#!/usr/bin/env python3
"""Experiment 2: STaR self-teacher loop for AAAI-27 reflex-RLVR paper.

Runs a Self-Taught Reasoner (STaR) loop on Qwen2.5-7B for N iterations.
Each iteration: (A) generate solutions, (B) filter, (C) LoRA fine-tune,
(D) evaluate.  No Modal dependencies -- uses vLLM for inference and
PEFT/LoRA + transformers for local training.

Usage
-----
    python run_star_loop.py \
        --substrate-jsonl /path/to/substrate_600.jsonl \
        --hk-jsonl /path/to/hk_51.jsonl \
        --math500-jsonl /path/to/math500.jsonl \
        --output-dir ./star_results
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Sibling-module imports (discrimination_standalone lives next to this file).
# ---------------------------------------------------------------------------
from discrimination_standalone import (
    build_verdict_prompt,
    corrupt_solution_consistent,
    create_llm,
    estimate_cost,
    load_problems,
    score_verdict_token,
    verdict_logit,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOLVE_PROMPT_TEMPLATE = (
    "Solve the following problem step by step. "
    "Put the final integer answer in \\boxed{{}}.\n\n"
    "Problem: {problem}\n\n"
    "Solution:"
)

VERDICT_TEMPLATE = (
    "You are a careful math grader. Read the problem and the proposed "
    "solution, then judge whether the proposed solution is correct.\n\n"
    "Problem: {problem}\n\n"
    "Proposed solution: {solution}\n\n"
    "Is the proposed solution correct? Answer YES or NO.\n\n"
    "Answer:"
)

# GPU cost rate for wall-clock cost estimation (local A100/H100 equivalent).
GPU_RATE_USD_PER_HOUR = 3.89  # Modal H100 rate, for comparison bookkeeping


# ===================================================================
# Utility helpers
# ===================================================================

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def extract_boxed_answer(text: str) -> str | None:
    """Return the content of the last \\boxed{...} in *text*, or None."""
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def verify_sympy(generated_answer: str | None, ground_truth: str) -> bool:
    """Check symbolic equality via SymPy, falling back to string match."""
    if generated_answer is None:
        return False
    try:
        from sympy import simplify, sympify

        gen = sympify(str(generated_answer))
        gt = sympify(str(ground_truth))
        return simplify(gen - gt) == 0
    except Exception:
        return str(generated_answer).strip() == str(ground_truth).strip()


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(obj: Any, path: str | Path) -> None:
    """Write *obj* as pretty-printed JSON to *path*."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  -> saved {path}")


# ===================================================================
# Phase A: generate solutions with vLLM
# ===================================================================

def phase_generate(
    model_path: str,
    problems: list[dict],
    k_samples: int,
    tensor_parallel: int,
    seed: int,
) -> list[dict]:
    """Generate K reasoning chains per problem using vLLM.

    Returns a list of dicts, each with keys:
        problem_idx, problem, answer, generation_idx, solution
    """
    from vllm import LLM, SamplingParams

    print(f"\n--- Phase A: Generating {k_samples} solutions per "
          f"{len(problems)} problems ({k_samples * len(problems)} total) ---")
    print(f"  Model: {model_path}")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel,
        trust_remote_code=True,
        seed=seed,
        dtype="bfloat16",
        max_model_len=4096,
    )

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=2048,
        n=k_samples,
        seed=seed,
    )

    prompts = [
        SOLVE_PROMPT_TEMPLATE.format(problem=p["problem"])
        for p in problems
    ]

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"  Generation complete in {elapsed:.1f}s")

    results: list[dict] = []
    for p_idx, (problem, output) in enumerate(zip(problems, outputs)):
        for g_idx, completion in enumerate(output.outputs):
            results.append({
                "problem_idx": p_idx,
                "problem": problem["problem"],
                "answer": str(problem["answer"]),
                "generation_idx": g_idx,
                "solution": completion.text,
            })

    print(f"  Total chains generated: {len(results)}")

    # Destroy the LLM to free GPU memory before training.
    del llm
    torch.cuda.empty_cache()

    return results


# ===================================================================
# Phase B: filter generated solutions
# ===================================================================

def _compute_verdict_logits(
    model_path: str,
    chains: list[dict],
    tensor_parallel: int,
    seed: int,
) -> list[float]:
    """Compute verdict_logit for each chain using a fresh vLLM instance.

    Returns a list of floats (one per chain).  verdict > 0 means the
    model judges the solution correct.
    """
    from vllm import LLM, SamplingParams

    print("  Computing verdict logits ...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel,
        trust_remote_code=True,
        seed=seed,
        dtype="bfloat16",
        max_model_len=4096,
    )
    tokenizer = llm.get_tokenizer()

    scoring_params = SamplingParams(
        prompt_logprobs=0,
        max_tokens=1,
        temperature=0.0,
    )

    verdicts: list[float] = []

    # Process in batches to avoid OOM on prompt construction.
    BATCH = 512
    for start in range(0, len(chains), BATCH):
        batch = chains[start : start + BATCH]
        batch_verdicts: list[float] = []

        for chain in batch:
            prompt = VERDICT_TEMPLATE.format(
                problem=chain["problem"],
                solution=chain["solution"],
            )

            def _score_token(token_str: str) -> float:
                full_text = prompt + token_str
                result = llm.generate([full_text], scoring_params)[0]
                full_ids = tokenizer.encode(full_text)
                prompt_ids = tokenizer.encode(prompt)
                n_ans = len(full_ids) - len(prompt_ids)
                if n_ans <= 0:
                    n_ans = max(
                        1,
                        len(tokenizer.encode(token_str, add_special_tokens=False)),
                    )
                answer_lp_list: list[float] = []
                prompt_lp = result.prompt_logprobs
                if prompt_lp is not None and len(prompt_lp) >= n_ans:
                    answer_positions = prompt_lp[-n_ans:]
                    for pos_idx, pos_lp in enumerate(answer_positions):
                        if pos_lp is None:
                            continue
                        actual_id = full_ids[len(full_ids) - n_ans + pos_idx]
                        lp_obj = pos_lp.get(actual_id)
                        if lp_obj is None:
                            lp_obj = next(iter(pos_lp.values()))
                        lp_val = getattr(lp_obj, "logprob", lp_obj)
                        answer_lp_list.append(float(lp_val))
                return (
                    sum(answer_lp_list) / len(answer_lp_list)
                    if answer_lp_list
                    else float("nan")
                )

            logp_yes = _score_token(" YES")
            logp_no = _score_token(" NO")
            v = logp_yes - logp_no
            batch_verdicts.append(round(v, 4) if not math.isnan(v) else -999.0)

        verdicts.extend(batch_verdicts)
        print(f"    Verdict batch {start}-{start + len(batch)}: done")

    del llm
    torch.cuda.empty_cache()

    return verdicts


def phase_filter(
    chains: list[dict],
    filter_mode: str,
    model_path: str,
    tensor_parallel: int,
    seed: int,
) -> tuple[list[dict], dict]:
    """Filter generated chains.

    Returns (filtered_chains, stats_dict).

    Each chain in the output gets an added key ``weight`` (1.0 for gold,
    0.5 for silver in "both" mode).
    """
    print(f"\n--- Phase B: Filtering ({filter_mode}) ---")

    # -- SymPy verification --
    sympy_pass: list[bool] = []
    for chain in chains:
        extracted = extract_boxed_answer(chain["solution"])
        sympy_pass.append(verify_sympy(extracted, chain["answer"]))
    n_sympy = sum(sympy_pass)
    print(f"  SymPy-verified correct: {n_sympy}/{len(chains)} "
          f"({100 * n_sympy / max(len(chains), 1):.1f}%)")

    # -- Verdict --
    need_verdict = filter_mode in ("both", "verdict_only")
    verdict_pass: list[bool] = [False] * len(chains)
    verdict_scores: list[float] = [0.0] * len(chains)

    if need_verdict:
        verdict_scores = _compute_verdict_logits(
            model_path, chains, tensor_parallel, seed
        )
        verdict_pass = [v > 0 for v in verdict_scores]
        n_verdict = sum(verdict_pass)
        print(f"  Verdict-positive: {n_verdict}/{len(chains)} "
              f"({100 * n_verdict / max(len(chains), 1):.1f}%)")

    # -- Combine --
    filtered: list[dict] = []
    n_gold = 0
    n_silver = 0

    for i, chain in enumerate(chains):
        chain_copy = dict(chain)
        chain_copy["sympy_correct"] = sympy_pass[i]
        chain_copy["verdict_score"] = verdict_scores[i]

        if filter_mode == "both":
            if sympy_pass[i] and verdict_pass[i]:
                chain_copy["weight"] = 1.0
                chain_copy["tier"] = "gold"
                filtered.append(chain_copy)
                n_gold += 1
            elif verdict_pass[i] and not sympy_pass[i]:
                chain_copy["weight"] = 0.5
                chain_copy["tier"] = "silver"
                filtered.append(chain_copy)
                n_silver += 1
        elif filter_mode == "verdict_only":
            if verdict_pass[i]:
                chain_copy["weight"] = 1.0
                chain_copy["tier"] = "verdict"
                filtered.append(chain_copy)
        elif filter_mode == "sympy_only":
            if sympy_pass[i]:
                chain_copy["weight"] = 1.0
                chain_copy["tier"] = "sympy"
                filtered.append(chain_copy)
        else:
            raise ValueError(f"Unknown filter_mode: {filter_mode}")

    stats = {
        "total_generated": len(chains),
        "sympy_correct": n_sympy,
        "verdict_positive": sum(verdict_pass) if need_verdict else None,
        "filter_mode": filter_mode,
        "n_gold": n_gold,
        "n_silver": n_silver,
        "n_kept": len(filtered),
        "keep_rate": round(len(filtered) / max(len(chains), 1), 4),
    }
    print(f"  Kept after filtering: {len(filtered)} "
          f"(gold={n_gold}, silver={n_silver})")

    return filtered, stats


# ===================================================================
# Phase C: LoRA fine-tune
# ===================================================================

@dataclass
class SFTSample:
    """A single supervised fine-tuning example."""
    prompt: str
    completion: str
    weight: float = 1.0


class WeightedSFTDataset(torch.utils.data.Dataset):
    """Dataset that tokenizes prompt+completion pairs with sample weights."""

    def __init__(
        self,
        samples: list[SFTSample],
        tokenizer,
        max_length: int = 2048,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        full_text = s.prompt + s.completion + self.tokenizer.eos_token
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Mask prompt tokens in labels (set to -100 so they are ignored
        # by the cross-entropy loss).
        prompt_encoding = self.tokenizer(
            s.prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        # Also mask padding.
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "weight": torch.tensor(s.weight, dtype=torch.float32),
        }


class WeightedSFTTrainer(torch.nn.Module):
    """Minimal trainer that applies per-sample weights to the SFT loss.

    Uses the HuggingFace Trainer API under the hood with a custom
    ``compute_loss`` override.
    """

    # This is a thin wrapper -- the actual training uses
    # transformers.Trainer subclassed below.
    pass


def _build_sft_samples(filtered_chains: list[dict]) -> list[SFTSample]:
    """Convert filtered chains into SFT training samples."""
    samples: list[SFTSample] = []
    for chain in filtered_chains:
        prompt = SOLVE_PROMPT_TEMPLATE.format(problem=chain["problem"])
        samples.append(SFTSample(
            prompt=prompt,
            completion=chain["solution"],
            weight=chain.get("weight", 1.0),
        ))
    return samples


def phase_train(
    model_path: str,
    filtered_chains: list[dict],
    output_checkpoint_dir: str,
    lora_rank: int = 32,
    lr: float = 1e-5,
    effective_batch: int = 256,
    batch_size: int = 8,
    seed: int = 1337,
) -> dict:
    """Fine-tune with LoRA on filtered chains.  Save merged checkpoint."""
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    print(f"\n--- Phase C: LoRA fine-tuning ---")
    print(f"  Model: {model_path}")
    print(f"  Training samples: {len(filtered_chains)}")
    print(f"  LoRA rank: {lora_rank}, LR: {lr}")

    set_seed(seed)

    if len(filtered_chains) == 0:
        print("  WARNING: No training samples after filtering.  Skipping training.")
        # Copy model as-is to checkpoint dir.
        os.makedirs(output_checkpoint_dir, exist_ok=True)
        return {
            "n_samples": 0,
            "skipped": True,
            "final_loss": None,
        }

    # Load tokenizer + model.
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Apply LoRA.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,  # alpha = 2 * rank
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Build dataset.
    sft_samples = _build_sft_samples(filtered_chains)
    dataset = WeightedSFTDataset(sft_samples, tokenizer, max_length=2048)

    grad_accum = max(1, effective_batch // batch_size)
    print(f"  Batch size: {batch_size}, Grad accumulation: {grad_accum} "
          f"(effective: {batch_size * grad_accum})")

    training_args = TrainingArguments(
        output_dir=output_checkpoint_dir,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="no",  # We save the merged model manually.
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
    )

    # Custom trainer with per-sample weighting.
    class _WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            weights = inputs.pop("weight")
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
            # Per-token loss is already averaged by the model; re-weight
            # per sample.  Note: the default loss is mean over the batch.
            # Here we apply per-sample weight to get weighted mean.
            if weights is not None and not torch.all(weights == 1.0):
                # Re-compute cross-entropy with weights.
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                labels = inputs["labels"][..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(
                    ignore_index=-100, reduction="none"
                )
                per_token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    labels.view(-1),
                )
                # Reshape to (batch, seq_len).
                per_token_loss = per_token_loss.view(labels.size())
                # Mean over tokens per sample.
                mask = (labels != -100).float()
                per_sample_loss = (per_token_loss * mask).sum(dim=-1) / (
                    mask.sum(dim=-1).clamp(min=1)
                )
                # Weighted mean over batch.
                loss = (per_sample_loss * weights).sum() / weights.sum()
            else:
                loss = outputs.loss

            return (loss, outputs) if return_outputs else loss

    t0 = time.time()
    trainer = _WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    train_result = trainer.train()
    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.1f}s")

    # Merge LoRA weights and save.
    print("  Merging LoRA weights ...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_checkpoint_dir)
    tokenizer.save_pretrained(output_checkpoint_dir)
    print(f"  Merged checkpoint saved to {output_checkpoint_dir}")

    train_stats = {
        "n_samples": len(sft_samples),
        "skipped": False,
        "train_loss": round(train_result.training_loss, 6),
        "train_runtime_s": round(elapsed, 1),
        "lora_rank": lora_rank,
        "lr": lr,
        "effective_batch": batch_size * grad_accum,
        "n_epochs": 1,
    }

    # Free memory.
    del model, merged_model, trainer
    torch.cuda.empty_cache()

    return train_stats


# ===================================================================
# Phase D: evaluate
# ===================================================================

def _generate_and_check(
    llm,
    problems: list[dict],
    n_samples: int,
    seed: int,
) -> list[dict]:
    """Generate n_samples solutions per problem; check correctness."""
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=0.8 if n_samples > 1 else 0.0,
        top_p=0.95 if n_samples > 1 else 1.0,
        max_tokens=2048,
        n=n_samples,
        seed=seed,
    )

    prompts = [
        SOLVE_PROMPT_TEMPLATE.format(problem=p["problem"])
        for p in problems
    ]
    outputs = llm.generate(prompts, sampling)

    results: list[dict] = []
    for p_idx, (problem, output) in enumerate(zip(problems, outputs)):
        any_correct = False
        per_gen: list[dict] = []
        for g_idx, completion in enumerate(output.outputs):
            extracted = extract_boxed_answer(completion.text)
            correct = verify_sympy(extracted, str(problem["answer"]))
            per_gen.append({
                "gen_idx": g_idx,
                "extracted_answer": extracted,
                "correct": correct,
            })
            if correct:
                any_correct = True

        results.append({
            "problem_idx": p_idx,
            "answer_gt": str(problem["answer"]),
            "n_samples": n_samples,
            "any_correct": any_correct,
            "n_correct": sum(g["correct"] for g in per_gen),
            "per_generation": per_gen,
        })

    return results


def _verdict_accuracy(
    model_path: str,
    problems: list[dict],
    tensor_parallel: int,
    seed: int,
) -> dict:
    """Measure verdict discrimination: oracle vs corrupted solutions."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel,
        trust_remote_code=True,
        seed=seed,
        dtype="bfloat16",
        max_model_len=4096,
    )
    tokenizer = llm.get_tokenizer()

    scoring_params = SamplingParams(
        prompt_logprobs=0,
        max_tokens=1,
        temperature=0.0,
    )

    def _score_tok(prompt: str, token_str: str) -> float:
        full_text = prompt + token_str
        result = llm.generate([full_text], scoring_params)[0]
        full_ids = tokenizer.encode(full_text)
        prompt_ids = tokenizer.encode(prompt)
        n_ans = len(full_ids) - len(prompt_ids)
        if n_ans <= 0:
            n_ans = max(
                1,
                len(tokenizer.encode(token_str, add_special_tokens=False)),
            )
        lp_list: list[float] = []
        prompt_lp = result.prompt_logprobs
        if prompt_lp is not None and len(prompt_lp) >= n_ans:
            for pos_idx, pos_lp in enumerate(prompt_lp[-n_ans:]):
                if pos_lp is None:
                    continue
                actual_id = full_ids[len(full_ids) - n_ans + pos_idx]
                lp_obj = pos_lp.get(actual_id)
                if lp_obj is None:
                    lp_obj = next(iter(pos_lp.values()))
                lp_val = getattr(lp_obj, "logprob", lp_obj)
                lp_list.append(float(lp_val))
        return sum(lp_list) / len(lp_list) if lp_list else float("nan")

    n_oracle_wins = 0
    rows: list[dict] = []

    for i, p in enumerate(problems):
        if "solution" not in p:
            continue
        oracle_sol = p["solution"]
        corrupted_sol, _ = corrupt_solution_consistent(
            oracle_sol, int(p["answer"]), seed=seed + i
        )

        prompt_oracle = VERDICT_TEMPLATE.format(
            problem=p["problem"], solution=oracle_sol
        )
        prompt_corrupt = VERDICT_TEMPLATE.format(
            problem=p["problem"], solution=corrupted_sol
        )

        logit_oracle = _score_tok(prompt_oracle, " YES") - _score_tok(prompt_oracle, " NO")
        logit_corrupt = _score_tok(prompt_corrupt, " YES") - _score_tok(prompt_corrupt, " NO")

        wins = logit_oracle > logit_corrupt
        if wins:
            n_oracle_wins += 1

        rows.append({
            "problem_idx": i,
            "logit_oracle": round(logit_oracle, 4),
            "logit_corrupt": round(logit_corrupt, 4),
            "oracle_wins": wins,
        })

    n_total = len(rows)
    accuracy = n_oracle_wins / max(n_total, 1)

    del llm
    torch.cuda.empty_cache()

    return {
        "n_problems": n_total,
        "oracle_wins": n_oracle_wins,
        "accuracy": round(accuracy, 4),
        "per_problem": rows,
    }


def phase_evaluate(
    model_path: str,
    hk_problems: list[dict],
    math500_problems: list[dict],
    tensor_parallel: int,
    k_samples: int,
    seed: int,
    iter_dir: str,
    measure_verdict: bool = True,
) -> dict:
    """Run full evaluation suite; save results to iter_dir."""
    from vllm import LLM

    print(f"\n--- Phase D: Evaluation ---")
    print(f"  Model: {model_path}")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel,
        trust_remote_code=True,
        seed=seed,
        dtype="bfloat16",
        max_model_len=4096,
    )

    # -- pass@1 on H_K --
    print(f"  Evaluating pass@1 on H_K ({len(hk_problems)} problems) ...")
    hk_p1_results = _generate_and_check(llm, hk_problems, n_samples=1, seed=seed)
    hk_pass1 = sum(r["any_correct"] for r in hk_p1_results) / max(len(hk_p1_results), 1)
    print(f"    pass@1 = {hk_pass1:.4f} ({sum(r['any_correct'] for r in hk_p1_results)}/{len(hk_p1_results)})")

    # -- pass@K on H_K --
    print(f"  Evaluating pass@{k_samples} on H_K ...")
    hk_pk_results = _generate_and_check(llm, hk_problems, n_samples=k_samples, seed=seed + 1)
    hk_passk = sum(r["any_correct"] for r in hk_pk_results) / max(len(hk_pk_results), 1)
    print(f"    pass@{k_samples} = {hk_passk:.4f} ({sum(r['any_correct'] for r in hk_pk_results)}/{len(hk_pk_results)})")

    # -- pass@1 on MATH-500 (forgetting control) --
    print(f"  Evaluating pass@1 on MATH-500 ({len(math500_problems)} problems) ...")
    m500_results = _generate_and_check(llm, math500_problems, n_samples=1, seed=seed + 2)
    m500_pass1 = sum(r["any_correct"] for r in m500_results) / max(len(m500_results), 1)
    print(f"    pass@1 = {m500_pass1:.4f} ({sum(r['any_correct'] for r in m500_results)}/{len(m500_results)})")

    del llm
    torch.cuda.empty_cache()

    eval_hk = {
        "pass_at_1": round(hk_pass1, 4),
        f"pass_at_{k_samples}": round(hk_passk, 4),
        "n_problems": len(hk_problems),
        "per_problem_p1": hk_p1_results,
        "per_problem_pk": hk_pk_results,
    }
    eval_m500 = {
        "pass_at_1": round(m500_pass1, 4),
        "n_problems": len(math500_problems),
        "per_problem": m500_results,
    }

    save_json(eval_hk, os.path.join(iter_dir, "eval_hk.json"))
    save_json(eval_m500, os.path.join(iter_dir, "eval_math500.json"))

    # -- Verdict accuracy on H_K --
    verdict_acc = None
    if measure_verdict:
        print("  Measuring verdict accuracy on H_K ...")
        verdict_acc = _verdict_accuracy(
            model_path, hk_problems, tensor_parallel, seed
        )
        save_json(verdict_acc, os.path.join(iter_dir, "verdict_accuracy.json"))
        print(f"    Verdict accuracy: {verdict_acc['accuracy']:.4f}")

    metrics = {
        "hk_pass_at_1": round(hk_pass1, 4),
        f"hk_pass_at_{k_samples}": round(hk_passk, 4),
        "math500_pass_at_1": round(m500_pass1, 4),
        "verdict_accuracy": verdict_acc["accuracy"] if verdict_acc else None,
    }
    print(f"  Eval summary: {metrics}")
    return metrics


# ===================================================================
# Main STaR loop
# ===================================================================

def run_star_loop(args: argparse.Namespace) -> None:
    """Execute the full STaR self-teacher loop."""

    print("=" * 70)
    print("STaR Self-Teacher Loop -- Experiment 2")
    print("=" * 70)
    print(f"  Model:           {args.model_id}")
    print(f"  Substrate:       {args.substrate_jsonl} ")
    print(f"  H_K set:         {args.hk_jsonl}")
    print(f"  MATH-500:        {args.math500_jsonl}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Iterations:      {args.n_iterations}")
    print(f"  K samples:       {args.k_samples}")
    print(f"  Filter mode:     {args.filter_mode}")
    print(f"  LoRA rank:       {args.lora_rank}")
    print(f"  LR:              {args.lr}")
    print(f"  Effective batch: {args.effective_batch}")
    print(f"  Seed:            {args.seed}")
    print(f"  Tensor parallel: {args.tensor_parallel}")
    print()

    # Load datasets.
    substrate = load_jsonl(args.substrate_jsonl)
    hk_problems = load_jsonl(args.hk_jsonl)
    math500 = load_jsonl(args.math500_jsonl)

    print(f"Loaded {len(substrate)} substrate problems, "
          f"{len(hk_problems)} H_K problems, "
          f"{len(math500)} MATH-500 problems.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Track metrics across iterations for the summary table.
    summary_rows: list[dict] = []
    current_model = args.model_id
    total_t0 = time.time()

    # ------------------------------------------------------------------
    # Iteration 0: baseline evaluation (no training)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ITERATION 0 (baseline -- no training)")
    print("=" * 70)

    iter0_dir = os.path.join(args.output_dir, "iter_0")
    os.makedirs(iter0_dir, exist_ok=True)

    metrics_0 = phase_evaluate(
        model_path=current_model,
        hk_problems=hk_problems,
        math500_problems=math500,
        tensor_parallel=args.tensor_parallel,
        k_samples=args.k_samples,
        seed=args.seed,
        iter_dir=iter0_dir,
        measure_verdict=True,
    )
    summary_rows.append({"iteration": 0, "model": current_model, **metrics_0})

    # ------------------------------------------------------------------
    # Iterations 1..N
    # ------------------------------------------------------------------
    for it in range(1, args.n_iterations + 1):
        print("\n" + "=" * 70)
        print(f"ITERATION {it} / {args.n_iterations}")
        print("=" * 70)
        iter_t0 = time.time()

        iter_dir = os.path.join(args.output_dir, f"iter_{it}")
        os.makedirs(iter_dir, exist_ok=True)
        ckpt_dir = os.path.join(iter_dir, "checkpoint")

        # Phase A: generate.
        chains = phase_generate(
            model_path=current_model,
            problems=substrate,
            k_samples=args.k_samples,
            tensor_parallel=args.tensor_parallel,
            seed=args.seed + it * 1000,
        )

        # Phase B: filter.
        filtered, gen_stats = phase_filter(
            chains=chains,
            filter_mode=args.filter_mode,
            model_path=current_model,
            tensor_parallel=args.tensor_parallel,
            seed=args.seed + it * 1000 + 500,
        )
        save_json(gen_stats, os.path.join(iter_dir, "generation_stats.json"))

        # Phase C: train.
        train_stats = phase_train(
            model_path=current_model,
            filtered_chains=filtered,
            output_checkpoint_dir=ckpt_dir,
            lora_rank=args.lora_rank,
            lr=args.lr,
            effective_batch=args.effective_batch,
            batch_size=args.batch_size,
            seed=args.seed + it * 2000,
        )
        save_json(train_stats, os.path.join(iter_dir, "train_stats.json"))

        # Update model path to the newly merged checkpoint.
        if not train_stats.get("skipped", False):
            current_model = ckpt_dir

        # Phase D: evaluate.
        metrics_it = phase_evaluate(
            model_path=current_model,
            hk_problems=hk_problems,
            math500_problems=math500,
            tensor_parallel=args.tensor_parallel,
            k_samples=args.k_samples,
            seed=args.seed + it * 3000,
            iter_dir=iter_dir,
            measure_verdict=True,
        )

        iter_elapsed = time.time() - iter_t0
        row = {
            "iteration": it,
            "model": current_model,
            "iter_time_s": round(iter_elapsed, 1),
            "n_train_samples": train_stats.get("n_samples", 0),
            "train_loss": train_stats.get("train_loss"),
            **metrics_it,
        }
        summary_rows.append(row)

        estimated_iter_cost = GPU_RATE_USD_PER_HOUR * args.tensor_parallel * (iter_elapsed / 3600.0)
        print(f"\n  Iteration {it} done in {iter_elapsed:.0f}s "
              f"(est. ${estimated_iter_cost:.2f})")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_elapsed = time.time() - total_t0
    total_est_cost = GPU_RATE_USD_PER_HOUR * args.tensor_parallel * (total_elapsed / 3600.0)

    summary = {
        "config": {
            "model_id": args.model_id,
            "n_iterations": args.n_iterations,
            "k_samples": args.k_samples,
            "filter_mode": args.filter_mode,
            "lora_rank": args.lora_rank,
            "lr": args.lr,
            "effective_batch": args.effective_batch,
            "seed": args.seed,
            "n_substrate": len(substrate),
            "n_hk": len(hk_problems),
            "n_math500": len(math500),
        },
        "total_time_s": round(total_elapsed, 1),
        "estimated_total_cost_usd": round(total_est_cost, 2),
        "iterations": summary_rows,
    }
    save_json(summary, os.path.join(args.output_dir, "summary.json"))

    # Print summary table.
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    header = (
        f"{'Iter':>4} | {'H_K p@1':>8} | {'H_K p@K':>8} | "
        f"{'M500 p@1':>8} | {'Verdict':>8} | {'#Train':>7} | {'Loss':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        hk_p1 = row.get("hk_pass_at_1", "---")
        hk_pk = row.get(f"hk_pass_at_{args.k_samples}", "---")
        m500 = row.get("math500_pass_at_1", "---")
        vacc = row.get("verdict_accuracy", "---")
        ntrain = row.get("n_train_samples", "---")
        loss = row.get("train_loss", "---")

        hk_p1_s = f"{hk_p1:.4f}" if isinstance(hk_p1, float) else str(hk_p1)
        hk_pk_s = f"{hk_pk:.4f}" if isinstance(hk_pk, float) else str(hk_pk)
        m500_s = f"{m500:.4f}" if isinstance(m500, float) else str(m500)
        vacc_s = f"{vacc:.4f}" if isinstance(vacc, float) else str(vacc)
        ntrain_s = str(ntrain)
        loss_s = f"{loss:.6f}" if isinstance(loss, float) else str(loss)

        print(
            f"{row['iteration']:>4} | {hk_p1_s:>8} | {hk_pk_s:>8} | "
            f"{m500_s:>8} | {vacc_s:>8} | {ntrain_s:>7} | {loss_s:>8}"
        )

    print(f"\nTotal elapsed: {total_elapsed:.0f}s")
    print(f"Estimated total cost: ${total_est_cost:.2f}")
    print(f"Results saved to: {args.output_dir}")


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STaR self-teacher loop (Experiment 2, AAAI-27).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-id", type=str, default="Qwen/Qwen2.5-7B",
        help="HuggingFace model ID or local path for the base model.",
    )
    parser.add_argument(
        "--substrate-jsonl", type=str, required=True,
        help="Path to the 600-problem substrate JSONL file.",
    )
    parser.add_argument(
        "--hk-jsonl", type=str, required=True,
        help="Path to the 51-problem hard set (H_K) JSONL file.",
    )
    parser.add_argument(
        "--math500-jsonl", type=str, required=True,
        help="Path to the MATH-500 JSONL file (forgetting control).",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory for checkpoints, results, and summary.",
    )
    parser.add_argument(
        "--n-iterations", type=int, default=3,
        help="Number of STaR iterations (each = generate + filter + train + eval).",
    )
    parser.add_argument(
        "--k-samples", type=int, default=32,
        help="Number of reasoning chains to sample per problem.",
    )
    parser.add_argument(
        "--filter-mode", type=str, default="both",
        choices=["both", "verdict_only", "sympy_only"],
        help="Filtering strategy for generated chains.",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=32,
        help="LoRA rank (alpha is set to 2 * rank).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5,
        help="Learning rate for LoRA fine-tuning.",
    )
    parser.add_argument(
        "--effective-batch", type=int, default=256,
        help="Effective batch size (batch_size * gradient_accumulation).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--seed", type=int, default=1337,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--tensor-parallel", type=int, default=4,
        help="Tensor parallelism degree for vLLM inference.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_star_loop(args)
