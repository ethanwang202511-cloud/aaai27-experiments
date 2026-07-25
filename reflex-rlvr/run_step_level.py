"""Experiment 4: Step-level verification via per-token surprise and
verdict-at-step protocols.

Phase 4a -- Per-token surprise as step-level verifier:
    Segment oracle solutions into steps, compute per-token surprise under the
    base model for oracle vs. corrupted solutions, then measure whether the
    corrupted step stands out (step-level AUROC, top-1 accuracy).

Phase 4b -- Verdict-at-step protocol:
    Present problem + solution truncated at each step boundary, ask the model
    whether the partial solution is on track (YES/NO verdict logit).  Track
    how the verdict trajectory diverges between oracle and corrupted.

No Modal.  Uses vLLM + scipy.  Standalone Python.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from discrimination_standalone import (
    load_problems,
    create_llm,
    corrupt_solution_consistent,
    build_verdict_prompt,
    score_verdict_token,
    estimate_cost,
    _nan_to_none,
    _sanitize_row,
)

# ---------------------------------------------------------------------------
# Step segmentation
# ---------------------------------------------------------------------------

_RE_STEP_MARKER = re.compile(r"(?:Step\s+\d+|^\s*\(\d+\)\.?|^\s*\d+\.)", re.MULTILINE)
_RE_SENTENCE = re.compile(r"\.\s+(?=[A-Z\n])")


def _split_paragraph(text: str) -> list[str]:
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return parts


def _split_sentence(text: str) -> list[str]:
    parts = _RE_SENTENCE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_step_marker(text: str) -> list[str]:
    positions = [m.start() for m in _RE_STEP_MARKER.finditer(text)]
    if len(positions) < 2:
        return []
    parts = []
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        chunk = text[pos:end].strip()
        if chunk:
            parts.append(chunk)
    return parts


def _split_equal_chunks(text: str, n: int = 3) -> list[str]:
    chunk_size = max(1, len(text) // n)
    parts = []
    for i in range(n):
        start = i * chunk_size
        end = start + chunk_size if i < n - 1 else len(text)
        chunk = text[start:end].strip()
        if chunk:
            parts.append(chunk)
    return parts


def segment_solution(text: str) -> tuple[list[str], str]:
    """Segment a solution into steps.

    Returns (steps, strategy_name).  Picks the strategy that yields the most
    steps (min 2, max 20).  Falls back to 3 equal-length chunks.
    """
    candidates: list[tuple[str, list[str]]] = [
        ("paragraph", _split_paragraph(text)),
        ("sentence", _split_sentence(text)),
        ("step_marker", _split_step_marker(text)),
    ]

    # Filter to valid candidates (2..20 steps).
    valid = [(name, parts) for name, parts in candidates if 2 <= len(parts) <= 20]

    if valid:
        # Pick the one with the most steps.
        best_name, best_parts = max(valid, key=lambda x: len(x[1]))
        return best_parts, best_name

    # Fallback: 3 equal-length chunks.
    return _split_equal_chunks(text, 3), "equal_chunks"


# ---------------------------------------------------------------------------
# Token-level helpers
# ---------------------------------------------------------------------------

def compute_per_token_surprise(
    llm,
    tokenizer,
    prompt_text: str,
    scoring_params,
) -> list[float]:
    """Return per-token surprise (= -logprob) for every token in prompt_text.

    Uses the prompt_logprobs=0 scoring trick: generate with max_tokens=1 and
    read prompt_logprobs from the result.  The first token has no logprob (it
    is the start token), so we return a list of length (n_tokens - 1).
    """
    result = llm.generate([prompt_text], scoring_params)[0]
    prompt_lp = result.prompt_logprobs
    if prompt_lp is None:
        return []

    token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    surprises: list[float] = []

    for idx in range(1, len(prompt_lp)):
        pos_lp = prompt_lp[idx]
        if pos_lp is None:
            surprises.append(float("nan"))
            continue
        actual_id = token_ids[idx] if idx < len(token_ids) else None
        if actual_id is not None and actual_id in pos_lp:
            lp_obj = pos_lp[actual_id]
        else:
            lp_obj = next(iter(pos_lp.values())) if pos_lp else None
        if lp_obj is None:
            surprises.append(float("nan"))
            continue
        lp_val = getattr(lp_obj, "logprob", lp_obj)
        surprises.append(-float(lp_val))

    return surprises


def map_steps_to_token_ranges(
    tokenizer,
    prompt_prefix: str,
    steps: list[str],
) -> list[tuple[int, int]]:
    """Map each step to a (start_token_idx, end_token_idx) range within the
    full prompt = prompt_prefix + "".join(steps).

    Returns a list of (start, end) pairs where indices refer to positions in
    the full token sequence.
    """
    prefix_len = len(tokenizer.encode(prompt_prefix, add_special_tokens=False))
    ranges: list[tuple[int, int]] = []
    cursor = prefix_len

    for step in steps:
        step_tokens = tokenizer.encode(step, add_special_tokens=False)
        n = len(step_tokens)
        ranges.append((cursor, cursor + n))
        cursor += n

    return ranges


def mean_surprise_per_step(
    surprises: list[float],
    step_ranges: list[tuple[int, int]],
) -> list[float]:
    """Compute mean surprise for each step from per-token surprise values.

    ``surprises`` is offset by 1 relative to token indices (surprises[0]
    corresponds to token index 1), so we adjust accordingly.
    """
    result: list[float] = []
    for start, end in step_ranges:
        # surprises is 0-indexed but corresponds to token indices 1..N.
        # So token index i maps to surprises[i-1].
        vals = []
        for tok_idx in range(start, end):
            surp_idx = tok_idx - 1  # offset
            if 0 <= surp_idx < len(surprises):
                v = surprises[surp_idx]
                if not math.isnan(v):
                    vals.append(v)
        result.append(float(np.mean(vals)) if vals else float("nan"))
    return result


def find_corrupted_step(
    tokenizer,
    prompt_prefix: str,
    oracle_text: str,
    corrupted_text: str,
    steps_oracle: list[str],
    step_ranges: list[tuple[int, int]],
) -> int | None:
    """Find which step index contains the first token where oracle and
    corrupted differ.  Returns the step index (0-based) or None.
    """
    oracle_ids = tokenizer.encode(prompt_prefix + oracle_text, add_special_tokens=False)
    corrupt_ids = tokenizer.encode(prompt_prefix + corrupted_text, add_special_tokens=False)
    min_len = min(len(oracle_ids), len(corrupt_ids))

    first_diff = None
    for i in range(min_len):
        if oracle_ids[i] != corrupt_ids[i]:
            first_diff = i
            break
    if first_diff is None:
        if len(oracle_ids) != len(corrupt_ids):
            first_diff = min_len
        else:
            return None

    # Map first_diff to a step.
    for step_idx, (start, end) in enumerate(step_ranges):
        if start <= first_diff < end:
            return step_idx

    # If the diff is past all step ranges, assign to the last step.
    if step_ranges and first_diff >= step_ranges[-1][1]:
        return len(step_ranges) - 1

    return None


# ---------------------------------------------------------------------------
# Phase 4a: Per-token surprise
# ---------------------------------------------------------------------------

def run_phase_4a(
    problems: list[dict],
    llm,
    tokenizer,
    scoring_params,
    *,
    seed: int = 1337,
    model_id: str = "",
    output_path: Path | None = None,
) -> list[dict]:
    """Phase 4a: step-level surprise verification."""
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    results: list[dict] = []
    n = len(problems)

    for i, prob in enumerate(problems):
        t0 = time.time()
        problem_text = prob["problem"]
        oracle_solution = prob["solution"]
        answer = prob["answer"]
        prob_id = prob.get("id", i)

        # Segment oracle solution.
        steps, strategy = segment_solution(oracle_solution)
        K = len(steps)

        # Build prompt prefix (problem statement portion).
        prompt_prefix = (
            "Solve the following problem. Show your reasoning briefly, "
            "then put the final integer answer in \\boxed{}.\n\n"
            f"Problem: {problem_text}\n\nSolution: "
        )

        # Reconstruct full solution text from steps (may differ slightly from
        # original due to whitespace, but that is fine for token-level work).
        oracle_text = "\n\n".join(steps) if strategy == "paragraph" else ". ".join(steps)
        full_oracle_prompt = prompt_prefix + oracle_text

        # Corrupt the solution.
        corrupted_solution, _n_corrupt = corrupt_solution_consistent(
            oracle_solution, answer, seed=seed + i,
        )
        steps_corrupt, _ = segment_solution(corrupted_solution)
        corrupted_text = "\n\n".join(steps_corrupt) if strategy == "paragraph" else ". ".join(steps_corrupt)
        full_corrupt_prompt = prompt_prefix + corrupted_text

        # Token ranges for oracle steps.
        step_ranges = map_steps_to_token_ranges(tokenizer, prompt_prefix, steps)

        # Per-token surprise.
        surprise_oracle = compute_per_token_surprise(
            llm, tokenizer, full_oracle_prompt, scoring_params,
        )
        surprise_corrupt = compute_per_token_surprise(
            llm, tokenizer, full_corrupt_prompt, scoring_params,
        )

        # Step-level mean surprise.
        step_ranges_corrupt = map_steps_to_token_ranges(
            tokenizer, prompt_prefix, steps_corrupt,
        )
        mean_surp_oracle = mean_surprise_per_step(surprise_oracle, step_ranges)
        mean_surp_corrupt = mean_surprise_per_step(surprise_corrupt, step_ranges_corrupt)

        # Identify which step is corrupted.
        corrupted_step = find_corrupted_step(
            tokenizer, prompt_prefix, oracle_solution, corrupted_solution,
            steps, step_ranges,
        )

        # Delta surprise per step (use min of the two lengths to align).
        n_steps_aligned = min(len(mean_surp_oracle), len(mean_surp_corrupt))
        delta_surprise = [
            mean_surp_corrupt[j] - mean_surp_oracle[j]
            for j in range(n_steps_aligned)
        ]

        # Step-level AUROC.
        step_auroc = float("nan")
        top1_correct = False
        if (
            corrupted_step is not None
            and 0 <= corrupted_step < n_steps_aligned
            and n_steps_aligned >= 2
        ):
            labels = [1 if j == corrupted_step else 0 for j in range(n_steps_aligned)]
            scores = delta_surprise
            # Filter out NaN scores.
            valid_pairs = [
                (l, s) for l, s in zip(labels, scores) if not math.isnan(s)
            ]
            if len(valid_pairs) >= 2 and sum(p[0] for p in valid_pairs) >= 1:
                try:
                    step_auroc = roc_auc_score(
                        [p[0] for p in valid_pairs],
                        [p[1] for p in valid_pairs],
                    )
                except ValueError:
                    pass

            # Top-1 accuracy.
            valid_deltas = [
                (j, d) for j, d in enumerate(delta_surprise) if not math.isnan(d)
            ]
            if valid_deltas:
                argmax_step = max(valid_deltas, key=lambda x: x[1])[0]
                top1_correct = argmax_step == corrupted_step

        random_baseline = 1.0 / K if K > 0 else 0.0

        row = _sanitize_row({
            "id": prob_id,
            "model_id": model_id,
            "n_steps": K,
            "strategy": strategy,
            "corrupted_step": corrupted_step,
            "mean_surprise_oracle": [_nan_to_none(v) for v in mean_surp_oracle],
            "mean_surprise_corrupt": [_nan_to_none(v) for v in mean_surp_corrupt],
            "delta_surprise": [_nan_to_none(v) for v in delta_surprise],
            "step_auroc": _nan_to_none(step_auroc),
            "top1_correct": top1_correct,
            "random_baseline": round(random_baseline, 4),
            "elapsed": round(time.time() - t0, 2),
        })
        results.append(row)

        print(
            f"  [4a] {i+1}/{n}  id={prob_id}  K={K} ({strategy})  "
            f"corrupted_step={corrupted_step}  "
            f"AUROC={_nan_to_none(step_auroc)}  top1={top1_correct}  "
            f"({time.time()-t0:.1f}s)"
        )

    # Write results.
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")
        print(f"  Phase 4a results written to {output_path}")

    return results


# ---------------------------------------------------------------------------
# Phase 4b: Verdict-at-step protocol
# ---------------------------------------------------------------------------

_VERDICT_AT_STEP_TEMPLATE = (
    "You are a careful math grader. Read the problem and the partial "
    "solution so far, then judge whether the solution is on the right track."
    "\n\n"
    "Problem: {problem}\n\n"
    "Partial solution (up to step {k} of {K}): {solution_prefix}\n\n"
    "Is this partial solution on the right track toward the correct answer? "
    "Answer YES or NO.\n\n"
    "Answer:"
)


def run_phase_4b(
    problems: list[dict],
    llm,
    tokenizer,
    scoring_params,
    *,
    seed: int = 1337,
    model_id: str = "",
    output_path: Path | None = None,
    divergence_threshold: float = 0.5,
) -> list[dict]:
    """Phase 4b: verdict-at-step trajectory analysis."""
    results: list[dict] = []
    n = len(problems)

    for i, prob in enumerate(problems):
        t0 = time.time()
        problem_text = prob["problem"]
        oracle_solution = prob["solution"]
        answer = prob["answer"]
        prob_id = prob.get("id", i)

        # Segment solutions.
        steps_oracle, strategy = segment_solution(oracle_solution)
        K = len(steps_oracle)

        corrupted_solution, _n_corrupt = corrupt_solution_consistent(
            oracle_solution, answer, seed=seed + i,
        )
        steps_corrupt, _ = segment_solution(corrupted_solution)

        # Identify corrupted step using token-level diff.
        prompt_prefix = (
            "Solve the following problem. Show your reasoning briefly, "
            "then put the final integer answer in \\boxed{}.\n\n"
            f"Problem: {problem_text}\n\nSolution: "
        )
        step_ranges = map_steps_to_token_ranges(tokenizer, prompt_prefix, steps_oracle)
        corrupted_step = find_corrupted_step(
            tokenizer, prompt_prefix, oracle_solution, corrupted_solution,
            steps_oracle, step_ranges,
        )

        # Verdict trajectory: at each step boundary k=1..K.
        oracle_verdicts: list[float] = []
        corrupt_verdicts: list[float] = []

        for k in range(1, K + 1):
            # Oracle prefix up to step k.
            oracle_prefix = "\n\n".join(steps_oracle[:k])
            prompt_o = _VERDICT_AT_STEP_TEMPLATE.format(
                problem=problem_text, k=k, K=K, solution_prefix=oracle_prefix,
            )
            logp_yes_o = score_verdict_token(llm, tokenizer, prompt_o, " YES", seed=seed)
            logp_no_o = score_verdict_token(llm, tokenizer, prompt_o, " NO", seed=seed)
            verdict_o = round(logp_yes_o - logp_no_o, 4)
            oracle_verdicts.append(verdict_o)

            # Corrupted prefix up to step k (use min(k, len) to handle
            # differing step counts).
            k_c = min(k, len(steps_corrupt))
            corrupt_prefix = "\n\n".join(steps_corrupt[:k_c])
            prompt_c = _VERDICT_AT_STEP_TEMPLATE.format(
                problem=problem_text, k=k, K=K, solution_prefix=corrupt_prefix,
            )
            logp_yes_c = score_verdict_token(llm, tokenizer, prompt_c, " YES", seed=seed)
            logp_no_c = score_verdict_token(llm, tokenizer, prompt_c, " NO", seed=seed)
            verdict_c = round(logp_yes_c - logp_no_c, 4)
            corrupt_verdicts.append(verdict_c)

        # Find divergence point: first step k where
        #   oracle_verdict[k] - corrupt_verdict[k] > threshold.
        divergence_point: int | None = None
        verdict_gaps = []
        for k_idx in range(K):
            gap = oracle_verdicts[k_idx] - corrupt_verdicts[k_idx]
            verdict_gaps.append(round(gap, 4))
            if divergence_point is None and gap > divergence_threshold:
                divergence_point = k_idx + 1  # 1-indexed step

        # Mean verdict gap before vs after corruption.
        if corrupted_step is not None and corrupted_step < K:
            gaps_before = verdict_gaps[:corrupted_step]
            gaps_after = verdict_gaps[corrupted_step:]
            mean_gap_before = float(np.mean(gaps_before)) if gaps_before else float("nan")
            mean_gap_after = float(np.mean(gaps_after)) if gaps_after else float("nan")
        else:
            mean_gap_before = float("nan")
            mean_gap_after = float("nan")

        # Does the divergence point coincide with corruption location?
        divergence_matches = (
            divergence_point is not None
            and corrupted_step is not None
            and divergence_point == corrupted_step + 1  # both become 1-indexed
        )
        # More lenient: within 1 step.
        divergence_within_1 = (
            divergence_point is not None
            and corrupted_step is not None
            and abs(divergence_point - (corrupted_step + 1)) <= 1
        )

        row = _sanitize_row({
            "id": prob_id,
            "model_id": model_id,
            "n_steps": K,
            "strategy": strategy,
            "corrupted_step": corrupted_step,
            "oracle_verdicts": [_nan_to_none(v) for v in oracle_verdicts],
            "corrupt_verdicts": [_nan_to_none(v) for v in corrupt_verdicts],
            "verdict_gaps": [_nan_to_none(v) for v in verdict_gaps],
            "divergence_point": divergence_point,
            "divergence_matches_exact": divergence_matches,
            "divergence_within_1": divergence_within_1,
            "mean_gap_before_corruption": _nan_to_none(round(mean_gap_before, 4)),
            "mean_gap_after_corruption": _nan_to_none(round(mean_gap_after, 4)),
            "elapsed": round(time.time() - t0, 2),
        })
        results.append(row)

        print(
            f"  [4b] {i+1}/{n}  id={prob_id}  K={K}  "
            f"div_point={divergence_point}  corrupt_step={corrupted_step}  "
            f"match={divergence_matches}  "
            f"gap_before={_nan_to_none(round(mean_gap_before, 4))}  "
            f"gap_after={_nan_to_none(round(mean_gap_after, 4))}  "
            f"({time.time()-t0:.1f}s)"
        )

    # Write results.
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")
        print(f"  Phase 4b results written to {output_path}")

    return results


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def build_summary(
    all_results: dict[str, dict[str, list[dict]]],
) -> dict:
    """Build aggregate summary across all models.

    all_results: {model_name: {"4a": [...], "4b": [...]}}
    """
    summary: dict = {"models": {}, "comparison": []}

    for model_name, phase_results in all_results.items():
        model_summary: dict = {"model": model_name}

        # Phase 4a aggregates.
        if "4a" in phase_results and phase_results["4a"]:
            rows = phase_results["4a"]
            aurocs = [r["step_auroc"] for r in rows if r.get("step_auroc") is not None]
            top1s = [r["top1_correct"] for r in rows if "top1_correct" in r]
            baselines = [r["random_baseline"] for r in rows if "random_baseline" in r]

            model_summary["phase_4a"] = {
                "n_problems": len(rows),
                "mean_step_auroc": round(float(np.mean(aurocs)), 4) if aurocs else None,
                "std_step_auroc": round(float(np.std(aurocs)), 4) if aurocs else None,
                "top1_accuracy": round(sum(top1s) / len(top1s), 4) if top1s else None,
                "mean_random_baseline": round(float(np.mean(baselines)), 4) if baselines else None,
                "n_valid_auroc": len(aurocs),
            }

        # Phase 4b aggregates.
        if "4b" in phase_results and phase_results["4b"]:
            rows = phase_results["4b"]
            div_points = [r["divergence_point"] for r in rows if r.get("divergence_point") is not None]
            exact_matches = [r["divergence_matches_exact"] for r in rows if "divergence_matches_exact" in r]
            within_1 = [r["divergence_within_1"] for r in rows if "divergence_within_1" in r]
            gaps_before = [r["mean_gap_before_corruption"] for r in rows if r.get("mean_gap_before_corruption") is not None]
            gaps_after = [r["mean_gap_after_corruption"] for r in rows if r.get("mean_gap_after_corruption") is not None]

            model_summary["phase_4b"] = {
                "n_problems": len(rows),
                "mean_divergence_step": round(float(np.mean(div_points)), 2) if div_points else None,
                "n_diverged": len(div_points),
                "pct_diverged": round(len(div_points) / len(rows), 4) if rows else None,
                "exact_match_rate": round(sum(exact_matches) / len(exact_matches), 4) if exact_matches else None,
                "within_1_rate": round(sum(within_1) / len(within_1), 4) if within_1 else None,
                "mean_gap_before": round(float(np.mean(gaps_before)), 4) if gaps_before else None,
                "mean_gap_after": round(float(np.mean(gaps_after)), 4) if gaps_after else None,
            }

        summary["models"][model_name] = model_summary

    # Build comparison table.
    for model_name, ms in summary["models"].items():
        row = {"model": model_name}
        if "phase_4a" in ms:
            row["step_auroc"] = ms["phase_4a"].get("mean_step_auroc")
            row["top1_acc"] = ms["phase_4a"].get("top1_accuracy")
            row["random_baseline"] = ms["phase_4a"].get("mean_random_baseline")
        if "phase_4b" in ms:
            row["pct_diverged"] = ms["phase_4b"].get("pct_diverged")
            row["exact_match_rate"] = ms["phase_4b"].get("exact_match_rate")
            row["mean_gap_after"] = ms["phase_4b"].get("mean_gap_after")
        summary["comparison"].append(row)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 4: Step-level verification (surprise + verdict-at-step).",
    )
    parser.add_argument(
        "--problems-jsonl", type=str, required=True,
        help="Path to AIME problems JSONL (expects fields: problem, solution, answer).",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory for output files.",
    )
    parser.add_argument(
        "--model-ids", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Comma-separated model IDs to test.",
    )
    parser.add_argument(
        "--seed", type=int, default=1337,
        help="Random seed.",
    )
    parser.add_argument(
        "--tensor-parallel", type=int, default=4,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--phases", type=str, default="4a,4b",
        help="Comma-separated phases to run (e.g. '4a', '4b', '4a,4b').",
    )
    args = parser.parse_args()

    # Parse arguments.
    model_ids = [m.strip() for m in args.model_ids.split(",") if m.strip()]
    phases = {p.strip() for p in args.phases.split(",") if p.strip()}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load problems.
    print(f"Loading problems from {args.problems_jsonl} ...")
    problems = load_problems(args.problems_jsonl)
    print(f"  Loaded {len(problems)} problems.")

    # Estimated cost reporting.
    print(f"\nModels: {model_ids}")
    print(f"Phases: {sorted(phases)}")
    est_4a = 2.0 * len(model_ids)
    est_4b = 10.0 * len(model_ids)
    est_total = (est_4a if "4a" in phases else 0) + (est_4b if "4b" in phases else 0)
    print(f"Estimated cost: ~${est_total:.0f} total")
    print()

    all_results: dict[str, dict[str, list[dict]]] = {}

    for model_id in model_ids:
        model_name = model_id.split("/")[-1]
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"{'='*70}")
        print(f"Model: {model_id}")
        print(f"{'='*70}")

        # Create the LLM engine.
        print(f"  Loading model {model_id} ...")
        t_load = time.time()
        llm, tokenizer = create_llm(
            model_id,
            seed=args.seed,
            tensor_parallel_size=args.tensor_parallel,
        )
        print(f"  Model loaded in {time.time()-t_load:.1f}s")

        # Scoring params -- same trick as discrimination_v2.
        from discrimination_standalone import SamplingParams
        scoring_params = SamplingParams(
            prompt_logprobs=0,
            max_tokens=1,
            temperature=0.0,
            seed=args.seed,
        )

        phase_results: dict[str, list[dict]] = {}

        # Phase 4a.
        if "4a" in phases:
            print(f"\n--- Phase 4a: Per-token surprise (step-level) ---")
            t_4a = time.time()
            results_4a = run_phase_4a(
                problems, llm, tokenizer, scoring_params,
                seed=args.seed,
                model_id=model_id,
                output_path=model_dir / "step_surprise_results.jsonl",
            )
            phase_results["4a"] = results_4a
            print(f"  Phase 4a complete in {time.time()-t_4a:.1f}s")

        # Phase 4b.
        if "4b" in phases:
            print(f"\n--- Phase 4b: Verdict-at-step ---")
            t_4b = time.time()
            results_4b = run_phase_4b(
                problems, llm, tokenizer, scoring_params,
                seed=args.seed,
                model_id=model_id,
                output_path=model_dir / "step_verdict_results.jsonl",
            )
            phase_results["4b"] = results_4b
            print(f"  Phase 4b complete in {time.time()-t_4b:.1f}s")

        all_results[model_name] = phase_results

        # Release GPU memory before loading next model.
        del llm, tokenizer
        try:
            import torch  # noqa: PLC0415
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        import gc
        gc.collect()

    # Build and write summary.
    print(f"\n{'='*70}")
    print("Building summary ...")
    summary = build_summary(all_results)
    summary_path = output_dir / "step_level_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")

    # Print comparison table.
    print(f"\n{'='*70}")
    print("Comparison across models:")
    print(f"{'='*70}")
    for row in summary.get("comparison", []):
        print(f"\n  Model: {row['model']}")
        if "step_auroc" in row:
            print(f"    Step AUROC:       {row.get('step_auroc')}")
            print(f"    Top-1 accuracy:   {row.get('top1_acc')}")
            print(f"    Random baseline:  {row.get('random_baseline')}")
        if "pct_diverged" in row:
            print(f"    Pct diverged:     {row.get('pct_diverged')}")
            print(f"    Exact match rate: {row.get('exact_match_rate')}")
            print(f"    Mean gap after:   {row.get('mean_gap_after')}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
