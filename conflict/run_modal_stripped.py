#!/usr/bin/env python3
"""
Consolidated Modal-free experiment runner for AAAI-27 conflict monitoring paper.

Strips Modal GPU orchestration from the original experiment scripts and runs
locally on CUDA/MPS/CPU. Three experiment modes:

  norm-vs-drop       Correlation between component L2 output norm and
                     single-component mean-ablation drop.
  audit              Matched-null audit with configurable null samples,
                     DLA-based circuit discovery, type-stratified null,
                     and EVT/GPD tail p-values.
  ablation-sensitivity  Run the audit across all 3 ablation types for a
                        single model-task cell.

Usage:
  python run_modal_stripped.py norm-vs-drop --model gpt2 --task ioi --n 200
  python run_modal_stripped.py audit --model gpt2 --task ioi --n-null 1000 --ablation zero
  python run_modal_stripped.py ablation-sensitivity --model gpt2 --task ioi --n-null 500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_MAP: Dict[str, str] = {
    "gpt2": "gpt2-small",
    "qwen2.5": "Qwen/Qwen2.5-0.5B",
    "gemma2": "gemma-2-2b",
    "llama3": "meta-llama/Meta-Llama-3.1-8B",
}

ABLATION_TYPES = ("mean", "zero", "resampling")

# IOI prompt ingredients
IOI_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
    "Grace", "Henry", "Iris", "Jack", "Kate", "Leo",
    "Mary", "Nick", "Olivia", "Paul", "Quinn", "Rachel",
    "Sam", "Tom", "Uma", "Victor", "Wendy",
]
IOI_PLACES = [
    "store", "park", "library", "beach", "restaurant",
    "museum", "market", "school", "office", "cafe",
]

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[device] Using CUDA ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[device] Using Apple MPS")
    else:
        dev = torch.device("cpu")
        print("[device] Using CPU")
    return dev


# ---------------------------------------------------------------------------
# IOI dataset generation
# ---------------------------------------------------------------------------

@dataclass
class IOIPrompt:
    """Single IOI (Indirect Object Identification) prompt."""
    text: str
    io_name: str       # Indirect Object = correct answer
    s_name: str        # Subject = incorrect answer
    io_token: int = -1
    s_token: int = -1


def generate_ioi_prompts(n: int, seed: int = 42) -> List[IOIPrompt]:
    """Generate n IOI prompts using the standard template.

    Template: "When {IO} and {S} went to the {place}, {IO} gave a gift to"
    Correct continuation: S   (the model should predict S)
    Wait -- standard IOI: name1=name3=IO appears twice, correct answer is IO.
    Actually in the standard IOI task:
      "When Mary and John went to the store, John gave a drink to"
      correct = Mary (IO), incorrect = John (S)

    Let's use the canonical framing:
      name1 = IO (indirect object, correct answer)
      name2 = S  (subject, distractor)
      name3 = S  (repeated subject)
      correct completion = name1 = IO
    """
    rng = random.Random(seed)
    prompts = []
    for _ in range(n):
        # Pick two distinct names
        io_name, s_name = rng.sample(IOI_NAMES, 2)
        place = rng.choice(IOI_PLACES)
        text = f"When {io_name} and {s_name} went to the {place}, {s_name} gave a gift to"
        prompts.append(IOIPrompt(text=text, io_name=io_name, s_name=s_name))
    return prompts


def tokenize_ioi(model, prompts: List[IOIPrompt]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """Tokenize IOI prompts and resolve correct/incorrect token ids.

    Returns:
        tokens:    (n, seq_len) padded token ids
        correct:   (n,) token id of the correct (IO) name
        incorrect: (n,) token id of the incorrect (S) name
        last_pos:  list of last-token positions per prompt
    """
    tokenizer = model.tokenizer

    # Resolve name -> first token id (with leading space, as in continuation)
    all_texts = [p.text for p in prompts]
    tokens_list = [model.to_tokens(t, prepend_bos=True).squeeze(0) for t in all_texts]

    # Pad to uniform length
    max_len = max(t.shape[0] for t in tokens_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded = torch.full((len(tokens_list), max_len), pad_id, dtype=torch.long)
    last_positions = []
    for i, t in enumerate(tokens_list):
        padded[i, :t.shape[0]] = t
        last_positions.append(t.shape[0] - 1)

    # Correct / incorrect token ids
    correct_ids = []
    incorrect_ids = []
    for p in prompts:
        # Tokenize " Name" (with space prefix) and take first token
        io_toks = model.to_tokens(f" {p.io_name}", prepend_bos=False).squeeze(0)
        s_toks = model.to_tokens(f" {p.s_name}", prepend_bos=False).squeeze(0)
        correct_ids.append(io_toks[0].item())
        incorrect_ids.append(s_toks[0].item())

    correct = torch.tensor(correct_ids, dtype=torch.long)
    incorrect = torch.tensor(incorrect_ids, dtype=torch.long)

    return padded, correct, incorrect, last_positions


# ---------------------------------------------------------------------------
# Metric: logit difference
# ---------------------------------------------------------------------------

def logit_diff_metric(logits: torch.Tensor, correct: torch.Tensor,
                      incorrect: torch.Tensor, last_pos: List[int]) -> float:
    """Mean logit difference: logit(correct) - logit(incorrect) at last position."""
    batch_idx = torch.arange(logits.shape[0])
    pos_idx = torch.tensor(last_pos, dtype=torch.long)
    final_logits = logits[batch_idx, pos_idx, :]  # (batch, vocab)
    correct_logits = final_logits[batch_idx, correct]
    incorrect_logits = final_logits[batch_idx, incorrect]
    return (correct_logits - incorrect_logits).mean().item()


# ---------------------------------------------------------------------------
# Batched forward pass helper
# ---------------------------------------------------------------------------

def batched_forward(model, tokens: torch.Tensor, batch_size: int,
                    fwd_hooks=None, return_type="logits") -> torch.Tensor:
    """Run forward passes in batches, optionally with hooks."""
    device = next(model.parameters()).device
    all_outputs = []
    n = tokens.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = tokens[start:end].to(device)
        if fwd_hooks:
            with model.hooks(fwd_hooks=fwd_hooks):
                out = model(batch, return_type=return_type)
        else:
            out = model(batch, return_type=return_type)
        all_outputs.append(out.cpu())
    return torch.cat(all_outputs, dim=0)


# ---------------------------------------------------------------------------
# Cache activations
# ---------------------------------------------------------------------------

def cache_activations(model, tokens: torch.Tensor, batch_size: int,
                      hook_names: List[str]) -> Dict[str, torch.Tensor]:
    """Run forward pass and cache specified hook activations."""
    device = next(model.parameters()).device
    cache = {name: [] for name in hook_names}
    n = tokens.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = tokens[start:end].to(device)
        _, batch_cache = model.run_with_cache(batch, names_filter=hook_names,
                                               return_type="logits")
        for name in hook_names:
            cache[name].append(batch_cache[name].cpu())
    return {name: torch.cat(parts, dim=0) for name, parts in cache.items()}


# ---------------------------------------------------------------------------
# Component enumeration
# ---------------------------------------------------------------------------

@dataclass
class Component:
    """Represents an attention head or MLP layer."""
    kind: Literal["head", "mlp"]
    layer: int
    head: Optional[int] = None  # None for MLP

    @property
    def hook_name(self) -> str:
        if self.kind == "head":
            return f"blocks.{self.layer}.attn.hook_result"
        return f"blocks.{self.layer}.hook_mlp_out"

    @property
    def label(self) -> str:
        if self.kind == "head":
            return f"L{self.layer}H{self.head}"
        return f"L{self.layer}_MLP"

    def __repr__(self) -> str:
        return self.label


def enumerate_components(model) -> List[Component]:
    """List all attention heads and MLP layers."""
    cfg = model.cfg
    components = []
    for layer in range(cfg.n_layers):
        for head in range(cfg.n_heads):
            components.append(Component("head", layer, head))
        components.append(Component("mlp", layer))
    return components


# ---------------------------------------------------------------------------
# DLA attribution
# ---------------------------------------------------------------------------

def compute_dla_scores(model, tokens: torch.Tensor, correct: torch.Tensor,
                       incorrect: torch.Tensor, last_pos: List[int],
                       batch_size: int) -> Dict[str, float]:
    """Compute Direct Logit Attribution for all components.

    DLA(C) = component_output @ (W_U[:, correct] - W_U[:, incorrect])
    evaluated at the last token position, averaged over the dataset.

    Returns dict mapping component label -> mean DLA score.
    """
    device = next(model.parameters()).device
    components = enumerate_components(model)

    # Collect all hook names we need
    hook_names = list({c.hook_name for c in components})

    # Cache all activations
    act_cache = cache_activations(model, tokens, batch_size, hook_names)

    # W_U: (d_model, vocab)
    W_U = model.W_U.cpu().float()

    scores = {}
    n = tokens.shape[0]
    batch_idx = torch.arange(n)
    pos_idx = torch.tensor(last_pos, dtype=torch.long)

    for comp in components:
        act = act_cache[comp.hook_name].float()  # (batch, seq, ...)

        if comp.kind == "head":
            # act shape: (batch, seq, n_heads, d_model)
            out = act[batch_idx, pos_idx, comp.head, :]  # (batch, d_model)
        else:
            # act shape: (batch, seq, d_model)
            out = act[batch_idx, pos_idx, :]  # (batch, d_model)

        # Direction vector per sample: W_U[:, correct_i] - W_U[:, incorrect_i]
        direction = W_U[:, correct] - W_U[:, incorrect]  # (d_model, batch)
        direction = direction.T  # (batch, d_model)

        # DLA = dot product
        dla = (out * direction).sum(dim=-1)  # (batch,)
        scores[comp.label] = dla.mean().item()

    return scores


# ---------------------------------------------------------------------------
# Ablation hooks
# ---------------------------------------------------------------------------

def make_mean_ablation_hook(comp: Component, mean_cache: Dict[str, torch.Tensor]):
    """Hook that replaces component output with its per-position mean."""
    mean_act = mean_cache[comp.hook_name]  # (seq, ...) or with head dim

    def hook_fn(activation, hook):
        if comp.kind == "head":
            # activation: (batch, seq, n_heads, d_model)
            seq_len = min(activation.shape[1], mean_act.shape[0])
            activation[:, :seq_len, comp.head, :] = mean_act[:seq_len, comp.head, :].to(activation.device)
        else:
            # activation: (batch, seq, d_model)
            seq_len = min(activation.shape[1], mean_act.shape[0])
            activation[:, :seq_len, :] = mean_act[:seq_len, :].to(activation.device)
        return activation

    return (comp.hook_name, hook_fn)


def make_zero_ablation_hook(comp: Component):
    """Hook that zeros out component output."""
    def hook_fn(activation, hook):
        if comp.kind == "head":
            activation[:, :, comp.head, :] = 0.0
        else:
            activation[:, :, :] = 0.0
        return activation
    return (comp.hook_name, hook_fn)


def make_resampling_hook(comp: Component, tokens: torch.Tensor,
                         resample_cache: Dict[str, torch.Tensor]):
    """Hook that replaces component output with activation from a random other prompt."""
    cached = resample_cache[comp.hook_name]  # (n, seq, ...)
    n = cached.shape[0]

    def hook_fn(activation, hook):
        batch = activation.shape[0]
        # Random permutation for resampling
        perm = torch.randint(0, n, (batch,))
        replacement = cached[perm].to(activation.device)
        seq_len = min(activation.shape[1], replacement.shape[1])
        if comp.kind == "head":
            activation[:, :seq_len, comp.head, :] = replacement[:, :seq_len, comp.head, :]
        else:
            activation[:, :seq_len, :] = replacement[:, :seq_len, :]
        return activation

    return (comp.hook_name, hook_fn)


def compute_mean_activations(act_cache: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Compute per-position mean across the batch dimension for each hook."""
    return {name: tensor.float().mean(dim=0) for name, tensor in act_cache.items()}


# ---------------------------------------------------------------------------
# Experiment 1: norm-vs-drop
# ---------------------------------------------------------------------------

def run_norm_vs_drop(model_name: str, task: str, n: int, batch_size: int,
                     seed: int, output_dir: Path):
    """Correlation between component L2 output norm and mean-ablation drop."""
    from transformer_lens import HookedTransformer

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: norm-vs-drop")
    print(f"  model={model_name}, task={task}, n={n}")
    print(f"{'='*60}\n")

    device = get_device()
    tl_name = MODEL_MAP[model_name]
    print(f"Loading model {tl_name} ...")
    model = HookedTransformer.from_pretrained(tl_name, device=device)
    model.eval()

    # Generate prompts
    prompts = generate_ioi_prompts(n, seed=seed)
    tokens, correct, incorrect, last_pos = tokenize_ioi(model, prompts)
    print(f"Generated {len(prompts)} IOI prompts, max seq len = {tokens.shape[1]}")

    components = enumerate_components(model)
    hook_names = list({c.hook_name for c in components})

    # Cache activations
    print("Caching activations ...")
    act_cache = cache_activations(model, tokens, batch_size, hook_names)
    mean_cache = compute_mean_activations(act_cache)

    # Clean metric
    print("Computing clean metric ...")
    clean_logits = batched_forward(model, tokens, batch_size)
    clean_metric = logit_diff_metric(clean_logits, correct, incorrect, last_pos)
    print(f"  Clean logit diff = {clean_metric:.4f}")

    # Per-component: norm and drop
    results_rows = []
    print(f"Evaluating {len(components)} components ...")
    for i, comp in enumerate(components):
        # --- Norm: E_x || out_C(x)[:, last_pos, :] ||_2 ---
        act = act_cache[comp.hook_name].float()
        batch_idx = torch.arange(n)
        pos_idx = torch.tensor(last_pos, dtype=torch.long)

        if comp.kind == "head":
            out = act[batch_idx, pos_idx, comp.head, :]  # (n, d_model)
        else:
            out = act[batch_idx, pos_idx, :]

        norm_val = out.norm(dim=-1).mean().item()

        # --- Drop: metric(clean) - metric(ablate_C_mean) ---
        hook = make_mean_ablation_hook(comp, mean_cache)
        abl_logits = batched_forward(model, tokens, batch_size, fwd_hooks=[hook])
        abl_metric = logit_diff_metric(abl_logits, correct, incorrect, last_pos)
        drop_val = clean_metric - abl_metric

        results_rows.append({
            "component": comp.label,
            "kind": comp.kind,
            "layer": comp.layer,
            "head": comp.head,
            "norm": norm_val,
            "drop": drop_val,
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(components):
            print(f"  [{i+1}/{len(components)}] {comp.label}: norm={norm_val:.4f}, drop={drop_val:.4f}")

    # Correlation
    norms = np.array([r["norm"] for r in results_rows])
    drops = np.array([r["drop"] for r in results_rows])
    pearson_r, pearson_p = scipy_stats.pearsonr(norms, drops)

    summary = {
        "experiment": "norm-vs-drop",
        "model": model_name,
        "tl_name": tl_name,
        "task": task,
        "n_prompts": n,
        "clean_logit_diff": clean_metric,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "n_components": len(components),
        "components": results_rows,
    }

    # Save
    out_file = output_dir / f"norm_vs_drop_{model_name}_{task}_n{n}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPearson r = {pearson_r:.4f} (p = {pearson_p:.2e})")
    print(f"Results saved to {out_file}")
    return summary


# ---------------------------------------------------------------------------
# Experiment 2: audit (matched-null with n null samples)
# ---------------------------------------------------------------------------

def gpd_tail_pvalue(null_samples: np.ndarray, observed: float,
                    threshold_quantile: float = 0.9) -> Optional[float]:
    """Fit GPD to exceedances above a high threshold and return tail p-value.

    Uses EVT/GPD to estimate p-values below the empirical floor
    (1/n_null). Returns None if fitting fails or if observed does not
    exceed the threshold.
    """
    try:
        from scipy.stats import genpareto
    except ImportError:
        return None

    threshold = np.quantile(null_samples, threshold_quantile)
    if observed <= threshold:
        return None  # Not in the tail

    exceedances = null_samples[null_samples > threshold] - threshold
    if len(exceedances) < 20:
        return None  # Not enough tail data

    try:
        shape, loc, scale = genpareto.fit(exceedances, floc=0)
        # P(X > observed | X > threshold) under GPD
        gpd_surv = genpareto.sf(observed - threshold, shape, loc=0, scale=scale)
        # P(X > threshold) empirically
        p_exceed = (null_samples > threshold).mean()
        # Combined tail p-value
        return float(p_exceed * gpd_surv)
    except Exception:
        return None


def run_audit(model_name: str, task: str, n_null: int, ablation: str,
              batch_size: int, seed: int, output_dir: Path,
              dla_threshold: float = 0.1):
    """Matched-null audit with DLA-based circuit discovery.

    Steps:
      1. Compute DLA for all components; select top-10% as circuit.
      2. Ablate circuit and measure metric drop (observed faithfulness).
      3. Generate n_null null circuits (type-stratified) and measure their
         metric drops to build null distribution.
      4. Compute empirical p-value; if it hits floor (1/n_null), use
         EVT/GPD for a tail p-value.
    """
    from transformer_lens import HookedTransformer

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: audit")
    print(f"  model={model_name}, task={task}, n_null={n_null}, ablation={ablation}")
    print(f"{'='*60}\n")

    device = get_device()
    tl_name = MODEL_MAP[model_name]
    print(f"Loading model {tl_name} ...")
    model = HookedTransformer.from_pretrained(tl_name, device=device)
    model.eval()

    # Generate prompts (use fixed 200 for the audit)
    n_prompts = 200
    prompts = generate_ioi_prompts(n_prompts, seed=seed)
    tokens, correct, incorrect, last_pos = tokenize_ioi(model, prompts)
    print(f"Generated {n_prompts} IOI prompts")

    components = enumerate_components(model)
    hook_names = list({c.hook_name for c in components})

    # Cache
    print("Caching activations ...")
    act_cache = cache_activations(model, tokens, batch_size, hook_names)
    mean_cache = compute_mean_activations(act_cache)

    # Clean metric
    clean_logits = batched_forward(model, tokens, batch_size)
    clean_metric = logit_diff_metric(clean_logits, correct, incorrect, last_pos)
    print(f"Clean logit diff = {clean_metric:.4f}")

    # DLA-based circuit discovery (top 10%)
    print("Computing DLA scores ...")
    dla_scores = compute_dla_scores(model, tokens, correct, incorrect, last_pos, batch_size)
    sorted_components = sorted(components, key=lambda c: abs(dla_scores[c.label]), reverse=True)
    k = max(1, int(len(components) * dla_threshold))
    circuit = sorted_components[:k]
    circuit_labels = {c.label for c in circuit}
    print(f"Circuit size: {k} / {len(components)} components (top {dla_threshold*100:.0f}% by |DLA|)")
    print(f"  Top-5: {[c.label for c in circuit[:5]]}")

    # Type composition of circuit
    n_heads_in_circuit = sum(1 for c in circuit if c.kind == "head")
    n_mlps_in_circuit = sum(1 for c in circuit if c.kind == "mlp")
    print(f"  Circuit composition: {n_heads_in_circuit} heads, {n_mlps_in_circuit} MLPs")

    # --- Helper to ablate a set of components and measure drop ---
    def ablate_and_measure(comp_set: List[Component]) -> float:
        hooks = []
        for comp in comp_set:
            if ablation == "mean":
                hooks.append(make_mean_ablation_hook(comp, mean_cache))
            elif ablation == "zero":
                hooks.append(make_zero_ablation_hook(comp))
            elif ablation == "resampling":
                hooks.append(make_resampling_hook(comp, tokens, act_cache))
            else:
                raise ValueError(f"Unknown ablation type: {ablation}")
        abl_logits = batched_forward(model, tokens, batch_size, fwd_hooks=hooks)
        abl_metric = logit_diff_metric(abl_logits, correct, incorrect, last_pos)
        return clean_metric - abl_metric

    # Observed faithfulness (drop from ablating circuit)
    print("Computing observed circuit faithfulness ...")
    observed_drop = ablate_and_measure(circuit)
    print(f"  Observed drop = {observed_drop:.4f}")

    # Type-stratified null sampling
    print(f"Running {n_null} null samples (type-stratified) ...")
    non_circuit = [c for c in components if c.label not in circuit_labels]
    non_circuit_heads = [c for c in non_circuit if c.kind == "head"]
    non_circuit_mlps = [c for c in non_circuit if c.kind == "mlp"]

    rng = random.Random(seed + 1)
    null_drops = []
    t0 = time.time()

    for i in range(n_null):
        # Sample a null circuit with the same type composition
        null_heads = rng.sample(non_circuit_heads,
                                min(n_heads_in_circuit, len(non_circuit_heads)))
        null_mlps = rng.sample(non_circuit_mlps,
                               min(n_mlps_in_circuit, len(non_circuit_mlps)))
        null_circuit = null_heads + null_mlps
        drop = ablate_and_measure(null_circuit)
        null_drops.append(drop)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_null - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n_null}] last_drop={drop:.4f}, "
                  f"rate={rate:.1f} samples/s, ETA={eta:.0f}s")

    null_drops = np.array(null_drops)
    elapsed = time.time() - t0

    # Empirical p-value: fraction of null drops >= observed
    empirical_p = (null_drops >= observed_drop).mean()
    print(f"\nEmpirical p-value = {empirical_p:.6f} (based on {n_null} null samples)")

    # EVT/GPD tail p-value if empirical p hits floor
    gpd_p = None
    if empirical_p <= 1.0 / n_null:
        print("Empirical p at floor, attempting EVT/GPD fit ...")
        gpd_p = gpd_tail_pvalue(null_drops, observed_drop)
        if gpd_p is not None:
            print(f"  GPD tail p-value = {gpd_p:.2e}")
        else:
            print("  GPD fit failed or observed not in tail")

    summary = {
        "experiment": "audit",
        "model": model_name,
        "tl_name": tl_name,
        "task": task,
        "ablation": ablation,
        "n_prompts": n_prompts,
        "n_null": n_null,
        "dla_threshold": dla_threshold,
        "circuit_size": k,
        "circuit_heads": n_heads_in_circuit,
        "circuit_mlps": n_mlps_in_circuit,
        "circuit_labels": [c.label for c in circuit],
        "clean_logit_diff": clean_metric,
        "observed_drop": observed_drop,
        "null_mean": float(null_drops.mean()),
        "null_std": float(null_drops.std()),
        "null_max": float(null_drops.max()),
        "empirical_p": float(empirical_p),
        "gpd_p": gpd_p,
        "elapsed_seconds": elapsed,
        "null_drops": null_drops.tolist(),
    }

    out_file = output_dir / f"audit_{model_name}_{task}_{ablation}_n{n_null}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {out_file}")
    return summary


# ---------------------------------------------------------------------------
# Experiment 3: ablation-sensitivity
# ---------------------------------------------------------------------------

def run_ablation_sensitivity(model_name: str, task: str, n_null: int,
                             batch_size: int, seed: int, output_dir: Path):
    """Run the audit across all 3 ablation types for a single model-task cell."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: ablation-sensitivity")
    print(f"  model={model_name}, task={task}, n_null={n_null}")
    print(f"  ablation types: {ABLATION_TYPES}")
    print(f"{'='*60}\n")

    all_results = {}
    for abl_type in ABLATION_TYPES:
        print(f"\n--- Ablation type: {abl_type} ---")
        result = run_audit(
            model_name=model_name,
            task=task,
            n_null=n_null,
            ablation=abl_type,
            batch_size=batch_size,
            seed=seed,
            output_dir=output_dir,
        )
        all_results[abl_type] = {
            "observed_drop": result["observed_drop"],
            "null_mean": result["null_mean"],
            "null_std": result["null_std"],
            "empirical_p": result["empirical_p"],
            "gpd_p": result["gpd_p"],
        }

    # Summary comparison
    summary = {
        "experiment": "ablation-sensitivity",
        "model": model_name,
        "task": task,
        "n_null": n_null,
        "results_by_ablation": all_results,
    }

    out_file = output_dir / f"ablation_sensitivity_{model_name}_{task}_n{n_null}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("ABLATION SENSITIVITY SUMMARY")
    print(f"{'='*60}")
    for abl_type, res in all_results.items():
        print(f"  {abl_type:12s}  drop={res['observed_drop']:.4f}  "
              f"null_mean={res['null_mean']:.4f}  p={res['empirical_p']:.6f}"
              + (f"  gpd_p={res['gpd_p']:.2e}" if res['gpd_p'] else ""))
    print(f"\nResults saved to {out_file}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AAAI-27 conflict monitoring experiments (Modal-free local runner)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for JSON output files (default: results/)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for forward passes (default: 32)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    sub = parser.add_subparsers(dest="command", required=True)

    # norm-vs-drop
    p_nvd = sub.add_parser("norm-vs-drop",
                           help="Component norm vs. ablation drop correlation")
    p_nvd.add_argument("--model", type=str, required=True, choices=list(MODEL_MAP.keys()),
                       help="Model shorthand")
    p_nvd.add_argument("--task", type=str, default="ioi",
                       help="Task name (default: ioi)")
    p_nvd.add_argument("--n", type=int, default=200,
                       help="Number of prompts (default: 200)")

    # audit
    p_audit = sub.add_parser("audit",
                             help="Matched-null circuit audit")
    p_audit.add_argument("--model", type=str, required=True, choices=list(MODEL_MAP.keys()),
                         help="Model shorthand")
    p_audit.add_argument("--task", type=str, default="ioi",
                         help="Task name (default: ioi)")
    p_audit.add_argument("--n-null", type=int, default=1000,
                         help="Number of null samples (default: 1000)")
    p_audit.add_argument("--ablation", type=str, default="mean",
                         choices=list(ABLATION_TYPES),
                         help="Ablation type (default: mean)")

    # ablation-sensitivity
    p_abl = sub.add_parser("ablation-sensitivity",
                           help="Audit across all ablation types")
    p_abl.add_argument("--model", type=str, required=True, choices=list(MODEL_MAP.keys()),
                        help="Model shorthand")
    p_abl.add_argument("--task", type=str, default="ioi",
                       help="Task name (default: ioi)")
    p_abl.add_argument("--n-null", type=int, default=500,
                       help="Number of null samples per ablation type (default: 500)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.command == "norm-vs-drop":
        run_norm_vs_drop(
            model_name=args.model,
            task=args.task,
            n=args.n,
            batch_size=args.batch_size,
            seed=args.seed,
            output_dir=output_dir,
        )
    elif args.command == "audit":
        run_audit(
            model_name=args.model,
            task=args.task,
            n_null=args.n_null,
            ablation=args.ablation,
            batch_size=args.batch_size,
            seed=args.seed,
            output_dir=output_dir,
        )
    elif args.command == "ablation-sensitivity":
        run_ablation_sensitivity(
            model_name=args.model,
            task=args.task,
            n_null=args.n_null,
            batch_size=args.batch_size,
            seed=args.seed,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
