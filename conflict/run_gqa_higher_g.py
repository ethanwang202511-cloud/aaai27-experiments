#!/usr/bin/env python3
"""
Experiment 3: Higher-G GQA Replication
AAAI-27 — Matched-Null Audit at Per-Head vs KV-Group Levels

Motivation: GQA architecture-matched ablation on Qwen-2.5-0.5B (G=2) failed
because sigma_null=0 with only 2 KV groups per layer. We need G>=4 for
informative nulls.

Primary model: Llama-3.1-8B (32 query heads, 8 KV heads, G=4, 32 layers).
Fallback: Mistral-7B.

Task: IOI (Indirect Object Identification)
"""

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import transformer_lens as tl
from transformer_lens import HookedTransformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "llama3": "meta-llama/Meta-Llama-3.1-8B",
    "mistral": "mistralai/Mistral-7B-v0.1",
}

NAMES_POOL = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Peter",
    "Quinn", "Ryan", "Sara", "Tom",
]

PLACES_POOL = [
    "store", "park", "school", "library", "museum", "market", "beach",
    "cafe", "station", "hospital", "garden", "office", "theater", "church",
    "restaurant", "gym", "hotel", "airport", "zoo", "mall",
]


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# IOI prompt generation
# ---------------------------------------------------------------------------
def generate_ioi_prompts(n: int, seed: int) -> list[dict]:
    """Generate IOI prompts:
    'When {IO} and {S} went to the {place}, {IO} gave a gift to'
    correct = IO, incorrect = S.
    """
    rng = random.Random(seed)
    prompts = []
    for _ in range(n):
        io, s = rng.sample(NAMES_POOL, 2)
        place = rng.choice(PLACES_POOL)
        text = f"When {io} and {s} went to the {place}, {io} gave a gift to"
        prompts.append({
            "text": text,
            "correct": f" {s}",   # IO gave gift TO the Subject
            "incorrect": f" {io}",
            "io": io,
            "s": s,
        })
    # NOTE on IOI semantics: "IO gave a gift to ___" — the answer is S
    # (the other person), so correct=S, incorrect=IO.  This matches the
    # standard IOI task definition (the indirect object is the *recipient*).
    return prompts


# ---------------------------------------------------------------------------
# GQA group utilities
# ---------------------------------------------------------------------------
def get_gqa_info(model: HookedTransformer) -> dict:
    """Extract GQA configuration from model."""
    cfg = model.cfg
    n_heads = cfg.n_heads
    # TransformerLens stores n_key_value_heads if GQA; defaults to n_heads (MHA)
    n_kv_heads = getattr(cfg, "n_key_value_heads", None)
    if n_kv_heads is None or n_kv_heads == 0:
        n_kv_heads = n_heads  # MHA fallback
    group_size = n_heads // n_kv_heads  # queries per KV group
    n_groups = n_kv_heads
    log.info(
        f"GQA config: n_heads={n_heads}, n_kv_heads={n_kv_heads}, "
        f"G={group_size}, n_groups={n_groups}"
    )
    return {
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "group_size": group_size,
        "n_groups": n_groups,
    }


def head_to_kv_group(head_idx: int, gqa: dict) -> int:
    """Map a query-head index to its KV group index."""
    return head_idx // gqa["group_size"]


# ---------------------------------------------------------------------------
# DLA (Direct Logit Attribution)
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_dla(
    model: HookedTransformer,
    prompts: list[dict],
    batch_size: int,
    device: str,
) -> dict:
    """Compute per-head DLA scores on the IOI task.

    Returns dict with:
      - head_scores: np.array [n_layers, n_heads]  (mean DLA across prompts)
      - mlp_scores: np.array [n_layers]
      - logit_diffs: list of per-prompt logit differences
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    head_scores_accum = np.zeros((n_layers, n_heads), dtype=np.float64)
    mlp_scores_accum = np.zeros(n_layers, dtype=np.float64)
    logit_diffs = []
    count = 0

    W_U = model.W_U  # [d_model, d_vocab]

    for batch_start in range(0, len(prompts), batch_size):
        batch = prompts[batch_start : batch_start + batch_size]
        texts = [p["text"] for p in batch]
        tokens = model.to_tokens(texts, prepend_bos=True)  # [B, seq_len]
        B, seq_len = tokens.shape
        last_pos = seq_len - 1

        # Get correct/incorrect token ids
        correct_ids = []
        incorrect_ids = []
        for p in batch:
            c_id = model.to_single_token(p["correct"])
            i_id = model.to_single_token(p["incorrect"])
            correct_ids.append(c_id)
            incorrect_ids.append(i_id)

        correct_ids = torch.tensor(correct_ids, device=device)
        incorrect_ids = torch.tensor(incorrect_ids, device=device)

        # Compute logit diff direction per prompt: W_U[:, correct] - W_U[:, incorrect]
        # Shape: [B, d_model]
        diff_dir = W_U[:, correct_ids].T - W_U[:, incorrect_ids].T  # [B, d_model]

        # Run with cache
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: (
                "hook_result" in name or  # attention head outputs
                "hook_mlp_out" in name    # MLP outputs
            ),
        )

        for layer in range(n_layers):
            # Head outputs at last position: [B, n_heads, d_model]
            head_out = cache[f"blocks.{layer}.attn.hook_result"][:, last_pos, :, :]

            # DLA per head: head_out @ diff_dir for each prompt
            # head_out: [B, n_heads, d_model], diff_dir: [B, d_model]
            # -> project: [B, n_heads]
            dla = torch.einsum("bhd,bd->bh", head_out, diff_dir)
            head_scores_accum[layer] += dla.cpu().numpy().sum(axis=0)

            # MLP output at last position: [B, d_model]
            mlp_out = cache[f"blocks.{layer}.hook_mlp_out"][:, last_pos, :]
            mlp_dla = torch.einsum("bd,bd->b", mlp_out, diff_dir)
            mlp_scores_accum[layer] += mlp_dla.cpu().numpy().sum()

        # Overall logit diff
        logits = model(tokens)[:, last_pos, :]  # [B, d_vocab]
        for i in range(B):
            ld = (logits[i, correct_ids[i]] - logits[i, incorrect_ids[i]]).item()
            logit_diffs.append(ld)

        count += B
        del cache
        if device == "cuda":
            torch.cuda.empty_cache()

    head_scores = head_scores_accum / count
    mlp_scores = mlp_scores_accum / count

    log.info(
        f"DLA complete: mean logit diff = {np.mean(logit_diffs):.4f}, "
        f"head score range = [{head_scores.min():.4f}, {head_scores.max():.4f}]"
    )

    return {
        "head_scores": head_scores,
        "mlp_scores": mlp_scores,
        "logit_diffs": logit_diffs,
    }


# ---------------------------------------------------------------------------
# Circuit selection
# ---------------------------------------------------------------------------
def select_circuit_top_k(
    head_scores: np.ndarray,
    mlp_scores: np.ndarray,
    fraction: float = 0.10,
    gqa: Optional[dict] = None,
    level: str = "head",
) -> dict:
    """Select top-fraction components by |DLA| score.

    level='head': rank individual heads and MLPs.
    level='kv_group': sum DLA within KV groups, rank groups + MLPs.
    """
    n_layers, n_heads = head_scores.shape

    if level == "head":
        # Flatten heads
        head_entries = []
        for l in range(n_layers):
            for h in range(n_heads):
                head_entries.append((abs(head_scores[l, h]), "head", l, h))

        mlp_entries = []
        for l in range(n_layers):
            mlp_entries.append((abs(mlp_scores[l]), "mlp", l, -1))

        all_entries = head_entries + mlp_entries
        all_entries.sort(key=lambda x: x[0], reverse=True)
        k = max(1, int(len(all_entries) * fraction))
        selected = all_entries[:k]

        circuit_heads = [(e[2], e[3]) for e in selected if e[1] == "head"]
        circuit_mlps = [e[2] for e in selected if e[1] == "mlp"]

        return {
            "heads": circuit_heads,
            "mlps": circuit_mlps,
            "level": "head",
            "n_selected": len(selected),
            "n_total": len(all_entries),
        }

    elif level == "kv_group":
        assert gqa is not None
        n_groups = gqa["n_groups"]
        group_size = gqa["group_size"]

        # Compute group-level DLA: sum over query heads in each group
        group_scores = np.zeros((n_layers, n_groups), dtype=np.float64)
        for l in range(n_layers):
            for g in range(n_groups):
                start_h = g * group_size
                end_h = start_h + group_size
                group_scores[l, g] = head_scores[l, start_h:end_h].sum()

        group_entries = []
        for l in range(n_layers):
            for g in range(n_groups):
                group_entries.append((abs(group_scores[l, g]), "group", l, g))

        mlp_entries = []
        for l in range(n_layers):
            mlp_entries.append((abs(mlp_scores[l]), "mlp", l, -1))

        all_entries = group_entries + mlp_entries
        all_entries.sort(key=lambda x: x[0], reverse=True)
        k = max(1, int(len(all_entries) * fraction))
        selected = all_entries[:k]

        circuit_groups = [(e[2], e[3]) for e in selected if e[1] == "group"]
        circuit_mlps = [e[2] for e in selected if e[1] == "mlp"]

        # Expand groups to heads
        circuit_heads = []
        for l, g in circuit_groups:
            for h in range(g * group_size, (g + 1) * group_size):
                circuit_heads.append((l, h))

        return {
            "groups": circuit_groups,
            "heads": circuit_heads,
            "mlps": circuit_mlps,
            "group_scores": group_scores,
            "level": "kv_group",
            "n_selected": len(selected),
            "n_total": len(all_entries),
        }
    else:
        raise ValueError(f"Unknown level: {level}")


# ---------------------------------------------------------------------------
# Zero-ablation measurement
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_logit_diff_with_ablation(
    model: HookedTransformer,
    prompts: list[dict],
    ablate_heads: list[tuple[int, int]],
    ablate_mlps: list[int],
    batch_size: int,
    device: str,
) -> float:
    """Measure mean logit diff when specified heads and MLPs are zero-ablated."""

    ablate_head_set = set(ablate_heads)
    ablate_mlp_set = set(ablate_mlps)

    def ablate_head_hook(value, hook):
        layer = hook.layer()
        for l, h in ablate_head_set:
            if l == layer:
                value[:, :, h, :] = 0.0
        return value

    def ablate_mlp_hook(value, hook):
        layer = hook.layer()
        if layer in ablate_mlp_set:
            value[:] = 0.0
        return value

    # Build hook list
    fwd_hooks = []
    # Determine which layers need head ablation
    head_layers = set(l for l, h in ablate_head_set)
    for layer in head_layers:
        fwd_hooks.append(
            (f"blocks.{layer}.attn.hook_result", ablate_head_hook)
        )
    for layer in ablate_mlp_set:
        fwd_hooks.append(
            (f"blocks.{layer}.hook_mlp_out", ablate_mlp_hook)
        )

    logit_diffs = []
    for batch_start in range(0, len(prompts), batch_size):
        batch = prompts[batch_start : batch_start + batch_size]
        texts = [p["text"] for p in batch]
        tokens = model.to_tokens(texts, prepend_bos=True)
        B, seq_len = tokens.shape
        last_pos = seq_len - 1

        correct_ids = [model.to_single_token(p["correct"]) for p in batch]
        incorrect_ids = [model.to_single_token(p["incorrect"]) for p in batch]

        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=fwd_hooks,
        )[:, last_pos, :]

        for i in range(B):
            ld = (logits[i, correct_ids[i]] - logits[i, incorrect_ids[i]]).item()
            logit_diffs.append(ld)

        if device == "cuda":
            torch.cuda.empty_cache()

    return float(np.mean(logit_diffs))


# ---------------------------------------------------------------------------
# Null sampling
# ---------------------------------------------------------------------------
def sample_head_null(
    circuit_heads: list[tuple[int, int]],
    circuit_mlps: list[int],
    n_layers: int,
    n_heads: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Type-stratified null: same count of heads from non-circuit heads,
    same count of MLPs from non-circuit MLPs."""
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    non_circuit_heads = [x for x in all_heads if x not in set(circuit_heads)]
    non_circuit_mlps = [l for l in range(n_layers) if l not in set(circuit_mlps)]

    n_h = min(len(circuit_heads), len(non_circuit_heads))
    n_m = min(len(circuit_mlps), len(non_circuit_mlps))

    null_heads = rng.sample(non_circuit_heads, n_h)
    null_mlps = rng.sample(non_circuit_mlps, n_m) if n_m > 0 else []

    return null_heads, null_mlps


def sample_kv_group_null(
    circuit_groups: list[tuple[int, int]],
    circuit_mlps: list[int],
    n_layers: int,
    n_groups: int,
    group_size: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Group-matched null: for each circuit KV group, sample a different group
    from the same layer if possible, else from a nearby layer."""
    circuit_group_set = set(circuit_groups)
    null_groups = []

    for l, g in circuit_groups:
        # Try same layer first
        candidates = [(l, gp) for gp in range(n_groups)
                       if gp != g and (l, gp) not in circuit_group_set
                       and (l, gp) not in set(null_groups)]
        if not candidates:
            # Try nearby layers (within +/- 3)
            for dl in range(1, n_layers):
                for direction in [1, -1]:
                    nl = l + dl * direction
                    if 0 <= nl < n_layers:
                        more = [(nl, gp) for gp in range(n_groups)
                                if (nl, gp) not in circuit_group_set
                                and (nl, gp) not in set(null_groups)]
                        candidates.extend(more)
                if candidates:
                    break

        if candidates:
            null_groups.append(rng.choice(candidates))

    # Expand groups to heads
    null_heads = []
    for l, g in null_groups:
        for h in range(g * group_size, (g + 1) * group_size):
            null_heads.append((l, h))

    # MLPs — same as head-level
    non_circuit_mlps = [l for l in range(n_layers) if l not in set(circuit_mlps)]
    n_m = min(len(circuit_mlps), len(non_circuit_mlps))
    null_mlps = rng.sample(non_circuit_mlps, n_m) if n_m > 0 else []

    return null_heads, null_mlps


# ---------------------------------------------------------------------------
# Matched-null audit
# ---------------------------------------------------------------------------
def run_matched_null_audit(
    model: HookedTransformer,
    prompts: list[dict],
    circuit: dict,
    gqa: dict,
    n_null_samples: int,
    batch_size: int,
    device: str,
    seed: int,
    level: str,
) -> dict:
    """Run matched-null audit at the given level."""
    log.info(f"=== Matched-null audit: level={level} ===")
    rng = random.Random(seed)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Baseline (no ablation)
    log.info("Computing baseline logit diff (no ablation)...")
    baseline_ld = measure_logit_diff_with_ablation(
        model, prompts, [], [], batch_size, device
    )
    log.info(f"Baseline logit diff: {baseline_ld:.4f}")

    # Circuit ablation
    log.info(f"Computing circuit ablation ({len(circuit['heads'])} heads, {len(circuit['mlps'])} MLPs)...")
    circuit_ld = measure_logit_diff_with_ablation(
        model, prompts, circuit["heads"], circuit["mlps"], batch_size, device
    )
    circuit_drop = baseline_ld - circuit_ld
    log.info(f"Circuit ablated logit diff: {circuit_ld:.4f}, drop: {circuit_drop:.4f}")

    # Sanity check with small n first
    log.info("Running sigma_null sanity check with n=10...")
    null_drops_check = []
    per_layer_null_drops = defaultdict(list)  # only for kv_group level

    for i in range(min(10, n_null_samples)):
        if level == "head":
            null_heads, null_mlps = sample_head_null(
                circuit["heads"], circuit["mlps"], n_layers, n_heads, rng
            )
        else:
            null_heads, null_mlps = sample_kv_group_null(
                circuit["groups"], circuit["mlps"],
                n_layers, gqa["n_groups"], gqa["group_size"], rng
            )

        null_ld = measure_logit_diff_with_ablation(
            model, prompts, null_heads, null_mlps, batch_size, device
        )
        null_drops_check.append(baseline_ld - null_ld)

    sigma_check = float(np.std(null_drops_check))
    log.info(f"Sanity check sigma_null (n=10): {sigma_check:.6f}")
    if sigma_check < 1e-8:
        log.warning(
            f"DEGENERATE: sigma_null ~ 0 at level={level}. "
            "Null samples have no variance. Audit is uninformative."
        )
        return {
            "level": level,
            "baseline_ld": baseline_ld,
            "circuit_ld": circuit_ld,
            "circuit_drop": circuit_drop,
            "sigma_null_check": sigma_check,
            "degenerate": True,
            "n_null_run": 10,
            "null_drops": null_drops_check,
        }

    # Full null sampling
    log.info(f"Running full null sampling (n={n_null_samples})...")
    # Reset RNG for reproducibility — re-seed so the first 10 are regenerated
    rng = random.Random(seed)
    null_drops = []
    per_layer_sigma = {}

    for i in range(n_null_samples):
        if level == "head":
            null_heads, null_mlps = sample_head_null(
                circuit["heads"], circuit["mlps"], n_layers, n_heads, rng
            )
        else:
            null_heads, null_mlps = sample_kv_group_null(
                circuit["groups"], circuit["mlps"],
                n_layers, gqa["n_groups"], gqa["group_size"], rng
            )

        null_ld = measure_logit_diff_with_ablation(
            model, prompts, null_heads, null_mlps, batch_size, device
        )
        null_drops.append(baseline_ld - null_ld)

        if (i + 1) % 50 == 0:
            running_mean = np.mean(null_drops)
            running_std = np.std(null_drops)
            log.info(
                f"  null sample {i+1}/{n_null_samples}: "
                f"mean={running_mean:.4f}, std={running_std:.4f}"
            )

    null_drops = np.array(null_drops)
    null_mean = float(np.mean(null_drops))
    null_std = float(np.std(null_drops))

    # z-score and empirical p
    z_score = (circuit_drop - null_mean) / null_std if null_std > 0 else float("inf")
    empirical_p = float(np.mean(null_drops >= circuit_drop))

    log.info(f"Results [{level}]:")
    log.info(f"  circuit_drop = {circuit_drop:.4f}")
    log.info(f"  null_mean    = {null_mean:.4f}")
    log.info(f"  null_std     = {null_std:.4f}")
    log.info(f"  z(C)         = {z_score:.4f}")
    log.info(f"  empirical_p  = {empirical_p:.6f}")

    # Per-layer sigma breakdown for KV-group level
    if level == "kv_group":
        # Compute per-layer sigma by running dedicated single-layer ablations
        # (Skip this for now — use the group_scores from circuit selection)
        circuit_groups = circuit.get("groups", [])
        layer_groups = defaultdict(list)
        for l, g in circuit_groups:
            layer_groups[l].append(g)
        per_layer_sigma = {
            f"layer_{l}": {
                "n_circuit_groups": len(gs),
                "groups": gs,
            }
            for l, gs in sorted(layer_groups.items())
        }

    result = {
        "level": level,
        "baseline_ld": baseline_ld,
        "circuit_ld": circuit_ld,
        "circuit_drop": circuit_drop,
        "null_mean": null_mean,
        "null_std": null_std,
        "sigma_null": null_std,
        "z_score": z_score,
        "empirical_p": empirical_p,
        "n_null_samples": n_null_samples,
        "degenerate": False,
        "n_circuit_heads": len(circuit["heads"]),
        "n_circuit_mlps": len(circuit["mlps"]),
    }
    if level == "kv_group":
        result["n_circuit_groups"] = len(circuit.get("groups", []))
        result["per_layer_sigma"] = per_layer_sigma

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Exp 3: Higher-G GQA Matched-Null Audit"
    )
    parser.add_argument(
        "--model", choices=["llama3", "mistral"], default="llama3",
        help="Model to use (default: llama3)"
    )
    parser.add_argument(
        "--n-null-samples", type=int, default=500,
        help="Number of null samples (default: 500)"
    )
    parser.add_argument(
        "--n-prompts", type=int, default=200,
        help="Number of IOI prompts (default: 200)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: results_<model>.json)"
    )
    parser.add_argument(
        "--circuit-fraction", type=float, default=0.10,
        help="Fraction of components to include in circuit (default: 0.10)"
    )
    args = parser.parse_args()

    t_start = time.time()
    device = get_device()
    log.info(f"Device: {device}")
    log.info(f"Model: {args.model} -> {MODEL_REGISTRY[args.model]}")

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ---- Load model ----
    log.info("Loading model...")
    model_name = MODEL_REGISTRY[args.model]
    try:
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            dtype=torch.float16 if device != "cpu" else torch.float32,
        )
    except Exception as e:
        if args.model == "llama3":
            log.warning(f"Failed to load Llama-3.1-8B: {e}")
            log.info("Falling back to Mistral-7B...")
            model_name = MODEL_REGISTRY["mistral"]
            model = HookedTransformer.from_pretrained(
                model_name,
                device=device,
                dtype=torch.float16 if device != "cpu" else torch.float32,
            )
        else:
            raise

    log.info(f"Model loaded: {model_name}")
    log.info(f"  n_layers={model.cfg.n_layers}, n_heads={model.cfg.n_heads}, "
             f"d_model={model.cfg.d_model}")

    # ---- GQA info ----
    gqa = get_gqa_info(model)
    if gqa["group_size"] < 2:
        log.warning("Model uses MHA (group_size=1), not GQA. KV-group audit = per-head audit.")
    if gqa["n_groups"] < 4:
        log.warning(f"Only {gqa['n_groups']} KV groups — may still get degenerate nulls.")

    # ---- Generate IOI prompts ----
    log.info(f"Generating {args.n_prompts} IOI prompts...")
    prompts = generate_ioi_prompts(args.n_prompts, args.seed)
    log.info(f"Example prompt: {prompts[0]['text']}")
    log.info(f"  correct={prompts[0]['correct']!r}, incorrect={prompts[0]['incorrect']!r}")

    # ---- DLA ----
    log.info("Running Direct Logit Attribution...")
    dla_results = run_dla(model, prompts, args.batch_size, device)
    head_scores = dla_results["head_scores"]
    mlp_scores = dla_results["mlp_scores"]

    # ---- Select circuits ----
    log.info("Selecting per-head circuit (top 10% by |DLA|)...")
    head_circuit = select_circuit_top_k(
        head_scores, mlp_scores,
        fraction=args.circuit_fraction,
        level="head",
    )
    log.info(f"Per-head circuit: {len(head_circuit['heads'])} heads, {len(head_circuit['mlps'])} MLPs "
             f"(out of {head_circuit['n_total']} total)")

    log.info("Selecting KV-group circuit (top 10% by |DLA|)...")
    group_circuit = select_circuit_top_k(
        head_scores, mlp_scores,
        fraction=args.circuit_fraction,
        gqa=gqa,
        level="kv_group",
    )
    log.info(f"KV-group circuit: {len(group_circuit['groups'])} groups "
             f"({len(group_circuit['heads'])} heads), {len(group_circuit['mlps'])} MLPs "
             f"(out of {group_circuit['n_total']} total)")

    # ---- Matched-null audits ----
    log.info("\n" + "="*60)
    log.info("AUDIT 1: Per-head matched null")
    log.info("="*60)
    head_audit = run_matched_null_audit(
        model, prompts, head_circuit, gqa,
        n_null_samples=args.n_null_samples,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed + 1000,
        level="head",
    )

    log.info("\n" + "="*60)
    log.info("AUDIT 2: KV-group matched null")
    log.info("="*60)
    group_audit = run_matched_null_audit(
        model, prompts, group_circuit, gqa,
        n_null_samples=args.n_null_samples,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed + 2000,
        level="kv_group",
    )

    # ---- Comparison ----
    log.info("\n" + "="*60)
    log.info("COMPARISON: Per-head vs KV-group")
    log.info("="*60)

    head_degen = head_audit.get("degenerate", False)
    group_degen = group_audit.get("degenerate", False)

    comparison = {
        "head_z": head_audit.get("z_score"),
        "group_z": group_audit.get("z_score"),
        "head_sigma_null": head_audit.get("sigma_null"),
        "group_sigma_null": group_audit.get("sigma_null"),
        "head_p": head_audit.get("empirical_p"),
        "group_p": group_audit.get("empirical_p"),
        "head_degenerate": head_degen,
        "group_degenerate": group_degen,
    }

    if not head_degen and not group_degen:
        z_diff = (head_audit["z_score"] - group_audit["z_score"])
        comparison["z_diff_head_minus_group"] = z_diff
        verdict = "SAME"
        if abs(z_diff) > 1.0:
            verdict = "HEAD_HIGHER" if z_diff > 0 else "GROUP_HIGHER"
        comparison["verdict"] = verdict
        log.info(f"  head z(C)  = {head_audit['z_score']:.3f} (p={head_audit['empirical_p']:.4f})")
        log.info(f"  group z(C) = {group_audit['z_score']:.3f} (p={group_audit['empirical_p']:.4f})")
        log.info(f"  verdict: {verdict} (z_diff={z_diff:.3f})")
    else:
        comparison["verdict"] = "DEGENERATE"
        log.info("  One or both audits degenerate. Cannot compare.")

    # ---- Output ----
    elapsed = time.time() - t_start
    output = {
        "experiment": "higher_g_gqa_replication",
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "config": {
            "model": model_name,
            "model_alias": args.model,
            "n_prompts": args.n_prompts,
            "n_null_samples": args.n_null_samples,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "circuit_fraction": args.circuit_fraction,
            "device": device,
        },
        "gqa": {
            "n_heads": gqa["n_heads"],
            "n_kv_heads": gqa["n_kv_heads"],
            "group_size": gqa["group_size"],
            "n_groups": gqa["n_groups"],
        },
        "dla": {
            "mean_logit_diff": float(np.mean(dla_results["logit_diffs"])),
            "std_logit_diff": float(np.std(dla_results["logit_diffs"])),
        },
        "per_head_audit": head_audit,
        "kv_group_audit": group_audit,
        "comparison": comparison,
    }

    # Clean up numpy arrays for JSON serialization
    def sanitize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    output = sanitize(output)

    out_path = args.output or f"results_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nResults written to {out_path}")
    log.info(f"Total elapsed: {elapsed:.1f}s")

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT 3 SUMMARY")
    print("="*60)
    print(f"Model:        {model_name}")
    print(f"GQA:          G={gqa['group_size']}, {gqa['n_groups']} KV groups")
    print(f"Prompts:      {args.n_prompts}")
    print(f"Null samples: {args.n_null_samples}")
    print(f"Device:       {device}")
    print()
    print(f"Per-head audit:")
    print(f"  sigma_null = {head_audit.get('sigma_null', 'N/A')}")
    print(f"  z(C)       = {head_audit.get('z_score', 'N/A')}")
    print(f"  p          = {head_audit.get('empirical_p', 'N/A')}")
    print(f"  degenerate = {head_audit.get('degenerate', False)}")
    print()
    print(f"KV-group audit:")
    print(f"  sigma_null = {group_audit.get('sigma_null', 'N/A')}")
    print(f"  z(C)       = {group_audit.get('z_score', 'N/A')}")
    print(f"  p          = {group_audit.get('empirical_p', 'N/A')}")
    print(f"  degenerate = {group_audit.get('degenerate', False)}")
    print()
    print(f"Verdict:       {comparison.get('verdict', 'N/A')}")
    print(f"Elapsed:       {elapsed:.1f}s")
    print("="*60)


if __name__ == "__main__":
    main()
