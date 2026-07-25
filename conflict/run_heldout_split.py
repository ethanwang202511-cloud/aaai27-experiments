#!/usr/bin/env python3
"""
Experiment 4: Expanded Held-Out-Split Audit
============================================
AAAI-27 — Selection-bias audit for mechanistic interpretability circuits.

Problem: When circuit selection and testing use the same data, p-values can
be deflated (selection bias). The original audit used only 1 cell. This
script expands to 10 cells spanning the verdict space.

Protocol per cell:
  1. Generate 200 prompts, split 50/50 into D_select and D_test (fixed seed).
  2. DLA attribution on D_select → top-10% circuit.
  3. Matched-null audit on D_test (type-stratified, n=1000 null samples,
     three ablation types: zero, mean, resampling).
  4. Record z(C), empirical p, BH-FDR at q=0.05.
  5. Non-split baseline: same cell, full 200 prompts for both.
  6. Compare rejection rates.

Usage:
  python run_heldout_split.py --cells ioi/gpt2,ioi/qwen2.5 --n-prompts 200 \
      --n-null 1000 --seed 42 --batch-size 32 --ablation zero,mean,resampling

Output: JSON to stdout (or --output path).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import random
import itertools
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import transformer_lens as tl
from transformer_lens import HookedTransformer

# ---------------------------------------------------------------------------
# Constants & model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, str] = {
    "gpt2": "gpt2-small",
    "qwen2.5": "Qwen/Qwen2.5-0.5B",
    "gemma2": "gemma-2-2b",
    "llama3": "meta-llama/Meta-Llama-3.1-8B",
}

# IOI prompt template
IOI_TEMPLATE = "When {name1} and {name2} went to the {place}, {name3} gave a gift to"

IOI_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo", "Mary", "Nick", "Olivia", "Paul",
    "Quinn", "Rose", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
]

IOI_PLACES = [
    "store", "park", "library", "museum", "beach", "restaurant",
    "school", "hospital", "airport", "station", "market", "church",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CellSpec:
    task: str
    model_key: str

    @property
    def tl_name(self) -> str:
        return MODEL_REGISTRY[self.model_key]

    @property
    def label(self) -> str:
        return f"{self.task}/{self.model_key}"

    @classmethod
    def from_str(cls, s: str) -> "CellSpec":
        task, model_key = s.strip().split("/")
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key '{model_key}'. Choose from {list(MODEL_REGISTRY)}")
        return cls(task=task, model_key=model_key)


@dataclass
class IOIPrompt:
    """Single IOI prompt with metadata."""
    text: str
    io_name: str       # indirect object (correct answer)
    s_name: str        # subject (incorrect answer)
    io_token: int = -1
    s_token: int = -1


@dataclass
class CircuitComponent:
    """A single component in the circuit (head or MLP)."""
    layer: int
    comp_type: str  # "head" or "mlp"
    head_idx: Optional[int] = None  # None for MLP
    dla_score: float = 0.0

    @property
    def hook_name(self) -> str:
        if self.comp_type == "head":
            return f"blocks.{self.layer}.attn.hook_result"
        return f"blocks.{self.layer}.hook_mlp_out"


@dataclass
class AuditResult:
    """Result for one ablation type on one split."""
    ablation_type: str
    baseline_metric: float
    ablated_metric: float
    effect_size: float
    null_effects: list[float] = field(default_factory=list)
    z_score: float = 0.0
    empirical_p: float = 1.0
    bh_fdr_pass: bool = False


@dataclass
class CellResult:
    """Complete result for one cell."""
    cell: str
    n_prompts: int
    n_select: int
    n_test: int
    n_null: int
    circuit_size: int
    total_components: int
    split_results: dict[str, dict] = field(default_factory=dict)
    full_results: dict[str, dict] = field(default_factory=dict)
    jaccard: float = 0.0
    split_circuit_components: list[str] = field(default_factory=list)
    full_circuit_components: list[str] = field(default_factory=list)
    wall_time_s: float = 0.0


# ---------------------------------------------------------------------------
# IOI prompt generation
# ---------------------------------------------------------------------------

def generate_ioi_prompts(n: int, rng: random.Random) -> list[IOIPrompt]:
    """Generate n IOI prompts with random name/place combinations."""
    prompts = []
    for _ in range(n):
        # Pick two distinct names
        io_name, s_name = rng.sample(IOI_NAMES, 2)
        place = rng.choice(IOI_PLACES)
        text = IOI_TEMPLATE.format(
            name1=io_name, name2=s_name, name3=io_name, place=place
        )
        prompts.append(IOIPrompt(text=text, io_name=io_name, s_name=s_name))
    return prompts


def tokenize_prompts(
    model: HookedTransformer, prompts: list[IOIPrompt]
) -> list[IOIPrompt]:
    """Resolve IO and S token IDs for each prompt."""
    for p in prompts:
        # Get single-token ID for each name. Prepend space for subword models.
        io_tokens = model.to_tokens(f" {p.io_name}", prepend_bos=False).squeeze()
        s_tokens = model.to_tokens(f" {p.s_name}", prepend_bos=False).squeeze()
        # Take the first token if multi-token
        p.io_token = int(io_tokens[0]) if io_tokens.dim() > 0 else int(io_tokens)
        p.s_token = int(s_tokens[0]) if s_tokens.dim() > 0 else int(s_tokens)
    return prompts


# ---------------------------------------------------------------------------
# DLA attribution
# ---------------------------------------------------------------------------

def compute_dla_scores(
    model: HookedTransformer,
    prompts: list[IOIPrompt],
    batch_size: int,
    device: str,
) -> list[CircuitComponent]:
    """
    Compute Direct Logit Attribution for every head and MLP layer.

    For each component, DLA = component_output @ (W_U[:, correct] - W_U[:, incorrect])
    at the last token position. We average over all prompts.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Accumulate DLA scores: heads[layer][head], mlps[layer]
    head_scores = np.zeros((n_layers, n_heads))
    mlp_scores = np.zeros(n_layers)
    count = 0

    # W_U: [d_model, d_vocab]
    W_U = model.W_U.detach()  # [d_model, d_vocab]

    texts = [p.text for p in prompts]
    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        batch_texts = texts[batch_start : batch_start + batch_size]

        tokens = model.to_tokens(batch_texts, prepend_bos=True)
        # last_pos = seq_len - 1 for each prompt (they may differ in length
        # after tokenization, but to_tokens pads so we use the true last)
        # Actually HookedTransformer pads to same length. Find last non-pad.
        seq_len = tokens.shape[1]

        # Collect all hook names we need
        hook_names = []
        for layer in range(n_layers):
            hook_names.append(f"blocks.{layer}.attn.hook_result")
            hook_names.append(f"blocks.{layer}.hook_mlp_out")

        _, cache = model.run_with_cache(
            tokens, names_filter=lambda name: name in hook_names
        )

        for i, p in enumerate(batch_prompts):
            # Direction: W_U[:, io] - W_U[:, s]
            direction = W_U[:, p.io_token] - W_U[:, p.s_token]  # [d_model]

            for layer in range(n_layers):
                # Head contributions: hook_result is [batch, pos, n_heads, d_model]
                head_out = cache[f"blocks.{layer}.attn.hook_result"][i, -1]  # [n_heads, d_model]
                for head in range(n_heads):
                    score = (head_out[head] @ direction).item()
                    head_scores[layer, head] += score

                # MLP contribution: [batch, pos, d_model]
                mlp_out = cache[f"blocks.{layer}.hook_mlp_out"][i, -1]  # [d_model]
                score = (mlp_out @ direction).item()
                mlp_scores[layer] += score

            count += 1

        del cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Average
    head_scores /= count
    mlp_scores /= count

    # Build component list
    components = []
    for layer in range(n_layers):
        for head in range(n_heads):
            components.append(CircuitComponent(
                layer=layer, comp_type="head", head_idx=head,
                dla_score=float(head_scores[layer, head])
            ))
        components.append(CircuitComponent(
            layer=layer, comp_type="mlp", head_idx=None,
            dla_score=float(mlp_scores[layer])
        ))

    return components


def select_top_circuit(
    components: list[CircuitComponent], percentile: float = 10.0
) -> list[CircuitComponent]:
    """Select top percentile of components by absolute DLA score."""
    abs_scores = [abs(c.dla_score) for c in components]
    threshold = np.percentile(abs_scores, 100 - percentile)
    circuit = [c for c in components if abs(c.dla_score) >= threshold]
    # Sort by absolute DLA descending
    circuit.sort(key=lambda c: abs(c.dla_score), reverse=True)
    return circuit


def component_label(c: CircuitComponent) -> str:
    if c.comp_type == "head":
        return f"L{c.layer}H{c.head_idx}"
    return f"L{c.layer}_MLP"


# ---------------------------------------------------------------------------
# Ablation & metric computation
# ---------------------------------------------------------------------------

def compute_logit_diff(
    model: HookedTransformer,
    prompts: list[IOIPrompt],
    batch_size: int,
    hooks: Optional[list] = None,
) -> float:
    """
    Compute mean logit difference (logit[IO] - logit[S]) at the last position.
    Optionally with hooks applied for ablation.
    """
    texts = [p.text for p in prompts]
    total_diff = 0.0
    count = 0

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        batch_texts = texts[batch_start : batch_start + batch_size]

        if hooks:
            logits = model.run_with_hooks(
                model.to_tokens(batch_texts, prepend_bos=True),
                fwd_hooks=hooks,
            )
        else:
            logits = model(model.to_tokens(batch_texts, prepend_bos=True))

        # logits: [batch, seq, vocab]
        for i, p in enumerate(batch_prompts):
            diff = (logits[i, -1, p.io_token] - logits[i, -1, p.s_token]).item()
            total_diff += diff
            count += 1

        del logits
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return total_diff / count


def collect_mean_activations(
    model: HookedTransformer,
    prompts: list[IOIPrompt],
    circuit: list[CircuitComponent],
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Compute mean activations for each circuit component across prompts."""
    hook_names = list({c.hook_name for c in circuit})
    means: dict[str, torch.Tensor] = {}
    counts = 0

    texts = [p.text for p in prompts]
    for batch_start in range(0, len(prompts), batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]
        tokens = model.to_tokens(batch_texts, prepend_bos=True)
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda name: name in hook_names
        )
        bs = tokens.shape[0]
        for name in hook_names:
            act = cache[name].detach().float().sum(dim=0)  # sum over batch
            if name not in means:
                means[name] = act
            else:
                means[name] = means[name] + act
        counts += bs
        del cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    for name in means:
        means[name] = means[name] / counts

    return means


def make_ablation_hooks(
    circuit: list[CircuitComponent],
    ablation_type: str,
    mean_acts: Optional[dict[str, torch.Tensor]] = None,
    all_prompts: Optional[list[IOIPrompt]] = None,
    model: Optional[HookedTransformer] = None,
    batch_size_for_resample: int = 32,
    rng: Optional[random.Random] = None,
) -> list[tuple]:
    """
    Build a list of (hook_name, hook_fn) pairs for ablating the circuit.

    ablation_type:
      - "zero": set component output to 0 at all positions
      - "mean": replace with dataset-mean activation
      - "resampling": replace with activation from a random permutation
    """
    hooks = []

    # Group circuit components by hook_name for efficiency
    from collections import defaultdict
    hook_components: dict[str, list[CircuitComponent]] = defaultdict(list)
    for c in circuit:
        hook_components[c.hook_name].append(c)

    for hook_name, comps in hook_components.items():
        if ablation_type == "zero":
            def make_zero_hook(hname, components):
                def hook_fn(activation, hook):
                    for c in components:
                        if c.comp_type == "head":
                            activation[:, :, c.head_idx, :] = 0.0
                        else:
                            activation[:, :, :] = 0.0
                    return activation
                return hook_fn
            hooks.append((hook_name, make_zero_hook(hook_name, comps)))

        elif ablation_type == "mean":
            assert mean_acts is not None, "mean_acts required for mean ablation"
            def make_mean_hook(hname, components, mean_val):
                def hook_fn(activation, hook):
                    for c in components:
                        if c.comp_type == "head":
                            # mean_val shape: [seq, n_heads, d_model]
                            seq_len = min(activation.shape[1], mean_val.shape[0])
                            activation[:, :seq_len, c.head_idx, :] = mean_val[:seq_len, c.head_idx, :].to(activation.device)
                        else:
                            seq_len = min(activation.shape[1], mean_val.shape[0])
                            activation[:, :seq_len, :] = mean_val[:seq_len, :].to(activation.device)
                    return activation
                return hook_fn
            hooks.append((hook_name, make_mean_hook(hook_name, comps, mean_acts[hook_name])))

        elif ablation_type == "resampling":
            def make_resample_hook(hname, components):
                def hook_fn(activation, hook):
                    batch_size = activation.shape[0]
                    if batch_size <= 1:
                        return activation
                    # Random permutation within batch
                    perm = torch.randperm(batch_size, device=activation.device)
                    # Avoid identity permutation
                    while (perm == torch.arange(batch_size, device=activation.device)).all():
                        perm = torch.randperm(batch_size, device=activation.device)
                    for c in components:
                        if c.comp_type == "head":
                            activation[:, :, c.head_idx, :] = activation[perm, :, c.head_idx, :]
                        else:
                            activation[:, :, :] = activation[perm, :, :]
                    return activation
                return hook_fn
            hooks.append((hook_name, make_resample_hook(hook_name, comps)))

    return hooks


# ---------------------------------------------------------------------------
# Matched-null sampling
# ---------------------------------------------------------------------------

def sample_type_matched_null(
    all_components: list[CircuitComponent],
    circuit: list[CircuitComponent],
    rng: random.Random,
) -> list[CircuitComponent]:
    """
    Sample a random subset of components matching the circuit's type distribution.
    Type = head vs mlp, per layer.
    """
    from collections import Counter

    # Count types in circuit
    type_counts: Counter = Counter()
    for c in circuit:
        type_counts[c.comp_type] += 1

    # Partition all components by type
    by_type: dict[str, list[CircuitComponent]] = {"head": [], "mlp": []}
    circuit_set = {(c.layer, c.comp_type, c.head_idx) for c in circuit}
    for c in all_components:
        key = (c.layer, c.comp_type, c.head_idx)
        if key not in circuit_set:
            by_type[c.comp_type].append(c)

    null_sample = []
    for comp_type, count in type_counts.items():
        pool = by_type[comp_type]
        if len(pool) < count:
            # Not enough non-circuit components of this type; sample with replacement
            null_sample.extend(rng.choices(pool, k=count))
        else:
            null_sample.extend(rng.sample(pool, count))

    return null_sample


# ---------------------------------------------------------------------------
# Jaccard similarity
# ---------------------------------------------------------------------------

def jaccard_similarity(
    circuit_a: list[CircuitComponent], circuit_b: list[CircuitComponent]
) -> float:
    set_a = {(c.layer, c.comp_type, c.head_idx) for c in circuit_a}
    set_b = {(c.layer, c.comp_type, c.head_idx) for c in circuit_b}
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# BH-FDR correction
# ---------------------------------------------------------------------------

def bh_fdr(p_values: list[float], q: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR correction. Returns list of pass/fail."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    passes = [False] * m
    # Find largest k where p_(k) <= k/m * q
    max_k = -1
    for rank, (orig_idx, p) in enumerate(indexed, 1):
        if p <= rank / m * q:
            max_k = rank
    if max_k > 0:
        for rank, (orig_idx, p) in enumerate(indexed, 1):
            if rank <= max_k:
                passes[orig_idx] = True
    return passes


# ---------------------------------------------------------------------------
# Single cell experiment
# ---------------------------------------------------------------------------

def run_single_cell(
    cell: CellSpec,
    n_prompts: int,
    n_null: int,
    seed: int,
    batch_size: int,
    ablation_types: list[str],
    device: str,
) -> CellResult:
    """Run the full held-out-split audit for one cell."""
    t0 = time.time()
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    print(f"\n{'='*60}")
    print(f"Cell: {cell.label}")
    print(f"Model: {cell.tl_name}")
    print(f"{'='*60}")

    # --- Load model ---
    print(f"  Loading model {cell.tl_name}...")
    model = HookedTransformer.from_pretrained(
        cell.tl_name,
        device=device,
    )
    model.cfg.use_attn_result = True
    model.eval()

    # --- Generate prompts ---
    print(f"  Generating {n_prompts} prompts...")
    if cell.task == "ioi":
        prompts = generate_ioi_prompts(n_prompts, rng)
        prompts = tokenize_prompts(model, prompts)
    else:
        raise ValueError(f"Unknown task: {cell.task}")

    # --- Split ---
    n_select = n_prompts // 2
    n_test = n_prompts - n_select
    d_select = prompts[:n_select]
    d_test = prompts[n_select:]

    print(f"  D_select: {len(d_select)}, D_test: {len(d_test)}")

    # === SPLIT PROTOCOL ===
    print(f"  [Split] Running DLA on D_select...")
    all_components_select = compute_dla_scores(model, d_select, batch_size, device)
    circuit_split = select_top_circuit(all_components_select, percentile=10.0)
    print(f"  [Split] Circuit size: {len(circuit_split)} / {len(all_components_select)}")

    # Collect mean activations on D_test for mean ablation
    mean_acts_test = None
    if "mean" in ablation_types:
        print(f"  [Split] Computing mean activations on D_test...")
        mean_acts_test = collect_mean_activations(model, d_test, circuit_split, batch_size)

    # Baseline metric on D_test
    print(f"  [Split] Computing baseline logit diff on D_test...")
    baseline_split = compute_logit_diff(model, d_test, batch_size)
    print(f"  [Split] Baseline logit diff: {baseline_split:.4f}")

    split_results = {}
    for abl_type in ablation_types:
        print(f"  [Split] Ablation: {abl_type}")

        # Ablate circuit
        hooks = make_ablation_hooks(
            circuit_split, abl_type,
            mean_acts=mean_acts_test, rng=rng,
        )
        ablated_metric = compute_logit_diff(model, d_test, batch_size, hooks=hooks)
        effect = baseline_split - ablated_metric

        # Null distribution
        null_effects = []
        for null_i in range(n_null):
            if (null_i + 1) % 200 == 0:
                print(f"    Null sample {null_i + 1}/{n_null}")
            null_circuit = sample_type_matched_null(all_components_select, circuit_split, rng)

            # Collect mean acts for null circuit if needed
            null_mean_acts = None
            if abl_type == "mean":
                null_mean_acts = collect_mean_activations(model, d_test, null_circuit, batch_size)

            null_hooks = make_ablation_hooks(
                null_circuit, abl_type,
                mean_acts=null_mean_acts, rng=rng,
            )
            null_ablated = compute_logit_diff(model, d_test, batch_size, hooks=null_hooks)
            null_effects.append(baseline_split - null_ablated)

        # Statistics
        null_arr = np.array(null_effects)
        null_mean = null_arr.mean()
        null_std = null_arr.std()
        z = (effect - null_mean) / null_std if null_std > 0 else 0.0
        empirical_p = (np.sum(null_arr >= effect) + 1) / (len(null_arr) + 1)

        split_results[abl_type] = {
            "baseline_metric": baseline_split,
            "ablated_metric": ablated_metric,
            "effect_size": effect,
            "z_score": float(z),
            "empirical_p": float(empirical_p),
            "null_mean": float(null_mean),
            "null_std": float(null_std),
        }
        print(f"    Effect: {effect:.4f}, z={z:.2f}, p={empirical_p:.4f}")

    # === FULL (NON-SPLIT) PROTOCOL ===
    print(f"\n  [Full] Running DLA on all {n_prompts} prompts...")
    all_components_full = compute_dla_scores(model, prompts, batch_size, device)
    circuit_full = select_top_circuit(all_components_full, percentile=10.0)
    print(f"  [Full] Circuit size: {len(circuit_full)} / {len(all_components_full)}")

    mean_acts_full = None
    if "mean" in ablation_types:
        print(f"  [Full] Computing mean activations on full dataset...")
        mean_acts_full = collect_mean_activations(model, prompts, circuit_full, batch_size)

    baseline_full = compute_logit_diff(model, prompts, batch_size)
    print(f"  [Full] Baseline logit diff: {baseline_full:.4f}")

    full_results = {}
    for abl_type in ablation_types:
        print(f"  [Full] Ablation: {abl_type}")
        hooks = make_ablation_hooks(
            circuit_full, abl_type,
            mean_acts=mean_acts_full, rng=rng,
        )
        ablated_metric = compute_logit_diff(model, prompts, batch_size, hooks=hooks)
        effect = baseline_full - ablated_metric

        null_effects = []
        for null_i in range(n_null):
            if (null_i + 1) % 200 == 0:
                print(f"    Null sample {null_i + 1}/{n_null}")
            null_circuit = sample_type_matched_null(all_components_full, circuit_full, rng)

            null_mean_acts = None
            if abl_type == "mean":
                null_mean_acts = collect_mean_activations(model, prompts, null_circuit, batch_size)

            null_hooks = make_ablation_hooks(
                null_circuit, abl_type,
                mean_acts=null_mean_acts, rng=rng,
            )
            null_ablated = compute_logit_diff(model, prompts, batch_size, hooks=null_hooks)
            null_effects.append(baseline_full - null_ablated)

        null_arr = np.array(null_effects)
        null_mean = null_arr.mean()
        null_std = null_arr.std()
        z = (effect - null_mean) / null_std if null_std > 0 else 0.0
        empirical_p = (np.sum(null_arr >= effect) + 1) / (len(null_arr) + 1)

        full_results[abl_type] = {
            "baseline_metric": baseline_full,
            "ablated_metric": ablated_metric,
            "effect_size": effect,
            "z_score": float(z),
            "empirical_p": float(empirical_p),
            "null_mean": float(null_mean),
            "null_std": float(null_std),
        }
        print(f"    Effect: {effect:.4f}, z={z:.2f}, p={empirical_p:.4f}")

    # === Jaccard ===
    jac = jaccard_similarity(circuit_split, circuit_full)
    print(f"\n  Jaccard(split, full): {jac:.4f}")

    # Clean up
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    wall_time = time.time() - t0
    print(f"  Wall time: {wall_time:.1f}s")

    return CellResult(
        cell=cell.label,
        n_prompts=n_prompts,
        n_select=n_select,
        n_test=n_test,
        n_null=n_null,
        circuit_size=len(circuit_split),
        total_components=len(all_components_select),
        split_results=split_results,
        full_results=full_results,
        jaccard=jac,
        split_circuit_components=[component_label(c) for c in circuit_split],
        full_circuit_components=[component_label(c) for c in circuit_full],
        wall_time_s=wall_time,
    )


# ---------------------------------------------------------------------------
# Aggregate analysis
# ---------------------------------------------------------------------------

def aggregate_analysis(results: list[CellResult], ablation_types: list[str]) -> dict:
    """Compute aggregate statistics across cells."""
    analysis = {
        "n_cells": len(results),
        "per_ablation": {},
        "jaccard_stats": {},
    }

    jaccards = [r.jaccard for r in results]
    analysis["jaccard_stats"] = {
        "mean": float(np.mean(jaccards)),
        "std": float(np.std(jaccards)),
        "min": float(np.min(jaccards)),
        "max": float(np.max(jaccards)),
        "values": jaccards,
    }

    for abl in ablation_types:
        split_ps = []
        full_ps = []
        split_zs = []
        full_zs = []
        for r in results:
            if abl in r.split_results:
                split_ps.append(r.split_results[abl]["empirical_p"])
                split_zs.append(r.split_results[abl]["z_score"])
            if abl in r.full_results:
                full_ps.append(r.full_results[abl]["empirical_p"])
                full_zs.append(r.full_results[abl]["z_score"])

        # BH-FDR
        split_bh = bh_fdr(split_ps, q=0.05)
        full_bh = bh_fdr(full_ps, q=0.05)

        # Annotate per-cell results with BH-FDR
        for i, r in enumerate(results):
            if abl in r.split_results and i < len(split_bh):
                r.split_results[abl]["bh_fdr_pass"] = split_bh[i]
            if abl in r.full_results and i < len(full_bh):
                r.full_results[abl]["bh_fdr_pass"] = full_bh[i]

        analysis["per_ablation"][abl] = {
            "split_rejection_rate": sum(split_bh) / len(split_bh) if split_bh else 0,
            "full_rejection_rate": sum(full_bh) / len(full_bh) if full_bh else 0,
            "split_mean_z": float(np.mean(split_zs)) if split_zs else 0,
            "full_mean_z": float(np.mean(full_zs)) if full_zs else 0,
            "split_mean_p": float(np.mean(split_ps)) if split_ps else 0,
            "full_mean_p": float(np.mean(full_ps)) if full_ps else 0,
            "n_split_reject": sum(split_bh),
            "n_full_reject": sum(full_bh),
        }

    return analysis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 4: Expanded Held-Out-Split Audit"
    )
    parser.add_argument(
        "--cells",
        type=str,
        default="ioi/gpt2,ioi/qwen2.5",
        help="Comma-separated list of task/model cells (default: ioi/gpt2,ioi/qwen2.5)",
    )
    parser.add_argument(
        "--n-prompts", type=int, default=200,
        help="Total prompts per cell (split 50/50)",
    )
    parser.add_argument(
        "--n-null", type=int, default=1000,
        help="Number of null samples for matched-null test",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for model forward passes",
    )
    parser.add_argument(
        "--ablation", type=str, default="zero,mean,resampling",
        help="Comma-separated ablation types",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: stdout)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (default: auto-detect cuda/mps/cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Device selection
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    cells = [CellSpec.from_str(c) for c in args.cells.split(",")]
    ablation_types = [a.strip() for a in args.ablation.split(",")]

    print(f"Cells: {[c.label for c in cells]}")
    print(f"Ablation types: {ablation_types}")
    print(f"Prompts per cell: {args.n_prompts}")
    print(f"Null samples: {args.n_null}")

    # Run all cells
    all_results: list[CellResult] = []
    for cell in cells:
        result = run_single_cell(
            cell=cell,
            n_prompts=args.n_prompts,
            n_null=args.n_null,
            seed=args.seed,
            batch_size=args.batch_size,
            ablation_types=ablation_types,
            device=device,
        )
        all_results.append(result)

    # Aggregate analysis
    analysis = aggregate_analysis(all_results, ablation_types)

    # Build output
    output = {
        "experiment": "held_out_split_audit",
        "config": {
            "cells": [c.label for c in cells],
            "n_prompts": args.n_prompts,
            "n_null": args.n_null,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "ablation_types": ablation_types,
            "device": device,
        },
        "per_cell_results": [],
        "aggregate_analysis": analysis,
    }

    for r in all_results:
        output["per_cell_results"].append({
            "cell": r.cell,
            "n_prompts": r.n_prompts,
            "n_select": r.n_select,
            "n_test": r.n_test,
            "n_null": r.n_null,
            "circuit_size": r.circuit_size,
            "total_components": r.total_components,
            "jaccard_split_vs_full": r.jaccard,
            "split_circuit": r.split_circuit_components,
            "full_circuit": r.full_circuit_components,
            "split_results": r.split_results,
            "full_results": r.full_results,
            "wall_time_s": r.wall_time_s,
        })

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    header = f"{'Cell':<20} {'Ablation':<12} {'z_split':>8} {'p_split':>8} {'z_full':>8} {'p_full':>8} {'Jaccard':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        for abl in ablation_types:
            sr = r.split_results.get(abl, {})
            fr = r.full_results.get(abl, {})
            print(
                f"{r.cell:<20} {abl:<12} "
                f"{sr.get('z_score', 0):>8.2f} {sr.get('empirical_p', 1):>8.4f} "
                f"{fr.get('z_score', 0):>8.2f} {fr.get('empirical_p', 1):>8.4f} "
                f"{r.jaccard:>8.4f}"
            )

    print(f"\nAggregate Jaccard: {analysis['jaccard_stats']['mean']:.4f} +/- {analysis['jaccard_stats']['std']:.4f}")
    for abl in ablation_types:
        pa = analysis["per_ablation"].get(abl, {})
        print(
            f"  {abl}: split reject={pa.get('n_split_reject', 0)}/{len(all_results)}, "
            f"full reject={pa.get('n_full_reject', 0)}/{len(all_results)}, "
            f"split_mean_z={pa.get('split_mean_z', 0):.2f}, "
            f"full_mean_z={pa.get('full_mean_z', 0):.2f}"
        )

    # Write output
    json_str = json.dumps(output, indent=2)
    if args.output:
        Path(args.output).write_text(json_str)
        print(f"\nResults written to {args.output}")
    else:
        print(f"\n--- JSON OUTPUT ---")
        print(json_str)


if __name__ == "__main__":
    main()
