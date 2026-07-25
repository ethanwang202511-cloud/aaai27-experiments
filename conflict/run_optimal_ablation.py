"""run_optimal_ablation.py -- Experiment 5: Optimal Ablation Comparison.

Integrates Li & Janson (NeurIPS 2024, Spotlight) optimal ablation as a
fourth intervention type alongside mean, zero, and resampling ablation.

Optimal ablation (OA) learns a per-component replacement vector by gradient
descent, initialized at the dataset-mean activation and optimized to minimize
KL divergence between the ablated and original model output distributions.
This produces the "least destructive" ablation, revealing which components
are truly necessary vs. which appear important only because standard
ablation methods are over-destructive.

Methodology (adapted from Li & Janson, arXiv:2409.09951):
  Phase 1: Learn OA vectors for every component in the model.
    - Initialize ablation vector to dataset-mean activation of that component.
    - Optimize via AdamW (lr=2e-3, batch_size=3, up to 10000 steps) to
      minimize KL(original || ablated) on the task prompts.
    - Early stopping on a 20% held-out validation split.
  Phase 2: Run the matched-null audit under OA.
    - Ablate circuit using learned OA vectors.
    - For null samples: ablate random type-matched subsets using THEIR OA
      vectors (learned for every component in Phase 1).
    - Compare rejection rates across mean, zero, resampling, and OA.

Runs locally on Mac MPS or CUDA. NO Modal, NO SLURM.

Usage:
    python run_optimal_ablation.py --model gpt2 --n-null 500 --seed 42
    python run_optimal_ablation.py --model gemma2 --oa-steps 5000 --oa-lr 2e-3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from transformer_lens import HookedTransformer


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [optimal-ablation] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def pick_device(model_name: str) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if model_name in ("gemma2", "llama3"):
            log(f"WARNING: {model_name} may not fit on MPS. Proceeding anyway.")
        return "mps"
    return "cpu"


def empty_cache(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# IOI prompt generation
# ---------------------------------------------------------------------------
@dataclass
class IOITrial:
    prompt: str
    correct_token_id: int
    incorrect_token_id: int
    io_name: str
    s_name: str


NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace",
    "Henry", "Iris", "Jack", "Kate", "Leo", "Mary", "Nick", "Olivia",
    "Paul", "Quinn", "Rose", "Sam", "Tom", "Uma", "Vera", "Will",
]
PLACES = [
    "store", "park", "beach", "library", "school", "cafe",
    "museum", "garden", "market", "hospital", "church", "office",
]
TEMPLATES = [
    "When {io} and {s} went to the {place}, {io2} gave a gift to",
    "When {io} and {s} arrived at the {place}, {io2} handed a present to",
    "After {io} and {s} visited the {place}, {io2} gave a book to",
]


def generate_ioi_trials(
    tokenizer, n: int = 200, seed: int = 42
) -> List[IOITrial]:
    rng = random.Random(seed)
    trials = []
    for i in range(n):
        io_name, s_name = rng.sample(NAMES, 2)
        place = rng.choice(PLACES)
        template = rng.choice(TEMPLATES)
        prompt = template.format(io=io_name, s=s_name, place=place, io2=io_name)

        io_tok = tokenizer.encode(f" {io_name}", add_special_tokens=False)
        s_tok = tokenizer.encode(f" {s_name}", add_special_tokens=False)
        if not io_tok or not s_tok:
            continue
        trials.append(IOITrial(
            prompt=prompt,
            correct_token_id=io_tok[0],
            incorrect_token_id=s_tok[0],
            io_name=io_name,
            s_name=s_name,
        ))
    return trials[:n]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_MAP = {
    "gpt2": "gpt2-small",
    "gemma2": "gemma-2-2b",
    "llama3": "meta-llama/Meta-Llama-3.1-8B",
}


def load_model(model_short: str, device: str) -> HookedTransformer:
    tl_name = MODEL_MAP[model_short]
    log(f"loading {tl_name} ...")
    t0 = time.time()
    dtype = torch.float32 if device != "cuda" else torch.bfloat16
    model = HookedTransformer.from_pretrained(tl_name, device=device, dtype=dtype)
    model.eval()
    log(f"model ready in {time.time() - t0:.1f}s -- "
        f"n_layers={model.cfg.n_layers}, n_heads={model.cfg.n_heads}, "
        f"d_model={model.cfg.d_model}")
    return model


# ---------------------------------------------------------------------------
# Logit difference computation
# ---------------------------------------------------------------------------
def compute_logit_diff(logits: torch.Tensor, trials: List[IOITrial]) -> np.ndarray:
    diffs = np.zeros(len(trials))
    for j, t in enumerate(trials):
        diffs[j] = (
            logits[j, -1, t.correct_token_id] - logits[j, -1, t.incorrect_token_id]
        ).item()
    return diffs


def batched_logit_diff(
    model, tokens: torch.Tensor, trials: List[IOITrial],
    batch_size: int = 32, fwd_hooks=None,
) -> float:
    """Mean logit difference over all trials, optionally with hooks."""
    all_diffs = []
    N = tokens.shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ids = tokens[start:end]
        sub_trials = trials[start:end]
        with torch.no_grad():
            if fwd_hooks:
                logits = model.run_with_hooks(ids, fwd_hooks=fwd_hooks)
            else:
                logits = model(ids)
        all_diffs.append(compute_logit_diff(logits, sub_trials))
    return float(np.concatenate(all_diffs).mean())


# ---------------------------------------------------------------------------
# DLA-based circuit discovery
# ---------------------------------------------------------------------------
def dla_attribution(
    model, tokens: torch.Tensor, trials: List[IOITrial], batch_size: int = 32
) -> Dict[str, float]:
    """Compute Direct Logit Attribution for all heads and MLPs."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    device = model.cfg.device

    hooks_needed = []
    for L in range(n_layers):
        hooks_needed.append(f"blocks.{L}.attn.hook_result")
        hooks_needed.append(f"blocks.{L}.hook_mlp_out")

    scores = {}
    N = tokens.shape[0]

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ids = tokens[start:end]
        sub_trials = trials[start:end]

        with torch.no_grad():
            _, cache = model.run_with_cache(
                ids, names_filter=lambda k: k in hooks_needed
            )

        for j, t in enumerate(sub_trials):
            diff_dir = (
                model.W_U[:, t.correct_token_id] - model.W_U[:, t.incorrect_token_id]
            )
            for L in range(n_layers):
                # Heads
                attn_out = cache[f"blocks.{L}.attn.hook_result"][j, -1]  # (n_heads, d_model)
                for h in range(n_heads):
                    key = f"blocks.{L}.attn.head_{h}"
                    val = (attn_out[h] @ diff_dir).item()
                    scores[key] = scores.get(key, 0.0) + val / N

                # MLPs
                mlp_out = cache[f"blocks.{L}.hook_mlp_out"][j, -1]  # (d_model,)
                key = f"blocks.{L}.mlp"
                val = (mlp_out @ diff_dir).item()
                scores[key] = scores.get(key, 0.0) + val / N

        del cache
        empty_cache(device)

    return scores


def select_circuit(scores: Dict[str, float], top_frac: float = 0.10) -> List[str]:
    """Select top-X% components by absolute DLA score."""
    ranked = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
    k = max(1, int(len(ranked) * top_frac))
    return [c for c, _ in ranked[:k]]


# ---------------------------------------------------------------------------
# Mean activations (for mean ablation init and comparison)
# ---------------------------------------------------------------------------
def compute_mean_activations(
    model, tokens: torch.Tensor, batch_size: int = 32
) -> Dict[str, torch.Tensor]:
    """Compute per-component mean activations over the dataset."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    device = model.cfg.device
    N = tokens.shape[0]

    hooks = []
    for L in range(n_layers):
        hooks.append(f"blocks.{L}.attn.hook_result")
        hooks.append(f"blocks.{L}.hook_mlp_out")

    means = {}
    counts = 0

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ids = tokens[start:end]
        bs = ids.shape[0]

        with torch.no_grad():
            _, cache = model.run_with_cache(
                ids, names_filter=lambda k: k in hooks
            )

        for L in range(n_layers):
            # Head means -- last position only
            attn = cache[f"blocks.{L}.attn.hook_result"][:, -1]  # (bs, n_heads, d_model)
            for h in range(n_heads):
                key = f"blocks.{L}.attn.head_{h}"
                val = attn[:, h].sum(dim=0).detach().cpu()
                if key not in means:
                    means[key] = val
                else:
                    means[key] += val

            # MLP means -- last position only
            mlp = cache[f"blocks.{L}.hook_mlp_out"][:, -1]  # (bs, d_model)
            key = f"blocks.{L}.mlp"
            val = mlp.sum(dim=0).detach().cpu()
            if key not in means:
                means[key] = val
            else:
                means[key] += val

        counts += bs
        del cache
        empty_cache(device)

    for key in means:
        means[key] = means[key] / counts

    return means


# ---------------------------------------------------------------------------
# Optimal Ablation: learn per-component replacement vectors
# ---------------------------------------------------------------------------
def learn_optimal_ablation_vectors(
    model,
    tokens: torch.Tensor,
    trials: List[IOITrial],
    mean_acts: Dict[str, torch.Tensor],
    components: List[str],
    lr: float = 2e-3,
    batch_size: int = 3,
    max_steps: int = 10000,
    early_stop_patience: int = 500,
    val_frac: float = 0.20,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Learn optimal ablation vectors via KL minimization.

    For each component, learn a constant replacement vector that minimizes
    KL(original || ablated) when that component's output is replaced.

    Following Li & Janson (2024):
    - Initialize to dataset-mean activation
    - Optimize with AdamW (lr=2e-3)
    - KL divergence loss
    - Early stopping on validation split

    Returns dict mapping component name -> learned ablation vector (cpu).
    """
    device = model.cfg.device
    N = tokens.shape[0]

    # Train/val split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)
    n_val = max(1, int(N * val_frac))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_tokens = tokens[train_idx]
    val_tokens = tokens[val_idx]

    oa_vectors = {}
    total = len(components)

    for ci, comp in enumerate(components):
        log(f"  learning OA vector for {comp} ({ci + 1}/{total}) ...")
        t0 = time.time()

        # Parse component
        is_head = ".attn.head_" in comp
        if is_head:
            parts = comp.split(".")
            layer = int(parts[1])
            head = int(parts[3].split("_")[1])
            hook_name = f"blocks.{layer}.attn.hook_result"
        else:
            parts = comp.split(".")
            layer = int(parts[1])
            head = None
            hook_name = f"blocks.{layer}.hook_mlp_out"

        # Initialize ablation vector to mean activation
        init_vec = mean_acts[comp].to(device).to(torch.float32)
        oa_vec = init_vec.clone().requires_grad_(True)

        optimizer = torch.optim.AdamW([oa_vec], lr=lr, weight_decay=0.0)

        # Precompute reference logits for validation
        val_ref_logits = []
        for start in range(0, len(val_tokens), 16):
            end = min(start + 16, len(val_tokens))
            with torch.no_grad():
                logits = model(val_tokens[start:end])
            val_ref_logits.append(logits[:, -1].detach())
        val_ref_logits = torch.cat(val_ref_logits, dim=0)

        best_val_loss = float("inf")
        patience_counter = 0
        best_vec = oa_vec.data.clone()

        n_train = len(train_tokens)

        for step in range(max_steps):
            # Sample mini-batch
            batch_idx = rng.choice(n_train, size=min(batch_size, n_train), replace=False)
            batch_tokens = train_tokens[batch_idx]

            # Reference logits (no ablation)
            with torch.no_grad():
                ref_logits = model(batch_tokens)[:, -1]  # (bs, vocab)
                ref_probs = F.softmax(ref_logits.float(), dim=-1)

            # Ablated forward pass with the learnable vector
            def ablation_hook(act, hook):
                a = act.clone()
                if is_head:
                    a[:, -1, head, :] = oa_vec.to(a.dtype)
                else:
                    a[:, -1, :] = oa_vec.to(a.dtype)
                return a

            abl_logits = model.run_with_hooks(
                batch_tokens, fwd_hooks=[(hook_name, ablation_hook)]
            )[:, -1]

            abl_log_probs = F.log_softmax(abl_logits.float(), dim=-1)

            # KL(ref || ablated) = sum ref_p * (log ref_p - log abl_p)
            loss = F.kl_div(abl_log_probs, ref_probs, reduction="batchmean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation check every 100 steps
            if (step + 1) % 100 == 0:
                with torch.no_grad():
                    val_abl_logits = []
                    for vs in range(0, len(val_tokens), 16):
                        ve = min(vs + 16, len(val_tokens))
                        vl = model.run_with_hooks(
                            val_tokens[vs:ve],
                            fwd_hooks=[(hook_name, ablation_hook)]
                        )[:, -1]
                        val_abl_logits.append(vl)
                    val_abl_logits = torch.cat(val_abl_logits, dim=0)
                    val_log_probs = F.log_softmax(val_abl_logits.float(), dim=-1)
                    val_ref_probs = F.softmax(val_ref_logits.float(), dim=-1)
                    val_loss = F.kl_div(
                        val_log_probs, val_ref_probs, reduction="batchmean"
                    ).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_vec = oa_vec.data.clone()
                    patience_counter = 0
                else:
                    patience_counter += 100
                    if patience_counter >= early_stop_patience:
                        log(f"    early stop at step {step + 1}, "
                            f"val_loss={best_val_loss:.6f}")
                        break

        oa_vectors[comp] = best_vec.detach().cpu()
        elapsed = time.time() - t0
        log(f"    {comp} done in {elapsed:.1f}s, "
            f"val_kl={best_val_loss:.6f}, steps={step + 1}")

        # Detach to free graph
        oa_vec = oa_vec.detach()
        del optimizer
        empty_cache(device)

    return oa_vectors


# ---------------------------------------------------------------------------
# Ablation audit (generic for any ablation type)
# ---------------------------------------------------------------------------
def run_ablation_audit(
    model,
    tokens: torch.Tensor,
    trials: List[IOITrial],
    circuit: List[str],
    ablation_type: str,
    mean_acts: Dict[str, torch.Tensor],
    oa_vectors: Optional[Dict[str, torch.Tensor]],
    n_null_samples: int,
    seed: int,
    batch_size: int = 32,
) -> dict:
    """Run matched-null audit under a specified ablation type.

    ablation_type: "mean", "zero", "resampling", "optimal"
    """
    device = model.cfg.device
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    rng = np.random.RandomState(seed)

    def _is_head(c): return ".attn.head_" in c
    def _parse(c):
        parts = c.split(".")
        layer = int(parts[1])
        if _is_head(c):
            head = int(parts[3].split("_")[1])
            return layer, head, f"blocks.{layer}.attn.hook_result"
        else:
            return layer, None, f"blocks.{layer}.hook_mlp_out"

    # Build component universe
    all_components = []
    for L in range(n_layers):
        for h in range(n_heads):
            all_components.append(f"blocks.{L}.attn.head_{h}")
        all_components.append(f"blocks.{L}.mlp")

    circuit_set = set(circuit)
    non_circuit_heads = [c for c in all_components if _is_head(c) and c not in circuit_set]
    non_circuit_mlps = [c for c in all_components if not _is_head(c) and c not in circuit_set]
    n_circuit_heads = sum(1 for c in circuit if _is_head(c))
    n_circuit_mlps = sum(1 for c in circuit if not _is_head(c))

    def make_ablation_hooks(comp_set: List[str]):
        hooks_by_name = {}
        for c in comp_set:
            layer, head, hook_name = _parse(c)
            if hook_name not in hooks_by_name:
                hooks_by_name[hook_name] = []
            hooks_by_name[hook_name].append((c, layer, head))

        def make_hook_fn(entries):
            def fn(act, hook):
                a = act.clone()
                for comp_name, layer, head in entries:
                    if ablation_type == "zero":
                        if head is not None:
                            a[:, -1, head, :] = 0.0
                        else:
                            a[:, -1, :] = 0.0
                    elif ablation_type == "mean":
                        vec = mean_acts[comp_name].to(a.device).to(a.dtype)
                        if head is not None:
                            a[:, -1, head, :] = vec
                        else:
                            a[:, -1, :] = vec
                    elif ablation_type == "optimal":
                        vec = oa_vectors[comp_name].to(a.device).to(a.dtype)
                        if head is not None:
                            a[:, -1, head, :] = vec
                        else:
                            a[:, -1, :] = vec
                    elif ablation_type == "resampling":
                        B = a.shape[0]
                        perm = torch.randperm(B, device=a.device)
                        if head is not None:
                            a[:, -1, head, :] = a[perm, -1, head, :]
                        else:
                            a[:, -1, :] = a[perm, -1, :]
                return a
            return fn

        fwd_hooks = []
        for hook_name, entries in hooks_by_name.items():
            fwd_hooks.append((hook_name, make_hook_fn(entries)))
        return fwd_hooks

    # Baseline
    baseline = batched_logit_diff(model, tokens, trials, batch_size)
    log(f"  [{ablation_type}] baseline logit_diff = {baseline:.4f}")

    # Circuit drop
    circuit_hooks = make_ablation_hooks(circuit)
    circuit_score = batched_logit_diff(model, tokens, trials, batch_size, circuit_hooks)
    circuit_drop = baseline - circuit_score

    # Null samples
    null_drops = []
    for i in range(n_null_samples):
        # Type-matched sampling
        null_heads = list(rng.choice(
            non_circuit_heads, size=min(n_circuit_heads, len(non_circuit_heads)), replace=False
        )) if n_circuit_heads > 0 else []
        null_mlps = list(rng.choice(
            non_circuit_mlps, size=min(n_circuit_mlps, len(non_circuit_mlps)), replace=False
        )) if n_circuit_mlps > 0 else []
        null_set = null_heads + null_mlps

        null_hooks = make_ablation_hooks(null_set)
        null_score = batched_logit_diff(model, tokens, trials, batch_size, null_hooks)
        null_drops.append(baseline - null_score)

        if (i + 1) % 50 == 0:
            log(f"    [{ablation_type}] null {i + 1}/{n_null_samples}")
            empty_cache(device)

    null_drops = np.array(null_drops)
    null_mean = float(null_drops.mean())
    null_std = float(null_drops.std())
    z_score = (circuit_drop - null_mean) / null_std if null_std > 1e-12 else float("nan")
    empirical_p = float(np.mean(null_drops >= circuit_drop))

    return {
        "ablation_type": ablation_type,
        "baseline": baseline,
        "circuit_drop": float(circuit_drop),
        "null_drops": null_drops.tolist(),
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": z_score,
        "empirical_p": empirical_p,
        "rejects_at_005": bool(empirical_p < 0.05),
        "n_null_samples": n_null_samples,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 5: Optimal Ablation Comparison"
    )
    parser.add_argument(
        "--model", default="gpt2", choices=["gpt2", "gemma2", "llama3"],
        help="Model to run on (default: gpt2)."
    )
    parser.add_argument("--n-prompts", type=int, default=200)
    parser.add_argument("--n-null", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--oa-lr", type=float, default=2e-3,
                        help="Learning rate for OA vector optimization.")
    parser.add_argument("--oa-steps", type=int, default=10000,
                        help="Max gradient steps for OA optimization.")
    parser.add_argument("--oa-batch-size", type=int, default=3,
                        help="Batch size for OA optimization.")
    parser.add_argument("--oa-patience", type=int, default=500,
                        help="Early stopping patience (in steps).")
    parser.add_argument("--top-frac", type=float, default=0.10,
                        help="Fraction of components in the circuit (default 10%%).")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--skip-ablation-types", nargs="*", default=[],
        help="Ablation types to skip (e.g., 'resampling' to save time)."
    )
    parser.add_argument("--out-dir", default="results/exp5_optimal_ablation")
    args = parser.parse_args()

    device = args.device or pick_device(args.model)
    log(f"device={device}, model={args.model}")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load model
    model = load_model(args.model, device)

    # Generate IOI trials
    log(f"generating {args.n_prompts} IOI trials ...")
    trials = generate_ioi_trials(model.tokenizer, n=args.n_prompts, seed=args.seed)
    prompts = [t.prompt for t in trials]
    tokens = model.tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to(device)
    log(f"{len(trials)} trials tokenized, shape={tokens.shape}")

    # DLA circuit discovery
    log("running DLA attribution ...")
    t0 = time.time()
    dla_scores = dla_attribution(model, tokens, trials, batch_size=args.batch_size)
    circuit = select_circuit(dla_scores, top_frac=args.top_frac)
    log(f"circuit: {len(circuit)} components selected in {time.time() - t0:.1f}s")

    # Compute mean activations
    log("computing mean activations for all components ...")
    t0 = time.time()
    mean_acts = compute_mean_activations(model, tokens, batch_size=args.batch_size)
    log(f"mean activations computed in {time.time() - t0:.1f}s")

    # Phase 1: Learn OA vectors for ALL components (not just circuit)
    # This is the expensive step but allows principled null construction
    all_components = list(mean_acts.keys())
    log(f"Phase 1: learning OA vectors for all {len(all_components)} components ...")
    t0 = time.time()
    oa_vectors = learn_optimal_ablation_vectors(
        model=model,
        tokens=tokens,
        trials=trials,
        mean_acts=mean_acts,
        components=all_components,
        lr=args.oa_lr,
        batch_size=args.oa_batch_size,
        max_steps=args.oa_steps,
        early_stop_patience=args.oa_patience,
        seed=args.seed,
    )
    phase1_time = time.time() - t0
    log(f"Phase 1 complete in {phase1_time:.1f}s ({phase1_time / 3600:.2f}h)")

    # Phase 2: Run audit under all ablation types
    ablation_types = ["mean", "zero", "resampling", "optimal"]
    ablation_types = [a for a in ablation_types if a not in args.skip_ablation_types]

    results = {}
    for abl in ablation_types:
        log(f"Phase 2: running {abl} ablation audit (n={args.n_null}) ...")
        t0 = time.time()
        results[abl] = run_ablation_audit(
            model=model,
            tokens=tokens,
            trials=trials,
            circuit=circuit,
            ablation_type=abl,
            mean_acts=mean_acts,
            oa_vectors=oa_vectors if abl == "optimal" else None,
            n_null_samples=args.n_null,
            seed=args.seed + hash(abl) % 10000,
            batch_size=args.batch_size,
        )
        elapsed = time.time() - t0
        r = results[abl]
        log(f"  {abl}: circuit_drop={r['circuit_drop']:.4f}, "
            f"z={r['z_score']:.2f}, p={r['empirical_p']:.4f}, "
            f"rejects={r['rejects_at_005']} ({elapsed:.1f}s)")

    # Summary comparison
    log("\n=== COMPARISON ===")
    for abl in ablation_types:
        r = results[abl]
        log(f"  {abl:12s}: drop={r['circuit_drop']:+.4f}  "
            f"z={r['z_score']:6.2f}  p={r['empirical_p']:.4f}  "
            f"reject={r['rejects_at_005']}")

    # Expected hierarchy: OA (~15%) < mean (33%) < resampling (43%) < zero (83%)
    rejection_rates = {abl: r["rejects_at_005"] for abl, r in results.items()}
    z_scores = {abl: r["z_score"] for abl, r in results.items()}

    # Save
    out = {
        "experiment": "optimal_ablation_comparison",
        "model": args.model,
        "task": "ioi",
        "n_prompts": args.n_prompts,
        "n_null_samples": args.n_null,
        "seed": args.seed,
        "circuit_size": len(circuit),
        "circuit_components": circuit,
        "top_frac": args.top_frac,
        "oa_hyperparams": {
            "lr": args.oa_lr,
            "batch_size": args.oa_batch_size,
            "max_steps": args.oa_steps,
            "patience": args.oa_patience,
            "val_frac": 0.20,
        },
        "phase1_wall_seconds": phase1_time,
        "results_by_ablation": {
            abl: {k: v for k, v in r.items() if k != "null_drops"}
            for abl, r in results.items()
        },
        "null_distributions": {
            abl: r["null_drops"] for abl, r in results.items()
        },
        "comparison": {
            "rejection_rates": rejection_rates,
            "z_scores": z_scores,
            "hierarchy_note": (
                "Expected hierarchy (Li & Janson 2024): "
                "OA produces smallest drops (11-33% of standard ablation), "
                "so rejection rate should be: OA < mean < resampling < zero."
            ),
        },
        "reference": (
            "Li, M. & Janson, L. (2024). Optimal Ablation for Interpretability. "
            "NeurIPS 2024 (Spotlight). arXiv:2409.09951. "
            "Code: github.com/maxtli/optimalablation."
        ),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    outpath = os.path.join(args.out_dir, f"{args.model}_ioi_optimal_ablation.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2, default=float)
    log(f"saved results to {outpath}")


if __name__ == "__main__":
    main()
