"""Experiment 1 (C3): Full SAE feature-circuit selectivity protocol.

AAAI-27 matched-null audit of mechanistic interpretability circuit claims.
Self-contained script — runs locally on Mac MPS or CUDA (no Modal, no SLURM).

Substrate: GPT-2 small, IOI task, 200 prompts (first 80 for eval).

SAE loading: Residual-stream SAEs from "jbloom/GPT2-Small-SAEs-Reformatted"
for layers 8 and 9.  Dictionary size 24,576 features per layer.  Loaded via
huggingface_hub + safetensors (no sae_lens dependency).

Feature selection: Top 15 features by integrated-gradients (IG) attribution
across both SAEs.  20-step IG with zero-activation baseline.

Audit protocol (SAE features):
  - Intervention 1: Feature-level zero ablation — set top-15 feature
    activations to zero in SAE encoding, reconstruct, patch residual stream.
  - Intervention 2: Causal feature patching — patch clean feature activations
    into corrupted (IO-swapped) forward pass, Marks et al. 2025 style.
  - Null constructions:
      (a) density-decile-matched — 15 features from same activation-density
          decile.
      (b) activation-magnitude-matched — match on mean activation magnitude.
  - n = 500 null samples per construction.
  - Metric: logit difference (IO - S1) at last token position.

Audit protocol (raw residual dimensions, paired baseline):
  - Top 15 residual-stream dimensions (of 768) by IG attribution at same
    layers.
  - Same zero-ablation protocol on raw dimensions.
  - Null: random 15-dimension subsets, matched on activation-magnitude decile.
  - n = 500 null samples.

Output: JSON with full null distributions (all 500 drops), z-scores
  z(C) = (circuit_drop - null_mean) / null_std for each condition.

Usage:
    python3 run_sae_selectivity.py \\
        [--n-null-samples 500] [--n-features 15] [--seed 42] [--ig-steps 20]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from transformer_lens import HookedTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAE_LAYERS = (8, 9)
SAE_REPO = "jbloom/GPT2-Small-SAEs-Reformatted"

# IOI prompt generation data
NAMES_IO = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul",
    "Quinn", "Rose", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
]
NAMES_S = [
    "James", "Mary", "John", "Sarah", "David", "Emma", "Michael", "Lisa",
    "Chris", "Anna", "Daniel", "Laura", "Ryan", "Sophie", "Mark", "Julia",
    "Luke", "Hailey", "Tom", "Nora", "Ben", "Claire", "Ian", "Fiona",
]
PLACES = [
    "store", "park", "school", "library", "hospital", "restaurant",
    "museum", "market", "office", "gym", "cafe", "beach",
    "zoo", "airport", "station", "theater", "garden", "church",
]
IOI_TEMPLATE = "When {io} and {s} went to the {place}, {io} gave a gift to"


# ---------------------------------------------------------------------------
# IOI trial dataclass (self-contained, no src imports)
# ---------------------------------------------------------------------------
@dataclass
class IOITrial:
    """Minimal IOI trial: holds prompt text and token IDs for the correct
    (indirect object = IO) and incorrect (subject = S) completions."""
    prompt: str
    io_name: str
    s_name: str
    correct_token_id: int   # tokenizer ID for " IO"
    incorrect_token_id: int  # tokenizer ID for " S"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [sae-selectivity] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mps_empty_cache(device: str) -> None:
    if device == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif device == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# IOI prompt generation
# ---------------------------------------------------------------------------
def generate_ioi_trials(
    tokenizer,
    n_trials: int = 200,
    seed: int = 42,
) -> List[IOITrial]:
    """Generate IOI trials using the standard template.

    Template: "When {IO} and {S} went to the {place}, {IO} gave a gift to"
    Correct completion: " {S}" (the subject receives the gift)
    Wait — in IOI the correct answer is the Indirect Object name.
    Template: "When {name1} and {name2} went to the {place}, {name3} gave
    a gift to" where name1=name3=IO (the one who gives), and the correct
    completion is " {name2}" = S?

    Actually, standard IOI:
      "When Mary and John went to the store, John gave a drink to"
      Answer: Mary (the indirect object).
      name1 = Mary (IO), name2 = John (S), name3 = John (S repeats).
      Correct = IO = Mary.

    So: template is "When {IO} and {S} went to the {place}, {S} gave a
    gift to" and the correct answer is IO.
    """
    rng = np.random.RandomState(seed)
    trials = []
    for _ in range(n_trials):
        io_idx = rng.randint(len(NAMES_IO))
        s_idx = rng.randint(len(NAMES_S))
        place_idx = rng.randint(len(PLACES))
        io_name = NAMES_IO[io_idx]
        s_name = NAMES_S[s_idx]
        place = PLACES[place_idx]
        # Standard IOI: S appears twice (once in intro, once as subject of
        # the giving clause), IO appears once at the start.
        # "When {IO} and {S} went to the {place}, {S} gave a gift to"
        # Correct completion: " {IO}"
        prompt = f"When {io_name} and {s_name} went to the {place}, {s_name} gave a gift to"

        # Token IDs: GPT-2 tokenizer prepends space to names
        io_tok = tokenizer.encode(f" {io_name}", add_special_tokens=False)
        s_tok = tokenizer.encode(f" {s_name}", add_special_tokens=False)
        if len(io_tok) != 1 or len(s_tok) != 1:
            # Skip multi-token names (rare with short names but be safe)
            continue

        trials.append(IOITrial(
            prompt=prompt,
            io_name=io_name,
            s_name=s_name,
            correct_token_id=io_tok[0],
            incorrect_token_id=s_tok[0],
        ))
    if len(trials) < n_trials:
        log(f"WARNING: generated only {len(trials)}/{n_trials} single-token IOI trials")
    return trials[:n_trials]


def logit_diff_batch(
    logits: torch.Tensor,
    trials: List[IOITrial],
) -> np.ndarray:
    """Compute per-trial logit difference: logit(IO) - logit(S) at last
    token position."""
    diffs = np.zeros(len(trials))
    for j, t in enumerate(trials):
        diffs[j] = float(
            logits[j, -1, t.correct_token_id]
            - logits[j, -1, t.incorrect_token_id]
        )
    return diffs


# ---------------------------------------------------------------------------
# SAE loading
# ---------------------------------------------------------------------------
def load_saes(device: str, hf_token: Optional[str] = None) -> Dict:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    saes: Dict = {}
    for layer in SAE_LAYERS:
        log(f"downloading SAE weights for blocks.{layer}.hook_resid_pre ...")
        sp = hf_hub_download(
            repo_id=SAE_REPO,
            filename=f"blocks.{layer}.hook_resid_pre/sae_weights.safetensors",
            token=hf_token,
        )
        w = load_file(sp)
        d_sae = w["W_enc"].shape[1]
        saes[layer] = {
            "W_enc": w["W_enc"].to(device),          # (d_model, d_sae)
            "W_dec": w["W_dec"].to(device),          # (d_sae, d_model)
            "b_enc": w.get("b_enc", torch.zeros(d_sae)).to(device),
            "b_dec": w.get("b_dec", torch.zeros(w["W_enc"].shape[0])).to(device),
        }
        log(f"  layer {layer}: d_sae={d_sae}")
    return saes


def sae_encode(resid: torch.Tensor, layer: int, saes: Dict) -> torch.Tensor:
    """SAE encoding: relu((resid - b_dec) @ W_enc + b_enc)."""
    z = (resid - saes[layer]["b_dec"]) @ saes[layer]["W_enc"] + saes[layer]["b_enc"]
    return torch.relu(z)


def sae_decode(feat_acts: torch.Tensor, layer: int, saes: Dict) -> torch.Tensor:
    """SAE decoding: feat_acts @ W_dec + b_dec."""
    return feat_acts @ saes[layer]["W_dec"] + saes[layer]["b_dec"]


# ---------------------------------------------------------------------------
# Decile utilities
# ---------------------------------------------------------------------------
def compute_deciles(arr: np.ndarray) -> np.ndarray:
    """Assign each element to a decile (0-9)."""
    q = np.quantile(arr, np.linspace(0, 1, 11))
    return np.clip(np.digitize(arr, q) - 1, 0, 9)


def build_bucket_sampler(
    bins: Dict[int, np.ndarray],
    target_counts: Dict[Tuple[int, int], int],
    circuit_set: set,
    saes: Dict,
    rng: np.random.RandomState,
):
    """Build a sampler that draws features matched on the given decile bins."""
    pools: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for (L, b) in target_counts:
        pools[(L, b)] = [
            (L, f)
            for f in range(saes[L]["W_enc"].shape[1])
            if int(bins[L][f]) == b and (L, f) not in circuit_set
        ]

    def sample() -> List[Tuple[int, int]]:
        out = []
        for (L, b), k in target_counts.items():
            pool = pools[(L, b)]
            if len(pool) >= k:
                idx = rng.choice(len(pool), size=k, replace=False)
            elif pool:
                idx = rng.choice(len(pool), size=k, replace=True)
            else:
                continue
            for i in idx:
                out.append(pool[i])
        return out

    return sample


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-null-samples", type=int, default=500,
                        help="Number of null samples per construction (default: 500)")
    parser.add_argument("--n-features", type=int, default=15,
                        help="Number of top features to select (default: 15)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ig-steps", type=int, default=20,
                        help="Number of IG interpolation steps (default: 20)")
    parser.add_argument("--device", default=None,
                        help="Override device; defaults to CUDA > MPS > CPU.")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: ./sae_selectivity_results.json)")
    args = parser.parse_args()

    n_null = args.n_null_samples
    n_features = args.n_features
    seed = args.seed
    ig_steps = args.ig_steps
    device = args.device or pick_device()
    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "sae_selectivity_results.json",
    )

    log(f"device={device}, n_null={n_null}, n_features={n_features}, "
        f"ig_steps={ig_steps}, seed={seed}")
    log(f"output -> {out_path}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token)
        except Exception as e:
            log(f"HF login failed (continuing anonymously): {e}")

    # ===================================================================
    # Step 1: Load SAEs
    # ===================================================================
    t0 = time.time()
    try:
        saes = load_saes(device, hf_token)
    except Exception as e:
        log(f"SAE load FAILED: {e}")
        traceback.print_exc()
        result = {"status": "sae_load_failed", "reason": str(e)}
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        return
    d_sae = saes[SAE_LAYERS[0]]["W_enc"].shape[1]
    log(f"SAEs loaded in {time.time() - t0:.1f}s  (d_sae={d_sae})")

    # ===================================================================
    # Step 2: Load GPT-2 small
    # ===================================================================
    log("loading GPT-2 small via TransformerLens ...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    d_model = model.cfg.d_model
    log(f"model ready in {time.time() - t0:.1f}s  "
        f"(n_layers={model.cfg.n_layers}, d_model={d_model})")

    # ===================================================================
    # Step 3: Generate 200 IOI prompts
    # ===================================================================
    log("generating 200 IOI prompts (first 80 for eval) ...")
    all_trials = generate_ioi_trials(model.tokenizer, n_trials=200, seed=seed)
    eval_trials = all_trials[:80]
    prompts = [t.prompt for t in eval_trials]
    tokens = model.tokenizer(
        prompts, return_tensors="pt", padding=True,
    ).input_ids.to(device)
    B = tokens.shape[0]
    log(f"{B} eval trials tokenized  (seq_len={tokens.shape[1]})")

    # ===================================================================
    # Step 4: Cache SAE feature activations at last position
    # ===================================================================
    log("caching SAE feature activations (last position) for all eval trials ...")
    hooks_needed = [f"blocks.{L}.hook_resid_pre" for L in SAE_LAYERS]
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda k: k in hooks_needed
        )

    # feat_last[L] : (B, d_sae) — clean feature activations at last position
    feat_last: Dict[int, torch.Tensor] = {}
    # resid_last[L] : (B, d_model) — raw residual activations at last position
    resid_last: Dict[int, torch.Tensor] = {}
    density: Dict[int, np.ndarray] = {}     # firing rate per feature
    magnitude: Dict[int, np.ndarray] = {}   # mean activation per feature

    for L in SAE_LAYERS:
        resid = cache[f"blocks.{L}.hook_resid_pre"]
        feats = sae_encode(resid, L, saes)
        feat_last[L] = feats[:, -1, :].detach().clone()
        resid_last[L] = resid[:, -1, :].detach().clone()
        # Density: fraction of (batch, seq) positions where feature fires
        density[L] = (feats > 1e-6).float().mean(dim=(0, 1)).cpu().numpy()
        # Magnitude: mean activation across (batch, seq) for each feature
        magnitude[L] = feats.mean(dim=(0, 1)).cpu().numpy()
    del cache
    mps_empty_cache(device)

    # ===================================================================
    # Step 5: Integrated-gradients attribution (20-step, zero baseline)
    # ===================================================================
    def run_ig_sae(
        ig_model: HookedTransformer,
        ig_saes: Dict,
        ig_tokens: torch.Tensor,
        ig_eval_trials: List[IOITrial],
        ig_feat_last: Dict[int, torch.Tensor],
        compute_device: str,
        n_alpha_steps: int,
    ) -> Dict[int, np.ndarray]:
        """20-step IG attribution through SAE features.

        Baseline: zero activation vector.
        Path: alpha * feat_last[L], alpha in linspace(0, 1, n_alpha_steps)
            but we skip alpha=0 (zero gradient there is uninformative).
        IG formula: sum over alpha of (grad w.r.t. scaled_feat) * feat_last / n_steps.
        """
        ig_scores = {
            L: np.zeros(ig_saes[L]["W_enc"].shape[1]) for L in SAE_LAYERS
        }
        alphas = np.linspace(1.0 / n_alpha_steps, 1.0, n_alpha_steps)

        for step, alpha in enumerate(alphas):
            # Scaled feature activations as leaf tensors for gradient
            leaf = {
                L: (ig_feat_last[L].to(compute_device) * alpha).clone().requires_grad_(True)
                for L in SAE_LAYERS
            }

            def make_hook_fn(L_fixed):
                def fn(activation, hook):
                    resid_in = activation.clone()
                    recon = sae_decode(leaf[L_fixed], L_fixed, ig_saes)
                    resid_in[:, -1, :] = recon
                    return resid_in
                return fn

            fwd_hooks = [
                (f"blocks.{L}.hook_resid_pre", make_hook_fn(L))
                for L in SAE_LAYERS
            ]
            logits = ig_model.run_with_hooks(ig_tokens, fwd_hooks=fwd_hooks)
            diffs = torch.zeros(len(ig_eval_trials), device=compute_device)
            for j, t in enumerate(ig_eval_trials):
                diffs[j] = (
                    logits[j, -1, t.correct_token_id]
                    - logits[j, -1, t.incorrect_token_id]
                )
            diffs.sum().backward()

            for L in SAE_LAYERS:
                g = leaf[L].grad
                if g is not None:
                    # IG: gradient * (endpoint - baseline) / n_steps
                    # baseline = 0, so endpoint - baseline = feat_last
                    contrib = (g * ig_feat_last[L].to(compute_device)).sum(dim=0)
                    ig_scores[L] += contrib.detach().abs().cpu().numpy()

            mps_empty_cache(compute_device)
            if (step + 1) % 5 == 0 or step == 0:
                log(f"  IG step {step + 1}/{n_alpha_steps} (alpha={alpha:.3f}) done")

        for L in SAE_LAYERS:
            ig_scores[L] /= float(n_alpha_steps)
        return ig_scores

    log(f"running {ig_steps}-step integrated-gradients SAE attribution on {device} ...")
    t0 = time.time()
    try:
        ig_scores = run_ig_sae(
            model, saes, tokens, eval_trials, feat_last, device, ig_steps,
        )
    except Exception as e:
        if device != "cpu":
            log(f"IG backward FAILED on {device} ({e!r}); falling back to CPU")
            traceback.print_exc()
            model_cpu = model.to("cpu")
            saes_cpu = {
                L: {k: v.to("cpu") for k, v in saes[L].items()} for L in SAE_LAYERS
            }
            tokens_cpu = tokens.to("cpu")
            feat_last_cpu = {L: v.to("cpu") for L, v in feat_last.items()}
            ig_scores = run_ig_sae(
                model_cpu, saes_cpu, tokens_cpu, eval_trials,
                feat_last_cpu, "cpu", ig_steps,
            )
            model = model_cpu.to(device)
            saes = {L: {k: v.to(device) for k, v in saes[L].items()} for L in SAE_LAYERS}
            tokens = tokens_cpu.to(device)
            mps_empty_cache(device)
        else:
            raise
    log(f"IG SAE attribution done in {time.time() - t0:.1f}s")

    # Report per-layer IG magnitudes
    for L in SAE_LAYERS:
        total = float(ig_scores[L].sum())
        top5 = float(np.sort(ig_scores[L])[-5:].sum())
        log(f"  layer {L}: total_ig={total:.2f}, top-5 sum={top5:.2f}")

    # Select top-N features across both SAEs
    all_scored = [
        (L, f, float(ig_scores[L][f]))
        for L in SAE_LAYERS
        for f in range(saes[L]["W_enc"].shape[1])
    ]
    all_scored.sort(key=lambda x: x[2], reverse=True)
    circuit = [(L, f) for L, f, _ in all_scored[:n_features]]
    circuit_set = set(circuit)

    # Check if all features land on one layer
    layer_counts = {L: sum(1 for (lL, _) in circuit if lL == L) for L in SAE_LAYERS}
    log(f"circuit (top-{n_features} SAE features): {circuit}")
    log(f"  per-layer counts: {layer_counts}")
    if any(c == n_features for c in layer_counts.values()):
        log("  NOTE: all circuit features on a single layer — per-layer IG "
            "magnitudes reported above")

    # ===================================================================
    # Step 5b: IG attribution for raw residual dimensions
    # ===================================================================
    def run_ig_resid(
        ig_model: HookedTransformer,
        ig_tokens: torch.Tensor,
        ig_eval_trials: List[IOITrial],
        ig_resid_last: Dict[int, torch.Tensor],
        compute_device: str,
        n_alpha_steps: int,
    ) -> Dict[int, np.ndarray]:
        """IG attribution on raw residual-stream dimensions."""
        ig_dim_scores = {L: np.zeros(ig_model.cfg.d_model) for L in SAE_LAYERS}
        alphas = np.linspace(1.0 / n_alpha_steps, 1.0, n_alpha_steps)

        for step, alpha in enumerate(alphas):
            leaf = {
                L: (ig_resid_last[L].to(compute_device) * alpha).clone().requires_grad_(True)
                for L in SAE_LAYERS
            }

            def make_hook_fn(L_fixed):
                def fn(activation, hook):
                    r = activation.clone()
                    r[:, -1, :] = leaf[L_fixed]
                    return r
                return fn

            fwd_hooks = [
                (f"blocks.{L}.hook_resid_pre", make_hook_fn(L))
                for L in SAE_LAYERS
            ]
            logits = ig_model.run_with_hooks(ig_tokens, fwd_hooks=fwd_hooks)
            diffs = torch.zeros(len(ig_eval_trials), device=compute_device)
            for j, t in enumerate(ig_eval_trials):
                diffs[j] = (
                    logits[j, -1, t.correct_token_id]
                    - logits[j, -1, t.incorrect_token_id]
                )
            diffs.sum().backward()

            for L in SAE_LAYERS:
                g = leaf[L].grad
                if g is not None:
                    contrib = (g * ig_resid_last[L].to(compute_device)).sum(dim=0)
                    ig_dim_scores[L] += contrib.detach().abs().cpu().numpy()

            mps_empty_cache(compute_device)
            if (step + 1) % 5 == 0 or step == 0:
                log(f"  resid-IG step {step + 1}/{n_alpha_steps} done")

        for L in SAE_LAYERS:
            ig_dim_scores[L] /= float(n_alpha_steps)
        return ig_dim_scores

    log(f"running {ig_steps}-step IG for raw residual dimensions ...")
    t0 = time.time()
    try:
        resid_ig_scores = run_ig_resid(
            model, tokens, eval_trials, resid_last, device, ig_steps,
        )
    except Exception as e:
        if device != "cpu":
            log(f"Resid IG FAILED on {device} ({e!r}); falling back to CPU")
            traceback.print_exc()
            model_cpu = model.to("cpu")
            tokens_cpu = tokens.to("cpu")
            resid_last_cpu = {L: v.to("cpu") for L, v in resid_last.items()}
            resid_ig_scores = run_ig_resid(
                model_cpu, tokens_cpu, eval_trials, resid_last_cpu, "cpu", ig_steps,
            )
            model = model_cpu.to(device)
            tokens = tokens_cpu.to(device)
            mps_empty_cache(device)
        else:
            raise
    log(f"Residual-dimension IG done in {time.time() - t0:.1f}s")

    # Select top-N residual dimensions
    resid_scored = [
        (L, d, float(resid_ig_scores[L][d]))
        for L in SAE_LAYERS
        for d in range(d_model)
    ]
    resid_scored.sort(key=lambda x: x[2], reverse=True)
    resid_circuit = [(L, d) for L, d, _ in resid_scored[:n_features]]
    resid_circuit_set = set(resid_circuit)
    log(f"residual-dim circuit (top-{n_features}): {resid_circuit}")

    # ===================================================================
    # Step 6: Compute baseline logit-diff
    # ===================================================================
    with torch.no_grad():
        baseline_logits = model(tokens)
    baseline = float(np.mean(logit_diff_batch(baseline_logits, eval_trials)))
    log(f"baseline logit-diff = {baseline:.4f}")
    del baseline_logits
    mps_empty_cache(device)

    rng = np.random.RandomState(seed + 111)

    # ===================================================================
    # Step 7: Feature-level zero ablation
    # ===================================================================
    def feature_zero_ablation_measure(
        feature_set: List[Tuple[int, int]],
    ) -> float:
        """Zero-ablate the given SAE features: set their activations to zero,
        reconstruct, and patch the residual stream difference."""
        feat_by_layer = {
            L: [f for (lL, f) in feature_set if lL == L] for L in SAE_LAYERS
        }

        def make_hook_fn(L_fixed, feats_here):
            def fn(activation, hook):
                resid_in = activation.clone()
                cur_feats = sae_encode(resid_in[:, -1:, :], L_fixed, saes)
                ablated_feats = cur_feats.clone()
                for f_idx in feats_here:
                    ablated_feats[:, 0, int(f_idx)] = 0.0
                # Patch: add the difference between ablated and original recon
                orig_recon = sae_decode(cur_feats, L_fixed, saes)
                abl_recon = sae_decode(ablated_feats, L_fixed, saes)
                resid_in[:, -1, :] = resid_in[:, -1, :] + (
                    abl_recon[:, 0, :] - orig_recon[:, 0, :]
                )
                return resid_in
            return fn

        hooks = [
            (f"blocks.{L}.hook_resid_pre", make_hook_fn(L, feat_by_layer[L]))
            for L in SAE_LAYERS
            if feat_by_layer[L]
        ]
        with torch.no_grad():
            if hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                logits = model(tokens)
        return float(np.mean(logit_diff_batch(logits, eval_trials)))

    log("measuring feature zero-ablation drop for circuit ...")
    t0 = time.time()
    circuit_ld_zero_abl = feature_zero_ablation_measure(circuit)
    circuit_drop_zero_abl = baseline - circuit_ld_zero_abl
    log(f"  circuit zero-ablation drop = {circuit_drop_zero_abl:.4f} ({time.time() - t0:.1f}s)")
    mps_empty_cache(device)

    # ===================================================================
    # Step 8: Causal feature patching (Marks et al. 2025)
    # ===================================================================
    def causal_patch_measure(
        feature_set: List[Tuple[int, int]],
    ) -> float:
        """Causal patching: replace each circuit feature's last-position value
        with a random donor trial's value."""
        feat_by_layer = {
            L: [f for (lL, f) in feature_set if lL == L] for L in SAE_LAYERS
        }
        donor_idx = np.array([rng.randint(B - 1) for _ in range(B)])
        donor_idx = donor_idx + (donor_idx >= np.arange(B)).astype(int)

        def make_hook_fn(L_fixed, feats_here, donors):
            def fn(activation, hook):
                resid_in = activation.clone()
                cur_feats = sae_encode(resid_in[:, -1:, :], L_fixed, saes)
                patched_feats = cur_feats.clone()
                for f_idx in feats_here:
                    for b in range(B):
                        patched_feats[b, 0, int(f_idx)] = (
                            feat_last[L_fixed][int(donors[b]), int(f_idx)]
                        )
                orig_recon = sae_decode(cur_feats, L_fixed, saes)
                patch_recon = sae_decode(patched_feats, L_fixed, saes)
                resid_in[:, -1, :] = resid_in[:, -1, :] + (
                    patch_recon[:, 0, :] - orig_recon[:, 0, :]
                )
                return resid_in
            return fn

        hooks = [
            (f"blocks.{L}.hook_resid_pre", make_hook_fn(L, feat_by_layer[L], donor_idx))
            for L in SAE_LAYERS
            if feat_by_layer[L]
        ]
        with torch.no_grad():
            if hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                logits = model(tokens)
        return float(np.mean(logit_diff_batch(logits, eval_trials)))

    log("measuring causal-patch drop for circuit ...")
    t0 = time.time()
    circuit_ld_patch = causal_patch_measure(circuit)
    circuit_drop_patch = baseline - circuit_ld_patch
    log(f"  circuit causal-patch drop = {circuit_drop_patch:.4f} ({time.time() - t0:.1f}s)")
    mps_empty_cache(device)

    # ===================================================================
    # Step 9: Build null samplers for SAE features
    # ===================================================================
    den_bins = {L: compute_deciles(density[L]) for L in SAE_LAYERS}
    mag_bins = {L: compute_deciles(np.abs(magnitude[L])) for L in SAE_LAYERS}

    circuit_den_counts: Dict[Tuple[int, int], int] = {}
    circuit_mag_counts: Dict[Tuple[int, int], int] = {}
    for (L, f) in circuit:
        dkey = (L, int(den_bins[L][f]))
        mkey = (L, int(mag_bins[L][f]))
        circuit_den_counts[dkey] = circuit_den_counts.get(dkey, 0) + 1
        circuit_mag_counts[mkey] = circuit_mag_counts.get(mkey, 0) + 1

    sample_den = build_bucket_sampler(den_bins, circuit_den_counts, circuit_set, saes, rng)
    sample_mag = build_bucket_sampler(mag_bins, circuit_mag_counts, circuit_set, saes, rng)

    # ===================================================================
    # Step 10: Run null distributions for zero-ablation
    # ===================================================================
    def run_null_distribution(
        measure_fn,
        sampler,
        name: str,
        circuit_drop: float,
        n_samples: int,
    ) -> Dict:
        """Run n_samples null iterations and return full distribution + z-score."""
        drops = []
        t0 = time.time()
        for i in range(n_samples):
            null_features = sampler()
            null_ld = measure_fn(null_features)
            drops.append(baseline - null_ld)
            if i % 50 == 0:
                elapsed = time.time() - t0
                log(f"  [{name}] null {i}/{n_samples}  ({elapsed:.0f}s elapsed)")
                mps_empty_cache(device)
        drops_arr = np.array(drops)
        null_mean = float(drops_arr.mean())
        null_std = float(drops_arr.std())
        z_score = (circuit_drop - null_mean) / null_std if null_std > 1e-10 else float("inf")
        empirical_p = float(np.mean(drops_arr >= circuit_drop))
        return {
            "circuit_drop": float(circuit_drop),
            "null_drops": [float(x) for x in drops_arr],
            "null_mean": null_mean,
            "null_std": null_std,
            "null_p95": float(np.percentile(drops_arr, 95)),
            "null_p99": float(np.percentile(drops_arr, 99)),
            "z_score": float(z_score),
            "empirical_p": empirical_p,
            "passes_p05": bool(empirical_p < 0.05),
        }

    # --- Zero-ablation nulls ---
    log(f"running zero-ablation density-decile null (n={n_null}) ...")
    za_den = run_null_distribution(
        feature_zero_ablation_measure, sample_den,
        "za-density", circuit_drop_zero_abl, n_null,
    )
    log(f"  z={za_den['z_score']:.2f}, p={za_den['empirical_p']:.4f}, "
        f"PASS={za_den['passes_p05']}")

    log(f"running zero-ablation magnitude-decile null (n={n_null}) ...")
    za_mag = run_null_distribution(
        feature_zero_ablation_measure, sample_mag,
        "za-magnitude", circuit_drop_zero_abl, n_null,
    )
    log(f"  z={za_mag['z_score']:.2f}, p={za_mag['empirical_p']:.4f}, "
        f"PASS={za_mag['passes_p05']}")

    # --- Causal patching nulls ---
    log(f"running causal-patch density-decile null (n={n_null}) ...")
    cp_den = run_null_distribution(
        causal_patch_measure, sample_den,
        "cp-density", circuit_drop_patch, n_null,
    )
    # Note: causal_patch_measure returns logit-diff, and the null_distribution
    # function computes baseline - measure_fn(). So circuit_drop_patch is correct.
    log(f"  z={cp_den['z_score']:.2f}, p={cp_den['empirical_p']:.4f}, "
        f"PASS={cp_den['passes_p05']}")

    log(f"running causal-patch magnitude-decile null (n={n_null}) ...")
    cp_mag = run_null_distribution(
        causal_patch_measure, sample_mag,
        "cp-magnitude", circuit_drop_patch, n_null,
    )
    log(f"  z={cp_mag['z_score']:.2f}, p={cp_mag['empirical_p']:.4f}, "
        f"PASS={cp_mag['passes_p05']}")

    # ===================================================================
    # Step 11: Raw residual-dimension baseline audit
    # ===================================================================
    def resid_dim_zero_ablation_measure(
        dim_set: List[Tuple[int, int]],
    ) -> float:
        """Zero-ablate raw residual-stream dimensions at last position."""
        dim_by_layer = {
            L: [d for (lL, d) in dim_set if lL == L] for L in SAE_LAYERS
        }

        def make_hook_fn(L_fixed, dims_here):
            def fn(activation, hook):
                r = activation.clone()
                for d in dims_here:
                    r[:, -1, int(d)] = 0.0
                return r
            return fn

        hooks = [
            (f"blocks.{L}.hook_resid_pre", make_hook_fn(L, dim_by_layer[L]))
            for L in SAE_LAYERS
            if dim_by_layer[L]
        ]
        with torch.no_grad():
            if hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                logits = model(tokens)
        return float(np.mean(logit_diff_batch(logits, eval_trials)))

    log("measuring residual-dim zero-ablation drop for circuit ...")
    resid_circuit_ld = resid_dim_zero_ablation_measure(resid_circuit)
    resid_circuit_drop = baseline - resid_circuit_ld
    log(f"  residual-dim circuit zero-ablation drop = {resid_circuit_drop:.4f}")

    # Build magnitude-decile matched null for residual dimensions
    # Compute per-dimension activation magnitude at last position
    resid_dim_magnitude: Dict[int, np.ndarray] = {}
    for L in SAE_LAYERS:
        resid_dim_magnitude[L] = resid_last[L].abs().mean(dim=0).cpu().numpy()

    resid_mag_bins = {L: compute_deciles(resid_dim_magnitude[L]) for L in SAE_LAYERS}

    resid_circuit_mag_counts: Dict[Tuple[int, int], int] = {}
    for (L, d) in resid_circuit:
        mkey = (L, int(resid_mag_bins[L][d]))
        resid_circuit_mag_counts[mkey] = resid_circuit_mag_counts.get(mkey, 0) + 1

    # Build pools for residual dimension sampling
    resid_pools: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for (L, b) in resid_circuit_mag_counts:
        resid_pools[(L, b)] = [
            (L, d) for d in range(d_model)
            if int(resid_mag_bins[L][d]) == b and (L, d) not in resid_circuit_set
        ]

    def sample_resid_null() -> List[Tuple[int, int]]:
        out = []
        for (L, b), k in resid_circuit_mag_counts.items():
            pool = resid_pools[(L, b)]
            if len(pool) >= k:
                idx = rng.choice(len(pool), size=k, replace=False)
            elif pool:
                idx = rng.choice(len(pool), size=k, replace=True)
            else:
                continue
            for i in idx:
                out.append(pool[i])
        return out

    log(f"running residual-dim magnitude-decile null (n={n_null}) ...")
    resid_null = run_null_distribution(
        resid_dim_zero_ablation_measure, sample_resid_null,
        "resid-magnitude", resid_circuit_drop, n_null,
    )
    log(f"  z={resid_null['z_score']:.2f}, p={resid_null['empirical_p']:.4f}, "
        f"PASS={resid_null['passes_p05']}")

    # ===================================================================
    # Step 12: Compute summary z-scores and write output
    # ===================================================================
    all_z_scores = {
        "zero_ablation_density_decile": za_den["z_score"],
        "zero_ablation_magnitude_decile": za_mag["z_score"],
        "causal_patch_density_decile": cp_den["z_score"],
        "causal_patch_magnitude_decile": cp_mag["z_score"],
        "residual_dim_magnitude_decile": resid_null["z_score"],
    }

    result = {
        "status": "ok",
        "experiment": "C3_sae_feature_circuit_selectivity",
        "paper": "AAAI-27 matched-null audit",
        "task": "ioi",
        "model": "gpt2-small",
        "device": device,
        "sae_repo": SAE_REPO,
        "sae_layers": list(SAE_LAYERS),
        "d_sae": d_sae,
        "d_model": d_model,
        "n_prompts_total": len(all_trials),
        "n_prompts_eval": len(eval_trials),
        "n_features": n_features,
        "n_null_samples": n_null,
        "ig_steps": ig_steps,
        "seed": seed,
        "baseline_logit_diff": baseline,
        # Per-layer IG magnitude totals (for reporting if all features land
        # on one layer)
        "ig_magnitude_per_layer": {
            str(L): float(ig_scores[L].sum()) for L in SAE_LAYERS
        },
        # Circuit specification
        "circuit_features": [(int(L), int(f)) for (L, f) in circuit],
        "circuit_features_per_layer": {str(L): int(c) for L, c in layer_counts.items()},
        # SAE feature-level zero ablation
        "zero_ablation": {
            "circuit_drop": float(circuit_drop_zero_abl),
            "density_decile_null": za_den,
            "magnitude_decile_null": za_mag,
        },
        # SAE feature-level causal patching
        "causal_patching": {
            "circuit_drop": float(circuit_drop_patch),
            "density_decile_null": cp_den,
            "magnitude_decile_null": cp_mag,
        },
        # Raw residual-dimension baseline
        "residual_dimension_baseline": {
            "circuit_dims": [(int(L), int(d)) for (L, d) in resid_circuit],
            "circuit_drop": float(resid_circuit_drop),
            "magnitude_decile_null": resid_null,
        },
        # Summary z-scores
        "z_scores": all_z_scores,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"wrote results to {out_path}")

    # Final summary
    log("=" * 72)
    log("SUMMARY — z-scores z(C) = (circuit_drop - null_mean) / null_std")
    log("-" * 72)
    for condition, z in all_z_scores.items():
        log(f"  {condition:45s}  z = {z:+.2f}")
    log("=" * 72)
    log("PASS/FAIL (empirical p < 0.05):")
    log(f"  zero-ablation  density-decile:    p={za_den['empirical_p']:.4f}  "
        f"{'PASS' if za_den['passes_p05'] else 'FAIL'}")
    log(f"  zero-ablation  magnitude-decile:  p={za_mag['empirical_p']:.4f}  "
        f"{'PASS' if za_mag['passes_p05'] else 'FAIL'}")
    log(f"  causal-patch   density-decile:    p={cp_den['empirical_p']:.4f}  "
        f"{'PASS' if cp_den['passes_p05'] else 'FAIL'}")
    log(f"  causal-patch   magnitude-decile:  p={cp_mag['empirical_p']:.4f}  "
        f"{'PASS' if cp_mag['passes_p05'] else 'FAIL'}")
    log(f"  resid-dim      magnitude-decile:  p={resid_null['empirical_p']:.4f}  "
        f"{'PASS' if resid_null['passes_p05'] else 'FAIL'}")
    log("=" * 72)


if __name__ == "__main__":
    main()
