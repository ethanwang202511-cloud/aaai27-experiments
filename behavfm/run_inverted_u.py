"""Experiment 2: Expanded Inverted-U Validation (3 -> 8 species).

Validates the paper's central theoretical claim: an inverted-U relationship
between phylogenetic distance (measured by cosine similarity of DINOv2 patch
features) and ITA adaptation gain.

Original paper: 3 species (fox, elephant, chimpanzee).
This experiment: 8 species spanning cos_max from 0.834 to 0.493.

Species selection (EXPERIMENT_PLAN.md):
  - deer      cos=0.834  (very close)
  - wolf      cos=0.767  (close)
  - fox       cos=0.746  (close)        [existing]
  - moose     cos=0.698  (mid-close)
  - raccoon   cos=0.626  (mid-far)
  - polar_bear cos=0.555 (far)
  - elephant  cos=0.501  (far)          [existing]
  - chimpanzee cos=0.493 (far)          [existing]

Methods: ITA (768 params), BitFit (~14K), LoRA-r4 (~75K), full decoder FT
Seeds: 10 per condition
Analysis: quadratic vs linear regression, bootstrap CI on quadratic coefficient

Usage:
    python run_inverted_u.py \
        --train-cache data/cache/dinov2_base_ap10k_train \
        --val-cache data/cache/dinov2_base_ap10k_val \
        --ckpt results/q2_within_species/iti_xattn_aux01_seed0/model.pt \
        --out-dir results/inverted_u \
        --device cuda

    # Smoke test
    python run_inverted_u.py \
        --train-cache data/cache/dinov2_base_ap10k_train \
        --val-cache data/cache/dinov2_base_ap10k_val \
        --ckpt results/model.pt \
        --species fox elephant --n-seeds 2 --n-adapt-steps 10
"""

from __future__ import annotations
import argparse
import copy
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
_SRC = Path(os.environ.get("BEHAVFM_ROOT", ROOT.parent.parent.parent)) / "src"
if (_SRC / "iti").is_dir():
    sys.path.insert(0, str(_SRC.parent))
    sys.path.insert(0, str(_SRC))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("inverted_u")


# ---------------------------------------------------------------------------
# Species metadata
# ---------------------------------------------------------------------------

# Cosine similarity to nearest training species (from zero_shot_distance_curve.csv).
SPECIES_COS_MAX = {
    "deer":       0.834,
    "wolf":       0.767,
    "fox":        0.746,
    "moose":      0.698,
    "raccoon":    0.626,
    "polar_bear": 0.555,
    "elephant":   0.501,
    "chimpanzee": 0.493,
}

ALL_SPECIES = list(SPECIES_COS_MAX.keys())

METHODS = ["ita_token_only", "bitfit", "lora_r4", "full_decoder_ft"]


# ---------------------------------------------------------------------------
# Feature cache dataset (standalone)
# ---------------------------------------------------------------------------

class CachedFeatures:
    """Minimal feature cache loader."""

    def __init__(self, cache_dir: Path, species_subset: set[str] | None = None):
        self.cache_dir = Path(cache_dir)
        meta = json.loads((self.cache_dir / "meta.json").read_text())
        self._features = np.load(self.cache_dir / "features.npy", mmap_mode="r")
        self.records = meta["records"]
        self.indices = []
        for i, r in enumerate(self.records):
            if species_subset and r["species_name"] not in species_subset:
                continue
            self.indices.append(i)
        self.indices = np.array(self.indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        ridx = int(self.indices[i])
        r = self.records[ridx]
        feat = torch.from_numpy(np.array(self._features[ridx])).float()
        return {
            "feature": feat,
            "identity_id": r["identity_id"],
            "keypoints_xy": torch.tensor(r["keypoints_xy"]).float(),
            "vis": torch.tensor(r["vis"]).float(),
            "bbox_diag_orig": float(r["bbox_diag_orig"]),
            "species_name": r["species_name"],
            "crop_side": float(r["crop_side"]),
        }


def collate_fn(batch):
    return {
        "feature": torch.stack([b["feature"] for b in batch]),
        "identity_id": torch.tensor([b["identity_id"] for b in batch], dtype=torch.long),
        "keypoints_xy": torch.stack([b["keypoints_xy"] for b in batch]),
        "vis": torch.stack([b["vis"] for b in batch]),
        "bbox_diag_orig": torch.tensor([b["bbox_diag_orig"] for b in batch]),
        "species_name": [b["species_name"] for b in batch],
        "crop_side": torch.tensor([b["crop_side"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_xy, gt_xy, vis, crop_side, bbox_diag, out_size=224):
    scale = crop_side / out_size
    diff = pred_xy - gt_xy
    dist = np.linalg.norm(diff, axis=-1)
    dist_orig = dist * scale[:, None]
    dist_orig = np.where(vis > 0, dist_orig, np.nan)
    norm = bbox_diag[:, None]
    dist_norm = dist_orig / norm
    flat = dist_norm[~np.isnan(dist_norm)]
    if flat.size == 0:
        return {"rmse_norm": float("nan"), "pck@0.05": float("nan"),
                "pck@0.10": float("nan"), "pck@0.20": float("nan")}
    return {
        "rmse_norm": float(np.sqrt((flat ** 2).mean())),
        "pck@0.05": float((flat < 0.05).mean()),
        "pck@0.10": float((flat < 0.10).mean()),
        "pck@0.20": float((flat < 0.20).mean()),
    }


def bootstrap_ci(vals, n=1000, ci=0.95, seed=0):
    rng = np.random.default_rng(seed)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.array([vals[rng.integers(0, len(vals), size=len(vals))].mean() for _ in range(n)])
    lo = float(np.quantile(means, (1 - ci) / 2))
    hi = float(np.quantile(means, 1 - (1 - ci) / 2))
    return float(vals.mean()), lo, hi


# ---------------------------------------------------------------------------
# PEFT adapter utilities (standalone)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0):
        super().__init__()
        self.r, self.alpha, self.scaling = r, alpha, alpha / r
        out_f, in_f = base.weight.shape
        self.in_features, self.out_features = in_f, out_f
        self.register_buffer("weight", base.weight.detach().clone())
        if base.bias is not None:
            self.register_buffer("bias", base.bias.detach().clone())
        else:
            self.bias = None
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x):
        from torch.nn.functional import linear
        base_out = linear(x, self.weight, self.bias)
        lora_out = linear(linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


def wrap_with_lora(module: nn.Module, r: int = 4, alpha: float = 8.0) -> int:
    n_params = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            lora = LoRALinear(child, r=r, alpha=alpha)
            setattr(module, name, lora)
            n_params += lora.lora_A.numel() + lora.lora_B.numel()
        else:
            n_params += wrap_with_lora(child, r, alpha)
    return n_params


# ---------------------------------------------------------------------------
# Adaptation + evaluation
# ---------------------------------------------------------------------------

def adapt_and_evaluate(
    model_state: dict,
    cfg_dict: dict,
    species: str,
    method: str,
    seed: int,
    train_cache: Path,
    val_cache: Path,
    k_shot: int,
    n_adapt_steps: int,
    adapt_lr: float,
    device: str,
) -> dict:
    """Run adaptation with one method on one species, return metrics."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Try to import from existing codebase; fall back to standalone
    try:
        from iti.model.iti import ITIModel, ITIConfig
        cfg = ITIConfig(**{k: v for k, v in cfg_dict.items()
                          if k in ITIConfig.__dataclass_fields__})
        model = ITIModel(cfg).to(device)
        model.load_state_dict(model_state["model_state_dict"])
        model.eval()
        use_existing = True
    except (ImportError, KeyError):
        use_existing = False
        logger.info("Falling back to standalone model (no ITI source)")
        return {"status": "error", "reason": "ITI source not available"}

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad_(False)

    # Load species data
    # Find the species' global ID from the train cache
    train_ds = CachedFeatures(train_cache, species_subset={species})
    val_ds = CachedFeatures(val_cache, species_subset={species})

    if len(train_ds) == 0 or len(val_ds) == 0:
        return {"status": "skipped", "reason": f"no data for {species}",
                "species": species, "method": method, "seed": seed}

    gid = train_ds.records[int(train_ds.indices[0])]["identity_id"]

    # Get support set
    train_all = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False, collate_fn=collate_fn)
    train_batch = next(iter(train_all))
    train_features = train_batch["feature"].to(device)

    idx = rng.choice(len(train_ds), size=min(k_shot, len(train_ds)), replace=False)
    sup_f = train_features[idx]
    sup_kp = train_batch["keypoints_xy"][idx].to(device)
    sup_vis = train_batch["vis"][idx].to(device)

    # Setup method
    with torch.no_grad():
        mean_tok = model.tokens.emb.weight.detach().mean(dim=0)

    if method == "ita_token_only":
        # Initialize via interpolation head if available
        if model.interpolation_head is not None:
            id_bank = model.tokens.emb.weight.detach()
            ex_feat = sup_f.mean(dim=0).unsqueeze(0)
            init_tok, _ = model.interpolation_head(ex_feat, id_bank)
            tok = nn.Parameter(init_tok.squeeze(0).clone())
        else:
            tok = nn.Parameter(mean_tok.clone())
        tok.requires_grad_(True)
        extra_params = []

    elif method == "bitfit":
        tok = nn.Parameter(mean_tok.clone())
        tok.requires_grad_(False)
        extra_params = [p for n, p in model.decoder.named_parameters() if n.endswith(".bias")]
        for p in extra_params:
            p.requires_grad_(True)

    elif method == "lora_r4":
        tok = nn.Parameter(mean_tok.clone())
        tok.requires_grad_(False)
        wrap_with_lora(model.decoder, r=4, alpha=8.0)
        model.to(device)
        extra_params = [p for p in model.decoder.parameters() if p.requires_grad]

    elif method == "full_decoder_ft":
        tok = nn.Parameter(mean_tok.clone())
        tok.requires_grad_(False)
        for p in model.decoder.parameters():
            p.requires_grad_(True)
        extra_params = list(model.decoder.parameters())

    else:
        return {"status": "error", "reason": f"unknown method: {method}"}

    params_to_opt = ([tok] + extra_params) if tok.requires_grad else extra_params
    n_trainable = sum(p.numel() for p in params_to_opt)
    opt = torch.optim.SGD(params_to_opt, lr=adapt_lr, momentum=0.9)

    # Adapt
    model.train()
    for step in range(n_adapt_steps):
        B = sup_f.shape[0]
        pred = model.forward_with_token(sup_f, tok.unsqueeze(0).expand(B, -1))
        target = sup_kp / 224.0
        per_kp = ((pred - target) ** 2).sum(dim=-1)
        loss = (per_kp * sup_vis).sum() / sup_vis.sum().clamp_min(1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Evaluate
    model.eval()
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    all_pred, all_gt, all_vis, all_crop, all_diag = [], [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            fv = batch["feature"].to(device)
            Bv = fv.shape[0]
            pred_unit = model.forward_with_token(
                fv, tok.detach().unsqueeze(0).expand(Bv, -1)
            ).cpu()
            pred_xy = pred_unit * 224.0
            all_pred.append(pred_xy.numpy())
            all_gt.append(batch["keypoints_xy"].numpy())
            all_vis.append(batch["vis"].numpy())
            all_crop.append(batch["crop_side"].numpy())
            all_diag.append(batch["bbox_diag_orig"].numpy())

    metrics = compute_metrics(
        np.concatenate(all_pred), np.concatenate(all_gt),
        np.concatenate(all_vis), np.concatenate(all_crop),
        np.concatenate(all_diag),
    )

    return {
        "species": species,
        "method": method,
        "seed": seed,
        "n_trainable": n_trainable,
        "cos_max": SPECIES_COS_MAX.get(species, float("nan")),
        **metrics,
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# No-adaptation baseline
# ---------------------------------------------------------------------------

def no_adapt_evaluate(
    model_state: dict,
    cfg_dict: dict,
    species: str,
    val_cache: Path,
    device: str,
) -> dict:
    """Evaluate with mean token (no adaptation) to compute baseline."""
    try:
        from iti.model.iti import ITIModel, ITIConfig
        cfg = ITIConfig(**{k: v for k, v in cfg_dict.items()
                          if k in ITIConfig.__dataclass_fields__})
        model = ITIModel(cfg).to(device)
        model.load_state_dict(model_state["model_state_dict"])
        model.eval()
    except (ImportError, KeyError):
        return {"status": "error"}

    for p in model.parameters():
        p.requires_grad_(False)

    val_ds = CachedFeatures(val_cache, species_subset={species})
    if len(val_ds) == 0:
        return {"status": "skipped"}

    with torch.no_grad():
        mean_tok = model.tokens.emb.weight.mean(dim=0)

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    all_pred, all_gt, all_vis, all_crop, all_diag = [], [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            fv = batch["feature"].to(device)
            Bv = fv.shape[0]
            pred_unit = model.forward_with_token(
                fv, mean_tok.unsqueeze(0).expand(Bv, -1)
            ).cpu()
            pred_xy = pred_unit * 224.0
            all_pred.append(pred_xy.numpy())
            all_gt.append(batch["keypoints_xy"].numpy())
            all_vis.append(batch["vis"].numpy())
            all_crop.append(batch["crop_side"].numpy())
            all_diag.append(batch["bbox_diag_orig"].numpy())

    metrics = compute_metrics(
        np.concatenate(all_pred), np.concatenate(all_gt),
        np.concatenate(all_vis), np.concatenate(all_crop),
        np.concatenate(all_diag),
    )
    return {
        "species": species,
        "method": "no_adapt",
        "cos_max": SPECIES_COS_MAX.get(species, float("nan")),
        **metrics,
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def fit_regression(results: list[dict]) -> dict:
    """Fit quadratic vs linear regression on adaptation gain vs cosine distance.

    Adaptation gain = (no_adapt_RMSE - method_RMSE) / no_adapt_RMSE
    """
    from scipy import stats as sp_stats

    analysis = {}

    # Get no-adapt baselines per species
    no_adapt = {r["species"]: r["rmse_norm"] for r in results
                if r.get("method") == "no_adapt" and r.get("status") == "ok"}

    for method in METHODS:
        method_rows = [r for r in results
                       if r.get("method") == method and r.get("status") == "ok"]
        if not method_rows:
            continue

        # Compute per-species mean adaptation gain
        species_gains = {}
        for sp in ALL_SPECIES:
            baseline = no_adapt.get(sp)
            if baseline is None or np.isnan(baseline):
                continue
            sp_rows = [r for r in method_rows if r["species"] == sp]
            if not sp_rows:
                continue
            mean_rmse = np.mean([r["rmse_norm"] for r in sp_rows])
            gain = (baseline - mean_rmse) / baseline
            species_gains[sp] = {
                "gain": gain,
                "cos_max": SPECIES_COS_MAX[sp],
                "n_runs": len(sp_rows),
                "rmse_mean": mean_rmse,
                "baseline_rmse": baseline,
            }

        if len(species_gains) < 3:
            continue

        x = np.array([v["cos_max"] for v in species_gains.values()])
        y = np.array([v["gain"] for v in species_gains.values()])

        # Linear fit
        slope_l, intercept_l, r_l, p_l, _ = sp_stats.linregress(x, y)
        y_pred_l = slope_l * x + intercept_l
        ss_res_l = np.sum((y - y_pred_l) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_linear = 1 - ss_res_l / ss_tot if ss_tot > 0 else float("nan")

        # Quadratic fit
        coeffs = np.polyfit(x, y, 2)
        y_pred_q = np.polyval(coeffs, x)
        ss_res_q = np.sum((y - y_pred_q) ** 2)
        r2_quad = 1 - ss_res_q / ss_tot if ss_tot > 0 else float("nan")

        # AIC comparison (assuming normal errors)
        n = len(x)
        aic_linear = n * np.log(ss_res_l / n + 1e-10) + 2 * 2
        aic_quad = n * np.log(ss_res_q / n + 1e-10) + 2 * 3

        # Bootstrap CI on quadratic coefficient
        n_boot = 2000
        rng = np.random.default_rng(42)
        boot_a2 = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            try:
                c = np.polyfit(x[idx], y[idx], 2)
                boot_a2.append(c[0])
            except np.linalg.LinAlgError:
                pass
        boot_a2 = np.array(boot_a2)
        a2_mean = float(boot_a2.mean()) if len(boot_a2) else float("nan")
        a2_lo = float(np.quantile(boot_a2, 0.025)) if len(boot_a2) else float("nan")
        a2_hi = float(np.quantile(boot_a2, 0.975)) if len(boot_a2) else float("nan")
        # p-value: fraction of bootstraps where a2 >= 0 (for inverted-U, expect a2 < 0)
        p_inverted_u = float((boot_a2 >= 0).mean()) if len(boot_a2) else float("nan")

        analysis[method] = {
            "per_species": species_gains,
            "linear": {
                "slope": float(slope_l),
                "intercept": float(intercept_l),
                "r2": float(r2_linear),
                "p_value": float(p_l),
                "aic": float(aic_linear),
            },
            "quadratic": {
                "coefficients": [float(c) for c in coeffs],
                "r2": float(r2_quad),
                "aic": float(aic_quad),
                "a2_mean": a2_mean,
                "a2_ci95": [a2_lo, a2_hi],
                "p_inverted_u": p_inverted_u,
            },
            "delta_aic": float(aic_linear - aic_quad),
            "inverted_u_supported": bool(a2_mean < 0 and p_inverted_u < 0.05),
        }

    return analysis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Experiment 2: Expanded Inverted-U Validation")
    p.add_argument("--train-cache", type=Path, required=True)
    p.add_argument("--val-cache", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Pretrained ITI model checkpoint.")
    p.add_argument("--out-dir", type=Path, default=Path("results/inverted_u"))
    p.add_argument("--species", nargs="+", default=ALL_SPECIES)
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--n-adapt-steps", type=int, default=50)
    p.add_argument("--adapt-lr", type=float, default=1.0)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Device: %s | species: %s | methods: %s", device, args.species, args.methods)

    # Load checkpoint
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg_dict = state.get("cfg", {})

    all_results = []

    # 1. No-adaptation baselines
    logger.info("=== Computing no-adaptation baselines ===")
    for species in args.species:
        result = no_adapt_evaluate(state, cfg_dict, species, args.val_cache, device)
        all_results.append(result)
        if result.get("status") == "ok":
            logger.info("  %s: no_adapt rmse=%.4f", species, result["rmse_norm"])

    # 2. Adaptation runs
    total = len(args.species) * len(args.methods) * args.n_seeds
    done = 0
    for species in args.species:
        for method in args.methods:
            for seed in range(args.n_seeds):
                done += 1
                logger.info("[%d/%d] species=%s method=%s seed=%d",
                            done, total, species, method, seed)

                result = adapt_and_evaluate(
                    model_state=state,
                    cfg_dict=cfg_dict,
                    species=species,
                    method=method,
                    seed=seed,
                    train_cache=args.train_cache,
                    val_cache=args.val_cache,
                    k_shot=args.k_shot,
                    n_adapt_steps=args.n_adapt_steps,
                    adapt_lr=args.adapt_lr,
                    device=device,
                )
                all_results.append(result)

                # Save incrementally
                with open(args.out_dir / "inverted_u_results.json", "w") as f:
                    json.dump(all_results, f, indent=2)

    # 3. Statistical analysis
    logger.info("=== Running regression analysis ===")
    try:
        analysis = fit_regression(all_results)
        with open(args.out_dir / "inverted_u_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

        for method, a in analysis.items():
            logger.info(
                "  %s: linear R2=%.3f, quad R2=%.3f, delta_AIC=%.1f, "
                "a2=%.4f [%.4f, %.4f], inverted_u=%s",
                method,
                a["linear"]["r2"], a["quadratic"]["r2"],
                a["delta_aic"],
                a["quadratic"]["a2_mean"],
                a["quadratic"]["a2_ci95"][0], a["quadratic"]["a2_ci95"][1],
                a["inverted_u_supported"],
            )
    except ImportError:
        logger.warning("scipy not available; skipping regression analysis. "
                       "Install scipy and re-run, or analyze inverted_u_results.json manually.")

    # 4. Per-species summary
    summary = {}
    for species in args.species:
        summary[species] = {"cos_max": SPECIES_COS_MAX.get(species)}
        for method in ["no_adapt"] + args.methods:
            rows = [r for r in all_results
                    if r.get("species") == species and r.get("method") == method
                    and r.get("status") == "ok"]
            if not rows:
                continue
            rmses = np.array([r["rmse_norm"] for r in rows])
            m, lo, hi = bootstrap_ci(rmses)
            summary[species][method] = {
                "rmse_mean": m, "rmse_ci95": [lo, hi], "n_runs": len(rows),
            }

    with open(args.out_dir / "inverted_u_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done. Output: %s", args.out_dir)


if __name__ == "__main__":
    main()
