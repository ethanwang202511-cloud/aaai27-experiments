"""Experiment 4 (stretch goal): Dormancy audit on sci-Plex K562 with dose-response.

Tests whether CPA dormancy detection interacts with dose encoding.  When a drug
embedding is dormant (untrained, held-out drug), the dose-encoder scales a noise
vector: h_k(dose) * e_k where e_k is random init.  The audit checks whether the
learned-vs-pop_mean gap is dose-dependent for held-out drugs vs trained drugs.

Dataset: sci-Plex (Srivatsan et al., Science 2020) -- K562 subset, top 50 drugs.
Architecture: CPA with shared nonlinear dose encoder (Lotfollahi et al.).

Usage:
    python run_dose_response.py --epochs 100 --seeds 1,2,3 --n_drugs 50
    python run_dose_response.py --epochs 50 --seed 1 --device cuda:0
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

INTERVENEFM_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "InterveneFM"
DATA_DIR = INTERVENEFM_ROOT / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "exp4_dose_response"
sys.path.insert(0, str(INTERVENEFM_ROOT))

AUDIT_MODES = ["learned", "mean", "zero", "random", "pop_mean", "identity"]


# ===========================================================================
# Data loading
# ===========================================================================

def load_sciplex_k562(
    data_path: Optional[Path] = None, n_hvg: int = 2000, n_drugs: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load sci-Plex K562: returns (X, drug_labels, dose_values, is_control, drug_names)."""
    import scanpy as sc

    path = data_path or DATA_DIR / "sciplex_k562.h5ad"
    print(f"[data] loading {path} ...")
    adata = sc.read_h5ad(str(path))
    print(f"[data] raw: {adata.shape[0]} cells, {adata.shape[1]} genes")

    # Subset to K562
    for ct_col in ["cell_type", "cell_line"]:
        if ct_col in adata.obs.columns:
            mask = adata.obs[ct_col].str.upper().str.contains("K562")
            if 0 < mask.sum() < adata.shape[0]:
                adata = adata[mask].copy()
                print(f"[data] K562 subset: {adata.shape[0]} cells")
            break

    # Find perturbation and dose columns
    pert_col = next(
        (c for c in ["perturbation", "treatment", "drug", "condition", "product_name"]
         if c in adata.obs.columns), None)
    dose_col = next(
        (c for c in ["dose", "dose_value", "concentration", "dose_um"]
         if c in adata.obs.columns), None)
    if pert_col is None or dose_col is None:
        raise ValueError("Cannot find perturbation/dose columns in .obs")

    perts = adata.obs[pert_col].astype(str)
    control_names = {"dmso", "control", "vehicle", "untreated", "unperturbed"}
    is_control = perts.str.lower().str.strip().isin(control_names).values

    # Normalize + log1p + HVG (on controls)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata_ctrl = adata[is_control].copy()
    if adata_ctrl.shape[0] >= 50:
        sc.pp.highly_variable_genes(adata_ctrl, n_top_genes=n_hvg,
                                    flavor="seurat_v3", span=0.5)
        adata.var["highly_variable"] = False
        adata.var.loc[adata_ctrl.var.index[adata_ctrl.var["highly_variable"]],
                      "highly_variable"] = True
    else:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3",
                                    span=0.5)

    X = adata.X[:, adata.var["highly_variable"].values]
    X = X.toarray().astype(np.float32) if hasattr(X, "toarray") else X.astype(np.float32)

    # Top drugs, build labels
    drug_counts = perts[~is_control].value_counts()
    top_drugs = drug_counts.head(n_drugs).index.tolist()
    keep = is_control | perts.isin(top_drugs).values
    X, perts_k, is_control = X[keep], perts.values[keep], is_control[keep]

    dose_raw = adata.obs[dose_col].values[keep].astype(np.float32)
    dose_values = np.where(dose_raw > 0, np.log10(dose_raw), 0.0).astype(np.float32)
    dose_values[is_control] = 0.0

    drug_names = ["control"] + sorted(top_drugs)
    d2i = {n: i for i, n in enumerate(drug_names)}
    drug_labels = np.array([0 if c else d2i.get(p, 0)
                            for p, c in zip(perts_k, is_control)], dtype=np.int64)
    print(f"[data] final: {X.shape[0]} cells, {X.shape[1]} genes, "
          f"{len(drug_names)-1} drugs + control")
    return X, drug_labels, dose_values, is_control, drug_names


def build_drug_split(
    drug_names: List[str], holdout_frac: float = 0.2, seed: int = 1,
) -> Tuple[List[int], List[int]]:
    """Hold out a fraction of drugs for testing. Returns (train_ids, test_ids)."""
    rng = np.random.RandomState(seed)
    idxs = list(range(1, len(drug_names)))
    rng.shuffle(idxs)
    n_test = max(1, int(len(idxs) * holdout_frac))
    test_ids = sorted(idxs[:n_test])
    train_ids = [0] + sorted(idxs[n_test:])
    print(f"[split] train: {len(train_ids)-1} drugs, test: {len(test_ids)} drugs")
    return train_ids, test_ids


# ===========================================================================
# CPA with dose encoder
# ===========================================================================

class CPADose(nn.Module):
    """CPA with shared nonlinear dose encoder (Lotfollahi et al.).

    z_pert = encoder(x) + drug_proj(dose_encoder(dose) * drug_embedding(idx))
    x_hat = decoder(z_pert)
    """
    def __init__(self, n_genes: int, n_drugs: int, z_dim: int = 128,
                 drug_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, z_dim))
        self.drug_embedding = nn.Embedding(n_drugs + 1, drug_dim, padding_idx=0)
        self.dose_encoder = nn.Sequential(
            nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 1), nn.Softplus())
        self.drug_proj = (nn.Linear(drug_dim, z_dim, bias=False)
                          if drug_dim != z_dim else nn.Identity())
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_genes))

    def get_drug_effect(self, drug_idx: torch.Tensor,
                        dose: torch.Tensor) -> torch.Tensor:
        e_k = self.drug_embedding(drug_idx)            # (B, drug_dim)
        h_d = self.dose_encoder(dose.unsqueeze(-1))     # (B, 1)
        return self.drug_proj(h_d * e_k)                # (B, z_dim)

    def forward(self, x: torch.Tensor, drug_idx: torch.Tensor,
                dose: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x) + self.get_drug_effect(drug_idx, dose)
        return self.decoder(z)

    def predict_with_effect(self, x: torch.Tensor,
                            effect: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x) + effect)


# ===========================================================================
# Training
# ===========================================================================

def train_cpa_dose(
    model: CPADose, X: np.ndarray, drug_labels: np.ndarray,
    doses: np.ndarray, epochs: int = 100, batch_size: int = 256,
    lr: float = 1e-3, device: str = "cpu",
) -> List[float]:
    """Train CPA-Dose with MSE loss. Returns per-epoch losses."""
    model.to(device).train()
    ds = TensorDataset(torch.tensor(X), torch.tensor(drug_labels),
                       torch.tensor(doses))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    losses: List[float] = []
    t0 = time.time()
    for ep in range(epochs):
        ep_loss, nb = 0.0, 0
        for xb, db, doseb in loader:
            xb, db, doseb = xb.to(device), db.to(device), doseb.to(device)
            loss = F.mse_loss(model(xb, db, doseb), xb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        sched.step()
        losses.append(ep_loss / max(nb, 1))
        if (ep + 1) % max(1, epochs // 10) == 0 or ep == 0:
            print(f"  [train] epoch {ep+1:4d}/{epochs}  loss={losses[-1]:.6f}  "
                  f"({time.time()-t0:.0f}s)")
    return losses


# ===========================================================================
# Audit
# ===========================================================================

def compute_audit_predictions(
    model: CPADose, X_ctrl: np.ndarray, drug_idx: int, dose_val: float,
    train_drug_ids: List[int], X_drugged_mean: np.ndarray, device: str,
) -> Dict[str, np.ndarray]:
    """Predictions under each audit mode for one drug at one dose."""
    model.eval()
    n = len(X_ctrl)
    xt = torch.tensor(X_ctrl, dtype=torch.float32, device=device)
    dt = torch.full((n,), dose_val, dtype=torch.float32, device=device)
    preds: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        # learned
        di = torch.full((n,), drug_idx, dtype=torch.long, device=device)
        preds["learned"] = model.predict_with_effect(
            xt, model.get_drug_effect(di, dt)).cpu().numpy()

        # mean trained embedding
        tids = [d for d in train_drug_ids if d != 0]
        mean_emb = model.drug_embedding.weight[tids].mean(0, keepdim=True)
        h1 = model.dose_encoder(dt[:1].unsqueeze(-1))
        mean_eff = model.drug_proj(h1 * mean_emb).expand(n, -1)
        preds["mean"] = model.predict_with_effect(xt, mean_eff).cpu().numpy()

        # zero
        preds["zero"] = model.predict_with_effect(
            xt, torch.zeros(n, model.z_dim, device=device)).cpu().numpy()

        # random (average 5 random trained embeddings)
        rng = np.random.RandomState(42)
        rp = []
        for _ in range(5):
            ri = rng.choice(tids)
            re = model.drug_proj(h1 * model.drug_embedding.weight[ri:ri+1]).expand(n,-1)
            rp.append(model.predict_with_effect(xt, re).cpu().numpy())
        preds["random"] = np.mean(rp, axis=0)

        # pop_mean (model-free)
        preds["pop_mean"] = np.tile(X_drugged_mean, (n, 1))
        # identity
        preds["identity"] = X_ctrl.copy()

    return preds


def audit_cpa_dose(
    model: CPADose, X: np.ndarray, drug_labels: np.ndarray,
    dose_values: np.ndarray, is_control: np.ndarray, drug_names: List[str],
    train_ids: List[int], test_ids: List[int], device: str = "cpu",
    n_ctrl: int = 200, n_de: int = 50,
) -> pd.DataFrame:
    """Run dormancy audit for all drugs at each dose. Returns results DataFrame."""
    import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)

    X_ctrl = X[is_control]
    if X_ctrl.shape[0] > n_ctrl:
        X_ctrl = X_ctrl[np.random.RandomState(0).choice(X_ctrl.shape[0], n_ctrl, False)]

    drugged_mask = np.isin(drug_labels, train_ids) & ~is_control
    X_drugged_mean = X[drugged_mask].mean(0).astype(np.float32)
    ctrl_mean = X_ctrl.mean(0)
    unique_doses = sorted(set(dose_values[~is_control]))
    print(f"[audit] {len(unique_doses)} doses, auditing "
          f"{len(train_ids)-1 + len(test_ids)} drugs")

    records: List[Dict] = []
    for drug_idx in (train_ids[1:] + test_ids):
        split = "train" if drug_idx in train_ids else "test"
        dmask = drug_labels == drug_idx
        for dv in unique_doses:
            gt_mask = dmask & (np.abs(dose_values - dv) < 0.01)
            if gt_mask.sum() < 5:
                continue
            gt_mean = X[gt_mask].mean(0)
            de_idx = np.argsort(np.abs(gt_mean - ctrl_mean))[-n_de:]
            preds = compute_audit_predictions(
                model, X_ctrl, drug_idx, dv, train_ids, X_drugged_mean, device)
            for mode, pred in preds.items():
                pm = pred.mean(0)
                rho, _ = spearmanr(gt_mean[de_idx], pm[de_idx])
                records.append(dict(
                    drug=drug_names[drug_idx], drug_idx=drug_idx, dose=dv,
                    n_cells=int(gt_mask.sum()), split=split, mode=mode,
                    de_spearman=float(rho if not np.isnan(rho) else 0.0),
                    mse=float(np.mean((gt_mean[de_idx] - pm[de_idx])**2))))

    df = pd.DataFrame(records)
    print(f"[audit] {len(df)} records")
    return df


# ===========================================================================
# Dose-response diagnostics and plotting
# ===========================================================================

def compute_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Learned-vs-pop_mean DE-Spearman gap per drug per dose."""
    l = df[df["mode"]=="learned"].set_index(["drug","dose"])
    p = df[df["mode"]=="pop_mean"].set_index(["drug","dose"])
    rows = []
    for idx in l.index.intersection(p.index):
        rows.append(dict(drug=idx[0], dose=idx[1], split=l.loc[idx,"split"],
                         gap=float(l.loc[idx,"de_spearman"]-p.loc[idx,"de_spearman"]),
                         rho_learned=float(l.loc[idx,"de_spearman"])))
    return pd.DataFrame(rows)


def dose_monotonicity_diagnostic(df: pd.DataFrame, out_dir: Path) -> Dict[str, float]:
    """Plot dose-response diagnostics; return monotonicity summary."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gap_df = compute_gap(df)
    summary: Dict[str, float] = {}

    # Monotonicity: Spearman(dose, rho_learned) per drug
    for split in ["train", "test"]:
        sdf = gap_df[gap_df["split"] == split]
        rhos = []
        for drug in sdf["drug"].unique():
            dd = sdf[sdf["drug"]==drug].sort_values("dose")
            if len(dd) >= 3:
                r, _ = spearmanr(dd["dose"], dd["rho_learned"])
                if not np.isnan(r):
                    rhos.append(r)
        rhos = np.array(rhos)
        summary[f"{split}_dose_rho_mean"] = float(np.mean(rhos)) if len(rhos) else 0.0
        summary[f"{split}_dose_rho_std"] = float(np.std(rhos)) if len(rhos) else 0.0
        summary[f"n_{split}"] = len(rhos)

    # -- Plot 1: gap vs dose --
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, split in zip(axes, ["train", "test"]):
        sdf = gap_df[gap_df["split"]==split]
        if sdf.empty: ax.set_title(f"{split} (no data)"); continue
        for d in sdf["drug"].unique():
            dd = sdf[sdf["drug"]==d].sort_values("dose")
            ax.plot(dd["dose"], dd["gap"], "o-", alpha=0.4, ms=3)
        agg = sdf.groupby("dose")["gap"].agg(["mean","std","count"])
        agg["sem"] = agg["std"]/np.sqrt(agg["count"])
        ax.errorbar(agg.index, agg["mean"], yerr=agg["sem"], color="black",
                     lw=2, capsize=4, zorder=10, label="mean+/-SEM")
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.set(xlabel="log10(dose uM)", ylabel="gap (learned-pop_mean)",
               title=f"{split} drugs (n={sdf['drug'].nunique()})")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "dose_response_gap.png", dpi=150); plt.close(fig)

    # -- Plot 2: DE-Spearman by mode --
    mcols = {"learned":"tab:blue","pop_mean":"tab:orange","mean":"tab:green",
             "zero":"tab:gray","identity":"tab:purple"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, split in zip(axes, ["train", "test"]):
        sdf = df[df["split"]==split]
        for mode in ["learned","pop_mean","mean","zero","identity"]:
            mdf = sdf[sdf["mode"]==mode]
            if mdf.empty: continue
            a = mdf.groupby("dose")["de_spearman"].agg(["mean","std","count"])
            a["sem"] = a["std"]/np.sqrt(a["count"])
            ax.errorbar(a.index, a["mean"], yerr=a["sem"], label=mode,
                         color=mcols.get(mode,"black"), marker="o", ms=4, capsize=3)
        ax.axhline(0, color="gray", ls="--", alpha=0.3)
        ax.set(xlabel="log10(dose uM)", ylabel="DE-Spearman",
               title=f"{split} drugs")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "dose_response_modes.png", dpi=150); plt.close(fig)

    # -- Plot 3: monotonicity histogram --
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-1, 1, 21)
    for split, color in [("train","steelblue"),("test","tomato")]:
        sdf = gap_df[gap_df["split"]==split]
        rhos = []
        for d in sdf["drug"].unique():
            dd = sdf[sdf["drug"]==d].sort_values("dose")
            if len(dd)>=3:
                r,_ = spearmanr(dd["dose"], dd["rho_learned"])
                if not np.isnan(r): rhos.append(r)
        if rhos:
            ax.hist(rhos, bins=bins, alpha=0.6, label=f"{split} drugs", color=color)
    ax.set(xlabel="Spearman(dose, rho_learned)", ylabel="count",
           title="Dose-monotonicity of learned predictions")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "dose_monotonicity.png", dpi=150); plt.close(fig)

    for s in ["train","test"]:
        print(f"[mono] {s}: rho={summary[f'{s}_dose_rho_mean']:.3f} "
              f"+/- {summary[f'{s}_dose_rho_std']:.3f} (n={summary[f'n_{s}']})")
    return summary


# ===========================================================================
# Main experiment
# ===========================================================================

def run_experiment(
    epochs: int = 100, seeds: List[int] = [1,2,3], n_drugs: int = 50,
    z_dim: int = 128, drug_dim: int = 128, hidden_dim: int = 256,
    batch_size: int = 256, lr: float = 1e-3, holdout_frac: float = 0.2,
    device: str = "auto", n_de: int = 50, data_path: Optional[str] = None,
) -> None:
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[exp4] device={device}, epochs={epochs}, seeds={seeds}")

    dp = Path(data_path) if data_path else None
    X, drug_labels, dose_values, is_control, drug_names = load_sciplex_k562(
        data_path=dp, n_drugs=n_drugs)
    n_genes = X.shape[1]
    n_drugs_total = len(drug_names) - 1

    all_results: List[pd.DataFrame] = []
    all_mono: List[Dict] = []

    for seed in seeds:
        print(f"\n{'='*60}\n[exp4] seed={seed}\n{'='*60}")
        sdir = RESULTS_DIR / f"seed_{seed}"
        os.makedirs(sdir, exist_ok=True)
        torch.manual_seed(seed); np.random.seed(seed)

        train_ids, test_ids = build_drug_split(drug_names, holdout_frac, seed)
        train_mask = np.isin(drug_labels, train_ids)
        print(f"[exp4] training cells: {train_mask.sum()}")

        model = CPADose(n_genes, n_drugs_total, z_dim, drug_dim, hidden_dim)
        print(f"[exp4] params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

        losses = train_cpa_dose(model, X[train_mask], drug_labels[train_mask],
                                dose_values[train_mask], epochs, batch_size, lr, device)
        pd.DataFrame({"epoch": range(1,len(losses)+1), "loss": losses}).to_csv(
            sdir / "training_loss.csv", index=False)

        adf = audit_cpa_dose(model, X, drug_labels, dose_values, is_control,
                             drug_names, train_ids, test_ids, device, n_de=n_de)
        adf["seed"] = seed
        adf.to_csv(sdir / "audit_results.csv", index=False)
        all_results.append(adf)

        mono = dose_monotonicity_diagnostic(adf, sdir)
        mono["seed"] = seed
        all_mono.append(mono)

        with open(sdir / "split_info.json", "w") as f:
            json.dump({"seed": seed, "n_train_cells": int(train_mask.sum()),
                        "train_drugs": [drug_names[i] for i in train_ids if i>0],
                        "test_drugs": [drug_names[i] for i in test_ids]}, f, indent=2)

    # Aggregate
    print(f"\n{'='*60}\n[exp4] Aggregating across seeds\n{'='*60}")
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(RESULTS_DIR / "audit_all_seeds.csv", index=False)

    gap_all = pd.concat([compute_gap(r) for r in all_results], ignore_index=True)
    summ = gap_all.groupby(["split","dose"])["gap"].agg(["mean","std","count"]).reset_index()
    summ["sem"] = summ["std"] / np.sqrt(summ["count"])
    summ.to_csv(RESULTS_DIR / "gap_summary.csv", index=False)
    print("\n[exp4] Gap summary (learned - pop_mean):")
    print(summ.to_string(index=False, float_format="%.4f"))

    # Aggregated plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for split, c, m in [("train","steelblue","o"),("test","tomato","s")]:
        s = gap_all[gap_all["split"]==split]
        if s.empty: continue
        a = s.groupby("dose")["gap"].agg(["mean","std","count"])
        a["sem"] = a["std"]/np.sqrt(a["count"])
        ax.errorbar(a.index, a["mean"], yerr=a["sem"], label=f"{split} drugs",
                     color=c, marker=m, capsize=4, lw=2, ms=6)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set(xlabel="log10(dose uM)", ylabel="gap (learned - pop_mean)",
           title="Dose-dependent dormancy gap (aggregated)")
    ax.legend(); fig.tight_layout()
    fig.savefig(RESULTS_DIR / "aggregated_dose_gap.png", dpi=150); plt.close(fig)

    pd.DataFrame(all_mono).to_csv(RESULTS_DIR / "monotonicity_summary.csv", index=False)
    print(f"\n[exp4] Results saved to {RESULTS_DIR}/")


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Exp4: Dose-response dormancy audit on sci-Plex K562")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seeds", type=str, default="1,2,3",
                   help="Comma-separated seeds")
    p.add_argument("--seed", type=int, default=None,
                   help="Single seed (overrides --seeds)")
    p.add_argument("--n_drugs", type=int, default=50)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--drug_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--holdout_frac", type=float, default=0.2)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--n_de_genes", type=int, default=50)
    p.add_argument("--data_path", type=str, default=None)
    args = p.parse_args()

    seeds = [args.seed] if args.seed is not None else [int(s) for s in args.seeds.split(",")]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_experiment(args.epochs, seeds, args.n_drugs, args.z_dim, args.drug_dim,
                   args.hidden_dim, args.batch_size, args.lr, args.holdout_frac,
                   args.device, args.n_de_genes, args.data_path)


if __name__ == "__main__":
    main()
