"""
Experiment 5 -- Conditional-helper effect replication and extension.

Replicates the conditional-helper effect from the MusIML workshop paper
(shown on Norman only) across K562 and RPE1, and extends to GEARS and scGPT.

The core observation: the gap (learned DE-Spearman minus pop_mean DE-Spearman)
decomposes by pop_mean performance tertile:
  LOW   (worst pop_mean, strongest gene-specific effects) -> gap > 0 (embedding helps)
  HIGH  (best pop_mean, weakest gene-specific effects)    -> gap < 0 (embedding hurts)

This script performs reanalysis of existing audit CSVs where available, and
can train + audit fresh models when results do not exist.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

INTERVENEFM_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "InterveneFM"
sys.path.insert(0, str(INTERVENEFM_ROOT))

from src.audit import get_top_deg_indices, predict_under_mode, run_audit  # noqa: E402
from src.cpa_minimal import CPAConfig, CPAMinimal  # noqa: E402
from src.data_norman import (  # noqa: E402
    PerturbSeqDataset,
    build_gene_vocab,
    build_split,
    load_norman,
)
from src.data_replogle import (  # noqa: E402
    build_gene_vocab_replogle,
    build_split_replogle,
    load_replogle,
)
from src.data_replogle_rpe1 import load_replogle_rpe1  # noqa: E402

RESULTS_DIR = INTERVENEFM_ROOT / "results"
OUT_DIR = RESULTS_DIR / "exp5_conditional_helper"


# ---------------------------------------------------------------------------
# Known audit CSV paths -- extend as new results appear
# ---------------------------------------------------------------------------

KNOWN_AUDITS: Dict[Tuple[str, str], List[str]] = {
    ("cpa", "norman"): [
        "audit_multiseed_norman02_all_seeds.csv",
    ],
    ("cpa", "k562"): [
        "audit_replogle_3seed_all.csv",
    ],
    ("gears", "norman"): [
        "audit_gears_norman_e5_seed1.csv",
    ],
}

# Column that identifies the perturbation condition across formats
PAIR_COL_MAP = {
    "norman": "pair",
    "k562": "test_gene",
    "rpe1": "test_gene",
}


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def load_existing_audit(path: Path) -> pd.DataFrame:
    """Load an audit CSV, normalising column names."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def _resolve_pair_col(df: pd.DataFrame, dataset: str) -> str:
    """Return the column name that identifies the perturbation condition."""
    candidate = PAIR_COL_MAP.get(dataset)
    if candidate and candidate in df.columns:
        return candidate
    # GEARS uses 'condition'
    if "condition" in df.columns:
        return "condition"
    for col in ("pair", "test_gene", "condition"):
        if col in df.columns:
            return col
    raise KeyError(f"Cannot find pair/gene column in {list(df.columns)}")


def compute_pop_mean_performance(
    df: pd.DataFrame, dataset: str
) -> pd.Series:
    """Return per-condition pop_mean DE-Spearman, averaged across seeds."""
    pair_col = _resolve_pair_col(df, dataset)
    pop = df[df["mode"] == "pop_mean"].copy()
    if pop.empty:
        raise ValueError("No pop_mean rows found in audit DataFrame")
    return pop.groupby(pair_col)["DE_Spearman"].mean()


def compute_learned_performance(
    df: pd.DataFrame, dataset: str
) -> pd.Series:
    """Return per-condition learned DE-Spearman, averaged across seeds."""
    pair_col = _resolve_pair_col(df, dataset)
    learned = df[df["mode"] == "learned"].copy()
    if learned.empty:
        raise ValueError("No learned rows found in audit DataFrame")
    return learned.groupby(pair_col)["DE_Spearman"].mean()


def stratify_by_tertile(
    pop_mean_perf: pd.Series,
) -> Dict[str, List[str]]:
    """Split conditions into LOW / MID / HIGH tertiles by pop_mean performance.

    LOW = bottom third of pop_mean performance (hardest conditions).
    HIGH = top third (easiest conditions).
    """
    terciles = pd.qcut(pop_mean_perf, q=3, labels=["LOW", "MID", "HIGH"])
    groups: Dict[str, List[str]] = {}
    for label in ("LOW", "MID", "HIGH"):
        groups[label] = list(terciles[terciles == label].index)
    return groups


def cluster_bootstrap_ci(
    gaps: np.ndarray,
    gene_ids: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Cluster-bootstrap CI for the mean gap, resampling by gene.

    Because multiple seeds for the same gene are correlated, we resample
    genes (drawing all seed observations per gene), not individual rows.
    """
    rng = np.random.default_rng(seed)
    unique_genes = np.unique(gene_ids)
    boot_means: List[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_genes, size=len(unique_genes), replace=True)
        boot_gaps = np.concatenate([gaps[gene_ids == g] for g in sampled])
        boot_means.append(float(boot_gaps.mean()))
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def compute_gap_per_tertile(
    df: pd.DataFrame,
    dataset: str,
    n_boot: int = 1000,
) -> pd.DataFrame:
    """Compute learned-minus-pop_mean gap per tertile with bootstrap CIs.

    Returns a DataFrame with columns:
        tertile, n_conditions, mean_gap, ci_lo, ci_hi, mean_pop_mean, mean_learned
    """
    pair_col = _resolve_pair_col(df, dataset)
    pop_mean_perf = compute_pop_mean_performance(df, dataset)
    learned_perf = compute_learned_performance(df, dataset)

    # Align to conditions present in both
    common = pop_mean_perf.index.intersection(learned_perf.index)
    pop_mean_perf = pop_mean_perf.loc[common]
    learned_perf = learned_perf.loc[common]

    tertile_groups = stratify_by_tertile(pop_mean_perf)

    rows = []
    for label in ("LOW", "MID", "HIGH"):
        conditions = tertile_groups[label]
        if not conditions:
            continue

        # Per-seed gaps for cluster bootstrap
        pop_rows = df[(df["mode"] == "pop_mean") & (df[pair_col].isin(conditions))]
        learned_rows = df[(df["mode"] == "learned") & (df[pair_col].isin(conditions))]

        if "seed" in df.columns:
            # Merge on pair_col + seed to get per-observation gap
            merged = learned_rows.merge(
                pop_rows[[pair_col, "seed", "DE_Spearman"]],
                on=[pair_col, "seed"],
                suffixes=("_learned", "_pop_mean"),
            )
            gaps = (
                merged["DE_Spearman_learned"].values
                - merged["DE_Spearman_pop_mean"].values
            )
            gene_ids = merged[pair_col].values
        else:
            # No seed column -- one observation per condition
            gap_series = learned_perf.loc[conditions] - pop_mean_perf.loc[conditions]
            gaps = gap_series.values
            gene_ids = np.array(conditions)

        mean_gap = float(gaps.mean())
        ci_lo, ci_hi = cluster_bootstrap_ci(gaps, gene_ids, n_boot=n_boot)

        rows.append(
            {
                "tertile": label,
                "n_conditions": len(conditions),
                "mean_gap": mean_gap,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "mean_pop_mean": float(pop_mean_perf.loc[conditions].mean()),
                "mean_learned": float(learned_perf.loc[conditions].mean()),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Effect-size distribution
# ---------------------------------------------------------------------------

def compute_effect_sizes(
    adata,
    test_conditions: List[str],
    dataset: str,
    deg_k: int = 200,
) -> pd.Series:
    """Per-condition absolute effect size: mean |delta-log| across top-k DEGs.

    For Norman (doubles), condition = 'geneA+geneB'.
    For Replogle (singles), condition = gene name.
    """
    ctrl_mask = adata.obs["condition"] == "ctrl"
    ctrl_x = adata[ctrl_mask].X
    if hasattr(ctrl_x, "toarray"):
        ctrl_x = ctrl_x.toarray()
    ctrl_mean = ctrl_x.mean(axis=0)

    effect_sizes: Dict[str, float] = {}
    for cond in test_conditions:
        cond_mask = adata.obs["condition"] == cond
        if cond_mask.sum() == 0:
            continue
        pert_x = adata[cond_mask].X
        if hasattr(pert_x, "toarray"):
            pert_x = pert_x.toarray()
        pert_mean = pert_x.mean(axis=0)

        delta = np.abs(pert_mean - ctrl_mean).flatten()
        if len(delta) < deg_k:
            top_k = delta
        else:
            top_idx = np.argsort(delta)[-deg_k:]
            top_k = delta[top_idx]
        effect_sizes[cond] = float(top_k.mean())

    return pd.Series(effect_sizes, name="effect_size")


def compute_effect_size_distribution(
    dataset: str,
    test_conditions: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load dataset, compute effect sizes, return summary DataFrame."""
    print(f"  Loading {dataset} for effect-size computation...")
    if dataset == "norman":
        adata = load_norman()
        split = build_split(adata)
        if test_conditions is None:
            test_conditions = [
                f"{a}+{b}" for a, b in split["test_pairs"]
            ]
    elif dataset == "k562":
        adata = load_replogle()
        split = build_split_replogle(adata)
        if test_conditions is None:
            test_conditions = list(split["test_genes"])
    elif dataset == "rpe1":
        adata = load_replogle_rpe1()
        split = build_split_replogle(adata)
        if test_conditions is None:
            test_conditions = list(split["test_genes"])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    es = compute_effect_sizes(adata, test_conditions, dataset)
    summary = pd.DataFrame(
        {
            "dataset": dataset,
            "condition": es.index,
            "effect_size": es.values,
        }
    )
    return summary


# ---------------------------------------------------------------------------
# Fresh training + audit (for datasets without existing results)
# ---------------------------------------------------------------------------

def _find_train_script() -> Path:
    """Locate the training script."""
    candidates = [
        INTERVENEFM_ROOT / "scripts" / "train_and_audit.py",
        INTERVENEFM_ROOT / "scripts" / "train_audit_replogle.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Cannot find training script in {INTERVENEFM_ROOT / 'scripts'}"
    )


def run_fresh_audit_cpa(
    dataset: str,
    seeds: List[int],
    epochs: int = 15,
    device: str = "cpu",
) -> pd.DataFrame:
    """Train CPA from scratch and return concatenated audit DataFrame.

    Delegates to the existing train_and_audit.py script for consistency.
    """
    all_dfs: List[pd.DataFrame] = []
    script = _find_train_script()

    for seed in seeds:
        out_tag = f"exp5_{dataset}_seed{seed}"
        cmd = [
            sys.executable,
            str(script),
            "--dataset", dataset,
            "--seed", str(seed),
            "--epochs", str(epochs),
            "--out_tag", out_tag,
            "--device", device,
        ]
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [WARN] Training failed for {dataset} seed={seed}:")
            print(result.stderr[-500:] if result.stderr else "(no stderr)")
            continue

        # Look for the output audit CSV
        audit_path = RESULTS_DIR / f"audit_{out_tag}.csv"
        if not audit_path.exists():
            # Try alternate naming
            candidates = list(RESULTS_DIR.glob(f"audit_{out_tag}*.csv"))
            if candidates:
                audit_path = candidates[0]
            else:
                print(f"  [WARN] No audit CSV found for {out_tag}")
                continue

        df = load_existing_audit(audit_path)
        df["seed"] = seed
        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError(f"No successful audits for CPA on {dataset}")
    return pd.concat(all_dfs, ignore_index=True)


def run_fresh_audit_gears(
    dataset: str,
    seeds: List[int],
    epochs: int = 5,
    device: str = "cpu",
) -> pd.DataFrame:
    """Train GEARS and return concatenated audit DataFrame.

    Looks for a GEARS training script in scripts/; falls back to a
    subprocess call to the GEARS package if installed.
    """
    gears_script = INTERVENEFM_ROOT / "scripts" / "train_audit_gears.py"
    if not gears_script.exists():
        raise FileNotFoundError(
            f"GEARS training script not found at {gears_script}. "
            "Please provide existing audit CSVs or add the script."
        )

    all_dfs: List[pd.DataFrame] = []
    for seed in seeds:
        out_tag = f"exp5_gears_{dataset}_seed{seed}"
        cmd = [
            sys.executable,
            str(gears_script),
            "--dataset", dataset,
            "--seed", str(seed),
            "--epochs", str(epochs),
            "--out_tag", out_tag,
            "--device", device,
        ]
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [WARN] GEARS training failed for {dataset} seed={seed}:")
            print(result.stderr[-500:] if result.stderr else "(no stderr)")
            continue

        audit_path = RESULTS_DIR / f"audit_{out_tag}.csv"
        if not audit_path.exists():
            candidates = list(RESULTS_DIR.glob(f"audit_{out_tag}*.csv"))
            if candidates:
                audit_path = candidates[0]
            else:
                print(f"  [WARN] No audit CSV found for {out_tag}")
                continue

        df = load_existing_audit(audit_path)
        df["seed"] = seed
        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError(f"No successful audits for GEARS on {dataset}")
    return pd.concat(all_dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Reanalysis pipeline
# ---------------------------------------------------------------------------

def _discover_audit_csvs(
    model: str, dataset: str
) -> List[Path]:
    """Find existing audit CSVs for a (model, dataset) pair."""
    known = KNOWN_AUDITS.get((model, dataset), [])
    found: List[Path] = []
    for name in known:
        path = RESULTS_DIR / name
        if path.exists():
            found.append(path)
    # Also search by glob for any we might have missed
    patterns = [
        f"audit*{model}*{dataset}*.csv",
        f"audit*{dataset}*{model}*.csv",
    ]
    if dataset == "k562":
        patterns.append("audit*replogle*3seed*.csv")
        patterns.append("audit*replogle*k562*.csv")
    if dataset == "rpe1":
        patterns.append("audit*rpe1*.csv")
    for pat in patterns:
        for p in RESULTS_DIR.glob(pat):
            if p not in found:
                found.append(p)
    return found


def reanalyze_existing(
    model: str,
    dataset: str,
    n_boot: int = 1000,
) -> Optional[pd.DataFrame]:
    """Reanalyze existing audit CSVs for conditional-helper effect.

    Returns tertile-gap DataFrame or None if no data found.
    """
    csv_paths = _discover_audit_csvs(model, dataset)
    if not csv_paths:
        print(f"  No existing audit CSVs for ({model}, {dataset})")
        return None

    print(f"  Found {len(csv_paths)} audit CSV(s) for ({model}, {dataset}):")
    for p in csv_paths:
        print(f"    {p.name}")

    dfs: List[pd.DataFrame] = []
    for p in csv_paths:
        df = load_existing_audit(p)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Check that both modes exist
    modes = set(combined["mode"].unique())
    if "learned" not in modes:
        print(f"  [WARN] No 'learned' mode in data for ({model}, {dataset})")
        return None
    if "pop_mean" not in modes:
        print(f"  [WARN] No 'pop_mean' mode in data for ({model}, {dataset})")
        return None

    tertile_df = compute_gap_per_tertile(combined, dataset, n_boot=n_boot)
    tertile_df.insert(0, "model", model)
    tertile_df.insert(1, "dataset", dataset)
    return tertile_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 5: Conditional-helper effect replication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["norman", "k562", "rpe1", "all"],
        help="Dataset to analyse (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["cpa", "gears", "scgpt", "all"],
        help="Model to analyse (default: all)",
    )
    parser.add_argument(
        "--reanalyze_only",
        action="store_true",
        help="Only reanalyze existing audit CSVs; skip fresh training",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds for fresh training runs (default: 0 1 2)",
    )
    parser.add_argument(
        "--n_boot",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for CIs (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Training epochs for fresh CPA runs (default: 15)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for fresh training (default: cpu)",
    )
    parser.add_argument(
        "--effect_sizes",
        action="store_true",
        help="Also compute per-dataset effect-size distributions",
    )
    return parser.parse_args()


def resolve_datasets(arg: str) -> List[str]:
    if arg == "all":
        return ["norman", "k562", "rpe1"]
    return [arg]


def resolve_models(arg: str) -> List[str]:
    if arg == "all":
        return ["cpa", "gears", "scgpt"]
    return [arg]


def print_tertile_table(df: pd.DataFrame) -> None:
    """Pretty-print the tertile gap table."""
    print()
    print(
        f"  {'Tertile':<8} {'N':>4}  {'Gap':>7}  "
        f"{'95% CI':>15}  {'PopMean':>8}  {'Learned':>8}"
    )
    print("  " + "-" * 62)
    for _, row in df.iterrows():
        ci_str = f"[{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]"
        print(
            f"  {row['tertile']:<8} {row['n_conditions']:>4}  "
            f"{row['mean_gap']:>+.4f}  {ci_str:>15}  "
            f"{row['mean_pop_mean']:>.4f}  {row['mean_learned']:>.4f}"
        )
    print()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = resolve_datasets(args.dataset)
    models = resolve_models(args.model)

    all_tertile_results: List[pd.DataFrame] = []
    all_effect_sizes: List[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # 1. Conditional-helper reanalysis / fresh audit
    # ------------------------------------------------------------------
    for model in models:
        for dataset in datasets:
            print(f"\n{'='*60}")
            print(f"  {model.upper()} x {dataset.upper()}")
            print(f"{'='*60}")

            # Try reanalysis first
            tertile_df = reanalyze_existing(model, dataset, n_boot=args.n_boot)

            if tertile_df is not None:
                print_tertile_table(tertile_df)
                all_tertile_results.append(tertile_df)
                continue

            # No existing data -- train fresh if allowed
            if args.reanalyze_only:
                print(f"  Skipping fresh training (--reanalyze_only)")
                continue

            print(f"  No existing results; launching fresh audit...")
            try:
                if model == "cpa":
                    audit_df = run_fresh_audit_cpa(
                        dataset,
                        args.seeds,
                        epochs=args.epochs,
                        device=args.device,
                    )
                elif model == "gears":
                    audit_df = run_fresh_audit_gears(
                        dataset,
                        args.seeds,
                        epochs=args.epochs,
                        device=args.device,
                    )
                elif model == "scgpt":
                    print("  [SKIP] scGPT fresh training not yet implemented")
                    continue
                else:
                    print(f"  [SKIP] Unknown model: {model}")
                    continue

                # Save the raw audit for reproducibility
                raw_path = OUT_DIR / f"audit_fresh_{model}_{dataset}.csv"
                audit_df.to_csv(raw_path, index=False)
                print(f"  Saved fresh audit to {raw_path.name}")

                # Now compute tertile gaps
                modes = set(audit_df["mode"].unique())
                if "learned" in modes and "pop_mean" in modes:
                    tertile_df = compute_gap_per_tertile(
                        audit_df, dataset, n_boot=args.n_boot
                    )
                    tertile_df.insert(0, "model", model)
                    tertile_df.insert(1, "dataset", dataset)
                    print_tertile_table(tertile_df)
                    all_tertile_results.append(tertile_df)
                else:
                    missing = {"learned", "pop_mean"} - modes
                    print(f"  [WARN] Missing modes {missing} in fresh audit")

            except Exception as e:
                print(f"  [ERROR] Fresh audit failed: {e}")
                continue

    # ------------------------------------------------------------------
    # 2. Save combined tertile results
    # ------------------------------------------------------------------
    if all_tertile_results:
        combined = pd.concat(all_tertile_results, ignore_index=True)
        out_path = OUT_DIR / "conditional_helper_tertile_gaps.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nSaved combined tertile results to {out_path}")

        # Also save a JSON summary for easy downstream consumption
        summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        for _, row in combined.iterrows():
            key = f"{row['model']}_{row['dataset']}"
            if key not in summary:
                summary[key] = {}
            summary[key][row["tertile"]] = {
                "mean_gap": row["mean_gap"],
                "ci_lo": row["ci_lo"],
                "ci_hi": row["ci_hi"],
                "n_conditions": int(row["n_conditions"]),
            }
        json_path = OUT_DIR / "conditional_helper_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved JSON summary to {json_path}")
    else:
        print("\nNo tertile results produced.")

    # ------------------------------------------------------------------
    # 3. Effect-size distributions (optional)
    # ------------------------------------------------------------------
    if args.effect_sizes:
        print(f"\n{'='*60}")
        print("  Effect-size distributions")
        print(f"{'='*60}")

        for dataset in datasets:
            try:
                es_df = compute_effect_size_distribution(dataset)
                all_effect_sizes.append(es_df)

                n = len(es_df)
                med = es_df["effect_size"].median()
                mean = es_df["effect_size"].mean()
                std = es_df["effect_size"].std()
                print(
                    f"  {dataset:>8}: n={n:>4}, "
                    f"mean={mean:.4f}, median={med:.4f}, std={std:.4f}"
                )
            except Exception as e:
                print(f"  [ERROR] Effect sizes for {dataset}: {e}")

        if all_effect_sizes:
            es_combined = pd.concat(all_effect_sizes, ignore_index=True)
            es_path = OUT_DIR / "effect_size_distributions.csv"
            es_combined.to_csv(es_path, index=False)
            print(f"\nSaved effect-size distributions to {es_path}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\nExperiment 5 complete.")


if __name__ == "__main__":
    main()
