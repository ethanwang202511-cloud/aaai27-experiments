"""run_flywire_n500.py -- Modal-free FlyWire N=500 extraction + evaluation.

Stripped from scripts/rigor_flywire_n500.py: all Modal decorators removed,
runs on local CPU/GPU. Performs FlyWire subgraph extraction at N=500,
fits ridge/RF/KNN/power-law degree predictors, evaluates Sinkhorn + reach_3hop
L1 pipeline, and tests whether predictors separate from the noise floor.

AAAI-27 Experiment 1: FlyWire at N >= 500.
CPU only. Output: results/aaai27/flywire_n500.{json,md}
"""
from __future__ import annotations

import json
import os
import sys
import time
import networkx as nx
import numpy as np

try:
    import scipy.linalg as _sla
    _SCIPY = True
except ImportError:
    _SCIPY = False

# ---------------------------------------------------------------------------
# Inlined helpers (standalone -- no Cell2Circuit src/ dependency)
# ---------------------------------------------------------------------------

def er_graph(N: int, density: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = (rng.random((N, N)) < density).astype(np.float32)
    np.fill_diagonal(a, 0)
    return a


def sinkhorn_degree_match(
    probs: np.ndarray, out_deg: np.ndarray, in_deg: np.ndarray,
    num_iters: int = 50, eps: float = 1e-8,
) -> np.ndarray:
    P = probs.copy().astype(np.float64)
    np.fill_diagonal(P, 0)
    for _ in range(num_iters):
        row_sums = P.sum(axis=1, keepdims=True).clip(eps)
        P = P * (out_deg[:, None] / row_sums)
        P = np.clip(P, 0, 1)
        col_sums = P.sum(axis=0, keepdims=True).clip(eps)
        P = P * (in_deg[None, :] / col_sums)
        P = np.clip(P, 0, 1)
    return P.astype(np.float32)


def configuration_model_sample(adj: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    adj = (adj > 0.5).astype(np.int32)
    N = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int).tolist()
    out_deg = adj.sum(axis=1).astype(int).tolist()
    if sum(in_deg) == 0 or sum(out_deg) == 0:
        return np.zeros((N, N), dtype=np.float32)
    if sum(in_deg) != sum(out_deg):
        diff = sum(out_deg) - sum(in_deg)
        if diff > 0:
            order = np.argsort(out_deg)[::-1]
            i = 0
            while diff > 0 and i < len(order):
                if out_deg[order[i]] > 0:
                    out_deg[order[i]] -= 1
                    diff -= 1
                i = (i + 1) % len(order)
        else:
            diff = -diff
            order = np.argsort(in_deg)[::-1]
            i = 0
            while diff > 0 and i < len(order):
                if in_deg[order[i]] > 0:
                    in_deg[order[i]] -= 1
                    diff -= 1
                i = (i + 1) % len(order)
    seed = int(rng.integers(0, 1 << 30))
    try:
        G = nx.directed_configuration_model(
            in_degree_sequence=in_deg,
            out_degree_sequence=out_deg,
            create_using=nx.MultiDiGraph,
            seed=seed,
        )
        G = nx.DiGraph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        out = np.zeros((N, N), dtype=np.float32)
        for u, v in G.edges():
            out[u, v] = 1.0
        return out
    except Exception:
        d = float(adj.mean())
        return er_graph(N, d, seed=seed)


def _adj_to_digraph(adj: np.ndarray) -> "nx.DiGraph":
    G = nx.DiGraph()
    N = adj.shape[0]
    G.add_nodes_from(range(N))
    rows, cols = np.where(adj > 0)
    for r, c in zip(rows.tolist(), cols.tolist()):
        if r != c:
            G.add_edge(int(r), int(c))
    return G


def reciprocity(adj: np.ndarray) -> float:
    A = np.asarray(adj, dtype=np.float32)
    np.fill_diagonal(A, 0.0)
    total = float(A.sum())
    if total == 0.0:
        return 0.0
    recip = float((A * A.T).sum())
    return recip / total


def spectral_eigenvalues(adj: np.ndarray, num_eigs: int = 20) -> np.ndarray:
    A = np.asarray(adj, dtype=np.float64)
    np.fill_diagonal(A, 0.0)
    N = A.shape[0]

    out_deg = A.sum(axis=1)
    L = np.diag(out_deg) - A

    k = min(num_eigs, N)

    if _SCIPY:
        eigvals = _sla.eigvals(L)
    else:
        eigvals = np.linalg.eigvals(L)

    order = np.argsort(-np.abs(eigvals))
    top = eigvals[order[:k]]
    return np.sort(top.real)[::-1].astype(np.float64)


def motif_3node_spectrum(adj: np.ndarray) -> np.ndarray:
    G = _adj_to_digraph(adj)
    try:
        triad_dict = nx.triads_by_type(G)
    except Exception:
        return np.zeros(16, dtype=np.float64)

    _TRIAD_TYPES = sorted([
        "003", "012", "021C", "021D", "021U",
        "030C", "030T", "102", "111D", "111U",
        "120C", "120D", "120U", "201", "210", "300",
    ])

    counts = np.zeros(16, dtype=np.float64)
    for i, t in enumerate(_TRIAD_TYPES):
        counts[i] = float(len(triad_dict.get(t, [])))
    return counts

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SUBSET_SIZE = 500
N_SUBSETS = 80          # 50 train / 20 test (+ margin)
N_TRAIN = 50
N_TEST = 20
SEED = 0

# Scales to sweep for Experiment 5 cross-validation
SCALE_SWEEP = [200, 300, 500]


# ---------------------------------------------------------------------------
# Feature engineering (standalone, no import dependency)
# ---------------------------------------------------------------------------

def features_full(item: dict, n_morph_classes: int = 15) -> np.ndarray:
    """Build feature matrix from item dict. Shape: (N, n_morph_classes + 4)."""
    N = item["adjacency"].shape[0]
    morph_idx = item["morph_idx"].astype(int)
    morph_oh = np.zeros((N, n_morph_classes), dtype=np.float32)
    for i in range(N):
        if 0 <= morph_idx[i] < n_morph_classes:
            morph_oh[i, morph_idx[i]] = 1.0
    pos = item.get("positions", np.zeros((N, 3), dtype=np.float32))
    depth = item.get("depth", np.zeros(N, dtype=np.float32))
    return np.hstack([morph_oh, pos, depth[:, None]]).astype(np.float32)


def _reconcile_totals(o: np.ndarray, i: np.ndarray):
    """Ensure sum(out_degrees) == sum(in_degrees) by trimming the larger."""
    o = o.copy(); i = i.copy()
    diff = int(o.sum() - i.sum())
    if diff > 0:
        order = np.argsort(-o)
        for idx in order:
            if diff <= 0:
                break
            trim = min(diff, int(o[idx]))
            o[idx] -= trim
            diff -= trim
    elif diff < 0:
        diff = -diff
        order = np.argsort(-i)
        for idx in order:
            if diff <= 0:
                break
            trim = min(diff, int(i[idx]))
            i[idx] -= trim
            diff -= trim
    return o, i


# ---------------------------------------------------------------------------
# Degree predictors
# ---------------------------------------------------------------------------

def fit_predictor(kind: str, train_items: list, feat_fn, n_morph: int = 15):
    """Fit a degree predictor (ridge, rf, or knn) on training items."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor

    X_l, yo_l, yi_l = [], [], []
    for it in train_items:
        X_l.append(feat_fn(it, n_morph))
        a = it["adjacency"]
        yo_l.append(a.sum(axis=1).astype(float))
        yi_l.append(a.sum(axis=0).astype(float))
    X = np.concatenate(X_l)
    yo = np.concatenate(yo_l)
    yi = np.concatenate(yi_l)
    fm = X.mean(0); fs = X.std(0).clip(1e-4)
    Xn = (X - fm) / fs

    if kind == "ridge":
        m_o = Ridge(alpha=1.0).fit(Xn, yo)
        m_i = Ridge(alpha=1.0).fit(Xn, yi)
    elif kind == "rf":
        m_o = RandomForestRegressor(
            n_estimators=500, max_depth=15, n_jobs=-1, random_state=42
        ).fit(Xn, yo)
        m_i = RandomForestRegressor(
            n_estimators=500, max_depth=15, n_jobs=-1, random_state=42
        ).fit(Xn, yi)
    elif kind == "knn":
        m_o = KNeighborsRegressor(n_neighbors=10).fit(Xn, yo)
        m_i = KNeighborsRegressor(n_neighbors=10).fit(Xn, yi)
    else:
        raise ValueError(f"Unknown predictor: {kind}")
    return m_o, m_i, fm, fs


def predict_degrees(m_o, m_i, fm, fs, item, feat_fn, n_morph=15):
    """Predict in/out degrees for a single item."""
    X = feat_fn(item, n_morph)
    Xn = (X - fm) / fs
    o = np.maximum(0, np.round(m_o.predict(Xn))).astype(int)
    i = np.maximum(0, np.round(m_i.predict(Xn))).astype(int)
    return _reconcile_totals(o, i)


def pearson_deg(predict_fn, test_items, rng, n_morph=15):
    """Compute Pearson r between predicted and GT degrees."""
    gt_out, pr_out, gt_in, pr_in = [], [], [], []
    for it in test_items:
        adj = it["adjacency"]
        o_pred, i_pred = predict_fn(it, rng)
        gt_out.extend(adj.sum(1).tolist())
        gt_in.extend(adj.sum(0).tolist())
        pr_out.extend(o_pred.tolist())
        pr_in.extend(i_pred.tolist())
    from scipy.stats import pearsonr
    r_out, _ = pearsonr(gt_out, pr_out) if len(gt_out) > 2 else (0.0, 1.0)
    r_in, _ = pearsonr(gt_in, pr_in) if len(gt_in) > 2 else (0.0, 1.0)
    return {"pearson_out_mean": float(r_out), "pearson_in_mean": float(r_in)}


# ---------------------------------------------------------------------------
# Power-law degree model
# ---------------------------------------------------------------------------

def fit_powerlaw_alpha(degrees: np.ndarray, xmin: int = 1) -> float:
    """MLE power-law exponent (discrete, Clauset et al. 2009)."""
    d = degrees[degrees >= xmin].astype(float)
    if len(d) < 5:
        return 2.0
    alpha = 1.0 + len(d) / np.sum(np.log(d / (xmin - 0.5)))
    return float(alpha)


def powerlaw_predict(alpha: float, N: int, total_edges: int,
                     rng: np.random.Generator):
    """Generate degree sequences from fitted power law."""
    xmin = 1
    degs = np.zeros(N, dtype=int)
    for k in range(N):
        u = rng.random()
        degs[k] = int(xmin * (1 - u) ** (-1.0 / (alpha - 1)))
    # Scale to match total edges
    if degs.sum() > 0:
        degs = np.round(degs * total_edges / degs.sum()).astype(int)
    return _reconcile_totals(degs.copy(), degs.copy())


# ---------------------------------------------------------------------------
# 3-hop reachability L1
# ---------------------------------------------------------------------------

def reach_3hop(adj: np.ndarray) -> float:
    """Fraction of node pairs reachable within 3 hops."""
    A = (adj > 0.5).astype(np.float32)
    np.fill_diagonal(A, 0)
    N = A.shape[0]
    R = A.copy()
    for _ in range(2):
        R = ((R @ A) > 0).astype(np.float32)
        R = np.clip(R + A, 0, 1)
    np.fill_diagonal(R, 0)
    return float(R.sum() / max(N * (N - 1), 1))


def reach_3hop_l1(adj_gt: np.ndarray, adj_sample: np.ndarray) -> float:
    return abs(reach_3hop(adj_gt) - reach_3hop(adj_sample))


# ---------------------------------------------------------------------------
# Motif L1
# ---------------------------------------------------------------------------

def motif_l1(adj_gt: np.ndarray, adj_sample: np.ndarray) -> float:
    """L1 distance between normalized 13-triad motif spectra."""
    m_gt = motif_3node_spectrum(adj_gt)
    m_s = motif_3node_spectrum(adj_sample)
    # Normalize
    s_gt = m_gt.sum()
    s_s = m_s.sum()
    if s_gt > 0:
        m_gt = m_gt / s_gt
    if s_s > 0:
        m_s = m_s / s_s
    return float(np.abs(m_gt - m_s).sum())


# ---------------------------------------------------------------------------
# FlyWire data loading
# ---------------------------------------------------------------------------

def load_flywire_n(subset_size: int, num_subsets: int, seed: int) -> list:
    """Load FlyWire subsets at the specified size.

    Requires the FlyWire data to be available locally via the Cell2Circuit
    data pipeline. Falls back to synthetic data if unavailable.
    """
    try:
        from data.flywire_loader import load_flywire, build_random_subsets_flywire
        ds = load_flywire()
        items = build_random_subsets_flywire(
            ds, num_subsets=num_subsets,
            subset_size=subset_size, seed=seed, pool="test"
        )
        out = []
        for it in items:
            out.append({
                "adjacency": it["adjacency"].numpy().astype(np.float32),
                "morph_idx": it["morph_idx"].numpy().astype(np.int64),
                "positions": it["positions"].numpy().astype(np.float32),
                "depth": it["depth"].numpy().astype(np.float32),
            })
        for o in out:
            np.fill_diagonal(o["adjacency"], 0)
        return out
    except ImportError:
        print("[WARN] FlyWire loader not available. Generating synthetic "
              "sparse directed graphs for pipeline validation.", flush=True)
        return _generate_synthetic_sparse(subset_size, num_subsets, seed)


def _generate_synthetic_sparse(N: int, num_subsets: int, seed: int) -> list:
    """Generate synthetic sparse directed graphs mimicking FlyWire statistics.

    Density ~0.001 (matching FlyWire whole-brain density).
    """
    rng = np.random.default_rng(seed)
    density = 0.001
    items = []
    for _ in range(num_subsets):
        adj = (rng.random((N, N)) < density).astype(np.float32)
        np.fill_diagonal(adj, 0)
        items.append({
            "adjacency": adj,
            "morph_idx": rng.integers(0, 15, size=N).astype(np.int64),
            "positions": rng.normal(0, 100, size=(N, 3)).astype(np.float32),
            "depth": rng.uniform(0, 800, size=N).astype(np.float32),
        })
    return items


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

def eval_on_test(predict_fn, test_items, rng, n_morph=15):
    """Evaluate a predictor via Sinkhorn -> sample -> reach_3hop L1 + motif L1."""
    reach_l1s = []
    motif_l1s = []
    recip_errs = []
    for it in test_items:
        adj_gt = it["adjacency"]
        N = adj_gt.shape[0]
        rho = float(adj_gt.sum() / max(N * (N - 1), 1))
        P_uniform = np.full((N, N), rho, dtype=np.float32)
        np.fill_diagonal(P_uniform, 0)

        o_pred, i_pred = predict_fn(it, rng)
        P_sink = sinkhorn_degree_match(P_uniform.copy(), o_pred, i_pred,
                                       num_iters=30)
        adj_s = (rng.random(P_sink.shape) < P_sink).astype(np.float32)
        np.fill_diagonal(adj_s, 0)

        reach_l1s.append(reach_3hop_l1(adj_gt, adj_s))
        motif_l1s.append(motif_l1(adj_gt, adj_s))
        recip_errs.append(abs(reciprocity(adj_gt) - reciprocity(adj_s)))

    reach_l1s = np.array(reach_l1s)
    motif_l1s = np.array(motif_l1s)
    recip_errs = np.array(recip_errs)
    return {
        "reach_3hop_L1": {
            "mean": float(reach_l1s.mean()),
            "std": float(reach_l1s.std()),
            "sem": float(reach_l1s.std() / np.sqrt(max(len(reach_l1s), 1))),
        },
        "motif_L1": {
            "mean": float(motif_l1s.mean()),
            "std": float(motif_l1s.std()),
            "sem": float(motif_l1s.std() / np.sqrt(max(len(motif_l1s), 1))),
        },
        "reciprocity_error": {
            "mean": float(recip_errs.mean()),
            "sem": float(recip_errs.std() / np.sqrt(max(len(recip_errs), 1))),
        },
    }


# ---------------------------------------------------------------------------
# Configuration model baseline
# ---------------------------------------------------------------------------

def eval_config_model(test_items, rng, n_mc=3):
    """Evaluate the directed configuration model on test items."""
    reach_l1s, motif_l1s, recip_errs = [], [], []
    for it in test_items:
        adj_gt = it["adjacency"]
        for _ in range(n_mc):
            adj_s = configuration_model_sample(adj_gt, rng)
            reach_l1s.append(reach_3hop_l1(adj_gt, adj_s))
            motif_l1s.append(motif_l1(adj_gt, adj_s))
            recip_errs.append(abs(reciprocity(adj_gt) - reciprocity(adj_s)))
    reach_l1s = np.array(reach_l1s)
    return {
        "reach_3hop_L1": {
            "mean": float(reach_l1s.mean()),
            "sem": float(reach_l1s.std() / np.sqrt(max(len(reach_l1s), 1))),
        },
        "motif_L1": {
            "mean": float(np.mean(motif_l1s)),
            "sem": float(np.std(motif_l1s) / np.sqrt(max(len(motif_l1s), 1))),
        },
        "reciprocity_error": {
            "mean": float(np.mean(recip_errs)),
            "sem": float(np.std(recip_errs) / np.sqrt(max(len(recip_errs), 1))),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print(f"Loading FlyWire subsets N={SUBSET_SIZE} ...", flush=True)
    items = load_flywire_n(SUBSET_SIZE, N_SUBSETS, SEED)
    train = items[:N_TRAIN]
    test = items[N_TRAIN:N_TRAIN + N_TEST]
    n_morph = max(int(it["morph_idx"].max()) + 1 for it in items) if items else 15

    mean_edges = float(np.mean([int(it["adjacency"].sum()) for it in items]))
    mean_deg = mean_edges / SUBSET_SIZE
    print(f"  {len(items)} subsets, n_morph={n_morph}, "
          f"mean_edges/subset={mean_edges:.1f} (N=100 was ~9.9), "
          f"mean_degree={mean_deg:.3f}", flush=True)

    # Power-law reference
    train_degs = np.concatenate(
        [it["adjacency"].sum(1).astype(int) for it in train]
        + [it["adjacency"].sum(0).astype(int) for it in train]
    )
    alpha = fit_powerlaw_alpha(train_degs, xmin=1)

    out = {
        "experiment": "flywire_n500_aaai27",
        "addresses": "AAAI-27 Exp 1: FlyWire at N>=500",
        "subset_size": SUBSET_SIZE,
        "n_train": len(train), "n_test": len(test),
        "n_morph": n_morph,
        "mean_edges_per_subset": mean_edges,
        "mean_degree": mean_deg,
        "by_predictor": {},
    }

    # Fit and evaluate degree predictors
    for kind in ["ridge", "rf", "knn"]:
        print(f"  Fitting {kind} predictor ...", flush=True)
        m_o, m_i, fm, fs = fit_predictor(kind, train, features_full, n_morph)
        pred_fn = lambda it, rng, _mo=m_o, _mi=m_i, _fm=fm, _fs=fs: (
            predict_degrees(_mo, _mi, _fm, _fs, it, features_full, n_morph)
        )
        diag = pearson_deg(pred_fn, test, rng, n_morph)
        metr = eval_on_test(pred_fn, test, rng, n_morph)
        out["by_predictor"][kind] = {**diag, **metr}
        print(f"    {kind:<5} pearson_out={diag['pearson_out_mean']:+.3f} "
              f"reach_L1={metr['reach_3hop_L1']['mean']:.4f}"
              f"+/-{metr['reach_3hop_L1']['sem']:.4f}", flush=True)

    # Power-law predictor
    def pl_pred(it, rng):
        N = it["adjacency"].shape[0]
        return powerlaw_predict(alpha, N, int(it["adjacency"].sum()), rng)
    diag_pl = pearson_deg(pl_pred, test, rng, n_morph)
    metr_pl = eval_on_test(pl_pred, test, rng, n_morph)
    out["by_predictor"]["powerlaw"] = {"alpha": alpha, **diag_pl, **metr_pl}
    print(f"    pl(a={alpha:.2f}) "
          f"reach_L1={metr_pl['reach_3hop_L1']['mean']:.4f}"
          f"+/-{metr_pl['reach_3hop_L1']['sem']:.4f}", flush=True)

    # Configuration model (GT degrees)
    print("  Evaluating config model (GT degrees) ...", flush=True)
    config_metr = eval_config_model(test, rng, n_mc=3)
    out["by_predictor"]["config_model_gt"] = config_metr
    print(f"    config_model reach_L1="
          f"{config_metr['reach_3hop_L1']['mean']:.4f}"
          f"+/-{config_metr['reach_3hop_L1']['sem']:.4f}", flush=True)

    # Separation analysis
    reaches = {
        k: v.get("reach_3hop_L1", v.get("reach_3hop_L1", {})).get("mean", 0.0)
        for k, v in out["by_predictor"].items()
    }
    spread = max(reaches.values()) - min(reaches.values()) if reaches else 0.0
    pears = {k: abs(v.get("pearson_out_mean", 0.0))
             for k, v in out["by_predictor"].items()}
    out["reach_L1_spread_across_predictors"] = float(spread)
    out["informative"] = bool(
        spread > 0.01 or max(pears.values(), default=0) > 0.2
    )
    out["interpretation"] = (
        f"At N={SUBSET_SIZE} mean edges/subset={mean_edges:.0f} "
        f"(vs ~9.9 at N=100). "
        f"Predictor reach-L1 spread={spread:.4f}; "
        f"max |degree-Pearson|="
        f"{max(pears.values(), default=0):.3f}. "
        + (
            "Predictors now SEPARATE from the noise floor -> FlyWire becomes "
            "informative at N>=500, supporting the paper's recommendation."
            if out["informative"]
            else "Predictors still collapse to the noise floor even at N=500 "
            "-> FlyWire sparsity is intrinsic, not a subset-size artifact."
        )
    )
    out["walltime_s"] = float(time.time() - t0)

    # Save results
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "aaai27")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "flywire_n500.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    # Human-readable summary
    md_lines = [
        f"# FlyWire predictor comparison at N={SUBSET_SIZE} (AAAI-27 Exp 1)",
        "",
        f"{len(items)} subsets, mean edges/subset = {mean_edges:.1f} "
        f"(N=100 was ~9.9), power-law alpha={alpha:.2f}.",
        "",
        "| predictor | degree Pearson (out) | reach_3hop L1 | motif L1 |",
        "|---|---|---|---|",
    ]
    for k, v in out["by_predictor"].items():
        r_l1 = v.get("reach_3hop_L1", {})
        m_l1 = v.get("motif_L1", {})
        md_lines.append(
            f"| {k} | {v.get('pearson_out_mean', 0.0):+.3f} | "
            f"{r_l1.get('mean', 0.0):.4f}+/-{r_l1.get('sem', 0.0):.4f} | "
            f"{m_l1.get('mean', 0.0):.4f}+/-{m_l1.get('sem', 0.0):.4f} |"
        )
    md_lines += ["", out["interpretation"]]
    md_path = os.path.join(out_dir, "flywire_n500.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n{out['interpretation']}")
    print(f"Saved: {json_path}")
    print(f"walltime {out['walltime_s']:.1f}s", flush=True)
    return out


if __name__ == "__main__":
    main()
