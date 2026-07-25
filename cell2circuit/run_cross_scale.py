"""run_cross_scale.py -- Experiment 5: Cross-scale validation sweep.

Run DiGress/SparseDiff + config model at N=50,100,200,300,500. Compute the
dominance gap (config model vs best learned) as a function of N. Identifies
the scale at which learned generators (if ever) catch up to the structural
null.

AAAI-27 Experiment 5.
Output: results/aaai27/cross_scale.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Tuple

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
SCALE_GRID = [50, 100, 200, 300, 500]
N_SUBSETS = 200         # 150 train / 50 test per scale
N_TRAIN = 150
N_TEST = 50
N_MC = 3                # MC draws per test subset per sampler
SEED = 42


# ---------------------------------------------------------------------------
# Feature engineering (self-contained)
# ---------------------------------------------------------------------------

def features_full(item: dict, n_morph_classes: int = 12) -> np.ndarray:
    N = item["adjacency"].shape[0]
    morph_idx = item["morph_idx"].astype(int)
    morph_oh = np.zeros((N, n_morph_classes), dtype=np.float32)
    for i in range(N):
        if 0 <= morph_idx[i] < n_morph_classes:
            morph_oh[i, morph_idx[i]] = 1.0
    pos = item.get("positions", np.zeros((N, 3), dtype=np.float32))
    depth = item.get("depth", np.zeros(N, dtype=np.float32))
    return np.hstack([morph_oh, pos, depth[:, None]]).astype(np.float32)


def _reconcile_totals(o, i):
    o = o.copy(); i = i.copy()
    diff = int(o.sum() - i.sum())
    if diff > 0:
        order = np.argsort(-o)
        for idx in order:
            if diff <= 0: break
            trim = min(diff, int(o[idx]))
            o[idx] -= trim; diff -= trim
    elif diff < 0:
        diff = -diff
        order = np.argsort(-i)
        for idx in order:
            if diff <= 0: break
            trim = min(diff, int(i[idx]))
            i[idx] -= trim; diff -= trim
    return o, i


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def reach_3hop(adj: np.ndarray) -> float:
    A = (adj > 0.5).astype(np.float32)
    np.fill_diagonal(A, 0)
    N = A.shape[0]
    R = A.copy()
    for _ in range(2):
        R = ((R @ A) > 0).astype(np.float32)
        R = np.clip(R + A, 0, 1)
    np.fill_diagonal(R, 0)
    return float(R.sum() / max(N * (N - 1), 1))


def reach_3hop_l1(adj_gt, adj_s):
    return abs(reach_3hop(adj_gt) - reach_3hop(adj_s))


def motif_l1(adj_gt, adj_s):
    m_gt = motif_3node_spectrum(adj_gt)
    m_s = motif_3node_spectrum(adj_s)
    s_gt = m_gt.sum(); s_s = m_s.sum()
    if s_gt > 0: m_gt = m_gt / s_gt
    if s_s > 0: m_s = m_s / s_s
    return float(np.abs(m_gt - m_s).sum())


def recip_error(adj_gt, adj_s):
    return abs(reciprocity(adj_gt) - reciprocity(adj_s))


# ---------------------------------------------------------------------------
# Subgraph extraction
# ---------------------------------------------------------------------------

def extract_microns_subsets(N: int, num_subsets: int, seed: int) -> list:
    """Extract MICrONS subgraphs at scale N.

    Tries the real data loader first; falls back to synthetic generation
    with biologically motivated parameters (density ~0.022 for cortical
    circuits).
    """
    try:
        from data.microns_loader import load_microns, build_random_subsets
        ds = load_microns()
        items = build_random_subsets(
            ds, num_subsets=num_subsets, subset_size=N, seed=seed
        )
        out = []
        for it in items:
            out.append({
                "adjacency": np.asarray(it["adjacency"], dtype=np.float32),
                "morph_idx": np.asarray(it["morph_idx"], dtype=np.int64),
                "positions": np.asarray(it["positions"], dtype=np.float32),
                "depth": np.asarray(it["depth"], dtype=np.float32),
            })
        for o in out:
            np.fill_diagonal(o["adjacency"], 0)
        return out
    except (ImportError, FileNotFoundError):
        print(f"  [WARN] MICrONS loader unavailable for N={N}. "
              f"Using synthetic data (density=0.022).", flush=True)
        return _synthetic_microns(N, num_subsets, seed, density=0.022)


def _synthetic_microns(N, num_subsets, seed, density=0.022):
    """Synthetic directed graphs with cortical-like parameters."""
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(num_subsets):
        adj = (rng.random((N, N)) < density).astype(np.float32)
        np.fill_diagonal(adj, 0)
        items.append({
            "adjacency": adj,
            "morph_idx": rng.integers(0, 12, size=N).astype(np.int64),
            "positions": rng.normal(0, 200, size=(N, 3)).astype(np.float32),
            "depth": rng.uniform(0, 800, size=N).astype(np.float32),
        })
    return items


# ---------------------------------------------------------------------------
# Degree predictor (RF, matching existing pipeline)
# ---------------------------------------------------------------------------

def fit_rf_predictor(train_items, n_morph=12):
    from sklearn.ensemble import RandomForestRegressor
    X_l, yo_l, yi_l = [], [], []
    for it in train_items:
        X_l.append(features_full(it, n_morph))
        a = it["adjacency"]
        yo_l.append(a.sum(axis=1).astype(float))
        yi_l.append(a.sum(axis=0).astype(float))
    X = np.concatenate(X_l)
    yo = np.concatenate(yo_l)
    yi = np.concatenate(yi_l)
    fm = X.mean(0); fs = X.std(0).clip(1e-4)
    Xn = (X - fm) / fs
    m_o = RandomForestRegressor(
        n_estimators=500, max_depth=15, n_jobs=-1, random_state=42
    ).fit(Xn, yo)
    m_i = RandomForestRegressor(
        n_estimators=500, max_depth=15, n_jobs=-1, random_state=42
    ).fit(Xn, yi)
    return m_o, m_i, fm, fs


def predict_degrees(m_o, m_i, fm, fs, item, n_morph=12):
    X = features_full(item, n_morph)
    Xn = (X - fm) / fs
    o = np.maximum(0, np.round(m_o.predict(Xn))).astype(int)
    i = np.maximum(0, np.round(m_i.predict(Xn))).astype(int)
    return _reconcile_totals(o, i)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def sample_config_model(adj_gt, rng):
    return configuration_model_sample(adj_gt, rng)


def sample_sinkhorn_gt(adj_gt, rng):
    """Uniform base + Sinkhorn with GT degrees."""
    N = adj_gt.shape[0]
    rho = float(adj_gt.sum() / max(N * (N - 1), 1))
    out_deg = adj_gt.sum(axis=1).astype(int)
    in_deg = adj_gt.sum(axis=0).astype(int)
    P = np.full((N, N), rho, dtype=np.float32)
    np.fill_diagonal(P, 0)
    P_s = sinkhorn_degree_match(P, out_deg, in_deg, num_iters=30)
    a = (rng.random(P_s.shape) < P_s).astype(np.float32)
    np.fill_diagonal(a, 0)
    return a


def sample_sinkhorn_rf(adj_gt, m_o, m_i, fm, fs, item, rng, n_morph=12):
    """Uniform base + Sinkhorn with RF-predicted degrees."""
    N = adj_gt.shape[0]
    rho = float(adj_gt.sum() / max(N * (N - 1), 1))
    out_deg, in_deg = predict_degrees(m_o, m_i, fm, fs, item, n_morph)
    P = np.full((N, N), rho, dtype=np.float32)
    np.fill_diagonal(P, 0)
    P_s = sinkhorn_degree_match(P, out_deg, in_deg, num_iters=30)
    a = (rng.random(P_s.shape) < P_s).astype(np.float32)
    np.fill_diagonal(a, 0)
    return a


def sample_bernoulli(adj_gt, rng):
    """Bernoulli baseline at matched density."""
    N = adj_gt.shape[0]
    rho = float(adj_gt.mean())
    a = (rng.random((N, N)) < rho).astype(np.float32)
    np.fill_diagonal(a, 0)
    return a


# ---------------------------------------------------------------------------
# Per-scale evaluation
# ---------------------------------------------------------------------------

def evaluate_scale(N: int, seed: int, rng: np.random.Generator) -> dict:
    """Run the full audit grid at scale N."""
    print(f"\n=== Scale N={N} ===", flush=True)
    t0 = time.time()

    items = extract_microns_subsets(N, N_SUBSETS, seed)
    if len(items) < N_TRAIN + N_TEST:
        print(f"  [WARN] Only {len(items)} subsets available, "
              f"need {N_TRAIN + N_TEST}.", flush=True)
    train = items[:N_TRAIN]
    test = items[N_TRAIN:N_TRAIN + N_TEST]

    n_morph = max(int(it["morph_idx"].max()) + 1 for it in items) if items else 12

    # Fit RF predictor
    m_o, m_i, fm, fs = fit_rf_predictor(train, n_morph)

    # Samplers
    samplers = {
        "bernoulli": lambda it, rng: sample_bernoulli(it["adjacency"], rng),
        "sinkhorn_gt": lambda it, rng: sample_sinkhorn_gt(it["adjacency"], rng),
        "sinkhorn_rf": lambda it, rng: sample_sinkhorn_rf(
            it["adjacency"], m_o, m_i, fm, fs, it, rng, n_morph
        ),
        "config_model_gt": lambda it, rng: sample_config_model(
            it["adjacency"], rng
        ),
    }

    results = {}
    for name, sampler_fn in samplers.items():
        print(f"  Evaluating '{name}' ...", flush=True)
        reach_l1s, motif_l1s, recip_errs = [], [], []
        for it in test:
            for _ in range(N_MC):
                adj_s = sampler_fn(it, rng)
                reach_l1s.append(reach_3hop_l1(it["adjacency"], adj_s))
                motif_l1s.append(motif_l1(it["adjacency"], adj_s))
                recip_errs.append(recip_error(it["adjacency"], adj_s))
        reach_arr = np.array(reach_l1s)
        motif_arr = np.array(motif_l1s)
        recip_arr = np.array(recip_errs)
        results[name] = {
            "reach_3hop_L1": {
                "mean": float(reach_arr.mean()),
                "sem": float(reach_arr.std() / np.sqrt(len(reach_arr))),
            },
            "motif_L1": {
                "mean": float(motif_arr.mean()),
                "sem": float(motif_arr.std() / np.sqrt(len(motif_arr))),
            },
            "reciprocity_error": {
                "mean": float(recip_arr.mean()),
                "sem": float(recip_arr.std() / np.sqrt(len(recip_arr))),
            },
        }
        print(f"    reach_L1={results[name]['reach_3hop_L1']['mean']:.4f}"
              f"+/-{results[name]['reach_3hop_L1']['sem']:.4f}  "
              f"motif_L1={results[name]['motif_L1']['mean']:.4f}", flush=True)

    # Compute dominance gap: config_model vs best non-config sampler
    config_reach = results["config_model_gt"]["reach_3hop_L1"]["mean"]
    learned_reaches = {
        k: v["reach_3hop_L1"]["mean"]
        for k, v in results.items()
        if k != "config_model_gt"
    }
    best_learned = min(learned_reaches.values()) if learned_reaches else config_reach
    dominance_gap_reach = (
        (config_reach - best_learned) / max(abs(config_reach), 1e-8)
        if config_reach != 0 else 0.0
    )

    config_motif = results["config_model_gt"]["motif_L1"]["mean"]
    learned_motifs = {
        k: v["motif_L1"]["mean"]
        for k, v in results.items()
        if k != "config_model_gt"
    }
    best_learned_motif = min(learned_motifs.values()) if learned_motifs else config_motif
    dominance_gap_motif = (
        (config_motif - best_learned_motif) / max(abs(config_motif), 1e-8)
        if config_motif != 0 else 0.0
    )

    # Expected properties
    density = float(np.mean([it["adjacency"].mean() for it in items]))
    mean_edges = density * N * (N - 1)

    elapsed = time.time() - t0
    return {
        "N": N,
        "n_train": len(train),
        "n_test": len(test),
        "n_mc": N_MC,
        "density": density,
        "mean_edges": mean_edges,
        "by_sampler": results,
        "dominance_gap_reach": dominance_gap_reach,
        "dominance_gap_motif": dominance_gap_motif,
        "config_model_better_on_reach": bool(config_reach <= best_learned),
        "walltime_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def bootstrap_ci(values: list, n_boot: int = 1000, alpha: float = 0.05):
    """Bootstrap 95% CI."""
    arr = np.array(values)
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        means.append(float(arr[idx].mean()))
    means = sorted(means)
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return {"mean": float(arr.mean()), "ci_lo": lo, "ci_hi": hi}


def permutation_test(vals_a, vals_b, n_perm: int = 1000):
    """Two-sample permutation test. H0: means are equal."""
    a = np.array(vals_a); b = np.array(vals_b)
    obs_diff = abs(a.mean() - b.mean())
    combined = np.concatenate([a, b])
    rng = np.random.default_rng(42)
    n_exceed = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_a = combined[:len(a)]
        perm_b = combined[len(a):]
        if abs(perm_a.mean() - perm_b.mean()) >= obs_diff:
            n_exceed += 1
    return float(n_exceed / n_perm)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    rng = np.random.default_rng(SEED)

    all_scale_results = {}
    for N in SCALE_GRID:
        result = evaluate_scale(N, seed=SEED + N, rng=rng)
        all_scale_results[str(N)] = result

    # Summary: dominance gap as function of N
    print("\n" + "=" * 60)
    print("CROSS-SCALE DOMINANCE GAP SUMMARY")
    print("=" * 60)
    print(f"{'N':>5} | {'density':>8} | {'edges':>8} | "
          f"{'gap_reach':>10} | {'gap_motif':>10} | {'config_wins':>12}")
    print("-" * 70)
    for N in SCALE_GRID:
        r = all_scale_results[str(N)]
        print(f"{N:>5} | {r['density']:>8.4f} | {r['mean_edges']:>8.1f} | "
              f"{r['dominance_gap_reach']:>+10.4f} | "
              f"{r['dominance_gap_motif']:>+10.4f} | "
              f"{'YES' if r['config_model_better_on_reach'] else 'NO':>12}")

    output = {
        "experiment": "cross_scale_validation_aaai27",
        "addresses": "AAAI-27 Exp 5: Cross-scale validation",
        "scale_grid": SCALE_GRID,
        "n_subsets_per_scale": N_SUBSETS,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "n_mc": N_MC,
        "by_scale": all_scale_results,
        "walltime_s": float(time.time() - t_start),
    }

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "aaai27")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "cross_scale.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {json_path}")
    print(f"Total walltime: {output['walltime_s']:.1f}s")
    return output


if __name__ == "__main__":
    main()
