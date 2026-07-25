"""Experiment 5: Scaling-law sweep for discrimination probes.

Runs consistent-corruption and verdict probes across Qwen2.5 model sizes,
fits power-law / sigmoid / linear-in-log scaling curves, and extrapolates
thresholds.  No Modal dependency -- uses vLLM locally + scipy + numpy.

Usage:
    python run_scaling_sweep.py \
        --problems-jsonl /path/to/problems.jsonl \
        --output-dir ./exp5_scaling
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

# Sibling module -- lives in the same directory.
from discrimination_standalone import (
    create_llm,
    estimate_cost,
    load_problems,
    run_consistent_corruption_probe,
    run_verdict_probe,
)

# ---------------------------------------------------------------------------
# Model roster
# ---------------------------------------------------------------------------

SCALING_MODELS: list[tuple[str, float]] = [
    ("Qwen/Qwen2.5-3B", 3e9),
    ("Qwen/Qwen2.5-3B-Instruct", 3e9),
    ("Qwen/Qwen2.5-7B", 7e9),            # already have data
    ("Qwen/Qwen2.5-7B-Instruct", 7e9),   # already have data
    ("Qwen/Qwen2.5-14B", 14e9),
    ("Qwen/Qwen2.5-14B-Instruct", 14e9),
    ("Qwen/Qwen2.5-32B", 32e9),
    ("Qwen/Qwen2.5-32B-Instruct", 32e9),
    ("Qwen/Qwen2.5-72B", 72e9),          # already have data
]

# Pre-existing measurements (from earlier experiments).
EXISTING_DATA: dict[str, dict[str, Any]] = {
    "Qwen/Qwen2.5-1.5B": {
        "params": 1.5e9,
        "sigma_disc": 2.58,
        "delta_verdict": 0.53,
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "params": 1.5e9,
        "sigma_disc": None,
        "delta_verdict": None,
    },
    "Qwen/Qwen2.5-7B": {
        "params": 7e9,
        "sigma_disc": 2.50,
        "delta_verdict": 1.39,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "params": 7e9,
        "sigma_disc": None,
        "delta_verdict": 7.34,
    },
    "Qwen/Qwen2.5-72B": {
        "params": 72e9,
        "sigma_disc": 2.38,
        "delta_verdict": 1.55,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_slug(model_id: str) -> str:
    """Turn 'Qwen/Qwen2.5-7B-Instruct' into 'Qwen2.5-7B-Instruct'."""
    return model_id.split("/")[-1]


def _output_exists(output_dir: Path, model_id: str) -> bool:
    """Check whether probe output files already exist for *model_id*."""
    slug = _model_slug(model_id)
    cc = output_dir / "probes" / f"{slug}_consistent_corruption.jsonl"
    vd = output_dir / "probes" / f"{slug}_verdict.jsonl"
    return cc.exists() and vd.exists()


def _load_existing_probes(output_dir: Path, model_id: str) -> dict[str, Any] | None:
    """Read previously saved probe JSONL files and extract summary metrics."""
    slug = _model_slug(model_id)
    cc_path = output_dir / "probes" / f"{slug}_consistent_corruption.jsonl"
    vd_path = output_dir / "probes" / f"{slug}_verdict.jsonl"

    result: dict[str, Any] = {}
    if cc_path.exists():
        rows = _read_jsonl(cc_path)
        signals = [r["discrimination_signal"] for r in rows
                   if "discrimination_signal" in r and r["discrimination_signal"] is not None
                   and not (isinstance(r["discrimination_signal"], float)
                            and np.isnan(r["discrimination_signal"]))]
        result["sigma_disc"] = float(np.mean(signals)) if signals else None
    else:
        result["sigma_disc"] = None

    if vd_path.exists():
        rows = _read_jsonl(vd_path)
        deltas = [r["delta_verdict"] for r in rows
                  if "delta_verdict" in r and r["delta_verdict"] is not None
                  and not (isinstance(r["delta_verdict"], float)
                           and np.isnan(r["delta_verdict"]))]
        result["delta_verdict"] = float(np.mean(deltas)) if deltas else None
    else:
        result["delta_verdict"] = None

    if result["sigma_disc"] is None and result["delta_verdict"] is None:
        return None
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    print(f"  -> saved {path}")


# ---------------------------------------------------------------------------
# Curve-fitting primitives
# ---------------------------------------------------------------------------


def _power_law(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """y = a * x^b"""
    return a * np.power(x, b)


def _sigmoid(log_x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
    """y = L / (1 + exp(-k * (log_x - x0)))"""
    return L / (1.0 + np.exp(-k * (log_x - x0)))


def _linear_in_log(log_x: np.ndarray, a: float, b: float) -> np.ndarray:
    """y = a * log(x) + b  (x already passed as log_x)"""
    return a * log_x + b


def compute_aic(n: int, k: int, rss: float) -> float:
    """Akaike Information Criterion."""
    if rss <= 0 or n <= 0:
        return float("inf")
    return n * np.log(rss / n) + 2 * k


def compute_bic(n: int, k: int, rss: float) -> float:
    """Bayesian Information Criterion."""
    if rss <= 0 or n <= 0:
        return float("inf")
    return n * np.log(rss / n) + k * np.log(n)


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    fit_func,
    p0=None,
    bounds=(-np.inf, np.inf),
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap confidence intervals on curve_fit parameters."""
    rng = np.random.RandomState(seed)
    params_list: list[np.ndarray] = []
    for _ in range(n_boot):
        idx = rng.choice(len(x), size=len(x), replace=True)
        try:
            popt, _ = curve_fit(fit_func, x[idx], y[idx], p0=p0, bounds=bounds,
                                maxfev=10000)
            params_list.append(popt)
        except Exception:
            continue
    if len(params_list) < 10:
        nan_arr = np.full_like(params_list[0] if params_list else np.array([np.nan]),
                               np.nan)
        return nan_arr, nan_arr
    params_arr = np.array(params_list)
    alpha = (1 - ci) / 2
    lower = np.percentile(params_arr, alpha * 100, axis=0)
    upper = np.percentile(params_arr, (1 - alpha) * 100, axis=0)
    return lower, upper


def _fit_all_curves(
    params_arr: np.ndarray,
    y_arr: np.ndarray,
    metric_name: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit power-law, sigmoid, and linear-in-log to (params, metric) data.

    Returns a dict with fit results keyed by curve type.
    """
    n = len(params_arr)
    log_params = np.log(params_arr)
    results: dict[str, Any] = {}

    # --- Power law: y = a * N^b (fit in log-log) ---
    try:
        log_y = np.log(np.clip(y_arr, 1e-12, None))
        popt_pl, _ = curve_fit(
            lambda log_x, log_a, b: log_a + b * log_x,
            log_params, log_y, p0=[0.0, 0.1], maxfev=10000,
        )
        y_pred_pl = np.exp(popt_pl[0] + popt_pl[1] * log_params)
        rss_pl = float(np.sum((y_arr - y_pred_pl) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        r2_pl = 1 - rss_pl / ss_tot if ss_tot > 0 else float("nan")
        ci_lo, ci_hi = bootstrap_ci(
            log_params, log_y,
            lambda log_x, log_a, b: log_a + b * log_x,
            p0=[0.0, 0.1], n_boot=1000, seed=seed,
        )
        results["power_law"] = {
            "params": {"a": float(np.exp(popt_pl[0])), "b": float(popt_pl[1])},
            "R2": r2_pl,
            "AIC": compute_aic(n, 2, rss_pl),
            "BIC": compute_bic(n, 2, rss_pl),
            "bootstrap_ci_95": {
                "log_a": [float(ci_lo[0]), float(ci_hi[0])],
                "b": [float(ci_lo[1]), float(ci_hi[1])],
            },
        }
    except Exception as e:
        results["power_law"] = {"error": str(e)}

    # --- Sigmoid: y = L / (1 + exp(-k*(log(N) - x0))) ---
    try:
        p0_sig = [float(np.max(y_arr) * 1.5), 1.0, float(np.median(log_params))]
        bounds_sig = ([0, -10, 0], [100, 100, 60])
        popt_sig, _ = curve_fit(_sigmoid, log_params, y_arr, p0=p0_sig,
                                bounds=bounds_sig, maxfev=10000)
        y_pred_sig = _sigmoid(log_params, *popt_sig)
        rss_sig = float(np.sum((y_arr - y_pred_sig) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        r2_sig = 1 - rss_sig / ss_tot if ss_tot > 0 else float("nan")
        ci_lo, ci_hi = bootstrap_ci(
            log_params, y_arr, _sigmoid,
            p0=p0_sig, bounds=bounds_sig, n_boot=1000, seed=seed,
        )
        results["sigmoid"] = {
            "params": {"L": float(popt_sig[0]), "k": float(popt_sig[1]),
                       "x0": float(popt_sig[2])},
            "R2": r2_sig,
            "AIC": compute_aic(n, 3, rss_sig),
            "BIC": compute_bic(n, 3, rss_sig),
            "bootstrap_ci_95": {
                "L": [float(ci_lo[0]), float(ci_hi[0])],
                "k": [float(ci_lo[1]), float(ci_hi[1])],
                "x0": [float(ci_lo[2]), float(ci_hi[2])],
            },
        }
    except Exception as e:
        results["sigmoid"] = {"error": str(e)}

    # --- Linear-in-log: y = a * log(N) + b ---
    try:
        popt_lin, _ = curve_fit(_linear_in_log, log_params, y_arr,
                                p0=[0.1, 0.0], maxfev=10000)
        y_pred_lin = _linear_in_log(log_params, *popt_lin)
        rss_lin = float(np.sum((y_arr - y_pred_lin) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        r2_lin = 1 - rss_lin / ss_tot if ss_tot > 0 else float("nan")
        ci_lo, ci_hi = bootstrap_ci(
            log_params, y_arr, _linear_in_log,
            p0=[0.1, 0.0], n_boot=1000, seed=seed,
        )
        results["linear_in_log"] = {
            "params": {"a": float(popt_lin[0]), "b": float(popt_lin[1])},
            "R2": r2_lin,
            "AIC": compute_aic(n, 2, rss_lin),
            "BIC": compute_bic(n, 2, rss_lin),
            "bootstrap_ci_95": {
                "a": [float(ci_lo[0]), float(ci_hi[0])],
                "b": [float(ci_lo[1]), float(ci_hi[1])],
            },
        }
    except Exception as e:
        results["linear_in_log"] = {"error": str(e)}

    # --- Model selection ---
    valid = {k: v for k, v in results.items() if "AIC" in v}
    if valid:
        best_aic = min(valid, key=lambda k: valid[k]["AIC"])
        best_bic = min(valid, key=lambda k: valid[k]["BIC"])
        results["_best_AIC"] = best_aic
        results["_best_BIC"] = best_bic
    results["_metric"] = metric_name
    results["_n_points"] = n

    return results


# ---------------------------------------------------------------------------
# Phase 5d: Extrapolation
# ---------------------------------------------------------------------------


def _extrapolate_thresholds(
    curve_fits: dict[str, Any],
    metric: str,
) -> dict[str, Any]:
    """Using the best-fit curve for *metric*, predict parameter-count thresholds.

    Thresholds:
    - delta_verdict > 0.2  (behavioral premise threshold)
    - verdict accuracy > 95%  (mapped to delta_verdict ~ 3.0 as proxy)
    """
    best_key = curve_fits.get("_best_AIC")
    if best_key is None or best_key not in curve_fits:
        return {"error": "no valid curve fit"}

    fit = curve_fits[best_key]
    params = fit["params"]
    predictions: dict[str, Any] = {"best_curve": best_key, "metric": metric}

    # Define thresholds based on metric
    if metric == "delta_verdict":
        thresholds = {
            "delta_verdict_gt_0.2": 0.2,
            "verdict_accuracy_gt_95pct": 3.0,  # proxy: delta ~ 3.0
        }
    elif metric == "sigma_disc":
        thresholds = {
            "sigma_disc_gt_2.0": 2.0,
            "sigma_disc_gt_3.0": 3.0,
        }
    else:
        thresholds = {}

    for label, threshold in thresholds.items():
        try:
            if best_key == "power_law":
                # y = a * N^b  =>  N = (y/a)^(1/b)
                a, b = params["a"], params["b"]
                if a > 0 and b != 0:
                    n_threshold = (threshold / a) ** (1.0 / b)
                else:
                    n_threshold = None
            elif best_key == "linear_in_log":
                # y = a * log(N) + b  =>  log(N) = (y - b) / a  =>  N = exp(...)
                a, b = params["a"], params["b"]
                if a != 0:
                    n_threshold = float(np.exp((threshold - b) / a))
                else:
                    n_threshold = None
            elif best_key == "sigmoid":
                # y = L / (1 + exp(-k*(log(N)-x0)))
                # Invert: log(N) = x0 - (1/k)*log(L/y - 1)
                L, k, x0 = params["L"], params["k"], params["x0"]
                ratio = L / threshold - 1
                if ratio > 0 and k != 0:
                    n_threshold = float(np.exp(x0 - (1.0 / k) * np.log(ratio)))
                else:
                    n_threshold = None
            else:
                n_threshold = None

            predictions[label] = {
                "threshold": threshold,
                "predicted_params": f"{n_threshold:.2e}" if n_threshold else None,
                "predicted_params_raw": n_threshold,
            }
        except Exception as e:
            predictions[label] = {"error": str(e)}

    return predictions


# ---------------------------------------------------------------------------
# Phase 5c: Generation-discrimination gap (optional)
# ---------------------------------------------------------------------------


def _run_generation_gap(
    model_id: str,
    problems: list[dict[str, Any]],
    seed: int,
    tensor_parallel: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate pass@32 solutions for generation-discrimination gap analysis."""
    from vllm import LLM, SamplingParams  # type: ignore[import-untyped]

    slug = _model_slug(model_id)
    gen_path = output_dir / "probes" / f"{slug}_generation.jsonl"

    if gen_path.exists():
        print(f"  [gen] output exists, loading: {gen_path}")
        rows = _read_jsonl(gen_path)
        n_correct = sum(1 for r in rows if r.get("any_correct", False))
        return {
            "model": model_id,
            "pass_at_32": n_correct / len(rows) if rows else 0.0,
            "n_problems": len(rows),
        }

    print(f"  [gen] loading model for generation: {model_id}")
    llm = LLM(
        model=model_id,
        seed=seed,
        tensor_parallel_size=tensor_parallel,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )

    sampling = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1024,
        n=32,
        seed=seed,
    )

    prompts = []
    for prob in problems:
        prompt = (
            f"Solve the following math problem step by step.\n\n"
            f"Problem: {prob['problem']}\n\n"
            f"Solution:"
        )
        prompts.append(prompt)

    print(f"  [gen] generating 32 samples for {len(prompts)} problems...")
    outputs = llm.generate(prompts, sampling)

    gen_results: list[dict[str, Any]] = []
    for i, (prob, out) in enumerate(zip(problems, outputs)):
        answer = str(prob.get("answer", ""))
        any_correct = False
        for completion in out.outputs:
            text = completion.text
            # Check if answer appears in boxed or at end
            if answer and (f"\\boxed{{{answer}}}" in text or text.strip().endswith(answer)):
                any_correct = True
                break
        gen_results.append({
            "idx": i,
            "answer": answer,
            "any_correct": any_correct,
            "n_samples": len(out.outputs),
        })

    gen_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gen_path, "w", encoding="utf-8") as fh:
        for row in gen_results:
            fh.write(json.dumps(row) + "\n")
    print(f"  [gen] saved {gen_path}")

    n_correct = sum(1 for r in gen_results if r["any_correct"])
    pass_at_32 = n_correct / len(gen_results) if gen_results else 0.0

    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "model": model_id,
        "pass_at_32": pass_at_32,
        "n_problems": len(gen_results),
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_probes(
    problems: list[dict[str, Any]],
    models: list[tuple[str, float]],
    output_dir: Path,
    seed: int,
    tensor_parallel: int,
    skip_existing: bool,
) -> dict[str, dict[str, Any]]:
    """Phase 5a: run discrimination probes on each model.

    Returns mapping from model_id -> {params, sigma_disc, delta_verdict}.
    """
    probe_dir = output_dir / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)

    collected: dict[str, dict[str, Any]] = {}

    for model_id, n_params in models:
        slug = _model_slug(model_id)
        print(f"\n{'='*60}")
        print(f"Model: {model_id}  ({n_params:.0e} params)")
        print(f"{'='*60}")

        if skip_existing and _output_exists(output_dir, model_id):
            print(f"  [skip] output already exists for {slug}")
            existing = _load_existing_probes(output_dir, model_id)
            if existing is not None:
                existing["params"] = n_params
                collected[model_id] = existing
            continue

        cc_path = probe_dir / f"{slug}_consistent_corruption.jsonl"
        vd_path = probe_dir / f"{slug}_verdict.jsonl"

        t0 = time.time()
        print(f"  Loading model into vLLM...")
        llm = create_llm(
            model_id=model_id,
            seed=seed,
            tensor_parallel_size=tensor_parallel,
        )
        tokenizer = llm.get_tokenizer()
        print(f"  Model loaded in {time.time() - t0:.1f}s")

        # Consistent corruption probe
        print(f"  Running consistent_corruption probe...")
        t1 = time.time()
        cc_result = run_consistent_corruption_probe(
            llm, tokenizer, problems, seed=seed, save_to=str(cc_path),
        )
        sigma_disc = cc_result.get("mean_discrimination_signal")
        print(f"  sigma_disc = {sigma_disc}")
        print(f"  Probe took {time.time() - t1:.1f}s")

        # Verdict probe
        print(f"  Running verdict probe...")
        t2 = time.time()
        vd_result = run_verdict_probe(
            llm, tokenizer, problems, seed=seed, save_to=str(vd_path),
        )
        delta_verdict = vd_result.get("mean_delta_verdict")
        print(f"  delta_verdict = {delta_verdict}")
        print(f"  Probe took {time.time() - t2:.1f}s")

        collected[model_id] = {
            "params": n_params,
            "sigma_disc": sigma_disc,
            "delta_verdict": delta_verdict,
        }

        # Free GPU memory
        print(f"  Unloading model...")
        del llm, tokenizer
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        elapsed = time.time() - t0
        print(f"  Total time for {slug}: {elapsed:.1f}s")

    return collected


def collect_all_data(
    probe_data: dict[str, dict[str, Any]],
    existing_data_dir: Path | None,
) -> dict[str, dict[str, Any]]:
    """Merge probe_data with EXISTING_DATA and any data from existing_data_dir."""
    all_data: dict[str, dict[str, Any]] = {}

    # Start with hardcoded existing data
    for model_id, vals in EXISTING_DATA.items():
        all_data[model_id] = dict(vals)

    # Override/add from existing_data_dir if provided
    if existing_data_dir is not None:
        data_dir = Path(existing_data_dir)
        if data_dir.exists():
            for f in data_dir.glob("*.json"):
                try:
                    with open(f) as fh:
                        extra = json.load(fh)
                    if isinstance(extra, dict):
                        for mid, vals in extra.items():
                            if isinstance(vals, dict) and "params" in vals:
                                all_data[mid] = vals
                except Exception:
                    pass

    # Override with fresh probe data
    for model_id, vals in probe_data.items():
        if model_id in all_data:
            # Merge: keep existing non-None values, override with new non-None
            for k, v in vals.items():
                if v is not None:
                    all_data[model_id][k] = v
        else:
            all_data[model_id] = vals

    return all_data


def fit_scaling_curves(
    all_data: dict[str, dict[str, Any]],
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Phase 5b: fit scaling curves separately for base and Instruct models."""

    def _is_instruct(model_id: str) -> bool:
        return "Instruct" in model_id

    results: dict[str, dict[str, Any]] = {}

    for metric in ["sigma_disc", "delta_verdict"]:
        for model_type in ["base", "instruct"]:
            label = f"{metric}_{model_type}"

            # Filter data points
            points_n: list[float] = []
            points_y: list[float] = []
            for model_id, vals in all_data.items():
                is_inst = _is_instruct(model_id)
                if model_type == "instruct" and not is_inst:
                    continue
                if model_type == "base" and is_inst:
                    continue
                val = vals.get(metric)
                if val is not None:
                    points_n.append(vals["params"])
                    points_y.append(val)

            if len(points_n) < 3:
                results[label] = {
                    "error": f"only {len(points_n)} data points, need >= 3",
                    "n_points": len(points_n),
                    "data": list(zip(points_n, points_y)),
                }
                continue

            params_arr = np.array(points_n)
            y_arr = np.array(points_y)

            print(f"\n--- Fitting {label} ({len(points_n)} points) ---")
            for p, y in zip(points_n, points_y):
                print(f"    N={p:.2e}  {metric}={y:.4f}")

            fit_result = _fit_all_curves(params_arr, y_arr, label, seed=seed)
            results[label] = fit_result

            # Print summary
            for curve_type in ["power_law", "sigmoid", "linear_in_log"]:
                if curve_type in fit_result and "R2" in fit_result[curve_type]:
                    r2 = fit_result[curve_type]["R2"]
                    aic = fit_result[curve_type]["AIC"]
                    bic = fit_result[curve_type]["BIC"]
                    print(f"    {curve_type:15s}: R2={r2:.4f}  AIC={aic:.2f}  BIC={bic:.2f}")

            if "_best_AIC" in fit_result:
                print(f"    Best by AIC: {fit_result['_best_AIC']}")
            if "_best_BIC" in fit_result:
                print(f"    Best by BIC: {fit_result['_best_BIC']}")

    return results


def run_extrapolation(
    curve_fits: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Phase 5d: extrapolate thresholds from best-fit curves."""
    extrapolations: dict[str, Any] = {}

    for label, fits in curve_fits.items():
        if label.startswith("_") or "error" in fits:
            continue
        metric = label.split("_")[0] + "_" + label.split("_")[1]
        # e.g. "sigma_disc" or "delta_verdict"
        extrap = _extrapolate_thresholds(fits, metric)
        extrapolations[label] = extrap

    return extrapolations


def build_summary(
    all_data: dict[str, dict[str, Any]],
    curve_fits: dict[str, dict[str, Any]],
    extrapolations: dict[str, Any],
) -> dict[str, Any]:
    """Build a full summary table."""
    rows: list[dict[str, Any]] = []
    for model_id in sorted(all_data.keys()):
        vals = all_data[model_id]
        rows.append({
            "model": model_id,
            "params": vals["params"],
            "sigma_disc": vals.get("sigma_disc"),
            "delta_verdict": vals.get("delta_verdict"),
            "is_instruct": "Instruct" in model_id,
        })

    return {
        "data_points": rows,
        "curve_fits": {k: v for k, v in curve_fits.items() if not k.startswith("_")},
        "extrapolations": extrapolations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 5: Scaling-law sweep for discrimination probes",
    )
    parser.add_argument(
        "--problems-jsonl", type=str, required=True,
        help="Path to JSONL file with problems (need 'problem', 'solution', 'answer').",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory for all output files.",
    )
    parser.add_argument(
        "--seed", type=int, default=1337,
        help="Random seed (default: 1337).",
    )
    parser.add_argument(
        "--tensor-parallel", type=int, default=4,
        help="Number of GPUs for tensor parallelism (default: 4).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip models whose output files already exist.",
    )
    parser.add_argument(
        "--existing-data-dir", type=str, default=None,
        help="Directory with pre-existing probe results (JSON files).",
    )
    parser.add_argument(
        "--run-generation", action="store_true",
        help="Also run pass@32 generation for gap analysis.",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated subset of model IDs; if None, run all SCALING_MODELS.",
    )
    parser.add_argument(
        "--fit-only", action="store_true",
        help="Skip probe runs; just fit curves from existing data.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to run
    if args.models is not None:
        selected = set(m.strip() for m in args.models.split(","))
        models = [(mid, np) for mid, np in SCALING_MODELS if mid in selected]
        if not models:
            # Try matching by slug
            models = [(mid, np) for mid, np in SCALING_MODELS
                      if _model_slug(mid) in selected]
        if not models:
            print(f"ERROR: no matching models found for: {args.models}")
            print(f"Available: {[mid for mid, _ in SCALING_MODELS]}")
            sys.exit(1)
    else:
        models = SCALING_MODELS

    print(f"Experiment 5: Scaling-law sweep")
    print(f"  Problems:       {args.problems_jsonl}")
    print(f"  Output:         {output_dir}")
    print(f"  Models:         {len(models)}")
    print(f"  Seed:           {args.seed}")
    print(f"  Tensor parallel: {args.tensor_parallel}")
    print(f"  Skip existing:  {args.skip_existing}")
    print(f"  Fit only:       {args.fit_only}")
    print(f"  Run generation: {args.run_generation}")

    # ---- Phase 5a: Run probes ----
    probe_data: dict[str, dict[str, Any]] = {}
    if not args.fit_only:
        problems = load_problems(args.problems_jsonl)
        print(f"\nLoaded {len(problems)} problems from {args.problems_jsonl}")
        probe_data = run_probes(
            problems=problems,
            models=models,
            output_dir=output_dir,
            seed=args.seed,
            tensor_parallel=args.tensor_parallel,
            skip_existing=args.skip_existing,
        )
    else:
        # In fit-only mode, load probe data from existing output files
        for model_id, n_params in models:
            existing = _load_existing_probes(output_dir, model_id)
            if existing is not None:
                existing["params"] = n_params
                probe_data[model_id] = existing

    # ---- Collect all data points ----
    existing_dir = Path(args.existing_data_dir) if args.existing_data_dir else None
    all_data = collect_all_data(probe_data, existing_dir)

    scaling_data_path = output_dir / "scaling_data.json"
    _save_json(all_data, scaling_data_path)

    print(f"\n{'='*60}")
    print(f"Collected scaling data: {len(all_data)} models")
    print(f"{'='*60}")
    for mid in sorted(all_data.keys()):
        v = all_data[mid]
        sd = v.get("sigma_disc")
        dv = v.get("delta_verdict")
        sd_str = f"{sd:.4f}" if sd is not None else "N/A"
        dv_str = f"{dv:.4f}" if dv is not None else "N/A"
        print(f"  {mid:40s}  N={v['params']:.0e}  sigma={sd_str}  delta={dv_str}")

    # ---- Phase 5b: Fit scaling curves ----
    print(f"\n{'='*60}")
    print(f"Phase 5b: Fitting scaling curves")
    print(f"{'='*60}")
    curve_fits = fit_scaling_curves(all_data, seed=args.seed)
    _save_json(curve_fits, output_dir / "curve_fits.json")

    # ---- Phase 5c: Generation-discrimination gap (optional) ----
    if args.run_generation and not args.fit_only:
        print(f"\n{'='*60}")
        print(f"Phase 5c: Generation-discrimination gap")
        print(f"{'='*60}")
        problems = load_problems(args.problems_jsonl)
        gen_results: list[dict[str, Any]] = []
        for model_id, n_params in models:
            print(f"\n  Generating for {model_id}...")
            gen_result = _run_generation_gap(
                model_id=model_id,
                problems=problems,
                seed=args.seed,
                tensor_parallel=args.tensor_parallel,
                output_dir=output_dir,
            )
            gen_result["params"] = n_params
            gen_results.append(gen_result)
            print(f"  pass@32 = {gen_result['pass_at_32']:.4f}")
        _save_json(gen_results, output_dir / "generation_gap.json")

    # ---- Phase 5d: Extrapolation ----
    print(f"\n{'='*60}")
    print(f"Phase 5d: Extrapolation")
    print(f"{'='*60}")
    extrapolations = run_extrapolation(curve_fits)
    _save_json(extrapolations, output_dir / "extrapolation.json")

    for label, extrap in extrapolations.items():
        print(f"\n  {label}:")
        if "error" in extrap:
            print(f"    {extrap['error']}")
            continue
        print(f"    Best curve: {extrap.get('best_curve', 'N/A')}")
        for key, val in extrap.items():
            if key in ("best_curve", "metric"):
                continue
            if isinstance(val, dict):
                pp = val.get("predicted_params", "N/A")
                thr = val.get("threshold", "?")
                print(f"    {key}: threshold={thr} -> predicted N={pp}")

    # ---- Full summary ----
    summary = build_summary(all_data, curve_fits, extrapolations)
    _save_json(summary, output_dir / "scaling_summary.json")

    print(f"\n{'='*60}")
    print(f"Done. Outputs in {output_dir}/")
    print(f"  scaling_data.json     - collected data points")
    print(f"  curve_fits.json       - fit parameters, R2, AIC, BIC")
    print(f"  extrapolation.json    - predicted thresholds")
    print(f"  scaling_summary.json  - full summary")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
