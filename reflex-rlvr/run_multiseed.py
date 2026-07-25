#!/usr/bin/env python3
"""
Experiment 1 (AAAI-27): Multi-seed consolidation of discrimination probes.

Runs consistent_corruption + verdict + per_token_surprise probes across
11 models x 4 seeds, then consolidates results with dispersion statistics
and ICC analysis.

Usage:
    python run_multiseed.py \
        --problems-jsonl data/hk_51_problems.jsonl \
        --output-dir results/multiseed
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from discrimination_standalone import (
    load_problems,
    create_llm,
    run_consistent_corruption_probe,
    run_verdict_probe,
    run_per_token_surprise_probe,
    estimate_cost,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_MODELS = [
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "meta-llama/Llama-3.1-8B",
    "mistralai/Mistral-7B-v0.3",
    "google/gemma-2-9b",
]

SEEDS = [1337, 37, 31415, 271828]

PROBE_FUNCTIONS = {
    "consistent_corruption": run_consistent_corruption_probe,
    "verdict": run_verdict_probe,
    "per_token_surprise": run_per_token_surprise_probe,
}

# ---------------------------------------------------------------------------
# Model-specific vLLM kwargs
# ---------------------------------------------------------------------------


def get_model_kwargs(model_name: str, tensor_parallel: int) -> dict:
    """Return vLLM engine kwargs tailored to known-quirky models."""
    kwargs = {
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel,
    }

    if "gemma-2-9b" in model_name:
        kwargs["gpu_memory_utilization"] = 0.85
        kwargs["max_model_len"] = 2048
    elif "Qwen2.5-72B" in model_name:
        kwargs["gpu_memory_utilization"] = 0.85
        # May need max_model_len=1536 if OOM; start without it and fall back.

    return kwargs


# ---------------------------------------------------------------------------
# ICC computation
# ---------------------------------------------------------------------------


def compute_icc(data_matrix: np.ndarray) -> float:
    """ICC(3,1) -- two-way mixed, single measures, consistency.

    Parameters
    ----------
    data_matrix : ndarray of shape (n_items, n_raters)
        e.g. (n_problems, n_seeds).

    Returns
    -------
    float
        ICC value.  Returns NaN when the denominator is zero.
    """
    n, k = data_matrix.shape
    mean_items = data_matrix.mean(axis=1)
    mean_raters = data_matrix.mean(axis=0)
    grand_mean = data_matrix.mean()

    ss_between = k * np.sum((mean_items - grand_mean) ** 2)
    ss_within = np.sum((data_matrix - mean_items[:, None]) ** 2)
    ss_raters = n * np.sum((mean_raters - grand_mean) ** 2)
    ss_error = ss_within - ss_raters

    ms_between = ss_between / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denom = ms_between + (k - 1) * ms_error
    if denom == 0:
        return float("nan")
    return float((ms_between - ms_error) / denom)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def model_dir_name(model_name: str) -> str:
    """Filesystem-safe directory name derived from the HF model id."""
    return model_name.replace("/", "__")


def run_output_path(output_dir: str, model_name: str, probe: str, seed: int) -> Path:
    return (
        Path(output_dir)
        / "runs"
        / model_dir_name(model_name)
        / f"{probe}_seed{seed}.jsonl"
    )


# ---------------------------------------------------------------------------
# Single (model, probe, seed) run
# ---------------------------------------------------------------------------


def run_single(
    model_name: str,
    probe: str,
    seed: int,
    problems: list,
    llm,
    tokenizer,
    output_dir: str,
) -> dict:
    """Run one probe for one model at one seed.  Returns a summary dict."""
    out_path = run_output_path(output_dir, model_name, probe, seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    probe_fn = PROBE_FUNCTIONS[probe]
    result_dict = probe_fn(llm, tokenizer, problems, seed=seed)
    per_problem = result_dict.get("per_problem", [])

    # Write per-problem JSONL
    with open(out_path, "w") as f:
        for r in per_problem:
            f.write(json.dumps(r, default=str) + "\n")

    # Quick aggregate — use the probe's primary signal field
    _SCORE_FIELDS = {
        "consistent_corruption": "discrimination_signal",
        "verdict": "delta_verdict",
        "per_token_surprise": "local_minus_global",
    }
    score_key = _SCORE_FIELDS.get(probe, "score")
    scores = [r.get(score_key) for r in per_problem
              if r.get(score_key) is not None and not (
                  isinstance(r.get(score_key), float) and math.isnan(r[score_key]))]
    mean_score = float(np.mean(scores)) if scores else None

    return {
        "model": model_name,
        "probe": probe,
        "seed": seed,
        "n_problems": len(per_problem),
        "mean_score": mean_score,
        "gate_passed": result_dict.get("gate_passed"),
        "output_path": str(out_path),
    }


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


def run_all(
    problems: list,
    output_dir: str,
    models: list[str],
    seeds: list[int],
    probes: list[str],
    tensor_parallel: int,
    skip_existing: bool,
) -> list[dict]:
    """Iterate over models x seeds x probes, loading each model once."""
    summaries: list[dict] = []
    total_combos = len(models) * len(seeds) * len(probes)
    done = 0

    for model_idx, model_name in enumerate(models, 1):
        print(
            f"\n{'='*72}\n"
            f"[{model_idx}/{len(models)}] Loading model: {model_name}\n"
            f"{'='*72}"
        )

        # Collect which (probe, seed) pairs we actually need
        needed: list[tuple[str, int]] = []
        for probe in probes:
            for seed in seeds:
                out_path = run_output_path(output_dir, model_name, probe, seed)
                if skip_existing and out_path.exists():
                    print(f"  [skip] {probe} seed={seed} -- output exists")
                    done += 1
                    summaries.append(
                        {
                            "model": model_name,
                            "probe": probe,
                            "seed": seed,
                            "skipped": True,
                            "output_path": str(out_path),
                        }
                    )
                else:
                    needed.append((probe, seed))

        if not needed:
            print("  All runs for this model already exist. Skipping model load.")
            continue

        # Load model
        kwargs = get_model_kwargs(model_name, tensor_parallel)
        t0 = time.time()
        try:
            llm, _tokenizer = create_llm(model_name, **kwargs)
        except Exception:
            # For 72B: retry with reduced max_model_len if OOM
            if "72B" in model_name:
                print(
                    "  [warn] Model load failed; retrying with max_model_len=1536 ..."
                )
                kwargs["max_model_len"] = 1536
                llm, _tokenizer = create_llm(model_name, **kwargs)
            else:
                raise
        load_time = time.time() - t0
        print(f"  Model loaded in {load_time:.1f}s")

        # Run probes
        for probe, seed in needed:
            done += 1
            tag = f"[{done}/{total_combos}]"
            print(f"\n  {tag} {model_name} | {probe} | seed={seed}")
            t1 = time.time()
            try:
                summary = run_single(
                    model_name, probe, seed, problems, llm, _tokenizer, output_dir
                )
                elapsed = time.time() - t1
                summary["elapsed_s"] = round(elapsed, 1)
                summaries.append(summary)
                score_str = (
                    f"{summary['mean_score']:.4f}"
                    if summary["mean_score"] is not None
                    else "N/A"
                )
                print(
                    f"      mean_score={score_str}  "
                    f"n={summary['n_problems']}  "
                    f"time={elapsed:.1f}s"
                )
            except Exception as exc:
                elapsed = time.time() - t1
                print(f"      [ERROR] {exc}  (after {elapsed:.1f}s)")
                traceback.print_exc()
                summaries.append(
                    {
                        "model": model_name,
                        "probe": probe,
                        "seed": seed,
                        "error": str(exc),
                    }
                )

        # Unload model to free GPU memory
        del llm, _tokenizer
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"  Model unloaded: {model_name}")

    return summaries


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


def load_run_scores(output_dir: str, model_name: str, probe: str, seed: int):
    """Load per-problem scores from a JSONL run file.  Returns list of floats."""
    path = run_output_path(output_dir, model_name, probe, seed)
    if not path.exists():
        return None
    scores = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            s = rec.get("score", rec.get("accuracy", None))
            if s is not None:
                scores.append(float(s))
    return scores


def consolidate(
    output_dir: str,
    models: list[str],
    seeds: list[int],
    probes: list[str],
):
    """Aggregate across seeds: dispersion, ICC, full panel."""
    cons_dir = Path(output_dir) / "consolidation"
    cons_dir.mkdir(parents=True, exist_ok=True)

    dispersion: dict = {}
    icc_analysis: dict = {}
    full_panel: dict = {}

    for model_name in models:
        m_key = model_dir_name(model_name)
        dispersion[m_key] = {}
        icc_analysis[m_key] = {}
        full_panel[m_key] = {}

        for probe in probes:
            seed_means = []
            all_item_vecs = []
            seeds_used = []

            for seed in seeds:
                scores = load_run_scores(output_dir, model_name, probe, seed)
                if scores is None:
                    continue
                seed_means.append(np.mean(scores))
                all_item_vecs.append(scores)
                seeds_used.append(seed)

            if not seed_means:
                dispersion[m_key][probe] = {"status": "no_data"}
                icc_analysis[m_key][probe] = {"status": "no_data"}
                full_panel[m_key][probe] = {"status": "no_data"}
                continue

            mean_val = float(np.mean(seed_means))
            std_val = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
            cv = (std_val / mean_val * 100) if mean_val != 0 else float("nan")

            dispersion[m_key][probe] = {
                "mean": round(mean_val, 6),
                "std": round(std_val, 6),
                "cv_pct": round(cv, 3),
                "seeds_used": seeds_used,
                "n_seeds": len(seeds_used),
            }

            # ICC -- requires all vectors to have the same length
            min_len = min(len(v) for v in all_item_vecs)
            if len(all_item_vecs) >= 2 and min_len > 1:
                matrix = np.array([v[:min_len] for v in all_item_vecs]).T
                icc_val = compute_icc(matrix)
                icc_analysis[m_key][probe] = {
                    "icc_3_1": round(icc_val, 6),
                    "n_items": min_len,
                    "n_seeds": len(all_item_vecs),
                }
            else:
                icc_analysis[m_key][probe] = {
                    "status": "insufficient_data",
                    "n_seeds": len(all_item_vecs),
                }

            full_panel[m_key][probe] = {
                "mean": round(mean_val, 6),
                "std": round(std_val, 6),
                "cv_pct": round(cv, 3),
                "icc": icc_analysis[m_key][probe].get("icc_3_1"),
                "n_seeds": len(seeds_used),
            }

    # Write outputs
    with open(cons_dir / "seed_dispersion.json", "w") as f:
        json.dump(dispersion, f, indent=2)
    with open(cons_dir / "icc_analysis.json", "w") as f:
        json.dump(icc_analysis, f, indent=2)
    with open(cons_dir / "full_panel.json", "w") as f:
        json.dump(full_panel, f, indent=2)

    # Summary table
    summary = build_summary(full_panel, models, probes)
    with open(Path(output_dir) / "multiseed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print_summary_table(full_panel, models, probes)
    return summary


def build_summary(full_panel: dict, models: list[str], probes: list[str]) -> dict:
    """Headline numbers across the full panel."""
    all_cvs = []
    all_iccs = []
    per_model = {}

    for model_name in models:
        m_key = model_dir_name(model_name)
        panel = full_panel.get(m_key, {})
        per_model[m_key] = {}
        for probe in probes:
            entry = panel.get(probe, {})
            if "mean" in entry:
                per_model[m_key][probe] = {
                    "mean": entry["mean"],
                    "std": entry["std"],
                    "cv_pct": entry["cv_pct"],
                    "icc": entry.get("icc"),
                }
                if not np.isnan(entry["cv_pct"]):
                    all_cvs.append(entry["cv_pct"])
                if entry.get("icc") is not None and not np.isnan(entry["icc"]):
                    all_iccs.append(entry["icc"])

    return {
        "n_models": len(models),
        "n_probes": len(probes),
        "n_seeds": 4,
        "median_cv_pct": round(float(np.median(all_cvs)), 3) if all_cvs else None,
        "mean_cv_pct": round(float(np.mean(all_cvs)), 3) if all_cvs else None,
        "median_icc": round(float(np.median(all_iccs)), 3) if all_iccs else None,
        "mean_icc": round(float(np.mean(all_iccs)), 3) if all_iccs else None,
        "per_model": per_model,
    }


def print_summary_table(full_panel: dict, models: list[str], probes: list[str]):
    """Pretty-print the consolidation table to stdout."""
    print(f"\n{'='*90}")
    print("MULTI-SEED CONSOLIDATION SUMMARY")
    print(f"{'='*90}")

    header = f"{'Model':<45}"
    for p in probes:
        short = p[:16]
        header += f" | {short:>16}"
    print(header)
    print("-" * len(header))

    for model_name in models:
        m_key = model_dir_name(model_name)
        panel = full_panel.get(m_key, {})
        short_model = model_name.split("/")[-1][:42]
        row = f"{short_model:<45}"
        for probe in probes:
            entry = panel.get(probe, {})
            if "mean" in entry:
                cell = f"{entry['mean']:.3f}+/-{entry['std']:.3f}"
            else:
                cell = "---"
            row += f" | {cell:>16}"
        print(row)

    print(f"{'='*90}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-seed discrimination probes (Experiment 1, AAAI-27)."
    )
    parser.add_argument(
        "--problems-jsonl",
        type=str,
        required=True,
        help="Path to the 51-problem H_K set (JSONL).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root output directory for runs and consolidation.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated subset of model names. Default: all 11.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1337,37,31415,271828",
        help="Comma-separated seeds. Default: 1337,37,31415,271828",
    )
    parser.add_argument(
        "--probes",
        type=str,
        default="consistent_corruption,verdict,per_token_surprise",
        help="Comma-separated probe names.",
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=4,
        help="Tensor parallel size for vLLM. Default: 4.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip (model, probe, seed) combos whose output file already exists.",
    )
    parser.add_argument(
        "--consolidate-only",
        action="store_true",
        help="Skip all runs; only aggregate existing results.",
    )
    return parser.parse_args()


def resolve_models(models_arg: str | None) -> list[str]:
    if models_arg is None:
        return list(ALL_MODELS)
    requested = [m.strip() for m in models_arg.split(",")]
    resolved = []
    for req in requested:
        # Allow matching by short name (e.g. "gemma-2-9b")
        matches = [m for m in ALL_MODELS if req in m]
        if len(matches) == 1:
            resolved.append(matches[0])
        elif len(matches) > 1:
            # Exact match takes priority
            exact = [m for m in matches if m == req]
            if exact:
                resolved.append(exact[0])
            else:
                print(
                    f"[warn] Ambiguous model query '{req}' matches: {matches}. "
                    f"Using first match: {matches[0]}"
                )
                resolved.append(matches[0])
        else:
            # Assume it is a full HF model id not in ALL_MODELS
            print(f"[warn] Model '{req}' not in ALL_MODELS; using as-is.")
            resolved.append(req)
    return resolved


def main():
    args = parse_args()

    models = resolve_models(args.models)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    probes = [p.strip() for p in args.probes.split(",")]

    # Validate probe names
    for p in probes:
        if p not in PROBE_FUNCTIONS:
            print(f"[error] Unknown probe '{p}'. Valid: {list(PROBE_FUNCTIONS.keys())}")
            sys.exit(1)

    print("=" * 72)
    print("Experiment 1: Multi-seed discrimination probes")
    print("=" * 72)
    print(f"  Models  : {len(models)} -- {[m.split('/')[-1] for m in models]}")
    print(f"  Seeds   : {seeds}")
    print(f"  Probes  : {probes}")
    print(f"  TP size : {args.tensor_parallel}")
    print(f"  Output  : {args.output_dir}")
    print(f"  Skip-ex : {args.skip_existing}")
    print(f"  Consol. : {args.consolidate_only}")
    total = len(models) * len(seeds) * len(probes)
    print(f"  Total combos: {total}")
    print("=" * 72)

    if not args.consolidate_only:
        # Cost estimate
        problems = load_problems(args.problems_jsonl)
        print(f"\nLoaded {len(problems)} problems from {args.problems_jsonl}")

        try:
            cost = estimate_cost(models, probes, seeds, problems)
            print(f"Estimated cost: ${cost:.2f}")
        except Exception:
            print("(Cost estimation not available)")

        t_start = time.time()
        summaries = run_all(
            problems=problems,
            output_dir=args.output_dir,
            models=models,
            seeds=seeds,
            probes=probes,
            tensor_parallel=args.tensor_parallel,
            skip_existing=args.skip_existing,
        )
        elapsed_total = time.time() - t_start

        # Save run summaries
        runs_meta_path = Path(args.output_dir) / "run_summaries.json"
        runs_meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(runs_meta_path, "w") as f:
            json.dump(summaries, f, indent=2, default=str)

        errors = [s for s in summaries if "error" in s]
        completed = [s for s in summaries if "mean_score" in s]
        skipped = [s for s in summaries if s.get("skipped")]
        print(f"\nRun phase complete in {elapsed_total:.0f}s")
        print(
            f"  Completed: {len(completed)}  Skipped: {len(skipped)}  "
            f"Errors: {len(errors)}"
        )
        if errors:
            print("  Errored runs:")
            for e in errors:
                print(f"    {e['model']} | {e['probe']} | seed={e['seed']}: {e['error']}")

    # Consolidation
    print("\n--- Consolidation ---")
    summary = consolidate(
        output_dir=args.output_dir,
        models=models,
        seeds=seeds,
        probes=probes,
    )
    print(f"Median CV: {summary.get('median_cv_pct')}%")
    print(f"Median ICC: {summary.get('median_icc')}")
    print(f"\nResults written to {args.output_dir}/")


if __name__ == "__main__":
    main()
