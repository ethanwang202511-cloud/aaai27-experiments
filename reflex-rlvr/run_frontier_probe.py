#!/usr/bin/env python3
"""Experiment 3 – Frontier-model discrimination probes (AAAI-27).

Runs verdict-based discrimination probes on GPT-4o and Claude Sonnet via their
APIs.  For each AIME problem the script:
  1. Presents problem + oracle solution  -> measures P(YES) vs P(NO).
  2. Presents problem + corrupted solution -> measures P(YES) vs P(NO).
  3. Computes  delta_verdict = L(oracle) - L(corrupted)
     where  L(c) = log P(YES|c) - log P(NO|c).

Usage:
    python run_frontier_probe.py \
        --problems-jsonl data/aime_51.jsonl \
        --output-dir results/frontier
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import corruption helper
# ---------------------------------------------------------------------------
try:
    from discrimination_standalone import corrupt_solution_consistent
except ImportError:
    # Fall back to the in-tree location used during development.
    try:
        sys.path.insert(
            0,
            str(
                Path(__file__).resolve().parents[1]
                / "src"
                / "reflex_rlvr"
                / "modal_app"
            ),
        )
        from discrimination_v2 import corrupt_solution_consistent
    except ImportError:
        # Inline a minimal version so the script is self-contained.
        import random as _random
        import re as _re

        def corrupt_solution_consistent(
            solution: str, answer, *, seed: int = 0
        ) -> tuple[str, int]:
            """Minimal fallback: shift boxed answer by a random delta."""
            rng = _random.Random(seed)
            original = int(answer)
            deltas = [1, 2, 3, 5, 7, -1, -2, -3, -5, -7]
            rng.shuffle(deltas)
            for d in deltas:
                candidate = original + d
                if candidate > 0 and candidate != original:
                    break
            else:
                candidate = original + 1

            def _replace_boxed(m: _re.Match) -> str:
                return m.group(0).replace(str(original), str(candidate))

            new_sol = _re.sub(
                r"\\boxed\{[^}]*\}", _replace_boxed, solution, count=1
            )
            if new_sol == solution:
                new_sol = solution + f"\n\\boxed{{{candidate}}}"
            return new_sol, candidate


# ---------------------------------------------------------------------------
# Retry decorator (tenacity)
# ---------------------------------------------------------------------------
try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False

    # Provide a no-op decorator so the code still works without tenacity.
    def retry(**_kw):  # type: ignore[misc]
        def _dec(fn):
            return fn

        return _dec

    class _Dummy:
        def __call__(self, *a, **kw):
            return True

    retry_if_exception_type = _Dummy  # type: ignore[assignment,misc]
    stop_after_attempt = _Dummy  # type: ignore[assignment,misc]
    wait_exponential = _Dummy  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verdict template
# ---------------------------------------------------------------------------
VERDICT_TEMPLATE = """\
You are a careful math grader. Read the problem and the proposed solution, \
then judge whether the proposed solution is correct.

Problem: {problem}

Proposed solution: {solution}

Is the proposed solution correct? Answer YES or NO.

Answer:"""


# ===================================================================
# GPT-4o probing
# ===================================================================

def _openai_logprob_probe(
    client,
    problem: str,
    solution: str,
    model: str = "gpt-4o",
) -> dict:
    """Query GPT-4o with logprobs to get P(YES) and P(NO)."""
    prompt = VERDICT_TEMPLATE.format(problem=problem, solution=solution)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful math grader."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1,
        logprobs=True,
        top_logprobs=5,
        temperature=0.0,
    )
    choice = response.choices[0]
    top_lps = choice.logprobs.content[0].top_logprobs

    lp_yes: float | None = None
    lp_no: float | None = None

    for entry in top_lps:
        tok = entry.token.strip().upper()
        if tok == "YES":
            lp_yes = entry.logprob
        elif tok == "NO":
            lp_no = entry.logprob

    # If both found, great.  If only one, estimate the other.
    if lp_yes is not None and lp_no is not None:
        method = "logprobs_both"
    elif lp_yes is not None:
        # Remaining probability mass after YES
        remaining = 1.0 - math.exp(lp_yes)
        lp_no = math.log(max(remaining, 1e-10))
        method = "logprobs_yes_only"
    elif lp_no is not None:
        remaining = 1.0 - math.exp(lp_no)
        lp_yes = math.log(max(remaining, 1e-10))
        method = "logprobs_no_only"
    else:
        # Neither in top-5 -- fall back to None so caller can use sampling
        method = "logprobs_neither"

    usage = response.usage
    return {
        "lp_yes": lp_yes,
        "lp_no": lp_no,
        "method": method,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "response_token": choice.logprobs.content[0].token.strip(),
    }


def _openai_sampling_probe(
    client,
    problem: str,
    solution: str,
    n_samples: int = 20,
    model: str = "gpt-4o",
) -> dict:
    """Fallback: sample N completions and estimate P(YES)/P(NO)."""
    prompt = VERDICT_TEMPLATE.format(problem=problem, solution=solution)
    eps = 1.0 / (2.0 * n_samples)
    count_yes = 0
    count_no = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for _ in range(n_samples):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a careful math grader."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=5,
            temperature=0.7,
        )
        text = response.choices[0].message.content.strip().upper()
        if text.startswith("YES"):
            count_yes += 1
        elif text.startswith("NO"):
            count_no += 1
        usage = response.usage
        if usage:
            total_prompt_tokens += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens
        time.sleep(0.1)  # gentle pacing within burst

    p_yes = (count_yes + eps) / (n_samples + 2 * eps)
    p_no = (count_no + eps) / (n_samples + 2 * eps)
    lp_yes = math.log(p_yes)
    lp_no = math.log(p_no)

    return {
        "lp_yes": lp_yes,
        "lp_no": lp_no,
        "method": "sampling",
        "count_yes": count_yes,
        "count_no": count_no,
        "n_samples": n_samples,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
    }


def _make_openai_retry():
    """Build a retry-wrapped probe function for OpenAI."""
    import openai as _oai

    @retry(
        retry=retry_if_exception_type((_oai.RateLimitError, _oai.APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(6),
    )
    def _probe(client, problem, solution, n_samples, model="gpt-4o"):
        result = _openai_logprob_probe(client, problem, solution, model=model)
        if result["method"] == "logprobs_neither":
            log.warning("Logprobs unavailable for GPT-4o, falling back to sampling")
            result = _openai_sampling_probe(
                client, problem, solution, n_samples=n_samples, model=model
            )
        return result

    return _probe


def run_gpt4o(
    problems: list[dict],
    api_key: str,
    n_samples: int = 20,
    seed: int = 1337,
) -> list[dict]:
    """Run GPT-4o probes on all problems."""
    import openai

    client = openai.OpenAI(api_key=api_key)
    probe_fn = _make_openai_retry()
    results = []

    for idx, prob in enumerate(tqdm(problems, desc="GPT-4o probing")):
        problem_text = prob["problem"]
        oracle_solution = prob["solution"]
        answer = prob.get("answer", prob.get("expected_answer", "0"))

        corrupted_solution, corrupted_answer = corrupt_solution_consistent(
            oracle_solution, answer, seed=seed + idx
        )

        # Oracle probe
        oracle_result = probe_fn(
            client, problem_text, oracle_solution, n_samples
        )
        time.sleep(0.5)

        # Corrupted probe
        corrupted_result = probe_fn(
            client, problem_text, corrupted_solution, n_samples
        )
        time.sleep(0.5)

        # Compute log-odds
        L_oracle = oracle_result["lp_yes"] - oracle_result["lp_no"]
        L_corrupted = corrupted_result["lp_yes"] - corrupted_result["lp_no"]
        delta_verdict = L_oracle - L_corrupted

        results.append(
            {
                "problem_idx": idx,
                "problem_id": prob.get("id", idx),
                "answer": str(answer),
                "corrupted_answer": str(corrupted_answer),
                "oracle": {
                    "lp_yes": oracle_result["lp_yes"],
                    "lp_no": oracle_result["lp_no"],
                    "L": L_oracle,
                    "method": oracle_result["method"],
                    "response_token": oracle_result.get("response_token"),
                },
                "corrupted": {
                    "lp_yes": corrupted_result["lp_yes"],
                    "lp_no": corrupted_result["lp_no"],
                    "L": L_corrupted,
                    "method": corrupted_result["method"],
                    "response_token": corrupted_result.get("response_token"),
                },
                "delta_verdict": delta_verdict,
                "oracle_above": delta_verdict > 0,
                "prompt_tokens": (
                    oracle_result["prompt_tokens"]
                    + corrupted_result["prompt_tokens"]
                ),
                "completion_tokens": (
                    oracle_result["completion_tokens"]
                    + corrupted_result["completion_tokens"]
                ),
            }
        )

    return results


# ===================================================================
# Claude Sonnet probing
# ===================================================================

def _make_anthropic_retry():
    """Build a retry-wrapped probe function for Anthropic."""
    import anthropic as _ant

    @retry(
        retry=retry_if_exception_type(
            (_ant.RateLimitError, _ant.APITimeoutError)
        ),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(6),
    )
    def _single_sample(client, prompt, model):
        response = client.messages.create(
            model=model,
            max_tokens=5,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
            system="You are a careful math grader.",
        )
        text = response.content[0].text.strip().upper()
        usage = response.usage
        return text, usage.input_tokens, usage.output_tokens

    return _single_sample


def _anthropic_sampling_probe(
    client,
    sample_fn,
    problem: str,
    solution: str,
    n_samples: int = 20,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Sample N completions from Claude and estimate P(YES)/P(NO)."""
    prompt = VERDICT_TEMPLATE.format(problem=problem, solution=solution)
    eps = 1.0 / (2.0 * n_samples)
    count_yes = 0
    count_no = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for _ in range(n_samples):
        text, in_tok, out_tok = sample_fn(client, prompt, model)
        if text.startswith("YES"):
            count_yes += 1
        elif text.startswith("NO"):
            count_no += 1
        total_input_tokens += in_tok
        total_output_tokens += out_tok
        time.sleep(0.05)  # gentle pacing within burst

    p_yes = (count_yes + eps) / (n_samples + 2 * eps)
    p_no = (count_no + eps) / (n_samples + 2 * eps)
    lp_yes = math.log(p_yes)
    lp_no = math.log(p_no)

    return {
        "lp_yes": lp_yes,
        "lp_no": lp_no,
        "method": "sampling",
        "count_yes": count_yes,
        "count_no": count_no,
        "n_samples": n_samples,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
    }


def run_claude_sonnet(
    problems: list[dict],
    api_key: str,
    n_samples: int = 20,
    seed: int = 1337,
) -> list[dict]:
    """Run Claude Sonnet probes on all problems."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    sample_fn = _make_anthropic_retry()
    model = "claude-sonnet-4-20250514"
    results = []

    for idx, prob in enumerate(tqdm(problems, desc="Claude Sonnet probing")):
        problem_text = prob["problem"]
        oracle_solution = prob["solution"]
        answer = prob.get("answer", prob.get("expected_answer", "0"))

        corrupted_solution, corrupted_answer = corrupt_solution_consistent(
            oracle_solution, answer, seed=seed + idx
        )

        # Oracle probe
        oracle_result = _anthropic_sampling_probe(
            client, sample_fn, problem_text, oracle_solution,
            n_samples=n_samples, model=model,
        )
        time.sleep(1.0)

        # Corrupted probe
        corrupted_result = _anthropic_sampling_probe(
            client, sample_fn, problem_text, corrupted_solution,
            n_samples=n_samples, model=model,
        )
        time.sleep(1.0)

        # Compute log-odds
        L_oracle = oracle_result["lp_yes"] - oracle_result["lp_no"]
        L_corrupted = corrupted_result["lp_yes"] - corrupted_result["lp_no"]
        delta_verdict = L_oracle - L_corrupted

        results.append(
            {
                "problem_idx": idx,
                "problem_id": prob.get("id", idx),
                "answer": str(answer),
                "corrupted_answer": str(corrupted_answer),
                "oracle": {
                    "lp_yes": oracle_result["lp_yes"],
                    "lp_no": oracle_result["lp_no"],
                    "L": L_oracle,
                    "method": oracle_result["method"],
                    "count_yes": oracle_result["count_yes"],
                    "count_no": oracle_result["count_no"],
                },
                "corrupted": {
                    "lp_yes": corrupted_result["lp_yes"],
                    "lp_no": corrupted_result["lp_no"],
                    "L": L_corrupted,
                    "method": corrupted_result["method"],
                    "count_yes": corrupted_result["count_yes"],
                    "count_no": corrupted_result["count_no"],
                },
                "delta_verdict": delta_verdict,
                "oracle_above": delta_verdict > 0,
                "input_tokens": (
                    oracle_result["input_tokens"]
                    + corrupted_result["input_tokens"]
                ),
                "output_tokens": (
                    oracle_result["output_tokens"]
                    + corrupted_result["output_tokens"]
                ),
            }
        )

    return results


# ===================================================================
# Summary / aggregation
# ===================================================================

# Known open-model baselines for comparison
OPEN_MODEL_BASELINES = {
    "Qwen2.5-7B-Instruct": {"delta_verdict": 7.34},
    "Qwen2.5-1.5B-base": {"delta_verdict": 0.53},
    "Llama-3.1-8B": {"delta_verdict": 0.0},
}

# Cost per token (USD) as of 2025-07
GPT4O_COST = {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000}
CLAUDE_SONNET_COST = {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000}


def _summarize_model(results: list[dict], model_name: str, cost_table: dict) -> dict:
    """Compute aggregate statistics for one model."""
    n = len(results)
    if n == 0:
        return {}

    deltas = [r["delta_verdict"] for r in results]
    mean_delta = sum(deltas) / n
    pct_above = sum(1 for d in deltas if d > 0) / n
    verdict_acc = pct_above  # fraction where oracle logit > corrupted logit

    # Token counts
    if "prompt_tokens" in results[0]:
        total_in = sum(r["prompt_tokens"] for r in results)
        total_out = sum(r["completion_tokens"] for r in results)
    else:
        total_in = sum(r["input_tokens"] for r in results)
        total_out = sum(r["output_tokens"] for r in results)

    cost = total_in * cost_table["input"] + total_out * cost_table["output"]

    return {
        "model": model_name,
        "n_problems": n,
        "mean_delta_verdict": round(mean_delta, 4),
        "median_delta_verdict": round(
            sorted(deltas)[n // 2], 4
        ),
        "std_delta_verdict": round(
            (sum((d - mean_delta) ** 2 for d in deltas) / max(n - 1, 1)) ** 0.5,
            4,
        ),
        "pct_oracle_above_corrupted": round(pct_above * 100, 2),
        "verdict_accuracy": round(verdict_acc, 4),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "estimated_cost_usd": round(cost, 4),
    }


def build_summary(
    gpt4o_results: list[dict] | None,
    claude_results: list[dict] | None,
) -> dict:
    """Build the aggregate comparison JSON."""
    summary: dict = {"frontier_models": {}, "open_model_baselines": OPEN_MODEL_BASELINES}

    if gpt4o_results:
        summary["frontier_models"]["gpt-4o"] = _summarize_model(
            gpt4o_results, "gpt-4o", GPT4O_COST
        )
    if claude_results:
        summary["frontier_models"]["claude-sonnet"] = _summarize_model(
            claude_results, "claude-sonnet-4", CLAUDE_SONNET_COST
        )

    # Build comparison table
    table = []
    for name, stats in OPEN_MODEL_BASELINES.items():
        table.append({"model": name, "delta_verdict": stats["delta_verdict"], "source": "open-model"})
    for name, stats in summary["frontier_models"].items():
        table.append({
            "model": name,
            "delta_verdict": stats["mean_delta_verdict"],
            "verdict_accuracy": stats["verdict_accuracy"],
            "source": "frontier-api",
        })
    summary["comparison_table"] = table

    return summary


# ===================================================================
# I/O helpers
# ===================================================================

def load_problems(path: str) -> list[dict]:
    """Load problems from a JSONL file."""
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    log.info("Loaded %d problems from %s", len(problems), path)
    return problems


def write_jsonl(results: list[dict], path: Path) -> None:
    """Write results to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    log.info("Wrote %d records to %s", len(results), path)


def write_json(obj: dict, path: Path) -> None:
    """Write a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    log.info("Wrote summary to %s", path)


# ===================================================================
# Main
# ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experiment 3: Frontier-model discrimination probes"
    )
    p.add_argument(
        "--problems-jsonl",
        type=str,
        required=True,
        help="Path to JSONL file with 51 AIME problems (H_K set)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output files",
    )
    p.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    p.add_argument(
        "--anthropic-api-key",
        type=str,
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of samples for sampling-based approach (default: 20)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for corruption (default: 1337)",
    )
    p.add_argument(
        "--models",
        type=str,
        default="gpt-4o,claude-sonnet",
        help="Comma-separated model list (default: gpt-4o,claude-sonnet)",
    )
    p.add_argument(
        "--skip-openai",
        action="store_true",
        help="Skip GPT-4o probing",
    )
    p.add_argument(
        "--skip-anthropic",
        action="store_true",
        help="Skip Claude Sonnet probing",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve API keys
    openai_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    anthropic_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")

    # Determine which models to run
    requested_models = {m.strip().lower() for m in args.models.split(",")}
    run_openai = "gpt-4o" in requested_models and not args.skip_openai
    run_anthropic = "claude-sonnet" in requested_models and not args.skip_anthropic

    if run_openai and not openai_key:
        log.warning(
            "No OpenAI API key provided (--openai-api-key or OPENAI_API_KEY). "
            "Skipping GPT-4o."
        )
        run_openai = False

    if run_anthropic and not anthropic_key:
        log.warning(
            "No Anthropic API key provided (--anthropic-api-key or ANTHROPIC_API_KEY). "
            "Skipping Claude Sonnet."
        )
        run_anthropic = False

    if not run_openai and not run_anthropic:
        log.error("No models to run. Provide at least one API key.")
        sys.exit(1)

    # Load problems
    problems = load_problems(args.problems_jsonl)
    log.info(
        "Running frontier probes: OpenAI=%s, Anthropic=%s, n_samples=%d, seed=%d",
        run_openai,
        run_anthropic,
        args.n_samples,
        args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpt4o_results = None
    claude_results = None

    # --- GPT-4o ---
    if run_openai:
        log.info("=" * 60)
        log.info("Running GPT-4o discrimination probes")
        log.info("=" * 60)
        try:
            gpt4o_results = run_gpt4o(
                problems, openai_key, n_samples=args.n_samples, seed=args.seed
            )
            write_jsonl(gpt4o_results, output_dir / "gpt4o_verdict.jsonl")

            # Quick inline summary
            deltas = [r["delta_verdict"] for r in gpt4o_results]
            acc = sum(1 for d in deltas if d > 0) / len(deltas)
            log.info(
                "GPT-4o: mean_delta=%.4f, verdict_acc=%.2f%%",
                sum(deltas) / len(deltas),
                acc * 100,
            )
        except Exception:
            log.exception("GPT-4o probing failed")

    # --- Claude Sonnet ---
    if run_anthropic:
        log.info("=" * 60)
        log.info("Running Claude Sonnet discrimination probes")
        log.info("=" * 60)
        try:
            claude_results = run_claude_sonnet(
                problems, anthropic_key, n_samples=args.n_samples, seed=args.seed
            )
            write_jsonl(claude_results, output_dir / "claude_sonnet_verdict.jsonl")

            # Quick inline summary
            deltas = [r["delta_verdict"] for r in claude_results]
            acc = sum(1 for d in deltas if d > 0) / len(deltas)
            log.info(
                "Claude Sonnet: mean_delta=%.4f, verdict_acc=%.2f%%",
                sum(deltas) / len(deltas),
                acc * 100,
            )
        except Exception:
            log.exception("Claude Sonnet probing failed")

    # --- Summary ---
    summary = build_summary(gpt4o_results, claude_results)
    write_json(summary, output_dir / "frontier_summary.json")

    # Print comparison table
    log.info("=" * 60)
    log.info("Comparison table (delta_verdict)")
    log.info("%-30s %12s %12s", "Model", "delta_v", "accuracy")
    log.info("-" * 56)
    for row in summary["comparison_table"]:
        acc_str = (
            f"{row['verdict_accuracy']:.2%}"
            if "verdict_accuracy" in row
            else "n/a"
        )
        log.info(
            "%-30s %12.4f %12s",
            row["model"],
            row["delta_verdict"],
            acc_str,
        )
    log.info("=" * 60)

    # Print cost estimate
    total_cost = 0.0
    for model_stats in summary["frontier_models"].values():
        cost = model_stats.get("estimated_cost_usd", 0)
        total_cost += cost
        log.info(
            "%s: %d input tokens, %d output tokens, est. $%.4f",
            model_stats["model"],
            model_stats["total_input_tokens"],
            model_stats["total_output_tokens"],
            cost,
        )
    log.info("Total estimated cost: $%.4f", total_cost)

    log.info("Done. Results in %s", output_dir)


if __name__ == "__main__":
    main()
