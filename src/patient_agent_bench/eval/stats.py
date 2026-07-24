# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Statistical utilities for benchmark summary generation.

Provides:
- Wilson score confidence intervals for pass rates
- Normal approximation confidence intervals for mean scores
- Per-attribute breakdown computation (severity, task_type, complexity)

No LLM calls, no side effects. Pure math.
"""

import math
from typing import Any, Dict, List, Tuple

from patient_agent_bench.eval.constants import PASS_THRESHOLD


def wilson_ci(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for a proportion.

    More accurate than normal approximation for small samples
    and extreme proportions.

    Args:
        successes: Number of successes (passes).
        total: Total number of trials.
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple of (lower, upper) bounds as percentages (0-100).
    """
    if total == 0:
        return (0.0, 0.0)

    # z-score for confidence level
    z = _z_score(confidence)
    p_hat = successes / total
    z2 = z * z

    denom = 1 + z2 / total
    center = (p_hat + z2 / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(
        p_hat * (1 - p_hat) / total + z2 / (4 * total * total)
    )

    lower = max(0.0, center - margin) * 100
    upper = min(1.0, center + margin) * 100
    return (round(lower, 1), round(upper, 1))


def mean_ci(
    values: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute confidence interval for a mean using normal approximation.

    Args:
        values: List of numeric values.
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple of (lower, upper) bounds.
    """
    n = len(values)
    if n < 2:
        if n == 1:
            return (values[0], values[0])
        return (0.0, 0.0)

    z = _z_score(confidence)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    se = math.sqrt(variance / n)

    lower = mean - z * se
    upper = mean + z * se
    return (round(lower, 2), round(upper, 2))


def _z_score(confidence: float) -> float:
    """Return z-score for common confidence levels."""
    # Hardcoded for common values to avoid scipy dependency
    z_table = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576,
    }
    return z_table.get(confidence, 1.96)


def compute_score_stats(
    values: List[float],
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    Compute mean, std, and CI for a list of scores.

    Args:
        values: List of numeric scores.
        confidence: Confidence level (default 0.95).

    Returns:
        Dict with keys: mean, std, ci (tuple of lower, upper).
    """
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci": (0.0, 0.0)}

    mean = sum(values) / n
    if n < 2:
        return {"mean": round(mean, 2), "std": 0.0, "ci": (round(mean, 2), round(mean, 2))}

    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    ci = mean_ci(values, confidence)

    return {"mean": round(mean, 2), "std": round(std, 2), "ci": ci}


def compute_rubric_score_stats(
    rubric_totals: Dict[str, List[float]],
    confidence: float = 0.95,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute mean, std, and CI for each rubric's scores across cases.

    Args:
        rubric_totals: Dict mapping rubric name to list of scores.
        confidence: Confidence level.

    Returns:
        Dict mapping rubric name to {mean, std, ci}.
    """
    return {
        name: compute_score_stats(vals, confidence)
        for name, vals in rubric_totals.items()
    }


def compute_pass_rate_cis(
    rubric_totals: Dict[str, List[float]],
    confidence: float = 0.95,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute Wilson CIs for pass rates (>= PASS_THRESHOLD) across all rubrics.

    Args:
        rubric_totals: Dict mapping rubric name to list of scores.
        confidence: Confidence level.

    Returns:
        Dict mapping rubric name to (lower, upper) CI as percentages.
    """
    cis = {}
    for name, vals in rubric_totals.items():
        total = len(vals)
        passes = sum(1 for v in vals if v >= PASS_THRESHOLD)
        cis[name] = wilson_ci(passes, total, confidence)
    return cis


def compute_aggregate_ci(
    scores: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute CI for the mean aggregate score.

    Args:
        scores: List of aggregate scores (1-5).
        confidence: Confidence level.

    Returns:
        Tuple of (lower, upper) bounds.
    """
    return mean_ci(scores, confidence)


# --- Attribute breakdown computation ---

# Attributes we break down by, and their known value sets
BREAKDOWN_ATTRIBUTES = {
    "severity_level": ["mild", "moderate", "severe"],
    "task_type": None,  # discovered dynamically from data
    "scenario_complexity": ["regular", "chronic", "complicated", "infeasible"],
    "personality": None,  # discovered dynamically from data
    "age_group": ["pediatric", "young_adult", "middle_aged", "senior"],
    "gender_identity": None,  # discovered dynamically from data
}


def compute_breakdowns(
    case_results: List[Dict[str, Any]],
    case_metadata: Dict[str, Dict[str, str]],
    rubric_names: List[str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Compute per-attribute breakdowns of scores and pass rates.

    Groups case results by each breakdown attribute and computes
    aggregate score, per-rubric pass rates, num_turns stats, and CIs
    for each group.

    Args:
        case_results: List of per-case result dicts with case_id,
            aggregate_score, num_turns, and rubric scores.
        case_metadata: Dict mapping case_id to metadata dict with
            severity_level, task_type, scenario_complexity, etc.
        rubric_names: List of rubric names to compute pass rates for.

    Returns:
        Nested dict: attribute_name -> attribute_value -> {
            "n": count,
            "aggregate_score_avg": float,
            "aggregate_score_ci": [lower, upper],
            "rubric_averages": {rubric: avg_score},
            "rubric_pass_rates": {rubric: rate},
            "rubric_pass_rate_cis": {rubric: [lower, upper]},
            "num_turns_avg": float,
            "num_turns_std": float,
        }
    """
    breakdowns: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for attr_name in BREAKDOWN_ATTRIBUTES:
        # Group cases by attribute value
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for case in case_results:
            case_id = case.get("case_id", "")
            meta = case_metadata.get(case_id, {})
            attr_val = meta.get(attr_name, "unknown")
            if not attr_val:
                attr_val = "unknown"
            groups.setdefault(attr_val, []).append(case)

        attr_breakdown: Dict[str, Dict[str, Any]] = {}
        for val, cases in sorted(groups.items()):
            n = len(cases)
            if n == 0:
                continue

            # Aggregate scores
            agg_scores = [
                c["aggregate_score"] for c in cases
                if c.get("aggregate_score") is not None
            ]
            if not agg_scores:
                continue
            n = len(agg_scores)
            agg_avg = sum(agg_scores) / n
            agg_ci = compute_aggregate_ci(agg_scores)

            # Per-rubric pass rates + CIs + averages
            rubric_pass_rates: Dict[str, float] = {}
            rubric_pass_cis: Dict[str, List[float]] = {}
            rubric_averages: Dict[str, float] = {}
            for rubric in rubric_names:
                scores = [c[rubric] for c in cases if c.get(rubric) is not None]
                total = len(scores)

                passes = sum(1 for s in scores if s >= PASS_THRESHOLD)
                rate = (passes / total * 100) if total > 0 else 0
                ci = wilson_ci(passes, total)
                rubric_pass_rates[rubric] = round(rate, 1)
                rubric_pass_cis[rubric] = list(ci)
                rubric_averages[rubric] = round(
                    sum(scores) / total, 2
                ) if total > 0 else 0

            # Num turns stats
            turns = [c["num_turns"] for c in cases if c.get("num_turns") is not None]
            turns_stats = compute_score_stats(turns) if turns else {"mean": 0, "std": 0, "ci": [0, 0]}

            attr_breakdown[val] = {
                "n": n,
                "aggregate_score_avg": round(agg_avg, 2),
                "aggregate_score_ci": list(agg_ci),
                "rubric_averages": rubric_averages,
                "rubric_pass_rates": rubric_pass_rates,
                "rubric_pass_rate_cis": rubric_pass_cis,
                "num_turns_avg": turns_stats["mean"],
                "num_turns_std": turns_stats["std"],
            }

        breakdowns[f"by_{attr_name}"] = attr_breakdown

    return breakdowns
