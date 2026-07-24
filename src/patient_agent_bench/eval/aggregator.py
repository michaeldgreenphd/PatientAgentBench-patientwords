# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Aggregator for multi-evaluator evaluation results.

Provides pure functions for:
- Adding pass labels to individual evaluation results
- Computing weighted aggregate scores
- Generating human-readable evaluation summaries
- Merging multiple evaluator results into a single aggregated evaluation
- Computing inter-rater agreement metrics (score_std, pass_agreement)

No LLM calls, no side effects.
"""

import math
from typing import Any, Dict, List, Optional

from patient_agent_bench.eval.constants import (
    MAX_SCORE,
    MIN_SCORE,
    PASS_THRESHOLD,
    SCORE_LABELS_EMOJI,
)
from patient_agent_bench.eval.rubrics import get_default_rubric_weights


def add_pass_labels(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add pass boolean to each rubric result.

    A single label is added:
    - 'pass': score >= PASS_THRESHOLD

    Applied to each individual evaluator's result immediately after
    evaluation, before merging.

    Args:
        evaluation: Evaluation dict with rubric_results containing score fields.

    Returns:
        The same evaluation dict with pass boolean added to each rubric result.
    """
    rubric_results = evaluation.get("rubric_results", {})
    for rubric_name, result in rubric_results.items():
        if isinstance(result, dict) and "score" in result:
            result["pass"] = result["score"] >= PASS_THRESHOLD
    return evaluation


def calculate_aggregate_score(
    evaluation: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute the weighted aggregate score and attach it to the evaluation dict.

    Post-processing step. If weights is None, uses default rubric weights.
    Rubrics not present in the weights dict default to 1.0.

    Args:
        evaluation: Evaluation dict with rubric_scores.
        weights: Optional mapping of rubric_name -> weight.

    Returns:
        The same evaluation dict with "aggregate_score" added/updated.
    """
    if weights is None:
        weights = get_default_rubric_weights()

    rubric_scores = evaluation.get("rubric_scores", {})
    if not rubric_scores:
        evaluation["aggregate_score"] = 0.0
        return evaluation

    weighted_sum = 0.0
    total_weight = 0.0
    for rubric_name, score in rubric_scores.items():
        w = weights.get(rubric_name, 1.0)
        weighted_sum += score * w
        total_weight += w

    evaluation["aggregate_score"] = round(
        (weighted_sum / total_weight) if total_weight > 0 else 0.0, 2
    )
    return evaluation


def generate_evaluation_summary(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a human-readable summary and attach it to the evaluation dict.

    Post-processing step called after add_pass_labels. Reads rubric_scores
    and aggregate_score from the evaluation dict and produces a markdown
    summary string.

    Args:
        evaluation: Evaluation dict with rubric_scores and aggregate_score.

    Returns:
        The same evaluation dict with a "summary" key added.
    """
    rubric_scores = evaluation.get("rubric_scores", {})
    aggregate_score = evaluation.get("aggregate_score", 0.0)

    lines = ["## Evaluation Summary", ""]
    lines.append(f"**Aggregate Score**: {aggregate_score:.2f}/{MAX_SCORE}")
    lines.append("")
    lines.append(f"### Rubric Scores ({MIN_SCORE}-{MAX_SCORE} scale)")
    lines.append("")

    for rubric_name, score in rubric_scores.items():
        label = SCORE_LABELS_EMOJI.get(
            score if isinstance(score, int) else round(score),
            f"Score: {score}",
        )
        display_name = rubric_name.replace("_", " ").title()
        lines.append(f"- **{display_name}**: {score}/{MAX_SCORE} ({label})")

    evaluation["summary"] = "\n".join(lines)
    return evaluation



def _compute_std(values: List[float]) -> float:
    """Compute population standard deviation of a list of values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _compute_pass_agreement(passes: List[bool]) -> float:
    """
    Compute pass agreement as proportion of evaluators in majority.

    For N evaluators, returns the fraction that agree with the majority.
    E.g., [True, True, False] → 2/3 = 0.67 (majority is True)
    E.g., [True, True, True] → 3/3 = 1.0 (unanimous)
    E.g., [True, False] → 1/2 = 0.5 (split)
    """
    if not passes:
        return 0.0
    num_true = sum(1 for p in passes if p)
    num_false = len(passes) - num_true
    majority_count = max(num_true, num_false)
    return majority_count / len(passes)


def merge_evaluations(
    evaluations: List[Dict[str, Any]],
    method: str = "average",
) -> Dict[str, Any]:
    """
    Merge N evaluation dicts (already with pass labels) into one.

    Args:
        evaluations: List of evaluation dicts, each with rubric_results,
                     rubric_scores, aggregate_score, and summary.
        method: Aggregation method - "average" or "majority_vote".

    Returns:
        Merged evaluation dict with:
        - rubric_scores: averaged scores
        - rubric_results: per-rubric with averaged score, pass label,
                          score_std, pass_agreement, concatenated explanations
        - aggregate_score: recalculated from merged rubric scores
        - aggregate_score_std: std of aggregate scores across evaluators
        - summary: concatenated summaries
    """
    if not evaluations:
        return {"error": "No evaluations to merge"}

    # Filter out errored evaluations
    valid_evals = [e for e in evaluations if "error" not in e]
    if not valid_evals:
        return {"error": "All evaluators failed"}

    # Collect all rubric names across evaluations
    all_rubric_names: List[str] = []
    for ev in valid_evals:
        for name in ev.get("rubric_scores", {}):
            if name not in all_rubric_names:
                all_rubric_names.append(name)

    # Merge rubric scores and results
    merged_rubric_scores: Dict[str, float] = {}
    merged_rubric_results: Dict[str, Dict[str, Any]] = {}

    for rubric_name in all_rubric_names:
        scores: List[float] = []
        passes: List[bool] = []
        explanations: List[str] = []

        for i, ev in enumerate(valid_evals):
            rubric_scores = ev.get("rubric_scores", {})
            rubric_results = ev.get("rubric_results", {})

            if rubric_name in rubric_scores:
                score = rubric_scores[rubric_name]
                scores.append(float(score))

                result = rubric_results.get(rubric_name, {})
                passes.append(result.get("pass", score >= PASS_THRESHOLD))

                explanation = result.get("explanation", "")
                if explanation:
                    explanations.append(f"[Evaluator {i}] {explanation}")

        if not scores:
            continue

        avg_score = sum(scores) / len(scores)
        merged_rubric_scores[rubric_name] = avg_score

        # Compute inter-rater agreement metrics
        score_std = _compute_std(scores)
        pass_agreement = _compute_pass_agreement(passes)

        # Determine pass label based on method
        if method == "majority_vote":
            num_passed = sum(1 for p in passes if p)
            merged_pass = num_passed > len(passes) / 2
        else:  # average
            merged_pass = avg_score >= PASS_THRESHOLD

        merged_rubric_results[rubric_name] = {
            "score": avg_score,
            "pass": merged_pass,
            "score_std": round(score_std, 3),
            "pass_agreement": round(pass_agreement, 3),
            "explanation": " ".join(explanations),
        }

    # Recalculate aggregate_score as mean of individual aggregate_scores
    agg_scores = [
        ev.get("aggregate_score", 0.0) for ev in valid_evals
    ]
    merged_aggregate = sum(agg_scores) / len(agg_scores) if agg_scores else 0.0
    aggregate_score_std = _compute_std(agg_scores)

    # Concatenate summaries
    summaries = []
    for i, ev in enumerate(valid_evals):
        s = ev.get("summary", "")
        if s:
            summaries.append(f"[Evaluator {i}] {s}")
    merged_summary = "\n\n".join(summaries)

    return {
        "rubric_results": merged_rubric_results,
        "rubric_scores": merged_rubric_scores,
        "aggregate_score": round(merged_aggregate, 2),
        "aggregate_score_std": round(aggregate_score_std, 3),
        "summary": merged_summary,
    }
