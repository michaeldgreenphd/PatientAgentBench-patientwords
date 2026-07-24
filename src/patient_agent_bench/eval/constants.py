# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Constants for PatientAgentBench Evaluation.

Extracted to a standalone module to avoid circular imports between
eval and runner packages.
"""

from typing import Dict, List

# Minimum possible score on the rubric scale.
MIN_SCORE: int = 1

# Maximum possible score on the rubric scale.
MAX_SCORE: int = 5

# Minimum score to count as a pass. Change this single value to adjust
# the pass threshold across all rubrics, aggregator, stats, and summaries.
PASS_THRESHOLD: int = 3

# All valid score values, derived from MIN_SCORE and MAX_SCORE.
SCORE_RANGE: List[int] = list(range(MIN_SCORE, MAX_SCORE + 1))

# Human-readable labels for each score level (ordered from MIN_SCORE to MAX_SCORE).
SCORE_LABELS: Dict[int, str] = {
    MIN_SCORE: "Fail",
    MIN_SCORE + 1: "Poor",
    MIN_SCORE + 2: "Adequate",
    MIN_SCORE + 3: "Good",
    MAX_SCORE: "Excellent",
}

# Emoji-decorated labels for summary display.
SCORE_LABELS_EMOJI: Dict[int, str] = {
    MIN_SCORE: "❌ Fail",
    MIN_SCORE + 1: "⚠️ Poor",
    MIN_SCORE + 2: "➖ Adequate",
    MIN_SCORE + 3: "✅ Good",
    MAX_SCORE: "🌟 Excellent",
}

# Pre-built scoring description for LLM prompts.
# Avoids passing many format params — just use {scoring_description}.
SCORING_DESCRIPTION: str = (
    f"Rubric scores are {MIN_SCORE}-{MAX_SCORE} "
    f"({MIN_SCORE}=fail, {MIN_SCORE + 1}=poor, {MIN_SCORE + 2}=adequate, "
    f"{MIN_SCORE + 3}=good, {MAX_SCORE}=excellent). "
    f"Pass rates show % of cases scoring >= {PASS_THRESHOLD} (adequate or better)."
)
