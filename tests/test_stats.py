# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for statistical utilities (stats.py).

Tests Wilson score CIs, mean CIs, pass rate CIs,
aggregate CIs, and per-attribute breakdowns.
"""

import pytest
from hypothesis import given, settings, strategies as st

from patient_agent_bench.eval.stats import (
    wilson_ci,
    mean_ci,
    compute_pass_rate_cis,
    compute_aggregate_ci,
    compute_breakdowns,
    compute_score_stats,
    compute_rubric_score_stats,
)
from patient_agent_bench.eval.constants import PASS_THRESHOLD


# =============================================================================
# wilson_ci tests
# =============================================================================


class TestWilsonCI:
    """Tests for Wilson score confidence interval."""

    def test_zero_total_returns_zeros(self):
        assert wilson_ci(0, 0) == (0.0, 0.0)

    def test_all_pass(self):
        lo, hi = wilson_ci(10, 10)
        assert lo > 0
        assert hi == 100.0 or hi > 95

    def test_none_pass(self):
        lo, hi = wilson_ci(0, 10)
        assert lo == 0.0
        assert hi < 30

    def test_half_pass(self):
        lo, hi = wilson_ci(50, 100)
        assert lo < 50
        assert hi > 50

    def test_bounds_are_percentages(self):
        lo, hi = wilson_ci(3, 10)
        assert 0 <= lo <= 100
        assert 0 <= hi <= 100
        assert lo <= hi

    def test_larger_sample_tighter_ci(self):
        lo_small, hi_small = wilson_ci(5, 10)
        lo_large, hi_large = wilson_ci(50, 100)
        width_small = hi_small - lo_small
        width_large = hi_large - lo_large
        assert width_large < width_small

    @given(
        successes=st.integers(min_value=0, max_value=200),
        total=st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=200)
    def test_property_bounds_valid(self, successes, total):
        """CI bounds are always in [0, 100] and lower <= upper."""
        successes = min(successes, total)
        lo, hi = wilson_ci(successes, total)
        assert 0 <= lo <= 100
        assert 0 <= hi <= 100
        assert lo <= hi


# =============================================================================
# mean_ci tests
# =============================================================================


class TestMeanCI:
    """Tests for mean confidence interval."""

    def test_empty_list(self):
        assert mean_ci([]) == (0.0, 0.0)

    def test_single_value(self):
        assert mean_ci([42.0]) == (42.0, 42.0)

    def test_identical_values(self):
        lo, hi = mean_ci([50.0, 50.0, 50.0])
        assert lo == 50.0
        assert hi == 50.0

    def test_two_values(self):
        lo, hi = mean_ci([40.0, 60.0])
        assert lo < 50.0
        assert hi > 50.0

    def test_ci_contains_mean(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        lo, hi = mean_ci(values)
        mean = sum(values) / len(values)
        assert lo <= mean <= hi

    @given(
        values=st.lists(
            st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=50,
        )
    )
    @settings(max_examples=200)
    def test_property_ci_contains_mean(self, values):
        """CI always contains the sample mean."""
        lo, hi = mean_ci(values)
        mean = sum(values) / len(values)
        assert lo <= mean + 0.01
        assert hi >= mean - 0.01


# =============================================================================
# compute_score_stats tests
# =============================================================================


class TestComputeScoreStats:
    """Tests for compute_score_stats."""

    def test_empty_list(self):
        result = compute_score_stats([])
        assert result["mean"] == 0.0
        assert result["std"] == 0.0
        assert result["ci"] == (0.0, 0.0)

    def test_single_value(self):
        result = compute_score_stats([3.0])
        assert result["mean"] == 3.0
        assert result["std"] == 0.0
        assert result["ci"] == (3.0, 3.0)

    def test_identical_values(self):
        result = compute_score_stats([2.0, 2.0, 2.0])
        assert result["mean"] == 2.0
        assert result["std"] == 0.0

    def test_known_values(self):
        result = compute_score_stats([1.0, 2.0, 3.0, 4.0])
        assert result["mean"] == 2.5
        assert result["std"] > 0
        lo, hi = result["ci"]
        assert lo < 2.5
        assert hi > 2.5

    def test_ci_contains_mean(self):
        result = compute_score_stats([0.0, 1.0, 2.0, 3.0, 4.0])
        lo, hi = result["ci"]
        assert lo <= result["mean"] <= hi

    @given(
        values=st.lists(
            st.floats(min_value=0, max_value=4, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_property_std_non_negative(self, values):
        result = compute_score_stats(values)
        assert result["std"] >= 0


class TestComputeRubricScoreStats:
    """Tests for compute_rubric_score_stats."""

    def test_single_rubric(self):
        totals = {"task_completion": [1.0, 2.0, 3.0]}
        result = compute_rubric_score_stats(totals)
        assert "task_completion" in result
        assert result["task_completion"]["mean"] == 2.0
        assert result["task_completion"]["std"] > 0

    def test_multiple_rubrics(self):
        totals = {
            "task_completion": [3.0, 4.0],
            "clinical_safety": [0.0, 1.0],
        }
        result = compute_rubric_score_stats(totals)
        assert len(result) == 2
        assert result["task_completion"]["mean"] == 3.5
        assert result["clinical_safety"]["mean"] == 0.5

    def test_empty_rubric(self):
        totals = {"empty": []}
        result = compute_rubric_score_stats(totals)
        assert result["empty"]["mean"] == 0.0
        assert result["empty"]["std"] == 0.0


# =============================================================================
# compute_pass_rate_cis tests
# =============================================================================


class TestComputePassRateCIs:
    """Tests for rubric pass rate CI computation (score >= PASS_THRESHOLD)."""

    def test_single_rubric(self):
        # Scores: 5, 3, 1, 4 — two pass (>= 3), two fail
        rubric_totals = {"task_completion": [5.0, 3.0, 1.0, 4.0]}
        cis = compute_pass_rate_cis(rubric_totals)
        assert "task_completion" in cis
        lo, hi = cis["task_completion"]
        assert lo > 0
        assert hi <= 100

    def test_multiple_rubrics(self):
        rubric_totals = {
            "task_completion": [4.0, 5.0],   # both pass (>= 3)
            "clinical_safety": [1.0, 2.0],   # neither passes (< 3)
        }
        cis = compute_pass_rate_cis(rubric_totals)
        assert len(cis) == 2
        assert cis["task_completion"][0] > 0
        assert cis["clinical_safety"][0] == 0.0

    def test_empty_rubric(self):
        rubric_totals = {"empty": []}
        cis = compute_pass_rate_cis(rubric_totals)
        assert cis["empty"] == (0.0, 0.0)

    def test_threshold_boundary(self):
        """Score exactly at PASS_THRESHOLD counts as pass."""
        rubric_totals = {"boundary": [float(PASS_THRESHOLD)]}
        cis = compute_pass_rate_cis(rubric_totals)
        lo, hi = cis["boundary"]
        assert lo > 0

    def test_below_threshold_fails(self):
        """Score one below PASS_THRESHOLD does not count as pass."""
        rubric_totals = {"below": [float(PASS_THRESHOLD - 1)]}
        cis = compute_pass_rate_cis(rubric_totals)
        lo, _hi = cis["below"]
        assert lo == 0.0


# =============================================================================
# compute_aggregate_ci tests
# =============================================================================


class TestComputeAggregateCI:
    """Tests for aggregate score CI computation."""

    def test_delegates_to_mean_ci(self):
        scores = [80.0, 90.0, 70.0]
        ci = compute_aggregate_ci(scores)
        expected = mean_ci(scores)
        assert ci == expected

    def test_empty_scores(self):
        assert compute_aggregate_ci([]) == (0.0, 0.0)


# =============================================================================
# compute_breakdowns tests
# =============================================================================


class TestComputeBreakdowns:
    """Tests for per-attribute breakdown computation."""

    def _make_case(self, case_id, agg_score, rubric_scores):
        row = {"case_id": case_id, "aggregate_score": agg_score}
        row.update(rubric_scores)
        return row

    def test_basic_severity_breakdown(self):
        cases = [
            self._make_case("c1", 80, {"task_completion": 3, "clinical_safety": 4}),
            self._make_case("c2", 40, {"task_completion": 0, "clinical_safety": 1}),
            self._make_case("c3", 90, {"task_completion": 4, "clinical_safety": 3}),
        ]
        metadata = {
            "c1": {"severity_level": "mild", "task_type": "refill", "scenario_complexity": "regular"},
            "c2": {"severity_level": "severe", "task_type": "refill", "scenario_complexity": "regular"},
            "c3": {"severity_level": "mild", "task_type": "appointment", "scenario_complexity": "chronic"},
        }
        rubrics = ["task_completion", "clinical_safety"]

        result = compute_breakdowns(cases, metadata, rubrics)

        assert "by_severity_level" in result
        assert "by_task_type" in result
        assert "by_scenario_complexity" in result

        sev = result["by_severity_level"]
        assert "mild" in sev
        assert "severe" in sev
        assert sev["mild"]["n"] == 2
        assert sev["severe"]["n"] == 1

    def test_breakdown_structure(self):
        cases = [
            self._make_case("c1", 80, {"task_completion": 3}),
        ]
        metadata = {
            "c1": {
                "severity_level": "mild",
                "task_type": "refill",
                "scenario_complexity": "regular",
            },
        }
        rubrics = ["task_completion"]

        result = compute_breakdowns(cases, metadata, rubrics)

        group = result["by_severity_level"]["mild"]
        assert "n" in group
        assert "aggregate_score_avg" in group
        assert "aggregate_score_ci" in group
        assert "rubric_pass_rates" in group
        assert "rubric_pass_rate_cis" in group
        assert isinstance(group["aggregate_score_ci"], list)
        assert len(group["aggregate_score_ci"]) == 2
        # No strict pass rate keys should exist
        assert "rubric_strict_pass_rates" not in group
        assert "rubric_strict_pass_rate_cis" not in group

    def test_missing_metadata_uses_unknown(self):
        cases = [
            self._make_case("c1", 80, {"task_completion": 3}),
        ]
        metadata = {}
        rubrics = ["task_completion"]

        result = compute_breakdowns(cases, metadata, rubrics)

        sev = result["by_severity_level"]
        assert "unknown" in sev

    def test_empty_cases(self):
        result = compute_breakdowns([], {}, ["task_completion"])
        assert "by_severity_level" in result
        assert "by_task_type" in result
        assert "by_scenario_complexity" in result

    def test_pass_rates_correct(self):
        """Pass rate uses PASS_THRESHOLD: count scores >= PASS_THRESHOLD."""
        cases = [
            self._make_case("c1", 80, {"safety": 4}),   # pass (4 >= any threshold)
            self._make_case("c2", 40, {"safety": 0}),   # fail
            self._make_case("c3", 60, {"safety": 2}),   # depends on threshold
        ]
        metadata = {
            "c1": {"severity_level": "mild", "task_type": "x", "scenario_complexity": "regular"},
            "c2": {"severity_level": "mild", "task_type": "x", "scenario_complexity": "regular"},
            "c3": {"severity_level": "mild", "task_type": "x", "scenario_complexity": "regular"},
        }
        rubrics = ["safety"]

        result = compute_breakdowns(cases, metadata, rubrics)
        group = result["by_severity_level"]["mild"]
        # Count how many of [4, 0, 2] pass at the current threshold
        scores = [4, 0, 2]
        expected_passes = sum(1 for s in scores if s >= PASS_THRESHOLD)
        expected_rate = round(expected_passes / len(scores) * 100, 1)
        assert group["rubric_pass_rates"]["safety"] == pytest.approx(
            expected_rate, abs=0.1
        )


# =============================================================================
# Summary integration tests (OutputManager.generate_summary)
# =============================================================================


class TestGenerateSummaryWithCIsAndBreakdowns:
    """
    Integration-style tests verifying that generate_summary produces
    the expected CI and breakdown keys (no strict pass rate keys).
    """

    def _make_output_manager(self, tmp_path):
        from patient_agent_bench.runner.output_manager import OutputManager

        om = OutputManager(
            input_file="data/sample_benchmark.json",
            base_output_dir=str(tmp_path),
            timestamp="20260101_000000",
        )
        om.setup()
        return om

    def _make_eval(self, case_id, agg_score, rubric_scores):
        return {
            "case_id": case_id,
            "evaluation": {
                "aggregate_score": agg_score,
                "rubric_scores": rubric_scores,
                "rubric_results": {
                    name: {"score": score, "explanation": "ok"}
                    for name, score in rubric_scores.items()
                },
            },
        }

    def test_summary_includes_cis_without_strict(self, tmp_path):
        om = self._make_output_manager(tmp_path)
        evaluations = [
            self._make_eval("c1", 75, {"task_completion": 3, "clinical_safety": 4}),
            self._make_eval("c2", 50, {"task_completion": 2, "clinical_safety": 1}),
            self._make_eval("c3", 90, {"task_completion": 4, "clinical_safety": 3}),
        ]
        metadata = {
            "c1": {"severity_level": "mild", "task_type": "refill", "scenario_complexity": "regular"},
            "c2": {"severity_level": "severe", "task_type": "refill", "scenario_complexity": "regular"},
            "c3": {"severity_level": "mild", "task_type": "appointment", "scenario_complexity": "chronic"},
        }

        summary = om.generate_summary(evaluations, case_metadata=metadata)

        # CIs present
        assert "confidence_intervals" in summary
        ci = summary["confidence_intervals"]
        assert "aggregate_score_ci" in ci
        assert "rubric_pass_rate_cis" in ci
        assert "rubric_score_cis" in ci
        # No strict pass rate CIs
        assert "rubric_strict_pass_rate_cis" not in ci

        # Score stats present (Family 1)
        assert "aggregate_score_stats" in summary
        agg_stats = summary["aggregate_score_stats"]
        assert "mean" in agg_stats
        assert "std" in agg_stats
        assert "ci" in agg_stats

        assert "rubric_score_stats" in summary
        for rubric in ["task_completion", "clinical_safety"]:
            assert rubric in summary["rubric_score_stats"]
            rs = summary["rubric_score_stats"][rubric]
            assert "mean" in rs
            assert "std" in rs
            assert "ci" in rs

        # Pass rates present, no strict
        assert "rubric_pass_rates" in summary
        assert "rubric_strict_pass_rates" not in summary

        # Breakdowns present
        assert "breakdowns" in summary

    def test_experiments_summary_no_strict_keys(self, tmp_path):
        om = self._make_output_manager(tmp_path)

        from patient_agent_bench.runner.experiment_config import ExperimentConfig
        from patient_agent_bench.config import AgentSpec, ModelConfig

        model = ModelConfig(model_id="test-model", temperature=0.0, max_tokens=1024)
        exp = ExperimentConfig(
            experiment_id="0_0_0",
            assistant_agent=AgentSpec(model=model),
            user_agent=AgentSpec(model=model),
            evaluator_models=[model],
            sandbox_model=model,
            assistant_idx=0,
            user_idx=0,
            max_turns=3,
            strip_thinking_content=True,
            analyzer_model=model,
        )

        evaluations = [
            self._make_eval("c1", 75, {"task_completion": 3}),
            self._make_eval("c2", 50, {"task_completion": 1}),
        ]

        results = {
            "0_0_0": {"evaluations": evaluations},
        }

        summary = om.generate_experiments_summary(results, [exp])

        # Check comparison table row has no strict keys
        row = summary["comparison_table"][0]
        for key in row:
            assert "strict" not in key.lower(), f"Found strict key: {key}"

        # Check comparison table row has score stats
        assert "average_score_std" in row
        assert "task_completion_std" in row
        assert "task_completion_ci" in row

        # Check pooled stats present
        assert "pooled_stats" in summary
        ps = summary["pooled_stats"]
        assert "aggregate_score_stats" in ps
        assert "rubric_score_stats" in ps
        assert "rubric_pass_rates" in ps
        assert "rubric_pass_rate_cis" in ps
