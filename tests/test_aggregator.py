# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for multi-evaluator aggregation.

Tests the Aggregator pure functions including property-based tests for:
- Property 3: Pass label correctness
- Property 4: Average aggregation
- Property 5: Majority vote aggregation
- Property 6: Merged evaluation output structure
- Property 7: Explanation concatenation
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from patient_agent_bench.eval.aggregator import add_pass_labels, merge_evaluations
from patient_agent_bench.eval.constants import PASS_THRESHOLD, MIN_SCORE, MAX_SCORE


# =============================================================================
# Strategies
# =============================================================================

RUBRIC_NAMES = [
    "task_completion",
    "clinical_safety",
    "workflow_accuracy",
    "triage_quality",
    "clinical_helpfulness",
]


@st.composite
def rubric_score(draw):
    """Generate a valid rubric score (MIN_SCORE-MAX_SCORE)."""
    return draw(st.integers(min_value=MIN_SCORE, max_value=MAX_SCORE))


@st.composite
def evaluation_dict(draw):
    """Generate a valid evaluation dict with rubric results."""
    scores = {}
    results = {}
    for name in RUBRIC_NAMES:
        score = draw(rubric_score())
        scores[name] = score
        results[name] = {
            "score": score,
            "explanation": draw(st.text(min_size=1, max_size=50)),
        }

    agg = draw(st.floats(min_value=MIN_SCORE, max_value=MAX_SCORE, allow_nan=False))
    return {
        "rubric_scores": scores,
        "rubric_results": results,
        "aggregate_score": agg,
        "summary": draw(st.text(min_size=1, max_size=100)),
    }


@st.composite
def evaluation_list(draw, min_size=1, max_size=5):
    """Generate a list of evaluation dicts."""
    return draw(st.lists(evaluation_dict(), min_size=min_size, max_size=max_size))


# =============================================================================
# Unit Tests
# =============================================================================


class TestAddPassLabels:
    """Unit tests for add_pass_labels.

    All assertions reference PASS_THRESHOLD so the tests adapt
    automatically when the constant changes.
    """

    @pytest.mark.parametrize("score", range(MIN_SCORE, MAX_SCORE + 1))
    def test_pass_label_matches_threshold(self, score):
        ev = {
            "rubric_results": {"safety": {"score": score, "explanation": "x"}},
            "rubric_scores": {"safety": score},
        }
        result = add_pass_labels(ev)
        assert result["rubric_results"]["safety"]["pass"] == (score >= PASS_THRESHOLD)

    def test_below_threshold_is_fail(self):
        score = PASS_THRESHOLD - 1
        ev = {
            "rubric_results": {"safety": {"score": score, "explanation": "below"}},
            "rubric_scores": {"safety": score},
        }
        result = add_pass_labels(ev)
        assert result["rubric_results"]["safety"]["pass"] is False

    def test_at_threshold_is_pass(self):
        score = PASS_THRESHOLD
        ev = {
            "rubric_results": {"safety": {"score": score, "explanation": "at"}},
            "rubric_scores": {"safety": score},
        }
        result = add_pass_labels(ev)
        assert result["rubric_results"]["safety"]["pass"] is True

    def test_multiple_rubrics(self):
        above = PASS_THRESHOLD
        below = PASS_THRESHOLD - 1
        ev = {
            "rubric_results": {
                "task_completion": {"score": above, "explanation": "done"},
                "clinical_safety": {"score": below, "explanation": "unsafe"},
            },
            "rubric_scores": {"task_completion": above, "clinical_safety": below},
        }
        result = add_pass_labels(ev)
        assert result["rubric_results"]["task_completion"]["pass"] is True
        assert result["rubric_results"]["clinical_safety"]["pass"] is False


class TestMergeEvaluations:
    """Unit tests for merge_evaluations."""

    def test_single_evaluation(self):
        ev = {
            "rubric_scores": {"task_completion": 4, "clinical_safety": 3},
            "rubric_results": {
                "task_completion": {
                    "score": 4, "pass": 4 >= PASS_THRESHOLD,
                    "explanation": "good",
                },
                "clinical_safety": {
                    "score": 3, "pass": 3 >= PASS_THRESHOLD,
                    "explanation": "ok",
                },
            },
            "aggregate_score": 75.0,
            "summary": "Summary text",
        }
        merged = merge_evaluations([ev])
        assert merged["rubric_scores"]["task_completion"] == 4.0
        assert merged["rubric_scores"]["clinical_safety"] == 3.0
        assert merged["aggregate_score"] == 75.0

    def test_two_evaluators_average(self):
        ev1 = {
            "rubric_scores": {"safety": 5},
            "rubric_results": {
                "safety": {
                    "score": 5, "pass": True,
                    "explanation": "safe",
                }
            },
            "aggregate_score": 100.0,
            "summary": "Good",
        }
        ev2 = {
            "rubric_scores": {"safety": 1},
            "rubric_results": {
                "safety": {
                    "score": 1, "pass": False,
                    "explanation": "unsafe",
                }
            },
            "aggregate_score": 0.0,
            "summary": "Bad",
        }
        merged = merge_evaluations([ev1, ev2], method="average")
        assert merged["rubric_scores"]["safety"] == 3.0
        assert merged["rubric_results"]["safety"]["pass"] is True  # 3.0 >= 3.0
        assert merged["aggregate_score"] == 50.0

    def test_two_evaluators_majority_vote(self):
        ev1 = {
            "rubric_scores": {"safety": 4},
            "rubric_results": {
                "safety": {
                    "score": 4, "pass": 4 >= PASS_THRESHOLD,
                    "explanation": "safe",
                }
            },
            "aggregate_score": 100.0,
            "summary": "Good",
        }
        ev2 = {
            "rubric_scores": {"safety": 1},
            "rubric_results": {
                "safety": {
                    "score": 1, "pass": 1 >= PASS_THRESHOLD,
                    "explanation": "unsafe",
                }
            },
            "aggregate_score": 0.0,
            "summary": "Bad",
        }
        merged = merge_evaluations([ev1, ev2], method="majority_vote")
        # 1 pass out of 2 = not strict majority
        individual = [4 >= PASS_THRESHOLD, 1 >= PASS_THRESHOLD]
        num_passed = sum(1 for p in individual if p)
        expected = num_passed > len(individual) / 2
        assert merged["rubric_results"]["safety"]["pass"] is expected

    def test_three_evaluators_majority_vote(self):
        pass_5 = 5 >= PASS_THRESHOLD
        pass_3 = 3 >= PASS_THRESHOLD
        pass_1 = 1 >= PASS_THRESHOLD
        evals = [
            {
                "rubric_scores": {"safety": 5},
                "rubric_results": {
                    "safety": {
                        "score": 5, "pass": pass_5,
                        "explanation": "a",
                    }
                },
                "aggregate_score": 100.0,
                "summary": "A",
            },
            {
                "rubric_scores": {"safety": 3},
                "rubric_results": {
                    "safety": {
                        "score": 3, "pass": pass_3,
                        "explanation": "b",
                    }
                },
                "aggregate_score": 50.0,
                "summary": "B",
            },
            {
                "rubric_scores": {"safety": 1},
                "rubric_results": {
                    "safety": {
                        "score": 1, "pass": pass_1,
                        "explanation": "c",
                    }
                },
                "aggregate_score": 0.0,
                "summary": "C",
            },
        ]
        merged = merge_evaluations(evals, method="majority_vote")
        # majority vote: pass if more than half of individual pass labels are True
        individual_passes = [pass_5, pass_3, pass_1]
        num_passed = sum(1 for p in individual_passes if p)
        expected = num_passed > len(individual_passes) / 2
        assert merged["rubric_results"]["safety"]["pass"] is expected

    def test_empty_evaluations(self):
        merged = merge_evaluations([])
        assert "error" in merged

    def test_all_errored_evaluations(self):
        merged = merge_evaluations([{"error": "fail1"}, {"error": "fail2"}])
        assert "error" in merged

    def test_skips_errored_evaluations(self):
        ev_good = {
            "rubric_scores": {"safety": 4},
            "rubric_results": {
                "safety": {
                    "score": 4, "pass": True,
                    "explanation": "ok",
                }
            },
            "aggregate_score": 100.0,
            "summary": "Good",
        }
        merged = merge_evaluations([ev_good, {"error": "fail"}])
        assert merged["rubric_scores"]["safety"] == 4.0

    def test_explanation_concatenation(self):
        ev1 = {
            "rubric_scores": {"safety": 5},
            "rubric_results": {
                "safety": {
                    "score": 5, "pass": 5 >= PASS_THRESHOLD,
                    "explanation": "first",
                }
            },
            "aggregate_score": 100.0,
            "summary": "S1",
        }
        ev2 = {
            "rubric_scores": {"safety": 3},
            "rubric_results": {
                "safety": {
                    "score": 3, "pass": 3 >= PASS_THRESHOLD,
                    "explanation": "second",
                }
            },
            "aggregate_score": 50.0,
            "summary": "S2",
        }
        merged = merge_evaluations([ev1, ev2])
        explanation = merged["rubric_results"]["safety"]["explanation"]
        assert "[Evaluator 0] first" in explanation
        assert "[Evaluator 1] second" in explanation


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestAggregatorPropertyBased:
    """Property-based tests for Aggregator."""

    @given(score=st.integers(min_value=MIN_SCORE, max_value=MAX_SCORE))
    @settings(max_examples=100)
    def test_property_3_pass_label_correctness(self, score):
        """
        Property 3: Pass label correctness

        *For any* rubric score (1-5), the pass label SHALL be True
        if and only if score >= PASS_THRESHOLD.
        """
        ev = {
            "rubric_results": {"test": {"score": score, "explanation": "x"}},
            "rubric_scores": {"test": score},
        }
        result = add_pass_labels(ev)
        assert result["rubric_results"]["test"]["pass"] == (score >= PASS_THRESHOLD)

    @given(evals=evaluation_list())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_4_average_aggregation(self, evals):
        """
        Property 4: Average aggregation computes correct scores and pass labels
        """
        labeled = [add_pass_labels(e) for e in evals]
        merged = merge_evaluations(labeled, method="average")

        for rubric_name in RUBRIC_NAMES:
            scores = [
                e["rubric_scores"][rubric_name]
                for e in labeled
                if rubric_name in e.get("rubric_scores", {})
            ]
            if scores:
                expected_avg = sum(scores) / len(scores)
                assert abs(merged["rubric_scores"][rubric_name] - expected_avg) < 1e-9
                assert merged["rubric_results"][rubric_name]["pass"] == (
                    expected_avg >= PASS_THRESHOLD
                )

    @given(evals=evaluation_list())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_5_majority_vote_aggregation(self, evals):
        """
        Property 5: Majority vote aggregation computes correct pass labels
        """
        labeled = [add_pass_labels(e) for e in evals]
        merged = merge_evaluations(labeled, method="majority_vote")

        for rubric_name in RUBRIC_NAMES:
            individual_passes = [
                e["rubric_results"][rubric_name]["pass"]
                for e in labeled
                if rubric_name in e.get("rubric_results", {})
            ]
            if individual_passes:
                num_passed = sum(1 for p in individual_passes if p)
                expected_pass = num_passed > len(individual_passes) / 2
                assert merged["rubric_results"][rubric_name]["pass"] == expected_pass

                # Score should still be average
                scores = [
                    e["rubric_scores"][rubric_name]
                    for e in labeled
                    if rubric_name in e.get("rubric_scores", {})
                ]
                expected_avg = sum(scores) / len(scores)
                assert abs(
                    merged["rubric_scores"][rubric_name] - expected_avg
                ) < 1e-9

    @given(evals=evaluation_list())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_merged_output_structure(self, evals):
        """
        Property 6: Merged evaluation output structure
        """
        labeled = [add_pass_labels(e) for e in evals]
        merged = merge_evaluations(labeled)

        assert "rubric_scores" in merged
        assert "rubric_results" in merged
        assert "aggregate_score" in merged
        assert "summary" in merged

        for rubric_name in merged["rubric_results"]:
            result = merged["rubric_results"][rubric_name]
            assert "score" in result
            assert "pass" in result
            assert "explanation" in result

    @given(evals=evaluation_list(min_size=1, max_size=3))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_7_explanation_contains_all(self, evals):
        """
        Property 7: Merged explanation contains all individual explanations
        """
        labeled = [add_pass_labels(e) for e in evals]
        merged = merge_evaluations(labeled)

        for rubric_name in RUBRIC_NAMES:
            merged_explanation = merged["rubric_results"].get(
                rubric_name, {}
            ).get("explanation", "")
            for i, ev in enumerate(labeled):
                individual_exp = (
                    ev.get("rubric_results", {})
                    .get(rubric_name, {})
                    .get("explanation", "")
                )
                if individual_exp:
                    assert individual_exp in merged_explanation


# =============================================================================
# Tests for Inter-Rater Agreement Metrics
# =============================================================================


class TestInterRaterAgreement:
    """Tests for inter-rater agreement metrics (score_std, pass_agreement)."""

    def test_score_std_perfect_agreement(self):
        """All evaluators give same score -> std = 0."""
        evals = [
            {
                "rubric_scores": {"safety": 3},
                "rubric_results": {
                    "safety": {
                        "score": 3, "pass": True,
                        "explanation": "a",
                    }
                },
                "aggregate_score": 100.0,
                "summary": "A",
            },
            {
                "rubric_scores": {"safety": 3},
                "rubric_results": {
                    "safety": {
                        "score": 3, "pass": True,
                        "explanation": "b",
                    }
                },
                "aggregate_score": 100.0,
                "summary": "B",
            },
        ]
        merged = merge_evaluations(evals)
        assert merged["rubric_results"]["safety"]["score_std"] == 0.0

    def test_score_std_disagreement(self):
        """Evaluators give different scores -> std > 0."""
        evals = [
            {
                "rubric_scores": {"safety": 5},
                "rubric_results": {
                    "safety": {
                        "score": 5, "pass": True,
                        "explanation": "a",
                    }
                },
                "aggregate_score": 100.0,
                "summary": "A",
            },
            {
                "rubric_scores": {"safety": 1},
                "rubric_results": {
                    "safety": {
                        "score": 1, "pass": False,
                        "explanation": "b",
                    }
                },
                "aggregate_score": 0.0,
                "summary": "B",
            },
        ]
        merged = merge_evaluations(evals)
        # std of [5, 1] = 2.0
        assert merged["rubric_results"]["safety"]["score_std"] == 2.0

    def test_pass_agreement_unanimous(self):
        """All evaluators agree on pass -> agreement = 1.0."""
        evals = [
            {
                "rubric_scores": {"safety": 4},
                "rubric_results": {
                    "safety": {"score": 4, "pass": True, "explanation": "a"}
                },
                "aggregate_score": 100.0,
                "summary": "A",
            },
            {
                "rubric_scores": {"safety": 3},
                "rubric_results": {
                    "safety": {"score": 3, "pass": True, "explanation": "b"}
                },
                "aggregate_score": 75.0,
                "summary": "B",
            },
            {
                "rubric_scores": {"safety": 4},
                "rubric_results": {
                    "safety": {"score": 4, "pass": True, "explanation": "c"}
                },
                "aggregate_score": 100.0,
                "summary": "C",
            },
        ]
        merged = merge_evaluations(evals)
        assert merged["rubric_results"]["safety"]["pass_agreement"] == 1.0

    def test_pass_agreement_split(self):
        """Evaluators split on pass -> agreement = 0.5."""
        evals = [
            {
                "rubric_scores": {"safety": 4},
                "rubric_results": {
                    "safety": {"score": 4, "pass": True, "explanation": "a"}
                },
                "aggregate_score": 100.0,
                "summary": "A",
            },
            {
                "rubric_scores": {"safety": 1},
                "rubric_results": {
                    "safety": {"score": 1, "pass": False, "explanation": "b"}
                },
                "aggregate_score": 0.0,
                "summary": "B",
            },
        ]
        merged = merge_evaluations(evals)
        assert merged["rubric_results"]["safety"]["pass_agreement"] == 0.5

    def test_aggregate_score_std(self):
        """Merged result includes aggregate_score_std."""
        evals = [
            {
                "rubric_scores": {"safety": 5},
                "rubric_results": {
                    "safety": {"score": 5, "pass": True, "explanation": "a"}
                },
                "aggregate_score": 100.0,
                "summary": "A",
            },
            {
                "rubric_scores": {"safety": 1},
                "rubric_results": {
                    "safety": {"score": 1, "pass": False, "explanation": "b"}
                },
                "aggregate_score": 0.0,
                "summary": "B",
            },
        ]
        merged = merge_evaluations(evals)
        assert "aggregate_score_std" in merged
        assert merged["aggregate_score_std"] == 50.0

    def test_single_evaluator_std_is_zero(self):
        """Single evaluator -> std = 0, agreement = 1.0."""
        ev = {
            "rubric_scores": {"safety": 3},
            "rubric_results": {
                "safety": {"score": 3, "pass": True, "explanation": "a"}
            },
            "aggregate_score": 100.0,
            "summary": "A",
        }
        merged = merge_evaluations([ev])
        assert merged["rubric_results"]["safety"]["score_std"] == 0.0
        assert merged["rubric_results"]["safety"]["pass_agreement"] == 1.0
        assert merged["aggregate_score_std"] == 0.0


class TestInterRaterAgreementPropertyBased:
    """Property-based tests for inter-rater agreement metrics."""

    @given(evals=evaluation_list(min_size=1, max_size=5))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_score_std_non_negative(self, evals):
        """score_std is always >= 0."""
        labeled = [add_pass_labels(e) for e in evals]
        merged = merge_evaluations(labeled)

        for rubric_name in merged["rubric_results"]:
            assert merged["rubric_results"][rubric_name]["score_std"] >= 0

    @given(evals=evaluation_list(min_size=1, max_size=5))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_pass_agreement_in_range(self, evals):
        """pass_agreement is always in [0.5, 1.0]."""
        labeled = [add_pass_labels(e) for e in evals]
        merged = merge_evaluations(labeled)

        for rubric_name in merged["rubric_results"]:
            agreement = merged["rubric_results"][rubric_name]["pass_agreement"]
            assert 0.5 <= agreement <= 1.0

    @given(evals=evaluation_list(min_size=1, max_size=5))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_aggregate_score_std_non_negative(self, evals):
        """aggregate_score_std is always >= 0."""
        labeled = [add_pass_labels(e) for e in evals]
        merged = merge_evaluations(labeled)

        assert merged["aggregate_score_std"] >= 0
