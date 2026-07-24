# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for evaluation system.

Tests Evaluator orchestration and aggregate scoring.
"""

import pytest
from unittest.mock import MagicMock, patch

from patient_agent_bench.config import ModelConfig
from patient_agent_bench.eval.evaluator import Evaluator
from patient_agent_bench.eval.aggregator import (
    calculate_aggregate_score,
    generate_evaluation_summary,
)
from patient_agent_bench.eval.base_rubric import BaseRubric
from patient_agent_bench.eval.constants import MIN_SCORE, MAX_SCORE, PASS_THRESHOLD

from tests.conftest import create_mock_rubric


@pytest.fixture
def mock_bedrock_llm():
    """Mock create_chat_model and create_bedrock_client_with_role to avoid AWS calls."""
    with patch("patient_agent_bench.eval.base_rubric.create_bedrock_client_with_role") as mock_client:
        mock_client.return_value = MagicMock()
        with patch("patient_agent_bench.eval.base_rubric.create_chat_model") as mock_factory:
            mock_llm = MagicMock()
            mock_factory.return_value = mock_llm
            yield mock_llm


class TestEvaluator:
    """Tests for Evaluator class."""

    def test_init_with_default_rubrics(self, model_config, mock_bedrock_llm):
        """Test evaluator initializes with all 6 default rubrics."""
        evaluator = Evaluator(model_config=model_config)

        assert len(evaluator.rubrics) == 6
        rubric_names = evaluator.get_rubric_names()
        assert "task_completion" in rubric_names
        assert "clinical_safety" in rubric_names
        assert "workflow_accuracy" in rubric_names
        assert "triage_quality" in rubric_names
        assert "clinical_helpfulness" in rubric_names
        assert "conversational_quality" in rubric_names

    def test_init_with_custom_rubrics(self, model_config):
        """Test evaluator with custom rubrics."""
        mock_rubrics = [
            create_mock_rubric("custom_rubric_1"),
            create_mock_rubric("custom_rubric_2"),
        ]

        evaluator = Evaluator(model_config=model_config, rubrics=mock_rubrics)

        assert len(evaluator.rubrics) == 2
        assert evaluator.get_rubric_names() == ["custom_rubric_1", "custom_rubric_2"]

    def test_evaluate_all_pass(self, model_config, sample_conversation):
        """Test evaluation when all rubrics pass with equal weights."""
        mock_rubrics = [
            create_mock_rubric("task_completion", score=4, weight=1.0),
            create_mock_rubric("clinical_safety", score=4, weight=1.0),
            create_mock_rubric("workflow_accuracy", score=4, weight=1.0),
        ]

        evaluator = Evaluator(model_config=model_config, rubrics=mock_rubrics)
        result = evaluator.evaluate(
            conversation_history=sample_conversation,
            user_profile="<profile>Test</profile>",
            scenario="Test scenario",
        )

        assert result["rubric_scores"]["task_completion"] == 4
        assert result["rubric_scores"]["clinical_safety"] == 4
        assert "aggregate_score" not in result  # Now computed in post-processing

    def test_evaluate_weighted_scores(self, model_config, sample_conversation):
        """Test that weights affect aggregate score when computed in post-processing."""
        mock_rubrics = [
            create_mock_rubric("task_completion", score=2, weight=1.0),
            create_mock_rubric("clinical_safety", score=0, weight=2.0),  # Higher weight
        ]

        evaluator = Evaluator(model_config=model_config, rubrics=mock_rubrics)
        result = evaluator.evaluate(
            conversation_history=sample_conversation,
            user_profile="<profile>Test</profile>",
        )

        # evaluate() no longer computes aggregate_score
        assert "aggregate_score" not in result

        # Post-processing with explicit weights
        weights = {"task_completion": 1.0, "clinical_safety": 2.0}
        result = calculate_aggregate_score(result, weights=weights)
        # weighted_sum = 2*1.0 + 0*2.0 = 2
        # total_weight = 1.0 + 2.0 = 3
        # score = 2/3 = 0.67
        assert 0.66 <= result["aggregate_score"] <= 0.68

    def test_evaluate_partial_scores(self, model_config, sample_conversation):
        """Test evaluation with mixed scores."""
        mock_rubrics = [
            create_mock_rubric("task_completion", score=2, weight=1.0),
            create_mock_rubric("clinical_safety", score=1, weight=1.0),
            create_mock_rubric("workflow_accuracy", score=1, weight=1.0),
        ]

        evaluator = Evaluator(model_config=model_config, rubrics=mock_rubrics)
        result = evaluator.evaluate(
            conversation_history=sample_conversation,
            user_profile="<profile>Test</profile>",
        )

        # Post-processing computes aggregate
        weights = {"task_completion": 1.0, "clinical_safety": 1.0, "workflow_accuracy": 1.0}
        result = calculate_aggregate_score(result, weights=weights)
        # weighted_sum = 2*1 + 1*1 + 1*1 = 4
        # total_weight = 1 + 1 + 1 = 3
        # score = 4/3 = 1.33
        assert 1.33 <= result["aggregate_score"] <= 1.34

    def test_evaluate_propagates_rubric_error(self, model_config, sample_conversation):
        """A rubric/infra error must propagate (not be scored as MIN_SCORE).

        Fabricating a numeric score for an infrastructure failure (e.g. a
        connection error) would poison the aggregate. Instead the exception
        propagates so the orchestrator records the conversation as a failed
        task and excludes it from scoring / re-runs it on resume.
        """
        import pytest

        mock_rubric_ok = create_mock_rubric("task_completion", score=2)
        mock_rubric_error = create_mock_rubric("clinical_safety", score=0)
        mock_rubric_error.evaluate.side_effect = Exception("Rubric error")

        evaluator = Evaluator(model_config=model_config, rubrics=[mock_rubric_ok, mock_rubric_error])
        with pytest.raises(Exception, match="Rubric error"):
            evaluator.evaluate(
                conversation_history=sample_conversation,
                user_profile="<profile>Test</profile>",
            )

    def test_evaluate_generates_summary(self, model_config, sample_conversation):
        """Test evaluation result can have summary added via post-processing."""
        mock_rubrics = [create_mock_rubric("task_completion", score=2)]

        evaluator = Evaluator(model_config=model_config, rubrics=mock_rubrics)
        result = evaluator.evaluate(
            conversation_history=sample_conversation,
            user_profile="<profile>Test</profile>",
        )

        # Summary is now added as a post-processing step, not by evaluate()
        assert "summary" not in result

        # After post-processing, summary should be present
        result = generate_evaluation_summary(result)
        assert "summary" in result
        assert isinstance(result["summary"], str)
        assert "Evaluation Summary" in result["summary"]


class TestAggregateScoreCalculation:
    """Tests for aggregate score calculation logic (now via calculate_aggregate_score)."""

    def test_calculate_aggregate_all_equal_weights(self, model_config):
        """Test aggregate score with all equal weights."""
        evaluation = {
            "rubric_scores": {
                "task_completion": 4,
                "clinical_safety": 4,
                "workflow_accuracy": 4,
            },
        }
        weights = {"task_completion": 1.0, "clinical_safety": 1.0, "workflow_accuracy": 1.0}
        result = calculate_aggregate_score(evaluation, weights=weights)
        assert result["aggregate_score"] == 4.0

    def test_calculate_aggregate_mixed_scores(self, model_config):
        """Test aggregate score with mixed scores and equal weights."""
        evaluation = {
            "rubric_scores": {
                "task_completion": 4,
                "clinical_safety": 2,
                "workflow_accuracy": 0,
            },
        }
        weights = {"task_completion": 1.0, "clinical_safety": 1.0, "workflow_accuracy": 1.0}
        result = calculate_aggregate_score(evaluation, weights=weights)
        # (4 + 2 + 0) / 3 = 2.0
        assert result["aggregate_score"] == 2.0

    def test_calculate_aggregate_with_weights(self, model_config):
        """Test aggregate score respects rubric weights."""
        evaluation = {
            "rubric_scores": {
                "task_completion": 4,
                "clinical_safety": 0,
            },
        }
        weights = {"task_completion": 1.0, "clinical_safety": 2.0}
        result = calculate_aggregate_score(evaluation, weights=weights)
        # weighted_sum = 4*1.0 + 0*2.0 = 4
        # total_weight = 1.0 + 2.0 = 3
        # score = 4/3 = 1.33
        assert 1.33 <= result["aggregate_score"] <= 1.34

    def test_calculate_aggregate_empty_scores(self, model_config):
        """Test aggregate score with empty scores."""
        evaluation = {"rubric_scores": {}}
        result = calculate_aggregate_score(evaluation)
        assert result["aggregate_score"] == 0.0

    def test_calculate_aggregate_default_weights(self, model_config):
        """Test aggregate score uses default rubric weights when none provided."""
        evaluation = {
            "rubric_scores": {
                "task_completion": 5,
                "clinical_safety": 5,
            },
        }
        result = calculate_aggregate_score(evaluation)
        assert result["aggregate_score"] == 5.0


class TestSummaryGeneration:
    """Tests for summary generation (now via generate_evaluation_summary)."""

    def test_generate_summary_pass(self, model_config):
        """Test summary for passing evaluation."""
        evaluation = {
            "rubric_scores": {"task_completion": MAX_SCORE, "clinical_safety": MAX_SCORE},
            "aggregate_score": float(MAX_SCORE),
        }
        result = generate_evaluation_summary(evaluation)
        summary = result["summary"]

        assert f"{float(MAX_SCORE):.2f}/{MAX_SCORE}" in summary
        assert "🌟 Excellent" in summary

    def test_generate_summary_fail(self, model_config):
        """Test summary for failing evaluation."""
        evaluation = {
            "rubric_scores": {"clinical_safety": MIN_SCORE},
            "aggregate_score": float(MIN_SCORE),
        }
        result = generate_evaluation_summary(evaluation)
        summary = result["summary"]

        assert f"{float(MIN_SCORE):.2f}/{MAX_SCORE}" in summary
        assert "❌ Fail" in summary

    def test_generate_summary_partial_pass(self, model_config):
        """Test summary for poor score."""
        evaluation = {
            "rubric_scores": {"task_completion": MIN_SCORE + 1},
            "aggregate_score": float(MIN_SCORE + 1),
        }
        result = generate_evaluation_summary(evaluation)
        summary = result["summary"]

        assert "⚠️ Poor" in summary

    def test_generate_summary_includes_rubric_names(self, model_config):
        """Test summary includes rubric display names."""
        evaluation = {
            "rubric_scores": {"clinical_safety": PASS_THRESHOLD},
            "aggregate_score": float(PASS_THRESHOLD),
        }
        result = generate_evaluation_summary(evaluation)
        summary = result["summary"]

        assert "Clinical Safety" in summary



