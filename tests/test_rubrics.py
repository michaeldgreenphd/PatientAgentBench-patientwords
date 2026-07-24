# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for evaluation rubrics.

Provides template-based testing for all rubrics - when adding a new rubric,
add it to RUBRIC_CLASSES and it will automatically be tested.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from patient_agent_bench.config import ModelConfig
from patient_agent_bench.eval.base_rubric import BaseRubric
from patient_agent_bench.eval.constants import MIN_SCORE, MAX_SCORE, PASS_THRESHOLD
from patient_agent_bench.eval.task_completion import TaskCompletionRubric
from patient_agent_bench.eval.clinical_safety import ClinicalSafetyRubric
from patient_agent_bench.eval.workflow_accuracy import WorkflowAccuracyRubric
from patient_agent_bench.eval.triage_quality import TriageQualityRubric
from patient_agent_bench.eval.clinical_helpfulness import ClinicalHelpfulnessRubric
from patient_agent_bench.eval.conversational_quality import ConversationalQualityRubric


@pytest.fixture
def mock_bedrock_llm():
    """Mock create_chat_model and create_bedrock_client_with_role to avoid AWS calls."""
    with patch("patient_agent_bench.eval.base_rubric.create_bedrock_client_with_role") as mock_client:
        mock_client.return_value = MagicMock()
        with patch("patient_agent_bench.eval.base_rubric.create_chat_model") as mock_factory:
            mock_llm = MagicMock()
            mock_factory.return_value = mock_llm
            yield mock_llm


# =============================================================================
# RUBRIC REGISTRY - Add new rubrics here for automatic testing
# =============================================================================

RUBRIC_CLASSES = [
    TaskCompletionRubric,
    ClinicalSafetyRubric,
    WorkflowAccuracyRubric,
    TriageQualityRubric,
    ClinicalHelpfulnessRubric,
    ConversationalQualityRubric,
]


# =============================================================================
# Template Tests - Applied to ALL rubrics
# =============================================================================

class TestRubricRegistry:
    """Tests to ensure all rubrics are properly registered and configured."""

    def test_all_rubrics_have_unique_names(self, model_config, mock_bedrock_llm):
        """Verify all rubrics have unique RUBRIC_NAME values."""
        names = []
        for rubric_class in RUBRIC_CLASSES:
            rubric = rubric_class(model_config)
            names.append(rubric.RUBRIC_NAME)

        assert len(names) == len(set(names)), f"Duplicate rubric names found: {names}"

    def test_all_rubrics_have_evaluation_prompt(self, model_config, mock_bedrock_llm):
        """Verify all rubrics have non-empty EVALUATION_PROMPT."""
        for rubric_class in RUBRIC_CLASSES:
            rubric = rubric_class(model_config)
            assert rubric.EVALUATION_PROMPT, f"{rubric_class.__name__} has empty EVALUATION_PROMPT"
            assert len(rubric.EVALUATION_PROMPT) > 100, \
                f"{rubric_class.__name__} EVALUATION_PROMPT too short"

    def test_all_rubrics_inherit_from_base(self):
        """Verify all rubrics inherit from BaseRubric."""
        for rubric_class in RUBRIC_CLASSES:
            assert issubclass(rubric_class, BaseRubric), \
                f"{rubric_class.__name__} must inherit from BaseRubric"


@pytest.mark.parametrize("rubric_class", RUBRIC_CLASSES)
class TestRubricTemplate:
    """Template tests applied to each rubric class."""

    def test_rubric_initialization(self, rubric_class, model_config, mock_bedrock_llm):
        """Test rubric can be initialized."""
        rubric = rubric_class(model_config)

        assert rubric.model_config == model_config
        assert rubric.RUBRIC_NAME is not None
        assert rubric.name == rubric.RUBRIC_NAME

    def test_rubric_format_conversation(
        self, rubric_class, model_config, sample_conversation, mock_bedrock_llm
    ):
        """Test conversation formatting with XML structure."""
        rubric = rubric_class(model_config)
        formatted = rubric.format_conversation(sample_conversation)

        assert isinstance(formatted, str)
        assert '<message index="' in formatted
        assert 'role="user"' in formatted or 'role="assistant"' in formatted
        assert "refill" in formatted.lower()  # From sample conversation

    def test_rubric_prepare_prompt(
        self, rubric_class, model_config, sample_conversation, mock_bedrock_llm
    ):
        """Test prompt preparation includes all required elements."""
        rubric = rubric_class(model_config)
        prompt = rubric.prepare_prompt(
            conversation_history=sample_conversation,
            user_profile="<profile>Test Patient</profile>",
            scenario="Test scenario description",
        )

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # Should contain conversation content
        assert "refill" in prompt.lower() or "medication" in prompt.lower()

    def test_rubric_parse_response_valid_json(self, rubric_class, model_config, mock_bedrock_llm):
        """Test parsing valid JSON response."""
        rubric = rubric_class(model_config)

        response = json.dumps({
            "score": 2,
            "explanation": "All criteria met.",
            "sub_scores": {"criterion_1": 2},
        })

        result = rubric.parse_response(response)

        assert result["score"] == 2
        assert result["explanation"] == "All criteria met."

    def test_rubric_parse_response_invalid_json(self, rubric_class, model_config, mock_bedrock_llm):
        """Test parsing falls back gracefully for invalid JSON."""
        rubric = rubric_class(model_config)

        response = "This is not JSON. Score: 2. The conversation passed."
        result = rubric.parse_response(response)

        assert "score" in result
        assert isinstance(result["score"], int)
        assert MIN_SCORE <= result["score"] <= MAX_SCORE

    def test_rubric_evaluate_with_mock_llm(
        self, rubric_class, model_config, sample_conversation, mock_bedrock_llm
    ):
        """Test full evaluation with mocked LLM."""
        rubric = rubric_class(model_config)

        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "score": 2,
            "explanation": "Evaluation passed.",
            "sub_scores": {},
        })
        mock_bedrock_llm.invoke.return_value = mock_response

        result = rubric.evaluate(
            conversation_history=sample_conversation,
            user_profile="<profile>Test</profile>",
            scenario="Test scenario",
        )

        assert result["score"] == 2
        assert result["rubric_name"] == rubric.RUBRIC_NAME
        assert "raw_response" in result


# =============================================================================
# BaseRubric Specific Tests
# =============================================================================

class TestBaseRubricParsing:
    """Tests for BaseRubric parsing logic."""

    def test_validate_result_clamps_score(self, model_config, mock_bedrock_llm):
        """Test score is clamped to MIN_SCORE-MAX_SCORE range."""
        rubric = TaskCompletionRubric(model_config)

        # Score too high
        result = rubric._validate_result({"score": MAX_SCORE + 1})
        assert result["score"] == MAX_SCORE

        # Score too low
        result = rubric._validate_result({"score": MIN_SCORE - 1})
        assert result["score"] == MIN_SCORE

    def test_validate_result_handles_non_numeric(self, model_config, mock_bedrock_llm):
        """Test handling of non-numeric scores."""
        rubric = TaskCompletionRubric(model_config)

        result = rubric._validate_result({"score": "high"})
        assert result["score"] == PASS_THRESHOLD  # Default

    def test_extract_score_from_text_patterns(self, model_config, mock_bedrock_llm):
        """Test score extraction from various text patterns."""
        rubric = TaskCompletionRubric(model_config)

        # Numeric patterns within valid range
        assert rubric._extract_score_from_text("Score: 5") == MAX_SCORE
        assert rubric._extract_score_from_text("Score: 4") == MAX_SCORE - 1
        assert rubric._extract_score_from_text("Score: 3") == PASS_THRESHOLD
        assert rubric._extract_score_from_text("Score: 2") == MIN_SCORE + 1
        assert rubric._extract_score_from_text("Score: 1") == MIN_SCORE

        # Label patterns (mapped to 1-5 scale)
        assert rubric._extract_score_from_text("EXCELLENT") == MAX_SCORE
        assert rubric._extract_score_from_text("GOOD") == MAX_SCORE - 1
        assert rubric._extract_score_from_text("ADEQUATE") == PASS_THRESHOLD
        assert rubric._extract_score_from_text("POOR") == MIN_SCORE + 1
        assert rubric._extract_score_from_text("FAIL") == MIN_SCORE

        # No match defaults to PASS_THRESHOLD
        assert rubric._extract_score_from_text("No clear score") == PASS_THRESHOLD

    def test_parse_response_extracts_json_from_text(self, model_config, mock_bedrock_llm):
        """Test JSON extraction from mixed text response."""
        rubric = TaskCompletionRubric(model_config)

        response = '''Here is my evaluation:

        {"score": 2, "explanation": "Good job", "sub_scores": {}}

        That concludes my review.'''

        result = rubric.parse_response(response)
        assert result["score"] == 2
        assert result["explanation"] == "Good job"


# =============================================================================
# Individual Rubric Specific Tests (if needed)
# =============================================================================

class TestClinicalSafetyRubric:
    """Specific tests for ClinicalSafetyRubric."""

    def test_safety_rubric_name(self, model_config, mock_bedrock_llm):
        """Verify clinical safety rubric has correct name."""
        rubric = ClinicalSafetyRubric(model_config)
        assert rubric.RUBRIC_NAME == "clinical_safety"

    def test_safety_prompt_contains_safety_keywords(self, model_config, mock_bedrock_llm):
        """Verify safety rubric prompt focuses on safety concerns."""
        rubric = ClinicalSafetyRubric(model_config)
        prompt_lower = rubric.EVALUATION_PROMPT.lower()

        # Should mention safety-related concepts
        safety_keywords = ["safety", "harm", "risk", "emergency", "dangerous"]
        assert any(kw in prompt_lower for kw in safety_keywords), \
            "Clinical safety rubric should mention safety concepts"


class TestTaskCompletionRubric:
    """Specific tests for TaskCompletionRubric."""

    def test_task_completion_rubric_name(self, model_config, mock_bedrock_llm):
        """Verify task completion rubric has correct name."""
        rubric = TaskCompletionRubric(model_config)
        assert rubric.RUBRIC_NAME == "task_completion"


class TestConversationalQualityRubric:
    """Specific tests for ConversationalQualityRubric."""

    def test_conversational_quality_rubric_name(self, model_config, mock_bedrock_llm):
        """Verify conversational quality rubric has correct name."""
        rubric = ConversationalQualityRubric(model_config)
        assert rubric.RUBRIC_NAME == "conversational_quality"

    def test_conversational_quality_prompt_contains_quality_keywords(self, model_config, mock_bedrock_llm):
        """Verify conversational quality rubric prompt focuses on quality concepts."""
        rubric = ConversationalQualityRubric(model_config)
        prompt_lower = rubric.EVALUATION_PROMPT.lower()

        # Should mention conversational quality concepts
        quality_keywords = ["brevity", "informative", "succinct", "clarity", "concise"]
        assert any(kw in prompt_lower for kw in quality_keywords), \
            "Conversational quality rubric should mention quality concepts"
