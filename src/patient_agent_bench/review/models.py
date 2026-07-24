# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Pydantic data models for the review annotation JSON structure."""

from typing import Dict, List, Optional

from pydantic import BaseModel, field_validator

from patient_agent_bench.eval.clinical_safety import ClinicalSafetyRubric
from patient_agent_bench.eval.triage_quality import TriageQualityRubric
from patient_agent_bench.eval.task_completion import TaskCompletionRubric
from patient_agent_bench.eval.workflow_accuracy import WorkflowAccuracyRubric
from patient_agent_bench.eval.clinical_helpfulness import ClinicalHelpfulnessRubric
from patient_agent_bench.eval.conversational_quality import (
    ConversationalQualityRubric,
)
from patient_agent_bench.eval.patient_profile_realism import (
    PatientProfileRealismDimension,
)
from patient_agent_bench.eval.conversation_realism import (
    ConversationRealismDimension,
)

# The 6 LLM rubric dimensions (shared with existing eval)
RUBRIC_DIMENSIONS = [
    ClinicalSafetyRubric.RUBRIC_NAME,
    TriageQualityRubric.RUBRIC_NAME,
    TaskCompletionRubric.RUBRIC_NAME,
    WorkflowAccuracyRubric.RUBRIC_NAME,
    ClinicalHelpfulnessRubric.RUBRIC_NAME,
    ConversationalQualityRubric.RUBRIC_NAME,
]

# The 2 additional human-only annotation dimensions
REALISM_DIMENSIONS = [
    PatientProfileRealismDimension.RUBRIC_NAME,
    ConversationRealismDimension.RUBRIC_NAME,
]

# All 8 annotation dimensions
ANNOTATION_DIMENSIONS = RUBRIC_DIMENSIONS + REALISM_DIMENSIONS

# Condensed 1-line scoring guides per level, sourced from each class
RUBRIC_GUIDE: Dict[str, List[str]] = {
    ClinicalSafetyRubric.RUBRIC_NAME: ClinicalSafetyRubric.SCORING_GUIDE,
    TriageQualityRubric.RUBRIC_NAME: TriageQualityRubric.SCORING_GUIDE,
    TaskCompletionRubric.RUBRIC_NAME: TaskCompletionRubric.SCORING_GUIDE,
    WorkflowAccuracyRubric.RUBRIC_NAME: WorkflowAccuracyRubric.SCORING_GUIDE,
    ClinicalHelpfulnessRubric.RUBRIC_NAME: ClinicalHelpfulnessRubric.SCORING_GUIDE,
    ConversationalQualityRubric.RUBRIC_NAME: ConversationalQualityRubric.SCORING_GUIDE,
    PatientProfileRealismDimension.RUBRIC_NAME: PatientProfileRealismDimension.SCORING_GUIDE,
    ConversationRealismDimension.RUBRIC_NAME: ConversationRealismDimension.SCORING_GUIDE,
}


class AnnotationMetadata(BaseModel):
    """Top-level metadata for an annotation file."""

    source_run_dir: str
    sampling_timestamp: str
    sample_count: int
    random_seed: Optional[int] = None
    experiments_sampled: List[str]


class ConversationMessage(BaseModel):
    """A single message in a conversation."""

    type: str  # "human", "ai", "tool"
    content: str


class RubricScore(BaseModel):
    """LLM evaluator score for a single rubric dimension."""

    score: float  # 1-5 (may be float from multi-evaluator averaging)
    explanation: str


class ExperimentInfo(BaseModel):
    """Experiment metadata for a sampled conversation."""

    experiment_id: str
    assistant_label: str
    user_label: str
    agent_class: str


class HumanAnnotation(BaseModel):
    """A single human annotator's scores for a conversation."""

    annotator_id: str
    scores: Dict[str, int]  # 8 annotation dimensions, each 1-5
    comments: Dict[str, str] = {}  # per-dimension comments
    comment: str = ""  # legacy single comment field
    timestamp: str = ""
    submitted: bool = True

    @field_validator("annotator_id")
    @classmethod
    def annotator_id_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("annotator_id must be non-empty after stripping whitespace")
        return v

    @field_validator("scores")
    @classmethod
    def scores_valid(cls, v: Dict[str, int]) -> Dict[str, int]:
        for dim, score in v.items():
            if not isinstance(score, int) or score < 1 or score > 5:
                raise ValueError(
                    f"Score for '{dim}' must be an integer in [1, 5], got {score}"
                )
        return v


class SampledConversation(BaseModel):
    """A conversation sampled for human review."""

    case_id: str
    experiment: ExperimentInfo
    conversation: List[ConversationMessage]
    patient_profile: str  # structured XML
    scenario: str
    num_turns: int
    llm_scores: Dict[str, RubricScore]  # 6 rubric dimensions
    aggregate_score: float
    annotations: List[HumanAnnotation] = []


class AnnotationFile(BaseModel):
    """Top-level model for the annotation JSON file."""

    metadata: AnnotationMetadata
    conversations: List[SampledConversation]
