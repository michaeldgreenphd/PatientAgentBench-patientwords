# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for review annotation data models."""

import json

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from patient_agent_bench.review.models import (
    ANNOTATION_DIMENSIONS,
    REALISM_DIMENSIONS,
    RUBRIC_DIMENSIONS,
    AnnotationFile,
    AnnotationMetadata,
    ConversationMessage,
    ExperimentInfo,
    HumanAnnotation,
    RubricScore,
    SampledConversation,
)

# =============================================================================
# Hypothesis strategies for generating model instances
# =============================================================================

st_score = st.integers(min_value=1, max_value=5)
st_nonempty_str = st.text(min_size=1, max_size=50).filter(lambda s: s.strip())
st_timestamp = st.from_regex(r"[0-9]{8}_[0-9]{6}", fullmatch=True)


@st.composite
def st_rubric_score(draw):
    return RubricScore(score=draw(st_score), explanation=draw(st.text(max_size=100)))


@st.composite
def st_conversation_message(draw):
    return ConversationMessage(
        type=draw(st.sampled_from(["human", "ai", "tool"])),
        content=draw(st.text(max_size=200)),
    )


@st.composite
def st_experiment_info(draw):
    return ExperimentInfo(
        experiment_id=draw(st_nonempty_str),
        assistant_label=draw(st_nonempty_str),
        user_label=draw(st_nonempty_str),
        agent_class=draw(st_nonempty_str),
    )


@st.composite
def st_human_annotation(draw):
    scores = {dim: draw(st_score) for dim in ANNOTATION_DIMENSIONS}
    return HumanAnnotation(
        annotator_id=draw(st_nonempty_str),
        scores=scores,
        comment=draw(st.text(max_size=200)),
        timestamp=draw(st_timestamp),
    )


@st.composite
def st_sampled_conversation(draw):
    llm_scores = {dim: draw(st_rubric_score()) for dim in RUBRIC_DIMENSIONS}
    annotations = draw(st.lists(st_human_annotation(), max_size=3))
    messages = draw(st.lists(st_conversation_message(), min_size=1, max_size=10))
    return SampledConversation(
        case_id=draw(st_nonempty_str),
        experiment=draw(st_experiment_info()),
        conversation=messages,
        patient_profile=draw(st.text(max_size=500)),
        scenario=draw(st.text(max_size=200)),
        num_turns=draw(st.integers(min_value=1, max_value=30)),
        llm_scores=llm_scores,
        aggregate_score=draw(st.floats(min_value=1.0, max_value=5.0, allow_nan=False)),
        annotations=annotations,
    )


@st.composite
def st_annotation_file(draw):
    convos = draw(st.lists(st_sampled_conversation(), min_size=0, max_size=5))
    metadata = AnnotationMetadata(
        source_run_dir=draw(st_nonempty_str),
        sampling_timestamp=draw(st_timestamp),
        sample_count=len(convos),
        random_seed=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=99999))),
        experiments_sampled=draw(st.lists(st_nonempty_str, min_size=0, max_size=5)),
    )
    return AnnotationFile(metadata=metadata, conversations=convos)


# =============================================================================
# Property 5: Annotation JSON Serialization Round-Trip
# =============================================================================
# Feature: review-annotation, Property 5: Annotation JSON Serialization Round-Trip


@given(annotation_file=st_annotation_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_annotation_json_round_trip(annotation_file: AnnotationFile):
    """Serializing to JSON and deserializing back produces an equivalent object."""
    json_str = annotation_file.model_dump_json()
    restored = AnnotationFile.model_validate_json(json_str)
    assert restored == annotation_file


@given(annotation_file=st_annotation_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_annotation_dict_round_trip(annotation_file: AnnotationFile):
    """Serializing to dict/JSON and back via dict produces an equivalent object."""
    d = annotation_file.model_dump()
    json_str = json.dumps(d)
    restored = AnnotationFile.model_validate(json.loads(json_str))
    assert restored == annotation_file


# =============================================================================
# Property 8: Annotation Score Validation
# =============================================================================
# Feature: review-annotation, Property 8: Annotation Score Validation


@given(
    scores=st.fixed_dictionaries(
        {dim: st.integers(min_value=-10, max_value=10) for dim in ANNOTATION_DIMENSIONS}
    ),
    annotator_id=st.text(max_size=50),
)
@settings(max_examples=100)
def test_annotation_score_validation(scores: dict, annotator_id: str):
    """Validation accepts iff all scores in [1,5] and annotator_id non-empty."""
    all_scores_valid = all(1 <= s <= 5 for s in scores.values())
    id_valid = bool(annotator_id.strip())
    should_accept = all_scores_valid and id_valid

    try:
        HumanAnnotation(
            annotator_id=annotator_id,
            scores=scores,
            comment="test",
            timestamp="20260310_120000",
        )
        accepted = True
    except (ValueError, Exception):
        accepted = False

    assert accepted == should_accept, (
        f"Expected {'accept' if should_accept else 'reject'} but got "
        f"{'accept' if accepted else 'reject'} for scores={scores}, "
        f"annotator_id={annotator_id!r}"
    )


# =============================================================================
# Unit tests for data models
# =============================================================================


class TestAnnotationDimensions:
    def test_rubric_dimensions_count(self):
        assert len(RUBRIC_DIMENSIONS) == 6

    def test_realism_dimensions_count(self):
        assert len(REALISM_DIMENSIONS) == 2

    def test_annotation_dimensions_count(self):
        assert len(ANNOTATION_DIMENSIONS) == 8

    def test_annotation_dimensions_contains_all_expected(self):
        expected = {
            "clinical_safety",
            "triage_quality",
            "task_completion",
            "workflow_accuracy",
            "clinical_helpfulness",
            "conversational_quality",
            "patient_profile_realism",
            "conversation_realism",
        }
        assert set(ANNOTATION_DIMENSIONS) == expected


class TestAnnotationFileSerialization:
    def test_serialize_known_data(self):
        af = AnnotationFile(
            metadata=AnnotationMetadata(
                source_run_dir="output/run_20260310/",
                sampling_timestamp="20260310_143022",
                sample_count=1,
                random_seed=42,
                experiments_sampled=["0_0"],
            ),
            conversations=[
                SampledConversation(
                    case_id="refill_001",
                    experiment=ExperimentInfo(
                        experiment_id="0_0",
                        assistant_label="Test Model",
                        user_label="Default User",
                        agent_class="default",
                    ),
                    conversation=[
                        ConversationMessage(type="human", content="Hello"),
                        ConversationMessage(type="ai", content="Hi there"),
                    ],
                    patient_profile="<patient_profile>test</patient_profile>",
                    scenario="Test scenario",
                    num_turns=2,
                    llm_scores={
                        dim: RubricScore(score=4, explanation="Good")
                        for dim in RUBRIC_DIMENSIONS
                    },
                    aggregate_score=4.0,
                    annotations=[],
                )
            ],
        )
        data = af.model_dump()
        assert data["metadata"]["source_run_dir"] == "output/run_20260310/"
        assert data["metadata"]["random_seed"] == 42
        assert len(data["conversations"]) == 1
        assert data["conversations"][0]["case_id"] == "refill_001"
        assert data["conversations"][0]["annotations"] == []

        # Round-trip
        restored = AnnotationFile.model_validate(data)
        assert restored == af


class TestHumanAnnotationValidation:
    def test_rejects_out_of_range_score_zero(self):
        with pytest.raises(ValueError):
            HumanAnnotation(
                annotator_id="reviewer_1",
                scores={dim: 0 for dim in ANNOTATION_DIMENSIONS},
                timestamp="20260310_120000",
            )

    def test_rejects_out_of_range_score_six(self):
        with pytest.raises(ValueError):
            HumanAnnotation(
                annotator_id="reviewer_1",
                scores={dim: 6 for dim in ANNOTATION_DIMENSIONS},
                timestamp="20260310_120000",
            )

    def test_rejects_empty_annotator_id(self):
        with pytest.raises(ValueError):
            HumanAnnotation(
                annotator_id="",
                scores={dim: 3 for dim in ANNOTATION_DIMENSIONS},
                timestamp="20260310_120000",
            )

    def test_rejects_whitespace_only_annotator_id(self):
        with pytest.raises(ValueError):
            HumanAnnotation(
                annotator_id="   ",
                scores={dim: 3 for dim in ANNOTATION_DIMENSIONS},
                timestamp="20260310_120000",
            )

    def test_accepts_valid_annotation(self):
        ann = HumanAnnotation(
            annotator_id="reviewer_1",
            scores={dim: 3 for dim in ANNOTATION_DIMENSIONS},
            comment="Looks good",
            timestamp="20260310_120000",
        )
        assert ann.annotator_id == "reviewer_1"
        assert all(s == 3 for s in ann.scores.values())

    def test_default_annotations_empty(self):
        conv = SampledConversation(
            case_id="test",
            experiment=ExperimentInfo(
                experiment_id="0_0",
                assistant_label="M",
                user_label="U",
                agent_class="default",
            ),
            conversation=[ConversationMessage(type="human", content="hi")],
            patient_profile="<p/>",
            scenario="s",
            num_turns=1,
            llm_scores={
                dim: RubricScore(score=3, explanation="ok")
                for dim in RUBRIC_DIMENSIONS
            },
            aggregate_score=3.0,
        )
        assert conv.annotations == []
