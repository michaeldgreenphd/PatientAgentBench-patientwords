# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for the HTML review generator."""

import html
import re
import tempfile
from pathlib import Path
from typing import List

from hypothesis import given, settings
from hypothesis import strategies as st

from patient_agent_bench.review.html_generator import HTMLGenerator
from patient_agent_bench.review.models import (
    ANNOTATION_DIMENSIONS,
    RUBRIC_DIMENSIONS,
    ConversationMessage,
    ExperimentInfo,
    RubricScore,
    SampledConversation,
)

# =============================================================================
# Helpers
# =============================================================================

st_nonempty = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=30,
).filter(lambda s: s.strip())


def _make_conversation(
    case_id: str = "case_001",
    experiment_id: str = "0_0",
    assistant_label: str = "TestModel",
) -> SampledConversation:
    return SampledConversation(
        case_id=case_id,
        experiment=ExperimentInfo(
            experiment_id=experiment_id,
            assistant_label=assistant_label,
            user_label="DefaultUser",
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
            dim: RubricScore(score=4, explanation="Good") for dim in RUBRIC_DIMENSIONS
        },
        aggregate_score=4.0,
    )


@st.composite
def st_conversations(draw):
    n = draw(st.integers(min_value=1, max_value=5))
    convos = []
    for i in range(n):
        case_id = draw(st_nonempty)
        exp_id = f"{draw(st.integers(min_value=0, max_value=3))}_{draw(st.integers(min_value=0, max_value=3))}"
        label = draw(st_nonempty)
        convos.append(_make_conversation(case_id=case_id, experiment_id=exp_id, assistant_label=label))
    return convos


# =============================================================================
# Property 6: Self-Contained HTML
# =============================================================================
# Feature: review-annotation, Property 6: Self-Contained HTML


@given(convos=st_conversations())
@settings(max_examples=100)
def test_self_contained_html(convos: List[SampledConversation]):
    """No external resource references in generated HTML."""
    gen = HTMLGenerator()
    html = gen.generate(convos, "annotations.json")

    # No external link tags (CSS)
    assert not re.search(r'<link\s[^>]*href=', html), "Found external <link href=>"
    # No external script tags
    assert not re.search(r'<script\s[^>]*src=', html), "Found external <script src=>"
    # No external images
    assert not re.search(r'<img\s[^>]*src="http', html), "Found external <img src=http>"


# =============================================================================
# Property 7: HTML Contains All Conversation Data
# =============================================================================
# Feature: review-annotation, Property 7: HTML Contains All Conversation Data


@given(convos=st_conversations())
@settings(max_examples=100)
def test_html_contains_all_conversation_data(convos: List[SampledConversation]):
    """Each case_id, experiment_id, and assistant label appears in the HTML."""
    gen = HTMLGenerator()
    output = gen.generate(convos, "annotations.json")

    for conv in convos:
        # Data appears in JSON blob and/or HTML-escaped nav items
        assert conv.case_id in output or html.escape(conv.case_id) in output, (
            f"case_id {conv.case_id!r} not found in HTML"
        )
        assert conv.experiment.experiment_id in output or html.escape(conv.experiment.experiment_id) in output, (
            f"experiment_id {conv.experiment.experiment_id!r} not found in HTML"
        )
        assert conv.experiment.assistant_label in output or html.escape(conv.experiment.assistant_label) in output, (
            f"assistant_label {conv.experiment.assistant_label!r} not found in HTML"
        )


# =============================================================================
# Unit tests for HTML generator
# =============================================================================


class TestHTMLGenerator:
    def test_valid_html_structure(self):
        gen = HTMLGenerator()
        html = gen.generate([_make_conversation()], "annotations.json")
        assert "<html" in html
        assert "<head>" in html
        assert "<body" in html
        assert "</html>" in html

    def test_annotation_mode_elements_present(self):
        gen = HTMLGenerator()
        html = gen.generate([_make_conversation()], "annotations.json")
        assert 'id="annotation-toggle"' in html
        assert 'id="annotator-id"' in html
        assert 'id="submit-annotation"' in html
        assert 'id="annotation-panel"' in html
        assert 'dim-comment' in html
        assert 'promptExitAnnotation' in html

    def test_conversation_message_styling(self):
        gen = HTMLGenerator()
        html = gen.generate([_make_conversation()], "annotations.json")
        assert "msg-ai" in html
        assert "msg-human" in html

    def test_llm_scores_panel_present(self):
        gen = HTMLGenerator()
        html = gen.generate([_make_conversation()], "annotations.json")
        assert 'id="llm-scores-panel"' in html
        assert "LLM Evaluator Scores" in html

    def test_download_button_present(self):
        gen = HTMLGenerator()
        html = gen.generate([_make_conversation()], "annotations.json")
        assert "downloadAnnotations" in html
        assert "Download" in html

    def test_patient_profile_panel(self):
        gen = HTMLGenerator()
        html = gen.generate([_make_conversation()], "annotations.json")
        assert "Patient Profile" in html
        assert 'id="patient-profile-content"' in html

    def test_navigation_sidebar(self):
        convos = [
            _make_conversation(case_id="case_A", experiment_id="0_0"),
            _make_conversation(case_id="case_B", experiment_id="1_0"),
        ]
        gen = HTMLGenerator()
        html = gen.generate(convos, "annotations.json")
        assert "case_A" in html
        assert "case_B" in html
        assert 'data-index="0"' in html
        assert 'data-index="1"' in html

    def test_annotation_json_filename_in_html(self):
        gen = HTMLGenerator()
        html = gen.generate([_make_conversation()], "annotations_20260310_143022.json")
        assert "annotations_20260310_143022.json" in html
