# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for the alignment analyzer (inter-rater agreement)."""

import json
import logging
import tempfile
from pathlib import Path
from typing import List

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from patient_agent_bench.review.alignment import (
    AlignmentAnalyzer,
    AlignmentReport,
    cohens_weighted_kappa,
    pearson_correlation,
    exact_agreement_rate,
    adjacent_agreement_rate,
    fleiss_kappa,
)
from patient_agent_bench.review.models import (
    ANNOTATION_DIMENSIONS,
    RUBRIC_DIMENSIONS,
)


# =============================================================================
# Helpers
# =============================================================================

st_score = st.integers(min_value=1, max_value=5)


def _make_annotation_json(
    conversations: list,
    source_run_dir: str = "output/test_run/",
) -> dict:
    """Build a minimal annotation file dict."""
    return {
        "metadata": {
            "source_run_dir": source_run_dir,
            "sampling_timestamp": "20260310_143022",
            "sample_count": len(conversations),
            "random_seed": 42,
            "experiments_sampled": ["0_0"],
        },
        "conversations": conversations,
    }


def _make_conversation_entry(
    case_id: str,
    llm_scores: dict,
    annotations: list = None,
) -> dict:
    """Build a minimal conversation entry for annotation JSON."""
    return {
        "case_id": case_id,
        "experiment": {
            "experiment_id": "0_0",
            "assistant_label": "TestModel",
            "user_label": "DefaultUser",
            "agent_class": "default",
        },
        "conversation": [
            {"type": "human", "content": "Hello"},
            {"type": "ai", "content": "Hi"},
        ],
        "patient_profile": "<patient_profile>test</patient_profile>",
        "scenario": "Test scenario",
        "num_turns": 2,
        "llm_scores": {
            dim: {"score": llm_scores.get(dim, 3), "explanation": "ok"}
            for dim in RUBRIC_DIMENSIONS
        },
        "aggregate_score": 3.0,
        "annotations": annotations or [],
    }


def _make_annotation(
    annotator_id: str,
    scores: dict,
) -> dict:
    """Build a minimal human annotation dict."""
    full_scores = {dim: 3 for dim in ANNOTATION_DIMENSIONS}
    full_scores.update(scores)
    return {
        "annotator_id": annotator_id,
        "scores": full_scores,
        "comment": "",
        "timestamp": "20260310_150000",
    }


def _write_annotation_file(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


# =============================================================================
# Property 9: Agreement Metrics Bounds
# =============================================================================
# Feature: review-annotation, Property 9: Agreement Metrics Bounds


@given(
    scores_a=st.lists(st_score, min_size=2, max_size=50),
    scores_b=st.lists(st_score, min_size=2, max_size=50),
)
@settings(max_examples=100)
def test_agreement_metrics_bounds(scores_a: List[int], scores_b: List[int]):
    """Kappa in [-1,1], Pearson in [-1,1], exact/adjacent in [0,100]."""
    # Ensure same length
    min_len = min(len(scores_a), len(scores_b))
    a = scores_a[:min_len]
    b = scores_b[:min_len]

    kappa = cohens_weighted_kappa(a, b)
    if kappa is not None:
        assert -1.0 - 1e-9 <= kappa <= 1.0 + 1e-9, (
            f"Kappa {kappa} out of bounds"
        )

    pearson = pearson_correlation(a, b)
    if pearson is not None:
        assert -1.0 - 1e-9 <= pearson <= 1.0 + 1e-9, (
            f"Pearson {pearson} out of bounds"
        )

    exact = exact_agreement_rate(a, b)
    assert 0.0 <= exact <= 100.0, f"Exact {exact} out of bounds"

    adjacent = adjacent_agreement_rate(a, b)
    assert 0.0 <= adjacent <= 100.0, f"Adjacent {adjacent} out of bounds"


# =============================================================================
# Property 10: Adjacent Agreement >= Exact Agreement
# =============================================================================
# Feature: review-annotation, Property 10: Adjacent Agreement >= Exact Agreement


@given(
    scores_a=st.lists(st_score, min_size=1, max_size=50),
    scores_b=st.lists(st_score, min_size=1, max_size=50),
)
@settings(max_examples=100)
def test_adjacent_gte_exact(scores_a: List[int], scores_b: List[int]):
    """Adjacent agreement rate >= exact agreement rate."""
    min_len = min(len(scores_a), len(scores_b))
    a = scores_a[:min_len]
    b = scores_b[:min_len]

    exact = exact_agreement_rate(a, b)
    adjacent = adjacent_agreement_rate(a, b)
    assert adjacent >= exact - 1e-9, (
        f"Adjacent {adjacent} < Exact {exact}"
    )


# =============================================================================
# Unit tests for alignment analyzer
# =============================================================================


class TestStatisticalFunctions:
    def test_known_kappa_perfect_agreement(self):
        """Identical arrays -> Kappa = 1."""
        a = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        result = cohens_weighted_kappa(a, a)
        assert result is not None
        assert abs(result - 1.0) < 1e-9

    def test_known_pearson_perfect_correlation(self):
        """Identical arrays -> Pearson = 1."""
        a = [1, 2, 3, 4, 5]
        result = pearson_correlation(a, a)
        assert result is not None
        assert abs(result - 1.0) < 1e-9

    def test_known_pearson_negative_correlation(self):
        """Reversed arrays -> Pearson = -1."""
        a = [1, 2, 3, 4, 5]
        b = [5, 4, 3, 2, 1]
        result = pearson_correlation(a, b)
        assert result is not None
        assert abs(result - (-1.0)) < 1e-9

    def test_exact_agreement_perfect(self):
        a = [3, 4, 5]
        assert abs(exact_agreement_rate(a, a) - 100.0) < 1e-9

    def test_exact_agreement_none(self):
        a = [1, 2, 3]
        b = [4, 5, 4]
        assert abs(exact_agreement_rate(a, b) - 0.0) < 1e-9

    def test_adjacent_agreement_perfect(self):
        a = [3, 4, 5]
        assert abs(adjacent_agreement_rate(a, a) - 100.0) < 1e-9

    def test_adjacent_agreement_within_one(self):
        a = [3, 3, 3]
        b = [2, 3, 4]
        assert abs(adjacent_agreement_rate(a, b) - 100.0) < 1e-9

    def test_constant_arrays_pearson_none(self):
        """Constant arrays have zero variance -> Pearson returns None."""
        a = [3, 3, 3, 3]
        b = [3, 3, 3, 3]
        assert pearson_correlation(a, b) is None

    def test_constant_arrays_kappa_handles_gracefully(self):
        """Constant identical arrays -> Kappa returns None (pe=1)."""
        a = [3, 3, 3, 3]
        result = cohens_weighted_kappa(a, a)
        # When all scores are the same, pe == 1, so kappa is undefined
        assert result is None

    def test_too_short_returns_none(self):
        assert cohens_weighted_kappa([1], [2]) is None
        assert pearson_correlation([1], [2]) is None

    def test_empty_returns_zero(self):
        assert exact_agreement_rate([], []) == 0.0
        assert adjacent_agreement_rate([], []) == 0.0

    def test_fleiss_kappa_perfect_agreement(self):
        """All raters agree on every subject -> Fleiss = 1."""
        # 5 subjects, 3 raters each, agreeing on different categories
        matrix = [
            [3, 0, 0, 0, 0],  # all rate category 1
            [0, 3, 0, 0, 0],  # all rate category 2
            [0, 0, 3, 0, 0],  # all rate category 3
            [0, 0, 0, 3, 0],  # all rate category 4
            [0, 0, 0, 0, 3],  # all rate category 5
        ]
        result = fleiss_kappa(matrix)
        assert result is not None
        assert abs(result - 1.0) < 1e-9

    def test_fleiss_kappa_too_few_raters(self):
        matrix = [[1, 0, 0, 0, 0]]
        assert fleiss_kappa(matrix) is None


class TestAlignmentAnalyzer:
    def test_basic_agreement(self):
        """Analyzer computes metrics for a simple case."""
        llm = {dim: 4 for dim in RUBRIC_DIMENSIONS}
        ann = _make_annotation("r1", {dim: 4 for dim in RUBRIC_DIMENSIONS})
        ann2 = _make_annotation("r1", {dim: 3 for dim in RUBRIC_DIMENSIONS})
        conv1 = _make_conversation_entry("c1", llm, [ann])
        conv2 = _make_conversation_entry("c2", llm, [ann2])
        data = _make_annotation_json([conv1, conv2])

        with tempfile.TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "ann.json"
            _write_annotation_file(fpath, data)
            analyzer = AlignmentAnalyzer([fpath])
            report = analyzer.compute_agreement()

        assert isinstance(report, AlignmentReport)
        assert report.num_conversations == 2
        assert report.num_annotators == 1
        for dim in RUBRIC_DIMENSIONS:
            m = report.per_rubric[dim]
            assert m.exact_agreement_pct >= 0
            assert m.adjacent_agreement_pct >= 0

    def test_no_annotations_skipped(self, caplog):
        """File with no annotations is skipped with warning."""
        conv = _make_conversation_entry("c1", {}, [])
        data = _make_annotation_json([conv])

        # Need a second file with annotations to avoid ValueError
        ann = _make_annotation("r1", {dim: 3 for dim in RUBRIC_DIMENSIONS})
        conv2 = _make_conversation_entry("c2", {}, [ann])
        data2 = _make_annotation_json([conv2])

        with tempfile.TemporaryDirectory() as tmp:
            f1 = Path(tmp) / "empty.json"
            f2 = Path(tmp) / "good.json"
            _write_annotation_file(f1, data)
            _write_annotation_file(f2, data2)

            with caplog.at_level(logging.WARNING):
                analyzer = AlignmentAnalyzer([f1, f2])

            assert "No human annotations" in caplog.text
            assert len(analyzer._conversations) == 1

    def test_all_empty_raises_value_error(self):
        """All files with no annotations raises ValueError."""
        conv = _make_conversation_entry("c1", {}, [])
        data = _make_annotation_json([conv])

        with tempfile.TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "empty.json"
            _write_annotation_file(fpath, data)
            with pytest.raises(ValueError, match="No annotated"):
                AlignmentAnalyzer([fpath])

    def test_format_report(self):
        """format_report produces readable console output."""
        llm = {dim: 4 for dim in RUBRIC_DIMENSIONS}
        ann = _make_annotation("r1", {dim: 4 for dim in RUBRIC_DIMENSIONS})
        conv = _make_conversation_entry("c1", llm, [ann])
        conv2 = _make_conversation_entry("c2", llm, [ann])
        data = _make_annotation_json([conv, conv2])

        with tempfile.TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "ann.json"
            _write_annotation_file(fpath, data)
            analyzer = AlignmentAnalyzer([fpath])
            report = analyzer.compute_agreement()
            text = AlignmentAnalyzer.format_report(report)

        assert "Alignment Report" in text
        assert "clinical_safety" in text
        assert "LLM vs Human" in text

    def test_save_report(self):
        """save_report writes valid JSON."""
        llm = {dim: 4 for dim in RUBRIC_DIMENSIONS}
        ann = _make_annotation("r1", {dim: 4 for dim in RUBRIC_DIMENSIONS})
        conv = _make_conversation_entry("c1", llm, [ann])
        conv2 = _make_conversation_entry("c2", llm, [ann])
        data = _make_annotation_json([conv, conv2])

        with tempfile.TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "ann.json"
            _write_annotation_file(fpath, data)
            analyzer = AlignmentAnalyzer([fpath])
            report = analyzer.compute_agreement()

            out = Path(tmp) / "report.json"
            AlignmentAnalyzer.save_report(report, out)

            assert out.exists()
            loaded = json.loads(out.read_text(encoding="utf-8"))
            assert "per_rubric" in loaded
            assert "num_conversations" in loaded

    def test_pairwise_with_two_annotators(self):
        """Two annotators produce pairwise metrics."""
        llm = {dim: 4 for dim in RUBRIC_DIMENSIONS}
        a1 = _make_annotation("r1", {dim: 4 for dim in RUBRIC_DIMENSIONS})
        a2 = _make_annotation("r2", {dim: 3 for dim in RUBRIC_DIMENSIONS})
        conv1 = _make_conversation_entry("c1", llm, [a1, a2])
        conv2 = _make_conversation_entry("c2", llm, [a1, a2])
        data = _make_annotation_json([conv1, conv2])

        with tempfile.TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "ann.json"
            _write_annotation_file(fpath, data)
            analyzer = AlignmentAnalyzer([fpath])
            report = analyzer.compute_agreement()

        assert report.num_annotators == 2
        assert len(report.pairwise_annotator_agreement) == 1
        pair_key = "r1_vs_r2"
        assert pair_key in report.pairwise_annotator_agreement
