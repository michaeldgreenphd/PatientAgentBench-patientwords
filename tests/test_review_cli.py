# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for review CLI subcommands (review-annotate and align-check)."""

import json
import re
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from patient_agent_bench.review.models import (
    ANNOTATION_DIMENSIONS,
    RUBRIC_DIMENSIONS,
)
from patient_agent_bench.run import setup_parser


# =============================================================================
# Helpers
# =============================================================================


def _make_evaluation_entry(case_id: str, aggregate_score: float) -> dict:
    rubric_results = {}
    for dim in RUBRIC_DIMENSIONS:
        rubric_results[dim] = {
            "score": max(1, min(5, round(aggregate_score))),
            "pass": aggregate_score >= 3.0,
            "score_std": 0.0,
            "pass_agreement": 1.0,
            "explanation": f"Score for {dim}",
        }
    return {
        "case_id": case_id,
        "conversation": [
            {"type": "human", "content": "Hello"},
            {"type": "ai", "content": "Hi there"},
        ],
        "user_profile": "<patient_profile>test</patient_profile>",
        "scenario": "Test scenario",
        "num_turns": 2,
        "evaluation": {
            "rubric_results": rubric_results,
            "rubric_scores": {
                dim: round(aggregate_score) for dim in RUBRIC_DIMENSIONS
            },
            "aggregate_score": aggregate_score,
            "aggregate_score_std": 0.0,
            "summary": "Test",
        },
    }


def _make_experiment_config(experiment_id: str, label: str = "TestModel") -> dict:
    return {
        "experiment_id": experiment_id,
        "assistant_agent": {
            "model": {"model": "test"},
            "prompt": "default_prompt",
            "agent_class": "default",
            "label": label,
        },
        "user_agent": {
            "model": {"model": "test"},
            "prompt": "default_prompt",
            "agent_class": "default",
            "label": None,
        },
    }


def _create_mock_run_dir(base: Path, num_cases: int = 3) -> Path:
    """Create a mock run directory with one experiment."""
    run_dir = base / "run"
    run_dir.mkdir()
    exp_dir = run_dir / "0_0"
    exp_dir.mkdir()

    config = _make_experiment_config("0_0", label="TestModel")
    (exp_dir / "experiment_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )

    evals = [
        _make_evaluation_entry(f"case_{i}", 2.0 + i)
        for i in range(num_cases)
    ]
    (exp_dir / "evaluations.json").write_text(
        json.dumps(evals), encoding="utf-8"
    )
    return run_dir


def _create_mock_annotation_file(base: Path, name: str = "ann.json") -> Path:
    """Create a mock annotation JSON file with human annotations."""
    data = {
        "metadata": {
            "source_run_dir": "output/test/",
            "sampling_timestamp": "20260310_143022",
            "sample_count": 1,
            "random_seed": 42,
            "experiments_sampled": ["0_0"],
        },
        "conversations": [
            {
                "case_id": "c1",
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
                "patient_profile": "<p/>",
                "scenario": "Test",
                "num_turns": 2,
                "llm_scores": {
                    dim: {"score": 4, "explanation": "ok"}
                    for dim in RUBRIC_DIMENSIONS
                },
                "aggregate_score": 4.0,
                "annotations": [
                    {
                        "annotator_id": "r1",
                        "scores": {
                            dim: 4 for dim in ANNOTATION_DIMENSIONS
                        },
                        "comment": "",
                        "timestamp": "20260310_150000",
                    }
                ],
            },
            {
                "case_id": "c2",
                "experiment": {
                    "experiment_id": "0_0",
                    "assistant_label": "TestModel",
                    "user_label": "DefaultUser",
                    "agent_class": "default",
                },
                "conversation": [
                    {"type": "human", "content": "Hi"},
                    {"type": "ai", "content": "Hello"},
                ],
                "patient_profile": "<p/>",
                "scenario": "Test2",
                "num_turns": 2,
                "llm_scores": {
                    dim: {"score": 3, "explanation": "ok"}
                    for dim in RUBRIC_DIMENSIONS
                },
                "aggregate_score": 3.0,
                "annotations": [
                    {
                        "annotator_id": "r1",
                        "scores": {
                            dim: 3 for dim in ANNOTATION_DIMENSIONS
                        },
                        "comment": "",
                        "timestamp": "20260310_150001",
                    }
                ],
            },
        ],
    }
    fpath = base / name
    fpath.write_text(json.dumps(data), encoding="utf-8")
    return fpath


# =============================================================================
# Property 11: Output File Naming Convention
# =============================================================================
# Feature: review-annotation, Property 11: Output File Naming Convention


@given(
    ts=st.from_regex(r"[0-9]{8}_[0-9]{6}", fullmatch=True),
)
@settings(max_examples=100)
def test_output_file_naming_convention(ts: str):
    """Filenames match expected patterns for any valid timestamp."""
    ann_name = f"annotations_{ts}.json"
    html_name = f"review_{ts}.html"
    report_name = f"alignment_report_{ts}.json"

    assert re.fullmatch(
        r"annotations_\d{8}_\d{6}\.json", ann_name
    ), f"Bad annotation name: {ann_name}"
    assert re.fullmatch(
        r"review_\d{8}_\d{6}\.html", html_name
    ), f"Bad HTML name: {html_name}"
    assert re.fullmatch(
        r"alignment_report_\d{8}_\d{6}\.json", report_name
    ), f"Bad report name: {report_name}"

    # All share the same timestamp
    ann_ts = ann_name.split("_", 1)[1].rsplit(".", 1)[0]
    html_ts = html_name.split("_", 1)[1].rsplit(".", 1)[0]
    assert ann_ts == html_ts == ts


# =============================================================================
# Unit tests for CLI integration
# =============================================================================


class TestReviewAnnotateArgParsing:
    def test_review_annotate_required_args(self):
        parser = setup_parser()
        args = parser.parse_args([
            "review-annotate", "--run-dir", "output/test_run"
        ])
        assert args.command == "review-annotate"
        assert args.run_dir == "output/test_run"
        assert args.num_samples == 20
        assert args.seed is None

    def test_review_annotate_all_args(self):
        parser = setup_parser()
        args = parser.parse_args([
            "review-annotate",
            "--run-dir", "output/test_run",
            "--num-samples", "50",
            "--seed", "42",
            "--output-dir", "/tmp/review_out",
        ])
        assert args.num_samples == 50
        assert args.seed == 42
        assert args.output_dir == "/tmp/review_out"

    def test_review_annotate_missing_run_dir_fails(self):
        parser = setup_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["review-annotate"])


class TestAlignCheckArgParsing:
    def test_align_check_required_args(self):
        parser = setup_parser()
        args = parser.parse_args([
            "align-check",
            "--annotation-files", "a.json", "b.json",
        ])
        assert args.command == "align-check"
        assert args.annotation_files == ["a.json", "b.json"]
        assert args.output is None

    def test_align_check_with_output(self):
        parser = setup_parser()
        args = parser.parse_args([
            "align-check",
            "--annotation-files", "a.json",
            "--output", "report.json",
        ])
        assert args.output == "report.json"

    def test_align_check_missing_files_fails(self):
        """align-check requires either --annotation-files or --annotation-dir."""
        parser = setup_parser()
        # No longer raises SystemExit since both args are optional;
        # the command itself prints an error and returns
        args = parser.parse_args(["align-check"])
        assert args.annotation_files is None
        assert args.annotation_dir is None


class TestReviewAnnotateEndToEnd:
    def test_generates_json_and_html(self):
        """End-to-end: review-annotate produces JSON + HTML files."""
        from patient_agent_bench.run import cmd_review_annotate

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = _create_mock_run_dir(tmp_path, num_cases=5)
            out_dir = tmp_path / "review_out"

            # Build a namespace mimicking parsed args
            class Args:
                pass

            args = Args()
            args.run_dir = str(run_dir)
            args.num_samples = 3
            args.seed = 42
            args.output_dir = str(out_dir)

            cmd_review_annotate(args)

            # Check output files exist
            files = list(out_dir.iterdir())
            json_files = [f for f in files if f.suffix == ".json"]
            html_files = [f for f in files if f.suffix == ".html"]
            assert len(json_files) == 1
            assert len(html_files) == 1

            # Validate JSON content
            data = json.loads(json_files[0].read_text(encoding="utf-8"))
            assert "metadata" in data
            assert "conversations" in data
            assert len(data["conversations"]) == 3

            # Validate HTML content
            html = html_files[0].read_text(encoding="utf-8")
            assert "<html" in html

    def test_default_output_dir(self):
        """Default output goes to review/ inside run-dir."""
        from patient_agent_bench.run import cmd_review_annotate

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = _create_mock_run_dir(tmp_path, num_cases=2)

            class Args:
                pass

            args = Args()
            args.run_dir = str(run_dir)
            args.num_samples = 2
            args.seed = 1
            args.output_dir = None

            cmd_review_annotate(args)

            review_dir = run_dir / "review"
            assert review_dir.exists()
            assert any(f.suffix == ".json" for f in review_dir.iterdir())
            assert any(f.suffix == ".html" for f in review_dir.iterdir())


class TestAlignCheckEndToEnd:
    def test_prints_report(self, capsys):
        """End-to-end: align-check prints agreement report."""
        from patient_agent_bench.run import cmd_align_check

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ann_file = _create_mock_annotation_file(tmp_path)

            class Args:
                pass

            args = Args()
            args.annotation_files = [str(ann_file)]
            args.annotation_dir = None
            args.output = None

            cmd_align_check(args)

            captured = capsys.readouterr()
            assert "Alignment Report" in captured.out
            assert "clinical_safety" in captured.out

    def test_saves_json_report(self):
        """align-check saves JSON when --output provided."""
        from patient_agent_bench.run import cmd_align_check

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ann_file = _create_mock_annotation_file(tmp_path)
            out_path = tmp_path / "report.json"

            class Args:
                pass

            args = Args()
            args.annotation_files = [str(ann_file)]
            args.annotation_dir = None
            args.output = str(out_path)

            cmd_align_check(args)

            assert out_path.exists()
            data = json.loads(out_path.read_text(encoding="utf-8"))
            assert "per_rubric" in data
