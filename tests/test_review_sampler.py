# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for the stratified conversation sampler."""

import json
import logging
import tempfile
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from patient_agent_bench.review.models import (
    RUBRIC_DIMENSIONS,
    ConversationMessage,
    ExperimentInfo,
    RubricScore,
    SampledConversation,
)
from patient_agent_bench.review.sampler import ConversationSampler


# =============================================================================
# Helpers to create mock run directories on disk
# =============================================================================


def _make_evaluation_entry(case_id: str, aggregate_score: float) -> dict:
    """Create a minimal evaluations.json entry with varied per-rubric scores."""
    # Use a deterministic hash of case_id to vary scores across dimensions,
    # keeping them centered around the aggregate score.
    import hashlib
    h = int(hashlib.md5(case_id.encode()).hexdigest()[:8], 16)
    rubric_results = {}
    for i, dim in enumerate(RUBRIC_DIMENSIONS):
        # Vary each dim's score by -1 to +1 from aggregate, deterministically
        offset = ((h >> (i * 2)) % 3) - 1  # -1, 0, or +1
        dim_score = max(1, min(5, round(aggregate_score) + offset))
        rubric_results[dim] = {
            "score": dim_score,
            "pass": dim_score >= 3,
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
            "rubric_scores": {dim: rubric_results[dim]["score"] for dim in RUBRIC_DIMENSIONS},
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


def _create_run_dir(
    tmp_path: Path,
    experiments: dict[str, list[float]],
) -> Path:
    """
    Create a mock run directory.

    experiments: mapping of experiment_id -> list of aggregate_scores
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    for exp_id, scores in experiments.items():
        exp_dir = run_dir / exp_id
        exp_dir.mkdir()

        config = _make_experiment_config(exp_id, label=f"Model_{exp_id}")
        (exp_dir / "experiment_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )

        evals = [
            _make_evaluation_entry(f"case_{exp_id}_{i}", score)
            for i, score in enumerate(scores)
        ]
        (exp_dir / "evaluations.json").write_text(
            json.dumps(evals), encoding="utf-8"
        )

    return run_dir


# =============================================================================
# Property 1: Proportional Experiment Representation
# =============================================================================
# Feature: review-annotation, Property 1: Proportional Experiment Representation


@given(
    counts=st.lists(
        st.integers(min_value=1, max_value=30),
        min_size=2,
        max_size=5,
    ),
    seed=st.integers(min_value=0, max_value=99999),
)
@settings(max_examples=100)
def test_proportional_experiment_representation(counts, seed):
    """Multiple experiments are represented when budget is generous."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        experiments = {}
        for i, count in enumerate(counts):
            exp_id = f"{i}_0"
            experiments[exp_id] = [1.0 + (j % 5) for j in range(count)]

        run_dir = _create_run_dir(tmp_path, experiments)
        sampler = ConversationSampler(run_dir, seed=seed)

        total = sum(counts)
        num_samples = min(total, max(len(counts), total // 2))
        assume(num_samples > 0)

        result = sampler.sample(num_samples)

        sampled_exp_ids = set(c.experiment.experiment_id for c in result)

        # With generous budget (>= 2x experiments), multiple experiments
        # should be represented
        if len(counts) >= 2 and num_samples >= 2 * len(counts):
            assert len(sampled_exp_ids) >= 2, (
                f"Only {len(sampled_exp_ids)} experiment(s) represented "
                f"out of {len(counts)} with {num_samples} samples"
            )


# =============================================================================
# Property 2: Score Bucket Coverage
# =============================================================================
# Feature: review-annotation, Property 2: Score Bucket Coverage


@given(
    scores=st.lists(
        st.floats(min_value=1.0, max_value=5.0, allow_nan=False),
        min_size=2,
        max_size=30,
    ),
    seed=st.integers(min_value=0, max_value=99999),
)
@settings(max_examples=100)
def test_score_bucket_coverage(scores, seed):
    """Sampled conversations cover diverse per-rubric score buckets."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        experiments = {"0_0": scores}
        run_dir = _create_run_dir(tmp_path, experiments)
        sampler = ConversationSampler(run_dir, seed=seed)

        num_samples = min(len(scores), max(2, len(scores) // 2))

        result = sampler.sample(num_samples)

        # The sample should contain conversations from more than one
        # per-rubric score bucket (i.e., not all identical scores)
        all_rubric_buckets: set[int] = set()
        for c in result:
            for dim in RUBRIC_DIMENSIONS:
                score = c.llm_scores.get(dim)
                if score and score.score > 0:
                    all_rubric_buckets.add(max(1, min(5, round(score.score))))

        # With varied per-dim scores and num_samples >= 2, we should see
        # at least 2 distinct score buckets across all rubric dimensions
        if num_samples >= 2 and len(scores) >= 2:
            assert len(all_rubric_buckets) >= 2, (
                f"Only {len(all_rubric_buckets)} rubric bucket(s) in sample"
            )


# =============================================================================
# Property 3: Reproducible Sampling
# =============================================================================
# Feature: review-annotation, Property 3: Reproducible Sampling


@given(
    num_convos=st.integers(min_value=2, max_value=20),
    num_samples=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=99999),
)
@settings(max_examples=100)
def test_reproducible_sampling(num_convos, num_samples, seed):
    """Same seed produces identical results."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        scores = [1.0 + (i % 5) for i in range(num_convos)]
        experiments = {"0_0": scores}

        run_dir = _create_run_dir(tmp_path, experiments)

        s1 = ConversationSampler(run_dir, seed=seed)
        r1 = s1.sample(num_samples)

        s2 = ConversationSampler(run_dir, seed=seed)
        r2 = s2.sample(num_samples)

        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.case_id == b.case_id
            assert a.experiment.experiment_id == b.experiment.experiment_id


# =============================================================================
# Property 4: Sample Count Correctness
# =============================================================================
# Feature: review-annotation, Property 4: Sample Count Correctness


@given(
    num_convos=st.integers(min_value=1, max_value=30),
    num_samples=st.integers(min_value=1, max_value=50),
    seed=st.integers(min_value=0, max_value=99999),
)
@settings(max_examples=100)
def test_sample_count_correctness(num_convos, num_samples, seed):
    """Result length is exactly min(K, N)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        scores = [1.0 + (i % 5) for i in range(num_convos)]
        experiments = {"0_0": scores}

        run_dir = _create_run_dir(tmp_path, experiments)
        sampler = ConversationSampler(run_dir, seed=seed)

        result = sampler.sample(num_samples)
        assert len(result) == min(num_samples, num_convos)


# =============================================================================
# Unit tests for sampler edge cases
# =============================================================================


class TestSamplerEdgeCases:
    def test_empty_run_dir_raises_value_error(self, tmp_path):
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        with pytest.raises(ValueError, match="No experiments"):
            ConversationSampler(run_dir)

    def test_nonexistent_run_dir_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ConversationSampler(tmp_path / "nonexistent")

    def test_single_experiment_single_conversation(self, tmp_path):
        run_dir = _create_run_dir(tmp_path, {"0_0": [3.5]})
        sampler = ConversationSampler(run_dir, seed=42)
        result = sampler.sample(1)
        assert len(result) == 1
        assert result[0].case_id == "case_0_0_0"

    def test_all_same_score_bucket(self, tmp_path):
        run_dir = _create_run_dir(tmp_path, {"0_0": [3.0, 3.1, 3.2, 2.8, 2.9]})
        sampler = ConversationSampler(run_dir, seed=42)
        result = sampler.sample(3)
        assert len(result) == 3
        # All should be from bucket 3
        for c in result:
            assert max(1, min(5, round(c.aggregate_score))) == 3

    def test_requested_more_than_available_returns_all(self, tmp_path, caplog):
        run_dir = _create_run_dir(tmp_path, {"0_0": [2.0, 4.0]})
        sampler = ConversationSampler(run_dir, seed=42)
        with caplog.at_level(logging.WARNING):
            result = sampler.sample(100)
        assert len(result) == 2
        assert "only 2 available" in caplog.text

    def test_missing_evaluations_json_skipped(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # Create experiment with config but no evaluations
        exp_dir = run_dir / "0_0"
        exp_dir.mkdir()
        (exp_dir / "experiment_config.json").write_text(
            json.dumps(_make_experiment_config("0_0")), encoding="utf-8"
        )
        # Create another valid experiment
        exp_dir2 = run_dir / "1_0"
        exp_dir2.mkdir()
        (exp_dir2 / "experiment_config.json").write_text(
            json.dumps(_make_experiment_config("1_0")), encoding="utf-8"
        )
        evals = [_make_evaluation_entry("case_1", 4.0)]
        (exp_dir2 / "evaluations.json").write_text(
            json.dumps(evals), encoding="utf-8"
        )

        sampler = ConversationSampler(run_dir, seed=42)
        assert sampler.total_conversations == 1

    def test_experiment_ids_populated(self, tmp_path):
        run_dir = _create_run_dir(tmp_path, {"0_0": [3.0], "1_0": [4.0]})
        sampler = ConversationSampler(run_dir, seed=42)
        assert set(sampler.experiment_ids) == {"0_0", "1_0"}
