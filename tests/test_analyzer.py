# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for benchmark seed analyzer."""

import json
import tempfile
from pathlib import Path

import pytest

from patient_agent_bench.benchmark_seed.analyzer import (
    AttributeDistribution,
    BenchmarkAnalyzer,
    BenchmarkSummary,
    analyze_entries,
    generate_summary_path,
    write_summary,
)
from patient_agent_bench.benchmark_seed.benchmark_entry import (
    BenchmarkSeed,
    EnrichedBenchmarkEntry,
)
from patient_agent_bench.benchmark_seed.config import DistributionConfig


@pytest.fixture
def sample_seeds():
    """Create sample benchmark seeds for testing."""
    return [
        BenchmarkSeed(
            condition_name="Hypertension",
            severity_level="moderate",
            task_category="prescription",
            task_subcategory="renewal",
            age_group="senior",
            gender="male",
            sex="male",
            care_preference="in person visit",
        ),
        BenchmarkSeed(
            condition_name="Diabetes",
            severity_level="severe",
            task_category="appointment",
            task_subcategory="schedule",
            age_group="middle_aged",
            gender="female",
            sex="female",
            care_preference="remote consultation",
        ),
        BenchmarkSeed(
            condition_name="Hypertension",
            severity_level="mild",
            task_category="health_qa",
            task_subcategory="symptoms",
            age_group="young_adult",
            gender="",
            sex="male",
            care_preference="no preference",
        ),
    ]


@pytest.fixture
def sample_entries(sample_seeds):
    """Create sample enriched entries for testing."""
    entries = []
    for i, seed in enumerate(sample_seeds):
        entries.append(
            EnrichedBenchmarkEntry(
                seed=seed,
                scenario_id=f"test-{i}",
                patient_story="Test patient story " * 20,
                patient_profile={
                    "medications": [{"name": f"Med{j}"} for j in range(i + 1)],
                    "account_info": {"age_in_years": "45"},
                },
                task_type=f"{seed.task_category}_{seed.task_subcategory}",
                preferred_care_option=seed.care_preference,
                has_image="False",
                scenario_complexity="regular" if i == 0 else "chronic",
            )
        )
    return entries


class TestAttributeDistribution:
    """Tests for AttributeDistribution class."""

    def test_add_and_count(self):
        """Test adding values and counting."""
        dist = AttributeDistribution(name="test")
        dist.add("a")
        dist.add("b")
        dist.add("a")

        assert dist.total == 3
        assert dist.counts["a"] == 2
        assert dist.counts["b"] == 1

    def test_get_percentages(self):
        """Test percentage calculation."""
        dist = AttributeDistribution(name="test")
        dist.add("a")
        dist.add("a")
        dist.add("b")
        dist.add("b")

        percentages = dist.get_percentages()
        assert percentages["a"] == 50.0
        assert percentages["b"] == 50.0

    def test_get_percentages_empty(self):
        """Test percentage calculation with no data."""
        dist = AttributeDistribution(name="test")
        assert dist.get_percentages() == {}

    def test_to_dict(self):
        """Test dictionary conversion."""
        dist = AttributeDistribution(name="test")
        dist.add("a")
        dist.add("b")

        result = dist.to_dict()
        assert result["total_count"] == 2
        assert result["unique_values"] == 2
        assert "a" in result["distribution"]
        assert "b" in result["distribution"]

    def test_to_dict_with_expected_weights(self):
        """Test dictionary conversion with expected weights comparison."""
        dist = AttributeDistribution(
            name="test",
            expected_weights={"a": 0.5, "b": 0.3, "c": 0.2},
        )
        dist.add("a")
        dist.add("a")
        dist.add("b")

        result = dist.to_dict()
        assert "expected_vs_actual" in result
        comparison = result["expected_vs_actual"]

        # Find the comparison for 'a'
        a_comp = next(c for c in comparison if c["value"] == "a")
        assert a_comp["expected_pct"] == 50.0
        assert a_comp["actual_pct"] == pytest.approx(66.67, rel=0.01)


class TestBenchmarkAnalyzer:
    """Tests for BenchmarkAnalyzer class."""

    def test_analyze_basic(self, sample_entries):
        """Test basic analysis without config."""
        analyzer = BenchmarkAnalyzer()
        summary = analyzer.analyze(sample_entries)

        assert summary.total_entries == 3
        assert "conditions" in summary.distributions
        assert "severity_levels" in summary.distributions
        assert "task_types" in summary.distributions

    def test_analyze_condition_distribution(self, sample_entries):
        """Test condition distribution analysis."""
        analyzer = BenchmarkAnalyzer()
        summary = analyzer.analyze(sample_entries)

        conditions = summary.distributions["conditions"]
        assert conditions.counts["Hypertension"] == 2
        assert conditions.counts["Diabetes"] == 1

    def test_analyze_severity_distribution(self, sample_entries):
        """Test severity distribution analysis."""
        analyzer = BenchmarkAnalyzer()
        summary = analyzer.analyze(sample_entries)

        severity = summary.distributions["severity_levels"]
        assert severity.counts["moderate"] == 1
        assert severity.counts["severe"] == 1
        assert severity.counts["mild"] == 1

    def test_analyze_task_types(self, sample_entries):
        """Test task type distribution analysis."""
        analyzer = BenchmarkAnalyzer()
        summary = analyzer.analyze(sample_entries)

        task_types = summary.distributions["task_types"]
        assert task_types.counts["prescription_renewal"] == 1
        assert task_types.counts["appointment_schedule"] == 1
        assert task_types.counts["health_qa_symptoms"] == 1

    def test_analyze_gender_not_specified(self, sample_entries):
        """Test handling of empty gender values."""
        analyzer = BenchmarkAnalyzer()
        summary = analyzer.analyze(sample_entries)

        genders = summary.distributions["genders"]
        assert genders.counts["(not specified)"] == 1
        assert genders.counts["male"] == 1
        assert genders.counts["female"] == 1

    def test_analyze_metadata(self, sample_entries):
        """Test metadata generation."""
        analyzer = BenchmarkAnalyzer()
        summary = analyzer.analyze(sample_entries)

        assert "medications" in summary.metadata
        assert "patient_stories" in summary.metadata

        # Entry 0 has 1 med, entry 1 has 2 meds, entry 2 has 3 meds
        assert summary.metadata["medications"]["average_count"] == 2.0
        assert summary.metadata["medications"]["min_count"] == 1
        assert summary.metadata["medications"]["max_count"] == 3

    def test_analyze_empty_entries(self):
        """Test analysis with empty entry list."""
        analyzer = BenchmarkAnalyzer()
        summary = analyzer.analyze([])

        assert summary.total_entries == 0
        assert summary.metadata == {}


class TestAnalyzeEntries:
    """Tests for analyze_entries convenience function."""

    def test_analyze_entries_basic(self, sample_entries):
        """Test convenience function."""
        summary = analyze_entries(sample_entries)
        assert summary.total_entries == 3


class TestWriteSummary:
    """Tests for write_summary function."""

    def test_write_summary(self, sample_entries):
        """Test writing summary to file."""
        summary = analyze_entries(sample_entries)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_summary.json"
            write_summary(summary, path)

            assert path.exists()

            with open(path) as f:
                data = json.load(f)

            assert data["summary"]["total_entries"] == 3
            assert "distributions" in data
            assert "conditions" in data["distributions"]


class TestGenerateSummaryPath:
    """Tests for generate_summary_path function."""

    def test_generate_summary_path(self):
        """Test summary path generation."""
        entries_path = Path("data/benchmark_20260211_171856.json")
        summary_path = generate_summary_path(entries_path)

        assert summary_path == Path("data/benchmark_20260211_171856_summary.json")

    def test_generate_summary_path_nested(self):
        """Test summary path generation with nested path."""
        entries_path = Path("output/run_123/entries.json")
        summary_path = generate_summary_path(entries_path)

        assert summary_path == Path("output/run_123/entries_summary.json")
