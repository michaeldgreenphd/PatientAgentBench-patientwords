# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for chart generation (eval/charts.py).

Tests heatmap, grouped bar chart, and high-level chart generators.
"""

import json
import pytest
from pathlib import Path

from patient_agent_bench.eval.charts import (
    generate_heatmap,
    generate_grouped_bar_chart,
    generate_experiment_breakdown_charts,
    generate_cross_experiment_heatmaps,
)


class TestGenerateHeatmap:
    """Tests for low-level heatmap generation."""

    def test_basic_heatmap(self, tmp_path):
        path = str(tmp_path / "test_heatmap.png")
        result = generate_heatmap(
            rows=["model_a", "model_b"],
            cols=["mild", "severe"],
            values=[[80.0, 40.0], [60.0, 30.0]],
            title="Test Heatmap",
            output_path=path,
        )
        assert Path(result).exists()
        assert Path(result).stat().st_size > 1000  # non-trivial PNG

    def test_single_cell_heatmap(self, tmp_path):
        path = str(tmp_path / "single.png")
        result = generate_heatmap(
            rows=["model"],
            cols=["mild"],
            values=[[100.0]],
            title="Single Cell",
            output_path=path,
        )
        assert Path(result).exists()

    def test_custom_range(self, tmp_path):
        path = str(tmp_path / "custom.png")
        result = generate_heatmap(
            rows=["a", "b"],
            cols=["x", "y"],
            values=[[1.5, 0.5], [1.0, 2.0]],
            title="Custom Range",
            output_path=path,
            vmin=0,
            vmax=2,
            value_label="Avg Score",
            annot_fmt=".1f",
        )
        assert Path(result).exists()


class TestGenerateGroupedBarChart:
    """Tests for low-level grouped bar chart generation."""

    def test_basic_bar_chart(self, tmp_path):
        path = str(tmp_path / "test_bar.png")
        result = generate_grouped_bar_chart(
            groups=["mild", "moderate", "severe"],
            series={
                "task_completion": [90.0, 80.0, 60.0],
                "clinical_safety": [70.0, 50.0, 30.0],
            },
            title="Test Bar Chart",
            output_path=path,
        )
        assert Path(result).exists()
        assert Path(result).stat().st_size > 1000

    def test_single_series(self, tmp_path):
        path = str(tmp_path / "single_series.png")
        result = generate_grouped_bar_chart(
            groups=["a", "b"],
            series={"safety": [80.0, 40.0]},
            title="Single Series",
            output_path=path,
        )
        assert Path(result).exists()

    def test_single_group(self, tmp_path):
        path = str(tmp_path / "single_group.png")
        result = generate_grouped_bar_chart(
            groups=["all"],
            series={"safety": [75.0], "task": [90.0]},
            title="Single Group",
            output_path=path,
        )
        assert Path(result).exists()


class TestExperimentBreakdownCharts:
    """Tests for high-level experiment breakdown chart generation."""

    def test_generates_charts_from_breakdowns(self, tmp_path):
        summary = {
            "breakdowns": {
                "by_severity_level": {
                    "mild": {
                        "n": 5,
                        "rubric_averages": {
                            "task_completion": 3.6,
                            "clinical_safety": 3.2,
                        },
                    },
                    "severe": {
                        "n": 3,
                        "rubric_averages": {
                            "task_completion": 2.4,
                            "clinical_safety": 1.6,
                        },
                    },
                },
                "by_task_type": {
                    "refill": {
                        "n": 4,
                        "rubric_averages": {
                            "task_completion": 3.4,
                            "clinical_safety": 2.8,
                        },
                    },
                },
            },
        }

        paths = generate_experiment_breakdown_charts(
            summary, str(tmp_path), experiment_label="claude-sonnet"
        )

        assert len(paths) == 2
        for p in paths:
            assert Path(p).exists()

    def test_no_breakdowns_returns_empty(self, tmp_path):
        summary = {"aggregate_metrics": {"average_score": 80}}
        paths = generate_experiment_breakdown_charts(summary, str(tmp_path))
        assert paths == []


class TestCrossExperimentHeatmaps:
    """Tests for cross-experiment heatmap generation."""

    def test_generates_heatmaps(self, tmp_path):
        experiments_summary = {
            "model_specs": [
                {"experiment_id": "0_0", "assistant": "model_a"},
                {"experiment_id": "1_0", "assistant": "model_b"},
            ],
        }

        per_exp = {
            "0_0": {
                "breakdowns": {
                    "by_severity_level": {
                        "mild": {
                            "rubric_averages": {"clinical_safety": 3.2},
                        },
                        "severe": {
                            "rubric_averages": {"clinical_safety": 1.6},
                        },
                    },
                    "by_task_type": {
                        "refill": {
                            "rubric_averages": {"clinical_safety": 2.8},
                        },
                    },
                    "by_scenario_complexity": {
                        "regular": {
                            "rubric_averages": {"clinical_safety": 3.0},
                        },
                    },
                },
            },
            "1_0": {
                "breakdowns": {
                    "by_severity_level": {
                        "mild": {
                            "rubric_averages": {"clinical_safety": 2.4},
                        },
                        "severe": {
                            "rubric_averages": {"clinical_safety": 0.8},
                        },
                    },
                    "by_task_type": {
                        "refill": {
                            "rubric_averages": {"clinical_safety": 2.0},
                        },
                    },
                    "by_scenario_complexity": {
                        "regular": {
                            "rubric_averages": {"clinical_safety": 2.2},
                        },
                    },
                },
            },
        }

        paths = generate_cross_experiment_heatmaps(
            experiments_summary, per_exp, str(tmp_path)
        )

        assert len(paths) == 3
        for p in paths:
            assert Path(p).exists()
            assert Path(p).stat().st_size > 1000

    def test_no_breakdowns_returns_empty(self, tmp_path):
        experiments_summary = {
            "model_specs": [
                {"experiment_id": "0_0", "assistant": "model_a"},
            ],
        }
        per_exp = {"0_0": {"aggregate_metrics": {"average_score": 80}}}

        paths = generate_cross_experiment_heatmaps(
            experiments_summary, per_exp, str(tmp_path)
        )
        assert paths == []
