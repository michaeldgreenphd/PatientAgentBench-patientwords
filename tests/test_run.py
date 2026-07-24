# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for CLI-layer guards in run.py.

Focus: the silent-success guard (_exit_if_no_evaluations) that makes a
benchmark/evaluate run exit non-zero when EVERY conversation failed, instead
of printing an all-zeros summary and exiting 0.
"""

import logging

import pytest

from patient_agent_bench.run import _exit_if_no_evaluations


class TestExitIfNoEvaluations:
    """Tests for the total-failure guard used by benchmark/evaluate."""

    def test_total_failure_exits_nonzero(self):
        """0 evaluated cases across all experiments -> sys.exit(1)."""
        summary = {
            "comparison_table": [
                {"experiment_id": "0_0", "num_cases": 5, "evaluated_cases": 0},
                {"experiment_id": "1_0", "num_cases": 5, "evaluated_cases": 0},
            ]
        }
        with pytest.raises(SystemExit) as exc:
            _exit_if_no_evaluations(summary)
        assert exc.value.code == 1

    def test_all_success_does_not_exit(self):
        """Every case evaluated -> no exit, no exception."""
        summary = {
            "comparison_table": [
                {"experiment_id": "0_0", "num_cases": 5, "evaluated_cases": 5},
            ]
        }
        # Should simply return None.
        assert _exit_if_no_evaluations(summary) is None

    def test_partial_failure_warns_but_does_not_exit(self, caplog):
        """Some cases fail, some succeed -> warn, but exit 0 (return normally)."""
        summary = {
            "comparison_table": [
                {"experiment_id": "0_0", "num_cases": 5, "evaluated_cases": 3},
            ]
        }
        with caplog.at_level(logging.WARNING, logger="patient_agent_bench.run"):
            assert _exit_if_no_evaluations(summary) is None
        assert any("failed to evaluate" in r.message for r in caplog.records)

    def test_empty_run_does_not_exit(self):
        """No cases at all (total_cases == 0) is not treated as a failure here;
        the empty-entries guard upstream already handles that."""
        assert _exit_if_no_evaluations({"comparison_table": []}) is None

    def test_mixed_experiments_one_all_fail_one_ok_does_not_exit(self):
        """If ANY experiment produced evaluations, it is not a total failure."""
        summary = {
            "comparison_table": [
                {"experiment_id": "0_0", "num_cases": 5, "evaluated_cases": 0},
                {"experiment_id": "1_0", "num_cases": 5, "evaluated_cases": 5},
            ]
        }
        # 5 of 10 evaluated -> partial, warns, but does not exit.
        assert _exit_if_no_evaluations(summary) is None
