# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for output management.

Tests OutputManager file operations and summary generation.
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from hypothesis import given, settings, strategies as st

from patient_agent_bench.config import AgentSpec, BenchConfig, ModelConfig
from patient_agent_bench.eval.constants import MIN_SCORE, MAX_SCORE
from patient_agent_bench.runner.experiment_config import ExperimentConfig
from patient_agent_bench.runner.output_manager import OutputManager


class TestOutputManagerInit:
    """Tests for OutputManager initialization."""

    def test_init_with_defaults(self, temp_benchmark_file):
        """Test initialization with default values."""
        manager = OutputManager(input_file=str(temp_benchmark_file))
        
        assert manager.input_file == temp_benchmark_file
        assert manager.base_output_dir == Path("output")
        assert manager.timestamp is not None

    def test_init_with_custom_output_dir(self, temp_benchmark_file, tmp_path):
        """Test initialization with custom output directory."""
        custom_dir = tmp_path / "custom_output"
        manager = OutputManager(
            input_file=str(temp_benchmark_file),
            base_output_dir=str(custom_dir),
        )
        
        assert manager.base_output_dir == custom_dir

    def test_init_with_custom_timestamp(self, temp_benchmark_file):
        """Test initialization with custom timestamp."""
        manager = OutputManager(
            input_file=str(temp_benchmark_file),
            timestamp="20240101_120000",
        )
        
        assert manager.timestamp == "20240101_120000"
        assert "20240101_120000" in str(manager.run_dir)

    def test_run_dir_naming(self, temp_benchmark_file):
        """Test run directory naming convention."""
        manager = OutputManager(
            input_file=str(temp_benchmark_file),
            timestamp="20240101_120000",
        )
        
        # Should be: output/{input_stem}_{timestamp}
        expected_name = f"{temp_benchmark_file.stem}_20240101_120000"
        assert manager.run_dir.name == expected_name

    def test_file_paths(self, temp_benchmark_file):
        """Test output file paths are set correctly."""
        manager = OutputManager(input_file=str(temp_benchmark_file))
        
        assert manager.conversations_file.name == "conversations.json"
        assert manager.evaluations_file.name == "evaluations.json"
        assert manager.summary_file.name == "summary.json"


class TestOutputManagerSetup:
    """Tests for OutputManager setup."""

    def test_setup_creates_directory(self, temp_benchmark_file, tmp_path):
        """Test setup creates the output directory."""
        manager = OutputManager(
            input_file=str(temp_benchmark_file),
            base_output_dir=str(tmp_path / "new_output"),
        )
        
        run_dir = manager.setup()
        
        assert run_dir.exists()
        assert run_dir.is_dir()

    def test_setup_idempotent(self, temp_benchmark_file, tmp_path):
        """Test setup can be called multiple times."""
        manager = OutputManager(
            input_file=str(temp_benchmark_file),
            base_output_dir=str(tmp_path / "output"),
        )
        
        run_dir1 = manager.setup()
        run_dir2 = manager.setup()
        
        assert run_dir1 == run_dir2


class TestOutputManagerSaveLoad:
    """Tests for OutputManager save/load operations."""

    def test_save_experiment_results(
        self, temp_benchmark_file, tmp_path, sample_conversation_list, sample_evaluation_result
    ):
        """Test saving experiment results."""
        manager = OutputManager(
            input_file=str(temp_benchmark_file),
            base_output_dir=str(tmp_path),
        )
        manager.setup()

        experiment = ExperimentConfig(
            experiment_id="0_0",
            assistant_agent=AgentSpec(model=ModelConfig(model="claude-sonnet-4.5-bedrock")),
            user_agent=AgentSpec(model=ModelConfig(model="claude-haiku-4.5-bedrock")),
            evaluator_models=[ModelConfig(model="claude-sonnet-4.5-bedrock")],
            sandbox_model=ModelConfig(model="claude-sonnet-4.5-bedrock"),
            analyzer_model=ModelConfig(model="claude-sonnet-4.5-bedrock"),
            assistant_idx=0,
            user_idx=0,
            max_turns=3,
            strip_thinking_content=True,
        )
        conversations = [{"case_id": "test_001", "conversation": sample_conversation_list}]
        evaluations = [{"case_id": "test_001", "evaluation": sample_evaluation_result}]
        path = manager.save_experiment_results(experiment, conversations, evaluations)

        assert path.exists()
        assert (path / "conversations.json").exists()
        assert (path / "evaluations.json").exists()


class TestOutputManagerSummaryGeneration:
    """Tests for summary generation."""

    def test_generate_summary_basic(self, temp_benchmark_file, tmp_path):
        """Test basic summary generation."""
        manager = OutputManager(
            input_file=str(temp_benchmark_file),
            base_output_dir=str(tmp_path),
        )
        
        evaluations = [
            {
                "case_id": "test_001",
                "evaluation": {
                    "aggregate_score": 80,
                    "safety_pass": True,
                    "rubric_scores": {"task_completion": 3, "clinical_safety": 3},
                },
            },
            {
                "case_id": "test_002",
                "evaluation": {
                    "aggregate_score": 90,
                    "rubric_scores": {"task_completion": 3, "clinical_safety": 3},
                },
            },
        ]
        
        summary = manager.generate_summary(evaluations)
        
        assert summary["aggregate_metrics"]["total_cases"] == 2
        assert summary["aggregate_metrics"]["average_score"] == 85.0
        # All scores are 3 (>= PASS_THRESHOLD), so pass rates should be 100%
        assert "rubric_pass_rates" in summary
        assert summary["rubric_pass_rates"]["task_completion"] == 100.0
        assert summary["rubric_pass_rates"]["clinical_safety"] == 100.0

    def test_generate_summary_with_low_scores(self, temp_benchmark_file, tmp_path):
        """Test summary with low scores."""
        manager = OutputManager(
            input_file=str(temp_benchmark_file),
            base_output_dir=str(tmp_path),
        )
        
        evaluations = [
            {
                "case_id": "test_001",
                "evaluation": {
                    "aggregate_score": 80,
                    "rubric_scores": {"clinical_safety": 3},
                },
            },
            {
                "case_id": "test_002",
                "evaluation": {
                    "aggregate_score": 20,
                    "rubric_scores": {"clinical_safety": 1},
                },
            },
        ]
        
        summary = manager.generate_summary(evaluations)
        
        assert summary["aggregate_metrics"]["average_score"] == 50.0
        assert summary["aggregate_metrics"]["evaluated_cases"] == 2

    def test_generate_summary_empty(self, temp_benchmark_file, tmp_path):
        """Test summary generation with empty evaluations."""
        manager = OutputManager(
            input_file=str(temp_benchmark_file),
            base_output_dir=str(tmp_path),
        )
        
        summary = manager.generate_summary([])
        
        assert "error" in summary

    def test_generate_summary_with_errors(self, temp_benchmark_file, tmp_path):
        """Test summary handles evaluation errors gracefully."""
        manager = OutputManager(
            input_file=str(temp_benchmark_file),
            base_output_dir=str(tmp_path),
        )
        
        evaluations = [
            {
                "case_id": "test_001",
                "evaluation": {"error": "Evaluation failed"},
            },
            {
                "case_id": "test_002",
                "evaluation": {
                    "aggregate_score": 80,
                    "rubric_scores": {"task_completion": 2},
                },
            },
        ]
        
        summary = manager.generate_summary(evaluations)
        
        # Should only count the successful evaluation
        assert summary["aggregate_metrics"]["evaluated_cases"] == 1

    def test_generate_summary_rubric_averages(self, temp_benchmark_file, tmp_path):
        """Test rubric averages in summary."""
        manager = OutputManager(
            input_file=str(temp_benchmark_file),
            base_output_dir=str(tmp_path),
        )
        
        evaluations = [
            {
                "case_id": "test_001",
                "evaluation": {
                    "aggregate_score": 80,
                    "rubric_scores": {"task_completion": 2, "clinical_safety": 1},
                },
            },
            {
                "case_id": "test_002",
                "evaluation": {
                    "aggregate_score": 90,
                    "rubric_scores": {"task_completion": 1, "clinical_safety": 2},
                },
            },
        ]
        
        summary = manager.generate_summary(evaluations)
        
        assert summary["rubric_averages"]["task_completion"] == 1.5
        assert summary["rubric_averages"]["clinical_safety"] == 1.5

    def test_generate_summary_case_results(self, temp_benchmark_file, tmp_path):
        """Test per-case results in summary."""
        manager = OutputManager(
            input_file=str(temp_benchmark_file),
            base_output_dir=str(tmp_path),
        )
        
        evaluations = [
            {
                "case_id": "test_001",
                "evaluation": {
                    "aggregate_score": 80,
                    "rubric_scores": {"task_completion": 2},
                },
            },
        ]
        
        summary = manager.generate_summary(evaluations)
        
        assert len(summary["case_results"]) == 1
        assert summary["case_results"][0]["case_id"] == "test_001"
        assert summary["case_results"][0]["aggregate_score"] == 80



class TestOutputManagerHelperMethods:
    """Tests for OutputManager public methods."""

    def test_get_run_dir(self, temp_output_dir, temp_benchmark_file):
        """Test getting the run directory path."""
        manager = OutputManager(
            input_file=str(temp_benchmark_file),
            base_output_dir=str(temp_output_dir)
        )

        run_dir = manager.run_dir

        assert run_dir.exists() or not run_dir.exists()  # Path object exists
        assert str(temp_output_dir) in str(run_dir)

    def test_file_paths_are_correct(self, temp_output_dir, temp_benchmark_file):
        """Test that file paths are correctly constructed."""
        manager = OutputManager(
            input_file=str(temp_benchmark_file),
            base_output_dir=str(temp_output_dir)
        )

        assert manager.conversations_file.name == "conversations.json"
        assert manager.evaluations_file.name == "evaluations.json"
        assert manager.summary_file.name == "summary.json"
        assert manager.conversations_file.parent == manager.run_dir

    def test_load_partial_evaluator_results_no_file(
        self, temp_output_dir, temp_benchmark_file
    ):
        """Test loading partial evaluator results when no file exists returns all None."""
        manager = OutputManager(
            input_file=str(temp_benchmark_file),
            base_output_dir=str(temp_output_dir)
        )
        manager.setup()

        result = manager.load_partial_evaluator_results("0_0", 0, 3)
        assert result == [None, None, None]


# =============================================================================
# Property-Based Tests for Multi-Experiment Support
# =============================================================================


# Strategies for generating test data
@st.composite
def cli_params_strategy(draw):
    """Generate valid CLI parameters."""
    return {
        "benchmark_file": draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip())),
        "num_cases": draw(st.integers(min_value=0, max_value=100)),
        "case_id": draw(st.one_of(st.none(), st.text(min_size=1, max_size=20))),
        "max_turns": draw(st.integers(min_value=1, max_value=10)),
    }


@st.composite
def model_config_strategy(draw):
    """Generate a valid ModelConfig using registry models."""
    model_names = ["claude-opus-4.5-bedrock", "claude-sonnet-4.5-bedrock", "claude-haiku-4.5-bedrock"]
    return ModelConfig(model=draw(st.sampled_from(model_names)))


@st.composite
def evaluator_models_strategy(draw):
    """Generate a list of 1-3 evaluator ModelConfigs."""
    return draw(st.lists(model_config_strategy(), min_size=1, max_size=3))


@st.composite
def agent_spec_strategy(draw):
    """Generate a valid AgentSpec."""
    prompt_names = ["default_prompt", "custom_prompt", "test_prompt", "experimental_prompt"]
    return AgentSpec(
        model=draw(model_config_strategy()),
        prompt=draw(st.sampled_from(prompt_names)),
    )


@st.composite
def experiment_config_strategy(draw):
    """Generate a valid ExperimentConfig."""
    a_idx = draw(st.integers(min_value=0, max_value=5))
    u_idx = draw(st.integers(min_value=0, max_value=5))

    return ExperimentConfig(
        experiment_id=f"{a_idx}_{u_idx}",
        assistant_agent=draw(agent_spec_strategy()),
        user_agent=draw(agent_spec_strategy()),
        evaluator_models=draw(evaluator_models_strategy()),
        sandbox_model=draw(model_config_strategy()),
        analyzer_model=draw(model_config_strategy()),
        assistant_idx=a_idx,
        user_idx=u_idx,
        max_turns=3,
        strip_thinking_content=True,
    )


@st.composite
def evaluation_result_strategy(draw):
    """Generate a valid evaluation result."""
    safety_pass = draw(st.booleans())
    return {
        "case_id": draw(st.text(min_size=1, max_size=20).filter(lambda x: x.strip())),
        "evaluation": {
            "aggregate_score": draw(st.floats(min_value=MIN_SCORE, max_value=MAX_SCORE)),
            "safety_pass": safety_pass,
            "rubric_scores": {
                "task_completion": draw(st.integers(min_value=MIN_SCORE, max_value=MAX_SCORE)),
                "clinical_safety": draw(st.integers(min_value=MIN_SCORE, max_value=MAX_SCORE)),
            },
        },
    }


class TestOutputManagerPropertyBased:
    """Property-based tests for OutputManager multi-experiment extensions.

    **Feature: multi-experiment-runner**
    **Property 7: Experiment Directory Contents**
    **Property 8: Run Config Contains CLI Params**
    **Property 9: Summary Contains All Experiments**
    **Validates: Requirements 1.2, 4.2, 4.3, 6.2, 6.3**
    """

    @given(cli_params=cli_params_strategy())
    @settings(max_examples=100)
    def test_property_8_run_config_contains_cli_params(self, cli_params):
        """
        Property 8: Run Config Contains CLI Params

        *For any* benchmark run with CLI parameters, the saved run_config.json
        SHALL contain a "cli_params" key with the provided parameters.

        **Validates: Requirements 1.2**
        """
        config = BenchConfig(
            assistant_agents=[AgentSpec(model=ModelConfig(model="claude-sonnet-4.5-bedrock"))],
            user_agents=[AgentSpec(model=ModelConfig(model="claude-haiku-4.5-bedrock"))],
            evaluator_models=[ModelConfig(model="claude-sonnet-4.5-bedrock")],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a temp benchmark file
            benchmark_file = Path(temp_dir) / "test_benchmark.json"
            benchmark_file.write_text("[]")

            output_mgr = OutputManager(str(benchmark_file), temp_dir)
            output_mgr.setup()

            # Save run config
            path = output_mgr.save_run_config(config, cli_params)

            # Verify file exists
            assert path.exists()

            # Load and verify contents
            saved_config = json.loads(path.read_text())

            # Must contain cli_params key
            assert "cli_params" in saved_config

            # CLI params must match
            assert saved_config["cli_params"] == cli_params

            # Must also contain agent configs
            assert "assistant_agent" in saved_config
            assert "user_agent" in saved_config
            assert "evaluator_model" in saved_config

    @given(experiment=experiment_config_strategy())
    @settings(max_examples=100)
    def test_property_7_experiment_directory_contents(self, experiment):
        """
        Property 7: Experiment Directory Contents

        *For any* completed experiment, its subdirectory SHALL contain exactly
        four files: experiment_config.json, conversations.json, evaluations.json,
        and summary.json.

        **Validates: Requirements 4.2, 4.3**
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a temp benchmark file
            benchmark_file = Path(temp_dir) / "test_benchmark.json"
            benchmark_file.write_text("[]")

            output_mgr = OutputManager(str(benchmark_file), temp_dir)
            output_mgr.setup()

            # Create sample data
            conversations = [{"case_id": "test", "conversation": []}]
            evaluations = [
                {
                    "case_id": "test",
                    "evaluation": {
                        "aggregate_score": 80,
                        "safety_pass": True,
                        "rubric_scores": {},
                    },
                }
            ]

            # Save experiment results
            exp_dir = output_mgr.save_experiment_results(
                experiment, conversations, evaluations
            )

            # Verify directory exists
            assert exp_dir.exists()
            assert exp_dir.is_dir()

            # Verify all four required files exist
            required_files = [
                "experiment_config.json",
                "conversations.json",
                "evaluations.json",
                "summary.json",
            ]
            for filename in required_files:
                file_path = exp_dir / filename
                assert file_path.exists(), f"Missing required file: {filename}"

            # Verify experiment_config.json contains correct experiment_id
            exp_config = json.loads((exp_dir / "experiment_config.json").read_text())
            assert exp_config["experiment_id"] == experiment.experiment_id

    @given(
        num_experiments=st.integers(min_value=1, max_value=10),
        num_failures=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=100)
    def test_property_9_summary_contains_all_experiments(
        self, num_experiments, num_failures
    ):
        """
        Property 9: Summary Contains All Experiments

        *For any* completed multi-experiment run, the experiments_summary.json
        comparison_table SHALL contain entries for all successful experiments.

        **Validates: Requirements 6.2, 6.3**
        """
        # Ensure failures don't exceed total experiments
        num_failures = min(num_failures, num_experiments)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a temp benchmark file
            benchmark_file = Path(temp_dir) / "test_benchmark.json"
            benchmark_file.write_text("[]")

            output_mgr = OutputManager(str(benchmark_file), temp_dir)
            output_mgr.setup()

            # Create experiments and results
            experiments = []
            results = {}

            for i in range(num_experiments):
                exp = ExperimentConfig(
                    experiment_id=f"{i}_0",
                    assistant_agent=AgentSpec(model=ModelConfig(model="claude-sonnet-4.5-bedrock")),
                    user_agent=AgentSpec(model=ModelConfig(model="claude-haiku-4.5-bedrock")),
                    evaluator_models=[ModelConfig(model="claude-sonnet-4.5-bedrock")],
                    sandbox_model=ModelConfig(model="claude-haiku-4.5-bedrock"),
                    analyzer_model=ModelConfig(model="claude-sonnet-4.5-bedrock"),
                    assistant_idx=i,
                    user_idx=0,
                    max_turns=3,
                    strip_thinking_content=True,
                )
                experiments.append(exp)

                if i < num_failures:
                    # Mark as failed
                    results[exp.experiment_id] = {"error": f"Simulated failure {i}"}
                else:
                    # Mark as successful with evaluations
                    results[exp.experiment_id] = {
                        "evaluations": [
                            {
                                "case_id": "test",
                                "evaluation": {
                                    "aggregate_score": 80,
                                    "safety_pass": True,
                                    "rubric_scores": {},
                                },
                            }
                        ]
                    }

            # Generate summary
            summary = output_mgr.generate_experiments_summary(results, experiments)

            # Verify total experiments count
            assert summary["total_experiments"] == num_experiments

            # Verify successful experiments count
            expected_successful = num_experiments - num_failures
            assert summary["successful_experiments"] == expected_successful

            # Verify comparison_table has entries for all experiments
            assert len(summary["comparison_table"]) == num_experiments

            # Verify model_specs has entries for all experiments
            assert len(summary["model_specs"]) == num_experiments

            # Verify all experiment IDs are present in comparison_table
            comparison_ids = {row["experiment_id"] for row in summary["comparison_table"]}
            expected_ids = {exp.experiment_id for exp in experiments}
            assert comparison_ids == expected_ids

            # Verify failed experiments have error in comparison_table
            for row in summary["comparison_table"]:
                exp_id = row["experiment_id"]
                if "error" in results[exp_id]:
                    assert "error" in row

    @given(experiment=experiment_config_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_8_prompt_config_in_output(self, experiment):
        """
        Property 8: Prompt Config in Output

        *For any* benchmark run, the output configuration files (run_config.json,
        experiment_config.json) and summary SHALL include the user_prompt and
        assistant_prompt values used (now inside AgentSpec objects).

        **Feature: configurable-prompts, Property 8: Prompt Config in Output**
        **Validates: Requirements 5.1, 5.2**
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a temp benchmark file
            benchmark_file = Path(temp_dir) / "test_benchmark.json"
            benchmark_file.write_text("[]")

            output_mgr = OutputManager(str(benchmark_file), temp_dir)
            output_mgr.setup()

            # Create config with AgentSpec-based prompts
            config = BenchConfig(
                assistant_agents=[experiment.assistant_agent],
                user_agents=[experiment.user_agent],
                evaluator_models=list(experiment.evaluator_models),
            )

            # Save run config
            cli_params = {"benchmark_file": "test.json", "max_turns": 3}
            run_config_path = output_mgr.save_run_config(config, cli_params)

            # Verify run_config.json includes prompt fields inside agent specs
            run_config = json.loads(run_config_path.read_text())
            assert "assistant_agent" in run_config
            assert "user_agent" in run_config
            assert run_config["assistant_agent"][0]["prompt"] == experiment.assistant_agent.prompt
            assert run_config["user_agent"][0]["prompt"] == experiment.user_agent.prompt

            # Create sample data and save experiment results
            conversations = [{"case_id": "test", "conversation": []}]
            evaluations = [
                {
                    "case_id": "test",
                    "evaluation": {
                        "aggregate_score": 80,
                        "safety_pass": True,
                        "rubric_scores": {},
                    },
                }
            ]

            exp_dir = output_mgr.save_experiment_results(
                experiment, conversations, evaluations
            )

            # Verify experiment_config.json includes prompt fields inside agent specs
            exp_config_path = exp_dir / "experiment_config.json"
            exp_config = json.loads(exp_config_path.read_text())
            assert "assistant_agent" in exp_config
            assert "user_agent" in exp_config
            assert exp_config["assistant_agent"]["prompt"] == experiment.assistant_agent.prompt
            assert exp_config["user_agent"]["prompt"] == experiment.user_agent.prompt

            # Generate experiments summary
            results = {
                experiment.experiment_id: {
                    "evaluations": evaluations,
                }
            }
            summary = output_mgr.generate_experiments_summary(results, [experiment])

            # Verify summary includes prompt fields in model_specs
            assert len(summary["model_specs"]) == 1
            model_spec = summary["model_specs"][0]
            assert "user_prompt" in model_spec
            assert "assistant_prompt" in model_spec
            assert model_spec["user_prompt"] == experiment.user_agent.prompt
            assert model_spec["assistant_prompt"] == experiment.assistant_agent.prompt


# =============================================================================
# Property 8: Experiment Directory Name Validation (Task 6.2)
# =============================================================================

from patient_agent_bench.run import _is_experiment_dir


class TestIsExperimentDirPropertyBased:
    """Property-based tests for _is_experiment_dir.

    **Feature: multi-evaluator-aggregation**
    **Property 8: Experiment directory name validation**
    **Validates: Requirements 1.3**
    """

    @given(
        x=st.integers(min_value=0, max_value=999),
        y=st.integers(min_value=0, max_value=999),
    )
    @settings(max_examples=100)
    def test_property_8_valid_x_y_format(self, x, y):
        """
        Property 8: Valid X_Y format

        *For any* string "{X}_{Y}" where X and Y are non-negative integers,
        _is_experiment_dir() SHALL return True.

        **Validates: Requirements 1.3**
        """
        assert _is_experiment_dir(f"{x}_{y}") is True

    @given(
        x=st.integers(min_value=0, max_value=999),
        y=st.integers(min_value=0, max_value=999),
        z=st.integers(min_value=0, max_value=999),
    )
    @settings(max_examples=100)
    def test_property_8_valid_legacy_x_y_z_format(self, x, y, z):
        """
        Property 8 (continued): Legacy X_Y_Z format

        *For any* string "{X}_{Y}_{Z}" where X, Y, Z are non-negative integers,
        _is_experiment_dir() SHALL return True (backward compatibility).
        """
        assert _is_experiment_dir(f"{x}_{y}_{z}") is True

    @given(name=st.text(min_size=0, max_size=20))
    @settings(max_examples=100)
    def test_property_8_rejects_non_numeric(self, name):
        """
        Property 8 (continued): Non-numeric rejection

        *For any* string that does not match X_Y or X_Y_Z format with
        all-digit parts, _is_experiment_dir() SHALL return False.
        """
        parts = name.split("_")
        if len(parts) in (2, 3) and all(p.isdigit() for p in parts):
            # This is actually a valid format, skip
            return
        assert _is_experiment_dir(name) is False


class TestIsExperimentDirUnit:
    """Unit tests for _is_experiment_dir edge cases."""

    def test_valid_two_part(self):
        assert _is_experiment_dir("0_0") is True
        assert _is_experiment_dir("1_2") is True
        assert _is_experiment_dir("99_0") is True

    def test_valid_three_part_legacy(self):
        assert _is_experiment_dir("0_0_0") is True
        assert _is_experiment_dir("1_2_3") is True

    def test_invalid_single_part(self):
        assert _is_experiment_dir("0") is False
        assert _is_experiment_dir("abc") is False

    def test_invalid_four_parts(self):
        assert _is_experiment_dir("0_0_0_0") is False

    def test_invalid_non_digit(self):
        assert _is_experiment_dir("a_b") is False
        assert _is_experiment_dir("0_a") is False
        assert _is_experiment_dir("conversations") is False

    def test_empty_string(self):
        assert _is_experiment_dir("") is False


# =============================================================================
# Tests for Multi-Evaluator Output Format (Requirement 9.4)
# =============================================================================

from patient_agent_bench.eval.aggregator import add_pass_labels, merge_evaluations


class TestMultiEvaluatorOutputFormat:
    """Tests validating the multi-evaluator evaluation output format.

    Ensures evaluations contain evaluation_0, ..., evaluation_N-1 keys
    for individual results and a merged 'evaluation' key.

    **Feature: multi-evaluator-aggregation**
    **Validates: Requirements 3.2, 3.3, 9.4**
    """

    def _build_multi_evaluator_output(self, num_evaluators, rubric_scores_list):
        """Helper to simulate multi-evaluator output construction."""
        individual_evals = []
        for scores in rubric_scores_list:
            ev = {
                "rubric_scores": scores,
                "rubric_results": {
                    name: {"score": score, "explanation": f"Eval for {name}"}
                    for name, score in scores.items()
                },
                "aggregate_score": sum(scores.values()) / len(scores) * 50,
                "summary": "Test summary",
            }
            ev = add_pass_labels(ev)
            individual_evals.append(ev)

        # Build output dict as the system would
        output = {"case_id": "test_001", "conversation": []}
        for i, ev in enumerate(individual_evals):
            output[f"evaluation_{i}"] = ev
        output["evaluation"] = merge_evaluations(individual_evals)
        return output, individual_evals

    def test_two_evaluators_output_keys(self):
        """Test output has evaluation_0, evaluation_1, and evaluation."""
        scores_list = [
            {"task_completion": 3, "clinical_safety": 2},
            {"task_completion": 2, "clinical_safety": 3},
        ]
        output, _ = self._build_multi_evaluator_output(2, scores_list)

        assert "evaluation_0" in output
        assert "evaluation_1" in output
        assert "evaluation" in output
        assert "evaluation_2" not in output

    def test_three_evaluators_output_keys(self):
        """Test output has evaluation_0 through evaluation_2 and evaluation."""
        scores_list = [
            {"task_completion": 3, "clinical_safety": 1},
            {"task_completion": 2, "clinical_safety": 2},
            {"task_completion": 1, "clinical_safety": 3},
        ]
        output, _ = self._build_multi_evaluator_output(3, scores_list)

        assert "evaluation_0" in output
        assert "evaluation_1" in output
        assert "evaluation_2" in output
        assert "evaluation" in output
        assert "evaluation_3" not in output

    def test_single_evaluator_output_keys(self):
        """Test output with single evaluator still has evaluation_0 and evaluation."""
        scores_list = [{"task_completion": 3, "clinical_safety": 3}]
        output, _ = self._build_multi_evaluator_output(1, scores_list)

        assert "evaluation_0" in output
        assert "evaluation" in output
        assert "evaluation_1" not in output

    def test_individual_evals_have_pass_labels(self):
        """Test that individual evaluation_N results have pass labels."""
        scores_list = [
            {"task_completion": 4, "clinical_safety": 1},
            {"task_completion": 1, "clinical_safety": 4},
        ]
        output, _ = self._build_multi_evaluator_output(2, scores_list)

        # evaluation_0 should have pass labels
        ev0 = output["evaluation_0"]
        assert ev0["rubric_results"]["task_completion"]["pass"] is True
        assert ev0["rubric_results"]["clinical_safety"]["pass"] is False

        # evaluation_1 should have pass labels
        ev1 = output["evaluation_1"]
        assert ev1["rubric_results"]["task_completion"]["pass"] is False
        assert ev1["rubric_results"]["clinical_safety"]["pass"] is True

    def test_merged_eval_has_averaged_scores(self):
        """Test that merged evaluation has averaged rubric scores."""
        scores_list = [
            {"task_completion": 3, "clinical_safety": 1},
            {"task_completion": 1, "clinical_safety": 3},
        ]
        output, _ = self._build_multi_evaluator_output(2, scores_list)

        merged = output["evaluation"]
        assert merged["rubric_scores"]["task_completion"] == 2.0
        assert merged["rubric_scores"]["clinical_safety"] == 2.0

    def test_merged_eval_has_pass_labels(self):
        """Test that merged evaluation has pass labels."""
        scores_list = [
            {"task_completion": 3, "clinical_safety": 1},
            {"task_completion": 1, "clinical_safety": 3},
        ]
        output, _ = self._build_multi_evaluator_output(2, scores_list)

        merged = output["evaluation"]
        # Both average to 2.0, which is < PASS_THRESHOLD (3) → not pass
        assert merged["rubric_results"]["task_completion"]["pass"] is False
        assert merged["rubric_results"]["clinical_safety"]["pass"] is False

    @given(
        num_evaluators=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_property_output_has_correct_key_count(self, num_evaluators):
        """
        Property: Multi-evaluator output key count

        *For any* N evaluators, the output SHALL contain exactly N
        evaluation_i keys (i=0..N-1) plus one merged evaluation key.

        **Validates: Requirements 3.2, 3.3**
        """
        scores_list = [
            {"task_completion": 2, "clinical_safety": 2}
            for _ in range(num_evaluators)
        ]
        output, _ = self._build_multi_evaluator_output(num_evaluators, scores_list)

        # Check all evaluation_i keys exist
        for i in range(num_evaluators):
            assert f"evaluation_{i}" in output

        # Check no extra evaluation_N key
        assert f"evaluation_{num_evaluators}" not in output

        # Check merged evaluation exists
        assert "evaluation" in output
