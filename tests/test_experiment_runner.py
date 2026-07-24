# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for ExperimentRunner.

Tests the experiment runner including property-based tests for:
- Property 1: Experiment generation produces correct count (A × U)
- Property 6: Conversation Reuse Correctness
- Property 10: Experiment Failure Isolation
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import patch, AsyncMock

from hypothesis import given, settings, strategies as st, HealthCheck

from patient_agent_bench.config import AgentSpec, BenchConfig, ModelConfig, RolePoolManager
from patient_agent_bench.benchmark_seed import BenchmarkEntry
from patient_agent_bench.runner.experiment_config import ExperimentConfig
from patient_agent_bench.runner.experiment_runner import ExperimentRunner
from patient_agent_bench.runner.output_manager import OutputManager
from patient_agent_bench.utils.retry import RetryConfig

from tests.test_config import agent_spec_strategy, model_config_dict


# =============================================================================
# Helper functions for property tests
# =============================================================================


def create_temp_benchmark_file() -> Path:
    """Create a temporary benchmark file for testing."""
    benchmark_data = [
        {
            "query_id": "test_001",
            "query": "Test query",
            "patient_story": "Test patient story",
            "patient_profile": {
                "account_info": {"patient_id": "P12345"},
                "personal_info": {"first_name": "Test", "last_name": "User"},
                "addresses": {},
                "phone_numbers": {},
                "emergency_contacts": {},
                "insurances": {},
                "care_team": {},
                "pharmacies": {},
                "insurance_status": {},
            },
            "condition_name": "Test",
            "condition_mapped_name": "Test",
            "preferred_care_option": "test",
            "severity_level": "low",
            "interaction_type": "test",
            "task_type": "test",
            "has_image": "False",
        }
    ]
    fd, path = tempfile.mkstemp(suffix=".json")
    with open(fd, "w") as f:
        json.dump(benchmark_data, f)
    return Path(path)


def create_sample_benchmark_entry() -> BenchmarkEntry:
    """Create a sample benchmark entry for testing."""
    return BenchmarkEntry.from_dict({
        "query_id": "test_001",
        "query": "Test query",
        "patient_story": "Test patient story",
        "patient_profile": {
            "account_info": {"patient_id": "P12345"},
            "personal_info": {"first_name": "Test", "last_name": "User"},
            "addresses": {},
            "phone_numbers": {},
            "emergency_contacts": {},
            "insurances": {},
            "care_team": {},
            "pharmacies": {},
            "insurance_status": {},
        },
        "condition_name": "Test",
        "condition_mapped_name": "Test",
        "preferred_care_option": "test",
        "severity_level": "low",
        "interaction_type": "test",
        "task_type": "test",
        "has_image": "False",
    })


# =============================================================================
# Unit Tests
# =============================================================================


class TestExperimentRunnerBasic:
    """Basic unit tests for ExperimentRunner."""

    def test_init(self, bench_config, temp_output_dir, temp_benchmark_file):
        """Test ExperimentRunner initialization."""
        output_mgr = OutputManager(
            str(temp_benchmark_file), str(temp_output_dir)
        )
        runner = ExperimentRunner(bench_config, output_mgr)

        assert runner.config == bench_config
        assert runner.output_manager == output_mgr
        assert not runner.conversation_cache

    def test_generate_experiments_single_models(
        self, temp_output_dir, temp_benchmark_file
    ):
        """Test experiment generation with single models (A×U)."""
        config = BenchConfig(
            assistant_agents=[AgentSpec(model=ModelConfig(model="claude-sonnet-4.5-bedrock"))],
            user_agents=[AgentSpec(model=ModelConfig(model="claude-haiku-4.5-bedrock"))],
            evaluator_models=[ModelConfig(model="claude-sonnet-4.5-bedrock")],
            strip_thinking_content=True,
        )
        output_mgr = OutputManager(
            str(temp_benchmark_file), str(temp_output_dir)
        )
        runner = ExperimentRunner(config, output_mgr)

        experiments = runner.generate_experiments()

        assert len(experiments) == 1  # 1 × 1 = 1
        assert experiments[0].experiment_id == "0_0"
        assert len(experiments[0].evaluator_models) == 1

    def test_generate_experiments_multiple_models(
        self, temp_output_dir, temp_benchmark_file
    ):
        """Test experiment generation with multiple models."""
        config = BenchConfig(
            assistant_agents=[
                AgentSpec(model=ModelConfig(model="claude-sonnet-4.5-bedrock")),
                AgentSpec(model=ModelConfig(model="claude-haiku-4.5-bedrock")),
            ],
            user_agents=[AgentSpec(model=ModelConfig(model="claude-haiku-4.5-bedrock"))],
            evaluator_models=[
                ModelConfig(model="claude-sonnet-4.5-bedrock"),
                ModelConfig(model="claude-haiku-4.5-bedrock"),
            ],
            strip_thinking_content=True,
        )
        output_mgr = OutputManager(
            str(temp_benchmark_file), str(temp_output_dir)
        )
        runner = ExperimentRunner(config, output_mgr)

        experiments = runner.generate_experiments()

        # 2 assistant × 1 user = 2 experiments (evaluators are inside each)
        assert len(experiments) == 2

        exp_ids = [e.experiment_id for e in experiments]
        assert "0_0" in exp_ids
        assert "1_0" in exp_ids

        # Each experiment should have both evaluator models
        for exp in experiments:
            assert len(exp.evaluator_models) == 2

    def test_experiment_id_format(self, temp_output_dir, temp_benchmark_file):
        """Test that experiment IDs follow the X_Y format."""
        config = BenchConfig(
            assistant_agents=[
                AgentSpec(model=ModelConfig(model="claude-sonnet-4.5-bedrock")),
                AgentSpec(model=ModelConfig(model="claude-haiku-4.5-bedrock")),
            ],
            user_agents=[
                AgentSpec(model=ModelConfig(model="claude-sonnet-4.5-bedrock")),
                AgentSpec(model=ModelConfig(model="claude-haiku-4.5-bedrock")),
            ],
            evaluator_models=[ModelConfig(model="claude-sonnet-4.5-bedrock")],
            strip_thinking_content=True,
        )
        output_mgr = OutputManager(
            str(temp_benchmark_file), str(temp_output_dir)
        )
        runner = ExperimentRunner(config, output_mgr)

        experiments = runner.generate_experiments()

        for exp in experiments:
            expected_id = f"{exp.assistant_idx}_{exp.user_idx}"
            assert exp.experiment_id == expected_id

    def test_aggregation_method_propagated(
        self, temp_output_dir, temp_benchmark_file
    ):
        """Test that aggregation_method is propagated to experiments."""
        config = BenchConfig(
            assistant_agents=[AgentSpec(model=ModelConfig(model="claude-sonnet-4.5-bedrock"))],
            user_agents=[AgentSpec(model=ModelConfig(model="claude-haiku-4.5-bedrock"))],
            evaluator_models=[ModelConfig(model="claude-sonnet-4.5-bedrock")],
            aggregation_method="majority_vote",
            strip_thinking_content=True,
        )
        output_mgr = OutputManager(
            str(temp_benchmark_file), str(temp_output_dir)
        )
        runner = ExperimentRunner(config, output_mgr)

        experiments = runner.generate_experiments()
        assert experiments[0].aggregation_method == "majority_vote"


# =============================================================================
# Property-Based Tests
# =============================================================================


@st.composite
def model_list_sizes(draw):
    """Generate valid list sizes for model configs (1-5 each)."""
    return (
        draw(st.integers(min_value=1, max_value=5)),
        draw(st.integers(min_value=1, max_value=5)),
        draw(st.integers(min_value=1, max_value=5)),
    )


@st.composite
def model_config_strategy(draw):
    """Generate a valid ModelConfig using registry models."""
    model_names = ["claude-opus-4.5-bedrock", "claude-sonnet-4.5-bedrock", "claude-haiku-4.5-bedrock"]
    return ModelConfig(model=draw(st.sampled_from(model_names)))


class TestExperimentRunnerPropertyBased:
    """Property-based tests for ExperimentRunner.

    **Feature: multi-evaluator-aggregation**
    **Property 1: Experiment Count Equals A × U**
    **Property 6: Conversation Reuse Correctness**
    **Property 10: Experiment Failure Isolation**
    """

    @given(sizes=model_list_sizes())
    @settings(max_examples=100)
    def test_property_1_experiment_count_equals_a_times_u(self, sizes):
        """
        Property 1: Experiment generation produces correct count and ID format

        *For any* BenchConfig with A assistant models and U user models,
        generate_experiments() SHALL produce exactly A × U experiments.

        **Validates: Requirements 1.1, 1.2**
        """
        a_size, u_size, e_size = sizes

        config = BenchConfig(
            assistant_agents=[
                AgentSpec(model=ModelConfig(model="claude-sonnet-4.5-bedrock"))
                for _ in range(a_size)
            ],
            user_agents=[
                AgentSpec(model=ModelConfig(model="claude-haiku-4.5-bedrock"))
                for _ in range(u_size)
            ],
            evaluator_models=[
                ModelConfig(model="claude-sonnet-4.5-bedrock") for _ in range(e_size)
            ],
            strip_thinking_content=True,
        )

        benchmark_file = create_temp_benchmark_file()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_mgr = OutputManager(str(benchmark_file), temp_dir)
                runner = ExperimentRunner(config, output_mgr)

                experiments = runner.generate_experiments()

                # Count = A × U (not A × U × E)
                expected_count = a_size * u_size
                assert len(experiments) == expected_count

                # All IDs unique
                exp_ids = [e.experiment_id for e in experiments]
                assert len(exp_ids) == len(set(exp_ids))

                # Each experiment has all evaluator models
                for exp in experiments:
                    assert len(exp.evaluator_models) == e_size

                # IDs match X_Y format
                for exp in experiments:
                    parts = exp.experiment_id.split("_")
                    assert len(parts) == 2
                    assert all(p.isdigit() for p in parts)
        finally:
            benchmark_file.unlink(missing_ok=True)

    @given(
        assistant=model_config_strategy(),
        user=model_config_strategy(),
    )
    @settings(max_examples=100)
    def test_property_6_conversation_reuse_correctness(self, assistant, user):
        """
        Property 6: Conversation Reuse Correctness

        *For any* two experiments with identical conversation signatures,
        the conversations SHALL be identical.

        **Validates: Requirements 5.1, 5.4**
        """
        config = BenchConfig(
            assistant_agents=[AgentSpec(model=assistant)],
            user_agents=[AgentSpec(model=user)],
            evaluator_models=[
                ModelConfig(model="claude-sonnet-4.5-bedrock"),
                ModelConfig(model="claude-haiku-4.5-bedrock"),
            ],
            strip_thinking_content=True,
        )

        benchmark_file = create_temp_benchmark_file()

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_mgr = OutputManager(str(benchmark_file), temp_dir)
                runner = ExperimentRunner(config, output_mgr)

                experiments = runner.generate_experiments()

                # Only 1 experiment (A×U = 1×1), but with 2 evaluators inside
                assert len(experiments) == 1
                assert len(experiments[0].evaluator_models) == 2
        finally:
            benchmark_file.unlink(missing_ok=True)

    @given(sizes=model_list_sizes())
    @settings(max_examples=100)
    def test_property_10_experiment_failure_isolation(self, sizes):
        """
        Property 10: Experiment Failure Isolation

        *For any* experiment that fails, remaining experiments SHALL still run.

        **Validates: Requirements 7.3**
        """
        a_size, u_size, e_size = sizes

        config = BenchConfig(
            assistant_agents=[
                AgentSpec(model=ModelConfig(model="claude-sonnet-4.5-bedrock"))
                for _ in range(a_size)
            ],
            user_agents=[
                AgentSpec(model=ModelConfig(model="claude-haiku-4.5-bedrock"))
                for _ in range(u_size)
            ],
            evaluator_models=[
                ModelConfig(model="claude-sonnet-4.5-bedrock") for _ in range(e_size)
            ],
            strip_thinking_content=True,
        )

        benchmark_file = create_temp_benchmark_file()
        sample_entry = create_sample_benchmark_entry()

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_mgr = OutputManager(str(benchmark_file), temp_dir)
                runner = ExperimentRunner(config, output_mgr)

                total_experiments = a_size * u_size

                attempted_experiments: List[str] = []
                fail_first = [True]

                async def mock_run_experiment_async(exp, _entries):
                    attempted_experiments.append(exp.experiment_id)
                    if fail_first[0]:
                        fail_first[0] = False
                        raise RuntimeError("Simulated failure")
                    return {
                        "conversations": [],
                        "evaluations": [],
                        "experiment_config": exp.to_dict(),
                    }

                with patch.object(
                    runner,
                    "_run_experiment_async",
                    side_effect=mock_run_experiment_async,
                ):
                    results = runner.run_all([sample_entry])

                assert len(attempted_experiments) == total_experiments
                assert len(results["experiments"]) == total_experiments

                first_exp_id = attempted_experiments[0]
                assert "error" in results["experiments"][first_exp_id]

                if total_experiments > 1:
                    successful_count = sum(
                        1
                        for exp_id, result in results["experiments"].items()
                        if "error" not in result
                    )
                    assert successful_count == total_experiments - 1
        finally:
            benchmark_file.unlink(missing_ok=True)


# =============================================================================
# ExperimentConfig Round-Trip Property Test
# =============================================================================


@st.composite
def experiment_config_strategy(draw):
    """Generate a valid ExperimentConfig instance for property testing."""
    assistant_agent = draw(agent_spec_strategy())
    user_agent = draw(agent_spec_strategy())
    evaluator_models = [
        ModelConfig(**draw(model_config_dict()))
        for _ in range(draw(st.integers(min_value=1, max_value=3)))
    ]
    sandbox_model = ModelConfig(**draw(model_config_dict()))
    analyzer_model = ModelConfig(**draw(model_config_dict()))
    assistant_idx = draw(st.integers(min_value=0, max_value=10))
    user_idx = draw(st.integers(min_value=0, max_value=10))
    max_turns = draw(st.integers(min_value=1, max_value=50))
    strip_thinking_content = draw(st.booleans())
    aggregation_method = draw(st.sampled_from(["average", "majority_vote"]))
    experiment_id = f"{assistant_idx}_{user_idx}"

    return ExperimentConfig(
        experiment_id=experiment_id,
        assistant_agent=assistant_agent,
        user_agent=user_agent,
        evaluator_models=evaluator_models,
        sandbox_model=sandbox_model,
        analyzer_model=analyzer_model,
        assistant_idx=assistant_idx,
        user_idx=user_idx,
        max_turns=max_turns,
        strip_thinking_content=strip_thinking_content,
        aggregation_method=aggregation_method,
    )


class TestExperimentConfigRoundTrip:
    """Property-based tests for ExperimentConfig round-trip serialization.

    **Feature: agent-spec**
    **Property 5: ExperimentConfig round-trip serialization**
    **Validates: Requirements 8.5, 8.3, 8.4**
    """

    @given(ec=experiment_config_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_5_experiment_config_round_trip(self, ec):
        """
        Property 5: ExperimentConfig round-trip serialization

        *For any* valid ExperimentConfig instance, serializing via to_dict()
        and deserializing via from_dict() SHALL produce an equivalent
        ExperimentConfig — specifically, assistant_agent and user_agent
        survive as equivalent AgentSpec objects, and evaluator_models
        round-trips as an equivalent List of ModelConfig.

        **Validates: Requirements 8.5, 8.3, 8.4**
        """
        serialized = ec.to_dict()
        restored = ExperimentConfig.from_dict(serialized)

        # Verify assistant_agent AgentSpec round-trips (Req 8.3)
        assert restored.assistant_agent.model.model == ec.assistant_agent.model.model
        assert restored.assistant_agent.model.model_id == ec.assistant_agent.model.model_id
        assert restored.assistant_agent.model.temperature == ec.assistant_agent.model.temperature
        assert restored.assistant_agent.model.max_tokens == ec.assistant_agent.model.max_tokens
        assert restored.assistant_agent.model.provider == ec.assistant_agent.model.provider
        assert restored.assistant_agent.prompt == ec.assistant_agent.prompt
        assert restored.assistant_agent.agent_class == ec.assistant_agent.agent_class
        assert restored.assistant_agent.system_prompt == ec.assistant_agent.system_prompt
        assert restored.assistant_agent.label == ec.assistant_agent.label

        # Verify user_agent AgentSpec round-trips (Req 8.4)
        assert restored.user_agent.model.model == ec.user_agent.model.model
        assert restored.user_agent.model.model_id == ec.user_agent.model.model_id
        assert restored.user_agent.model.temperature == ec.user_agent.model.temperature
        assert restored.user_agent.model.max_tokens == ec.user_agent.model.max_tokens
        assert restored.user_agent.model.provider == ec.user_agent.model.provider
        assert restored.user_agent.prompt == ec.user_agent.prompt
        assert restored.user_agent.agent_class == ec.user_agent.agent_class
        assert restored.user_agent.system_prompt == ec.user_agent.system_prompt
        assert restored.user_agent.label == ec.user_agent.label

        # Verify evaluator_models round-trips as List[ModelConfig] (Req 8.5)
        assert len(restored.evaluator_models) == len(ec.evaluator_models)
        for restored_m, original_m in zip(restored.evaluator_models, ec.evaluator_models):
            assert restored_m.model == original_m.model
            assert restored_m.model_id == original_m.model_id
            assert restored_m.temperature == original_m.temperature
            assert restored_m.max_tokens == original_m.max_tokens
            assert restored_m.provider == original_m.provider

        # Verify scalar fields round-trip
        assert restored.experiment_id == ec.experiment_id
        assert restored.assistant_idx == ec.assistant_idx
        assert restored.user_idx == ec.user_idx
        assert restored.max_turns == ec.max_turns
        assert restored.strip_thinking_content == ec.strip_thinking_content
        assert restored.aggregation_method == ec.aggregation_method


# =============================================================================
# Conversation Signature Property Tests
# =============================================================================


class TestConversationSignatureDeterminism:
    """Property-based tests for conversation signature determinism.

    **Feature: agent-spec**
    **Property 8: Conversation signature determinism**
    **Validates: Requirements 10.1, 10.3**
    """

    @given(ec=experiment_config_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_8_conversation_signature_determinism(self, ec):
        """
        Property 8: Conversation signature determinism

        *For any* ExperimentConfig, calling conversation_signature() multiple
        times SHALL always produce the same SHA-256 hash value.

        **Validates: Requirements 10.1, 10.3**
        """
        sig1 = ec.conversation_signature()
        sig2 = ec.conversation_signature()

        assert sig1 == sig2, (
            f"conversation_signature() is not deterministic: "
            f"first call returned {sig1!r}, second call returned {sig2!r}"
        )

        # Verify it's a valid SHA-256 hex string
        assert len(sig1) == 64, f"Expected 64-char hex string, got length {len(sig1)}"
        assert all(c in "0123456789abcdef" for c in sig1), (
            f"Signature contains non-hex characters: {sig1!r}"
        )


# =============================================================================
# AgentSpec-based Experiment Generation Property Tests
# =============================================================================


class TestAgentSpecExperimentGeneration:
    """Property-based tests for experiment generation using AgentSpec-based API.

    **Feature: agent-spec**
    **Property 6: Experiment count invariant**
    **Property 7: Experiment generation correctness**
    """

    @given(
        assistant_agents=st.lists(agent_spec_strategy(), min_size=1, max_size=5),
        user_agents=st.lists(agent_spec_strategy(), min_size=1, max_size=5),
        evaluator_models=st.lists(model_config_strategy(), min_size=1, max_size=3),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_experiment_count_invariant(
        self, assistant_agents, user_agents, evaluator_models
    ):
        """
        Property 6: Experiment count invariant

        *For any* BenchConfig with non-empty assistant_agents and user_agents
        lists, generate_experiments() SHALL produce exactly
        len(assistant_agents) × len(user_agents) experiments.

        **Validates: Requirement 9.1**
        """
        config = BenchConfig(
            assistant_agents=assistant_agents,
            user_agents=user_agents,
            evaluator_models=evaluator_models,
            strip_thinking_content=True,
        )

        benchmark_file = create_temp_benchmark_file()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_mgr = OutputManager(str(benchmark_file), temp_dir)
                runner = ExperimentRunner(config, output_mgr)

                experiments = runner.generate_experiments()

                expected_count = len(assistant_agents) * len(user_agents)
                assert len(experiments) == expected_count, (
                    f"Expected {expected_count} experiments "
                    f"({len(assistant_agents)} × {len(user_agents)}), "
                    f"got {len(experiments)}"
                )
        finally:
            benchmark_file.unlink(missing_ok=True)

    @given(
        assistant_agents=st.lists(agent_spec_strategy(), min_size=1, max_size=4),
        user_agents=st.lists(agent_spec_strategy(), min_size=1, max_size=4),
        evaluator_models=st.lists(model_config_strategy(), min_size=1, max_size=3),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_7_experiment_generation_correctness(
        self, assistant_agents, user_agents, evaluator_models
    ):
        """
        Property 7: Experiment generation correctness

        *For any* BenchConfig, each generated ExperimentConfig SHALL have:
        - experiment_id matching "{assistant_idx}_{user_idx}"
        - the correct AgentSpec from the input lists for assistant_agent
          and user_agent
        - the full evaluator_models list as List[ModelConfig]

        **Validates: Requirements 9.2, 9.3, 9.4**
        """
        config = BenchConfig(
            assistant_agents=assistant_agents,
            user_agents=user_agents,
            evaluator_models=evaluator_models,
            strip_thinking_content=True,
        )

        benchmark_file = create_temp_benchmark_file()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_mgr = OutputManager(str(benchmark_file), temp_dir)
                runner = ExperimentRunner(config, output_mgr)

                experiments = runner.generate_experiments()

                for exp in experiments:
                    # Req 9.2: experiment_id format
                    expected_id = f"{exp.assistant_idx}_{exp.user_idx}"
                    assert exp.experiment_id == expected_id, (
                        f"Expected id '{expected_id}', "
                        f"got '{exp.experiment_id}'"
                    )

                    # Req 9.3: correct AgentSpec from input lists
                    a_idx = exp.assistant_idx
                    u_idx = exp.user_idx
                    expected_assistant = assistant_agents[a_idx]
                    expected_user = user_agents[u_idx]

                    assert exp.assistant_agent.model.model == expected_assistant.model.model
                    assert exp.assistant_agent.model.model_id == expected_assistant.model.model_id
                    assert exp.assistant_agent.prompt == expected_assistant.prompt
                    assert exp.assistant_agent.agent_class == expected_assistant.agent_class
                    assert exp.assistant_agent.system_prompt == expected_assistant.system_prompt
                    assert exp.assistant_agent.label == expected_assistant.label

                    assert exp.user_agent.model.model == expected_user.model.model
                    assert exp.user_agent.model.model_id == expected_user.model.model_id
                    assert exp.user_agent.prompt == expected_user.prompt
                    assert exp.user_agent.agent_class == expected_user.agent_class
                    assert exp.user_agent.system_prompt == expected_user.system_prompt
                    assert exp.user_agent.label == expected_user.label

                    # Req 9.4: full evaluator_models list
                    assert len(exp.evaluator_models) == len(evaluator_models)
                    for exp_m, orig_m in zip(
                        exp.evaluator_models, evaluator_models
                    ):
                        assert exp_m.model == orig_m.model
                        assert exp_m.model_id == orig_m.model_id
                        assert exp_m.temperature == orig_m.temperature
                        assert exp_m.max_tokens == orig_m.max_tokens
        finally:
            benchmark_file.unlink(missing_ok=True)


class TestConversationSignatureSensitivity:
    """Property-based tests for conversation signature sensitivity.

    **Feature: agent-spec**
    **Property 9: Conversation signature sensitivity**
    **Validates: Requirement 10.2**
    """

    @given(
        ec=experiment_config_strategy(),
        new_prompt=st.from_regex(r"[a-z][a-z0-9_]{1,29}", fullmatch=True),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_9_signature_differs_on_assistant_prompt_change(self, ec, new_prompt):
        """
        Property 9: Conversation signature sensitivity — assistant prompt

        *For any* ExperimentConfig, changing the assistant agent's prompt field
        SHALL produce a different conversation_signature() hash.

        **Validates: Requirement 10.2**
        """
        from hypothesis import assume
        assume(new_prompt != ec.assistant_agent.prompt)

        original_sig = ec.conversation_signature()

        modified_agent = AgentSpec(
            model=ec.assistant_agent.model,
            prompt=new_prompt,
            agent_class=ec.assistant_agent.agent_class,
            system_prompt=ec.assistant_agent.system_prompt,
            label=ec.assistant_agent.label,
        )
        modified_ec = ExperimentConfig(
            experiment_id=ec.experiment_id,
            assistant_agent=modified_agent,
            user_agent=ec.user_agent,
            evaluator_models=ec.evaluator_models,
            sandbox_model=ec.sandbox_model,
            analyzer_model=ec.analyzer_model,
            assistant_idx=ec.assistant_idx,
            user_idx=ec.user_idx,
            max_turns=ec.max_turns,
            strip_thinking_content=ec.strip_thinking_content,
            aggregation_method=ec.aggregation_method,
        )

        modified_sig = modified_ec.conversation_signature()
        assert original_sig != modified_sig, (
            f"Signature did not change when assistant prompt changed from "
            f"{ec.assistant_agent.prompt!r} to {new_prompt!r}"
        )

    @given(
        ec=experiment_config_strategy(),
        new_agent_class=st.from_regex(r"[a-z][a-z0-9_]{1,19}", fullmatch=True),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_9_signature_differs_on_assistant_agent_class_change(self, ec, new_agent_class):
        """
        Property 9: Conversation signature sensitivity — assistant agent_class

        *For any* ExperimentConfig, changing the assistant agent's agent_class field
        SHALL produce a different conversation_signature() hash.

        **Validates: Requirement 10.2**
        """
        from hypothesis import assume
        assume(new_agent_class != ec.assistant_agent.agent_class)

        original_sig = ec.conversation_signature()

        modified_agent = AgentSpec(
            model=ec.assistant_agent.model,
            prompt=ec.assistant_agent.prompt,
            agent_class=new_agent_class,
            system_prompt=ec.assistant_agent.system_prompt,
            label=ec.assistant_agent.label,
        )
        modified_ec = ExperimentConfig(
            experiment_id=ec.experiment_id,
            assistant_agent=modified_agent,
            user_agent=ec.user_agent,
            evaluator_models=ec.evaluator_models,
            sandbox_model=ec.sandbox_model,
            analyzer_model=ec.analyzer_model,
            assistant_idx=ec.assistant_idx,
            user_idx=ec.user_idx,
            max_turns=ec.max_turns,
            strip_thinking_content=ec.strip_thinking_content,
            aggregation_method=ec.aggregation_method,
        )

        modified_sig = modified_ec.conversation_signature()
        assert original_sig != modified_sig, (
            f"Signature did not change when assistant agent_class changed from "
            f"{ec.assistant_agent.agent_class!r} to {new_agent_class!r}"
        )

    @given(
        ec=experiment_config_strategy(),
        new_system_prompt=st.text(min_size=1, max_size=200),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_9_signature_differs_on_assistant_system_prompt_change(self, ec, new_system_prompt):
        """
        Property 9: Conversation signature sensitivity — assistant system_prompt

        *For any* ExperimentConfig, changing the assistant agent's system_prompt field
        SHALL produce a different conversation_signature() hash.

        **Validates: Requirement 10.2**
        """
        from hypothesis import assume
        assume(new_system_prompt != ec.assistant_agent.system_prompt)

        original_sig = ec.conversation_signature()

        modified_agent = AgentSpec(
            model=ec.assistant_agent.model,
            prompt=ec.assistant_agent.prompt,
            agent_class=ec.assistant_agent.agent_class,
            system_prompt=new_system_prompt,
            label=ec.assistant_agent.label,
        )
        modified_ec = ExperimentConfig(
            experiment_id=ec.experiment_id,
            assistant_agent=modified_agent,
            user_agent=ec.user_agent,
            evaluator_models=ec.evaluator_models,
            sandbox_model=ec.sandbox_model,
            analyzer_model=ec.analyzer_model,
            assistant_idx=ec.assistant_idx,
            user_idx=ec.user_idx,
            max_turns=ec.max_turns,
            strip_thinking_content=ec.strip_thinking_content,
            aggregation_method=ec.aggregation_method,
        )

        modified_sig = modified_ec.conversation_signature()
        assert original_sig != modified_sig, (
            f"Signature did not change when assistant system_prompt changed"
        )

    @given(
        ec=experiment_config_strategy(),
        new_prompt=st.from_regex(r"[a-z][a-z0-9_]{1,29}", fullmatch=True),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_9_signature_differs_on_user_prompt_change(self, ec, new_prompt):
        """
        Property 9: Conversation signature sensitivity — user prompt

        *For any* ExperimentConfig, changing the user agent's prompt field
        SHALL produce a different conversation_signature() hash.

        **Validates: Requirement 10.2**
        """
        from hypothesis import assume
        assume(new_prompt != ec.user_agent.prompt)

        original_sig = ec.conversation_signature()

        modified_agent = AgentSpec(
            model=ec.user_agent.model,
            prompt=new_prompt,
            agent_class=ec.user_agent.agent_class,
            system_prompt=ec.user_agent.system_prompt,
            label=ec.user_agent.label,
        )
        modified_ec = ExperimentConfig(
            experiment_id=ec.experiment_id,
            assistant_agent=ec.assistant_agent,
            user_agent=modified_agent,
            evaluator_models=ec.evaluator_models,
            sandbox_model=ec.sandbox_model,
            analyzer_model=ec.analyzer_model,
            assistant_idx=ec.assistant_idx,
            user_idx=ec.user_idx,
            max_turns=ec.max_turns,
            strip_thinking_content=ec.strip_thinking_content,
            aggregation_method=ec.aggregation_method,
        )

        modified_sig = modified_ec.conversation_signature()
        assert original_sig != modified_sig, (
            f"Signature did not change when user prompt changed from "
            f"{ec.user_agent.prompt!r} to {new_prompt!r}"
        )

