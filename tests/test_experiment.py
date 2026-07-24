# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for ExperimentConfig data model.

Tests the experiment configuration dataclass including property-based tests
for experiment ID format and conversation signature consistency.
"""

from dataclasses import fields as dataclass_fields

import pytest
from hypothesis import given, strategies as st, settings

from patient_agent_bench.config import AgentSpec, ModelConfig, parse_model_config
from patient_agent_bench.runner.experiment_config import ExperimentConfig


# =============================================================================
# Unit Tests
# =============================================================================


class TestExperimentConfig:
    """Unit tests for ExperimentConfig dataclass."""

    def test_basic_creation(self):
        """Test basic ExperimentConfig creation."""
        assistant = AgentSpec(model=ModelConfig(model="claude-sonnet-4.5-bedrock"))
        user = AgentSpec(model=ModelConfig(model="claude-haiku-4.5-bedrock"))
        evaluators = [ModelConfig(model="claude-sonnet-4.5-bedrock")]
        sandbox = ModelConfig(model="claude-haiku-4.5-bedrock")
        analyzer = ModelConfig(model="claude-sonnet-4.5-bedrock")

        exp = ExperimentConfig(
            experiment_id="0_0",
            assistant_agent=assistant,
            user_agent=user,
            evaluator_models=evaluators,
            sandbox_model=sandbox,
            analyzer_model=analyzer,
            assistant_idx=0,
            user_idx=0,
            max_turns=3,
            strip_thinking_content=True,
        )

        assert exp.experiment_id == "0_0"
        assert exp.assistant_idx == 0
        assert exp.user_idx == 0
        assert len(exp.evaluator_models) == 1

    def test_to_dict(self):
        """Test ExperimentConfig serialization."""
        assistant = AgentSpec(model=ModelConfig(model="claude-sonnet-4.5-bedrock"))
        user = AgentSpec(model=ModelConfig(model="claude-haiku-4.5-bedrock"))
        evaluators = [
            ModelConfig(model="claude-sonnet-4.5-bedrock"),
            ModelConfig(model="claude-haiku-4.5-bedrock"),
        ]
        sandbox = ModelConfig(model="claude-haiku-4.5-bedrock")
        analyzer = ModelConfig(model="claude-sonnet-4.5-bedrock")

        exp = ExperimentConfig(
            experiment_id="1_2",
            assistant_agent=assistant,
            user_agent=user,
            evaluator_models=evaluators,
            sandbox_model=sandbox,
            analyzer_model=analyzer,
            assistant_idx=1,
            user_idx=2,
            max_turns=3,
            strip_thinking_content=True,
        )

        d = exp.to_dict()

        assert d["experiment_id"] == "1_2"
        assert d["assistant_idx"] == 1
        assert d["user_idx"] == 2
        assert "evaluator_models" in d
        assert len(d["evaluator_models"]) == 2
        assert "assistant_agent" in d
        assert "user_agent" in d
        assert "sandbox_model" in d
        assert "analyzer_model" in d
        assert d["aggregation_method"] == "average"

    def test_from_dict(self):
        """Test ExperimentConfig deserialization."""
        data = {
            "experiment_id": "2_1",
            "assistant_agent": {
                "model": {
                    "model": "claude-sonnet-4.5-bedrock",
                    "model_id": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
            },
            "user_agent": {
                "model": {
                    "model": "claude-haiku-4.5-bedrock",
                    "model_id": "global.anthropic.claude-haiku-4-5-20250929-v1:0",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
            },
            "evaluator_models": [
                {
                    "model": "claude-sonnet-4.5-bedrock",
                    "model_id": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
                    "temperature": 0.0,
                    "max_tokens": 4096,
                },
                {
                    "model": "claude-haiku-4.5-bedrock",
                    "model_id": "global.anthropic.claude-haiku-4-5-20250929-v1:0",
                    "temperature": 0.0,
                    "max_tokens": 4096,
                },
            ],
            "sandbox_model": {
                "model": "claude-haiku-4.5-bedrock",
                "model_id": "global.anthropic.claude-haiku-4-5-20250929-v1:0",
                "temperature": 0.5,
                "max_tokens": 8192,
            },
            "analyzer_model": {
                "model": "claude-sonnet-4.5-bedrock",
                "model_id": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "temperature": 0.0,
                "max_tokens": 4096,
            },
            "assistant_idx": 2,
            "user_idx": 1,
            "aggregation_method": "majority_vote",
        }

        exp = ExperimentConfig.from_dict(data)

        assert exp.experiment_id == "2_1"
        assert exp.assistant_idx == 2
        assert exp.user_idx == 1
        assert len(exp.evaluator_models) == 2
        assert exp.aggregation_method == "majority_vote"

    def test_from_dict_legacy_format(self):
        """Test ExperimentConfig deserialization from legacy format."""
        data = {
            "experiment_id": "0_0_0",
            "assistant_agent": {"model": {"model": "claude-sonnet-4.5-bedrock"}},
            "user_agent": {"model": {"model": "claude-haiku-4.5-bedrock"}},
            "evaluator_model": {"model": "claude-sonnet-4.5-bedrock"},
            "sandbox_model": {"model": "claude-haiku-4.5-bedrock"},
            "analyzer_model": {"model": "claude-sonnet-4.5-bedrock"},
            "assistant_idx": 0,
            "user_idx": 0,
        }

        exp = ExperimentConfig.from_dict(data)

        assert exp.experiment_id == "0_0_0"
        assert len(exp.evaluator_models) == 1
        assert exp.aggregation_method == "average"

    def test_from_dict_preserves_all_custom_model_fields(self):
        """
        Test from_dict() preserves every custom ModelConfig field for the roles
        parsed directly (evaluator / sandbox / analyzer), not just the ones
        routed through AgentSpec.

        Regression: from_dict() used to carry its own reduced copy of
        parse_model_config that silently dropped developer, auth, api_prefix,
        use_responses_api, base_url_env, api_key_env, bedrock_invoke_provider
        and region — so `evaluate --run-dir` rebuilt these roles on the wrong
        channel.
        """
        spec = {
            "model_id": "arn:aws:bedrock:us-east-2:123456789012:imported-model/abc",
            "temperature": 0.2,
            "max_tokens": 2048,
            "provider": "bedrock",
            "developer": "qwen",
            "auth": "api_key",
            "api_prefix": "/v1",
            "use_responses_api": True,
            "thinking_budget": 1024,
            "reasoning_effort": "low",
            "additional_fields": {"custom": 1},
            "thinking_prompt_suffix": " /no_think",
            "base_url_env": "FOO_BASE_URL",
            "api_key_env": "FOO_API_KEY",
            "bedrock_invoke_provider": "qwen",
            "region": "us-east-2",
        }
        agent = {"model": spec, "prompt": "default_prompt", "agent_class": "default"}
        data = {
            "experiment_id": "0_0",
            "assistant_agent": agent,
            "user_agent": agent,
            "evaluator_models": [spec],
            "sandbox_model": spec,
            "analyzer_model": spec,
            "assistant_idx": 0,
            "user_idx": 0,
        }

        exp = ExperimentConfig.from_dict(data)
        expected = parse_model_config(spec)

        for restored in (
            exp.evaluator_models[0],
            exp.sandbox_model,
            exp.analyzer_model,
            exp.assistant_model,
            exp.user_model,
        ):
            for f in dataclass_fields(ModelConfig):
                assert getattr(restored, f.name) == getattr(expected, f.name), (
                    f"field '{f.name}' not preserved through from_dict()"
                )

    def test_conversation_signature_deterministic(self):
        """Test that conversation signature is deterministic."""
        exp = ExperimentConfig(
            experiment_id="0_0",
            assistant_agent=AgentSpec(model=ModelConfig(model="claude-sonnet-4.5-bedrock")),
            user_agent=AgentSpec(model=ModelConfig(model="claude-haiku-4.5-bedrock")),
            evaluator_models=[ModelConfig(model="claude-sonnet-4.5-bedrock")],
            sandbox_model=ModelConfig(model="claude-haiku-4.5-bedrock"),
            analyzer_model=ModelConfig(model="claude-sonnet-4.5-bedrock"),
            assistant_idx=0,
            user_idx=0,
            max_turns=3,
            strip_thinking_content=True,
        )

        sig1 = exp.conversation_signature()
        sig2 = exp.conversation_signature()

        assert sig1 == sig2
        assert len(sig1) == 64

    def test_conversation_signature_ignores_evaluator(self):
        """Test that conversation signature ignores evaluator models."""
        base_kwargs = dict(
            assistant_agent=AgentSpec(model=ModelConfig(model="claude-sonnet-4.5-bedrock")),
            user_agent=AgentSpec(model=ModelConfig(model="claude-haiku-4.5-bedrock")),
            sandbox_model=ModelConfig(model="claude-haiku-4.5-bedrock"),
            analyzer_model=ModelConfig(model="claude-sonnet-4.5-bedrock"),
            assistant_idx=0,
            user_idx=0,
            max_turns=3,
            strip_thinking_content=True,
        )

        exp1 = ExperimentConfig(
            experiment_id="0_0",
            evaluator_models=[ModelConfig(model="claude-sonnet-4.5-bedrock")],
            **base_kwargs,
        )

        exp2 = ExperimentConfig(
            experiment_id="0_0",
            evaluator_models=[
                ModelConfig(model="claude-sonnet-4.5-bedrock"),
                ModelConfig(model="claude-haiku-4.5-bedrock"),
            ],
            **base_kwargs,
        )

        assert exp1.conversation_signature() == exp2.conversation_signature()

    def test_conversation_signature_differs_for_different_models(self):
        """Test that different assistant/user models produce different signatures."""
        base_kwargs = dict(
            evaluator_models=[ModelConfig(model="claude-sonnet-4.5-bedrock")],
            sandbox_model=ModelConfig(model="claude-haiku-4.5-bedrock"),
            analyzer_model=ModelConfig(model="claude-sonnet-4.5-bedrock"),
            user_idx=0,
            max_turns=3,
            strip_thinking_content=True,
        )

        exp1 = ExperimentConfig(
            experiment_id="0_0",
            assistant_agent=AgentSpec(model=ModelConfig(model="claude-sonnet-4.5-bedrock")),
            user_agent=AgentSpec(model=ModelConfig(model="claude-haiku-4.5-bedrock")),
            assistant_idx=0,
            **base_kwargs,
        )

        exp2 = ExperimentConfig(
            experiment_id="1_0",
            assistant_agent=AgentSpec(model=ModelConfig(model="claude-haiku-4.5-bedrock")),
            user_agent=AgentSpec(model=ModelConfig(model="claude-haiku-4.5-bedrock")),
            assistant_idx=1,
            **base_kwargs,
        )

        assert exp1.conversation_signature() != exp2.conversation_signature()


# =============================================================================
# Property-Based Tests
# =============================================================================


@st.composite
def valid_indices(draw):
    """Generate valid indices for experiment ID."""
    return (
        draw(st.integers(min_value=0, max_value=99)),
        draw(st.integers(min_value=0, max_value=99)),
    )


@st.composite
def model_config_strategy(draw):
    """Generate a valid ModelConfig using registry models."""
    model_names = ["claude-opus-4.5-bedrock", "claude-sonnet-4.5-bedrock", "claude-haiku-4.5-bedrock"]
    return ModelConfig(model=draw(st.sampled_from(model_names)))


@st.composite
def evaluator_models_strategy(draw):
    """Generate a list of evaluator ModelConfigs."""
    return draw(st.lists(model_config_strategy(), min_size=1, max_size=3))


@st.composite
def experiment_config_strategy(draw):
    """Generate a valid ExperimentConfig."""
    a_idx, u_idx = draw(valid_indices())
    return ExperimentConfig(
        experiment_id=f"{a_idx}_{u_idx}",
        assistant_agent=AgentSpec(model=draw(model_config_strategy())),
        user_agent=AgentSpec(model=draw(model_config_strategy())),
        evaluator_models=draw(evaluator_models_strategy()),
        sandbox_model=draw(model_config_strategy()),
        analyzer_model=draw(model_config_strategy()),
        assistant_idx=a_idx,
        user_idx=u_idx,
        aggregation_method=draw(st.sampled_from(["average", "majority_vote"])),
        max_turns=3,
        strip_thinking_content=True,
    )


class TestExperimentConfigPropertyBased:
    """Property-based tests for ExperimentConfig.

    **Feature: multi-evaluator-aggregation**
    **Property 2: ExperimentConfig round-trip serialization**
    **Property 4: Experiment ID Format**
    **Property 5: Conversation Signature Consistency**
    """

    @given(indices=valid_indices())
    @settings(max_examples=100)
    def test_property_4_experiment_id_format(self, indices):
        """
        Property 4: Experiment ID Format

        *For any* generated experiment with indices (a, u), the experiment_id
        SHALL equal the string "{a}_{u}".

        **Validates: Requirements 1.1**
        """
        a_idx, u_idx = indices

        exp = ExperimentConfig(
            experiment_id=f"{a_idx}_{u_idx}",
            assistant_agent=AgentSpec(model=ModelConfig(model="claude-sonnet-4.5-bedrock")),
            user_agent=AgentSpec(model=ModelConfig(model="claude-haiku-4.5-bedrock")),
            evaluator_models=[ModelConfig(model="claude-sonnet-4.5-bedrock")],
            sandbox_model=ModelConfig(model="claude-haiku-4.5-bedrock"),
            analyzer_model=ModelConfig(model="claude-sonnet-4.5-bedrock"),
            assistant_idx=a_idx,
            user_idx=u_idx,
            max_turns=3,
            strip_thinking_content=True,
        )

        expected_id = f"{a_idx}_{u_idx}"
        assert exp.experiment_id == expected_id

        parts = exp.experiment_id.split("_")
        assert len(parts) == 2
        assert int(parts[0]) == a_idx
        assert int(parts[1]) == u_idx

    @given(exp=experiment_config_strategy())
    @settings(max_examples=100)
    def test_property_2_round_trip_serialization(self, exp):
        """
        Property 2: ExperimentConfig round-trip serialization

        *For any* valid ExperimentConfig, serializing via to_dict() and then
        deserializing via from_dict() SHALL produce an equivalent config.

        **Validates: Requirements 2.3, 2.4, 2.5**
        """
        d = exp.to_dict()
        restored = ExperimentConfig.from_dict(d)

        assert restored.experiment_id == exp.experiment_id
        assert restored.assistant_idx == exp.assistant_idx
        assert restored.user_idx == exp.user_idx
        assert len(restored.evaluator_models) == len(exp.evaluator_models)
        assert restored.aggregation_method == exp.aggregation_method
        assert restored.max_turns == exp.max_turns
        assert restored.strip_thinking_content == exp.strip_thinking_content

        # Model configs must survive field-for-field, not just in count.
        for original, rebuilt in [
            (exp.assistant_model, restored.assistant_model),
            (exp.user_model, restored.user_model),
            (exp.sandbox_model, restored.sandbox_model),
            (exp.analyzer_model, restored.analyzer_model),
            *zip(exp.evaluator_models, restored.evaluator_models),
        ]:
            for f in dataclass_fields(ModelConfig):
                assert getattr(rebuilt, f.name) == getattr(original, f.name), (
                    f"field '{f.name}' not preserved through to_dict/from_dict"
                )

    @given(
        assistant=model_config_strategy(),
        user=model_config_strategy(),
        evaluators1=evaluator_models_strategy(),
        evaluators2=evaluator_models_strategy(),
    )
    @settings(max_examples=100)
    def test_property_5_conversation_signature_consistency(
        self, assistant, user, evaluators1, evaluators2
    ):
        """
        Property 5: Conversation Signature Consistency

        *For any* two ExperimentConfigs with identical assistant_model AND
        user_model, their conversation_signature() SHALL return identical values.

        **Validates: Requirements 5.3**
        """
        exp1 = ExperimentConfig(
            experiment_id="0_0",
            assistant_agent=AgentSpec(model=assistant),
            user_agent=AgentSpec(model=user),
            evaluator_models=evaluators1,
            sandbox_model=ModelConfig(model="claude-haiku-4.5-bedrock"),
            analyzer_model=ModelConfig(model="claude-sonnet-4.5-bedrock"),
            assistant_idx=0,
            user_idx=0,
            max_turns=3,
            strip_thinking_content=True,
        )

        exp2 = ExperimentConfig(
            experiment_id="0_0",
            assistant_agent=AgentSpec(model=assistant),
            user_agent=AgentSpec(model=user),
            evaluator_models=evaluators2,
            sandbox_model=ModelConfig(model="claude-haiku-4.5-bedrock"),
            analyzer_model=ModelConfig(model="claude-sonnet-4.5-bedrock"),
            assistant_idx=0,
            user_idx=0,
            max_turns=3,
            strip_thinking_content=True,
        )

        assert exp1.conversation_signature() == exp2.conversation_signature()

    @given(exp=experiment_config_strategy())
    @settings(max_examples=100)
    def test_property_5_signature_determinism(self, exp):
        """
        Property 5 (continued): Signature is deterministic.
        """
        sig1 = exp.conversation_signature()
        sig2 = exp.conversation_signature()

        assert sig1 == sig2
        assert isinstance(sig1, str)
        assert len(sig1) == 64

    @given(exp=experiment_config_strategy())
    @settings(max_examples=100)
    def test_to_dict_contains_all_fields(self, exp):
        """Test that to_dict() contains all required fields."""
        d = exp.to_dict()

        assert "experiment_id" in d
        assert "assistant_agent" in d
        assert "user_agent" in d
        assert "evaluator_models" in d
        assert "assistant_idx" in d
        assert "user_idx" in d
        assert "aggregation_method" in d

        assert d["experiment_id"] == exp.experiment_id
        assert d["assistant_idx"] == exp.assistant_idx
        assert d["user_idx"] == exp.user_idx
        assert isinstance(d["evaluator_models"], list)
        assert len(d["evaluator_models"]) == len(exp.evaluator_models)


class TestExperimentConfigValidation:
    """Tests for ExperimentConfig validation."""

    def test_empty_evaluator_models_raises(self):
        """Test that empty evaluator_models list raises ValueError."""
        with pytest.raises(ValueError, match="at least one evaluator"):
            ExperimentConfig(
                experiment_id="0_0",
                assistant_agent=AgentSpec(model=ModelConfig(model="claude-sonnet-4.5-bedrock")),
                user_agent=AgentSpec(model=ModelConfig(model="claude-haiku-4.5-bedrock")),
                evaluator_models=[],
                sandbox_model=ModelConfig(model="claude-haiku-4.5-bedrock"),
                analyzer_model=ModelConfig(model="claude-sonnet-4.5-bedrock"),
                assistant_idx=0,
                user_idx=0,
                max_turns=3,
                strip_thinking_content=True,
            )

    def test_no_evaluator_idx_field(self):
        """Test that ExperimentConfig does not have evaluator_idx field (Req 2.2)."""
        exp = ExperimentConfig(
            experiment_id="0_0",
            assistant_agent=AgentSpec(model=ModelConfig(model="claude-sonnet-4.5-bedrock")),
            user_agent=AgentSpec(model=ModelConfig(model="claude-haiku-4.5-bedrock")),
            evaluator_models=[ModelConfig(model="claude-sonnet-4.5-bedrock")],
            sandbox_model=ModelConfig(model="claude-haiku-4.5-bedrock"),
            analyzer_model=ModelConfig(model="claude-sonnet-4.5-bedrock"),
            assistant_idx=0,
            user_idx=0,
            max_turns=3,
            strip_thinking_content=True,
        )
        assert not hasattr(exp, "evaluator_idx")
        d = exp.to_dict()
        assert "evaluator_idx" not in d
