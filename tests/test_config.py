# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for configuration management.

Tests ModelConfig, BenchConfig, and AWS credential handling.
"""

import json
import logging
import os
import pytest
from unittest.mock import patch, MagicMock

from patient_agent_bench.config import (
    AgentSpec,
    BenchConfig,
    ModelConfig,
    RolePoolManager,
    DEFAULT_MODEL_NAME,
    check_credentials_valid,
    refresh_credentials_hook,
    assume_target_role,
    ensure_credentials,
    load_prompt,
    format_prompt_safe,
    parse_model_config,
)
from patient_agent_bench.model_registry import get_model_spec


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        """Test ModelConfig with default values (uses registry)."""
        config = ModelConfig()
        spec = get_model_spec(DEFAULT_MODEL_NAME)
        assert config.model_id == spec.model_id
        assert config.temperature == spec.default_temperature
        assert config.max_tokens == spec.default_max_tokens

    def test_registry_model(self):
        """Test ModelConfig with registry model name."""
        config = ModelConfig(model="claude-haiku-4.5-bedrock")
        spec = get_model_spec("claude-haiku-4.5-bedrock")
        assert config.model_id == spec.model_id
        assert config.temperature == spec.default_temperature
        assert config.max_tokens == spec.default_max_tokens

    def test_registry_model_alias(self):
        """Test ModelConfig with registry key."""
        config = ModelConfig(model="claude-sonnet-5-bedrock")
        spec = get_model_spec("claude-sonnet-5-bedrock")
        assert config.model_id == spec.model_id

    def test_custom_model(self):
        """Test ModelConfig with custom model spec."""
        config = ModelConfig(
            model_id="custom-model-id",
            temperature=0.5,
            max_tokens=2048,
        )
        assert config.model_id == "custom-model-id"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048

    def test_custom_model_missing_fields(self):
        """Test custom model fails when fields are missing."""
        with pytest.raises(ValueError, match="missing required fields"):
            ModelConfig(model_id="custom-model", temperature=0.5)

    def test_mixed_format_error(self):
        """Test error when mixing registry and custom formats."""
        with pytest.raises(ValueError, match="Cannot mix"):
            ModelConfig(model="claude-sonnet-5-bedrock", temperature=0.5)

    def test_unknown_registry_model(self):
        """Test error for unknown registry model."""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelConfig(model="nonexistent-model")

    def test_requires_bedrock_by_provider(self):
        """requires_bedrock is True only for the bedrock channel; the API-key
        protocol channels (OpenAI, Anthropic) are False."""
        bedrock = ModelConfig(model_id="x", max_tokens=100, provider="bedrock")
        openai = ModelConfig(model_id="x", max_tokens=100, provider="openai-protocol-api")
        anthropic = ModelConfig(model_id="x", max_tokens=100, provider="anthropic-protocol-api")
        assert bedrock.requires_bedrock is True
        assert openai.requires_bedrock is False
        assert anthropic.requires_bedrock is False
        # is_openai_api stays specific to the OpenAI protocol (back-compat).
        assert anthropic.is_openai_api is False

    def test_anthropic_protocol_channel_builds_chat_anthropic(self, monkeypatch):
        """create_chat_model routes anthropic-protocol-api to ChatAnthropic,
        omitting temperature when None and enabling thinking when a budget is set."""
        from langchain_anthropic import ChatAnthropic
        from patient_agent_bench.config import create_chat_model

        # Dummy non-secret value; the api_key branch only needs it to be set
        # (no real request is made). Deliberately not shaped like a real key so
        # secret scanners don't flag it.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "unit-test-placeholder")
        cfg = ModelConfig(
            model_id="claude-sonnet-5", temperature=None, max_tokens=8192,
            provider="anthropic-protocol-api", auth="api_key", thinking_budget=4096,
        )
        llm = create_chat_model(cfg)  # no bedrock client needed
        assert isinstance(llm, ChatAnthropic)
        assert llm.model == "claude-sonnet-5"
        assert llm.temperature is None  # omitted, not passed as None-rejecting value
        assert llm.thinking == {"type": "enabled", "budget_tokens": 4096}

    def test_anthropic_protocol_requires_api_key(self, monkeypatch):
        """The Anthropic api_key channel raises a clear error when the key env is unset."""
        from patient_agent_bench.config import create_chat_model

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cfg = ModelConfig(
            model_id="claude-sonnet-5", max_tokens=4096,
            provider="anthropic-protocol-api", auth="api_key",
        )
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            create_chat_model(cfg)

    def test_anthropic_protocol_reasoning_effort(self, monkeypatch):
        """reasoning_effort passes through as ChatAnthropic reasoning_effort."""
        from langchain_anthropic import ChatAnthropic
        from patient_agent_bench.config import create_chat_model

        monkeypatch.setenv("ANTHROPIC_API_KEY", "unit-test-placeholder")
        cfg = ModelConfig(
            model_id="claude-sonnet-5", max_tokens=8192,
            provider="anthropic-protocol-api", auth="api_key", reasoning_effort="low",
        )
        llm = create_chat_model(cfg)
        assert isinstance(llm, ChatAnthropic)
        assert llm.reasoning_effort == "low"

    def test_anthropic_protocol_additional_fields_passthrough(self, monkeypatch):
        """additional_fields flow through to ChatAnthropic as top-level kwargs,
        so raw thinking/output_config configs work identically to Bedrock."""
        from langchain_anthropic import ChatAnthropic
        from patient_agent_bench.config import create_chat_model

        monkeypatch.setenv("ANTHROPIC_API_KEY", "unit-test-placeholder")
        cfg = ModelConfig(
            model_id="claude-sonnet-5", max_tokens=16384,
            provider="anthropic-protocol-api", auth="api_key",
            additional_fields={"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}},
        )
        llm = create_chat_model(cfg)
        assert isinstance(llm, ChatAnthropic)
        assert llm.thinking == {"type": "adaptive"}
        assert llm.output_config == {"effort": "low"}

    def test_openai_protocol_additional_fields_passthrough(self, monkeypatch):
        """additional_fields flow through to ChatOpenAI: known fields land in their
        named slot, unknown provider params route into model_kwargs."""
        from langchain_openai import ChatOpenAI
        from patient_agent_bench.config import create_chat_model

        monkeypatch.setenv("OPENAI_API_KEY", "unit-test-placeholder")
        cfg = ModelConfig(
            model_id="gpt-5.2", max_tokens=4096,
            provider="openai-protocol-api", auth="api_key", reasoning_effort="low",
            additional_fields={"seed": 42, "custom_provider_param": "xyz"},
        )
        llm = create_chat_model(cfg)
        assert isinstance(llm, ChatOpenAI)
        assert llm.reasoning_effort == "low"
        assert llm.seed == 42  # real ChatOpenAI field
        assert llm.model_kwargs.get("custom_provider_param") == "xyz"  # unknown -> model_kwargs


class TestParseModelConfig:
    """Tests for parse_model_config, especially the registry-alias override warning."""

    def test_registry_alias_ignores_explicit_params(self):
        """A registry key always resolves to the spec defaults; explicit
        temperature/max_tokens are ignored (by design — use model_id to customize)."""
        spec = get_model_spec("claude-sonnet-4.6-bedrock")
        cfg = parse_model_config(
            {"model": "claude-sonnet-4.6-bedrock", "temperature": 0.1, "max_tokens": 8192}
        )
        assert cfg.model_id == spec.model_id
        assert cfg.temperature == spec.default_temperature
        assert cfg.max_tokens == spec.default_max_tokens

    def test_registry_alias_with_differing_params_warns(self, caplog):
        """When the explicit params differ from the registry defaults, a warning
        tells the user the values were ignored."""
        with caplog.at_level(logging.WARNING, logger="patient_agent_bench.config"):
            parse_model_config(
                {"model": "claude-sonnet-4.6-bedrock", "temperature": 0.1, "max_tokens": 8192}
            )
        assert any("registry key" in r.message for r in caplog.records)

    def test_registry_roundtrip_does_not_warn(self, caplog):
        """A to_dict() round-trip re-emits the resolved defaults, which must NOT
        trigger the override warning."""
        roundtrip = parse_model_config({"model": "claude-sonnet-4.6-bedrock"}).to_dict()
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="patient_agent_bench.config"):
            parse_model_config(roundtrip)
        assert not any("registry key" in r.message for r in caplog.records)

    def test_custom_spec_honors_explicit_params(self):
        """A full custom spec (model_id) uses the explicit values verbatim."""
        cfg = parse_model_config(
            {"model_id": "some-model", "temperature": 0.1, "max_tokens": 8192}
        )
        assert cfg.model_id == "some-model"
        assert cfg.temperature == 0.1
        assert cfg.max_tokens == 8192


class TestBenchConfig:
    """Tests for BenchConfig dataclass."""

    def test_default_values(self):
        """Test BenchConfig with default values."""
        config = BenchConfig.default()
        assert config.max_turns == 3
        assert isinstance(config.assistant_model, ModelConfig)
        assert isinstance(config.user_model, ModelConfig)
        assert isinstance(config.evaluator_model, ModelConfig)

    def test_from_dict_registry_models(self):
        """Test creating BenchConfig with registry models via AgentSpec format."""
        data = {
            "max_turns": 5,
            "assistant_agent": [{"model": {"model": "claude-opus-4.8-bedrock"}}],
            "user_agent": [{"model": {"model": "claude-haiku-4.5-bedrock"}}],
            "evaluator_model": {"model": "claude-sonnet-5-bedrock"},
            "strip_thinking_content": True,
        }
        config = BenchConfig.from_dict(data)

        assert config.max_turns == 5
        assert "opus" in config.assistant_model.model_id
        assert "haiku" in config.user_model.model_id

    def test_from_dict_custom_models(self):
        """Test creating BenchConfig with custom model specs via AgentSpec format."""
        data = {
            "assistant_agent": [
                {
                    "model": {
                        "model_id": "custom-assistant",
                        "temperature": 0.8,
                        "max_tokens": 2048,
                    },
                }
            ],
            "evaluator_model": {
                "model_id": "custom-evaluator",
                "temperature": 0.0,
                "max_tokens": 4096,
            },
            "strip_thinking_content": False,
        }
        config = BenchConfig.from_dict(data)

        assert config.assistant_model.model_id == "custom-assistant"
        assert config.assistant_model.temperature == 0.8
        assert config.evaluator_model.temperature == 0.0

    def test_from_dict_empty(self):
        """Test creating BenchConfig from an empty dictionary uses all defaults."""
        config = BenchConfig.from_dict({})
        assert config.max_turns == 3  # Default value
        assert config.strip_thinking_content is True  # Default value

    def test_from_dict_minimal(self):
        """Test creating BenchConfig with a single explicit field."""
        config = BenchConfig.from_dict({"strip_thinking_content": True})
        assert config.max_turns == 3  # Default value
        assert config.strip_thinking_content is True

    def test_from_dict_strip_thinking_content_opt_out(self):
        """Test strip_thinking_content can be explicitly disabled."""
        config = BenchConfig.from_dict({"strip_thinking_content": False})
        assert config.strip_thinking_content is False

    def test_from_file(self, temp_config_file):
        """Test loading BenchConfig from file."""
        config = BenchConfig.from_file(str(temp_config_file))

        assert config.max_turns == 5
        assert config.assistant_agents[0].model.model_id == "test-assistant-model"

    def test_sandbox_model_from_dict(self):
        """Test creating BenchConfig with sandbox_model."""
        data = {
            "max_turns": 5,
            "assistant_agent": [{"model": {"model": "claude-sonnet-5-bedrock"}}],
            "sandbox_model": {
                "model_id": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "temperature": 0.5,
                "max_tokens": 8192,
            },
            "strip_thinking_content": True,
        }
        config = BenchConfig.from_dict(data)

        assert config.sandbox_model is not None
        assert config.sandbox_model.model_id == "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
        assert config.sandbox_model.temperature == 0.5
        assert config.sandbox_model.max_tokens == 8192

    def test_sandbox_model_default(self):
        """Test sandbox_model uses default when not specified."""
        data = {
            "max_turns": 5,
            "assistant_agent": [{"model": {"model": "claude-sonnet-5-bedrock"}}],
            "strip_thinking_content": True,
        }
        config = BenchConfig.from_dict(data)

        # Should use default model (claude-sonnet-4.5)
        assert config.sandbox_model is not None
        assert config.sandbox_model.model == DEFAULT_MODEL_NAME
        assert config.sandbox_model.model_id is not None

    def test_sandbox_model_to_dict(self):
        """Test sandbox_model is included in to_dict output."""
        data = {
            "sandbox_model": {
                "model_id": "test-model",
                "temperature": 0.5,
                "max_tokens": 4096,
            },
            "strip_thinking_content": True,
        }
        config = BenchConfig.from_dict(data)
        result = config.to_dict()

        assert "sandbox_model" in result
        assert result["sandbox_model"]["model_id"] == "test-model"

    def test_sandbox_model_always_in_to_dict(self):
        """Test sandbox_model is always in to_dict (uses default)."""
        config = BenchConfig.from_dict({"strip_thinking_content": True})
        result = config.to_dict()

        assert "sandbox_model" in result
        assert result["sandbox_model"]["model_id"] is not None


class TestCredentialManagement:
    """Tests for AWS credential management functions."""

    def test_check_credentials_valid_success(self, mock_boto3_client):
        """Test credential check passes when the STS identity call succeeds."""
        mock_boto3_client["sts"].get_caller_identity.return_value = {
            "Account": "123456789012"
        }

        assert check_credentials_valid() is True

    def test_check_credentials_valid_any_account(self, mock_boto3_client):
        """Any account is accepted: the check confirms usable credentials, not
        which account they belong to."""
        mock_boto3_client["sts"].get_caller_identity.return_value = {
            "Account": "999999999999"
        }

        assert check_credentials_valid() is True

    def test_check_credentials_valid_failure(self, mock_boto3_client):
        """Test credential check when credentials are invalid."""
        mock_boto3_client["sts"].get_caller_identity.side_effect = Exception("Invalid credentials")
        
        result = check_credentials_valid()
        assert result is False

    def test_refresh_credentials_hook_is_noop(self, mock_aws_env):
        """The public credential hook is a no-op that always returns False."""
        result = refresh_credentials_hook()
        assert result is False

    @patch("subprocess.run")
    def test_refresh_credentials_hook_runs_no_subprocess(self, mock_run, mock_aws_env):
        """The public no-op hook must not shell out to any external tool."""
        result = refresh_credentials_hook()
        assert result is False
        mock_run.assert_not_called()

    def test_assume_target_role_no_role_configured(self, monkeypatch):
        """Test assume role when no target role is configured."""
        monkeypatch.delenv("AWS_ARN_ROLE", raising=False)
        
        result = assume_target_role()
        assert result is True  # Should succeed (no-op)

    def test_assume_target_role_success(self, mock_boto3_client, monkeypatch):
        """Test successful role assumption."""
        monkeypatch.setenv("AWS_ARN_ROLE", "arn:aws:iam::123456789012:role/TestRole")
        
        mock_boto3_client["sts"].assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "new-access-key",
                "SecretAccessKey": "new-secret-key",
                "SessionToken": "new-session-token",
                "Expiration": "2024-01-01T00:00:00Z",
            }
        }
        
        result = assume_target_role()
        assert result is True

    def test_assume_target_role_failure(self, mock_boto3_client, monkeypatch):
        """Test role assumption failure."""
        monkeypatch.setenv("AWS_ARN_ROLE", "arn:aws:iam::123456789012:role/TestRole")
        mock_boto3_client["sts"].assume_role.side_effect = Exception("Access denied")
        
        result = assume_target_role()
        assert result is False

    def test_ensure_credentials_already_valid(self, mock_boto3_client, mock_aws_env):
        """Test ensure_credentials when credentials are already valid."""
        mock_boto3_client["sts"].get_caller_identity.return_value = {
            "Account": "123456789012"
        }

        result = ensure_credentials()
        assert result is True


# =============================================================================
# Property-Based Tests for Multi-Model Config Support
# =============================================================================

from hypothesis import given, strategies as st, settings, HealthCheck


# Strategies for generating model configs
@st.composite
def registry_model_config_dict(draw):
    """Generate a registry model config dict."""
    model_names = [
        "claude-opus-4.5-bedrock", "claude-sonnet-4.5-bedrock", "claude-haiku-4.5-bedrock",
        "claude-sonnet-5-bedrock", "claude-haiku-4.5-bedrock", "claude-opus-4.8-bedrock",  # aliases
    ]
    return {"model": draw(st.sampled_from(model_names))}


@st.composite
def custom_model_config_dict(draw):
    """Generate a custom model config dict."""
    # Use from_regex to generate non-empty strings without filtering
    model_id = draw(st.from_regex(r"[a-zA-Z][a-zA-Z0-9._-]{0,49}", fullmatch=True))
    return {
        "model_id": model_id,
        "temperature": draw(st.floats(min_value=0.0, max_value=2.0, allow_nan=False)),
        "max_tokens": draw(st.integers(min_value=1, max_value=100000)),
    }


@st.composite
def model_config_dict(draw):
    """Generate either a registry or custom model config dict."""
    return draw(st.one_of(registry_model_config_dict(), custom_model_config_dict()))


@st.composite
def model_config_list(draw):
    """Generate a list of model config dicts."""
    return draw(st.lists(model_config_dict(), min_size=1, max_size=5))


@st.composite
def model_config_input(draw):
    """Generate either a single dict or list of dicts for model config."""
    return draw(st.one_of(model_config_dict(), model_config_list()))


# =============================================================================
# AgentSpec Hypothesis Strategies
# =============================================================================


@st.composite
def agent_spec_strategy(draw):
    """Generate an arbitrary valid AgentSpec instance."""
    model_cfg = ModelConfig(**draw(model_config_dict()))
    prompt = draw(st.from_regex(r"[a-z][a-z0-9_]{0,29}", fullmatch=True))
    agent_class = draw(st.from_regex(r"[a-z][a-z0-9_]{0,19}", fullmatch=True))
    system_prompt = draw(st.one_of(st.none(), st.text(min_size=1, max_size=200)))
    label = draw(st.one_of(st.none(), st.text(min_size=1, max_size=50)))
    return AgentSpec(
        model=model_cfg,
        prompt=prompt,
        agent_class=agent_class,
        system_prompt=system_prompt,
        label=label,
    )


class TestAgentSpecRoundTrip:
    """Property-based tests for AgentSpec round-trip serialization.

    **Feature: agent-spec**
    **Property 1: AgentSpec round-trip serialization**
    **Validates: Requirements 1.2, 1.5**
    """

    @given(spec=agent_spec_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_agent_spec_round_trip(self, spec):
        """
        Property 1: AgentSpec round-trip serialization

        *For any* valid AgentSpec instance, serializing via to_dict() and
        deserializing via from_dict() SHALL produce an equivalent object
        with identical field values.

        **Validates: Requirements 1.2, 1.5**
        """
        serialized = spec.to_dict()
        restored = AgentSpec.from_dict(serialized)

        assert restored.model.model == spec.model.model
        assert restored.model.model_id == spec.model.model_id
        assert restored.model.temperature == spec.model.temperature
        assert restored.model.max_tokens == spec.model.max_tokens
        assert restored.model.provider == spec.model.provider
        assert restored.prompt == spec.prompt
        assert restored.agent_class == spec.agent_class
        assert restored.system_prompt == spec.system_prompt
        assert restored.label == spec.label


@st.composite
def bench_config_strategy(draw):
    """Generate a valid BenchConfig instance for property testing."""
    assistant_agents = draw(st.lists(agent_spec_strategy(), min_size=1, max_size=3))
    user_agents = draw(st.lists(agent_spec_strategy(), min_size=1, max_size=3))
    evaluator_models = [ModelConfig(**draw(model_config_dict())) for _ in range(draw(st.integers(min_value=1, max_value=3)))]
    sandbox_model = ModelConfig(**draw(model_config_dict()))
    seed_generator_model = ModelConfig(**draw(model_config_dict()))
    analyzer_model = ModelConfig(**draw(model_config_dict()))
    max_turns = draw(st.integers(min_value=1, max_value=50))
    strip_thinking_content = draw(st.booleans())
    aggregation_method = draw(st.sampled_from(["average", "majority_vote"]))
    return BenchConfig(
        assistant_agents=assistant_agents,
        user_agents=user_agents,
        evaluator_models=evaluator_models,
        sandbox_model=sandbox_model,
        seed_generator_model=seed_generator_model,
        analyzer_model=analyzer_model,
        max_turns=max_turns,
        strip_thinking_content=strip_thinking_content,
        aggregation_method=aggregation_method,
    )


class TestBenchConfigRoundTrip:
    """Property-based tests for BenchConfig round-trip serialization.

    **Feature: agent-spec**
    **Property 4: BenchConfig round-trip serialization**
    **Validates: Requirements 7.6, 13.3**
    """

    @given(config=bench_config_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_4_bench_config_round_trip(self, config):
        """
        Property 4: BenchConfig round-trip serialization

        *For any* valid BenchConfig instance, serializing via to_dict() and
        deserializing via from_dict() SHALL preserve assistant_agents,
        user_agents, and evaluator_models.

        **Validates: Requirements 7.6, 13.3**
        """
        serialized = config.to_dict()
        restored = BenchConfig.from_dict(serialized)

        # Verify assistant_agents preserved
        assert len(restored.assistant_agents) == len(config.assistant_agents)
        for orig, rest in zip(config.assistant_agents, restored.assistant_agents):
            assert rest.model.model == orig.model.model
            assert rest.model.model_id == orig.model.model_id
            assert rest.model.temperature == orig.model.temperature
            assert rest.model.max_tokens == orig.model.max_tokens
            assert rest.model.provider == orig.model.provider
            assert rest.prompt == orig.prompt
            assert rest.agent_class == orig.agent_class
            assert rest.system_prompt == orig.system_prompt
            assert rest.label == orig.label

        # Verify user_agents preserved
        assert len(restored.user_agents) == len(config.user_agents)
        for orig, rest in zip(config.user_agents, restored.user_agents):
            assert rest.model.model == orig.model.model
            assert rest.model.model_id == orig.model.model_id
            assert rest.model.temperature == orig.model.temperature
            assert rest.model.max_tokens == orig.model.max_tokens
            assert rest.model.provider == orig.model.provider
            assert rest.prompt == orig.prompt
            assert rest.agent_class == orig.agent_class
            assert rest.system_prompt == orig.system_prompt
            assert rest.label == orig.label

        # Verify evaluator_models preserved (plain ModelConfig, not AgentSpec)
        assert len(restored.evaluator_models) == len(config.evaluator_models)
        for orig, rest in zip(config.evaluator_models, restored.evaluator_models):
            assert rest.model == orig.model
            assert rest.model_id == orig.model_id
            assert rest.temperature == orig.temperature
            assert rest.max_tokens == orig.max_tokens
            assert rest.provider == orig.provider


class TestConfigPropertyBased:
    """Property-based tests for config parsing.

    **Feature: multi-experiment-runner**
    **Property 1: Config Parsing Flexibility**
    **Property 2: Single Dict Normalization**
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
    """

    @given(model_input=model_config_input())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_config_parsing_flexibility(self, model_input):
        """
        Property 1: Config Parsing Flexibility

        *For any* model configuration input (single dict or list of dicts) for any role
        (assistant, user, evaluator), parsing SHALL succeed and the internal representation
        SHALL be a non-empty list of AgentSpec objects (for assistant/user) or ModelConfig objects (for evaluator).

        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        """
        # Wrap model_input in AgentSpec format for assistant_agent key
        if isinstance(model_input, list):
            agent_input = [{"model": m} for m in model_input]
        else:
            agent_input = [{"model": model_input}]

        data = {"assistant_agent": agent_input, "strip_thinking_content": True}
        config = BenchConfig.from_dict(data)

        # Internal representation must be a non-empty list of AgentSpec
        assert isinstance(config.assistant_agents, list)
        assert len(config.assistant_agents) > 0

        # All elements must be AgentSpec instances with valid ModelConfig
        for agent_spec in config.assistant_agents:
            assert isinstance(agent_spec, AgentSpec)
            assert isinstance(agent_spec.model, ModelConfig)
            assert agent_spec.model.model_id is not None
            # temperature may legitimately be None: some models (e.g. Sonnet 5,
            # Opus 4.8) reject the `temperature` param, so their specs
            # resolve default_temperature=None and it is omitted from the request.
            assert agent_spec.model.max_tokens is not None

    @given(model_input=model_config_input())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_user_model_parsing(self, model_input):
        """
        Property 1 (continued): Test user_agent parsing flexibility.

        **Validates: Requirements 7.2**
        """
        # Wrap model_input in AgentSpec format for user_agent key
        if isinstance(model_input, list):
            agent_input = [{"model": m} for m in model_input]
        else:
            agent_input = [{"model": model_input}]

        data = {"user_agent": agent_input, "strip_thinking_content": True}
        config = BenchConfig.from_dict(data)

        assert isinstance(config.user_agents, list)
        assert len(config.user_agents) > 0
        for agent_spec in config.user_agents:
            assert isinstance(agent_spec, AgentSpec)
            assert agent_spec.model.model_id is not None

    @given(model_input=model_config_input())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_evaluator_model_parsing(self, model_input):
        """
        Property 1 (continued): Test evaluator_model parsing flexibility.

        **Validates: Requirements 2.3**
        """
        data = {"evaluator_model": model_input, "strip_thinking_content": True}
        config = BenchConfig.from_dict(data)

        assert isinstance(config.evaluator_models, list)
        assert len(config.evaluator_models) > 0
        for mc in config.evaluator_models:
            assert isinstance(mc, ModelConfig)
            assert mc.model_id is not None

    @given(single_dict=model_config_dict())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_2_single_dict_normalization(self, single_dict):
        """
        Property 2: Single Dict Normalization

        *For any* single dict model configuration wrapped in AgentSpec format,
        the parsed result SHALL be a list containing exactly one AgentSpec
        with equivalent model parameters.

        **Validates: Requirements 7.4**
        """
        data = {"assistant_agent": {"model": single_dict}, "strip_thinking_content": True}
        config = BenchConfig.from_dict(data)

        # Must be normalized to a list of exactly one element
        assert isinstance(config.assistant_agents, list)
        assert len(config.assistant_agents) == 1

        # The single AgentSpec must have a valid ModelConfig
        agent_spec = config.assistant_agents[0]
        assert isinstance(agent_spec, AgentSpec)
        mc = agent_spec.model
        assert isinstance(mc, ModelConfig)

        # If it was a registry model, check model name was preserved
        if "model" in single_dict:
            assert mc.model == single_dict["model"] or mc.model is not None

        # If it was a custom model, check values match
        if "model_id" in single_dict:
            assert mc.model_id == single_dict["model_id"]
            assert mc.temperature == single_dict["temperature"]
            assert mc.max_tokens == single_dict["max_tokens"]

    @given(
        assistant_input=model_config_input(),
        user_input=model_config_input(),
        evaluator_input=model_config_input(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_all_roles_combined(self, assistant_input, user_input, evaluator_input):
        """
        Property 1 (combined): All three roles can be parsed together.

        **Validates: Requirements 7.1, 7.2, 7.5**
        """
        # Wrap assistant and user inputs in AgentSpec format
        if isinstance(assistant_input, list):
            assistant_agent_input = [{"model": m} for m in assistant_input]
        else:
            assistant_agent_input = [{"model": assistant_input}]

        if isinstance(user_input, list):
            user_agent_input = [{"model": m} for m in user_input]
        else:
            user_agent_input = [{"model": user_input}]

        data = {
            "assistant_agent": assistant_agent_input,
            "user_agent": user_agent_input,
            "evaluator_model": evaluator_input,
            "strip_thinking_content": True,
        }
        config = BenchConfig.from_dict(data)

        # All three must be non-empty lists
        assert len(config.assistant_agents) > 0
        assert len(config.user_agents) > 0
        assert len(config.evaluator_models) > 0

        # Count should match input
        expected_assistant = len(assistant_input) if isinstance(assistant_input, list) else 1
        expected_user = len(user_input) if isinstance(user_input, list) else 1
        expected_evaluator = len(evaluator_input) if isinstance(evaluator_input, list) else 1

        assert len(config.assistant_agents) == expected_assistant
        assert len(config.user_agents) == expected_user
        assert len(config.evaluator_models) == expected_evaluator

    def test_backward_compatibility_properties(self):
        """
        Test that backward compatibility properties work correctly.

        The assistant_model, user_model, evaluator_model properties should
        return the first element of their respective lists/agents.
        """
        data = {
            "assistant_agent": [
                {"model": {"model": "claude-sonnet-4.5-bedrock"}},
                {"model": {"model": "claude-haiku-4.5-bedrock"}},
            ],
            "user_agent": [{"model": {"model": "claude-haiku-4.5-bedrock"}}],
            "evaluator_model": {"model": "claude-sonnet-5-bedrock"},
            "strip_thinking_content": True,
        }
        config = BenchConfig.from_dict(data)

        # Properties should return first element's model
        assert config.assistant_model is config.assistant_agents[0].model
        assert config.user_model is config.user_agents[0].model
        assert config.evaluator_model is config.evaluator_models[0]

        # First assistant should be sonnet
        assert "sonnet" in config.assistant_model.model_id

    def test_to_dict_round_trip(self):
        """
        Test that to_dict produces valid output that can be re-parsed.
        """
        original_data = {
            "assistant_agent": [
                {"model": {"model": "claude-sonnet-4.5-bedrock"}},
                {"model": {"model": "claude-haiku-4.5-bedrock"}},
            ],
            "user_agent": [{"model": {"model": "claude-haiku-4.5-bedrock"}}],
            "evaluator_model": [{"model": "claude-sonnet-5-bedrock"}],
            "max_turns": 5,
            "strip_thinking_content": True,
        }
        config1 = BenchConfig.from_dict(original_data)
        serialized = config1.to_dict()
        config2 = BenchConfig.from_dict(serialized)

        # Should have same structure
        assert len(config2.assistant_agents) == len(config1.assistant_agents)
        assert len(config2.user_agents) == len(config1.user_agents)
        assert len(config2.evaluator_models) == len(config1.evaluator_models)
        assert config2.max_turns == config1.max_turns

        # Model IDs should match
        for i, agent_spec in enumerate(config1.assistant_agents):
            assert config2.assistant_agents[i].model.model_id == agent_spec.model.model_id

    def test_model_config_to_dict(self):
        """Test ModelConfig.to_dict() serialization."""
        # Registry model
        mc1 = ModelConfig(model="claude-sonnet-4.5-bedrock")
        d1 = mc1.to_dict()
        assert d1["model"] == "claude-sonnet-4.5-bedrock"
        assert d1["model_id"] is not None
        assert d1["temperature"] is not None
        assert d1["max_tokens"] is not None

        # Custom model - model is set to "custom" for clarity
        mc2 = ModelConfig(model_id="custom-id", temperature=0.5, max_tokens=2048)
        d2 = mc2.to_dict()
        assert d2["model"] == "custom"
        assert d2["model_id"] == "custom-id"
        assert d2["temperature"] == 0.5
        assert d2["max_tokens"] == 2048


# =============================================================================
# RolePoolManager Tests
# =============================================================================


class TestRolePoolManager:
    """Tests for RolePoolManager class."""

    def test_empty_pool(self):
        """Test RolePoolManager with empty roles list."""
        pool = RolePoolManager(roles=[])
        assert len(pool) == 0
        assert pool.get_role() is None
        assert pool.get_random_role() is None

    def test_single_role(self):
        """Test RolePoolManager with single role."""
        pool = RolePoolManager(roles=["arn:aws:iam::123:role/test"])
        assert len(pool) == 1
        assert pool.get_role() == "arn:aws:iam::123:role/test"
        assert pool.get_role() == "arn:aws:iam::123:role/test"

    def test_multiple_roles_round_robin(self):
        """Test round-robin distribution with multiple roles."""
        pool = RolePoolManager(roles=["role1", "role2", "role3"])
        assert len(pool) == 3

        # Should cycle through roles in order
        assert pool.get_role() == "role1"
        assert pool.get_role() == "role2"
        assert pool.get_role() == "role3"
        assert pool.get_role() == "role1"  # Wraps around

    def test_from_env_empty(self, monkeypatch):
        """Test from_env with no AWS_ARN_ROLE set."""
        monkeypatch.delenv("AWS_ARN_ROLE", raising=False)
        pool = RolePoolManager.from_env()
        assert len(pool) == 0
        assert pool.roles == []

    def test_from_env_single_role(self, monkeypatch):
        """Test from_env with single role."""
        monkeypatch.setenv("AWS_ARN_ROLE", "arn:aws:iam::123:role/single")
        pool = RolePoolManager.from_env()
        assert len(pool) == 1
        assert pool.roles == ["arn:aws:iam::123:role/single"]

    def test_from_env_multiple_roles(self, monkeypatch):
        """Test from_env with comma-separated roles."""
        monkeypatch.setenv(
            "AWS_ARN_ROLE",
            "arn:aws:iam::123:role/role1,arn:aws:iam::456:role/role2"
        )
        pool = RolePoolManager.from_env()
        assert len(pool) == 2
        assert pool.roles == [
            "arn:aws:iam::123:role/role1",
            "arn:aws:iam::456:role/role2"
        ]

    def test_from_env_strips_whitespace(self, monkeypatch):
        """Test from_env strips whitespace from roles."""
        monkeypatch.setenv("AWS_ARN_ROLE", "  role1  ,  role2  ,  role3  ")
        pool = RolePoolManager.from_env()
        assert pool.roles == ["role1", "role2", "role3"]

    def test_from_env_filters_empty_strings(self, monkeypatch):
        """Test from_env filters out empty strings."""
        monkeypatch.setenv("AWS_ARN_ROLE", "role1,,role2,  ,role3")
        pool = RolePoolManager.from_env()
        assert pool.roles == ["role1", "role2", "role3"]

    def test_get_random_role(self):
        """Test get_random_role returns a role from the pool."""
        roles = ["role1", "role2", "role3"]
        pool = RolePoolManager(roles=roles)

        # Call multiple times and verify all results are from the pool
        for _ in range(10):
            role = pool.get_random_role()
            assert role in roles


# =============================================================================
# Property-Based Tests for RolePoolManager
# =============================================================================


# Strategy for generating unique role ARNs
@st.composite
def unique_role_list(draw):
    """Generate a list of unique role ARNs."""
    num_roles = draw(st.integers(min_value=1, max_value=10))
    roles = []
    for i in range(num_roles):
        account_id = 100000000000 + i
        role_name = f"role{i}"
        roles.append(f"arn:aws:iam::{account_id}:role/{role_name}")
    return roles


@st.composite
def arn_string_unique(draw, index=0):
    """Generate a valid ARN-like string with unique index."""
    account_id = draw(st.integers(min_value=100000000000, max_value=999999999999))
    role_name = draw(st.from_regex(r"[a-zA-Z][a-zA-Z0-9_-]{0,19}", fullmatch=True))
    return f"arn:aws:iam::{account_id}:role/{role_name}"


class TestRolePoolManagerPropertyBased:
    """Property-based tests for RolePoolManager.

    **Feature: parallel-execution**
    **Property 2: Role Parsing and Normalization**
    **Property 4: Role Distribution Balance**
    **Validates: Requirements 2.1, 2.2, 2.4, 3.2**
    """

    @given(roles=unique_role_list())
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture]
    )
    def test_property_2_role_parsing_and_normalization(self, roles):
        """
        Property 2: Role Parsing and Normalization

        *For any* AWS_ARN_ROLE environment variable value (single ARN or comma-separated
        list), parsing SHALL produce a non-empty list of valid ARN strings, and a single
        ARN SHALL result in a list of exactly one element.

        **Validates: Requirements 2.1, 2.2, 2.4**
        """
        # Create env var value from roles
        env_value = ",".join(roles)
        original_value = os.environ.get("AWS_ARN_ROLE")

        try:
            os.environ["AWS_ARN_ROLE"] = env_value
            pool = RolePoolManager.from_env()

            # Must produce a non-empty list
            assert isinstance(pool.roles, list)
            assert len(pool.roles) > 0

            # Number of roles should match input
            assert len(pool.roles) == len(roles)

            # All roles should be strings
            for role in pool.roles:
                assert isinstance(role, str)
                assert len(role) > 0

            # Roles should match input (order preserved)
            assert pool.roles == roles
        finally:
            # Restore original value
            if original_value is None:
                os.environ.pop("AWS_ARN_ROLE", None)
            else:
                os.environ["AWS_ARN_ROLE"] = original_value

    @given(roles=unique_role_list())
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture]
    )
    def test_property_2_whitespace_handling(self, roles):
        """
        Property 2 (continued): Whitespace handling

        *For any* env var value with optional whitespace around roles,
        parsing SHALL strip whitespace and produce clean role strings.

        **Validates: Requirements 2.1, 2.4**
        """
        # Add whitespace around roles
        roles_with_whitespace = [f"  {r}  " for r in roles]
        env_value = ",".join(roles_with_whitespace)
        original_value = os.environ.get("AWS_ARN_ROLE")

        try:
            os.environ["AWS_ARN_ROLE"] = env_value
            pool = RolePoolManager.from_env()

            # All roles should be stripped of whitespace
            for role in pool.roles:
                assert role == role.strip()
                assert not role.startswith(" ")
                assert not role.endswith(" ")

            # Should match original roles (without whitespace)
            assert pool.roles == roles
        finally:
            if original_value is None:
                os.environ.pop("AWS_ARN_ROLE", None)
            else:
                os.environ["AWS_ARN_ROLE"] = original_value

    @given(account_id=st.integers(min_value=100000000000, max_value=999999999999))
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture]
    )
    def test_property_2_single_role_normalization(self, account_id):
        """
        Property 2 (continued): Single role normalization

        *For any* single ARN, parsing SHALL result in a list of exactly one element.

        **Validates: Requirements 2.4**
        """
        single_role = f"arn:aws:iam::{account_id}:role/test-role"
        original_value = os.environ.get("AWS_ARN_ROLE")

        try:
            os.environ["AWS_ARN_ROLE"] = single_role
            pool = RolePoolManager.from_env()

            # Single role should produce list of exactly one element
            assert len(pool.roles) == 1
            assert pool.roles[0] == single_role
        finally:
            if original_value is None:
                os.environ.pop("AWS_ARN_ROLE", None)
            else:
                os.environ["AWS_ARN_ROLE"] = original_value

    @given(roles=unique_role_list(), num_calls=st.integers(min_value=1, max_value=100))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_4_role_distribution_balance(self, roles, num_calls):
        """
        Property 4: Role Distribution Balance

        *For any* role pool with R roles and N tasks where N > R, after all tasks
        complete, each role SHALL have been assigned approximately N/R times
        (within statistical variance for round-robin selection).

        For round-robin, the distribution should be exactly balanced when
        num_calls is a multiple of len(roles).

        **Validates: Requirements 3.2**
        """
        pool = RolePoolManager(roles=roles)
        num_roles = len(roles)

        # Collect all role assignments
        assignments = [pool.get_role() for _ in range(num_calls)]

        # Count assignments per role
        from collections import Counter
        counts = Counter(assignments)

        # For round-robin, each role should be assigned floor(N/R) or ceil(N/R) times
        expected_min = num_calls // num_roles
        expected_max = expected_min + 1

        for role in roles:
            count = counts.get(role, 0)
            assert expected_min <= count <= expected_max, (
                f"Role {role} assigned {count} times, expected between "
                f"{expected_min} and {expected_max}"
            )

        # Total assignments should equal num_calls
        assert sum(counts.values()) == num_calls

    @given(roles=unique_role_list())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_4_exact_balance_on_full_cycles(self, roles):
        """
        Property 4 (continued): Exact balance on full cycles

        *For any* role pool, after exactly N * len(roles) calls, each role
        SHALL have been assigned exactly N times.

        **Validates: Requirements 3.2**
        """
        pool = RolePoolManager(roles=roles)
        num_roles = len(roles)
        num_cycles = 5  # Fixed number of full cycles

        # Make exactly num_cycles * num_roles calls
        num_calls = num_cycles * num_roles
        assignments = [pool.get_role() for _ in range(num_calls)]

        # Count assignments per role
        from collections import Counter
        counts = Counter(assignments)

        # Each role should be assigned exactly num_cycles times
        for role in roles:
            assert counts[role] == num_cycles, (
                f"Role {role} assigned {counts[role]} times, expected {num_cycles}"
            )


# =============================================================================
# Tests for Prompt Loading Utilities
# =============================================================================


class TestLoadPrompt:
    """Tests for load_prompt() function.

    **Feature: configurable-prompts**
    **Validates: Requirements 2.3, 2.4, 2.5**
    """

    def test_load_user_agent_default_prompt(self):
        """Test loading the default user agent prompt."""
        prompt = load_prompt("user_agent", "default_prompt")
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # Verify it contains expected placeholders
        assert "{scenario}" in prompt
        assert "{user_profile}" in prompt

    def test_load_assistant_agent_default_prompt(self):
        """Test loading the default assistant agent prompt."""
        prompt = load_prompt("assistant_agent", "default_prompt")
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # Verify it contains expected placeholder
        assert "{user_profile}" in prompt

    def test_load_prompt_invalid_agent_type(self):
        """Test error for invalid agent type."""
        with pytest.raises(ValueError, match="Invalid agent_type"):
            load_prompt("invalid_agent", "default_prompt")

    def test_load_prompt_missing_file(self):
        """Test error for missing prompt file."""
        with pytest.raises(FileNotFoundError, match="Prompt file not found"):
            load_prompt("user_agent", "nonexistent_prompt")

    def test_load_prompt_file_path_in_error(self):
        """Test that error message contains expected file path."""
        try:
            load_prompt("assistant_agent", "missing_prompt")
            pytest.fail("Expected FileNotFoundError")
        except FileNotFoundError as e:
            assert "assistant_agent/missing_prompt.py" in str(e)

    def test_load_prompt_absolute_path(self, tmp_path):
        """Test loading prompt from an absolute file path."""
        # Create a temporary prompt file
        prompt_content = "This is a custom prompt from {name}."
        prompt_file = tmp_path / "custom_prompt.txt"
        prompt_file.write_text(prompt_content)

        # Load using absolute path - agent_type is ignored for absolute paths
        result = load_prompt("user_agent", str(prompt_file))
        assert result == prompt_content

    def test_load_prompt_absolute_path_missing_file(self, tmp_path):
        """Test error when absolute path file doesn't exist."""
        missing_file = tmp_path / "nonexistent.txt"
        with pytest.raises(FileNotFoundError, match="Prompt file not found"):
            load_prompt("user_agent", str(missing_file))

    def test_load_prompt_absolute_path_ignores_agent_type(self, tmp_path):
        """Test that agent_type is ignored when using absolute path."""
        prompt_content = "Custom assistant prompt."
        prompt_file = tmp_path / "assistant_custom.txt"
        prompt_file.write_text(prompt_content)

        # Even with "user_agent", it should load the file directly
        result = load_prompt("user_agent", str(prompt_file))
        assert result == prompt_content


class TestFormatPromptSafe:
    """Tests for format_prompt_safe() function.

    **Feature: configurable-prompts**
    **Validates: Requirements 4.1, 4.2, 4.3**
    """

    def test_format_with_all_placeholders(self):
        """Test formatting when all placeholders are present."""
        template = "Hello {name}, your age is {age}."
        result = format_prompt_safe(template, name="Alice", age=30)
        assert result == "Hello Alice, your age is 30."

    def test_format_with_missing_placeholder_in_template(self):
        """Test formatting when template doesn't have all expected placeholders."""
        template = "Hello {name}."
        # extra_field is not in template - should warn but not error
        result = format_prompt_safe(template, name="Bob", extra_field="ignored")
        assert result == "Hello Bob."

    def test_format_with_no_placeholders(self):
        """Test formatting a template with no placeholders."""
        template = "This is a static prompt."
        result = format_prompt_safe(template, name="Alice", age=30)
        assert result == "This is a static prompt."

    def test_format_preserves_unmatched_braces(self):
        """Test that unmatched braces in template are preserved."""
        template = "Hello {name}, use {{literal_braces}}."
        result = format_prompt_safe(template, name="Charlie")
        # The {{literal_braces}} should remain as-is since it's not a placeholder
        assert "Hello Charlie" in result

    def test_format_with_empty_kwargs(self):
        """Test formatting with no kwargs provided."""
        template = "Hello {name}."
        result = format_prompt_safe(template)
        # Placeholder should remain since no value provided
        assert result == "Hello {name}."

    def test_format_with_complex_template(self):
        """Test formatting with a complex multi-line template."""
        template = """<role>
You are a {role_type} assistant.
</role>

<context>
User: {user_name}
Profile: {profile}
</context>
"""
        result = format_prompt_safe(
            template,
            role_type="health",
            user_name="John",
            profile="Patient with diabetes"
        )
        assert "health assistant" in result
        assert "John" in result
        assert "Patient with diabetes" in result

    def test_format_logs_warning_for_missing_placeholder(self, caplog):
        """Test that warning is logged for placeholders not in template."""
        import logging
        caplog.set_level(logging.WARNING)

        template = "Hello {name}."
        format_prompt_safe(template, name="Alice", missing_key="value")

        # Check that warning was logged
        assert any("missing_key" in record.message for record in caplog.records)

    def test_format_with_numeric_values(self):
        """Test formatting with numeric values."""
        template = "Count: {count}, Price: {price}"
        result = format_prompt_safe(template, count=42, price=19.99)
        assert result == "Count: 42, Price: 19.99"

    def test_format_with_none_value(self):
        """Test formatting with None value."""
        template = "Value: {value}"
        result = format_prompt_safe(template, value=None)
        assert result == "Value: None"


# =============================================================================
# Property-Based Tests for Configurable Prompts
# =============================================================================


# Strategy for generating valid prompt names
@st.composite
def valid_prompt_name(draw):
    """Generate a valid prompt file name (without .py extension)."""
    # Prompt names should be valid Python identifiers
    return draw(st.from_regex(r"[a-z][a-z0-9_]{0,29}", fullmatch=True))


class TestConfigPromptFieldsPropertyBased:
    """Property-based tests for prompt configuration fields via AgentSpec.

    **Feature: agent-spec**
    **Property 2: Config Prompt Field Parsing via AgentSpec**
    **Property 9: Backward Compatibility**
    **Validates: Requirements 1.1, 1.2, 7.1, 7.2**
    """

    @given(
        user_prompt=valid_prompt_name(),
        assistant_prompt=valid_prompt_name(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_2_config_prompt_field_parsing(self, user_prompt, assistant_prompt):
        """
        Property 2: Config Prompt Field Parsing via AgentSpec

        *For any* valid benchmark configuration JSON containing prompt fields inside
        AgentSpec objects, the BenchConfig SHALL correctly parse and store these values.

        **Validates: Requirements 1.1, 7.1, 7.2**
        """
        data = {
            "assistant_agent": [{"model": {"model": "claude-sonnet-5-bedrock"}, "prompt": assistant_prompt}],
            "user_agent": [{"model": {"model": "claude-haiku-4.5-bedrock"}, "prompt": user_prompt}],
            "strip_thinking_content": True,
        }
        config = BenchConfig.from_dict(data)

        # Verify prompts are correctly parsed inside AgentSpec
        assert isinstance(config.user_agents[0].prompt, str)
        assert isinstance(config.assistant_agents[0].prompt, str)

        assert config.user_agents[0].prompt == user_prompt
        assert config.assistant_agents[0].prompt == assistant_prompt

    @given(user_prompt=valid_prompt_name())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_2_user_prompt_only(self, user_prompt):
        """
        Property 2 (continued): Test parsing with only user_agent prompt specified.

        **Validates: Requirements 7.2**
        """
        data = {
            "user_agent": [{"model": {"model": "claude-haiku-4.5-bedrock"}, "prompt": user_prompt}],
            "strip_thinking_content": True,
        }
        config = BenchConfig.from_dict(data)

        # user_agent prompt should be parsed
        assert config.user_agents[0].prompt == user_prompt
        # assistant_agent should use default prompt
        assert config.assistant_agents[0].prompt == "default_prompt"

    @given(assistant_prompt=valid_prompt_name())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_2_assistant_prompt_only(self, assistant_prompt):
        """
        Property 2 (continued): Test parsing with only assistant_agent prompt specified.

        **Validates: Requirements 7.1**
        """
        data = {
            "assistant_agent": [{"model": {"model": "claude-sonnet-5-bedrock"}, "prompt": assistant_prompt}],
            "strip_thinking_content": True,
        }
        config = BenchConfig.from_dict(data)

        # assistant_agent prompt should be parsed
        assert config.assistant_agents[0].prompt == assistant_prompt
        # user_agent should use default prompt
        assert config.user_agents[0].prompt == "default_prompt"

    @given(
        user_prompt=valid_prompt_name(),
        assistant_prompt=valid_prompt_name(),
        model_input=model_config_input(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_2_prompt_fields_with_model_config(
        self, user_prompt, assistant_prompt, model_input
    ):
        """
        Property 2 (continued): Test prompt fields work alongside model config in AgentSpec.

        **Validates: Requirements 1.1, 7.1, 7.2**
        """
        # Use the first model_input dict for the assistant agent
        if isinstance(model_input, list):
            model_dict = model_input[0]
        else:
            model_dict = model_input

        data = {
            "assistant_agent": [{"model": model_dict, "prompt": assistant_prompt}],
            "user_agent": [{"model": {"model": "claude-haiku-4.5-bedrock"}, "prompt": user_prompt}],
            "strip_thinking_content": True,
        }
        config = BenchConfig.from_dict(data)

        # Prompt fields should be correctly parsed
        assert config.user_agents[0].prompt == user_prompt
        assert config.assistant_agents[0].prompt == assistant_prompt

        # Model config should also be correctly parsed
        assert len(config.assistant_agents) > 0

    @given(
        user_prompt=valid_prompt_name(),
        assistant_prompt=valid_prompt_name(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_2_to_dict_includes_prompt_fields(self, user_prompt, assistant_prompt):
        """
        Property 2 (continued): Test that to_dict() includes prompt fields inside AgentSpec.

        **Validates: Requirements 1.2, 7.6**
        """
        data = {
            "assistant_agent": [{"model": {"model": "claude-sonnet-5-bedrock"}, "prompt": assistant_prompt}],
            "user_agent": [{"model": {"model": "claude-haiku-4.5-bedrock"}, "prompt": user_prompt}],
            "strip_thinking_content": True,
        }
        config = BenchConfig.from_dict(data)
        serialized = config.to_dict()

        # Serialized dict should include prompt fields inside agent specs
        assert "assistant_agent" in serialized
        assert "user_agent" in serialized
        assert serialized["assistant_agent"][0]["prompt"] == assistant_prompt
        assert serialized["user_agent"][0]["prompt"] == user_prompt

    @given(model_input=model_config_input())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_9_backward_compatibility_no_prompt_fields(self, model_input):
        """
        Property 9: Backward Compatibility

        *For any* benchmark configuration that omits prompt fields in AgentSpec,
        the system SHALL use the default prompts ("default_prompt").

        **Validates: Requirements 1.5, 7.3**
        """
        # Wrap model_input in AgentSpec format without prompt
        if isinstance(model_input, list):
            agent_input = [{"model": m} for m in model_input]
        else:
            agent_input = [{"model": model_input}]

        data = {
            "assistant_agent": agent_input,
            "max_turns": 5,
            "strip_thinking_content": True,
        }
        config = BenchConfig.from_dict(data)

        # Should use default prompt values inside AgentSpec
        for agent_spec in config.assistant_agents:
            assert agent_spec.prompt == "default_prompt"
        for agent_spec in config.user_agents:
            assert agent_spec.prompt == "default_prompt"

        # Model config should still work
        assert len(config.assistant_agents) > 0
        assert config.max_turns == 5

    def test_property_9_backward_compatibility_empty_dict(self):
        """
        Property 9 (continued): Test backward compatibility with empty config.

        **Validates: Requirements 7.3**
        """
        # An empty config is valid: every field falls back to its default.
        config = BenchConfig.from_dict({})
        assert config.strip_thinking_content is True
        assert config.max_turns == 3

    def test_property_9_backward_compatibility_default_factory(self):
        """
        Property 9 (continued): Test backward compatibility with default() factory.

        **Validates: Requirements 7.3**
        """
        config = BenchConfig.default()

        # Should use all defaults — prompts live inside AgentSpec
        assert config.assistant_agents[0].prompt == "default_prompt"
        assert config.user_agents[0].prompt == "default_prompt"

    @given(
        user_prompt=valid_prompt_name(),
        assistant_prompt=valid_prompt_name(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_9_round_trip_preserves_prompt_fields(self, user_prompt, assistant_prompt):
        """
        Property 9 (continued): Test that round-trip serialization preserves prompt fields in AgentSpec.

        **Validates: Requirements 1.2, 7.6**
        """
        original_data = {
            "assistant_agent": [{"model": {"model": "claude-sonnet-5-bedrock"}, "prompt": assistant_prompt}],
            "user_agent": [{"model": {"model": "claude-haiku-4.5-bedrock"}, "prompt": user_prompt}],
            "max_turns": 5,
            "strip_thinking_content": True,
        }
        config1 = BenchConfig.from_dict(original_data)
        serialized = config1.to_dict()
        config2 = BenchConfig.from_dict(serialized)

        # Prompt fields should be preserved through round-trip
        assert config2.assistant_agents[0].prompt == config1.assistant_agents[0].prompt
        assert config2.user_agents[0].prompt == config1.user_agents[0].prompt
        assert config2.max_turns == config1.max_turns


# =============================================================================
# Property-Based Tests for Agent Prompt Loading
# =============================================================================


class TestAgentPromptLoadingPropertyBased:
    """Property-based tests for agent prompt loading.

    **Feature: configurable-prompts**
    **Property 1: Default Prompt Loading**
    **Property 6: Flexible Placeholder Handling**
    **Validates: Requirements 1.4, 4.1, 4.2**
    """

    @pytest.fixture(autouse=True)
    def _mock_boto3(self, mock_boto3_client):
        """Auto-use mock_boto3_client so agents can create bedrock clients."""
        pass

    @given(
        scenario=st.text(min_size=1, max_size=100),
        user_profile=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_property_1_user_agent_default_prompt_loading(
        self, scenario, user_profile
    ):
        """
        Property 1: Default Prompt Loading

        *For any* user agent initialized without a custom prompt specification,
        the agent SHALL use the prompt loaded from `default_prompt.py` in its directory.

        **Validates: Requirements 1.4**
        """
        from patient_agent_bench.user_agent.default_agent import DefaultUserAgent as UserAgent
        from patient_agent_bench.user_agent.default_prompt import SYSTEM_PROMPT as DEFAULT_PROMPT

        # Create agent without prompt_name or system_prompt
        agent = UserAgent(
            scenario=scenario,
            user_profile=user_profile,
            model_config=ModelConfig(),
            current_datetime="Tuesday, February 10, 2026 at 2:35 PM",
            personality="cooperative",
        )

        # Agent should use the default prompt template
        assert agent.system_prompt_template == DEFAULT_PROMPT

        # System prompt should be formatted with the provided values
        assert isinstance(agent.system_prompt, str)
        assert len(agent.system_prompt) > 0

    @given(user_profile=st.text(min_size=1, max_size=100))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_assistant_agent_default_prompt_loading(self, user_profile):
        """
        Property 1 (continued): Default Prompt Loading for AssistantAgent

        *For any* assistant agent initialized without a custom prompt specification,
        the agent SHALL use the prompt loaded from `default_prompt.py` in its directory.

        **Validates: Requirements 1.4**
        """
        from patient_agent_bench.assistant_agent.default_agent import DefaultAssistantAgent as AssistantAgent
        from patient_agent_bench.assistant_agent.default_prompt import (
            SYSTEM_PROMPT as DEFAULT_PROMPT,
        )

        # Create agent without prompt_name or system_prompt
        agent = AssistantAgent(
            model_config=ModelConfig(model_id="test-model", temperature=0.7, max_tokens=1024),
            current_datetime="Tuesday, February 10, 2026 at 2:35 PM",
        )

        # Agent should use the default prompt template
        assert agent.system_prompt_template == DEFAULT_PROMPT

    @given(
        scenario=st.text(min_size=1, max_size=100),
        user_profile=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_user_agent_with_prompt_name(
        self, scenario, user_profile
    ):
        """
        Property 1 (continued): Test loading with prompt_name parameter.

        *For any* user agent initialized with prompt_name="default_prompt",
        the agent SHALL load and use that prompt.

        **Validates: Requirements 1.4**
        """
        from patient_agent_bench.user_agent.default_agent import DefaultUserAgent as UserAgent
        from patient_agent_bench.user_agent.default_prompt import SYSTEM_PROMPT as DEFAULT_PROMPT

        # Create agent with explicit prompt_name
        agent = UserAgent(
            scenario=scenario,
            user_profile=user_profile,
            model_config=ModelConfig(),
            current_datetime="Tuesday, February 10, 2026 at 2:35 PM",
            prompt_name="default_prompt",
            personality="cooperative",
        )

        # Agent should use the loaded prompt
        assert agent.system_prompt_template == DEFAULT_PROMPT

    @given(user_profile=st.text(min_size=1, max_size=100))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_assistant_agent_with_prompt_name(self, user_profile):
        """
        Property 1 (continued): Test loading with prompt_name parameter for assistant.

        **Validates: Requirements 1.4**
        """
        from patient_agent_bench.assistant_agent.default_agent import DefaultAssistantAgent as AssistantAgent
        from patient_agent_bench.assistant_agent.default_prompt import (
            SYSTEM_PROMPT as DEFAULT_PROMPT,
        )

        # Create agent with explicit prompt_name
        agent = AssistantAgent(
            model_config=ModelConfig(model_id="test-model", temperature=0.7, max_tokens=1024),
            current_datetime="Tuesday, February 10, 2026 at 2:35 PM",
            prompt_name="default_prompt",
        )

        # Agent should use the loaded prompt
        assert agent.system_prompt_template == DEFAULT_PROMPT

    @given(
        template=st.text(min_size=1, max_size=200),
        placeholder_values=st.dictionaries(
            keys=st.from_regex(r"[a-z_][a-z0-9_]{0,19}", fullmatch=True),
            values=st.text(min_size=0, max_size=50),
            min_size=0,
            max_size=5,
        ),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_flexible_placeholder_handling(self, template, placeholder_values):
        """
        Property 6: Flexible Placeholder Handling

        *For any* prompt template and set of runtime values, the system SHALL
        substitute all placeholders that exist in the template while ignoring
        placeholders that are not present, without raising an error.

        **Validates: Requirements 4.1, 4.2**
        """
        # format_prompt_safe should never raise an error
        result = format_prompt_safe(template, **placeholder_values)

        # Result should be a string
        assert isinstance(result, str)

        # For any placeholder that exists in the template and has a value,
        # the placeholder should be replaced
        for key, value in placeholder_values.items():
            placeholder = f"{{{key}}}"
            if placeholder in template:
                # The placeholder should be replaced with the value
                assert placeholder not in result or value == ""

    @given(
        scenario=st.text(min_size=1, max_size=100),
        user_profile=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_user_agent_placeholder_substitution(
        self, scenario, user_profile
    ):
        """
        Property 6 (continued): Test placeholder substitution in UserAgent.

        *For any* user agent with scenario and user_profile values,
        the formatted system prompt SHALL contain those values.

        **Validates: Requirements 4.1**
        """
        from patient_agent_bench.user_agent.default_agent import DefaultUserAgent as UserAgent

        agent = UserAgent(
            scenario=scenario,
            user_profile=user_profile,
            model_config=ModelConfig(),
            current_datetime="Tuesday, February 10, 2026 at 2:35 PM",
            personality="cooperative",
        )

        # The formatted prompt should contain the substituted values
        # (if the default template has those placeholders)
        assert isinstance(agent.system_prompt, str)
        assert len(agent.system_prompt) > 0

        # If the values are non-empty and the template has the placeholders,
        # they should appear in the formatted prompt
        if scenario and "{scenario}" in agent.system_prompt_template:
            assert scenario in agent.system_prompt

    @given(user_profile=st.text(min_size=1, max_size=100))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_assistant_agent_placeholder_substitution(self, user_profile):
        """
        Property 6 (continued): Test placeholder substitution in AssistantAgent.

        *For any* assistant agent with user_profile value, when the agent is created,
        the formatted system prompt SHALL contain that value.

        **Validates: Requirements 4.1**
        """
        from unittest.mock import MagicMock, patch
        from patient_agent_bench.assistant_agent.default_agent import DefaultAssistantAgent as AssistantAgent

        agent = AssistantAgent(
            model_config=ModelConfig(model_id="test-model", temperature=0.7, max_tokens=1024),
            current_datetime="Tuesday, February 10, 2026 at 2:35 PM",
        )

        # Mock create_agent to capture the system_prompt
        with patch("patient_agent_bench.assistant_agent.default_agent.create_agent") as mock_create:
            mock_create.return_value = MagicMock()
            agent._create_agent(user_profile)

            # Verify create_agent was called with system_prompt containing user_profile
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            if user_profile and "{user_profile}" in agent.system_prompt_template:
                assert user_profile in call_kwargs.get("system_prompt", "")

    @given(
        template_with_missing=st.just("Hello {name}, welcome!"),
        extra_kwargs=st.dictionaries(
            keys=st.from_regex(r"[a-z_][a-z0-9_]{0,19}", fullmatch=True),
            values=st.text(min_size=0, max_size=50),
            min_size=1,
            max_size=3,
        ),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_ignores_missing_placeholders(self, template_with_missing, extra_kwargs):
        """
        Property 6 (continued): Test that extra kwargs are ignored without error.

        *For any* template and extra kwargs not present in the template,
        formatting SHALL succeed and ignore the extra values.

        **Validates: Requirements 4.2**
        """
        # Add a valid placeholder
        kwargs = {"name": "Alice"}
        kwargs.update(extra_kwargs)

        # Should not raise an error
        result = format_prompt_safe(template_with_missing, **kwargs)

        # Result should have the valid placeholder substituted
        assert "Alice" in result
        assert "{name}" not in result


# =============================================================================
# Tests for BenchConfig aggregation_method (Task 1.2)
# =============================================================================


class TestBenchConfigAggregationMethod:
    """Tests for BenchConfig aggregation_method field.

    **Feature: multi-evaluator-aggregation**
    **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
    """

    def test_default_aggregation_method(self):
        """Test that aggregation_method defaults to 'average'."""
        config = BenchConfig.from_dict({"strip_thinking_content": True})
        assert config.aggregation_method == "average"

    def test_average_aggregation_method(self):
        """Test explicit 'average' aggregation method."""
        config = BenchConfig.from_dict({
            "strip_thinking_content": True,
            "aggregation_method": "average",
        })
        assert config.aggregation_method == "average"

    def test_majority_vote_aggregation_method(self):
        """Test 'majority_vote' aggregation method."""
        config = BenchConfig.from_dict({
            "strip_thinking_content": True,
            "aggregation_method": "majority_vote",
        })
        assert config.aggregation_method == "majority_vote"

    def test_invalid_aggregation_method_raises(self):
        """Test that invalid aggregation_method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid aggregation_method"):
            BenchConfig.from_dict({
                "strip_thinking_content": True,
                "aggregation_method": "invalid_method",
            })

    def test_aggregation_method_in_to_dict(self):
        """Test that aggregation_method is included in to_dict output."""
        config = BenchConfig.from_dict({
            "strip_thinking_content": True,
            "aggregation_method": "majority_vote",
        })
        d = config.to_dict()
        assert "aggregation_method" in d
        assert d["aggregation_method"] == "majority_vote"

    def test_aggregation_method_round_trip(self):
        """Test aggregation_method survives to_dict/from_dict round trip."""
        for method in ["average", "majority_vote"]:
            config1 = BenchConfig.from_dict({
                "strip_thinking_content": True,
                "aggregation_method": method,
            })
            d = config1.to_dict()
            config2 = BenchConfig.from_dict(d)
            assert config2.aggregation_method == method

    def test_aggregation_method_default_in_to_dict(self):
        """Test that default aggregation_method appears in to_dict."""
        config = BenchConfig.from_dict({"strip_thinking_content": True})
        d = config.to_dict()
        assert d["aggregation_method"] == "average"
