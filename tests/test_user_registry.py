# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for user agent registry."""

from typing import Any, List, Optional

import pytest
from langchain_core.messages import BaseMessage

from patient_agent_bench.config import AgentSpec, ModelConfig
from patient_agent_bench.user_agent.base import BaseUserAgent
from patient_agent_bench.user_agent.registry import (
    _USER_AGENT_REGISTRY,
    create_user_agent_from_spec,
    get_user_agent_class,
    register_user_agent,
)


class FakeUserAgent(BaseUserAgent):
    """Minimal concrete subclass for testing."""

    def __init__(
        self,
        scenario: str,
        user_profile: str,
        model_config: ModelConfig,
        current_datetime: str,
        prompt_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        personality: Optional[str] = None,
        role_arn: Optional[str] = None,
    ) -> None:
        self.scenario = scenario
        self.user_profile = user_profile
        self.model_config = model_config
        self.current_datetime = current_datetime
        self.prompt_name = prompt_name
        self.system_prompt = system_prompt
        self.personality = personality
        self._role_arn = role_arn

    def invoke(self, messages: List[BaseMessage]) -> str:
        return "fake response"

    def start_conversation(self) -> str:
        return "fake opening"

    def respond(self, assistant_message: str) -> str:
        return "fake response"

    def is_conversation_complete(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def _clean_registry():
    """Save and restore registry state around each test."""
    saved = dict(_USER_AGENT_REGISTRY)
    yield
    _USER_AGENT_REGISTRY.clear()
    _USER_AGENT_REGISTRY.update(saved)


class TestRegisterUserAgent:
    def test_register_and_lookup(self):
        register_user_agent("fake", FakeUserAgent)
        assert get_user_agent_class("fake") is FakeUserAgent

    def test_overwrite_registration(self):
        register_user_agent("fake", FakeUserAgent)
        register_user_agent("fake", FakeUserAgent)
        assert get_user_agent_class("fake") is FakeUserAgent


class TestGetUserAgentClass:
    def test_unknown_name_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown user agent_class 'nonexistent'"):
            get_user_agent_class("nonexistent")

    def test_error_lists_available(self):
        register_user_agent("alpha", FakeUserAgent)
        register_user_agent("beta", FakeUserAgent)
        with pytest.raises(KeyError, match="Available: alpha, beta"):
            get_user_agent_class("missing")


class TestCreateUserAgentFromSpec:
    def test_creates_agent_from_spec(self):
        register_user_agent("fake", FakeUserAgent)
        spec = AgentSpec(
            model=ModelConfig(),
            prompt="my_prompt",
            agent_class="fake",
            system_prompt="You are a test patient.",
        )
        agent = create_user_agent_from_spec(
            spec=spec,
            scenario="Test scenario",
            user_profile="Test profile",
            current_datetime="2025-01-01T00:00:00",
        )
        assert isinstance(agent, FakeUserAgent)
        assert agent.prompt_name == "my_prompt"
        assert agent.system_prompt == "You are a test patient."
        assert agent.scenario == "Test scenario"
        assert agent.user_profile == "Test profile"

    def test_no_system_prompt_passes_none(self):
        register_user_agent("fake", FakeUserAgent)
        spec = AgentSpec(model=ModelConfig(), agent_class="fake")
        agent = create_user_agent_from_spec(
            spec=spec,
            scenario="Scenario",
            user_profile="Profile",
            current_datetime="2025-01-01T00:00:00",
        )
        assert agent.system_prompt is None
        assert agent.prompt_name == "default_prompt"

    def test_unregistered_class_raises_key_error(self):
        spec = AgentSpec(model=ModelConfig(), agent_class="nonexistent")
        with pytest.raises(KeyError):
            create_user_agent_from_spec(
                spec=spec,
                scenario="Scenario",
                user_profile="Profile",
                current_datetime="2025-01-01T00:00:00",
            )

# =============================================================================
# Property-Based Tests
# =============================================================================

from hypothesis import given, settings, strategies as st


class TestUserRegistryProperties:
    """Property-based tests for user agent registry."""

    @given(name=st.from_regex(r"[a-z][a-z0-9_]{0,19}", fullmatch=True))
    @settings(max_examples=50)
    def test_property_register_round_trip(self, name: str):
        """
        **Validates: Requirements 4.2, 4.3**

        Property 3: User registry round-trip.
        Register a class under a name, look it up, assert same class returned.
        """
        register_user_agent(name, FakeUserAgent)
        assert get_user_agent_class(name) is FakeUserAgent
