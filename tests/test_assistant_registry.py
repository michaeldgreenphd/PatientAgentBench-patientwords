# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for assistant agent registry."""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from patient_agent_bench.assistant_agent.base import BaseAssistantAgent
from patient_agent_bench.assistant_agent.registry import (
    _ASSISTANT_AGENT_REGISTRY,
    create_assistant_agent_from_spec,
    get_assistant_agent_class,
    register_assistant_agent,
)
from patient_agent_bench.config import AgentSpec, ModelConfig
from patient_agent_bench.tools.registry import ToolRegistry


class FakeAssistantAgent(BaseAssistantAgent):
    """Minimal concrete subclass for testing."""

    def __init__(
        self,
        model_config: ModelConfig,
        current_datetime: str,
        tools: Optional[List[BaseTool]] = None,
        prompt_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        role_arn: Optional[str] = None,
    ) -> None:
        self.model_config = model_config
        self.current_datetime = current_datetime
        self.tools = tools or []
        self.prompt_name = prompt_name
        self.system_prompt = system_prompt
        self._role_arn = role_arn

    def invoke(self, messages: List[BaseMessage], user_profile: str) -> Dict[str, Any]:
        return {"messages": messages}

    def get_tools(self) -> List[BaseTool]:
        return self.tools


@pytest.fixture(autouse=True)
def _clean_registry():
    """Save and restore registry state around each test."""
    saved = dict(_ASSISTANT_AGENT_REGISTRY)
    yield
    _ASSISTANT_AGENT_REGISTRY.clear()
    _ASSISTANT_AGENT_REGISTRY.update(saved)


class TestRegisterAssistantAgent:
    def test_register_and_lookup(self):
        register_assistant_agent("fake", FakeAssistantAgent)
        assert get_assistant_agent_class("fake") is FakeAssistantAgent

    def test_overwrite_registration(self):
        register_assistant_agent("fake", FakeAssistantAgent)
        register_assistant_agent("fake", FakeAssistantAgent)
        assert get_assistant_agent_class("fake") is FakeAssistantAgent


class TestGetAssistantAgentClass:
    def test_unknown_name_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown assistant agent_class 'nonexistent'"):
            get_assistant_agent_class("nonexistent")

    def test_error_lists_available(self):
        register_assistant_agent("alpha", FakeAssistantAgent)
        register_assistant_agent("beta", FakeAssistantAgent)
        with pytest.raises(KeyError, match="alpha.*beta"):
            get_assistant_agent_class("missing")


class TestCreateAssistantAgentFromSpec:
    def test_creates_agent_from_spec(self):
        register_assistant_agent("fake", FakeAssistantAgent)
        spec = AgentSpec(
            model=ModelConfig(),  # uses default model
            prompt="my_prompt",
            agent_class="fake",
            system_prompt="You are a test agent.",
        )
        agent = create_assistant_agent_from_spec(
            spec=spec,
            current_datetime="2025-01-01T00:00:00",
        )
        assert isinstance(agent, FakeAssistantAgent)
        assert agent.prompt_name == "my_prompt"
        assert agent.system_prompt == "You are a test agent."

    def test_uses_tool_registry_when_tools_none(self):
        register_assistant_agent("fake", FakeAssistantAgent)
        spec = AgentSpec(model=ModelConfig(), agent_class="fake")

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "mock_tool"
        registry = ToolRegistry()
        registry.register(mock_tool)

        agent = create_assistant_agent_from_spec(
            spec=spec,
            current_datetime="2025-01-01T00:00:00",
            tool_registry=registry,
        )
        assert len(agent.tools) == 1

    def test_explicit_tools_override_registry(self):
        register_assistant_agent("fake", FakeAssistantAgent)
        spec = AgentSpec(model=ModelConfig(), agent_class="fake")

        explicit_tool = MagicMock(spec=BaseTool)
        explicit_tool.name = "explicit"
        registry = ToolRegistry()

        agent = create_assistant_agent_from_spec(
            spec=spec,
            current_datetime="2025-01-01T00:00:00",
            tools=[explicit_tool],
            tool_registry=registry,
        )
        assert len(agent.tools) == 1

    def test_unregistered_class_raises_key_error(self):
        spec = AgentSpec(model=ModelConfig(), agent_class="nonexistent")
        with pytest.raises(KeyError):
            create_assistant_agent_from_spec(
                spec=spec,
                current_datetime="2025-01-01T00:00:00",
            )

# =============================================================================
# Property-Based Tests
# =============================================================================

from hypothesis import given, settings, strategies as st


class TestAssistantRegistryProperties:
    """Property-based tests for assistant agent registry."""

    @given(name=st.from_regex(r"[a-z][a-z0-9_]{0,19}", fullmatch=True))
    @settings(max_examples=50)
    def test_property_register_round_trip(self, name: str):
        """
        **Validates: Requirements 2.2, 2.3**

        Property 2: Assistant registry round-trip.
        Register a class under a name, look it up, assert same class returned.
        """
        register_assistant_agent(name, FakeAssistantAgent)
        assert get_assistant_agent_class(name) is FakeAssistantAgent

