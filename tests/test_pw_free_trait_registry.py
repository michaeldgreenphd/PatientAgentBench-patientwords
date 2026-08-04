# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for the PatientWords free-trait user agent and its registration.

Modelled on tests/test_assistant_registry.py: the same registry save/restore
fixture, the same round-trip and factory coverage, plus the checks specific to
this adapter -- that a preset arm stays byte-identical to the default agent, and
that nothing from the placeholder preset the base constructor needs survives
into the prompt.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from patient_agent_bench.config import AgentSpec, format_prompt_safe
from patient_agent_bench.user_agent.default_agent import DefaultUserAgent
from patient_agent_bench.user_agent.default_prompt import SYSTEM_PROMPT
from patient_agent_bench.user_agent.personalities import (
    PERSONALITY_TYPE_NAMES,
    PERSONALITY_TYPES,
    get_personality_prompt,
)
from patient_agent_bench.user_agent.registry import (
    _USER_AGENT_REGISTRY,
    create_user_agent_from_spec,
    get_user_agent_class,
    register_user_agent,
)
from patientwords_pab.free_trait_agent import (
    FreeTraitUserAgent,
    _placeholder_preset,
    register_free_trait_agent,
)
from patientwords_pab.trait_spec import (
    PW_PREFIX,
    TraitSpecError,
    parse_trait_spec,
    render_trait_block,
)

DECISIVE = f"{PW_PREFIX}health_literacy=low;clarity=high"
SCENARIO = "Case background stand-in."
PROFILE = "Patient profile stand-in."


@pytest.fixture(autouse=True)
def _clean_registry():
    """Save and restore registry state around each test."""
    saved = dict(_USER_AGENT_REGISTRY)
    yield
    _USER_AGENT_REGISTRY.clear()
    _USER_AGENT_REGISTRY.update(saved)


def make_agent(model_config, current_datetime, personality, **kwargs):
    return FreeTraitUserAgent(
        scenario=SCENARIO,
        user_profile=PROFILE,
        model_config=model_config,
        current_datetime=current_datetime,
        personality=personality,
        **kwargs,
    )


class TestRegistration:
    def test_importing_the_package_registers_the_agent(self):
        """Upstream's registry docstring says new agents are imported into
        registry.py; that would be an upstream edit, so registration goes
        through the documented register_user_agent() hook at import time."""
        assert get_user_agent_class(FreeTraitUserAgent.NAME) is FreeTraitUserAgent

    def test_name_is_stable(self):
        assert FreeTraitUserAgent.NAME == "pw_free_trait"

    def test_register_is_idempotent(self):
        register_free_trait_agent()
        register_free_trait_agent()
        assert get_user_agent_class("pw_free_trait") is FreeTraitUserAgent

    def test_registration_does_not_displace_the_default_agent(self):
        assert get_user_agent_class(DefaultUserAgent.NAME) is DefaultUserAgent

    def test_re_registering_under_another_name_round_trips(self):
        register_user_agent("pw_alias", FreeTraitUserAgent)
        assert get_user_agent_class("pw_alias") is FreeTraitUserAgent


class TestCreateFromSpec:
    def test_factory_builds_the_agent_with_a_trait_spec(
        self, model_config, mock_boto3_client, current_datetime
    ):
        spec = AgentSpec(model=model_config, agent_class="pw_free_trait")
        agent = create_user_agent_from_spec(
            spec=spec,
            scenario=SCENARIO,
            user_profile=PROFILE,
            current_datetime=current_datetime,
            personality=DECISIVE,
        )
        assert isinstance(agent, FreeTraitUserAgent)
        assert agent.trait_levels["health_literacy"] == "low"
        assert agent.trait_levels["clarity"] == "high"

    def test_factory_rejects_a_malformed_spec_before_any_client_work(
        self, model_config, mock_boto3_client, current_datetime
    ):
        spec = AgentSpec(model=model_config, agent_class="pw_free_trait")
        with pytest.raises(TraitSpecError, match="unknown trait"):
            create_user_agent_from_spec(
                spec=spec,
                scenario=SCENARIO,
                user_profile=PROFILE,
                current_datetime=current_datetime,
                personality=f"{PW_PREFIX}not_a_trait=low",
            )


class TestFreeTraitPrompt:
    def test_prompt_is_the_template_with_our_block(
        self, model_config, mock_boto3_client, current_datetime
    ):
        agent = make_agent(model_config, current_datetime, DECISIVE)
        expected = format_prompt_safe(
            SYSTEM_PROMPT,
            scenario=SCENARIO,
            user_profile=PROFILE,
            current_datetime=current_datetime,
            personality_traits=render_trait_block(parse_trait_spec(DECISIVE).traits),
        )
        assert agent.system_prompt == expected

    def test_placeholder_preset_leaves_no_trace(
        self, model_config, mock_boto3_client, current_datetime
    ):
        """The base constructor is handed a real preset name because it calls
        get_personality_prompt() unconditionally. None of that preset may reach
        the final prompt."""
        agent = make_agent(model_config, current_datetime, DECISIVE)
        placeholder_block = get_personality_prompt(_placeholder_preset())
        assert placeholder_block not in agent.system_prompt
        assert agent.system_prompt.count("<personality-traits") == 1

    def test_personality_type_holds_the_real_spec(
        self, model_config, mock_boto3_client, current_datetime
    ):
        agent = make_agent(model_config, current_datetime, DECISIVE)
        assert agent.personality_type == DECISIVE
        assert agent.trait_spec is not None
        assert agent.arm_label == parse_trait_spec(DECISIVE).canonical

    def test_case_content_still_reaches_the_prompt(
        self, model_config, mock_boto3_client, current_datetime
    ):
        agent = make_agent(model_config, current_datetime, DECISIVE)
        assert SCENARIO in agent.system_prompt
        assert PROFILE in agent.system_prompt
        assert current_datetime in agent.system_prompt
        assert "{personality_traits}" not in agent.system_prompt

    def test_arms_differ_only_in_the_swept_trait_line(
        self, model_config, mock_boto3_client, current_datetime
    ):
        low = make_agent(model_config, current_datetime,
                         f"{PW_PREFIX}health_literacy=low").system_prompt
        high = make_agent(model_config, current_datetime,
                          f"{PW_PREFIX}health_literacy=high").system_prompt
        differing = [
            (a, b) for a, b in zip(low.splitlines(), high.splitlines(), strict=True) if a != b
        ]
        assert len(differing) == 1
        assert "health_literacy" in differing[0][0]

    def test_custom_system_prompt_template_is_honoured(
        self, model_config, mock_boto3_client, current_datetime
    ):
        template = "Persona:\n{personality_traits}\nCase: {scenario}"
        agent = make_agent(model_config, current_datetime, DECISIVE,
                           system_prompt=template)
        assert agent.system_prompt.startswith("Persona:")
        assert agent.system_prompt.endswith(f"Case: {SCENARIO}")


class TestPresetPassthrough:
    @pytest.mark.parametrize("preset", sorted(PERSONALITY_TYPES))
    def test_preset_arm_matches_the_default_agent_byte_for_byte(
        self, preset, model_config, mock_boto3_client, current_datetime
    ):
        """A config can mix preset arms and free-trait arms under one
        agent_class only if the preset path is untouched."""
        ours = make_agent(model_config, current_datetime, preset)
        theirs = DefaultUserAgent(
            scenario=SCENARIO,
            user_profile=PROFILE,
            model_config=model_config,
            current_datetime=current_datetime,
            personality=preset,
        )
        assert ours.system_prompt == theirs.system_prompt
        assert ours.trait_spec is None
        assert ours.arm_label == preset
        assert ours.trait_levels == PERSONALITY_TYPES[preset]

    def test_unknown_preset_still_raises_upstream_key_error(
        self, model_config, mock_boto3_client, current_datetime
    ):
        with pytest.raises(KeyError, match="Unknown personality type"):
            make_agent(model_config, current_datetime, "not_a_preset")

    def test_missing_personality_still_raises_upstream_key_error(
        self, model_config, mock_boto3_client, current_datetime
    ):
        with pytest.raises(KeyError):
            make_agent(model_config, current_datetime, None)


class TestConversationBehaviourUnchanged:
    def test_inherits_the_default_conversation_loop(
        self, model_config, mock_boto3_client, current_datetime
    ):
        agent = make_agent(model_config, current_datetime, DECISIVE)
        opening = agent.start_conversation()
        assert opening == "Mock LLM response"
        agent.respond("How can I help?")
        assert len(agent.get_chat_history()) == 4
        agent.reset()
        assert agent.get_chat_history() == []

    def test_drop_off_signal_still_ends_the_conversation(
        self, model_config, mock_boto3_client, current_datetime
    ):
        agent = make_agent(model_config, current_datetime, DECISIVE)
        assert not agent.is_conversation_complete()
        agent.chat_history.append(
            {"role": "assistant", "content": FreeTraitUserAgent.CONVERSATION_END_SIGNAL}
        )
        assert agent.is_conversation_complete()


# =============================================================================
# Property-Based Tests
# =============================================================================

class TestFreeTraitRegistryProperties:
    """Property-based tests for registry round-tripping of this agent."""

    @given(name=st.from_regex(r"pw_[a-z0-9_]{0,16}", fullmatch=True))
    @settings(max_examples=50)
    def test_property_register_round_trip(self, name: str):
        register_user_agent(name, FreeTraitUserAgent)
        assert get_user_agent_class(name) is FreeTraitUserAgent

    @given(preset=st.sampled_from(PERSONALITY_TYPE_NAMES))
    @settings(max_examples=len(PERSONALITY_TYPE_NAMES), deadline=None)
    def test_property_preset_base_reproduces_that_preset(self, preset: str):
        """pw:base=<preset> with no overrides resolves to exactly that preset."""
        spec = parse_trait_spec(f"{PW_PREFIX}base={preset}")
        for trait, level in PERSONALITY_TYPES[preset].items():
            assert spec.traits[trait] == level
