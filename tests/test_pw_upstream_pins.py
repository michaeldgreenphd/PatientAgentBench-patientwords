# SPDX-License-Identifier: CC-BY-NC-4.0
"""Pins on the upstream surfaces the PatientWords adapter depends on.

Layer 1 never edits upstream, so it can only break by upstream moving under it.
These are the load-bearing surfaces, each with the failure it would cause:

1. ``personalities.TRAIT_DEFINITIONS`` -- the trait text the persona block is
   assembled from. Rename or restructure it and the adapter cannot build a
   block at all (loud).
2. ``default_prompt.SYSTEM_PROMPT`` containing ``{personality_traits}`` -- the
   slot the block is substituted into. This one is the dangerous pin:
   ``format_prompt_safe`` only *logs* a warning for a keyword whose placeholder
   is absent, so a rename produces a persona-free prompt with no error and no
   crash. Every arm would then run the same personality, and the sweep would
   measure nothing while looking healthy.
3. ``registry.register_user_agent`` -- the documented dynamic-registration hook
   the adapter uses instead of editing ``registry.py``.
4. ``DefaultUserAgent.__init__`` accepting the constructor keywords the subclass
   forwards, and rejecting a non-preset ``personality`` -- the reason the
   subclass passes a placeholder preset up and rebuilds the prompt afterwards.

A failure here is not necessarily a bug in either project: it is a signal that
the adapter needs re-reading against the new upstream.
"""

import inspect

import pytest

from patient_agent_bench.config import format_prompt_safe
from patient_agent_bench.user_agent.default_agent import DefaultUserAgent
from patient_agent_bench.user_agent.default_prompt import SYSTEM_PROMPT
from patient_agent_bench.user_agent.personalities import (
    PERSONALITY_TYPE_NAMES,
    PERSONALITY_TYPES,
    TRAIT_DEFINITIONS,
    get_personality_prompt,
)
from patient_agent_bench.user_agent.registry import register_user_agent
from patientwords_pab.trait_spec import NEUTRAL_LEVEL

# The two traits the integration design identifies as the register axes: the
# wording arm varies terminology (health_literacy), the dialect arm holds the
# term fixed and varies surrounding syntax (communication).
IDENTIFICATION_TRAITS = ("health_literacy", "communication")

PLACEHOLDER = "personality_traits"


class TestTraitDefinitionsPin:
    def test_importable_and_non_empty(self):
        assert isinstance(TRAIT_DEFINITIONS, dict)
        assert TRAIT_DEFINITIONS

    def test_shape_is_trait_to_level_to_description(self):
        for trait, levels in TRAIT_DEFINITIONS.items():
            assert isinstance(trait, str) and trait
            assert isinstance(levels, dict) and levels
            for level, description in levels.items():
                assert isinstance(level, str) and level
                assert isinstance(description, str) and description.strip()

    def test_every_trait_has_the_neutral_level(self):
        """The all-<NEUTRAL_LEVEL> base is the sweep's control arm; a trait
        without that level makes it unbuildable."""
        missing = [t for t, levels in TRAIT_DEFINITIONS.items()
                   if NEUTRAL_LEVEL not in levels]
        assert missing == []

    @pytest.mark.parametrize("trait", IDENTIFICATION_TRAITS)
    def test_identification_traits_still_exist(self, trait):
        """Rename either and the sweep silently changes what it measures."""
        assert trait in TRAIT_DEFINITIONS

    def test_presets_reference_only_defined_traits_and_levels(self):
        for name, preset in PERSONALITY_TYPES.items():
            for trait, level in preset.items():
                assert trait in TRAIT_DEFINITIONS, (name, trait)
                assert level in TRAIT_DEFINITIONS[trait], (name, trait, level)

    def test_preset_names_list_matches_the_preset_mapping(self):
        assert set(PERSONALITY_TYPE_NAMES) == set(PERSONALITY_TYPES)
        assert PERSONALITY_TYPE_NAMES, "the adapter needs one preset to pass upstream"


class TestPromptPlaceholderPin:
    def test_placeholder_present_in_the_template(self):
        assert "{" + PLACEHOLDER + "}" in SYSTEM_PROMPT

    def test_placeholder_is_actually_substituted(self):
        """The pin that matters. Presence in the source is not enough -- assert
        the real formatter really replaces it."""
        sentinel = "<<PW-PERSONA-SENTINEL>>"
        rendered = format_prompt_safe(SYSTEM_PROMPT, **{PLACEHOLDER: sentinel})
        assert sentinel in rendered
        assert "{" + PLACEHOLDER + "}" not in rendered

    def test_a_renamed_placeholder_would_fail_silently(self):
        """Documents the failure mode this pin guards: format_prompt_safe drops
        a keyword whose placeholder is missing, without raising."""
        renamed = SYSTEM_PROMPT.replace("{" + PLACEHOLDER + "}", "{persona_traits}")
        rendered = format_prompt_safe(renamed, **{PLACEHOLDER: "<<SENTINEL>>"})
        assert "<<SENTINEL>>" not in rendered  # no exception, no persona

    @pytest.mark.parametrize("name", ["scenario", "user_profile", "current_datetime"])
    def test_other_forwarded_placeholders_present(self, name):
        """The subclass re-renders the whole template, so every placeholder it
        passes must still exist or that content is dropped too."""
        assert "{" + name + "}" in SYSTEM_PROMPT


class TestRegistryHookPin:
    def test_register_user_agent_signature(self):
        params = list(inspect.signature(register_user_agent).parameters)
        assert params == ["name", "cls"]

    def test_registry_module_documents_dynamic_additions(self):
        """The hook is used instead of editing registry.py; its docstring is the
        upstream statement that this is a supported path."""
        assert "dynamic" in (register_user_agent.__doc__ or "").lower()


class TestDefaultUserAgentPin:
    def test_constructor_accepts_the_keywords_the_subclass_forwards(self):
        params = inspect.signature(DefaultUserAgent.__init__).parameters
        for name in ("scenario", "user_profile", "model_config", "current_datetime",
                     "prompt_name", "system_prompt", "personality", "role_arn"):
            assert name in params

    def test_constructor_exposes_the_template_it_resolved(self):
        """The subclass re-renders from ``self.system_prompt_template``; if the
        base class stops publishing it the rebuild has nothing to work from."""
        source = inspect.getsource(DefaultUserAgent.__init__)
        assert "self.system_prompt_template" in source
        assert "self.system_prompt" in source

    def test_non_preset_personality_is_rejected_upstream(self):
        """The reason a placeholder preset is passed to super().__init__ at all.
        If this ever stops raising, the placeholder becomes unnecessary -- but
        harmless, since the prompt is rebuilt regardless."""
        with pytest.raises(KeyError):
            get_personality_prompt("pw:health_literacy=low")

    def test_preset_block_shape_is_what_the_renderer_mirrors(self):
        block = get_personality_prompt(PERSONALITY_TYPE_NAMES[0])
        lines = block.splitlines()
        assert lines[0].startswith('<personality-traits type="')
        assert lines[-1] == "</personality-traits>"
        assert all(line.startswith("  <") for line in lines[1:-1])
