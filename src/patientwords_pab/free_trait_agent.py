# SPDX-License-Identifier: CC-BY-NC-4.0
"""User agent that accepts an arbitrary trait mapping instead of a preset.

Registered as ``agent_class="pw_free_trait"``. A benchmark entry selects the
persona through its ``personality`` field, which upstream carries as an opaque
string all the way to the agent constructor:

    {"scenario_id": "...", "personality": "pw:health_literacy=low", ...}

A personality WITHOUT the ``pw:`` prefix is an upstream preset name and is
handled entirely by :class:`DefaultUserAgent` -- the prompt is byte-identical to
what ``agent_class="default"`` would have produced. So one config can hold
preset arms and free-trait arms side by side.

Why a placeholder preset is passed to ``super().__init__``. Upstream's
constructor calls ``get_personality_prompt(personality)`` unconditionally while
building the system prompt, and that raises ``KeyError`` for anything that is
not a preset key. Editing upstream to relax it is out of scope by design (it
would end the clean ``git pull upstream main``), so the base constructor is
handed a real preset name it can resolve, and the system prompt is then rebuilt
from the same template with the real trait block. Nothing from the placeholder
survives into ``self.system_prompt`` -- ``tests/test_pw_free_trait_registry.py``
asserts exactly that.
"""

from __future__ import annotations

from typing import Dict, Optional

from patient_agent_bench.config import ModelConfig, format_prompt_safe
from patient_agent_bench.user_agent.default_agent import DefaultUserAgent
from patient_agent_bench.user_agent.personalities import (
    PERSONALITY_TYPE_NAMES,
    PERSONALITY_TYPES,
)
from patient_agent_bench.user_agent.registry import register_user_agent
from patientwords_pab.trait_spec import (
    TraitSpec,
    TraitSpecError,
    is_free_trait_spec,
    parse_trait_spec,
    render_trait_block,
)


def _placeholder_preset() -> str:
    """A preset name that ``super().__init__`` can resolve. Never reaches the
    final prompt; see the module docstring."""
    if not PERSONALITY_TYPE_NAMES:
        raise TraitSpecError(
            "upstream defines no personality presets, so the base constructor "
            "has nothing to resolve"
        )
    return PERSONALITY_TYPE_NAMES[0]


class FreeTraitUserAgent(DefaultUserAgent):
    """A :class:`DefaultUserAgent` whose persona block is built from a trait
    mapping rather than a preset.

    Extra attributes beyond the base class:
        trait_spec: The parsed :class:`TraitSpec`, or ``None`` on a preset arm.
        trait_levels: The resolved ``trait -> level`` mapping for this agent
            (populated for preset arms too, so analysis code has one shape).
        arm_label: Canonical spec string for a free-trait arm, or the preset
            name. Stable across equivalent specs, so it groups runs.
    """

    NAME = "pw_free_trait"

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
        spec: Optional[TraitSpec] = (
            parse_trait_spec(personality) if is_free_trait_spec(personality) else None
        )

        super().__init__(
            scenario=scenario,
            user_profile=user_profile,
            model_config=model_config,
            current_datetime=current_datetime,
            prompt_name=prompt_name,
            system_prompt=system_prompt,
            personality=_placeholder_preset() if spec is not None else personality,
            role_arn=role_arn,
        )

        self.trait_spec = spec

        if spec is None:
            # Preset arm: leave the base class's prompt untouched.
            self.trait_levels: Dict[str, str] = dict(
                PERSONALITY_TYPES.get(personality or "", {})
            )
            self.arm_label = personality or ""
            return

        # Free-trait arm: restore the real personality string the base class
        # overwrote with the placeholder, then rebuild the prompt.
        self.personality_type = spec.raw
        self.trait_levels = dict(spec.traits)
        self.arm_label = spec.canonical
        self.system_prompt = format_prompt_safe(
            self.system_prompt_template,
            scenario=scenario,
            user_profile=user_profile,
            current_datetime=current_datetime,
            personality_traits=render_trait_block(spec.traits),
        )


def register_free_trait_agent() -> None:
    """Register the agent under :data:`FreeTraitUserAgent.NAME`.

    Idempotent (upstream's registry is a plain dict keyed by name). Called on
    import of this module, so ``import patientwords_pab`` is enough to make
    ``"agent_class": "pw_free_trait"`` resolvable.
    """
    register_user_agent(FreeTraitUserAgent.NAME, FreeTraitUserAgent)


register_free_trait_agent()
