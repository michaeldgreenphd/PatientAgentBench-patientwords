# SPDX-License-Identifier: CC-BY-NC-4.0
"""Free-trait persona specs: parse, resolve, render, and characterise.

Upstream ships seven behavioural traits x three levels in ``TRAIT_DEFINITIONS``
but exposes them only through six fixed presets. Every preset moves several
traits at once, so a difference between two presets cannot be attributed to any
one trait. This module removes that restriction without touching upstream: it
resolves an arbitrary ``trait -> level`` mapping and renders it in exactly the
XML shape ``get_personality_prompt()`` produces, using upstream's own trait
descriptions as the text.

Spec strings. The ``personality`` field flows from the benchmark JSON to the
user agent as an opaque string, so the sweep is encoded there:

    pw:health_literacy=low;clarity=high        # neutral base, two traits set
    pw:base=confused;health_literacy=high      # a preset base, one trait flipped
    pw:base=neutral                            # the all-medium control arm

Anything without the ``pw:`` prefix is an upstream preset name and is left alone
(:func:`is_free_trait_spec` returns False), so a config can mix preset arms and
free-trait arms in one run.

Base. Unspecified traits come from the base. The default base is ``neutral`` --
*every* trait at ``medium`` -- not a preset. That matters for identification: a
preset base would silently re-bundle the confounds the sweep exists to separate.
``base=<preset>`` is available for ablations against a validated persona.

Rendering. The block's ``type`` attribute is a fixed literal (:data:`TRAIT_BLOCK_TYPE`)
and never carries the spec. It is the only free text in the block, so letting it
vary across arms would put an uncontrolled label in the prompt alongside the
trait being swept. Traits render in ``TRAIT_DEFINITIONS`` order, so two arms
differ by exactly the lines whose levels differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from patient_agent_bench.user_agent.personalities import (
    PERSONALITY_TYPES,
    TRAIT_DEFINITIONS,
)

# Prefix that marks a personality string as a free-trait spec rather than an
# upstream preset name. Chosen so it can never collide with a preset key.
PW_PREFIX = "pw:"

# Reserved assignment key naming the base the unspecified traits come from.
BASE_KEY = "base"

# Base name for the synthetic all-<NEUTRAL_LEVEL> starting point.
NEUTRAL_BASE = "neutral"

# The level the neutral base holds every trait at.
NEUTRAL_LEVEL = "medium"

# Fixed value of the rendered block's ``type`` attribute. Constant by design:
# see the module docstring.
TRAIT_BLOCK_TYPE = "custom"

_ASSIGN_SEP = ";"
_KV_SEP = "="

# A double quote would break out of the ``type`` / ``level`` attributes, and
# angle brackets would break out of the element itself.
_FORBIDDEN_IN_SPEC = ('"', "<", ">")


class TraitSpecError(ValueError):
    """A personality spec is malformed, or names a trait/level/base that does
    not exist upstream."""


@dataclass(frozen=True)
class TraitSpec:
    """A resolved free-trait persona.

    Attributes:
        raw: The spec string exactly as it arrived from the benchmark JSON.
        base: Name of the base -- ``neutral`` or an upstream preset key.
        overrides: Traits set explicitly by the spec, in the order written.
        traits: Every trait in ``TRAIT_DEFINITIONS`` order, base merged with
            overrides. This is what gets rendered.
    """

    raw: str
    base: str
    overrides: Dict[str, str] = field(default_factory=dict)
    traits: Dict[str, str] = field(default_factory=dict)

    @property
    def canonical(self) -> str:
        """The spec re-emitted in canonical form (base first, traits in
        ``TRAIT_DEFINITIONS`` order). Two specs that resolve identically share
        one canonical string, which makes it usable as an arm label."""
        return format_trait_spec(self.overrides, base=self.base)


def is_free_trait_spec(personality: Optional[str]) -> bool:
    """True if ``personality`` is a free-trait spec rather than a preset name."""
    return isinstance(personality, str) and personality.strip().startswith(PW_PREFIX)


def neutral_traits() -> Dict[str, str]:
    """Every upstream trait at :data:`NEUTRAL_LEVEL`.

    Raises:
        TraitSpecError: If upstream defines a trait without the neutral level.
            That is drift worth failing on, not papering over: the neutral base
            is the sweep's control arm.
    """
    if not TRAIT_DEFINITIONS:
        raise TraitSpecError("upstream TRAIT_DEFINITIONS is empty")
    missing = [t for t, levels in TRAIT_DEFINITIONS.items() if NEUTRAL_LEVEL not in levels]
    if missing:
        raise TraitSpecError(
            f"upstream trait(s) {', '.join(sorted(missing))} have no {NEUTRAL_LEVEL!r} "
            f"level, so the {NEUTRAL_BASE!r} base cannot be built"
        )
    return dict.fromkeys(TRAIT_DEFINITIONS, NEUTRAL_LEVEL)


def base_traits(base: str) -> Dict[str, str]:
    """Resolve a base name to a full trait mapping.

    ``neutral`` builds the all-medium control; any other name must be an
    upstream preset. A preset that omits a trait upstream defines gets the
    neutral level, so the result always covers every trait.
    """
    if base == NEUTRAL_BASE:
        return neutral_traits()
    if base not in PERSONALITY_TYPES:
        available = ", ".join([NEUTRAL_BASE] + sorted(PERSONALITY_TYPES))
        raise TraitSpecError(f"unknown base {base!r}. Available: {available}")
    resolved = neutral_traits()
    for trait, level in PERSONALITY_TYPES[base].items():
        if trait not in TRAIT_DEFINITIONS:
            # Upstream preset naming a trait with no definition: it renders as
            # nothing upstream, so drop it here too rather than invent text.
            continue
        resolved[trait] = level
    return resolved


def _validate(trait: str, level: str) -> None:
    if trait not in TRAIT_DEFINITIONS:
        raise TraitSpecError(
            f"unknown trait {trait!r}. Available: {', '.join(sorted(TRAIT_DEFINITIONS))}"
        )
    if level not in TRAIT_DEFINITIONS[trait]:
        raise TraitSpecError(
            f"unknown level {level!r} for trait {trait!r}. "
            f"Available: {', '.join(sorted(TRAIT_DEFINITIONS[trait]))}"
        )


def parse_trait_spec(personality: str) -> TraitSpec:
    """Parse a ``pw:`` spec string into a resolved :class:`TraitSpec`.

    Raises:
        TraitSpecError: On a missing prefix, a malformed assignment, a repeated
            trait, an unknown trait/level/base, or a character that would break
            the rendered XML.
    """
    if not is_free_trait_spec(personality):
        raise TraitSpecError(
            f"not a free-trait spec (expected the {PW_PREFIX!r} prefix): {personality!r}"
        )
    raw = personality.strip()
    for bad in _FORBIDDEN_IN_SPEC:
        if bad in raw:
            raise TraitSpecError(f"spec may not contain {bad!r}: {raw!r}")

    body = raw[len(PW_PREFIX):]
    base = NEUTRAL_BASE
    overrides: Dict[str, str] = {}

    for chunk in body.split(_ASSIGN_SEP):
        item = chunk.strip()
        if not item:
            continue  # tolerate a trailing or doubled ';'
        if _KV_SEP not in item:
            raise TraitSpecError(
                f"malformed assignment {item!r} in {raw!r} "
                f"(expected trait{_KV_SEP}level)"
            )
        key, _, value = item.partition(_KV_SEP)
        key, value = key.strip(), value.strip()
        if not key or not value:
            raise TraitSpecError(f"malformed assignment {item!r} in {raw!r}")
        if key == BASE_KEY:
            if base != NEUTRAL_BASE:
                raise TraitSpecError(f"{BASE_KEY!r} set more than once in {raw!r}")
            base_traits(value)  # validates
            base = value
            continue
        if key in overrides:
            raise TraitSpecError(f"trait {key!r} set more than once in {raw!r}")
        _validate(key, value)
        overrides[key] = value

    resolved = base_traits(base)
    resolved.update(overrides)
    # Re-key in TRAIT_DEFINITIONS order so rendering and diffing are stable.
    traits = {trait: resolved[trait] for trait in TRAIT_DEFINITIONS}
    return TraitSpec(raw=raw, base=base, overrides=overrides, traits=traits)


def format_trait_spec(overrides: Dict[str, str], base: str = NEUTRAL_BASE) -> str:
    """Build a canonical spec string from overrides (round-trips through
    :func:`parse_trait_spec`)."""
    parts = []
    if base != NEUTRAL_BASE:
        parts.append(f"{BASE_KEY}{_KV_SEP}{base}")
    for trait in TRAIT_DEFINITIONS:
        if trait in overrides:
            parts.append(f"{trait}{_KV_SEP}{overrides[trait]}")
    for trait, level in overrides.items():  # any trait upstream no longer defines
        if trait not in TRAIT_DEFINITIONS:
            parts.append(f"{trait}{_KV_SEP}{level}")
    return PW_PREFIX + _ASSIGN_SEP.join(parts)


def render_trait_block(
    traits: Dict[str, str],
    block_type: str = TRAIT_BLOCK_TYPE,
) -> str:
    """Render a trait mapping in upstream's ``<personality-traits>`` shape.

    Same element names, attribute order, indentation, and description text as
    ``get_personality_prompt()`` -- the simulator sees a block it cannot
    distinguish from a preset one except by the ``type`` value.
    """
    lines = [f'<personality-traits type="{block_type}">']
    for trait in TRAIT_DEFINITIONS:
        if trait not in traits:
            continue
        level = traits[trait]
        description = TRAIT_DEFINITIONS.get(trait, {}).get(level, "")
        if description:
            lines.append(f'  <{trait} level="{level}">{description}</{trait}>')
    lines.append("</personality-traits>")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Characterising an arm against upstream's validated persona space
# --------------------------------------------------------------------------- #

def preset_distances(traits: Dict[str, str]) -> List[Tuple[str, int, List[str]]]:
    """How far a trait mapping sits from each upstream preset.

    Returns ``(preset_name, n_differing, differing_traits)`` sorted nearest
    first. A free-trait arm is a claim about a persona upstream never validated;
    this says how large that claim is.
    """
    rows: List[Tuple[str, int, List[str]]] = []
    for name in PERSONALITY_TYPES:
        preset = base_traits(name)
        differing = [t for t in TRAIT_DEFINITIONS if traits.get(t) != preset.get(t)]
        rows.append((name, len(differing), differing))
    rows.sort(key=lambda row: (row[1], row[0]))
    return rows


def off_preset_level_pairs(traits: Dict[str, str]) -> List[Tuple[str, str, str, str]]:
    """Trait-level pairs in this arm that no upstream preset ever combines.

    Returns ``(trait_a, level_a, trait_b, level_b)`` for each unattested pair, in
    ``TRAIT_DEFINITIONS`` order. Empty means every pairwise combination appears
    in some preset -- the arm is novel only in its whole, not in any pair.
    """
    names = [t for t in TRAIT_DEFINITIONS if t in traits]
    presets = [base_traits(name) for name in PERSONALITY_TYPES]
    unattested: List[Tuple[str, str, str, str]] = []
    for i, trait_a in enumerate(names):
        for trait_b in names[i + 1:]:
            level_a, level_b = traits[trait_a], traits[trait_b]
            attested = any(
                p.get(trait_a) == level_a and p.get(trait_b) == level_b for p in presets
            )
            if not attested:
                unattested.append((trait_a, level_a, trait_b, level_b))
    return unattested


def single_factor_arms(
    trait: str,
    base: str = NEUTRAL_BASE,
    levels: Optional[Iterable[str]] = None,
    hold: Optional[Dict[str, str]] = None,
) -> List[TraitSpec]:
    """The arms of a single-factor sweep over one trait.

    Every arm shares the same base and the same ``hold`` overrides; only
    ``trait`` moves. Levels default to every level upstream defines for it, in
    definition order.
    """
    if trait not in TRAIT_DEFINITIONS:
        raise TraitSpecError(
            f"unknown trait {trait!r}. Available: {', '.join(sorted(TRAIT_DEFINITIONS))}"
        )
    hold = dict(hold or {})
    if trait in hold:
        raise TraitSpecError(f"cannot hold {trait!r} fixed and sweep it at the same time")
    for held_trait, held_level in hold.items():
        _validate(held_trait, held_level)
    chosen = list(levels) if levels is not None else list(TRAIT_DEFINITIONS[trait])
    arms = []
    for level in chosen:
        _validate(trait, level)
        overrides = dict(hold)
        overrides[trait] = level
        arms.append(parse_trait_spec(format_trait_spec(overrides, base=base)))
    return arms
