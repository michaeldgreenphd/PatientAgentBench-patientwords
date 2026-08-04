# SPDX-License-Identifier: CC-BY-NC-4.0
"""Dry-run persona renderer: print what a trait spec builds, calling no model.

The open question this answers, before any spend: does a trait combination that
no upstream preset contains -- ``health_literacy=low`` with ``clarity=high``, say
-- produce a coherent instruction block, or a self-contradicting one the patient
simulator would partly ignore? The block is deterministic text assembled from
upstream's own trait descriptions, so it can be read in full offline.

It also reports two things the eye cannot compute: how far the arm sits from
every upstream preset, and which of its trait-level pairs no preset attests. An
arm whose pairs are all attested is a recombination of validated ground; an arm
with unattested pairs is a persona upstream never checked, and any result from
it carries that caveat.

    # one arm
    python -m patientwords_pab.render_persona "pw:health_literacy=low;clarity=high"

    # the three arms of a single-factor sweep, plus the identification check
    python -m patientwords_pab.render_persona --sweep health_literacy

    # a free-trait arm beside the preset it is nearest to
    python -m patientwords_pab.render_persona "pw:health_literacy=low" confused

    # the whole system prompt, with abstract stand-ins for the case content
    python -m patientwords_pab.render_persona --sweep communication --full-prompt

No LLM client is constructed and no network call is made: this module touches
only ``TRAIT_DEFINITIONS``, ``PERSONALITY_TYPES``, the prompt template, and
upstream's ``format_prompt_safe``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from patient_agent_bench.config import format_prompt_safe
from patient_agent_bench.user_agent.default_prompt import SYSTEM_PROMPT
from patient_agent_bench.user_agent.personalities import (
    PERSONALITY_TYPES,
    TRAIT_DEFINITIONS,
    get_personality_prompt,
)
from patientwords_pab.trait_spec import (
    NEUTRAL_BASE,
    NEUTRAL_LEVEL,
    TraitSpec,
    TraitSpecError,
    base_traits,
    is_free_trait_spec,
    off_preset_level_pairs,
    parse_trait_spec,
    preset_distances,
    render_trait_block,
    single_factor_arms,
)

# Stand-ins for the per-case content, so the rendered prompt shows structure
# without pulling any clinical text into this file.
PLACEHOLDER_SCENARIO = "<case background omitted in dry run>"
PLACEHOLDER_PROFILE = "<patient profile omitted in dry run>"
PLACEHOLDER_DATETIME = "<run datetime omitted in dry run>"


def _arm_from_argument(argument: str) -> Dict[str, Any]:
    """Resolve a CLI argument -- a ``pw:`` spec or an upstream preset name --
    into a uniform arm record."""
    if is_free_trait_spec(argument):
        spec: Optional[TraitSpec] = parse_trait_spec(argument)
        return _arm_from_spec(spec)
    if argument not in PERSONALITY_TYPES:
        available = ", ".join(sorted(PERSONALITY_TYPES))
        raise TraitSpecError(
            f"{argument!r} is neither a free-trait spec nor an upstream preset. "
            f"Presets: {available}"
        )
    traits = base_traits(argument)
    return {
        "label": argument,
        "kind": "preset",
        "base": argument,
        "overrides": {},
        "traits": traits,
        # Preset arms render through upstream's own function, so the dry run
        # shows the real text including its type attribute.
        "block": get_personality_prompt(argument),
    }


def _arm_from_spec(spec: TraitSpec) -> Dict[str, Any]:
    return {
        "label": spec.canonical,
        "kind": "free_trait",
        "base": spec.base,
        "overrides": dict(spec.overrides),
        "traits": dict(spec.traits),
        "block": render_trait_block(spec.traits),
    }


def build_arms(
    arguments: List[str],
    sweep: Optional[str] = None,
    base: str = NEUTRAL_BASE,
    hold: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Arms for the requested rendering, sweep arms first."""
    arms: List[Dict[str, Any]] = []
    if sweep:
        arms.extend(
            _arm_from_spec(s) for s in single_factor_arms(sweep, base=base, hold=hold)
        )
    arms.extend(_arm_from_argument(a) for a in arguments)
    return arms


def describe_arm(arm: Dict[str, Any]) -> Dict[str, Any]:
    """Add the novelty characterisation to an arm record."""
    traits = arm["traits"]
    distances = preset_distances(traits)
    unattested = off_preset_level_pairs(traits)
    described = dict(arm)
    described["nearest_presets"] = [
        {"preset": name, "n_differing": n, "differing": differing}
        for name, n, differing in distances
    ]
    overrides = arm["overrides"]
    described["unattested_pairs"] = [
        {
            "trait_a": a, "level_a": la, "trait_b": b, "level_b": lb,
            # Pairs that involve a trait this spec sets are the ones the sweep
            # introduced; the rest come from the base and are novel whatever the
            # spec does (the all-medium neutral base is itself off-manifold).
            "involves_override": a in overrides or b in overrides,
        }
        for a, la, b, lb in unattested
    ]
    return described


def diff_arms(arms: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Which traits vary across the arms, and which are held fixed.

    For a single-factor sweep, ``varying`` must have exactly one entry -- that is
    the identification condition, and it is checkable here for free.
    """
    varying, fixed = [], []
    for trait in TRAIT_DEFINITIONS:
        levels = {arm["traits"].get(trait) for arm in arms}
        (varying if len(levels) > 1 else fixed).append(trait)
    return {"varying": varying, "fixed": fixed}


def full_prompt(block: str) -> str:
    """The whole user-agent system prompt with this persona block in place."""
    return format_prompt_safe(
        SYSTEM_PROMPT,
        scenario=PLACEHOLDER_SCENARIO,
        user_profile=PLACEHOLDER_PROFILE,
        current_datetime=PLACEHOLDER_DATETIME,
        personality_traits=block,
    )


def _print_arm(arm: Dict[str, Any], show_prompt: bool) -> None:
    print(f"arm: {arm['label']}")
    base_note = (
        f"{NEUTRAL_BASE} (every trait at {NEUTRAL_LEVEL!r})"
        if arm["base"] == NEUTRAL_BASE
        else f"preset {arm['base']!r}"
    )
    print(f"  kind: {arm['kind']}    base: {base_note}")
    print(f"  resolved traits ({len(arm['traits'])}):")
    width = max((len(t) for t in arm["traits"]), default=0)
    for trait, level in arm["traits"].items():
        origin = "set" if trait in arm["overrides"] else "base"
        print(f"    {trait:<{width}}  {level:<7} ({origin})")

    nearest = arm["nearest_presets"][0] if arm["nearest_presets"] else None
    if nearest:
        if nearest["n_differing"] == 0:
            print(f"  novelty: identical to upstream preset {nearest['preset']!r}")
        else:
            print(
                f"  novelty: nearest upstream preset is {nearest['preset']!r}, "
                f"differing on {nearest['n_differing']} of {len(arm['traits'])} "
                f"traits ({', '.join(nearest['differing'])})"
            )
    pairs = arm["unattested_pairs"]
    if pairs:
        from_spec = sum(1 for p in pairs if p["involves_override"])
        print(
            f"  trait-level pairs no preset attests ({len(pairs)}; "
            f"{from_spec} involve a trait this spec sets, marked *):"
        )
        for p in pairs:
            mark = "*" if p["involves_override"] else " "
            print(
                f"   {mark} {p['trait_a']}={p['level_a']} + "
                f"{p['trait_b']}={p['level_b']}"
            )
    else:
        print("  trait-level pairs no preset attests: none")

    print("  persona block:")
    for line in arm["block"].splitlines():
        print(f"    {line}")
    if show_prompt:
        print("  full system prompt:")
        for line in full_prompt(arm["block"]).splitlines():
            print(f"    {line}")
    print()


def _parse_hold(values: Optional[List[str]]) -> Dict[str, str]:
    hold: Dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise TraitSpecError(f"malformed --hold {item!r} (expected trait=level)")
        trait, _, level = item.partition("=")
        hold[trait.strip()] = level.strip()
    return hold


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "specs", nargs="*",
        help="free-trait specs (pw:trait=level;...) and/or upstream preset names",
    )
    parser.add_argument("--sweep", help="render every level of this trait as an arm")
    parser.add_argument(
        "--base", default=NEUTRAL_BASE,
        help=f"base for --sweep arms: {NEUTRAL_BASE} or a preset name "
             f"(default: {NEUTRAL_BASE})",
    )
    parser.add_argument(
        "--hold", action="append", metavar="TRAIT=LEVEL",
        help="hold an extra trait fixed across --sweep arms (repeatable)",
    )
    parser.add_argument(
        "--full-prompt", action="store_true",
        help="also print the complete system prompt for each arm",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    if not args.specs and not args.sweep:
        parser.error("give at least one spec or --sweep TRAIT")

    try:
        arms = [
            describe_arm(a)
            for a in build_arms(
                args.specs, sweep=args.sweep, base=args.base,
                hold=_parse_hold(args.hold),
            )
        ]
    except TraitSpecError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    contrast = diff_arms(arms) if len(arms) > 1 else None

    if args.json:
        payload: Dict[str, Any] = {"arms": arms}
        if args.full_prompt:
            for arm in payload["arms"]:
                arm["full_prompt"] = full_prompt(arm["block"])
        if contrast:
            payload["contrast"] = contrast
        print(json.dumps(payload, indent=2))
        return 0

    for arm in arms:
        _print_arm(arm, args.full_prompt)

    if contrast:
        varying = contrast["varying"]
        print(f"across {len(arms)} arms:")
        print(f"  varying ({len(varying)}): {', '.join(varying) or 'none'}")
        print(f"  held fixed ({len(contrast['fixed'])}): {', '.join(contrast['fixed'])}")
        if args.sweep:
            identified = varying == [args.sweep]
            print(
                f"  single-factor identification: "
                f"{'OK' if identified else 'BROKEN'} "
                f"(exactly one trait may vary; it must be {args.sweep!r})"
            )
            if not identified:
                return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
