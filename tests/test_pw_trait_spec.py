# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for the PatientWords free-trait spec parser, renderer, and dry run.

Nothing here constructs an agent or an LLM client: the spec layer is pure
functions over upstream's TRAIT_DEFINITIONS, and the dry-run renderer is the
cheap way to inspect a persona before spending anything on it.
"""

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from patient_agent_bench.user_agent.personalities import (
    PERSONALITY_TYPES,
    TRAIT_DEFINITIONS,
    get_personality_prompt,
)
from patientwords_pab import render_persona
from patientwords_pab.trait_spec import (
    NEUTRAL_BASE,
    NEUTRAL_LEVEL,
    PW_PREFIX,
    TRAIT_BLOCK_TYPE,
    TraitSpecError,
    base_traits,
    format_trait_spec,
    is_free_trait_spec,
    neutral_traits,
    off_preset_level_pairs,
    parse_trait_spec,
    preset_distances,
    render_trait_block,
    single_factor_arms,
)

# The combination the integration design calls decisive: a register split that
# no upstream preset contains (only "confused" carries health_literacy=low, and
# it carries clarity=low with it).
DECISIVE = f"{PW_PREFIX}health_literacy=low;clarity=high"


class TestIsFreeTraitSpec:
    def test_prefixed_string_is_a_spec(self):
        assert is_free_trait_spec(DECISIVE)

    def test_preset_name_is_not_a_spec(self):
        for preset in PERSONALITY_TYPES:
            assert not is_free_trait_spec(preset)

    def test_none_and_empty_are_not_specs(self):
        assert not is_free_trait_spec(None)
        assert not is_free_trait_spec("")


class TestParseTraitSpec:
    def test_neutral_base_fills_unset_traits(self):
        spec = parse_trait_spec(DECISIVE)
        assert spec.base == NEUTRAL_BASE
        assert spec.overrides == {"health_literacy": "low", "clarity": "high"}
        assert set(spec.traits) == set(TRAIT_DEFINITIONS)
        assert spec.traits["health_literacy"] == "low"
        assert spec.traits["clarity"] == "high"
        for trait, level in spec.traits.items():
            if trait not in spec.overrides:
                assert level == NEUTRAL_LEVEL

    def test_traits_render_in_upstream_definition_order(self):
        spec = parse_trait_spec(DECISIVE)
        assert list(spec.traits) == list(TRAIT_DEFINITIONS)

    def test_preset_base_inherits_that_preset(self):
        spec = parse_trait_spec(f"{PW_PREFIX}base=confused;clarity=high")
        assert spec.base == "confused"
        assert spec.traits["health_literacy"] == PERSONALITY_TYPES["confused"]["health_literacy"]
        assert spec.traits["clarity"] == "high"

    def test_whitespace_and_trailing_separator_tolerated(self):
        spec = parse_trait_spec(f"  {PW_PREFIX} health_literacy = low ; ; clarity=high ;")
        assert spec.overrides == {"health_literacy": "low", "clarity": "high"}

    def test_raw_is_preserved_verbatim(self):
        spec = parse_trait_spec(f"  {DECISIVE}  ")
        assert spec.raw == DECISIVE

    def test_missing_prefix_rejected(self):
        with pytest.raises(TraitSpecError, match="prefix"):
            parse_trait_spec("health_literacy=low")

    def test_unknown_trait_rejected(self):
        with pytest.raises(TraitSpecError, match="unknown trait"):
            parse_trait_spec(f"{PW_PREFIX}not_a_trait=low")

    def test_unknown_level_rejected(self):
        with pytest.raises(TraitSpecError, match="unknown level"):
            parse_trait_spec(f"{PW_PREFIX}health_literacy=extremely_low")

    def test_unknown_base_rejected(self):
        with pytest.raises(TraitSpecError, match="unknown base"):
            parse_trait_spec(f"{PW_PREFIX}base=not_a_preset")

    def test_repeated_trait_rejected(self):
        with pytest.raises(TraitSpecError, match="more than once"):
            parse_trait_spec(f"{PW_PREFIX}health_literacy=low;health_literacy=high")

    def test_repeated_base_rejected(self):
        with pytest.raises(TraitSpecError, match="more than once"):
            parse_trait_spec(f"{PW_PREFIX}base=confused;base=skeptical")

    def test_malformed_assignment_rejected(self):
        with pytest.raises(TraitSpecError, match="malformed"):
            parse_trait_spec(f"{PW_PREFIX}health_literacy")

    @pytest.mark.parametrize("bad", ['"', "<", ">"])
    def test_xml_breaking_characters_rejected(self, bad):
        # These would escape the rendered attribute or element.
        with pytest.raises(TraitSpecError, match="may not contain"):
            parse_trait_spec(f"{PW_PREFIX}health_literacy=low{bad}")

    def test_canonical_is_stable_across_orderings(self):
        a = parse_trait_spec(f"{PW_PREFIX}clarity=high;health_literacy=low")
        b = parse_trait_spec(f"{PW_PREFIX}health_literacy=low;clarity=high")
        assert a.canonical == b.canonical
        assert a.traits == b.traits


class TestBases:
    def test_neutral_covers_every_trait_at_the_neutral_level(self):
        traits = neutral_traits()
        assert set(traits) == set(TRAIT_DEFINITIONS)
        assert set(traits.values()) == {NEUTRAL_LEVEL}

    def test_preset_base_matches_upstream_preset(self):
        for name, preset in PERSONALITY_TYPES.items():
            resolved = base_traits(name)
            for trait, level in preset.items():
                if trait in TRAIT_DEFINITIONS:
                    assert resolved[trait] == level


class TestRenderTraitBlock:
    def test_structurally_matches_upstream_for_the_same_traits(self):
        """A preset rendered through our renderer differs from upstream's own
        output only in the type attribute -- same elements, levels, text,
        ordering, indentation."""
        for name in PERSONALITY_TYPES:
            ours = render_trait_block(base_traits(name))
            theirs = get_personality_prompt(name)
            assert ours.replace(f'type="{TRAIT_BLOCK_TYPE}"', f'type="{name}"') == theirs

    def test_type_attribute_never_carries_the_spec(self):
        """The type attribute is the only free text in the block; if it varied
        by arm it would be an uncontrolled label riding alongside the swept
        trait."""
        blocks = [
            render_trait_block(parse_trait_spec(s).traits)
            for s in (DECISIVE, f"{PW_PREFIX}health_literacy=high", f"{PW_PREFIX}base=stoic")
        ]
        for block in blocks:
            assert block.startswith(f'<personality-traits type="{TRAIT_BLOCK_TYPE}">')
            assert "health_literacy=" not in block.splitlines()[0]

    def test_uses_upstream_description_text_verbatim(self):
        block = render_trait_block(parse_trait_spec(DECISIVE).traits)
        assert TRAIT_DEFINITIONS["health_literacy"]["low"] in block
        assert TRAIT_DEFINITIONS["clarity"]["high"] in block

    def test_two_arms_differ_only_on_the_swept_line(self):
        low = render_trait_block(parse_trait_spec(f"{PW_PREFIX}health_literacy=low").traits)
        high = render_trait_block(parse_trait_spec(f"{PW_PREFIX}health_literacy=high").traits)
        differing = [
            (a, b) for a, b in zip(low.splitlines(), high.splitlines(), strict=True) if a != b
        ]
        assert len(differing) == 1
        assert "health_literacy" in differing[0][0]


class TestNovelty:
    def test_preset_is_zero_distance_from_itself(self):
        for name in PERSONALITY_TYPES:
            nearest = preset_distances(base_traits(name))[0]
            assert nearest[1] == 0

    def test_decisive_pair_is_unattested_upstream(self):
        """No upstream preset combines low health literacy with high clarity --
        which is why the sweep needs the adapter at all."""
        traits = parse_trait_spec(f"{PW_PREFIX}base=confused;clarity=high").traits
        pairs = off_preset_level_pairs(traits)
        assert ("health_literacy", "low", "clarity", "high") in pairs

    def test_preset_arms_have_no_unattested_pairs(self):
        for name in PERSONALITY_TYPES:
            assert off_preset_level_pairs(base_traits(name)) == []


class TestSingleFactorArms:
    def test_one_arm_per_level_varying_only_that_trait(self):
        arms = single_factor_arms("health_literacy")
        assert len(arms) == len(TRAIT_DEFINITIONS["health_literacy"])
        swept = {arm.traits["health_literacy"] for arm in arms}
        assert swept == set(TRAIT_DEFINITIONS["health_literacy"])
        for trait in TRAIT_DEFINITIONS:
            if trait == "health_literacy":
                continue
            assert len({arm.traits[trait] for arm in arms}) == 1

    def test_hold_applies_to_every_arm(self):
        arms = single_factor_arms("health_literacy", hold={"clarity": "high"})
        assert all(arm.traits["clarity"] == "high" for arm in arms)

    def test_cannot_hold_and_sweep_the_same_trait(self):
        with pytest.raises(TraitSpecError, match="hold"):
            single_factor_arms("clarity", hold={"clarity": "high"})

    def test_unknown_trait_rejected(self):
        with pytest.raises(TraitSpecError, match="unknown trait"):
            single_factor_arms("not_a_trait")

    def test_preset_base_carries_into_every_arm(self):
        arms = single_factor_arms("health_literacy", base="confused")
        assert all(arm.base == "confused" for arm in arms)
        assert all(arm.traits["urgency"] == PERSONALITY_TYPES["confused"]["urgency"]
                   for arm in arms)


class TestDryRunRenderer:
    def test_renders_a_free_trait_arm(self, capsys):
        assert render_persona.main([DECISIVE]) == 0
        out = capsys.readouterr().out
        assert TRAIT_DEFINITIONS["health_literacy"]["low"] in out
        assert TRAIT_DEFINITIONS["clarity"]["high"] in out

    def test_renders_a_preset_arm_through_upstream(self, capsys):
        assert render_persona.main(["confused"]) == 0
        assert 'type="confused"' in capsys.readouterr().out

    def test_sweep_reports_single_factor_identification(self, capsys):
        assert render_persona.main(["--sweep", "health_literacy"]) == 0
        out = capsys.readouterr().out
        assert "single-factor identification: OK" in out
        assert "varying (1): health_literacy" in out

    def test_json_mode_is_machine_readable(self, capsys):
        assert render_persona.main(["--sweep", "communication", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["arms"]) == len(TRAIT_DEFINITIONS["communication"])
        assert payload["contrast"]["varying"] == ["communication"]
        arm = payload["arms"][0]
        assert arm["kind"] == "free_trait"
        assert "unattested_pairs" in arm and "nearest_presets" in arm

    def test_full_prompt_embeds_the_block_and_no_case_content(self, capsys):
        assert render_persona.main([DECISIVE, "--full-prompt"]) == 0
        out = capsys.readouterr().out
        assert render_persona.PLACEHOLDER_SCENARIO in out
        assert render_persona.PLACEHOLDER_PROFILE in out
        assert "{personality_traits}" not in out

    def test_unattested_pairs_flag_the_ones_the_spec_introduced(self, capsys):
        assert render_persona.main([f"{PW_PREFIX}base=confused;clarity=high", "--json"]) == 0
        pairs = json.loads(capsys.readouterr().out)["arms"][0]["unattested_pairs"]
        assert pairs and all(p["involves_override"] for p in pairs)

    def test_bad_spec_exits_two(self, capsys):
        assert render_persona.main([f"{PW_PREFIX}not_a_trait=low"]) == 2
        assert "unknown trait" in capsys.readouterr().err

    def test_diff_across_mixed_arms(self, capsys):
        assert render_persona.main([f"{PW_PREFIX}base=confused",
                                    f"{PW_PREFIX}base=confused;clarity=high"]) == 0
        out = capsys.readouterr().out
        assert "varying (1): clarity" in out


# =============================================================================
# Property-Based Tests
# =============================================================================

_TRAITS = sorted(TRAIT_DEFINITIONS)


@st.composite
def trait_overrides(draw):
    chosen = draw(st.lists(st.sampled_from(_TRAITS), unique=True, min_size=1,
                           max_size=len(_TRAITS)))
    return {t: draw(st.sampled_from(sorted(TRAIT_DEFINITIONS[t]))) for t in chosen}


class TestTraitSpecProperties:
    """Property-based tests for the spec round trip."""

    @given(overrides=trait_overrides(),
           base=st.sampled_from([NEUTRAL_BASE] + sorted(PERSONALITY_TYPES)))
    @settings(max_examples=50)
    def test_property_format_parse_round_trip(self, overrides, base):
        """Any valid (base, overrides) formats to a spec string that parses back
        to the same base, the same overrides, and the same resolved traits."""
        spec = parse_trait_spec(format_trait_spec(overrides, base=base))
        assert spec.base == base
        assert spec.overrides == overrides
        expected = base_traits(base)
        expected.update(overrides)
        assert spec.traits == expected

    @given(overrides=trait_overrides())
    @settings(max_examples=50)
    def test_property_block_has_one_line_per_trait(self, overrides):
        spec = parse_trait_spec(format_trait_spec(overrides))
        lines = render_trait_block(spec.traits).splitlines()
        assert len(lines) == len(TRAIT_DEFINITIONS) + 2  # open + close tags
