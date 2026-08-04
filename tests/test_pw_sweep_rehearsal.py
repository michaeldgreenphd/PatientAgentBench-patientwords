# SPDX-License-Identifier: CC-BY-NC-4.0
"""Build a sweep and run it end to end offline, before a dollar is spent.

Two jobs. The first is testing `build_sweep`: a sweep is only interpretable if
the arms of a case differ in nothing but `personality`, and that is a property
of the file, checkable for free.

The second is a full-pipeline rehearsal. Every model is mocked -- following
upstream's own patching points in tests/test_conversation_runner.py -- but
`ExperimentRunner`, `ConversationRunner` and `OutputManager` are the real
things, so the run directory that comes out is genuinely what a paid run would
write. That directory is then checked against the same invariants the
measurement engine's Layer-2 validator enforces, which closes the loop between
the two repositories without either importing the other.

The point is narrow and worth stating: after this passes, the only untested
thing between here and a live pilot is whether the models themselves behave.
The plumbing is known good.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from patient_agent_bench.benchmark_seed import load_benchmark_entries
from patient_agent_bench.config import AgentSpec, BenchConfig, ModelConfig
from patient_agent_bench.runner.experiment_runner import ExperimentRunner
from patient_agent_bench.runner.output_manager import OutputManager
from patientwords_pab.build_sweep import (
    SweepError,
    arm_slug,
    build,
    build_entries,
    resolve_arms,
    verify,
    write,
)

SAMPLE = "data/sample_benchmark.json"
FOUR_ARMS = ["confused",
             "pw:base=confused;health_literacy=low",
             "pw:base=confused;health_literacy=medium",
             "pw:base=confused;health_literacy=high"]


# =============================================================================
# Arm resolution
# =============================================================================

class TestResolveArms:
    def test_the_documented_four_arm_design(self):
        arms = resolve_arms(sweep="health_literacy", base="confused",
                            anchor_preset="confused")
        assert arms == FOUR_ARMS

    def test_anchor_preset_comes_first(self):
        """It renders through upstream's own function, so it is the arm whose
        score is comparable to published preset numbers."""
        arms = resolve_arms(sweep="health_literacy", base="confused",
                            anchor_preset="stoic")
        assert arms[0] == "stoic"

    def test_explicit_arms_are_kept_in_order(self):
        arms = resolve_arms(arms=["confused", "pw:health_literacy=high"])
        assert arms == ["confused", "pw:health_literacy=high"]

    def test_one_arm_is_not_a_sweep(self):
        with pytest.raises(SweepError, match="at least two arms"):
            resolve_arms(arms=["confused"])

    def test_duplicate_arms_rejected(self):
        with pytest.raises(SweepError, match="duplicate arm"):
            resolve_arms(arms=["confused", "confused"])

    def test_unknown_preset_rejected(self):
        with pytest.raises(SweepError, match="unknown preset"):
            resolve_arms(sweep="health_literacy", anchor_preset="not_a_preset")

    def test_nonsense_arm_rejected(self):
        with pytest.raises(SweepError, match="neither a free-trait spec"):
            resolve_arms(arms=["confused", "gibberish"])

    def test_malformed_spec_rejected(self):
        with pytest.raises(Exception, match="unknown trait"):
            resolve_arms(arms=["confused", "pw:not_a_trait=low"])

    def test_hold_applies_across_sweep_arms(self):
        arms = resolve_arms(sweep="health_literacy", hold={"clarity": "high"})
        assert all("clarity=high" in a for a in arms)


# =============================================================================
# Building and verifying the benchmark file
# =============================================================================

class TestBuildSweep:
    def test_builds_one_entry_per_case_per_arm(self):
        entries, summary = build(SAMPLE, 3, FOUR_ARMS)
        assert summary == {"cases": 3, "arms": 4, "entries": 12}
        assert len(entries) == 12

    def test_arms_of_a_case_differ_only_in_personality(self):
        """The invariant the whole design rests on, and the one the engine's
        Layer-2 validator independently re-checks on the transcripts."""
        entries, _ = build(SAMPLE, 2, FOUR_ARMS)
        groups = {}
        for entry in entries:
            groups.setdefault(entry["scenario_id"].rsplit("--", 1)[0], []).append(entry)
        assert len(groups) == 2
        for group in groups.values():
            assert [e["personality"] for e in group] == FOUR_ARMS
            reference = group[0]
            for entry in group[1:]:
                differing = {
                    k for k in set(reference) | set(entry)
                    if reference.get(k) != entry.get(k)
                }
                assert differing == {"personality", "scenario_id"}

    def test_scenario_ids_are_unique_and_derived(self):
        entries, _ = build(SAMPLE, 3, FOUR_ARMS)
        ids = [e["scenario_id"] for e in entries]
        assert len(set(ids)) == len(ids)
        assert all("--arm" in i for i in ids)

    def test_case_content_is_carried_through_untouched(self):
        source = json.loads(Path(SAMPLE).read_text(encoding="utf-8"))[0]
        entries, _ = build(SAMPLE, 1, FOUR_ARMS)
        for entry in entries:
            assert entry["patient_story"] == source["patient_story"]
            assert entry["patient_profile"] == source["patient_profile"]
            assert entry["condition_name"] == source["condition_name"]

    def test_case_offset_selects_a_different_slice(self):
        first, _ = build(SAMPLE, 1, FOUR_ARMS)
        second, _ = build(SAMPLE, 1, FOUR_ARMS, case_offset=1)
        assert first[0]["patient_story"] != second[0]["patient_story"]

    def test_asking_for_more_cases_than_exist_is_refused(self):
        with pytest.raises(SweepError, match="cannot take"):
            build(SAMPLE, 10_000, FOUR_ARMS)

    def test_missing_source_file_is_refused(self, tmp_path):
        with pytest.raises(SweepError, match="cannot read"):
            build(str(tmp_path / "absent.json"), 1, FOUR_ARMS)

    def test_verify_catches_a_confounded_group(self):
        """The failure that would silently ruin a run: an attribute other than
        personality differing between two arms of the same case."""
        entries, _ = build(SAMPLE, 1, FOUR_ARMS)
        entries[1]["severity_level"] = "severe"
        with pytest.raises(SweepError, match="differs between arms"):
            verify(entries, FOUR_ARMS, 1)

    def test_verify_catches_a_missing_arm(self):
        entries, _ = build(SAMPLE, 2, FOUR_ARMS)
        with pytest.raises(SweepError, match="expected 8 entries"):
            verify(entries[:-1], FOUR_ARMS, 2)

    def test_case_without_an_id_is_refused(self):
        with pytest.raises(SweepError, match="no scenario_id"):
            build_entries([{"patient_story": "x"}], FOUR_ARMS)


class TestWrite:
    def test_written_file_loads_through_upstream(self, tmp_path):
        entries, _ = build(SAMPLE, 2, FOUR_ARMS)
        path = write(entries, tmp_path / "sweep.json")
        loaded = load_benchmark_entries(str(path))
        assert len(loaded) == 8
        assert [e.personality for e in loaded[:4]] == FOUR_ARMS

    def test_arm_labels_survive_the_round_trip(self, tmp_path):
        """The label is the arm. If upstream's reader dropped or mangled it,
        every conversation would run the same persona."""
        entries, _ = build(SAMPLE, 1, FOUR_ARMS)
        path = write(entries, tmp_path / "sweep.json")
        for entry, parsed in zip(entries, load_benchmark_entries(str(path)),
                                 strict=True):
            assert parsed.personality == entry["personality"]

    def test_arm_slug_is_id_safe(self):
        for index, arm in enumerate(FOUR_ARMS):
            slug = arm_slug(arm, index)
            assert slug.isalnum()


# =============================================================================
# Full-pipeline rehearsal
# =============================================================================

SANDBOX_JSON = {
    "offices": [{"id": "off_1", "name": "Office One", "address": "1 Main St",
                 "city": "Boston", "state": "MA", "zip_code": "02101"}],
    "doctors": [{"id": "doc_1", "name": "Dr. One", "specialty": "Internal Medicine",
                 "credentials": "MD", "office_id": "off_1"}],
    "pcp_id": "doc_1",
}
RUBRIC_JSON = json.dumps({"score": 3, "explanation": "stand-in verdict"})


# The pilot's own model keys. Using them here means the rehearsal exercises the
# real config path -- registry lookup, channel resolution, no Bedrock client --
# rather than a synthetic one that would hide a resolution failure until the
# first paid run.
PILOT_ASSISTANT = "openrouter:openai/gpt-5.4-mini"
PILOT_PATIENT = "openrouter:x-ai/grok-4.3"
PILOT_JURY = "claude-opus-4.8-api"


@pytest.fixture
def rehearsal_config():
    """The pilot's config, with every model call mocked."""
    return BenchConfig(
        assistant_agents=[AgentSpec(model=ModelConfig(model=PILOT_ASSISTANT),
                                    label="Rehearsal")],
        user_agents=[AgentSpec(model=ModelConfig(model=PILOT_PATIENT),
                               agent_class="pw_free_trait")],
        evaluator_models=[ModelConfig(model=PILOT_JURY)],
        sandbox_model=ModelConfig(model=PILOT_ASSISTANT),
        analyzer_model=ModelConfig(model=PILOT_ASSISTANT),
        max_turns=2,
    )


@pytest.fixture
def mocked_pipeline():
    """Patch exactly where upstream's own runner tests patch."""
    sandbox_llm = MagicMock()
    sandbox_llm.ainvoke = AsyncMock(
        return_value=MagicMock(content=json.dumps(SANDBOX_JSON))
    )

    assistant = MagicMock()

    def _invoke(messages, user_profile):  # noqa: ARG001
        ai = AIMessage(content="", tool_calls=[
            {"name": "list_doctors", "args": {}, "id": "c1", "type": "tool_call"},
        ])
        tool = ToolMessage(content="[]", tool_call_id="c1", name="list_doctors")
        return {"messages": list(messages) + [ai, tool,
                                              AIMessage(content="Here is my advice.")]}

    assistant.invoke.side_effect = _invoke

    rubric_llm = MagicMock()
    rubric_llm.invoke.return_value = MagicMock(content=RUBRIC_JSON)

    with patch("patient_agent_bench.runner.conversation_runner.create_sandbox_llm",
               return_value=sandbox_llm), \
         patch("patient_agent_bench.runner.conversation_runner."
               "create_assistant_agent_from_spec", return_value=assistant), \
         patch("patient_agent_bench.runner.conversation_runner."
               "create_bedrock_client_with_role", return_value=MagicMock()), \
         patch("patient_agent_bench.eval.base_rubric.create_chat_model",
               return_value=rubric_llm), \
         patch("patient_agent_bench.eval.base_rubric.create_bedrock_client_with_role",
               return_value=MagicMock()), \
         patch("patient_agent_bench.user_agent.default_agent.create_chat_model",
               return_value=MagicMock(**{
                   "invoke.return_value": MagicMock(content="I need some help."),
               })), \
         patch("patient_agent_bench.user_agent.default_agent."
               "create_bedrock_client_with_role", return_value=MagicMock()):
        yield


@pytest.fixture
def rehearsed_run(tmp_path, rehearsal_config, mocked_pipeline):
    """Build a 2-arm x 2-case sweep and run it through the real pipeline."""
    sweep_path = tmp_path / "sweep.json"
    arms = ["pw:base=confused;health_literacy=low",
            "pw:base=confused;health_literacy=high"]
    entries, _ = build(SAMPLE, 2, arms)
    write(entries, sweep_path)

    manager = OutputManager(input_file=str(sweep_path),
                            base_output_dir=str(tmp_path / "output"),
                            timestamp="rehearsal")
    manager.setup()
    manager.save_benchmark_cases(str(sweep_path))
    runner = ExperimentRunner(config=rehearsal_config, output_manager=manager)
    runner.run_all(load_benchmark_entries(str(sweep_path)))
    return manager.run_dir, arms


class TestPipelineRehearsal:
    def test_pilot_model_keys_resolve_without_a_bedrock_client(self,
                                                               rehearsal_config):
        """The registry lookup the live run depends on, exercised here rather
        than discovered at the first paid call."""
        assistant = rehearsal_config.assistant_agents[0].model
        patient = rehearsal_config.user_agents[0].model
        assert assistant.provider == "openai-protocol-api"
        assert assistant.requires_bedrock is False
        assert patient.requires_bedrock is False
        assert assistant.api_key_env == "OPENROUTER_API_KEY"

    def test_run_directory_has_the_expected_artifacts(self, rehearsed_run):
        run_dir, _arms = rehearsed_run
        assert (run_dir / "benchmark_cases.json").is_file()
        experiment = run_dir / "0_0"
        for name in ("conversations.json", "evaluations.json", "summary.json",
                     "experiment_config.json"):
            assert (experiment / name).is_file(), name

    def test_conversations_carry_the_arm_labels(self, rehearsed_run):
        run_dir, arms = rehearsed_run
        records = json.loads((run_dir / "0_0" / "conversations.json").read_text())
        assert len(records) == 4
        assert sorted({r["personality"] for r in records}) == sorted(arms)

    def test_arms_are_balanced_across_cases(self, rehearsed_run):
        """The pairing the engine's Layer-2 validator requires: every case
        present in every arm, joined on the scenario."""
        run_dir, arms = rehearsed_run
        records = json.loads((run_dir / "0_0" / "conversations.json").read_text())
        by_case = {}
        for record in records:
            by_case.setdefault(record["scenario"], []).append(record["personality"])
        assert len(by_case) == 2
        for labels in by_case.values():
            assert sorted(labels) == sorted(arms)

    def test_the_scenario_is_the_only_stable_join_key(self, rehearsed_run):
        """Found by this rehearsal on 2026-08-04, and the reason the engine's
        pair key is the scenario alone.

        initialize_sandbox() attaches a generated PCP to the patient profile
        before the transcript records it. The scenario survives byte-identical
        across arms; the profile does not, and its rendering varies between
        runs. A pairing keyed on the profile reports every case as absent from
        every arm -- a broken-design verdict on a perfectly good run.
        """
        run_dir, _arms = rehearsed_run
        records = json.loads((run_dir / "0_0" / "conversations.json").read_text())
        by_scenario = {}
        for record in records:
            by_scenario.setdefault(record["scenario"], []).append(record)
        assert len(by_scenario) == 2, "the scenario must group the arms of a case"
        for group in by_scenario.values():
            assert len(group) == 2
            # The profile is carried through the sandbox, so it is not asserted
            # stable -- that is precisely what makes it unusable as a join key.
            assert all(r["scenario"] == group[0]["scenario"] for r in group)

    def test_evaluations_join_slot_for_slot(self, rehearsed_run):
        run_dir, _arms = rehearsed_run
        convs = json.loads((run_dir / "0_0" / "conversations.json").read_text())
        evals = json.loads((run_dir / "0_0" / "evaluations.json").read_text())
        assert len(evals) == len(convs)
        for conv, ev in zip(convs, evals, strict=True):
            assert ev["case_id"] == conv["case_id"]

    def test_every_conversation_is_scored_on_the_six_rubrics(self, rehearsed_run):
        run_dir, _arms = rehearsed_run
        evals = json.loads((run_dir / "0_0" / "evaluations.json").read_text())
        for ev in evals:
            scores = ev["evaluation"]["rubric_scores"]
            assert "triage_quality" in scores
            assert len(scores) == 6
            assert all(1 <= v <= 5 for v in scores.values())

    def test_tool_calls_survive_into_the_transcript(self, rehearsed_run):
        """If they did not, workflow accuracy and triage would be scoring
        nothing -- the failure the live smoke test exists to catch."""
        run_dir, _arms = rehearsed_run
        convs = json.loads((run_dir / "0_0" / "conversations.json").read_text())
        for record in convs:
            types = [m.get("type") for m in record["conversation"]]
            assert "tool" in types

    def test_no_conversation_errored(self, rehearsed_run):
        run_dir, _arms = rehearsed_run
        convs = json.loads((run_dir / "0_0" / "conversations.json").read_text())
        assert [r for r in convs if r.get("error")] == []


# =============================================================================
# The committed pilot artifacts
# =============================================================================

PILOT_DIR = "pw_pilot"
PILOT_CONFIG = f"{PILOT_DIR}/config_pilot.json"
PILOT_SWEEPS = {
    f"{PILOT_DIR}/sweep_health_literacy_n8.json": (8, 4),
    f"{PILOT_DIR}/sweep_health_literacy_n13.json": (13, 4),
    f"{PILOT_DIR}/sweep_health_literacy_2arm_n13.json": (13, 2),
}


class TestCommittedPilotArtifacts:
    """The stimulus set and config that a live run would use, checked here so a
    rot in either surfaces offline rather than partway through a paid run."""

    @pytest.mark.parametrize("path,shape", sorted(PILOT_SWEEPS.items()))
    def test_sweep_loads_with_the_expected_shape(self, path, shape):
        cases, arms = shape
        entries = load_benchmark_entries(path)
        assert len(entries) == cases * arms
        assert len({e.personality for e in entries}) == arms

    @pytest.mark.parametrize("path,shape", sorted(PILOT_SWEEPS.items()))
    def test_sweep_arms_differ_only_in_personality(self, path, shape):
        _cases, arms = shape
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        groups = {}
        for entry in raw:
            groups.setdefault(entry["scenario_id"].rsplit("--", 1)[0], []).append(entry)
        assert all(len(g) == arms for g in groups.values())
        for group in groups.values():
            reference = group[0]
            for entry in group[1:]:
                differing = {k for k in set(reference) | set(entry)
                             if reference.get(k) != entry.get(k)}
                assert differing == {"personality", "scenario_id"}

    def test_pilot_config_resolves_every_role(self):
        config = BenchConfig.from_file(PILOT_CONFIG)
        assistant = config.assistant_agents[0].model
        patient = config.user_agents[0].model
        jury = config.evaluator_models[0]
        assert assistant.model_id == "openai/gpt-5.4-mini"
        assert patient.model_id == "x-ai/grok-4.3"
        assert jury.model_id == "claude-opus-4-8"
        assert config.sandbox_model.requires_bedrock is False

    def test_pilot_config_selects_the_free_trait_agent(self):
        """Without this the arms would all run upstream's preset path and the
        pw: labels would raise before the first call."""
        from patient_agent_bench.user_agent.registry import get_user_agent_class
        config = BenchConfig.from_file(PILOT_CONFIG)
        assert config.user_agents[0].agent_class == "pw_free_trait"
        assert get_user_agent_class("pw_free_trait").NAME == "pw_free_trait"

    def test_pilot_config_needs_no_aws_credentials(self):
        config = BenchConfig.from_file(PILOT_CONFIG)
        for model in (config.assistant_agents[0].model, config.user_agents[0].model,
                      config.sandbox_model, config.evaluator_models[0]):
            assert model.requires_bedrock is False

    def test_pilot_turns_match_the_costed_plan(self):
        assert BenchConfig.from_file(PILOT_CONFIG).max_turns == 3
