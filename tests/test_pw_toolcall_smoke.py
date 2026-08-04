# SPDX-License-Identifier: CC-BY-NC-4.0
"""Offline tests for the tool-calling smoke test.

Every model call is mocked. These tests cover the parts that decide whether a
live run is trustworthy: that a missing key skips instead of failing, that the
trace check actually rejects the failure modes it claims to, that the cost
arithmetic is right, and that nothing resembling key material can reach the
sidecar -- both repositories are public.
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from patient_agent_bench.runner.conversation import Conversation
from patientwords_pab import toolcall_smoke as smoke
from patientwords_pab.openrouter_specs import API_KEY_ENV, OPENROUTER_MODELS

PRICED = "openrouter:openai/gpt-5.4-mini"
FAKE_KEY = "sk-or-v1-TESTKEY-not-a-real-credential"
TOOLS = ["list_doctors", "get_available_appointments", "schedule_appointment"]


@pytest.fixture(autouse=True)
def _no_key(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    yield


def ai_with_calls(*calls):
    return AIMessage(content="", tool_calls=list(calls))


def call(name, cid, args=None):
    return {"name": name, "args": {} if args is None else args, "id": cid,
            "type": "tool_call"}


def conversation_with(*messages):
    return Conversation(list(messages))


# =============================================================================
# The trace check
# =============================================================================

class TestInspectToolTrace:
    def test_well_formed_trace_passes(self):
        conv = conversation_with(
            HumanMessage(content="opening"),
            ai_with_calls(call("list_doctors", "c1", {"specialty": "x"})),
            ToolMessage(content="[]", tool_call_id="c1", name="list_doctors"),
            AIMessage(content="here is what I found"),
        )
        trace = smoke.inspect_tool_trace(conv, TOOLS)
        assert trace.ok
        assert trace.problems == []
        assert trace.to_dict()["tools_called"] == ["list_doctors"]

    def test_no_tool_calls_fails(self):
        """The failure the paper describes: a model that narrates tool use in
        prose instead of emitting a call."""
        conv = conversation_with(
            HumanMessage(content="opening"),
            AIMessage(content="I will look up your doctors now."),
        )
        trace = smoke.inspect_tool_trace(conv, TOOLS)
        assert not trace.ok
        assert any("no tool calls" in p for p in trace.problems)

    def test_unregistered_tool_name_fails(self):
        conv = conversation_with(
            ai_with_calls(call("teleport_patient", "c1")),
            ToolMessage(content="?", tool_call_id="c1", name="teleport_patient"),
        )
        trace = smoke.inspect_tool_trace(conv, TOOLS)
        assert not trace.ok
        assert any("did not register" in p for p in trace.problems)

    def test_non_dict_arguments_fail(self):
        """Some providers return the arguments as a JSON string. AIMessage's
        validator rejects that at construction, so the failure shows up
        post-construction -- which is why upstream carries a sanitizer for it."""
        message = ai_with_calls(call("list_doctors", "c1", {"specialty": "x"}))
        message.tool_calls[0]["args"] = '{"specialty": "x"}'
        conv = conversation_with(
            message,
            ToolMessage(content="[]", tool_call_id="c1", name="list_doctors"),
        )
        trace = smoke.inspect_tool_trace(conv, TOOLS)
        assert not trace.ok
        assert any("not a dict" in p for p in trace.problems)

    def test_unanswered_call_fails(self):
        conv = conversation_with(ai_with_calls(call("list_doctors", "c1")))
        trace = smoke.inspect_tool_trace(conv, TOOLS)
        assert not trace.ok
        assert any("no tool result" in p for p in trace.problems)

    def test_multiple_calls_are_all_counted(self):
        conv = conversation_with(
            ai_with_calls(call("list_doctors", "c1"),
                          call("get_available_appointments", "c2")),
            ToolMessage(content="[]", tool_call_id="c1", name="list_doctors"),
            ToolMessage(content="[]", tool_call_id="c2",
                        name="get_available_appointments"),
        )
        trace = smoke.inspect_tool_trace(conv, TOOLS)
        assert trace.ok
        assert trace.to_dict()["n_calls"] == 2
        assert trace.to_dict()["n_results"] == 2


# =============================================================================
# Metering and cost
# =============================================================================

class TestLegUsage:
    def test_cost_from_registry_prices(self):
        leg = smoke.LegUsage(model=PRICED)
        leg.add({"input_tokens": 1_000_000, "output_tokens": 1_000_000})
        # gpt-5.4-mini: 0.80 in / 4.75 out per 1M
        assert leg.cost() == pytest.approx(5.55)
        assert leg.calls == 1

    def test_missing_usage_still_counts_the_call(self):
        leg = smoke.LegUsage(model=PRICED)
        leg.add(None)
        assert leg.calls == 1
        assert leg.input_tokens == 0
        assert leg.cost() == 0.0

    def test_unpriced_model_costs_none(self):
        leg = smoke.LegUsage(model="openrouter:nobody/unpriced")
        leg.add({"input_tokens": 10, "output_tokens": 10})
        assert leg.cost() is None
        assert leg.to_dict()["cost"] is None


class TestMeteredChatModel:
    def test_records_usage_and_returns_the_response(self):
        leg = smoke.LegUsage(model=PRICED)

        class Inner:
            model_name = "inner"

            def invoke(self, *_args, **_kwargs):
                msg = AIMessage(content="hi")
                msg.usage_metadata = {"input_tokens": 10, "output_tokens": 5,
                                      "total_tokens": 15}
                return msg

        metered = smoke.MeteredChatModel(Inner(), leg)
        out = metered.invoke([])
        assert out.content == "hi"
        assert leg.calls == 1
        assert (leg.input_tokens, leg.output_tokens) == (10, 5)

    def test_delegates_unknown_attributes(self):
        class Inner:
            model_name = "inner"

        assert smoke.MeteredChatModel(Inner(), smoke.LegUsage(PRICED)).model_name == "inner"


# =============================================================================
# Key handling and the live gate
# =============================================================================

class TestKeyGate:
    def test_run_smoke_refuses_without_a_key(self):
        with pytest.raises(RuntimeError, match=API_KEY_ENV):
            smoke.run_smoke()

    def test_cli_skips_without_a_key(self, capsys):
        assert smoke.main([]) == 0
        err = capsys.readouterr().err
        assert "skip" in err
        assert "nothing was spent" in err

    def test_dry_run_makes_no_call_and_reports_the_ceiling(self, capsys):
        assert smoke.main(["--dry-run"]) == 0
        plan = json.loads(capsys.readouterr().out)
        assert plan["api_key_present"] is False
        assert plan["max_spend_usd"] == smoke.DEFAULT_MAX_SPEND_USD
        assert plan["pricing"][smoke.DEFAULT_ASSISTANT]["input_price_per_1m"] > 0

    def test_dry_run_works_with_a_key_present_and_never_prints_it(
        self, capsys, monkeypatch
    ):
        monkeypatch.setenv(API_KEY_ENV, FAKE_KEY)
        assert smoke.main(["--dry-run"]) == 0
        out = capsys.readouterr().out
        assert json.loads(out)["api_key_present"] is True
        assert FAKE_KEY not in out


# =============================================================================
# The sidecar
# =============================================================================

def sample_record(**overrides):
    record = {
        "task": "pab-toolcall-smoke",
        "run_timestamp": "2026-08-04T12:00:00Z",
        "model": PRICED,
        "models": {"assistant": PRICED, "patient": PRICED, "sandbox": PRICED},
        "case_id": "case-1",
        "max_spend_usd": 0.10,
        "cost_usd": 0.0123,
        "usage": {
            "total_cost_usd": 0.0123,
            "per_model": {
                PRICED: {"calls": 3, "input_tokens": 100, "output_tokens": 50,
                         "cost": 0.0123},
            },
        },
        "tool_trace": {"n_calls": 1, "ok": True},
        "conversation": [{"type": "human", "content": f"key is {FAKE_KEY}"}],
    }
    record.update(overrides)
    return record


class TestSidecar:
    def test_shape_matches_the_engine_ledger_convention(self, tmp_path):
        """scripts/ledger_update.py folds on run_timestamp + cost_usd, and the
        established sidecar carries model, max_spend_usd and usage.per_model."""
        path = smoke.write_report(sample_record(), tmp_path / "smoke.report.json")
        data = json.loads(path.read_text())
        for key in ("run_timestamp", "model", "max_spend_usd", "cost_usd", "usage"):
            assert key in data
        per_model = data["usage"]["per_model"][PRICED]
        for key in ("calls", "input_tokens", "output_tokens", "cost"):
            assert key in per_model

    def test_task_is_not_the_tier_b_task(self):
        """ledger_update.attribute_tierb gates on task == 'pairs'; anything else
        is background spend and never touches Tier B rows."""
        assert sample_record()["task"] != "pairs"

    def test_transcript_is_dropped_by_default(self, tmp_path):
        path = smoke.write_report(sample_record(), tmp_path / "smoke.report.json")
        text = path.read_text()
        assert "conversation" not in json.loads(text)
        assert FAKE_KEY not in text

    def test_transcript_is_opt_in(self, tmp_path):
        path = smoke.write_report(sample_record(), tmp_path / "smoke.report.json",
                                  with_transcript=True)
        assert "conversation" in json.loads(path.read_text())

    def test_unpriced_run_writes_null_not_a_guess(self, tmp_path):
        record = sample_record(cost_usd=None, unpriced_legs=["assistant"])
        record["usage"]["total_cost_usd"] = None
        path = smoke.write_report(record, tmp_path / "smoke.report.json")
        data = json.loads(path.read_text())
        assert data["cost_usd"] is None
        assert data["unpriced_legs"] == ["assistant"]


# =============================================================================
# Assembly, fully mocked
# =============================================================================

class _FakeTool:
    def __init__(self, name):
        self.name = name


class _FakeRegistry:
    def get_all(self):
        return [_FakeTool(n) for n in TOOLS]


class _FakeAssistant:
    """Emits one well-formed tool call, then an answer."""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages, user_profile):  # noqa: ARG002
        self.calls += 1
        ai = ai_with_calls(call("list_doctors", f"c{self.calls}", {"specialty": "x"}))
        ai.usage_metadata = {"input_tokens": 2000, "output_tokens": 40,
                             "total_tokens": 2040}
        tool = ToolMessage(content="[]", tool_call_id=f"c{self.calls}",
                           name="list_doctors")
        final = AIMessage(content="Here are your options.")
        final.usage_metadata = {"input_tokens": 2100, "output_tokens": 120,
                                "total_tokens": 2220}
        return {"messages": list(messages) + [ai, tool, final]}


class _FakePatient:
    def __init__(self):
        self.llm = object()

    def start_conversation(self):
        return "I need help with something."

    def respond(self, _assistant_text):
        return "Okay, thanks."


class _FakeEntry:
    id = "case-alpha"
    scenario = "case background stand-in"
    patient_profile = object()
    patient_profile_xml = "<profile/>"


@pytest.fixture
def unpriced_key(monkeypatch):
    """A registered spec with no verified price, per upstream's None -> N/A rule."""
    from patient_agent_bench.model_registry import MODEL_STORE, ModelSpec

    key = "openrouter:probe/unpriced"
    saved = dict(MODEL_STORE)
    MODEL_STORE[key] = ModelSpec(
        model_id="probe/unpriced", display_name="Unpriced probe",
        provider="openai-protocol-api", auth="api_key",
        base_url_env="OPENROUTER_BASE_URL", api_key_env=API_KEY_ENV,
        default_max_tokens=4096,
    )
    yield key
    MODEL_STORE.clear()
    MODEL_STORE.update(saved)


@pytest.fixture
def wired(monkeypatch):
    """Patch every upstream factory the smoke test assembles."""
    async def _init_sandbox(**_kwargs):
        return None

    monkeypatch.setattr(smoke, "load_benchmark_entries", lambda _p: [_FakeEntry()])
    monkeypatch.setattr(smoke, "HealthcareSandbox", lambda: object())
    monkeypatch.setattr(smoke, "create_sandbox_llm", lambda _c, _b: object())
    monkeypatch.setattr(smoke, "initialize_sandbox", _init_sandbox)
    monkeypatch.setattr(smoke, "create_tool_registry", lambda _s: _FakeRegistry())
    monkeypatch.setattr(smoke, "create_assistant_agent_from_spec",
                        lambda **_kw: _FakeAssistant())
    monkeypatch.setattr(smoke, "create_user_agent_from_spec",
                        lambda **_kw: _FakePatient())
    monkeypatch.setenv(API_KEY_ENV, FAKE_KEY)
    yield


class TestAssembly:
    def test_smoke_run_passes_and_prices_every_leg(self, wired):
        record = smoke.run_smoke(assistant_key=PRICED, patient_key=PRICED,
                                 sandbox_key=PRICED, turns=1)
        assert record["passed"] is True
        assert record["tool_trace"]["ok"] is True
        assert record["tool_trace"]["tools_called"] == ["list_doctors"]
        assert record["cost_usd"] > 0
        assert record["unpriced_legs"] == []
        assert record["task"] == "pab-toolcall-smoke"

    def test_assistant_usage_is_summed_across_intermediate_messages(self, wired):
        record = smoke.run_smoke(assistant_key=PRICED, patient_key=PRICED,
                                 sandbox_key=PRICED, turns=1)
        assistant = record["usage"]["per_role"]["assistant"]
        # 2000 + 2100 input across the tool-call message and the final answer.
        assert assistant["input_tokens"] == 4100
        assert assistant["output_tokens"] == 160
        # All three roles share one model here, so per_model must sum them
        # rather than keep whichever was written last.
        assert record["usage"]["per_model"][PRICED]["input_tokens"] == 4100
        assert record["usage"]["per_model"][PRICED]["calls"] == (
            assistant["calls"]
            + record["usage"]["per_role"]["patient"]["calls"]
            + record["usage"]["per_role"]["sandbox"]["calls"]
        )

    def test_unpriced_assistant_reports_na_not_a_number(self, wired, unpriced_key):
        record = smoke.run_smoke(assistant_key=unpriced_key,
                                 patient_key=PRICED, sandbox_key=PRICED, turns=1)
        assert "assistant" in record["unpriced_legs"]
        assert record["usage"]["per_model"][unpriced_key]["cost"] is None
        # A partially priced run still reports the legs it can price, and the
        # total stays a real number for those -- never a filled-in guess.
        assert record["usage"]["per_role"]["patient"]["cost"] is not None

    def test_unknown_model_key_is_rejected_before_any_call(self, wired):
        with pytest.raises(ValueError, match="Unknown model"):
            smoke.run_smoke(assistant_key="openrouter:nobody/not-registered",
                            patient_key=PRICED, sandbox_key=PRICED, turns=1)

    def test_assistant_exception_is_reported_not_masked(self, wired, monkeypatch):
        class _Broken:
            def invoke(self, messages, user_profile):  # noqa: ARG002
                raise RuntimeError("400 tool_choice unsupported")

        monkeypatch.setattr(smoke, "create_assistant_agent_from_spec",
                            lambda **_kw: _Broken())
        record = smoke.run_smoke(assistant_key=PRICED, patient_key=PRICED,
                                 sandbox_key=PRICED, turns=1)
        assert record["passed"] is False
        assert "tool_choice unsupported" in record["error"]

    def test_record_never_contains_key_material(self, wired):
        record = smoke.run_smoke(assistant_key=PRICED, patient_key=PRICED,
                                 sandbox_key=PRICED, turns=1)
        assert FAKE_KEY not in json.dumps(record)


class TestDefaults:
    def test_default_models_are_registered_openrouter_slugs(self):
        keys = {f"openrouter:{m.slug}" for m in OPENROUTER_MODELS}
        for key in (smoke.DEFAULT_ASSISTANT, smoke.DEFAULT_PATIENT,
                    smoke.DEFAULT_SANDBOX):
            assert key in keys

    def test_default_ceiling_is_small(self):
        """A smoke test that can spend real money by accident is not a smoke
        test."""
        assert 0 < smoke.DEFAULT_MAX_SPEND_USD <= 0.25
