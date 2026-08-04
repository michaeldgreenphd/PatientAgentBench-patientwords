# SPDX-License-Identifier: CC-BY-NC-4.0
"""Offline tests for the live spend ceiling.

The guard's whole value is that it bounds an unattended paid run, so these tests
target the ways a budget guard fails *silently* -- metering a fraction of the
calls, losing the assistant leg behind ``bind_tools``, being swallowed by an
``except Exception`` retry, or leaving no record when the run dies. A guard that
under-reports is worse than none: it converts an unbounded run into an unbounded
run that looks bounded.

No network, no keys, no model calls.
"""

import json
import sys

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from patient_agent_bench import config as pab_config
from patient_agent_bench.model_registry import MODEL_STORE, ModelSpec
from patientwords_pab import budget_guard as bg
from patientwords_pab import run as pw_run
from patientwords_pab.openrouter_specs import API_KEY_ENV

PRICED = "openrouter:openai/gpt-5.4-mini"


@pytest.fixture(autouse=True)
def _uninstall():
    """No test may leak a patched factory into the next one."""
    yield
    bg.uninstall()


@pytest.fixture
def unpriced_key():
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


def fake_model(n_responses=50, in_tokens=1_000_000, out_tokens=0):
    """A chat model whose replies each report a fixed, large token usage."""
    responses = []
    for _ in range(n_responses):
        msg = AIMessage(content="ok")
        msg.usage_metadata = {"input_tokens": in_tokens,
                              "output_tokens": out_tokens,
                              "total_tokens": in_tokens + out_tokens}
        responses.append(msg)
    return FakeMessagesListChatModel(responses=responses)


# =============================================================================
# Accounting
# =============================================================================

class TestSpendGuard:
    def test_ceiling_must_be_positive(self):
        with pytest.raises(bg.BudgetGuardError, match="unbounded"):
            bg.SpendGuard(ceiling_usd=0)

    def test_cost_uses_registry_prices(self):
        guard = bg.SpendGuard(ceiling_usd=100)
        # 1M input tokens at $0.80/1M (the registered ceiling-side price).
        guard.record(PRICED, {"input_tokens": 1_000_000, "output_tokens": 0})
        assert guard.total() == pytest.approx(0.80)

    def test_calls_are_summed_across_roles_sharing_a_model(self):
        guard = bg.SpendGuard(ceiling_usd=100)
        for _ in range(3):
            guard.record(PRICED, {"input_tokens": 1_000_000, "output_tokens": 0})
        assert guard.total() == pytest.approx(2.40)
        assert guard.per_model[PRICED].calls == 3

    def test_check_trips_at_the_ceiling(self):
        guard = bg.SpendGuard(ceiling_usd=1.0)
        guard.record(PRICED, {"input_tokens": 1_000_000, "output_tokens": 0})
        guard.check()  # $0.80 < $1.00
        guard.record(PRICED, {"input_tokens": 500_000, "output_tokens": 0})
        with pytest.raises(bg.BudgetExceeded, match="ceiling reached"):
            guard.check()
        assert guard.tripped

    def test_budget_exceeded_survives_a_bare_except_exception(self):
        """Upstream retries anything deriving from Exception, with backoff.

        A ceiling that retried would spend more on the way out, so the abort is
        a BaseException. This is the property, not an implementation detail.
        """
        assert not issubclass(bg.BudgetExceeded, Exception)
        guard = bg.SpendGuard(ceiling_usd=0.01)
        guard.record(PRICED, {"input_tokens": 1_000_000, "output_tokens": 0})
        with pytest.raises(bg.BudgetExceeded):
            try:
                guard.check()
            except Exception:  # noqa: BLE001 - deliberately the wrong handler
                pytest.fail("an except Exception handler swallowed the abort")

    def test_missing_usage_metadata_still_counts_the_call(self):
        """A response with no usage is unmeasured, not free -- the call count
        keeps it visible in the report instead of vanishing."""
        guard = bg.SpendGuard(ceiling_usd=1.0)
        guard.record(PRICED, None)
        assert guard.per_model[PRICED].calls == 1
        assert guard.total() == 0.0


class TestPreflight:
    def test_unpriced_model_refuses_to_run(self, unpriced_key):
        guard = bg.SpendGuard(ceiling_usd=1.0)
        with pytest.raises(bg.BudgetGuardError, match="no registry price"):
            guard.require_priced([PRICED, unpriced_key])

    def test_allow_unpriced_records_instead_of_refusing(self, unpriced_key):
        guard = bg.SpendGuard(ceiling_usd=1.0, allow_unpriced=True)
        guard.require_priced([PRICED, unpriced_key])
        assert guard.unpriced == [unpriced_key]
        assert guard.report()["unpriced_models"] == [unpriced_key]

    def test_fully_priced_config_passes(self):
        bg.SpendGuard(ceiling_usd=1.0).require_priced([PRICED])


# =============================================================================
# Installation: the part that decides whether the ceiling is real
# =============================================================================

#: Every upstream module that binds the factory into its own namespace, from
#: `grep -rn "from patient_agent_bench.config import .*create_chat_model"`.
#: Five of these are the ones conftest.py session-mocks; seed_runner is not,
#: which is why this list is derived from the source and not from conftest.
CALL_SITES = [
    "patient_agent_bench.assistant_agent.default_agent",
    "patient_agent_bench.user_agent.default_agent",
    "patient_agent_bench.eval.base_rubric",
    "patient_agent_bench.sandbox.generator",
    "patient_agent_bench.analyzer.generator",
    "patient_agent_bench.benchmark_seed.seed_runner",
]

# Run in a clean interpreter: this test session's conftest replaces
# create_chat_model with a MagicMock in five of those modules, so an in-process
# check would compare against the mock and prove nothing about the real thing.
_COVERAGE_PROBE = """
import importlib, json, sys
sites = {sites!r}
for name in sites:
    importlib.import_module(name)
from patient_agent_bench import config
from patientwords_pab import budget_guard as bg

original = config.create_chat_model
patched = bg.install(bg.SpendGuard(ceiling_usd=1.0))
missed = [n for n in sites if sys.modules[n].create_chat_model is original]
still_original = config.create_chat_model is original

bg.uninstall()
restored = [n for n in sites if sys.modules[n].create_chat_model is original]

print(json.dumps({{"patched": patched, "missed": missed,
                   "config_still_original": still_original,
                   "restored": restored}}))
"""


def _run_coverage_probe(tmp_path):
    import subprocess

    script = tmp_path / "probe.py"
    script.write_text(_COVERAGE_PROBE.format(sites=CALL_SITES))
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestInstall:
    def test_patches_every_module_that_bound_the_factory(self, tmp_path):
        """Callers do `from ... import create_chat_model`, which copies the
        function into their own namespace. Patching only `config` would leave
        those pointing at the original and meter a fraction of the run, while
        the report still looked complete."""
        out = _run_coverage_probe(tmp_path)
        assert out["missed"] == [], (
            f"unmetered call sites remain: {out['missed']}"
        )
        assert not out["config_still_original"]
        for name in CALL_SITES:
            assert name in out["patched"]
        assert "patient_agent_bench.config" in out["patched"]

    def test_uninstall_restores_every_site(self, tmp_path):
        out = _run_coverage_probe(tmp_path)
        assert sorted(out["restored"]) == sorted(CALL_SITES)

    def test_call_sites_list_matches_the_source(self):
        """If upstream adds a caller, this fails here rather than by
        under-metering a paid run."""
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "src" / "patient_agent_bench"
        found = set()
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(
                r"from patient_agent_bench\.config import \(?([^)\n]*(?:\n[^)]*)?)\)?",
                text,
            ):
                if "create_chat_model" in match.group(1):
                    rel = path.relative_to(root.parent).with_suffix("")
                    found.add(str(rel).replace("/", "."))
        assert found == set(CALL_SITES), (
            f"call sites drifted: only in source {found - set(CALL_SITES)}, "
            f"only in list {set(CALL_SITES) - found}"
        )

    def test_double_install_is_refused(self):
        bg.install(bg.SpendGuard(ceiling_usd=1.0))
        with pytest.raises(bg.BudgetGuardError, match="already installed"):
            bg.install(bg.SpendGuard(ceiling_usd=1.0))

    def test_the_factory_returns_a_metered_model(self, monkeypatch):
        guard = bg.SpendGuard(ceiling_usd=100.0)
        monkeypatch.setattr(pab_config, "create_chat_model",
                            lambda cfg, client=None, role_arn=None: fake_model())
        bg.install(guard)
        model = pab_config.create_chat_model(pab_config.ModelConfig(model=PRICED))
        model.invoke("hello")
        assert guard.per_model[PRICED].input_tokens == 1_000_000
        assert guard.total() == pytest.approx(0.80)


# =============================================================================
# The wrapper
# =============================================================================

class TestGuardedChatModel:
    def test_bind_tools_returns_a_guarded_wrapper(self):
        """The hole that matters. The assistant is the only leg that binds
        tools and the most expensive one; returning the bare inner runnable
        would let its calls escape while the report still looked complete.

        Uses a real ChatOpenAI -- constructed, never called -- because no fake
        chat model in langchain_core implements bind_tools, and a fake that did
        would be pinning this test to itself rather than to langchain.
        """
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        @tool
        def probe(x: str) -> str:
            """A tool that is never called."""
            return x

        inner = ChatOpenAI(model="probe", api_key="not-a-real-credential",
                           base_url="http://localhost:1")
        guard = bg.SpendGuard(ceiling_usd=100.0)
        bound = bg.GuardedChatModel(inner, PRICED, guard).bind_tools([probe])
        assert isinstance(bound, bg.GuardedChatModel)

    def test_bind_returns_a_guarded_wrapper(self):
        guard = bg.SpendGuard(ceiling_usd=100.0)
        model = bg.GuardedChatModel(fake_model(), PRICED, guard)
        assert isinstance(model.bind(stop=None), bg.GuardedChatModel)

    def test_invoke_trips_the_ceiling(self):
        guard = bg.SpendGuard(ceiling_usd=1.0)
        model = bg.GuardedChatModel(fake_model(), PRICED, guard)
        model.invoke("first")           # $0.80
        with pytest.raises(bg.BudgetExceeded):
            model.invoke("second")      # $1.60 -- over

    def test_a_tripped_guard_refuses_the_next_call_before_spending(self):
        """Checked before as well as after: once tripped, no further call may
        be made, however many workers are still in flight."""
        guard = bg.SpendGuard(ceiling_usd=0.5)
        model = bg.GuardedChatModel(fake_model(), PRICED, guard)
        with pytest.raises(bg.BudgetExceeded):
            model.invoke("first")
        calls_after_trip = guard.per_model[PRICED].calls
        with pytest.raises(bg.BudgetExceeded):
            model.invoke("second")
        assert guard.per_model[PRICED].calls == calls_after_trip, (
            "a second call was made after the ceiling tripped"
        )

    def test_create_agent_accepts_the_wrapper(self):
        """Pins the duck-typing assumption: the wrapper is not a BaseChatModel,
        and langgraph's create_agent must still take it."""
        from patient_agent_bench.assistant_agent.default_agent import create_agent

        guard = bg.SpendGuard(ceiling_usd=100.0)
        model = bg.GuardedChatModel(fake_model(), PRICED, guard)
        graph = create_agent(model=model, tools=[], system_prompt="x")
        result = graph.invoke({"messages": [("user", "hello")]})
        assert any(isinstance(m, AIMessage) for m in result["messages"])
        assert guard.per_model[PRICED].calls >= 1

    def test_unknown_attributes_delegate(self):
        guard = bg.SpendGuard(ceiling_usd=100.0)
        inner = fake_model()
        assert bg.GuardedChatModel(inner, PRICED, guard).responses is inner.responses


# =============================================================================
# Reporting -- an unrecorded spend is an unbounded one
# =============================================================================

class TestReport:
    def test_report_written_on_abort(self, tmp_path):
        guard = bg.SpendGuard(ceiling_usd=1.0)
        guard.record(PRICED, {"input_tokens": 2_000_000, "output_tokens": 0})
        try:
            guard.check()
        except bg.BudgetExceeded:
            pass
        path = guard.write_report(tmp_path / "run.report.json", outcome="ceiling_tripped")
        data = json.loads(path.read_text())
        assert data["cost_usd"] == pytest.approx(1.60)
        assert data["ceiling_tripped"] is True
        assert data["outcome"] == "ceiling_tripped"
        assert data["max_spend_usd"] == 1.0
        assert "not the provider's invoice" in data["cost_basis"]

    def test_report_names_the_ledger_fields_the_engine_reads(self, tmp_path):
        """ledger_update.py folds data/pab/*.report.json on cost_usd; a sidecar
        without it books nothing and the spend goes unseen."""
        guard = bg.SpendGuard(ceiling_usd=1.0)
        guard.record(PRICED, {"input_tokens": 1_000, "output_tokens": 100})
        data = json.loads(guard.write_report(tmp_path / "r.json").read_text())
        assert data["cost_usd"] is not None
        assert data["max_spend_usd"] == 1.0
        assert data["task"] == "pab-run"
        assert data["run_timestamp"].endswith("Z")

    def test_no_key_material_reaches_the_report(self, tmp_path, monkeypatch):
        """Both repositories are public."""
        monkeypatch.setenv(API_KEY_ENV, "sk-or-v1-TESTKEY-not-a-real-credential")
        guard = bg.SpendGuard(ceiling_usd=1.0)
        guard.record(PRICED, {"input_tokens": 10, "output_tokens": 1})
        text = guard.write_report(tmp_path / "r.json").read_text()
        assert "sk-or-v1" not in text
        assert "TESTKEY" not in text


# =============================================================================
# Environment wiring
# =============================================================================

class TestEnv:
    def test_absent_ceiling_means_no_guard(self, monkeypatch):
        monkeypatch.delenv(bg.MAX_SPEND_ENV, raising=False)
        assert bg.guard_from_env() is None

    def test_malformed_ceiling_is_an_error_not_a_silent_skip(self, monkeypatch):
        """`PW_MAX_SPEND_USD=abc` means someone intended a ceiling. Running
        unbounded is the one outcome they did not intend."""
        monkeypatch.setenv(bg.MAX_SPEND_ENV, "abc")
        with pytest.raises(bg.BudgetGuardError, match="refusing to run unbounded"):
            bg.guard_from_env()

    def test_ceiling_and_allow_unpriced_are_read(self, monkeypatch):
        monkeypatch.setenv(bg.MAX_SPEND_ENV, "2.5")
        monkeypatch.setenv(bg.ALLOW_UNPRICED_ENV, "1")
        guard = bg.guard_from_env()
        assert guard.ceiling_usd == 2.5
        assert guard.allow_unpriced is True


class TestRunWrapper:
    def test_config_path_parsed_in_both_forms(self):
        assert pw_run.config_path(["generate", "--config", "c.json"]) == "c.json"
        assert pw_run.config_path(["generate", "--config=c.json"]) == "c.json"
        assert pw_run.config_path(["generate"]) is None

    def test_configured_models_covers_every_role(self, tmp_path):
        """A role missed here is a leg the price check never sees, and the
        first evidence of that is an unmetered bill."""
        # Seven distinct registered keys, one per role, so a role the function
        # forgets shows up as a missing key rather than being masked by another
        # role that happens to use the same model.
        roles = {
            "assistant_agent": ["openrouter:openai/gpt-5.4-mini",
                                "openrouter:openai/gpt-5.5"],
            "user_agent": "openrouter:x-ai/grok-4.3",
            "evaluator_model": "claude-opus-4.8-api",
            "sandbox_model": "openrouter:moonshotai/kimi-k2.5",
            "seed_generator_model": "openrouter:deepseek/deepseek-v4-flash",
            "analyzer_model": "openrouter:google/gemini-3.5-flash",
        }
        config = {
            "assistant_agent": [{"model": {"model": m}}
                                for m in roles["assistant_agent"]],
            "user_agent": {"model": {"model": roles["user_agent"]}},
            "evaluator_model": [{"model": roles["evaluator_model"]}],
            "sandbox_model": {"model": roles["sandbox_model"]},
            "seed_generator_model": {"model": roles["seed_generator_model"]},
            "analyzer_model": {"model": roles["analyzer_model"]},
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        expected = sorted(set(roles["assistant_agent"])
                          | {v for k, v in roles.items() if k != "assistant_agent"})
        assert pw_run.configured_models(str(path)) == expected
