# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for the OpenRouter spec injection and the registry surface it needs.

Nothing here makes a network call or reads key material: injection is a dict
write, and resolution is a dict read.

The drift pins in TestModelStorePin are the important part. Upstream's
``MODEL_STORE`` is a plain module-level dict that ``get_model_spec()`` reads at
call time, which is the only reason specs can be added without editing
``model_registry.py``. If upstream ever precomputes or freezes that lookup, the
injection would keep "succeeding" -- the dict write still works -- while every
config naming an injected key resolved to nothing. That is a silent failure, so
the round-trip is asserted rather than the write.
"""

import os

import pytest

import patientwords_pab.openrouter_specs as ors
from patient_agent_bench.config import ModelConfig
from patient_agent_bench.model_registry import (
    MODEL_STORE,
    ModelCapability,
    ModelSpec,
    get_model_pricing,
    get_model_spec,
    list_models,
)

PROBE_KEY = "openrouter:probe/drift-sentinel"


@pytest.fixture(autouse=True)
def _clean_store():
    """Save and restore the registry around each test."""
    saved = dict(MODEL_STORE)
    yield
    MODEL_STORE.clear()
    MODEL_STORE.update(saved)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Keep environment writes inside the test that made them."""
    monkeypatch.delenv(ors.API_KEY_ENV, raising=False)
    yield


# =============================================================================
# Drift pins on the upstream registry surface
# =============================================================================

class TestModelStorePin:
    def test_model_store_is_importable_and_populated(self):
        assert isinstance(MODEL_STORE, dict)
        assert MODEL_STORE, "upstream registry is empty; nothing to extend"

    def test_model_store_is_a_mutable_mapping(self):
        """A frozen or read-only mapping would make injection impossible."""
        MODEL_STORE[PROBE_KEY] = ors.build_spec(ors.OPENROUTER_MODELS[0])
        assert PROBE_KEY in MODEL_STORE
        del MODEL_STORE[PROBE_KEY]
        assert PROBE_KEY not in MODEL_STORE

    def test_get_model_spec_reads_the_store_at_call_time(self):
        """The dangerous one. A precomputed lookup would ignore injected specs
        without raising anything."""
        assert get_model_spec(PROBE_KEY) is None
        probe = ors.build_spec(ors.OPENROUTER_MODELS[0])
        MODEL_STORE[PROBE_KEY] = probe
        assert get_model_spec(PROBE_KEY) is probe
        del MODEL_STORE[PROBE_KEY]
        assert get_model_spec(PROBE_KEY) is None

    def test_get_model_pricing_reads_injected_specs(self):
        MODEL_STORE[PROBE_KEY] = ors.build_spec(ors.OPENROUTER_MODELS[0])
        pricing = get_model_pricing(PROBE_KEY)
        assert pricing == {
            "input_price_per_1m": ors.OPENROUTER_MODELS[0].input_price_per_1m,
            "output_price_per_1m": ors.OPENROUTER_MODELS[0].output_price_per_1m,
        }

    def test_list_models_sees_injected_specs(self):
        MODEL_STORE[PROBE_KEY] = ors.build_spec(ors.OPENROUTER_MODELS[0])
        assert PROBE_KEY in list_models()

    def test_model_spec_still_accepts_the_fields_the_injection_sets(self):
        fields = ModelSpec.__dataclass_fields__
        for name in ("model_id", "provider", "developer", "auth", "base_url_env",
                     "api_key_env", "use_responses_api", "notes",
                     "input_price_per_1m", "output_price_per_1m"):
            assert name in fields

    def test_unpriced_spec_renders_as_none(self):
        """Upstream's rule: None -> unpriced, cost renders as N/A, never guessed."""
        MODEL_STORE[PROBE_KEY] = ModelSpec(
            model_id="probe/unpriced", display_name="Unpriced",
            provider="openai-protocol-api",
        )
        assert get_model_pricing(PROBE_KEY) is None


# =============================================================================
# The injection itself
# =============================================================================

class TestInjection:
    def test_importing_the_package_registers_every_slug(self):
        for key in ors.openrouter_model_specs():
            assert get_model_spec(key) is not None, key

    def test_registration_is_idempotent(self):
        first = ors.register_openrouter_models()
        second = ors.register_openrouter_models()
        assert first == second

    def test_registration_does_not_disturb_upstream_entries(self):
        before = {k: v for k, v in MODEL_STORE.items()
                  if not k.startswith(ors.KEY_PREFIX)}
        ors.register_openrouter_models()
        after = {k: v for k, v in MODEL_STORE.items()
                 if not k.startswith(ors.KEY_PREFIX)}
        assert before == after

    def test_an_upstream_slug_of_the_same_name_wins(self):
        """If upstream starts shipping one of these keys, theirs must stand --
        being silently shadowed by ours is the worse failure."""
        key = ors.registry_key(ors.OPENROUTER_MODELS[0].slug)
        upstream_spec = ModelSpec(
            model_id="upstream/version", display_name="Upstream", provider="bedrock",
        )
        MODEL_STORE[key] = upstream_spec
        ors.register_openrouter_models()
        assert get_model_spec(key) is upstream_spec

    def test_keys_follow_the_engine_naming_convention(self):
        for key in ors.openrouter_model_specs():
            assert key.startswith("openrouter:")
            assert "/" in key[len("openrouter:"):]

    def test_vendor_pack_attribution(self):
        assert ors.vendor_pack("openrouter:qwen/qwen3-235b") == "qwen"
        assert ors.vendor_pack("openrouter:x-ai/grok-4.3") == "x-ai"
        assert ors.vendor_pack("qwen/qwen3-235b") == "qwen"

    def test_developer_matches_the_vendor_pack(self):
        for key, spec in ors.openrouter_model_specs().items():
            assert spec.developer == ors.vendor_pack(key)


class TestSpecShape:
    @pytest.mark.parametrize("model", ors.OPENROUTER_MODELS, ids=lambda m: m.slug)
    def test_channel_and_auth(self, model):
        spec = ors.build_spec(model)
        assert spec.provider == "openai-protocol-api"
        assert spec.auth == "api_key"
        assert spec.api_key_env == ors.API_KEY_ENV
        assert spec.base_url_env == ors.BASE_URL_ENV

    @pytest.mark.parametrize("model", ors.OPENROUTER_MODELS, ids=lambda m: m.slug)
    def test_model_id_is_the_bare_slug(self, model):
        """OpenRouter's API takes vendor/model; the prefix is our registry key,
        not part of the wire id."""
        spec = ors.build_spec(model)
        assert spec.model_id == model.slug
        assert not spec.model_id.startswith(ors.KEY_PREFIX)

    @pytest.mark.parametrize("model", ors.OPENROUTER_MODELS, ids=lambda m: m.slug)
    def test_responses_api_is_off(self, model):
        """OpenRouter speaks Chat Completions. Upstream's gpt-5.x-api specs set
        use_responses_api=True; copying it would send the wrong request shape."""
        assert ors.build_spec(model).use_responses_api is False

    @pytest.mark.parametrize("model", ors.OPENROUTER_MODELS, ids=lambda m: m.slug)
    def test_price_provenance_is_recorded(self, model):
        spec = ors.build_spec(model)
        if model.input_price_per_1m is None:
            assert "UNPRICED" in spec.notes
        else:
            assert model.price_source
            assert model.price_source in spec.notes

    def test_half_priced_spec_is_rejected(self):
        broken = ors.OPENROUTER_MODELS[0]._replace(output_price_per_1m=None)
        with pytest.raises(ors.OpenRouterSpecError, match="half-priced"):
            ors.build_spec(broken)

    def test_priced_without_a_source_is_rejected(self):
        broken = ors.OPENROUTER_MODELS[0]._replace(price_source="")
        with pytest.raises(ors.OpenRouterSpecError, match="guess"):
            ors.build_spec(broken)

    def test_tool_capability_is_declared(self):
        """The assistant role needs it; the smoke test exists because declaring
        it and delivering it are different things."""
        for spec in ors.openrouter_model_specs().values():
            assert ModelCapability.TOOL_USE in spec.capabilities


class TestResolutionThroughConfig:
    @pytest.mark.parametrize("model", ors.OPENROUTER_MODELS, ids=lambda m: m.slug)
    def test_model_config_resolves_an_injected_key(self, model):
        """The path a benchmark config actually takes:
        {"model": {"model": "<key>"}} -> ModelConfig -> create_chat_model."""
        cfg = ModelConfig(model=ors.registry_key(model.slug))
        assert cfg.model_id == model.slug
        assert cfg.provider == "openai-protocol-api"
        assert cfg.auth == "api_key"
        assert cfg.api_key_env == ors.API_KEY_ENV
        assert cfg.base_url_env == ors.BASE_URL_ENV
        assert cfg.requires_bedrock is False

    def test_unknown_openrouter_key_still_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            ModelConfig(model="openrouter:nobody/not-a-model")

    def test_round_trips_through_to_dict(self):
        key = ors.registry_key(ors.OPENROUTER_MODELS[0].slug)
        assert ModelConfig(model=key).to_dict()["model"] == key


class TestBaseUrlGuard:
    def test_ensure_base_url_sets_the_endpoint(self, monkeypatch):
        monkeypatch.delenv(ors.BASE_URL_ENV, raising=False)
        assert ors.ensure_base_url() == ors.BASE_URL
        assert os.environ[ors.BASE_URL_ENV] == ors.BASE_URL

    def test_ensure_base_url_never_overrides_the_operator(self, monkeypatch):
        monkeypatch.setenv(ors.BASE_URL_ENV, "https://proxy.example.invalid/v1")
        assert ors.ensure_base_url() == "https://proxy.example.invalid/v1"

    def test_base_url_env_is_set_so_chatopenai_cannot_fall_back(self):
        """Without base_url_env, ChatOpenAI defaults to api.openai.com -- an
        OpenRouter key fails there, and an ambient OPENAI_API_KEY would make the
        call succeed against the wrong vendor at the wrong price."""
        for spec in ors.openrouter_model_specs().values():
            assert spec.base_url_env


class TestSecrets:
    def test_api_key_present_reports_without_revealing(self, monkeypatch):
        assert ors.api_key_present() is False
        monkeypatch.setenv(ors.API_KEY_ENV, "sk-test-not-a-real-key")
        assert ors.api_key_present() is True

    def test_blank_key_counts_as_absent(self, monkeypatch):
        monkeypatch.setenv(ors.API_KEY_ENV, "   ")
        assert ors.api_key_present() is False

    def test_specs_carry_only_the_env_var_name(self, monkeypatch):
        """A spec is serialised into run configs and reports; it must name the
        variable, never hold the value."""
        monkeypatch.setenv(ors.API_KEY_ENV, "sk-test-not-a-real-key")
        blob = repr(sorted(ors.openrouter_model_specs().items()))
        assert "sk-test-not-a-real-key" not in blob
        assert ors.API_KEY_ENV in blob
