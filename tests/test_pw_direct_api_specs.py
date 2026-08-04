# SPDX-License-Identifier: CC-BY-NC-4.0
"""Offline tests for the priced direct-API variants.

Upstream registers its ``*-api`` keys unpriced, which makes the jury leg
unmeterable and the Anthropic budget unenforceable. These tests pin the two
properties that make the fix safe: upstream's own entries are never modified,
and a priced variant is never invented for a key upstream does not have.
"""

import pytest

from patient_agent_bench.model_registry import MODEL_STORE, get_model_pricing
from patientwords_pab import direct_api_specs as das


class TestUpstreamIsUntouched:
    def test_upstream_keys_stay_unpriced(self):
        """The whole point of the pw: prefix. If this fails, a config naming
        upstream's key is silently getting our price instead of refusing."""
        for price in das.PRICED_DIRECT_API:
            assert get_model_pricing(price.upstream_key) is None, (
                f"{price.upstream_key} was priced in place; upstream's spec "
                "must be left as upstream wrote it"
            )

    def test_variant_is_a_distinct_object(self):
        """`dataclasses.replace`, not mutation: the spec object is shared with
        MODEL_STORE, so setting a field in place would price upstream's key as
        a side effect."""
        upstream = MODEL_STORE["claude-opus-4.8-api"]
        variant = MODEL_STORE["pw:claude-opus-4.8-api"]
        assert variant is not upstream
        assert upstream.input_price_per_1m is None
        assert variant.input_price_per_1m == 5.00

    def test_variant_keeps_the_upstream_access_channel(self):
        """A priced variant that routed differently would be a different model
        wearing the same rubric."""
        upstream = MODEL_STORE["claude-opus-4.8-api"]
        variant = MODEL_STORE["pw:claude-opus-4.8-api"]
        for field in ("model_id", "provider", "auth", "base_url_env", "api_key_env"):
            assert getattr(variant, field) == getattr(upstream, field), field


class TestPricing:
    def test_every_variant_is_priced_and_sourced(self):
        for price in das.PRICED_DIRECT_API:
            assert price.input_price_per_1m > 0
            assert price.output_price_per_1m > 0
            assert price.price_source, f"{price.upstream_key} priced without a source"

    def test_registered_variants_are_usable_by_the_guard(self):
        from patientwords_pab.budget_guard import SpendGuard

        keys = [das.registry_key(p.upstream_key) for p in das.PRICED_DIRECT_API]
        SpendGuard(ceiling_usd=1.0).require_priced(keys)

    def test_notes_record_the_billing_account(self):
        """OpenRouter and the direct Anthropic key are separate budgets; a
        transcript that does not say which one paid cannot be reconciled."""
        notes = MODEL_STORE["pw:claude-opus-4.8-api"].notes
        assert "direct Anthropic" in notes
        assert "NOT the OpenRouter balance" in notes


class TestRefusals:
    def test_unknown_upstream_key_is_refused_not_invented(self):
        bogus = das.DirectAPIPrice("no-such-model-api", 1.0, 2.0, "nowhere")
        with pytest.raises(das.DirectAPISpecError, match="not in upstream's registry"):
            das.build_spec(bogus)

    def test_registration_skips_a_key_upstream_dropped(self):
        saved = list(das.PRICED_DIRECT_API)
        try:
            das.PRICED_DIRECT_API.append(
                das.DirectAPIPrice("no-such-model-api", 1.0, 2.0, "nowhere")
            )
            injected = das.register_direct_api_models()
            assert "pw:no-such-model-api" not in injected
            assert "pw:no-such-model-api" not in MODEL_STORE
        finally:
            das.PRICED_DIRECT_API[:] = saved

    def test_registration_is_idempotent(self):
        first = das.register_direct_api_models()
        assert das.register_direct_api_models() == first


class TestKeys:
    def test_registry_key_round_trips(self):
        assert das.registry_key("claude-opus-4.8-api") == "pw:claude-opus-4.8-api"
        assert das.registry_key("pw:claude-opus-4.8-api") == "pw:claude-opus-4.8-api"
        assert das.upstream_key_for("pw:claude-opus-4.8-api") == "claude-opus-4.8-api"
        assert das.upstream_key_for("claude-opus-4.8-api") is None
