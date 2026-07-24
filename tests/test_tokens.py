# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for token-usage statistics (eval/tokens.py)."""

from patient_agent_bench.eval.tokens import compute_token_stats


def _ai(input_tokens, output_tokens, reasoning=None, with_usage=True):
    """Build a minimal AI message with (or without) usage_metadata."""
    msg = {"type": "ai", "content": "..."}
    if with_usage:
        um = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        if reasoning is not None:
            um["output_token_details"] = {"reasoning": reasoning}
        msg["usage_metadata"] = um
    return msg


def _human(text="hi"):
    return {"type": "human", "content": text}


def _conv(messages):
    return {"case_id": "c", "conversation": messages}


class TestComputeTokenStats:
    def test_basic_output_and_peak(self):
        # 2 AI turns: outputs 100 + 200; inputs grow 500 -> 800 (peak=800)
        conv = _conv([
            _human(), _ai(500, 100),
            _human(), _ai(800, 200),
        ])
        ts = compute_token_stats([conv])
        assert ts["conversations_measured"] == 1
        assert ts["total_output_tokens"]["total"] == 300  # 100 + 200
        assert ts["peak_input_tokens"]["max"] == 800       # last/largest input
        assert ts["usage_coverage"] == 1.0
        assert ts["reasoning_tokens_reported"] is False

    def test_reasoning_tokens_summed_when_present(self):
        conv = _conv([
            _human(), _ai(500, 100, reasoning=40),
            _human(), _ai(700, 150, reasoning=60),
        ])
        ts = compute_token_stats([conv])
        assert ts["reasoning_tokens"]["total"] == 100  # 40 + 60
        assert ts["reasoning_tokens_reported"] is True

    def test_missing_usage_metadata_lowers_coverage_not_crash(self):
        # One AI msg reports usage, one does not.
        conv = _conv([
            _human(), _ai(500, 100, with_usage=True),
            _human(), _ai(0, 0, with_usage=False),
        ])
        ts = compute_token_stats([conv])
        # 2 AI messages, 1 with usage -> coverage 0.5
        assert ts["ai_messages_total"] == 2
        assert ts["ai_messages_with_usage"] == 1
        assert ts["usage_coverage"] == 0.5
        # totals only reflect the message that reported usage
        assert ts["total_output_tokens"]["total"] == 100

    def test_errored_and_empty_conversations_skipped(self):
        good = _conv([_human(), _ai(500, 100)])
        errored = {"case_id": "e", "error": "timeout",
                   "conversation": [_human(), _ai(500, 100)]}
        empty = None
        no_ai = _conv([_human(), _human()])
        ts = compute_token_stats([good, errored, empty, no_ai])
        # only `good` is measured (errored skipped, None skipped, no-AI skipped)
        assert ts["conversations_measured"] == 1
        assert ts["total_output_tokens"]["total"] == 100

    def test_no_conversations_returns_zeroed(self):
        ts = compute_token_stats([])
        assert ts["conversations_measured"] == 0
        assert ts["usage_coverage"] == 0.0
        assert ts["total_output_tokens"]["total"] == 0.0

    def test_aggregation_across_conversations(self):
        # two convos, output totals 300 and 500 -> mean 400, max 500
        c1 = _conv([_human(), _ai(500, 300)])
        c2 = _conv([_human(), _ai(500, 500)])
        ts = compute_token_stats([c1, c2])
        assert ts["conversations_measured"] == 2
        assert ts["total_output_tokens"]["mean"] == 400.0
        assert ts["total_output_tokens"]["max"] == 500.0

    def test_distribution_percentiles(self):
        # 5 convos with outputs 100..500 -> median 300, min 100, max 500,
        # p25≈200, p75≈400 (linear-interpolated percentiles).
        convs = [_conv([_human(), _ai(500, v)]) for v in (100, 200, 300, 400, 500)]
        o = compute_token_stats(convs)["total_output_tokens"]
        assert o["median"] == 300.0
        assert o["min"] == 100.0
        assert o["max"] == 500.0
        assert o["p25"] == 200.0
        assert o["p75"] == 400.0
        # right-skew sanity: mean present alongside percentiles
        assert o["mean"] == 300.0


class TestCostBlock:
    def _ts(self, out_med, peak_med):
        return {
            "total_output_tokens": {"median": out_med},
            "peak_input_tokens": {"median": peak_med},
        }

    def test_cost_none_when_price_missing(self):
        from patient_agent_bench.eval.tokens import compute_cost_block
        ts = self._ts(2000, 10000)
        assert compute_cost_block(ts, None, 10.0) is None
        assert compute_cost_block(ts, 2.0, None) is None

    def test_cost_split_and_combined(self):
        from patient_agent_bench.eval.tokens import compute_cost_block
        # output 2000 @ $10/1M = 0.02 ; peak-input 10000 @ $2/1M = 0.02
        c = compute_cost_block(self._ts(2000, 10000), 2.0, 10.0)
        assert c["output_cost_per_conv_usd"] == 0.02
        assert c["input_cost_per_conv_usd"] == 0.02
        assert c["approx_cost_per_conv_usd"] == 0.04
        assert c["input_price_per_1m"] == 2.0
        assert c["output_price_per_1m"] == 10.0

    def test_cost_none_when_medians_missing(self):
        from patient_agent_bench.eval.tokens import compute_cost_block
        assert compute_cost_block({"total_output_tokens": {}}, 2.0, 10.0) is None


class TestModelPricing:
    def test_pricing_by_key_and_id(self):
        from patient_agent_bench.model_registry import get_model_pricing
        # registry key resolves
        assert get_model_pricing("claude-opus-4.8-bedrock") == {"input_price_per_1m": 5.0, "output_price_per_1m": 25.0}
        assert get_model_pricing("claude-sonnet-5-bedrock") == {"input_price_per_1m": 2.0, "output_price_per_1m": 10.0}
        # full model_id resolves
        assert get_model_pricing("global.anthropic.claude-opus-4-8") == {"input_price_per_1m": 5.0, "output_price_per_1m": 25.0}

    def test_unpriced_and_unknown_return_none(self):
        from patient_agent_bench.model_registry import get_model_pricing
        assert get_model_pricing("gpt-5.2-pro-api") is None   # in registry, no price
        assert get_model_pricing("no-such-model") is None  # unknown model
