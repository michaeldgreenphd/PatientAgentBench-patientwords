# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Token-usage statistics for generated conversations.

Reads LangChain's provider-agnostic ``usage_metadata`` (populated by
ChatBedrockConverse, ChatOpenAI, and LiteLLM alike) off each assistant
message and rolls it up per conversation, then per experiment.

Metrics (per conversation):
- ``total_output_tokens`` — sum of ``output_tokens`` over AI messages. Real
  generation cost. (Input tokens are NOT summed: the full history is re-sent
  each turn, so summing input double-counts context.)
- ``peak_input_tokens`` — the last AI message's ``input_tokens``: the largest
  context the model was asked to process in the conversation.
- ``reasoning_tokens`` — sum of ``output_token_details.reasoning``. Populated
  only by models that emit reasoning/thinking tokens (e.g. GPT-5.x thinking,
  Claude extended-thinking); 0 otherwise. Read defensively so models that omit
  it simply contribute 0 rather than erroring.

Coverage: ``usage_coverage`` reports the fraction of AI messages that actually
carried ``usage_metadata``. A value < 1.0 flags a model/provider that failed
to report token counts, so the aggregates are known to be under-counts rather
than silently wrong.

No LLM calls, no side effects.
"""

import math
from typing import Any, Dict, List, Optional


def _percentile(sorted_vals: List[float], pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = pct / 100.0 * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def _summarize(values: List[float]) -> Dict[str, float]:
    """Summary stats over a list of per-conversation values.

    Token usage is right-skewed, so charts use the robust median + IQR
    (p25-p75) + p10/p90 to compare distributions. We also keep mean/std and a
    95% CI of the mean (normal approx, mean ± 1.96 * std / sqrt(n)) in the data
    for reference / significance checks.
    """
    if not values:
        return {"mean": 0.0, "median": 0.0, "total": 0.0, "max": 0.0, "min": 0.0,
                "std": 0.0, "ci": [0.0, 0.0],
                "p10": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}
    n = len(values)
    s = sorted(values)
    mean = sum(values) / n
    if n < 2:
        std = 0.0
        ci = [round(mean, 1), round(mean, 1)]
    else:
        std = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
        half = 1.96 * std / math.sqrt(n)
        ci = [round(mean - half, 1), round(mean + half, 1)]
    return {
        "mean": round(mean, 1),
        "median": round(_percentile(s, 50), 1),
        "total": round(sum(values), 1),
        "min": round(s[0], 1),
        "max": round(s[-1], 1),
        "std": round(std, 1),
        "ci": ci,
        "p10": round(_percentile(s, 10), 1),
        "p25": round(_percentile(s, 25), 1),
        "p75": round(_percentile(s, 75), 1),
        "p90": round(_percentile(s, 90), 1),
    }


def _conversation_token_usage(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Per-conversation token metrics from its AI messages.

    Returns None if the conversation has no AI messages at all. Otherwise
    returns a dict with the per-conversation metrics plus the AI-message
    usage-coverage counts (so the caller can aggregate coverage).
    """
    ai_messages = [m for m in messages if isinstance(m, dict) and m.get("type") == "ai"]
    if not ai_messages:
        return None

    total_output = 0
    reasoning = 0
    peak_input = 0
    with_usage = 0

    for m in ai_messages:
        um = m.get("usage_metadata")
        if not isinstance(um, dict):
            continue  # provider did not report usage for this message
        with_usage += 1
        total_output += um.get("output_tokens") or 0
        # peak input = the input_tokens of the last AI message that reported it;
        # since context grows monotonically, track the max defensively.
        peak_input = max(peak_input, um.get("input_tokens") or 0)
        details = um.get("output_token_details") or {}
        reasoning += details.get("reasoning") or 0

    return {
        "total_output_tokens": total_output,
        "peak_input_tokens": peak_input,
        "reasoning_tokens": reasoning,
        "ai_messages": len(ai_messages),
        "ai_messages_with_usage": with_usage,
    }


def compute_token_stats(conversations: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Aggregate token usage across an experiment's conversations.

    Args:
        conversations: list of conversation records (as stored in
            conversations.json). None/empty slots and errored conversations
            are skipped.

    Returns:
        Dict with per-metric summaries (mean/median/p90/total/max) for
        total_output_tokens, peak_input_tokens, reasoning_tokens; plus
        ``usage_coverage`` and counts. Returns a zeroed structure with
        ``conversations_measured: 0`` when there is nothing to measure.
    """
    per_conv_output: List[float] = []
    per_conv_peak: List[float] = []
    per_conv_reasoning: List[float] = []
    total_ai = 0
    total_ai_with_usage = 0
    measured = 0

    for conv in conversations:
        if not conv or not isinstance(conv, dict):
            continue
        # Skip conversations that ended in an agent error — their token counts
        # are partial/meaningless.
        if conv.get("error"):
            continue
        messages = conv.get("conversation") or []
        usage = _conversation_token_usage(messages)
        if usage is None:
            continue
        measured += 1
        per_conv_output.append(usage["total_output_tokens"])
        per_conv_peak.append(usage["peak_input_tokens"])
        per_conv_reasoning.append(usage["reasoning_tokens"])
        total_ai += usage["ai_messages"]
        total_ai_with_usage += usage["ai_messages_with_usage"]

    coverage = round(total_ai_with_usage / total_ai, 4) if total_ai else 0.0
    any_reasoning = any(r > 0 for r in per_conv_reasoning)

    return {
        "conversations_measured": measured,
        "usage_coverage": coverage,  # fraction of AI messages that reported usage
        "ai_messages_total": total_ai,
        "ai_messages_with_usage": total_ai_with_usage,
        "reasoning_tokens_reported": any_reasoning,
        "total_output_tokens": _summarize(per_conv_output),
        "peak_input_tokens": _summarize(per_conv_peak),
        "reasoning_tokens": _summarize(per_conv_reasoning),
    }


def compute_cost_block(
    token_stats: Dict[str, Any],
    input_price_per_1m: Optional[float],
    output_price_per_1m: Optional[float],
) -> Optional[Dict[str, Any]]:
    """APPROXIMATE per-(typical)-conversation USD cost, split input vs output.

    Uses the MEDIAN token values from ``token_stats``:
      input_cost  = median_peak_input_tokens * in_price  / 1e6
      output_cost = median_output_tokens     * out_price / 1e6
      total       = input_cost + output_cost

    IMPORTANT — this UNDER-counts real input cost. In a multi-turn agent
    conversation the full history is re-sent each turn, so true billed input is
    the sum over turns (grows ~quadratically), not the peak. We use median peak
    input as a cheap proxy (true billable-input tracking + prompt-cache
    discounts are out of scope), so treat this as a lower-bound / relative-cost
    indicator, not an invoice.

    Returns None when either price is missing (model unpriced) so callers render
    cost as N/A.
    """
    if input_price_per_1m is None or output_price_per_1m is None:
        return None
    out_med = token_stats.get("total_output_tokens", {}).get("median")
    peak_med = token_stats.get("peak_input_tokens", {}).get("median")
    if out_med is None or peak_med is None:
        return None
    input_cost = peak_med * input_price_per_1m / 1_000_000.0
    output_cost = out_med * output_price_per_1m / 1_000_000.0
    return {
        "input_cost_per_conv_usd": round(input_cost, 6),
        "output_cost_per_conv_usd": round(output_cost, 6),
        "approx_cost_per_conv_usd": round(input_cost + output_cost, 6),
        "input_price_per_1m": input_price_per_1m,
        "output_price_per_1m": output_price_per_1m,
        "cost_basis": "median output tokens + median peak-input tokens; "
                      "on-demand list price; input is a lower bound "
                      "(peak, not summed over turns).",
    }
