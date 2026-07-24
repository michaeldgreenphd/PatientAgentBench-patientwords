# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Helpers for normalizing LLM chat-model response content.

A chat model's ``AIMessage.content`` is polymorphic — it may be a plain string
or a list of content blocks — and the exact shape depends on the channel and on
whether the model emitted a reasoning/thinking block on a given turn:

- ``ChatAnthropic`` (adaptive thinking on): ``[{"type": "thinking", ...},
  {"type": "text", "text": ...}]``. LangChain collapses to a bare string only
  when the response is a lone non-citation text block, so this is *intermittent*.
- ``ChatBedrockConverse`` (extended thinking on, Anthropic models):
  ``[{"type": "reasoning_content", ...}, {"type": "text", "text": ...}]`` — also
  intermittent for the same single-block-collapse reason.
- ``ChatOpenAI`` Responses API (``use_responses_api=True``, e.g. Mantle GPT-5.x):
  *always* a list, so it is deterministic.

Any code that feeds ``response.content`` into a string operation (``.strip()``,
JSON parsing, a regex, etc.) must first normalize it, or it crashes with
"'list' object has no attribute 'strip'" the moment a reasoning block appears.
"""

from typing import Any


def flatten_response_content(content: Any) -> str:
    """Flatten an LLM response's ``content`` to plain answer text.

    Returns strings unchanged. For list-of-blocks content, concatenates the text
    of the answer blocks (``type`` in ``text``/``output_text``, or any block that
    carries a ``text`` key) and drops reasoning / thinking / tool blocks — the
    JSON-bearing answer lives only in the text blocks. Works across all three
    channels' block shapes (Anthropic ``thinking``, Bedrock ``reasoning_content``,
    OpenAI Responses ``reasoning``), which are all correctly skipped because they
    do not expose a top-level ``text`` key.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                # Text blocks carry the model's actual answer; skip reasoning /
                # thinking / tool blocks (downstream parsers only need the text).
                if block.get("type") in ("text", "output_text") or "text" in block:
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return str(content) if content is not None else ""
