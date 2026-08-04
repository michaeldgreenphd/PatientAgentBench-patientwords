# SPDX-License-Identifier: CC-BY-NC-4.0
"""OpenRouter model specs, injected into upstream's registry without editing it.

Upstream's ``model_registry.py`` already supports a second access channel:
``provider="openai-protocol-api"`` with ``auth="api_key"``, ``base_url_env`` and
``api_key_env``, documented as the path for a non-default OpenAI-compatible
endpoint. OpenRouter is exactly that shape, so no upstream change is needed --
only entries. ``MODEL_STORE`` is a plain module-level dict and
``get_model_spec()`` reads it at call time, so the specs are injected at import
the same way ``register_user_agent()`` adds an agent.

**Registry keys are the engine's names.** They are ``openrouter:<vendor>/<model>``
rather than upstream's ``<family>-<version>-api`` convention, deliberately: the
measurement engine already names OpenRouter models that way, groups vendor packs
by the prefix (``openrouter:qwen/...`` is qwen's), and writes that string into
cost sidecars. One name across both repos means the benchmark config, the
transcript, and the ledger all say the same thing. Upstream's convention is a
docstring guideline over a plain dict, not a constraint.

**Prices are not guessed.** Every price below is transcribed from an
owner-verified in-repo source, named per spec in ``price_source``, and lands in
``ModelSpec.notes`` so provenance travels with the spec. A slug whose price
cannot be verified is registered with ``None``, and upstream's own rule then
applies: *None → unpriced; cost renders as N/A (never guessed)*.

**Two traps this module exists to avoid.**

1. ``base_url_env`` is required. Without it ``ChatOpenAI`` silently defaults to
   ``api.openai.com`` -- an OpenRouter key would fail there, and worse, an
   ambient ``OPENAI_API_KEY`` could make the call succeed against the wrong
   vendor at the wrong price. :func:`ensure_base_url` sets the variable with
   ``setdefault`` so an operator-provided value always wins.
2. ``use_responses_api`` must stay False. Upstream's ``gpt-5.x-api`` specs set it
   True for OpenAI's Responses API; OpenRouter speaks Chat Completions, so
   copying that flag would send the wrong request shape.

Nothing here reads, prints, or stores key material: the specs name the *env var*,
and the value is read by ``ChatOpenAI`` at call time.
"""

from __future__ import annotations

import os
from typing import Dict, List, NamedTuple, Optional

from patient_agent_bench.model_registry import (
    MODEL_STORE,
    ModelCapability,
    ModelSpec,
)

# Env var naming the OpenRouter API key. Matches the engine's established name
# (docs/advice_arm_handoff.md, docs/preregistration_advice.md): one prepaid key
# whose balance is a hard external ceiling above any script-level ceiling.
API_KEY_ENV = "OPENROUTER_API_KEY"

# Env var naming the endpoint. Upstream reads the URL from an env var rather
# than the spec, so the name is the spec's and the value is the operator's.
BASE_URL_ENV = "OPENROUTER_BASE_URL"

# The endpoint itself, as recorded in the engine's provider registry
# (data/advice_providers.json, every OpenRouter-routed entry).
BASE_URL = "https://openrouter.ai/api/v1"

# Prefix that makes a registry key an OpenRouter key, and the vendor attribution
# rule: the segment after the prefix, up to the slash, is the vendor pack.
KEY_PREFIX = "openrouter:"


class OpenRouterModel(NamedTuple):
    """One registerable slug, with where its price came from.

    Attributes:
        slug: OpenRouter model id, ``<vendor>/<model>``. Also the registry key
            once prefixed with :data:`KEY_PREFIX`.
        display_name: Human-readable name.
        input_price_per_1m: USD per 1M input tokens, or None when unverified.
        output_price_per_1m: USD per 1M output tokens, or None when unverified.
        price_source: Where the numbers came from. Required whenever a price is
            set -- an unsourced price is a guess.
        default_temperature: Passed through to ModelSpec. None omits the
            parameter entirely, which is the safe choice for models that reject
            it.
        tool_use: Whether the slug is registered as tool-capable. The assistant
            role needs this; the patient agent has no tools and does not.
        note: Anything an operator should know before selecting it.
    """

    slug: str
    display_name: str
    input_price_per_1m: Optional[float]
    output_price_per_1m: Optional[float]
    price_source: str
    default_temperature: Optional[float] = 0
    tool_use: bool = True
    note: str = ""


# Source for every price below, quoted so the provenance is checkable without
# leaving this file:
#
#   patientwords-engine/data/advice_providers.json -- the REVIEWED copy of the
#   engine's provider registry. Its own header states: "pricing = [input,
#   output] USD per Mtok, used only by the local spend-ceiling math (worst-case,
#   cache-miss rates); the true bill is the provider's." OpenRouter-routed
#   entries record list price plus roughly a 5% aggregator margin -- deliberately
#   conservative for ceiling arithmetic. Consumer tiers were verified against
#   vendor docs on 2026-07-21, and the file carries at least one recorded
#   correction from a live 400 (openai/gpt-5.5-mini does not exist on
#   OpenRouter; openai/gpt-5.4-mini does), so the slugs are evidence-backed and
#   not transcribed from memory.
#
# These are therefore *ceiling-side* prices: an estimate built on them is an
# upper bound, and the provider's invoice is the ground truth. Re-verify against
# OpenRouter's live model list before the first fire -- slugs and prices drift,
# and this repo's sandbox cannot reach openrouter.ai to do it automatically.
_ENGINE_REGISTRY = "patientwords-engine/data/advice_providers.json"

OPENROUTER_MODELS: List[OpenRouterModel] = [
    OpenRouterModel(
        slug="openai/gpt-5.4-mini",
        display_name="GPT-5.4 Mini (OpenRouter)",
        input_price_per_1m=0.80,
        output_price_per_1m=4.75,
        price_source=f"{_ENGINE_REGISTRY} :: openai.pricing['openai/gpt-5.4-mini']",
        # The gpt-5 family accepts only temperature=1; upstream's own gpt-5.x
        # specs pin it there.
        default_temperature=1,
        note="Slug existence confirmed by the engine's registry, which records "
             "a live 400 for openai/gpt-5.5-mini and the correction to this id.",
    ),
    OpenRouterModel(
        slug="openai/gpt-5.5",
        display_name="GPT-5.5 (OpenRouter)",
        input_price_per_1m=5.25,
        output_price_per_1m=31.50,
        price_source=f"{_ENGINE_REGISTRY} :: openai.default_pricing",
        default_temperature=1,
        note="Served by OpenAI's backend through OpenRouter's routing; an "
             "intermediary sits in the request path.",
    ),
    OpenRouterModel(
        slug="x-ai/grok-4.3",
        display_name="Grok 4.3 (OpenRouter)",
        input_price_per_1m=1.32,
        output_price_per_1m=2.63,
        price_source=f"{_ENGINE_REGISTRY} :: xai.default_pricing",
    ),
    OpenRouterModel(
        slug="moonshotai/kimi-k2.5",
        display_name="Kimi K2.5 (OpenRouter)",
        input_price_per_1m=0.63,
        output_price_per_1m=3.15,
        price_source=f"{_ENGINE_REGISTRY} :: moonshot.default_pricing",
    ),
    OpenRouterModel(
        slug="deepseek/deepseek-v4-flash",
        display_name="DeepSeek V4 Flash (OpenRouter)",
        input_price_per_1m=0.15,
        output_price_per_1m=0.30,
        price_source=f"{_ENGINE_REGISTRY} :: deepseek.default_pricing",
    ),
    OpenRouterModel(
        slug="google/gemini-3.5-flash",
        display_name="Gemini 3.5 Flash (OpenRouter)",
        input_price_per_1m=0.35,
        output_price_per_1m=2.75,
        price_source=f"{_ENGINE_REGISTRY} :: openrouter.pricing['google/gemini-3.5-flash']",
    ),
]


class OpenRouterSpecError(RuntimeError):
    """The OpenRouter specs cannot be injected or used as configured."""


def registry_key(slug: str) -> str:
    """Registry key for an OpenRouter slug (``openrouter:vendor/model``)."""
    return slug if slug.startswith(KEY_PREFIX) else KEY_PREFIX + slug


def vendor_pack(key: str) -> str:
    """Vendor a registry key belongs to, per the engine's attribution rule."""
    slug = key[len(KEY_PREFIX):] if key.startswith(KEY_PREFIX) else key
    return slug.split("/", 1)[0]


def ensure_base_url() -> str:
    """Point :data:`BASE_URL_ENV` at OpenRouter unless the operator set it.

    ``setdefault``, never overwrite: an operator routing through a proxy or a
    regional endpoint keeps their value. Called at import so selecting an
    injected spec needs only the API key.
    """
    return os.environ.setdefault(BASE_URL_ENV, BASE_URL)


def build_spec(model: OpenRouterModel) -> ModelSpec:
    """Build the upstream ModelSpec for one OpenRouter model."""
    if (model.input_price_per_1m is None) != (model.output_price_per_1m is None):
        raise OpenRouterSpecError(
            f"{model.slug}: half-priced spec. Both prices or neither -- upstream "
            "renders cost as N/A only when both are None."
        )
    if model.input_price_per_1m is not None and not model.price_source:
        raise OpenRouterSpecError(
            f"{model.slug}: priced without a price_source. An unsourced price is a guess."
        )
    capabilities = [ModelCapability.TEXT]
    if model.tool_use:
        capabilities.append(ModelCapability.TOOL_USE)
    notes = [f"Routed via OpenRouter ({BASE_URL}); key from ${API_KEY_ENV}."]
    if model.input_price_per_1m is None:
        notes.append("UNPRICED: no verified list price; cost renders as N/A.")
    else:
        notes.append(f"Price source: {model.price_source}.")
    if model.note:
        notes.append(model.note)
    return ModelSpec(
        model_id=model.slug,
        display_name=model.display_name,
        provider="openai-protocol-api",
        developer=vendor_pack(model.slug),
        auth="api_key",
        default_temperature=model.default_temperature,
        default_max_tokens=4096,
        capabilities=tuple(capabilities),
        description=f"{model.display_name} via the OpenRouter aggregator",
        notes=" ".join(notes),
        # Required. Without it ChatOpenAI falls back to api.openai.com.
        base_url_env=BASE_URL_ENV,
        api_key_env=API_KEY_ENV,
        # OpenRouter speaks Chat Completions. Upstream's gpt-5.x-api specs set
        # this True for OpenAI's Responses API; copying that here would send the
        # wrong request shape.
        use_responses_api=False,
        input_price_per_1m=model.input_price_per_1m,
        output_price_per_1m=model.output_price_per_1m,
    )


def openrouter_model_specs() -> Dict[str, ModelSpec]:
    """Every OpenRouter spec, keyed by registry key. Built fresh each call."""
    return {registry_key(m.slug): build_spec(m) for m in OPENROUTER_MODELS}


def register_openrouter_models() -> List[str]:
    """Inject the specs into upstream's ``MODEL_STORE``.

    Idempotent (a plain dict keyed by name), and additive: it refuses to
    overwrite a key upstream already defines, so an upstream release that starts
    shipping one of these slugs wins rather than being silently shadowed.

    Returns the registry keys now available.
    """
    ensure_base_url()
    injected = []
    for key, spec in openrouter_model_specs().items():
        existing = MODEL_STORE.get(key)
        if existing is not None and getattr(existing, "base_url_env", None) != BASE_URL_ENV:
            # Upstream now defines this key itself. Leave theirs alone.
            continue
        MODEL_STORE[key] = spec
        injected.append(key)
    return injected


def api_key_present() -> bool:
    """Whether the OpenRouter key is set. Never returns or logs its value."""
    return bool(os.environ.get(API_KEY_ENV, "").strip())


register_openrouter_models()
