# SPDX-License-Identifier: CC-BY-NC-4.0
"""Priced variants of upstream's direct-API Anthropic specs.

Upstream registers its ``*-api`` keys -- the direct Anthropic and OpenAI channel,
as opposed to Bedrock -- with ``input_price_per_1m=None``. That is a reasonable
default for a benchmark that bills through Bedrock, but it makes the whole
channel unmeterable: :mod:`patientwords_pab.budget_guard` cannot bound a run
whose legs have no price, and the jury is exactly such a leg. Registering a
price is what turns the $2 Anthropic budget into an enforceable ceiling rather
than an intention.

**New keys, not edits.** Each spec here is a copy of upstream's, with prices
filled in, stored under a ``pw:`` prefix. Upstream's own entry is left exactly as
it is -- this module never mutates it -- so nothing is shadowed and a config that
names ``claude-opus-4.8-api`` still gets upstream's unpriced spec and still
refuses to run under a ceiling. Choosing the priced variant is explicit, and the
transcript records which one was used.

**Prices, and where they came from.** Both sources are named per spec, and they
agree, which is the point of listing two:

1. ``medlang_circuits/evaluate_models.py :: PRICING`` in the measurement engine
   -- the table that study already uses to price its own Claude spend, and the
   source ``docs/pab_first_probe_costing.md`` prices the jury leg from.
2. The live OpenRouter catalogue captured in CI on 2026-08-04, which lists
   ``anthropic/claude-opus-4.8`` at the same $5/$25 and
   ``anthropic/claude-haiku-4.5`` at $1/$5.

They are transcribed rather than imported: this package is CC-BY-NC-4.0 and the
engine is MIT, and the licence boundary is one-directional by design -- the
engine never imports the fork, and the fork does not take a build dependency on
the engine to read six numbers. A price whose two independent sources disagree
would be registered as None instead.

Direct-API billing is a *different account* from OpenRouter. A ceiling that
summed the two would answer the wrong question, which is why
``pab_probe_cost.py :: billing_channel`` splits them and why the jury stage
declares its own ceiling.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, NamedTuple, Optional

from patient_agent_bench.model_registry import MODEL_STORE, ModelSpec

#: Prefix marking a spec this package priced, as opposed to upstream's own.
KEY_PREFIX = "pw:"

_ENGINE_PRICING = "patientwords-engine/medlang_circuits/evaluate_models.py :: PRICING"
_LIVE_CATALOGUE = (
    "patientwords-engine/data/pab/openrouter_catalogue_20260804T025116Z.json"
)


class DirectAPIPrice(NamedTuple):
    """A price for an upstream key that upstream left unpriced."""

    upstream_key: str
    input_price_per_1m: float
    output_price_per_1m: float
    price_source: str


PRICED_DIRECT_API: List[DirectAPIPrice] = [
    DirectAPIPrice(
        upstream_key="claude-opus-4.8-api",
        input_price_per_1m=5.00,
        output_price_per_1m=25.00,
        price_source=f"{_ENGINE_PRICING}; confirmed by {_LIVE_CATALOGUE}",
    ),
    DirectAPIPrice(
        upstream_key="claude-sonnet-5-api",
        input_price_per_1m=2.00,
        output_price_per_1m=10.00,
        price_source=f"{_ENGINE_PRICING}; confirmed by {_LIVE_CATALOGUE}",
    ),
    DirectAPIPrice(
        upstream_key="claude-haiku-4.5-api",
        input_price_per_1m=1.00,
        output_price_per_1m=5.00,
        price_source=f"{_ENGINE_PRICING}; confirmed by {_LIVE_CATALOGUE}",
    ),
]


class DirectAPISpecError(RuntimeError):
    """A priced variant cannot be built from upstream's spec."""


def registry_key(upstream_key: str) -> str:
    """Key for the priced variant of an upstream key."""
    return upstream_key if upstream_key.startswith(KEY_PREFIX) else KEY_PREFIX + upstream_key


def build_spec(price: DirectAPIPrice) -> ModelSpec:
    """Upstream's spec with prices filled in, as a new object.

    ``dataclasses.replace`` rather than mutation: upstream's spec object is
    shared with ``MODEL_STORE``, and setting a field on it in place would price
    upstream's key as a side effect -- the silent edit this module exists to
    avoid.
    """
    base = MODEL_STORE.get(price.upstream_key)
    if base is None:
        raise DirectAPISpecError(
            f"{price.upstream_key!r} is not in upstream's registry; a priced "
            "variant of a key that does not exist would be a guess about the "
            "access channel as well as the price"
        )
    note = (
        f"Priced variant of {price.upstream_key!r}, which upstream registers "
        f"unpriced. Price source: {price.price_source}. Billed to the direct "
        "Anthropic account, NOT the OpenRouter balance."
    )
    notes = f"{base.notes} {note}".strip() if base.notes else note
    return dataclasses.replace(
        base,
        display_name=f"{base.display_name} (priced)",
        notes=notes,
        input_price_per_1m=price.input_price_per_1m,
        output_price_per_1m=price.output_price_per_1m,
    )


def direct_api_specs() -> Dict[str, ModelSpec]:
    """Every priced variant, keyed by registry key. Built fresh each call."""
    return {registry_key(p.upstream_key): build_spec(p) for p in PRICED_DIRECT_API}


def register_direct_api_models() -> List[str]:
    """Inject the priced variants. Never touches an upstream key.

    A variant whose upstream key has gone away is skipped rather than invented:
    upstream removing a model is a signal, not something to paper over.
    """
    injected = []
    for price in PRICED_DIRECT_API:
        if price.upstream_key not in MODEL_STORE:
            continue
        key = registry_key(price.upstream_key)
        MODEL_STORE[key] = build_spec(price)
        injected.append(key)
    return injected


def upstream_key_for(key: str) -> Optional[str]:
    """The upstream key a priced variant was built from, if it is one."""
    return key[len(KEY_PREFIX):] if key.startswith(KEY_PREFIX) else None


register_direct_api_models()
