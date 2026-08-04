# SPDX-License-Identifier: CC-BY-NC-4.0
"""Live OpenRouter catalogue: verify slugs exist and read their real prices.

The development sandbox cannot reach openrouter.ai, so the specs in
``openrouter_specs.py`` carry prices transcribed from an in-repo registry and
slugs that were never confirmed against the live list. CI can reach it. This
module closes that gap at the only moment it matters -- immediately before
spending -- by fetching the public model list, failing on any slug that does not
exist, and reading each price from the vendor rather than from a transcription.

    # discovery: what does OpenRouter actually call these?
    python -m patientwords_pab.openrouter_catalog --search qwen3 --search gpt-oss

    # pre-flight: refuse to proceed unless every slug exists and is priced
    python -m patientwords_pab.openrouter_catalog \\
        --require openai/gpt-5.5 --require qwen/qwen3-235b-a22b --json

The endpoint is public and unauthenticated: listing models costs nothing and
sends no key. If a key is present it is not used here.

Prices come back as USD per token and are converted to USD per 1M to match the
registry's units. A model the catalogue prices at 0 for both directions is
reported as free rather than unpriced -- OpenRouter does list genuinely free
endpoints, and conflating "free" with "unknown" would be its own guess.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

CATALOGUE_URL = "https://openrouter.ai/api/v1/models"
TIMEOUT_SECONDS = 30
PER_MILLION = 1_000_000


class CatalogueError(RuntimeError):
    """The catalogue could not be fetched, or a required slug is not in it."""


def fetch_catalogue(url: str = CATALOGUE_URL, timeout: int = TIMEOUT_SECONDS) -> dict:
    """Fetch the public model list. No credentials are sent."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, ValueError) as err:
        raise CatalogueError(f"cannot fetch {url}: {err}") from err
    models = payload.get("data")
    if not isinstance(models, list) or not models:
        raise CatalogueError(f"{url} returned no model list")
    return {m["id"]: m for m in models if isinstance(m, dict) and m.get("id")}


def price_per_million(model: dict) -> Optional[Tuple[float, float]]:
    """``(input, output)`` USD per 1M tokens, or None when not quoted.

    OpenRouter quotes per token as strings. A quote of "0" is a real price --
    free endpoints exist -- so only a missing or unparseable quote is None.
    """
    pricing = model.get("pricing") or {}
    try:
        prompt = float(pricing["prompt"])
        completion = float(pricing["completion"])
    except (KeyError, TypeError, ValueError):
        return None
    return round(prompt * PER_MILLION, 6), round(completion * PER_MILLION, 6)


def search(catalogue: dict, term: str, limit: int = 12) -> List[dict]:
    """Slugs containing ``term``, cheapest input first, for discovery."""
    needle = term.lower()
    hits = [m for slug, m in catalogue.items() if needle in slug.lower()]

    def sort_key(model):
        price = price_per_million(model)
        return (price[0] if price else float("inf"), model["id"])

    return sorted(hits, key=sort_key)[:limit]


def resolve(catalogue: dict, slugs: List[str]) -> Dict[str, dict]:
    """Prices for every requested slug, or raise naming the ones that are missing.

    Fails loudly and before any spend. A near-miss suggestion is included
    because the usual cause is a plausible-but-wrong slug -- the engine's
    provider registry records exactly that happening once, a live 400 for a
    model id that did not exist.
    """
    resolved: Dict[str, dict] = {}
    missing: List[str] = []
    for slug in slugs:
        model = catalogue.get(slug)
        if model is None:
            missing.append(slug)
            continue
        price = price_per_million(model)
        resolved[slug] = {
            "slug": slug,
            "name": model.get("name", ""),
            "input_price_per_1m": None if price is None else price[0],
            "output_price_per_1m": None if price is None else price[1],
            "context_length": model.get("context_length"),
            "priced": price is not None,
        }
    if missing:
        lines = []
        for slug in missing:
            stem = slug.split("/")[-1].split("-")[0]
            near = [m["id"] for m in search(catalogue, stem, limit=5)]
            lines.append(f"  {slug!r} not in the catalogue"
                         + (f"; nearest: {', '.join(near)}" if near else ""))
        raise CatalogueError(
            "slug(s) OpenRouter does not serve:\n" + "\n".join(lines)
        )
    return resolved


def compare_with_specs(resolved: Dict[str, dict]) -> List[dict]:
    """Live prices against the prices the specs were built with.

    The specs carry ceiling-side numbers (list plus roughly a 5% aggregator
    margin). A live price *above* the spec is the one that matters: it means an
    estimate built on the spec was not the upper bound it claimed to be.
    """
    from patientwords_pab.openrouter_specs import OPENROUTER_MODELS

    by_slug = {m.slug: m for m in OPENROUTER_MODELS}
    rows = []
    for slug, live in sorted(resolved.items()):
        spec = by_slug.get(slug)
        row = {"slug": slug,
               "live_input": live["input_price_per_1m"],
               "live_output": live["output_price_per_1m"],
               "spec_input": spec.input_price_per_1m if spec else None,
               "spec_output": spec.output_price_per_1m if spec else None}
        if spec and spec.input_price_per_1m is not None and live["priced"]:
            row["spec_is_upper_bound"] = (
                live["input_price_per_1m"] <= spec.input_price_per_1m
                and live["output_price_per_1m"] <= spec.output_price_per_1m
            )
        rows.append(row)
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--search", action="append", default=[],
                        help="print catalogue entries whose slug contains this (repeatable)")
    parser.add_argument("--require", action="append", default=[],
                        help="fail unless this slug exists and is priced (repeatable)")
    parser.add_argument("--allow-unpriced", action="store_true",
                        help="accept a required slug the catalogue does not price")
    parser.add_argument("--compare-specs", action="store_true",
                        help="check the built-in spec prices are upper bounds")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    if not args.search and not args.require:
        parser.error("give --search and/or --require")

    try:
        catalogue = fetch_catalogue()
    except CatalogueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    out: Dict[str, object] = {"catalogue_size": len(catalogue)}

    for term in args.search:
        hits = search(catalogue, term)
        out.setdefault("search", {})[term] = [  # type: ignore[union-attr]
            {"slug": m["id"], "name": m.get("name", ""),
             "price_per_1m": price_per_million(m)} for m in hits
        ]
        if not args.as_json:
            print(f"\n== {term!r}: {len(hits)} match(es)")
            for model in hits:
                price = price_per_million(model)
                quote = "unpriced" if price is None else f"${price[0]:g} / ${price[1]:g}"
                print(f"  {model['id']:<45} {quote:<22} {model.get('name', '')}")

    if args.require:
        try:
            resolved = resolve(catalogue, args.require)
        except CatalogueError as err:
            print(f"error: {err}", file=sys.stderr)
            return 2
        unpriced = sorted(s for s, r in resolved.items() if not r["priced"])
        if unpriced and not args.allow_unpriced:
            print(f"error: catalogue does not price {', '.join(unpriced)}; "
                  "refusing to run against an unpriced model", file=sys.stderr)
            return 2
        out["resolved"] = resolved
        if args.compare_specs:
            out["spec_comparison"] = compare_with_specs(resolved)
        if not args.as_json:
            print("\n== required slugs")
            for slug, row in sorted(resolved.items()):
                print(f"  {slug:<45} ${row['input_price_per_1m']:g} / "
                      f"${row['output_price_per_1m']:g} per 1M")
            for row in out.get("spec_comparison", []):  # type: ignore[union-attr]
                if row.get("spec_is_upper_bound") is False:
                    print(f"  WARNING {row['slug']}: live price exceeds the spec "
                          f"({row['live_input']}/{row['live_output']} vs "
                          f"{row['spec_input']}/{row['spec_output']}); estimates "
                          "built on the spec were not upper bounds")

    if args.as_json:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
