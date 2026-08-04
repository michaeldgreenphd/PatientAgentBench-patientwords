# SPDX-License-Identifier: CC-BY-NC-4.0
"""Offline tests for the live OpenRouter catalogue check.

The fetch is mocked. What is tested is the part that protects spend: a slug the
catalogue does not serve must fail loudly *before* a run, and an unpriced model
must not be silently accepted.
"""

import json
from unittest.mock import patch

import pytest

from patientwords_pab import openrouter_catalog as cat

CATALOGUE = {
    "data": [
        {"id": "openai/gpt-5.5", "name": "GPT-5.5",
         "pricing": {"prompt": "0.00000500", "completion": "0.00003000"},
         "context_length": 400000},
        {"id": "qwen/qwen3-235b-a22b", "name": "Qwen3 235B",
         "pricing": {"prompt": "0.00000022", "completion": "0.00000088"}},
        {"id": "vendor/free-model", "name": "Free",
         "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "vendor/unquoted", "name": "Unquoted", "pricing": {}},
    ]
}


@pytest.fixture
def catalogue():
    class _Response:
        def read(self):
            return json.dumps(CATALOGUE).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    with patch.object(cat.urllib.request, "urlopen", return_value=_Response()):
        yield cat.fetch_catalogue()


class TestFetch:
    def test_indexes_by_slug(self, catalogue):
        assert set(catalogue) == {"openai/gpt-5.5", "qwen/qwen3-235b-a22b",
                                  "vendor/free-model", "vendor/unquoted"}

    def test_network_failure_is_an_error_not_a_default(self):
        with patch.object(cat.urllib.request, "urlopen",
                          side_effect=OSError("no route")):
            with pytest.raises(cat.CatalogueError, match="cannot fetch"):
                cat.fetch_catalogue()


class TestPricing:
    def test_per_token_is_converted_to_per_million(self, catalogue):
        assert cat.price_per_million(catalogue["openai/gpt-5.5"]) == (5.0, 30.0)

    def test_zero_is_a_real_price_not_a_missing_one(self, catalogue):
        """OpenRouter lists genuinely free endpoints; treating free as unknown
        would be its own guess."""
        assert cat.price_per_million(catalogue["vendor/free-model"]) == (0.0, 0.0)

    def test_missing_quote_is_none(self, catalogue):
        assert cat.price_per_million(catalogue["vendor/unquoted"]) is None


class TestResolve:
    def test_resolves_known_slugs(self, catalogue):
        resolved = cat.resolve(catalogue, ["openai/gpt-5.5"])
        assert resolved["openai/gpt-5.5"]["input_price_per_1m"] == 5.0
        assert resolved["openai/gpt-5.5"]["priced"] is True

    def test_unknown_slug_fails_with_a_suggestion(self, catalogue):
        """The failure the engine's provider registry records happening for
        real: a plausible slug that does not exist, found by a live 400."""
        with pytest.raises(cat.CatalogueError) as err:
            cat.resolve(catalogue, ["qwen/qwen3-nonexistent"])
        assert "not in the catalogue" in str(err.value)
        assert "qwen/qwen3-235b-a22b" in str(err.value)

    def test_unpriced_slug_resolves_but_is_flagged(self, catalogue):
        resolved = cat.resolve(catalogue, ["vendor/unquoted"])
        assert resolved["vendor/unquoted"]["priced"] is False
        assert resolved["vendor/unquoted"]["input_price_per_1m"] is None


class TestSearch:
    def test_finds_by_substring_cheapest_first(self, catalogue):
        hits = [m["id"] for m in cat.search(catalogue, "vendor")]
        assert hits[0] == "vendor/free-model"
        assert "vendor/unquoted" in hits


class TestCli:
    def test_require_exits_two_on_a_missing_slug(self, capsys):
        class _Response:
            def read(self):
                return json.dumps(CATALOGUE).encode()

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        with patch.object(cat.urllib.request, "urlopen", return_value=_Response()):
            assert cat.main(["--require", "qwen/nope"]) == 2
        assert "not in the catalogue" in capsys.readouterr().err

    def test_unpriced_required_slug_is_refused_by_default(self, capsys):
        class _Response:
            def read(self):
                return json.dumps(CATALOGUE).encode()

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        with patch.object(cat.urllib.request, "urlopen", return_value=_Response()):
            assert cat.main(["--require", "vendor/unquoted"]) == 2
        assert "does not price" in capsys.readouterr().err

    def test_search_json(self, capsys):
        class _Response:
            def read(self):
                return json.dumps(CATALOGUE).encode()

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        with patch.object(cat.urllib.request, "urlopen", return_value=_Response()):
            assert cat.main(["--search", "qwen", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["search"]["qwen"][0]["slug"] == "qwen/qwen3-235b-a22b"
