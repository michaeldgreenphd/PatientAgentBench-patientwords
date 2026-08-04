# SPDX-License-Identifier: CC-BY-NC-4.0
"""PatientWords trait-sweep adapter for PatientAgentBench (Layer 1).

This package is the ONLY place in this fork that PatientWords code lives. It is
additive: no file under ``src/patient_agent_bench/`` (Layer 0, upstream) is
edited, so ``git pull upstream main`` stays a fast-forward.

Licensing. This package imports PatientAgentBench code and is therefore a
derivative work of it: it inherits CC-BY-NC-4.0 and must stay in this fork.
The measurement engine (``patientwords-engine``, MIT) reads the runner's JSON
output as data and never imports any of this.

What it adds. Upstream's persona system bundles seven traits into six fixed
presets, and ``get_personality_prompt()`` accepts only a preset name — so no
single trait is identified. :mod:`patientwords_pab.trait_spec` builds a persona
block from an arbitrary trait -> level mapping using upstream's own
``TRAIT_DEFINITIONS``, and :class:`patientwords_pab.free_trait_agent.
FreeTraitUserAgent` feeds that block through upstream's own prompt template.

Importing this package registers the agent under the name ``pw_free_trait`` via
upstream's documented ``register_user_agent()`` hook, so a benchmark config can
select it with ``"agent_class": "pw_free_trait"``.
"""

from patientwords_pab.free_trait_agent import (
    FreeTraitUserAgent,
    register_free_trait_agent,
)
from patientwords_pab.direct_api_specs import (
    PRICED_DIRECT_API,
    register_direct_api_models,
)
from patientwords_pab.openrouter_specs import (
    API_KEY_ENV,
    BASE_URL,
    BASE_URL_ENV,
    KEY_PREFIX,
    OPENROUTER_MODELS,
    api_key_present,
    openrouter_model_specs,
    register_openrouter_models,
    registry_key,
    vendor_pack,
)
from patientwords_pab.trait_spec import (
    NEUTRAL_BASE,
    NEUTRAL_LEVEL,
    PW_PREFIX,
    TraitSpec,
    TraitSpecError,
    format_trait_spec,
    is_free_trait_spec,
    neutral_traits,
    off_preset_level_pairs,
    parse_trait_spec,
    preset_distances,
    render_trait_block,
    single_factor_arms,
)

__all__ = [
    "API_KEY_ENV",
    "BASE_URL",
    "BASE_URL_ENV",
    "KEY_PREFIX",
    "NEUTRAL_BASE",
    "NEUTRAL_LEVEL",
    "OPENROUTER_MODELS",
    "PRICED_DIRECT_API",
    "PW_PREFIX",
    "FreeTraitUserAgent",
    "TraitSpec",
    "TraitSpecError",
    "api_key_present",
    "format_trait_spec",
    "is_free_trait_spec",
    "neutral_traits",
    "off_preset_level_pairs",
    "openrouter_model_specs",
    "parse_trait_spec",
    "preset_distances",
    "register_direct_api_models",
    "register_free_trait_agent",
    "register_openrouter_models",
    "registry_key",
    "render_trait_block",
    "single_factor_arms",
    "vendor_pack",
]
