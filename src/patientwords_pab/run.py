# SPDX-License-Identifier: CC-BY-NC-4.0
"""CLI wrapper that registers the trait-sweep agent, then defers to upstream.

Upstream's ``patient-agent-bench`` console script imports only its own registry,
which knows about ``agent_class="default"``. Teaching it about ``pw_free_trait``
would mean editing ``user_agent/registry.py`` -- an upstream file -- so instead
this wrapper imports :mod:`patientwords_pab` first (which registers the agent
through the documented ``register_user_agent()`` hook) and then calls upstream's
``main()`` unchanged. Arguments, exit behaviour, and output are upstream's:

    python -m patientwords_pab.run benchmark --cases data/my_sweep.json \\
        --config data/my_config.json

Use this in place of ``patient-agent-bench`` for any run whose config selects
``"agent_class": "pw_free_trait"``. With upstream's own entry point that config
raises ``KeyError: Unknown user agent_class 'pw_free_trait'`` -- a loud failure
before any spend, not a silent fallback.
"""

from __future__ import annotations

import patientwords_pab  # noqa: F401  -- import registers the agent
from patient_agent_bench.run import main

if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
