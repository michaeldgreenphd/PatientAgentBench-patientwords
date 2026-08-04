# SPDX-License-Identifier: CC-BY-NC-4.0
"""CLI wrapper that registers the trait-sweep agent and bounds the spend.

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

**The spend ceiling.** Setting ``PW_MAX_SPEND_USD`` installs
:mod:`patientwords_pab.budget_guard`, which meters every LLM call and aborts the
run when the cumulative cost crosses the limit. Without it a run is bounded only
by the prepaid balance -- the wrong instrument and the wrong amount. The ceiling
is checked in three places, earliest first:

1. Before any call, every model named in the config must have a registry price.
   An unpriced model cannot be metered, so a ceiling covering it would be a
   ceiling in name only.
2. Before and after every call, against the running total.
3. In ``finally``, where the spend report is written whether the run finished,
   tripped the ceiling, or died -- a crashed run that spent money and left no
   sidecar is invisible to the ledger, which is how spend goes unrecorded.

``PW_SPEND_REPORT`` names the sidecar path. ``PW_ALLOW_UNPRICED=1`` opts out of
(1) and records the affected models in the report instead.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import patientwords_pab  # noqa: F401  -- import registers the agent
from patient_agent_bench.config import BenchConfig
from patient_agent_bench.run import main
from patientwords_pab import budget_guard


def config_path(argv: List[str]) -> Optional[str]:
    """The ``--config`` value in an upstream-shaped argv, if present."""
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def configured_models(path: str) -> List[str]:
    """Every model key a config can reach, across all six roles.

    Enumerated explicitly rather than by scanning the dataclass: a role missed
    here becomes a leg the pre-flight price check never sees, and the first
    evidence of that is an unmetered bill.
    """
    config = BenchConfig.from_file(path)
    models = [spec.model.model for spec in config.assistant_agents]
    models += [spec.model.model for spec in config.user_agents]
    models += [m.model for m in config.evaluator_models]
    models += [
        config.sandbox_model.model,
        config.seed_generator_model.model,
        config.analyzer_model.model,
    ]
    return sorted({m for m in models if m})


def run() -> None:
    guard = budget_guard.guard_from_env()
    if guard is None:
        main()
        return

    path = config_path(sys.argv)
    if path and os.path.exists(path):
        guard.require_priced(configured_models(path))

    sites = budget_guard.install(guard)
    print(f"[budget] ceiling ${guard.ceiling_usd:.4f}; "
          f"metering {len(sites)} call sites: {', '.join(sites)}", flush=True)

    outcome = "completed"
    try:
        main()
    except budget_guard.BudgetExceeded as exc:
        outcome = "ceiling_tripped"
        print(f"[budget] ABORTED: {exc}", file=sys.stderr, flush=True)
    except BaseException:
        outcome = "failed"
        raise
    finally:
        report_path = budget_guard.report_path_from_env()
        print(f"[budget] {outcome}: ${guard.total():.4f} of "
              f"${guard.ceiling_usd:.4f}", flush=True)
        if report_path is not None:
            guard.write_report(report_path, outcome=outcome,
                               argv=sys.argv[1:], config_file=path)
            print(f"[budget] report: {report_path}", flush=True)

    # A tripped ceiling is a deliberate stop, not a success. Exit non-zero so CI
    # shows it -- after the report has been written.
    if outcome == "ceiling_tripped":
        sys.exit(2)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    run()
