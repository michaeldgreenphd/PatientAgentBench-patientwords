# SPDX-License-Identifier: CC-BY-NC-4.0
"""A live spend ceiling for benchmark runs, installed without editing upstream.

The benchmark has no budget enforcement: ``max_spend`` in the CI trigger is a
bookkeeping number that the fire guard counts, and nothing at all stops a run
once it starts. For an attended run that is fine. For an unattended overnight
run against a prepaid key it is not -- the only real ceiling would be the
balance itself, which is the wrong instrument and the wrong amount.

This module adds the missing ceiling. It meters every LLM call the run makes,
prices it from the registry, and aborts the moment the cumulative cost crosses
a declared limit.

**The chokepoint is upstream's, not an assumption.** ``config.create_chat_model``
documents itself as "the single factory for all LLM creation in the project",
and every agent, rubric, sandbox and analyzer goes through it. Wrapping that one
function therefore meters everything -- assistant, patient, sandbox, jury.

**Rebinding is the trap.** Callers do ``from patient_agent_bench.config import
create_chat_model``, which copies the function object into their own module
namespace at import time. Patching only ``config`` would leave every
already-imported caller pointing at the original, and the guard would silently
meter a fraction of the run while reporting a number that looked fine.
:func:`install` therefore patches the factory in ``config`` *and* in every module
that has already bound it, discovered by scanning ``sys.modules`` for the
original function object rather than from a hand-maintained list -- a caller
upstream adds later is covered without an edit here. :func:`installed_sites`
reports what was patched so a run can assert its own coverage.

**Fail closed on unpriced models.** A model with no registry price cannot be
metered, so a ceiling that included one would be a ceiling in name only. The
guard refuses to start rather than run blind; ``allow_unpriced=True`` is the
explicit opt-out, and it downgrades the ceiling to "best effort over the priced
legs" in the report rather than pretending.

**The abort must not be swallowed.** :class:`BudgetExceeded` derives from
``BaseException``, like ``KeyboardInterrupt``: upstream wraps agent failures in
``except Exception`` and retries them with backoff, which for an overspend would
mean spending more. Conversations already finished are safe -- upstream writes
each one to disk as it completes -- so an abort costs the run, not the results.

Prices are the registry's list prices times reported token usage. That is an
estimate, not the provider's invoice, and it is labelled as such everywhere it
is written.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from patient_agent_bench import config as _config
from patient_agent_bench.model_registry import get_model_pricing

#: Env var naming the ceiling in USD. Absent means no guard is installed.
MAX_SPEND_ENV = "PW_MAX_SPEND_USD"

#: Env var naming where to write the spend report. Absent means no file.
REPORT_ENV = "PW_SPEND_REPORT"

#: Env var opting out of the unpriced-model refusal. Any non-empty value.
ALLOW_UNPRICED_ENV = "PW_ALLOW_UNPRICED"

COST_BASIS = "registry list prices x reported token usage; not the provider's invoice"


class BudgetExceeded(BaseException):
    """The declared spend ceiling was reached; the run must stop.

    Deliberately a ``BaseException``. Upstream retries anything deriving from
    ``Exception`` with backoff, and retrying an overspend spends more.
    """


class BudgetGuardError(RuntimeError):
    """The guard cannot be installed or cannot bound the run as configured."""


@dataclass
class ModelUsage:
    """Token usage and cost for one model, summed across every role using it."""

    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, usage: Optional[Dict[str, Any]]) -> None:
        self.calls += 1
        if not usage:
            return
        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or 0)

    def cost(self) -> Optional[float]:
        """USD from the registry's list prices, or None when unpriced."""
        pricing = get_model_pricing(self.model)
        if pricing is None:
            return None
        return (
            self.input_tokens * pricing["input_price_per_1m"]
            + self.output_tokens * pricing["output_price_per_1m"]
        ) / 1_000_000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost(),
        }


@dataclass
class SpendGuard:
    """Accumulates spend across every metered call and trips at the ceiling.

    Thread-safe: upstream can run experiments in parallel, so the accumulator is
    shared across threads and every mutation takes the lock. Without it the
    ceiling would be checked against a torn read.
    """

    ceiling_usd: float
    allow_unpriced: bool = False
    per_model: Dict[str, ModelUsage] = field(default_factory=dict)
    tripped: bool = False
    unpriced: List[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.ceiling_usd <= 0:
            raise BudgetGuardError(
                f"ceiling must be > 0, got {self.ceiling_usd!r}; a run with no "
                "positive ceiling is an unbounded run"
            )

    # -- pre-flight ---------------------------------------------------------

    def require_priced(self, models: List[str]) -> None:
        """Refuse a run whose models cannot all be metered.

        Called before the first call, so an unpriced model costs nothing to
        discover. ``allow_unpriced`` downgrades this to a recorded warning.
        """
        missing = sorted({m for m in models if get_model_pricing(m) is None})
        if not missing:
            return
        self.unpriced = missing
        if self.allow_unpriced:
            return
        raise BudgetGuardError(
            "cannot enforce a ceiling: no registry price for "
            + ", ".join(repr(m) for m in missing)
            + f". Register a verified price, or set {ALLOW_UNPRICED_ENV}=1 to "
            "accept a ceiling that only covers the priced legs."
        )

    # -- metering -----------------------------------------------------------

    def record(self, model: str, usage: Optional[Dict[str, Any]]) -> None:
        with self._lock:
            self.per_model.setdefault(model, ModelUsage(model=model)).add(usage)

    def total(self) -> float:
        """Cost of the priced legs. Unpriced legs contribute nothing.

        Never None: a guard that returned None here could not compare against a
        ceiling. The report carries ``unpriced`` so the number is read with the
        right caveat.
        """
        with self._lock:
            return round(sum(u.cost() or 0.0 for u in self.per_model.values()), 6)

    def check(self) -> None:
        """Raise if the accumulated spend has reached the ceiling."""
        spent = self.total()
        if spent < self.ceiling_usd:
            return
        self.tripped = True
        raise BudgetExceeded(
            f"spend ceiling reached: ${spent:.4f} of ${self.ceiling_usd:.4f} "
            f"({COST_BASIS}). Conversations already finished are on disk."
        )

    # -- reporting ----------------------------------------------------------

    def report(self, **extra: Any) -> Dict[str, Any]:
        with self._lock:
            per_model = {m: u.to_dict() for m, u in sorted(self.per_model.items())}
        priced = [u for u in per_model.values() if u["cost"] is not None]
        return {
            "task": "pab-run",
            "run_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "max_spend_usd": self.ceiling_usd,
            "cost_usd": round(sum(u["cost"] for u in priced), 6) if priced else None,
            "cost_basis": COST_BASIS,
            "ceiling_tripped": self.tripped,
            "unpriced_models": list(self.unpriced),
            "usage": {"per_model": per_model},
            **extra,
        }

    def write_report(self, path: Path, **extra: Any) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report(**extra), indent=2) + "\n",
                        encoding="utf-8")
        return path


class GuardedChatModel:
    """Pass-through wrapper that meters every response and checks the ceiling.

    Not a subclass: the wrapped object is a pydantic ``BaseChatModel`` whose
    fields would reject an extra attribute, and after ``bind_tools`` it is a
    plain ``Runnable`` rather than a chat model at all. Forwarding
    ``invoke``/``ainvoke`` and delegating the rest is enough for every caller in
    the benchmark, and ``create_agent`` accepts it (pinned by a test).

    ``bind_tools``, ``bind`` and ``with_structured_output`` re-wrap their result.
    Returning the bare inner runnable is the hole that matters: the assistant is
    the only leg that binds tools, it is also the most expensive leg, and its
    calls would have escaped the meter entirely while the report still looked
    complete.
    """

    def __init__(self, inner: Any, model: str, guard: SpendGuard):
        self._inner = inner
        self._model = model
        self._guard = guard

    def _rewrap(self, inner: Any) -> "GuardedChatModel":
        return GuardedChatModel(inner, self._model, self._guard)

    def _record(self, response: Any) -> Any:
        self._guard.record(self._model, getattr(response, "usage_metadata", None))
        self._guard.check()
        return response

    def invoke(self, *args, **kwargs):
        # Checked before as well as after: a ceiling already crossed on another
        # thread must not buy one more call per remaining worker.
        self._guard.check()
        return self._record(self._inner.invoke(*args, **kwargs))

    async def ainvoke(self, *args, **kwargs):
        self._guard.check()
        return self._record(await self._inner.ainvoke(*args, **kwargs))

    def bind_tools(self, *args, **kwargs):
        return self._rewrap(self._inner.bind_tools(*args, **kwargs))

    def bind(self, *args, **kwargs):
        return self._rewrap(self._inner.bind(*args, **kwargs))

    def with_structured_output(self, *args, **kwargs):
        return self._rewrap(self._inner.with_structured_output(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._inner, name)


# Modules patched by the last install(), for assertions and uninstall.
_patched: List[str] = []
_original = None


def _binding_modules(original) -> List[Any]:
    """Every imported module holding a reference to the original factory.

    Discovered by object identity rather than a hand-kept list, so a caller
    upstream adds later is covered without an edit here. ``list(...)`` because
    importing during the scan would mutate ``sys.modules``.
    """
    found = []
    for module in list(sys.modules.values()):
        if module is None:
            continue
        try:
            if getattr(module, "create_chat_model", None) is original:
                found.append(module)
        except Exception:  # noqa: BLE001 - a module raising on getattr is not a caller
            continue
    return found


def install(guard: SpendGuard) -> List[str]:
    """Meter every model the run creates. Returns the patched module names.

    Patches ``config`` first so modules imported *after* this call pick up the
    guarded factory through their own import, then patches the modules that
    already copied the original into their namespace.
    """
    global _original
    if _original is not None:
        raise BudgetGuardError("a guard is already installed")
    original = _config.create_chat_model
    _original = original

    def guarded(model_config, bedrock_client=None, role_arn=None):
        inner = original(model_config, bedrock_client, role_arn=role_arn)
        return GuardedChatModel(inner, getattr(model_config, "model", "unknown"), guard)

    guarded.__name__ = original.__name__
    guarded.__doc__ = original.__doc__

    sites = _binding_modules(original)
    _config.create_chat_model = guarded
    for module in sites:
        module.create_chat_model = guarded
    _patched[:] = sorted({m.__name__ for m in sites} | {_config.__name__})
    return list(_patched)


def uninstall() -> None:
    """Restore the original factory. Mainly for tests."""
    global _original
    if _original is None:
        return
    for name in _patched:
        module = sys.modules.get(name)
        if module is not None:
            module.create_chat_model = _original
    _original = None
    _patched.clear()


def installed_sites() -> List[str]:
    """Module names the guard is currently patched into."""
    return list(_patched)


def guard_from_env() -> Optional[SpendGuard]:
    """Build a guard from the environment, or None when no ceiling is declared.

    A malformed ceiling is an error, not a silent skip: ``PW_MAX_SPEND_USD=abc``
    means someone intended a ceiling, and running unbounded is the one outcome
    they did not intend.
    """
    raw = os.environ.get(MAX_SPEND_ENV, "").strip()
    if not raw:
        return None
    try:
        ceiling = float(raw)
    except ValueError as exc:
        raise BudgetGuardError(
            f"{MAX_SPEND_ENV}={raw!r} is not a number; refusing to run unbounded"
        ) from exc
    return SpendGuard(
        ceiling_usd=ceiling,
        allow_unpriced=bool(os.environ.get(ALLOW_UNPRICED_ENV, "").strip()),
    )


def report_path_from_env() -> Optional[Path]:
    raw = os.environ.get(REPORT_ENV, "").strip()
    return Path(raw) if raw else None
