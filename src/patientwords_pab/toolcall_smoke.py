# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tool-calling smoke test: one live conversation, before any arm spend.

PatientAgentBench is agentic -- the assistant calls sandbox tools, and the
workflow-accuracy and triage rubrics score what it does with them. The paper
omits models "solely because they lacked reliable native tool-calling for
agentic workflows", so routing the assistant through a new access channel puts
that capability at risk before it puts anything else at risk. If tool-calling
does not survive the OpenRouter path, every conversation in Stage 1 is garbage
and the plan needs a different assistant. One conversation answers it.

The patient agent carries none of this risk: it has no tools. Only the
assistant under test does.

    # dry: prints the plan and the ceiling, makes no call
    python -m patientwords_pab.toolcall_smoke --dry-run

    # live: needs OPENROUTER_API_KEY; refuses to start without it
    python -m patientwords_pab.toolcall_smoke \\
        --assistant openrouter:openai/gpt-5.4-mini \\
        --report out/pab_toolcall_smoke.report.json

What it asserts: the transcript contains at least one tool call; every call
names a tool the sandbox actually registered; every call carries a dict of
arguments; and every call is answered by a matching tool result. A model that
narrates tool use in prose instead of emitting a call fails all four.

**Assembly.** This drives upstream's own factories in the order
``ConversationRunner._run_conversation`` uses them -- sandbox, tool registry,
assistant, user agent -- rather than calling the runner itself, for one reason:
the runner creates the sandbox's LLM client internally and discards its
response, so its cost is unobservable. A smoke test whose whole output is "what
did this cost" cannot leave a third of the calls unmetered. Each leg is metered
separately and the sidecar reports all three.

**Cost.** Computed from token usage times the registry's list prices, so it is a
list-price reconstruction, not the provider's invoice; the sidecar says so. With
an OpenRouter prepaid key the balance is a hard external ceiling above
``--max-spend``, which is checked before the run and reported after.

**Secrets.** The key is read by the HTTP client from the environment. This module
never reads its value, never puts it in the sidecar, and never logs it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from patient_agent_bench.assistant_agent.registry import create_assistant_agent_from_spec
from patient_agent_bench.benchmark_seed import load_benchmark_entries
from patient_agent_bench.config import AgentSpec, ModelConfig
from patient_agent_bench.model_registry import get_model_pricing
from patient_agent_bench.runner.conversation import Conversation
from patient_agent_bench.sandbox import (
    HealthcareSandbox,
    create_sandbox_llm,
    initialize_sandbox,
)
from patient_agent_bench.tools.registry import create_tool_registry
from patient_agent_bench.user_agent.registry import create_user_agent_from_spec
from patientwords_pab.openrouter_specs import (
    API_KEY_ENV,
    api_key_present,
    ensure_base_url,
)

# Defaults chosen in docs/pab_first_probe_costing.md (patientwords-engine).
DEFAULT_ASSISTANT = "openrouter:openai/gpt-5.4-mini"
DEFAULT_PATIENT = "openrouter:x-ai/grok-4.3"
DEFAULT_SANDBOX = "openrouter:openai/gpt-5.4-mini"
DEFAULT_MAX_SPEND_USD = 0.10
DEFAULT_TURNS = 2

# Cases ship with the benchmark; the smoke test only needs one.
DEFAULT_CASES = "data/sample_benchmark.json"


@dataclass
class LegUsage:
    """Token usage and cost for one model leg."""

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


class MeteredChatModel:
    """Pass-through wrapper recording usage of every response.

    Wraps a LangChain chat model without subclassing it: the agents call
    ``invoke``/``ainvoke`` and read ``.content``, so forwarding those two and
    delegating everything else is enough, and it avoids inheriting a pydantic
    model whose fields would reject an extra attribute.
    """

    def __init__(self, inner: Any, leg: LegUsage):
        self._inner = inner
        self._leg = leg

    def _record(self, response: Any) -> Any:
        self._leg.add(getattr(response, "usage_metadata", None))
        return response

    def invoke(self, *args, **kwargs):
        return self._record(self._inner.invoke(*args, **kwargs))

    async def ainvoke(self, *args, **kwargs):
        return self._record(await self._inner.ainvoke(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._inner, name)


@dataclass
class ToolTrace:
    """What the assistant actually did with the tools."""

    calls: List[Dict[str, Any]] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.calls) and not self.problems

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_calls": len(self.calls),
            "n_results": len(self.results),
            "tools_called": sorted({c["name"] for c in self.calls}),
            "problems": self.problems,
            "ok": self.ok,
        }


def inspect_tool_trace(conversation: Conversation, tool_names: List[str]) -> ToolTrace:
    """Extract and check the tool trace of a finished conversation.

    Well-formed means: at least one call; every call names a registered tool;
    every call carries a dict of arguments; every call is answered by a tool
    result with the matching id.
    """
    trace = ToolTrace()
    known = set(tool_names)
    answered: set = set()

    for message in conversation.messages:
        if isinstance(message, ToolMessage):
            trace.results.append({
                "tool_call_id": getattr(message, "tool_call_id", None),
                "name": getattr(message, "name", None),
            })
            answered.add(getattr(message, "tool_call_id", None))
            continue
        if not isinstance(message, AIMessage):
            continue
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name")
            args = call.get("args")
            trace.calls.append({"id": call.get("id"), "name": name})
            if name not in known:
                trace.problems.append(
                    f"call to {name!r}, which the sandbox did not register"
                )
            if not isinstance(args, dict):
                trace.problems.append(
                    f"call to {name!r} carried {type(args).__name__} arguments, not a dict"
                )

    if not trace.calls:
        trace.problems.append(
            "no tool calls in the transcript -- the assistant never invoked the "
            "sandbox, so workflow accuracy and triage cannot be scored"
        )
    for call in trace.calls:
        if call["id"] not in answered:
            trace.problems.append(f"call {call['name']!r} received no tool result")
    return trace


def _agent_spec(model_key: str, agent_class: str = "default") -> AgentSpec:
    return AgentSpec(model=ModelConfig(model=model_key), agent_class=agent_class)


def run_smoke(
    assistant_key: str = DEFAULT_ASSISTANT,
    patient_key: str = DEFAULT_PATIENT,
    sandbox_key: str = DEFAULT_SANDBOX,
    cases_file: str = DEFAULT_CASES,
    case_index: int = 0,
    turns: int = DEFAULT_TURNS,
    personality: str = "confused",
    max_spend_usd: float = DEFAULT_MAX_SPEND_USD,
) -> Dict[str, Any]:
    """Run one conversation and return the smoke-test record.

    Mirrors ``ConversationRunner._run_conversation``'s assembly order, metering
    each leg. Raises RuntimeError if the key is absent -- the caller decides
    whether that is a skip or a failure.
    """
    if not api_key_present():
        raise RuntimeError(
            f"{API_KEY_ENV} is not set; refusing to attempt a live call"
        )
    ensure_base_url()

    entries = load_benchmark_entries(cases_file)
    if not entries:
        raise RuntimeError(f"{cases_file} contains no benchmark entries")
    entry = entries[case_index]

    current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    legs = {
        "sandbox": LegUsage(model=sandbox_key),
        "assistant": LegUsage(model=assistant_key),
        "patient": LegUsage(model=patient_key),
    }

    # 1. Sandbox. Its client is ours to create, so it is metered directly --
    #    exactly the call the runner makes and discards.
    sandbox = HealthcareSandbox()
    sandbox_llm = MeteredChatModel(
        create_sandbox_llm(ModelConfig(model=sandbox_key), None), legs["sandbox"]
    )
    asyncio.run(
        initialize_sandbox(
            sandbox=sandbox,
            patient_profile=entry.patient_profile,
            llm_client=sandbox_llm,
            current_datetime=current_datetime,
        )
    )

    # 2. Tools, from the initialised sandbox.
    tool_registry = create_tool_registry(sandbox)
    tool_names = [t.name for t in tool_registry.get_all()]

    # 3. Assistant under test. Its usage rides on the messages it returns, so
    #    no wrapper is needed (and the react agent gives none to wrap).
    assistant = create_assistant_agent_from_spec(
        spec=_agent_spec(assistant_key),
        current_datetime=current_datetime,
        tool_registry=tool_registry,
    )

    # 4. Patient. Metered by wrapping the public .llm attribute after
    #    construction; DefaultUserAgent reads it on every turn.
    patient = create_user_agent_from_spec(
        spec=_agent_spec(patient_key, agent_class="pw_free_trait"),
        scenario=entry.scenario,
        user_profile=entry.patient_profile_xml,
        current_datetime=current_datetime,
        personality=personality,
    )
    patient.llm = MeteredChatModel(patient.llm, legs["patient"])

    conversation = Conversation()
    error: Optional[str] = None
    conversation.add_message(HumanMessage(content=patient.start_conversation()))

    seen_ai = 0
    for turn in range(turns):
        try:
            result = assistant.invoke(
                messages=conversation.messages,
                user_profile=entry.patient_profile_xml,
            )
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            error = f"{type(exc).__name__}: {exc}"
            break
        before = len(conversation)
        conversation.extend_from_agent_result(result["messages"])
        # Assistant usage: every new AIMessage carries its own usage_metadata.
        for message in conversation.messages[before:]:
            if isinstance(message, AIMessage):
                legs["assistant"].add(getattr(message, "usage_metadata", None))
                seen_ai += 1
        conversation.set_all_new_assistant_responses(from_index=before, strip_thinking=True)
        if turn < turns - 1:
            reply = patient.respond(conversation.get_all_new_assistant_text(before))
            conversation.add_message(HumanMessage(content=reply))

    trace = inspect_tool_trace(conversation, tool_names)
    costs = {name: leg.cost() for name, leg in legs.items()}
    known = [c for c in costs.values() if c is not None]
    total = round(sum(known), 6) if known else None

    # Roles routinely share a model (the sandbox and the assistant do by
    # default), so per_model sums across roles rather than keying on the last
    # one seen. per_role keeps the breakdown that makes the run diagnosable.
    merged: Dict[str, LegUsage] = {}
    for leg in legs.values():
        into = merged.setdefault(leg.model, LegUsage(model=leg.model))
        into.calls += leg.calls
        into.input_tokens += leg.input_tokens
        into.output_tokens += leg.output_tokens

    return {
        "task": "pab-toolcall-smoke",
        "run_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": assistant_key,
        "models": {"assistant": assistant_key, "patient": patient_key,
                   "sandbox": sandbox_key},
        "case_id": entry.id,
        "personality": personality,
        "turns_requested": turns,
        "messages": len(conversation),
        "assistant_messages": seen_ai,
        "tools_registered": len(tool_names),
        "tool_trace": trace.to_dict(),
        "error": error,
        "max_spend_usd": max_spend_usd,
        "cost_usd": total,
        "cost_basis": "registry list prices x reported token usage; not the "
                      "provider's invoice",
        "unpriced_legs": sorted(n for n, c in costs.items() if c is None),
        "usage": {
            "total_cost_usd": total,
            "per_model": {model: leg.to_dict() for model, leg in merged.items()},
            "per_role": {role: leg.to_dict() for role, leg in legs.items()},
        },
        "passed": trace.ok and error is None,
        "conversation": conversation.to_dicts(),
    }


def _plan(args) -> Dict[str, Any]:
    """What a live run would do, and what it would be allowed to spend."""
    return {
        "assistant": args.assistant,
        "patient": args.patient,
        "sandbox": args.sandbox,
        "turns": args.turns,
        "max_spend_usd": args.max_spend,
        "api_key_present": api_key_present(),
        "pricing": {
            key: get_model_pricing(key)
            for key in (args.assistant, args.patient, args.sandbox)
        },
    }


def write_report(record: Dict[str, Any], path: Path,
                 with_transcript: bool = False) -> Path:
    """Write the cost sidecar in the engine's established shape.

    The transcript is dropped by default: this file is destined for a public
    repository's ledger, and it only needs to carry the numbers.
    """
    payload = {k: v for k, v in record.items()
               if k != "conversation" or with_transcript}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--assistant", default=DEFAULT_ASSISTANT)
    parser.add_argument("--patient", default=DEFAULT_PATIENT)
    parser.add_argument("--sandbox", default=DEFAULT_SANDBOX)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    parser.add_argument("--personality", default="confused")
    parser.add_argument("--max-spend", type=float, default=DEFAULT_MAX_SPEND_USD)
    parser.add_argument("--report", help="write the cost sidecar here")
    parser.add_argument("--with-transcript", action="store_true",
                        help="keep the conversation in the sidecar (not for a public ledger)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without calling anything")
    args = parser.parse_args(argv)

    if args.dry_run:
        print(json.dumps(_plan(args), indent=2))
        return 0

    if not api_key_present():
        print(
            f"skip: {API_KEY_ENV} is not set. The tool-calling smoke test needs a "
            "live call and makes none without it; nothing was spent.",
            file=sys.stderr,
        )
        return 0

    record = run_smoke(
        assistant_key=args.assistant,
        patient_key=args.patient,
        sandbox_key=args.sandbox,
        cases_file=args.cases,
        case_index=args.case_index,
        turns=args.turns,
        personality=args.personality,
        max_spend_usd=args.max_spend,
    )

    if args.report:
        written = write_report(record, Path(args.report), args.with_transcript)
        print(f"report: {written}")

    trace = record["tool_trace"]
    cost = record["cost_usd"]
    print(f"assistant      : {record['model']}")
    print(f"messages       : {record['messages']}")
    print(f"tools registered: {record['tools_registered']}")
    print(f"tool calls     : {trace['n_calls']} -> {', '.join(trace['tools_called']) or 'none'}")
    print(f"tool results   : {trace['n_results']}")
    print(f"cost           : {'N/A (unpriced)' if cost is None else f'${cost:.6f}'} "
          f"of ${record['max_spend_usd']:.2f} allowed")
    if record["unpriced_legs"]:
        print(f"unpriced legs  : {', '.join(record['unpriced_legs'])}")
    for problem in trace["problems"]:
        print(f"PROBLEM        : {problem}")
    if record["error"]:
        print(f"ERROR          : {record['error']}")
    print(f"verdict        : {'PASS' if record['passed'] else 'FAIL'}")
    if cost is not None and cost > args.max_spend:
        print(f"WARNING        : spend exceeded --max-spend (${args.max_spend:.2f})")
    return 0 if record["passed"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
