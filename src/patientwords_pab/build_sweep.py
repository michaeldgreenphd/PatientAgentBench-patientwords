# SPDX-License-Identifier: CC-BY-NC-4.0
"""Build the benchmark file for a single-factor sweep, offline and free.

A sweep is *one clinical case per arm*, identical in every respect except the
``personality`` field. Upstream's benchmark format has no notion of that -- each
entry is an independent case -- so the pairing has to be constructed, and
constructed exactly, because the whole design rests on it: if any other
attribute differs between two arms of the same case, the contrast is confounded
and the measurement engine's Layer-2 validator will (correctly) refuse the run.

This copies each selected case once per arm, changes only ``personality``,
and mints a derived ``scenario_id``. Everything else -- the story, the profile,
the condition, severity, task type -- is carried through byte-identical.

    # the four-arm literacy sweep over 13 cases, from the shipped sample
    python -m patientwords_pab.build_sweep \\
        --cases data/sample_benchmark.json --n 13 \\
        --sweep health_literacy --base confused --anchor-preset confused \\
        --out data/sweep_health_literacy.json

    # arbitrary arms, explicitly
    python -m patientwords_pab.build_sweep --n 8 \\
        --arm confused --arm "pw:base=confused;health_literacy=high" \\
        --out data/sweep_pair.json

Every emitted file is verified before it is written: it must load through
upstream's own ``load_benchmark_entries``, every case group must carry one entry
per arm, and every non-personality field must match across a group. A file that
fails those checks is not written at all.

$0 and offline. No model is constructed and no network call is made.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from patient_agent_bench.benchmark_seed import load_benchmark_entries
from patient_agent_bench.user_agent.personalities import PERSONALITY_TYPES
from patientwords_pab.trait_spec import (
    NEUTRAL_BASE,
    TraitSpecError,
    is_free_trait_spec,
    parse_trait_spec,
    single_factor_arms,
)

# Fields that identify the arm rather than the case. Everything else must be
# identical across a case's arms, and is asserted to be.
ARM_FIELDS = ("personality", "scenario_id", "query_id")

DEFAULT_CASES = "data/sample_benchmark.json"


class SweepError(ValueError):
    """The requested sweep cannot be built, or the built one is not valid."""


def arm_slug(arm: str, index: int) -> str:
    """Short, id-safe tag for an arm, used to derive scenario ids.

    Positional (``arm0``, ``arm1``) rather than derived from the label: labels
    contain ``:``, ``;`` and ``=``, and an id is a filename component often
    enough that smuggling those through is asking for trouble.
    """
    return f"arm{index}"


def resolve_arms(
    arms: Optional[List[str]] = None,
    sweep: Optional[str] = None,
    base: str = NEUTRAL_BASE,
    anchor_preset: Optional[str] = None,
    hold: Optional[Dict[str, str]] = None,
) -> List[str]:
    """The arm labels for a sweep, in order.

    ``anchor_preset`` puts an upstream preset first. That arm renders through
    upstream's own function, so it doubles as an external anchor (its score is
    comparable to published preset numbers) and, against the adapter's
    equivalent arm, isolates the effect of the persona *label* itself.
    """
    labels: List[str] = []
    if anchor_preset:
        if anchor_preset not in PERSONALITY_TYPES:
            raise SweepError(
                f"unknown preset {anchor_preset!r}. "
                f"Available: {', '.join(sorted(PERSONALITY_TYPES))}"
            )
        labels.append(anchor_preset)
    if sweep:
        labels.extend(
            spec.canonical for spec in single_factor_arms(sweep, base=base, hold=hold)
        )
    labels.extend(arms or [])

    if len(labels) < 2:
        raise SweepError("a sweep needs at least two arms")
    if len(set(labels)) != len(labels):
        raise SweepError(f"duplicate arm labels: {labels}")
    for label in labels:
        if is_free_trait_spec(label):
            parse_trait_spec(label)  # validates, raises TraitSpecError
        elif label not in PERSONALITY_TYPES:
            raise SweepError(
                f"arm {label!r} is neither a free-trait spec nor an upstream preset"
            )
    return labels


def build_entries(cases: List[Dict[str, Any]], arms: List[str]) -> List[Dict[str, Any]]:
    """One entry per (case, arm), grouped by case so runs read in pair order."""
    entries: List[Dict[str, Any]] = []
    for case in cases:
        source_id = case.get("scenario_id") or case.get("query_id") or ""
        if not source_id:
            raise SweepError("a source case has no scenario_id/query_id to derive from")
        for index, arm in enumerate(arms):
            entry = dict(case)
            entry.pop("query_id", None)  # collapse the legacy id onto scenario_id
            entry["scenario_id"] = f"{source_id}--{arm_slug(arm, index)}"
            entry["personality"] = arm
            entries.append(entry)
    return entries


def verify(entries: List[Dict[str, Any]], arms: List[str], n_cases: int) -> Dict[str, Any]:
    """Check the built sweep before anything is written.

    Enforces exactly what the measurement engine's Layer-2 validator enforces on
    the resulting transcripts, so a run cannot be started against a benchmark
    that is already confounded.
    """
    expected = n_cases * len(arms)
    if len(entries) != expected:
        raise SweepError(f"expected {expected} entries, built {len(entries)}")

    ids = [e["scenario_id"] for e in entries]
    if len(set(ids)) != len(ids):
        raise SweepError("duplicate scenario_id in the built sweep")

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(entry["scenario_id"].rsplit("--", 1)[0], []).append(entry)

    for case_id, group in sorted(groups.items()):
        labels = [e["personality"] for e in group]
        if labels != arms:
            raise SweepError(
                f"case {case_id} carries arms {labels}, expected {arms}"
            )
        reference = group[0]
        for entry in group[1:]:
            for field in set(reference) | set(entry):
                if field in ARM_FIELDS:
                    continue
                if reference.get(field) != entry.get(field):
                    raise SweepError(
                        f"case {case_id}: field {field!r} differs between arms "
                        f"{reference['personality']!r} and {entry['personality']!r}; "
                        "the arms would not be the same clinical case"
                    )
    return {"cases": len(groups), "arms": len(arms), "entries": len(entries)}


def build(cases_file: str, n_cases: int, arms: List[str],
          case_offset: int = 0) -> tuple:
    """Build and verify a sweep. Returns ``(entries, summary)``."""
    try:
        raw = json.loads(Path(cases_file).read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise SweepError(f"cannot read {cases_file}: {err}") from err
    if not isinstance(raw, list) or not raw:
        raise SweepError(f"{cases_file} is not a non-empty list of benchmark entries")

    selected = raw[case_offset:case_offset + n_cases]
    if len(selected) < n_cases:
        raise SweepError(
            f"{cases_file} has {len(raw)} cases; cannot take {n_cases} from "
            f"offset {case_offset}. Generate more with `patient-agent-bench "
            "generate-seeds` (which costs money) or lower --n."
        )

    entries = build_entries(selected, arms)
    summary = verify(entries, arms, len(selected))
    return entries, summary


def write(entries: List[Dict[str, Any]], out_path: Path) -> Path:
    """Write the sweep, then load it back through upstream's own reader.

    A file that upstream cannot parse is worse than no file: the failure would
    surface partway through a paid run.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    loaded = load_benchmark_entries(str(out_path))
    if len(loaded) != len(entries):
        out_path.unlink()
        raise SweepError(
            f"upstream read back {len(loaded)} of {len(entries)} entries; "
            "file removed"
        )
    for entry, parsed in zip(entries, loaded, strict=True):
        if parsed.personality != entry["personality"]:
            out_path.unlink()
            raise SweepError(
                f"upstream parsed personality {parsed.personality!r} for "
                f"{entry['scenario_id']}; file removed"
            )
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", default=DEFAULT_CASES,
                        help=f"source benchmark JSON (default: {DEFAULT_CASES})")
    parser.add_argument("--n", type=int, required=True, help="cases per arm")
    parser.add_argument("--case-offset", type=int, default=0,
                        help="skip this many source cases first")
    parser.add_argument("--sweep", help="trait to sweep across its levels")
    parser.add_argument("--base", default=NEUTRAL_BASE,
                        help=f"base for --sweep arms (default: {NEUTRAL_BASE})")
    parser.add_argument("--hold", action="append", metavar="TRAIT=LEVEL",
                        help="hold an extra trait fixed across --sweep arms")
    parser.add_argument("--anchor-preset",
                        help="prepend an upstream preset arm as an external anchor")
    parser.add_argument("--arm", action="append", dest="arms",
                        help="explicit arm label (repeatable)")
    parser.add_argument("--out", help="write here; omit to print a summary only")
    args = parser.parse_args(argv)

    hold: Dict[str, str] = {}
    for item in args.hold or []:
        if "=" not in item:
            print(f"error: malformed --hold {item!r} (expected trait=level)",
                  file=sys.stderr)
            return 2
        trait, _, level = item.partition("=")
        hold[trait.strip()] = level.strip()

    try:
        arms = resolve_arms(args.arms, args.sweep, args.base, args.anchor_preset, hold)
        entries, summary = build(args.cases, args.n, arms, args.case_offset)
        if args.out:
            written = write(entries, Path(args.out))
            summary["path"] = str(written)
    except (SweepError, TraitSpecError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    print(f"{summary['cases']} cases x {summary['arms']} arms = "
          f"{summary['entries']} benchmark entries")
    for index, arm in enumerate(arms):
        print(f"  {arm_slug(arm, index)}  {arm}")
    if "path" in summary:
        print(f"written: {summary['path']} (verified through upstream's reader)")
    else:
        print("dry run: nothing written (pass --out to write)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
