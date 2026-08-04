# pw_pilot — runbook for the literacy sweep

Everything a live pilot needs, pre-built and verified offline. Costed in
`patientwords-engine/docs/pab_first_probe_costing.md`.

**Licence.** The sweep files are copies of upstream's `data/sample_benchmark.json`
cases with one field changed, so they are derivative of CC-BY-NC-4.0 data and stay
in this fork. Nothing here goes to the measurement engine except run *output*.

## Files

| file | shape | for |
|---|---|---|
| `sweep_health_literacy_n8.json` | 4 arms × 8 cases = 32 | Stage 1, the manipulation check |
| `sweep_health_literacy_n13.json` | 4 arms × 13 cases = 52 | Stage 2, full outcome pilot |
| `sweep_health_literacy_2arm_n13.json` | 2 arms × 13 cases = 26 | Stage 2-lite, low vs high only |
| `sweep_health_literacy_2arm_n3.json` | 2 arms × 3 cases = 6 | the cross-model run |
| `config_pilot.json` | — | models for every role, single assistant |
| `config_paper10.json` | — | the ten models PatientAgentBench's paper evaluates |

The arms, in order:

```
arm0  confused                                    upstream preset, rendered by upstream
arm1  pw:base=confused;health_literacy=low        the same seven traits, adapter-rendered
arm2  pw:base=confused;health_literacy=medium     swept
arm3  pw:base=confused;health_literacy=high       swept
```

Rebuild any of them (deterministic, $0):

```bash
python -m patientwords_pab.build_sweep --n 13 \
    --sweep health_literacy --base confused --anchor-preset confused \
    --out pw_pilot/sweep_health_literacy_n13.json
```

## Order of operations

```bash
export OPENROUTER_API_KEY=...      # patient, assistant, sandbox
export ANTHROPIC_API_KEY=...       # jury only, and only from Stage 2

# 0. tool-calling smoke test — ~$0.03, and the gate on everything below
python -m patientwords_pab.toolcall_smoke --dry-run          # free, prints the plan
python -m patientwords_pab.toolcall_smoke \
    --report ../patientwords-engine/data/pab/toolcall_smoke.report.json

# 1. manipulation check — ~$1.19, generation only, no jury, no ANTHROPIC_API_KEY needed
python -m patientwords_pab.run generate \
    --cases pw_pilot/sweep_health_literacy_n8.json \
    --config pw_pilot/config_pilot.json

# 2. outcome pilot — scores an existing run directory; conversations are reused
python -m patientwords_pab.run benchmark \
    --cases pw_pilot/sweep_health_literacy_n13.json \
    --config pw_pilot/config_pilot.json \
    --run-dir output/<the stage 1 run dir>
```

Then, in the measurement engine:

```bash
python scripts/validate_pab_contract.py --run <run-dir>   # before any analysis
python scripts/ledger_update.py                           # folds data/pab/*.report.json
```

**Use `python -m patientwords_pab.run`, not `patient-agent-bench`.** The console
script is upstream's entry point and does not import this package, so
`agent_class: "pw_free_trait"` raises `KeyError` — loudly, before any spend, which
is the desired failure but not the desired outcome.

## What is already known good

`tests/test_pw_sweep_rehearsal.py` runs this exact config and stimulus set through
the real `ExperimentRunner`, `ConversationRunner` and `OutputManager` with every
model mocked, and checks the resulting run directory against the same invariants
the engine's Layer-2 validator enforces. So the plumbing — registry resolution,
agent selection, arm labels reaching the transcript, tool calls surviving,
evaluations joining slot-for-slot, arms staying balanced — is tested. The only
untested thing left is whether the models themselves behave, which is what
Stage 0 buys.

One finding from that rehearsal worth knowing before reading any output:
`initialize_sandbox()` attaches a generated PCP to the patient profile *before* the
transcript records it, so `user_profile` differs between arms of the same case and
between runs. The scenario does not. Anything joining arms must key on the
scenario; the engine's contract file does.


## The cross-model run

`config_paper10.json` runs the sweep across the ten models in the
PatientAgentBench paper's Table 4, so their published aggregate / triage numbers
are an external reference point for whatever the literacy manipulation does.
Each assistant label carries those three numbers.

Patient (`x-ai/grok-4.3`) and sandbox (`gpt-5.4-mini`) are held **fixed**: the
assistant is the only thing that varies, and the patient leg *is* the
manipulation, so it is deliberately not the leg to cheapen.

Assistants are listed **most expensive first**. Upstream runs them in list order,
generation is sequential (`--max-parallel` stays at 1), and the spend guard
aborts hard on the ceiling — so a trip loses a known tail of cheap models, which
are also the cheap ones to re-run. `generate --run-dir <dir>` resumes and skips
what already finished.

`sweep_health_literacy_2arm_n3.json` is a subset of the 2-arm n13 file, cases
unmodified, one paired case per severity level (mild / moderate / severe) —
severity being the dimension the paper's triage scores separate models on:

| case | severity | task |
|---|---|---|
| `301ed132` | severe | mental health crisis — highest-stakes downgrade probe |
| `ae13749a` | moderate | emergency symptoms — classic triage test |
| `f9029aec` | mild | provider access / scheduling — workflow accuracy; also the case the tool-calling smoke test ran end to end |

Sizing, from the smoke test's measured 3-turn token mix: 10 models × 6
conversations ≈ **$3.60** against a **$4.00** ceiling.

## The spend ceiling is enforced, not declared

`max_spend` in the CI trigger is bookkeeping — it is counted by the fire guard
and folded into the ledger, but on its own nothing stops a run. Setting
`PW_MAX_SPEND_USD` installs `patientwords_pab.budget_guard`, which meters every
call through upstream's single model factory and aborts when the cumulative cost
crosses the limit:

```bash
PW_MAX_SPEND_USD=4.00 \
PW_SPEND_REPORT=../data/pab/run.report.json \
python -m patientwords_pab.run generate \
    --cases pw_pilot/sweep_health_literacy_2arm_n3.json \
    --config pw_pilot/config_paper10.json
```

Three checks, earliest first: every model in the config must be priced (an
unpriced leg cannot be metered, so the run refuses to start); the running total
is checked before and after every call; and the report is written in a `finally`
so an aborted or crashed run still leaves a sidecar for the ledger.

Upstream registers its direct-API keys (`claude-opus-4.8-api` and friends)
*unpriced*, which would make the Anthropic jury unmeterable.
`direct_api_specs.py` registers priced `pw:`-prefixed variants alongside them;
upstream's own entries are never modified.
