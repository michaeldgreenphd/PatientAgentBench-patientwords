# patientwords_pab — trait-sweep adapter (Layer 1)

A PatientWords addition to this PatientAgentBench fork. It lets the benchmark run
a **single-factor persona sweep**: hold the clinical case and six behavioural
traits fixed, vary exactly one, and read the rubric deltas.

## Why it exists

Upstream defines seven traits × three levels in `TRAIT_DEFINITIONS`, then bundles
them into six fixed presets, and `get_personality_prompt()` accepts only a preset
name. Every preset moves several traits at once, so a difference between two
presets cannot be attributed to any single trait. The data structure is already
factored the right way; only the public API is preset-locked. This package
supplies the missing API without changing upstream.

## Layer boundaries

| Layer | Where | Licence | Rule |
|---|---|---|---|
| 0 — upstream | `src/patient_agent_bench/`, `tests/` (upstream files), `pyproject.toml`, `data/` | CC-BY-NC-4.0 | **Never edited.** `git pull upstream main` stays a fast-forward. |
| 1 — adapter | `src/patientwords_pab/`, `tests/test_pw_*.py` | CC-BY-NC-4.0 (imports upstream ⇒ derivative) | Lives only in this fork. |
| 2 — analysis | `patientwords-engine` | MIT | Reads `conversations.json` / `evaluations.json` as data. Never imports this code. |

Layer 2 is the licence boundary: reading JSON output is not a derivative work,
so the interpretability engine stays MIT. Importing anything from this package
into that repo would spread NonCommercial terms across it.

Adding files is not editing files — the fork contains no modified upstream file.
To verify:

```bash
git diff --stat upstream/main -- src/patient_agent_bench tests pyproject.toml data
# (empty)
```

## Install

`src/` is placed on `sys.path` by the editable install, which is upstream's
supported install path, so no packaging change is needed:

```bash
pip install -e ".[dev]"
python -c "import patientwords_pab; print(patientwords_pab.FreeTraitUserAgent.NAME)"
```

A non-editable `pip install .` builds only `patient_agent_bench` (the wheel's
package list lives in upstream's `pyproject.toml`, which is Layer 0), so this
package would be absent. Use the editable install.

## Spec strings

The `personality` field travels from the benchmark JSON to the user agent as an
opaque string, so the arm is encoded there:

```
pw:health_literacy=low;clarity=high     # neutral base, two traits set
pw:base=confused;clarity=high           # a validated preset with one trait flipped
pw:base=neutral                         # the all-medium control arm
confused                                # an upstream preset — handled unchanged
```

* Prefix `pw:` marks a free-trait spec; anything else is a preset name.
* Unspecified traits come from the **base**. The default base is `neutral` —
  every trait at `medium` — not a preset, because a preset base would silently
  re-bundle the confounds the sweep exists to separate.
* Unknown trait, unknown level, unknown base, a repeated key, or a character that
  would break the rendered XML all raise `TraitSpecError` at construction, before
  any model call.

## Running a sweep

```jsonc
// benchmark entries: same case, one field apart
[
  {"scenario_id": "case-001-lo", "personality": "pw:health_literacy=low",    ...},
  {"scenario_id": "case-001-md", "personality": "pw:health_literacy=medium", ...},
  {"scenario_id": "case-001-hi", "personality": "pw:health_literacy=high",   ...}
]
```

```jsonc
// config: select the adapter
{"user_agent": {"model": {"model": "<registry-name>"}, "agent_class": "pw_free_trait"}}
```

```bash
python -m patientwords_pab.run benchmark --cases data/my_sweep.json --config data/my_config.json
```

Use `python -m patientwords_pab.run` rather than the `patient-agent-bench`
console script: the console script is upstream's entry point and does not import
this package, so `agent_class: "pw_free_trait"` would raise
`KeyError: Unknown user agent_class` — loudly, before any spend.

## OpenRouter models

Upstream's registry already supports a second access channel —
`provider="openai-protocol-api"` with `auth="api_key"`, `base_url_env` and
`api_key_env`, documented for a non-default OpenAI-compatible endpoint. OpenRouter
is that shape, so `openrouter_specs.py` adds entries rather than changing code:
`MODEL_STORE` is a plain dict and `get_model_spec()` reads it at call time, so the
specs are injected at import, the same pattern as `register_user_agent`.

Registry keys are the *engine's* names — `openrouter:<vendor>/<model>` — not
upstream's `<family>-<version>-api` convention. The measurement engine already
names OpenRouter models that way and writes that string into cost sidecars; one
name across both repos means the config, the transcript, and the ledger agree.

```jsonc
{"user_agent":      {"model": {"model": "openrouter:x-ai/grok-4.3"},
                     "agent_class": "pw_free_trait"},
 "assistant_agent": [{"model": {"model": "openrouter:openai/gpt-5.4-mini"}}]}
```

Set `OPENROUTER_API_KEY`. `OPENROUTER_BASE_URL` is set for you at import via
`setdefault`, so an operator-supplied value always wins — but it must be set
*somehow*: without `base_url_env`, `ChatOpenAI` silently falls back to
`api.openai.com`, where an OpenRouter key fails and an ambient `OPENAI_API_KEY`
would quietly succeed against the wrong vendor at the wrong price.

Prices are transcribed from an owner-verified in-repo source, recorded per spec in
`price_source` and carried into `ModelSpec.notes`. They are *ceiling-side* numbers
(list price plus roughly a 5% aggregator margin), so an estimate built on them is
an upper bound and the provider's invoice is the truth. A slug whose price cannot
be verified is registered with `None`, and upstream's own rule applies: **None →
unpriced; cost renders as N/A, never guessed.** Re-verify against OpenRouter's
live model list before the first fire.

## Tool-calling smoke test (the first thing to spend money on)

PatientAgentBench is agentic. The paper omits models "solely because they lacked
reliable native tool-calling for agentic workflows", so routing the assistant
through a new channel risks that capability before it risks anything else — and a
model that narrates tool use instead of emitting calls produces transcripts that
look fine and score meaningless. One conversation settles it:

```bash
python -m patientwords_pab.toolcall_smoke --dry-run          # plan only, $0
python -m patientwords_pab.toolcall_smoke \
    --assistant openrouter:openai/gpt-5.4-mini \
    --report out/pab_toolcall_smoke.report.json
```

It asserts the transcript holds at least one tool call, that every call names a
tool the sandbox registered, carries a dict of arguments, and is answered by a
matching result. Without `OPENROUTER_API_KEY` it skips and exits 0 — it never
half-runs. The sidecar is written in the measurement engine's established cost
shape (`run_timestamp`, `model`, `max_spend_usd`, `cost_usd`,
`usage.per_model{…}`) so that spend folds into the existing ledger, and it drops
the transcript unless `--with-transcript` is passed, because that file is destined
for a public repository.

The patient agent has no tools and carries none of this risk. Only the assistant
under test does.

## Dry run (free)

Renders exactly what a spec builds, calling no model:

```bash
python -m patientwords_pab.render_persona "pw:health_literacy=low;clarity=high"
python -m patientwords_pab.render_persona --sweep health_literacy
python -m patientwords_pab.render_persona --sweep communication --hold clarity=high --full-prompt
python -m patientwords_pab.render_persona "pw:base=confused;clarity=high" confused --json
```

Beyond the rendered block it reports two things worth knowing before committing
to a sweep:

* **nearest presets** — how many traits separate this arm from each upstream
  preset. Distance 0 means the arm reproduces a validated persona exactly.
* **unattested trait-level pairs** — pairs of trait levels that no upstream
  preset ever combines, with the ones this spec introduced marked `*`. An arm
  with none is a recombination of validated ground; an arm with some is a
  persona upstream never checked, and results from it carry that caveat.

For a `--sweep`, the last block is the identification check: exactly one trait
may vary across the arms. It exits non-zero if more than one does.

## Upstream surfaces this depends on

Pinned by `tests/test_pw_upstream_pins.py`, which is the alarm if upstream moves:

| Surface | Used for | If it moves |
|---|---|---|
| `personalities.TRAIT_DEFINITIONS` | the trait descriptions the block is built from | loud failure |
| `default_prompt.SYSTEM_PROMPT` containing `{personality_traits}` | the slot the block is substituted into | **silent** — `format_prompt_safe` only logs a warning for a keyword with no matching placeholder, so every arm would run persona-free and the sweep would measure nothing |
| `registry.register_user_agent(name, cls)` | registering without editing `registry.py` | loud failure |
| `DefaultUserAgent.__init__` keywords and `self.system_prompt_template` | the subclass forwards to it, then re-renders | loud failure |
| `get_personality_prompt()` raising `KeyError` on a non-preset | the reason a placeholder preset is passed up before the prompt is rebuilt | placeholder becomes unnecessary but stays harmless |
| `PERSONALITY_TYPES` | preset bases, and the novelty comparison | loud failure |
