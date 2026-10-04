# GP-BO and reasoning graphs

One loop: select an experiment, observe its result, refit one GP, update the evidence graph.
Stop at the wall-clock or model-token limit (with an experiment safety cap), then submit the **best observed result**:
maximum for enzyme activity,
minimum for drug survival. Ties keep the earliest experiment; no observed result means no answer.
Also stop after `patience` (default 30) consecutive experiments without improving the best observed value (`stagnation`); this is not a proof of optimality.
Hidden dataset means and GP forecasts never determine the final submission.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m epistemic run --task enzyme --max-seconds 300 --max-model-tokens 12000 --out logs/current/enzyme
.venv/bin/python -m epistemic run --task drug --max-seconds 300 --max-model-tokens 12000 --out logs/current/drug
```

These are new numerical GP-BO runs against measured tables, not LLM or mock evaluations.
No credentials or network calls are needed. Use a new output directory for every run.

The first three experiments are a seeded initial design; subsequent experiments maximise
expected improvement under a Matérn GP. Without priors this is plain GP-BO.
The deadline is checked between numerical fits; model/retrieval request timeouts use the remaining time.
Before each model call, UTF-8 prompt bytes plus output allowance and framing reserve are checked against
the remaining token budget. Actual reported model usage is recorded. Numerical GP steps use no model tokens.

### Outputs

- `graph.html`: one compact Mermaid-style graph with pan/zoom and click-to-open records.
- `actions.svg` / `actions.json`: actions as nodes, reasons as edges, and explicit final selection.
- `graph.svg`: the compact graph; `evidence.svg` / `experiments.svg`: detailed printable exports.
- `final.json`: selected candidate, observed value, selection rule and uncertainty references.
- `graph.json`, `trajectory.jsonl`, `audit.md`, `config.json`: records and decision provenance.
- `termination.json`, `literature_setup.json`: stopping reason and initial evidence/prior provenance.
- `hidden_truth.jsonl`, `hidden_provenance.json`, `metrics.json`: evaluator-only diagnostics.

Best observed is not a proven optimum. Noise can misrank candidates. Reported delivery is not
the hidden realised condition, and GP intervals do not establish calibration. Evaluator metrics
separately record the best realised mean, final intended mean and regret; they are not model inputs.
Compact branches identify the best-observed EI reference, not causal ancestry or closed regions.
The GP uses all prior observations. Source and claim links appear only when recorded; unused
literature is explicitly labelled, not presented as influencing acquisition.

## Tasks and limitations

| Task | Candidates | Objective | Noise |
|---|---:|---|---|
| Enzyme UPO/ABTS | 814 distinct settings from 818 rows | Maximise recorded rate | Reported SD, unresolved replicate design |
| A549 three-drug | 512 settings, complete 8×8×8 grid | Minimise survival (%) | Estimated neighbour residual spread, not replicates |

Measurements use simulated Gaussian noise, clipped below at zero. Enzyme rate units are inherited
from the existing adapter, not verified upstream; several parameter units remain undocumented.
Duplicate enzyme conditions pool summary means and squared spreads, not raw replicates.
The drug task does not measure synergy.

Perfect execution is the default. `--execution perturbed` perturbs intended settings, clips to
measured bounds and maps to the nearest candidate. This is a finite-table simulation, not validated
instrument variability. Dataset contents are unchanged.

## Optional evidence

Immutable source records retain full titles, abstracts, references and queries. Separate concise
claims carry scope, source/transfer/mechanistic uncertainty and a discriminating experiment.
Edges distinguish `supports`, `qualifies`, `contradicts`, `not_transferable`, `depends_on`, and
`tests`; `tests` is exclusively decision → observation.

Default `provider=none` makes no model calls. A configured `openai:<model>` endpoint can annotate
evidence at most three times per episode. Malformed responses, invalid claim revisions and provider
errors cannot discard an experiment. `--max-searches` bounds targeted Amass retrieval; failed
requests consume the cap, and sources become available next round. Supply credentials through
environment secrets, never source control. A fixed bundle can be supplied with `--evidence-file`.
The source drug-response paper is filtered by normalised title, DOI, PubMed and PMC identifiers
before evidence reaches the model. Exact attribution stays evaluator-only. This cannot erase
pretrained knowledge.

`--acquisition gated_ei --max-searches 2` instead makes at most two initial model calls:
propose up to two numerical hypotheses, retrieve bounded sources, and calibrate their initial trust.
Each valid prior is a Gaussian basis in normalised parameter coordinates; one residual GP jointly
updates its signed gate mean and uncertainty from observations. Learned gates are not probabilities.
The same EI selector uses this posterior. There is no separate LLM controller or per-experiment model call.
Invalid proposals or failed retrieval fall back to plain GP-BO; active prior counts are recorded explicitly.

For an Anthropic-backed Inspect episode (credentials in environment secrets):

```bash
.venv/bin/inspect eval epistemic/inspect_task.py \
  --model anthropic/claude-sonnet-4-20250514 --max-tokens 1400 --max-retries 0 \
  -T task_name=enzyme -T provider=inspect -T acquisition=gated_ei -T max_searches=2 \
  -T max_seconds=300 -T max_model_tokens=12000
```

## Reuse

```python
from epistemic import Config, run_episode
from epistemic.render import export

episode = run_episode(Config(max_seconds=300), domain=(task_spec, evaluator))
export(episode, "logs/new-domain")
```

`TaskSpec` supplies public candidates, parameters, units, noise assumptions and objective direction.
`Evaluator` owns outcomes. New finite-candidate domains reuse the same GP, loop and graph.
`Config` module switches (`acquisition`, `with_evidence`, `with_edges`, `observation_noise`,
`execution`) support matched ablations without additional controllers or orchestration.
The evidence switch removes priors as well as annotations; edge visibility affects interpretation only.
Inspect is an optional adapter in `epistemic/inspect_task.py`, not a core dependency.

## Checks

```bash
.venv/bin/python -m pytest -q tests/test_epistemic.py -k 'not inspect'
.venv/bin/ruff check epistemic tests/test_epistemic.py
.venv/bin/mypy epistemic --check-untyped-defs
.venv/bin/python -m build --wheel --outdir logs/wheels
```

Small-data fits may reach kernel bounds; warnings do not establish reliable calibration.
Provider mocks are unit-test fixtures only. There are no alternative LLM controllers,
research agents, intervention pipelines or historical partial runs.
Robotics, scenes, models and all datasets are untouched.
