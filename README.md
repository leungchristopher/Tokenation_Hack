# Epistemically aware experimental design

A minimal two-task benchmark, not a general agent framework:

```text
observe → update one GP → update the evidence graph → select → execute → append the audit
```

Three policies share that loop: `random`, ordinary `bo` (Matérn GP + expected improvement), and
`llm` (structured decisions using the same numerical summaries and evidence state). Literature
does not modify the numerical GP here. Earlier gated, ReAct and tool-layer implementations remain
in historical branches/PRs, not as competing active paths. There are no subagents, persistent claim
ledger, branch-closing proofs or paid deployment.

## Start without Inspect or credentials

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m epistemic run --task enzyme --policy llm --provider mock \
  --budget 8 --seed 0 --out logs/enzyme-demo
.venv/bin/python -m epistemic run --task drug --policy bo \
  --budget 8 --seed 0 --out logs/drug-demo
```

Use a fresh output directory: trajectories cannot be overwritten. Each run exports:

- `graph.svg`: standalone objective curve, decisions, evidence, contradictions and revisions;
  hover records for uncertainty, scope, alternatives and delivery reports;
- `audit.md` and `graph.json`: complete human-readable and typed evidence graphs;
- `trajectory.jsonl`: agent state, exact prompt, action, pre-experiment numerical prediction,
  accessible observation and state revision;
- `hidden_truth.jsonl`: **evaluator-only** realised settings, true outcomes and simulation metadata;
- `metrics.json`: evaluator-only objective/regret curves, error, coverage and cost;
- `config.json`: settings and provider/model/decode metadata.

Events are appended in memory then written once with exclusive creation; this is not a
crash-recoverable live event store. The mock deterministically hashes the visible prompt and
can choose any measured candidate. It is **plumbing only**, not scientific reasoning; its token
counts are character-based estimates.

The direct API uses the same engine:

```python
from epistemic import Config, run_episode
from epistemic.render import export

episode = run_episode(Config(task="drug", policy="bo", budget=8, seed=0))
export(episode, "logs/python-api-demo")
```

## Verified tasks and limitations

| Task | Feasible set | Objective | Noise/provenance |
|---|---|---|---|
| `enzyme` | 818 source rows, 814 distinct measured UPO/ABTS settings; sparse, not an interpolated Cartesian environment | Maximise recorded specific rate; U/mg is inherited from the old adapter, not verified upstream | Reported SD; raw replicate design and collection protocol are undocumented |
| `drug` | Exactly 512 distinct combinations, complete 8×8×8 taxol × cisplatin × doxorubicin grid in µM | Minimise A549 survival (%) | Existing converter cites Zimmer/Tendler archive; constant SD is a MAD estimate from grid-neighbour residuals, **not replicate variability** |

The existing drug adapter labels the assay as 48-hour A549 survival. The source paper is
[Zimmer et al. (2016)](https://doi.org/10.1073/pnas.1606301113); conversion details remain in
`data/zimmer/convert.py`. This work verifies the local table, not every upstream protocol or
raw archive conversion. There is **no synergy objective**: this table does not supply the necessary
single-agent/vehicle controls or a reference definition.

Enzyme salt, cosubstrate, solvent and temperature units are not established by this CSV.
Four repeated parameter conditions have differing summary outcomes. As in the original adapter,
their recorded means and squared spreads are averaged. They are **not counted as raw
biological/technical replicates**; this pooling does not capture between-summary variance.
No dataset has been changed.

Both observation noise and execution faults are explicitly **simulated**. Observation draws
are Gaussian using reported/estimated spread and lower-clipped at zero; the enzyme spread's
experimental interpretation is unresolved. Perturbed execution applies additive noise in
pH/temperature coordinates and at zero-valued settings, otherwise multiplicative positive noise,
clips to measured bounds, then snaps in scaled/log-scaled coordinates to the nearest measured
candidate. This is a finite-table approximation, not validated instrument variability.
The agent sees a noisy delivery report and clipping flags, not the hidden realised ID.
Perfect execution is the control.

## State and graph

`TaskSpec` contains public candidates and metadata only. `Evaluator` owns outcome labels.
Unqueried means, SDs, optimum and full outcome tables are never sent to the model.

The four uncertainties remain separate:

1. **Response:** GP mean, latent SD, fitted observation-noise SD and predictive interval.
2. **Execution:** intention, delivery report, report SD and clipping flags.
3. **Model:** assumptions, prequential error/coverage and unresolved explanations
   (execution error, observation noise or misspecification).
4. **Evidence:** source, scope, supporting/contradicting IDs, qualitative status, transfer
   assumptions and discriminating observable result. No arbitrary certainty probability.

Only `observation`, `claim`, `assumption`, `decision` records and
`supports`, `contradicts`, `depends_on`, `tests` edges exist. Observations and nested tool metadata
are immutable through the API. References are checked. Revisions preserve prior statements,
scope, criteria and evidence; dependencies are not causal attribution.

The LLM receives recent observations, up to eight active claims, assumptions and the **full**
feasible pool. Numerical suggestions are not its only choices. It may repeat a setting (using
budget), cite existing records and add/refine at most two scoped model conjectures per decision.
Interpretations never become measured facts. Invalid JSON, unknown references, non-finite
predictions and unknown candidates receive at most two retries; then the episode stops with
a logged rejection—**no silent BO replacement**.

LLM point forecasts are labelled self-reported, not statistical response uncertainty. Numerical
pre-experiment means/intervals are computed independently by the GP and saved separately.

## Feedback and graph controls

```bash
# Frozen prefix; true/removed/permuted/influential-result-contradicted arms.
# Both total-system and direct-LLM-only modes, plus true-vs-true decoding controls.
.venv/bin/python -m epistemic feedback --task drug --policy llm --provider mock \
  --budget 8 --prefix 4 --repeats 2 --mode both --out logs/feedback-demo

# Equal-budget true/missing/corrupted closed loops; seeded fault on round 3.
.venv/bin/python -m epistemic closed-loop --task drug --policy llm --budget 8 \
  --seeds 0 --fault-round 3 --out logs/closed-loop-demo

# Small offline 2 tasks × 2 execution modes × 2 evidence conditions × edges/flat.
.venv/bin/python -m epistemic graph-control --provider mock --budget 3 \
  --seeds 0 --out logs/graph-control-demo
```

Every paired arm rebuilds model fits, diagnostics and observation-derived claims from modified
history. Intended settings, delivery reports, source evidence, declared budget and decoding setup
are fixed. Old LLM interpretations are discarded **in all arms**, preventing stale true-feedback
reasoning; this tests fresh re-conditioned choices, not persistence of old rhetoric.
Direct mode has no numerical advice or GP-derived calibration claim.
Sensitivity uses action identity and physical/scaled parameter distances, not changed explanations;
true-vs-true decoding variation is separate.

Closed-loop histories diverge after choices diverge: performance there is not a frozen-history
causal test. Scoring uses realised evaluator truth, not displayed feedback. Simple regret is
the absolute objective gap divided by `max(abs(optimum), 1e-9)`. Final selection is the intended
setting with best **shown** outcome (or the last decision if all feedback is missing);
its evaluator regret is reported separately.
Recovery is the first *new* post-fault trial with true regret within 0.25 of the pre-fault best
regret. No recovery within budget is `null`, not success at the horizon.

Graph/flat conditions retain identical records and uncertainty summaries, omitting explicit edges
only. The common candidate payload approximately matches prompt sizes; actual characters are
logged, not assumed token equality. Initial sources/seeds match, but subsequent histories can
diverge. Misleading evidence is an unverified hypothesis, not a fake publication; construction
is marked in evaluator metadata. Changed choices are not necessarily better choices.
**No improvement claim follows from mocks.**

## Models, existing retrieval, and the first real-model test

```bash
.venv/bin/python -m pip install -e '.[model]'
# Supply EPISTEMIC_API_KEY through your secret manager, not source control.
# Set EPISTEMIC_BASE_URL to your existing OpenAI-compatible endpoint (include /v1).
# Set EPISTEMIC_MODEL, EPISTEMIC_MODEL_REVISION and EPISTEMIC_INFERENCE_STACK.
# Set EPISTEMIC_SEED_SUPPORTED=true only if the endpoint actually supports seed.
```

Existing hosted open-weight/vLLM-compatible services work through this interface. It does not
provision Modal or paid infrastructure. Record actual model revision/quantisation and stack in
the environment metadata. Unsupported seeds are omitted; supported seeds still do not guarantee
determinism. Server fingerprint, temperature, prompt version, calls, usage and latency are logged
when available. Anonymous local servers may use a non-secret dummy key.

**Exact first real-model feedback command**, after endpoint/model selection and approval:

```bash
.venv/bin/python -m epistemic feedback --task drug --policy llm \
  --provider "openai:${EPISTEMIC_MODEL}" --budget 8 --prefix 4 --repeats 2 \
  --mode both --seed 0 --temperature 0 --out logs/first-real-feedback
```

This is 24 successful generations (shared prefix, both modes, decoding controls), plus bounded
retries if needed. **It has not been run.** Start here, not with a real-model sweep.

The existing Amass backend is retained without bibliometric “trust probabilities”:

```bash
.venv/bin/python -m pip install -e '.[literature]'
# Requires AMASS_API_KEY; explicit retrieval makes external requests.
.venv/bin/python -m epistemic literature --query "your task-specific question" \
  --out logs/source-bundle
.venv/bin/python -m epistemic run --task drug --policy llm --provider mock \
  --evidence-file logs/source-bundle/evidence.json --out logs/evidence-demo
```

Alternatively provide a curated JSON list of `Claim` records with attributable sources.
Retraction/bibliographic metadata are retained; applicability still needs review. The default
fixed bundle records dataset provenance/limitations, not invented mechanistic findings.

Live retrieval is optional: pass `--max-searches 2` to `run`, or `-T max_searches=2` to Inspect.
An LLM decision may request one focused `search_query`; at most three records are returned per
request, sources are added to the graph after execution and become available **next round**.
There is no researcher agent or extra decision generation. Failed requests consume budget and
are logged without fabricated sources; last-round requests are not executed. Trajectories retain
queries, raw returned records, attributable claims, credit-cost headers when provided, and errors.
Paired feedback decisions freeze the retrieved sources across arms. Graph/flat comparisons
require fixed evidence bundles rather than independently changing live searches.

## Optional Inspect wrapper

```bash
.venv/bin/python -m pip install -e '.[inspect]'
inspect eval epistemic/inspect_task.py --model mockllm/model \
  -T task_name=drug -T policy=bo -T budget=3
```

`provider=inspect` uses Inspect's configured model for LLM calls; `provider=mock` uses the core
mock. Loop, execution, graph and scoring are shared. Normal core imports do not require Inspect.

## Checks and portability

```bash
.venv/bin/ruff check epistemic tests/test_epistemic.py
.venv/bin/mypy epistemic --check-untyped-defs
.venv/bin/python -m pytest -q tests/test_epistemic.py
.venv/bin/python -m build --wheel --outdir logs/wheels
```

Small-data GP fits can hit kernel bounds and emit convergence warnings. These are not evidence
of reliable calibration; prequential diagnostics remain exposed.

New domains would replace the task factory/evaluator and implement the narrow `Executor`
protocol, not a new orchestration framework. Only two adapters are registered now.
`robot/`, `scenes/`, `models/`, robotics requirements and all datasets are untouched.
