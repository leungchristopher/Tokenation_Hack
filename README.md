# Epistemically aware experimental design

A minimal two-task benchmark, not a general agent framework:

```text
observe → update one GP → update the evidence graph → select → execute → append the audit
```

The default `bo_evidence` policy lets GP-BO select every experiment. A small LLM companion may
request targeted literature or add scoped interpretations; it cannot change acquisition, forecasts,
or final selection. Evidence errors are logged and never stop the optimiser. It makes at most three
model calls, triggered by initial retrieval, available sources, the first three observations, or a
predictive discrepancy.
There is no full candidate pool in its prompt and no JSON experiment decision to reject.

`bo` is the same numerical selector without model calls; `random` is the sanity baseline. The direct
`llm` selector remains an explicit experimental comparator, not the default. Literature does not
modify the numerical GP here; no optimisation benefit from these annotations is claimed. Earlier
gated and ReAct implementations remain in historical branches/PRs. There are no subagents,
persistent claim ledger, branch-closing proofs or paid deployment.

## Start without Inspect or credentials

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m epistemic run --task enzyme --policy bo_evidence --provider mock \
  --budget 8 --seed 0 --out logs/enzyme-demo
.venv/bin/python -m epistemic run --task drug --policy bo \
  --budget 8 --seed 0 --out logs/drug-demo
```

Use a fresh output directory: trajectories cannot be overwritten. Each run exports:

- `graph.html`: self-contained interactive evidence and experiment views; select a record for
  full source metadata, uncertainty snapshots and revisions, with matching IDs highlighted across views;
- `evidence.svg` and `experiments.svg`: separate printable graph views;
- `graph.svg`: both views in one SVG, with typed links and qualitative uncertainty badges;
- `audit.md` and `graph.json`: complete human-readable and typed evidence graphs;
- `trajectory.jsonl`: agent state, exact prompt, action, pre-experiment numerical prediction,
  accessible observation and state revision;
- `hidden_truth.jsonl`: **evaluator-only** realised settings, true outcomes and simulation metadata;
- `hidden_provenance.json`: **evaluator-only** dataset attribution withheld from the agent;
- `metrics.json`: evaluator-only objective/regret curves, error, coverage and cost;
- `config.json`: settings and provider/model/decode metadata.

Events are appended in memory then written once with exclusive creation; this is not a
crash-recoverable live event store. The mock deterministically hashes the visible prompt and
can choose any measured candidate. It is **plumbing only**, not scientific reasoning; its token
counts are character-based estimates. In `bo_evidence` it returns no scientific claims; GP-BO still
selects every condition.

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

The source paper **“Prediction of multidimensional drug dose responses based on measurements of
drug pairs”** is held out from agent evidence. Its title, DOI (`10.1073/pnas.1606301113`), PubMed ID
(`27562164`) and PMC ID (`PMC5027409`) are blocked in retrieval, raw tool results, fixed/frozen source
bundles and model-added evidence. Agent-facing task provenance omits both the source citation and
the archive citation; exact attribution is kept in the evaluator-only export. The human-readable
attribution above is not sent to the model. This prevents retrieval leakage, not knowledge already
present in a model's training.

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

Records are `evidence`, `uncertainty`, `observation`, `claim`, `assumption` and `decision`.
Raw evidence title, abstract, reference, query and nested metadata are immutable; claims are
separate, concise propositions. Immutable uncertainty snapshots distinguish source, transfer,
mechanistic, response, model and execution limitations without invented certainty probabilities.
Claim badges group response/model uncertainty and explicitly label unassessed categories.

Edges are `supports`, `qualifies`, `contradicts`, `not_transferable`, `depends_on` and `tests`,
with distinct labels and line styles. `tests` is reserved for decision → observation links.
References and globally unique IDs are checked. Revisions preserve statements, scope, criteria,
evidence and uncertainty IDs. Dependencies are declared inputs, not causal attribution.
The experiment view separates the intended experiment from its accessible observation and
shows dated claim updates; execution cards are display projections, not fabricated graph records.

The evidence companion receives recent observations, up to eight active claims, assumptions, model
diagnostics and the next GP-selected experiment. It may add/refine at most two scoped model
conjectures per call. Interpretations never become measurements. Unknown citations, attempts to
rewrite protected records, malformed output and provider failures are logged and ignored without
losing an experiment. The graph does not pretend these annotations caused GP choices.

The optional direct `llm` comparator receives the full feasible pool and may choose outside numerical
suggestions. Its invalid actions receive at most two retries, then stop with a logged rejection;
there is no silent BO replacement in that comparator.

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

`records_to_claims()` returns a graph-shaped bundle containing immutable source records,
concise propositions, explicit scope, source/transfer/mechanistic uncertainty, discriminating
experiments and typed links. Full abstracts appear only in source metadata and hover/inspector
details, not claim labels or the compact model context. Retraction/bibliographic metadata are
retained; applicability still needs review. Legacy curated claim lists remain readable; attach
raw records when available. Import remaps IDs safely and deduplicates attributable references.
The default fixed bundle records dataset provenance/limitations, not invented mechanistic findings.

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

`provider=inspect` uses Inspect's configured model; `provider=mock` uses the core mock. The
default is `policy=bo_evidence`, with the same GP/EI selector as `policy=bo`. For a capped real-model
smoke test, after approval and with credentials supplied through the environment:

```bash
inspect eval epistemic/inspect_task.py --model anthropic/claude-sonnet-4-6 \
  -T task_name=enzyme -T budget=8 -T max_searches=2 \
  --max-tokens 1000 --token-limit 50000 --time-limit 300 --max-retries 1 --retry-on-error 0
inspect eval epistemic/inspect_task.py --model anthropic/claude-sonnet-4-6 \
  -T task_name=drug -T budget=8 -T max_searches=2 \
  --max-tokens 1000 --token-limit 50000 --time-limit 300 --max-retries 1 --retry-on-error 0
```

Each task can use at most three evidence calls and two searches while the optimiser executes eight
experiments. Loop, execution, graph and scoring are shared. Normal core imports do not require Inspect.

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
