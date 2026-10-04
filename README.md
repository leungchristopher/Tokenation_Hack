# GP-EI experimental design

Finite-candidate Bayesian optimisation for enzyme activity and drug survival. One Gaussian process selects experiments with expected improvement; optional `gated_ei` learns literature-informed prior weights. The final answer is the greedy best observed result, not a proof of optimality.

## Tasks

| Task | Objective | Observation noise |
|---|---|---|
| `enzyme` | Maximise recorded enzyme rate | Dataset-reported spread; replicate design is undocumented |
| `drug` | Minimise A549 survival | 5% relative CV by default; a modelling choice because the dataset has no replicates |

Drug CV can be changed with `--noise-cv` or `Config(noise_cv=...)`. Both tasks use simulated Gaussian measurement noise.

## Run

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m epistemic run --task drug --budget 20 --out logs/drug
.venv/bin/python -m epistemic run --task enzyme --budget 20 --out logs/enzyme
.venv/bin/python -m epistemic run --task drug --acquisition gated_ei --provider openai:<model> --max-searches 2
```

Runs stop at the time, token, experiment-budget, or stagnation limit. Stagnation means 30 consecutive experiments without improving the best observed result by default; it is not evidence of optimality. `gated_ei` performs bounded literature setup before the first experiment. The default `provider=none` makes no model calls.

Inspect usage:

```bash
.venv/bin/pip install -e '.[inspect]'
.venv/bin/inspect eval epistemic/inspect_task.py --model anthropic/claude-sonnet-4-20250514 \
  -T task_name=drug -T provider=inspect -T acquisition=gated_ei -T max_searches=2
```

## Domain adapter

Pass a `(TaskSpec, Evaluator)` pair to `run_episode(config, domain=(task, evaluator))`. `TaskSpec` supplies candidate parameters, objective direction, and noise assumptions; `Evaluator` owns hidden outcomes.

## Outputs

`export` writes `graph.html`, `graph.svg`, `actions.json`, and `metrics.json`, plus the `Episode.save` files: `trajectory.jsonl`, `hidden_truth.jsonl`, `graph.json`, `final.json`, `config.json`, `termination.json`, `literature_setup.json`, and `hidden_provenance.json`. Hidden truth and provenance are evaluator-only artifacts.
