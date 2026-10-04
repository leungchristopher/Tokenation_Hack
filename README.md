# Tokenation_Hack

Inspect eval for LLM-driven experimental optimisation, using a Bayesian-optimisation tool and a legible reasoning graph.

- **Environment** (`bo_eval/env.py`): `upo_abts` uses `data/upo_abts.csv`. Each experiment snaps to the nearest measured condition and returns a draw from N(mean rate, SD).
- **Tools** (`bo_eval/tools/`): `run_experiment`, `bayes_opt_suggest` (GP + EI), `add_reasoning`, `close_branch` (the node and its descendants are closed and can't be extended), `view_graph`, `submit`.
=======
- **Environments** (`bo_eval/env.py`): each task is a CSV of measured conditions (parameters, mean, SD) plus one entry in `ENVS`. Choose one with `-T env=<name>`. Each experiment snaps to the nearest measured condition and returns a draw from N(mean, SD).
  - `upo_abts`: UPO specific rate (`data/upo_abts.csv`).
  - `icfree_cole1`, `icfree_colm`, `icfree_colm_eu`: cell-free colicin yield from the active-learning data of [Borkowski et al., iScience 2025](https://doi.org/10.1016/j.isci.2025.113599) ([Zenodo](https://doi.org/10.5281/zenodo.14904992)). `data/icfree/convert.py` converts Echo transfer volumes to final concentrations with Table S1 (conc = max conc × volume / max volume) and pools the replicate yields.
- **Tools** (`bo_eval/tools/`): `run_experiment`, `bayes_opt_suggest` (GP + EI), `add_reasoning`, `close_branch` (Hintikka-style: the node and its descendants are closed and can't be extended), `view_graph`, `submit`.

```bash
pip install -e .
inspect eval bo_eval/task.py --model openai/gpt-4o -T solver=react -T budget=30 --epochs 5
inspect eval bo_eval/task.py -T solver=bo --model mockllm/model
```

## epistemic

`epistemic` provides finite-candidate GP-EI for enzyme activity and drug survival, with optional literature-informed priors via `gated_ei`. Final selection is the greedy best observed result.

The enzyme task uses its dataset-reported spread. The drug task uses 5% relative CV by default; this is a modelling choice because the dataset has no replicates. Set `noise_cv` to change it.

```bash
python -m epistemic run --task drug --budget 20
python -m epistemic run --task drug --acquisition gated_ei --provider openai:<model> --max-searches 2
inspect eval epistemic/inspect_task.py --model anthropic/claude-sonnet-4-20250514 \
  -T task_name=drug -T provider=inspect -T acquisition=gated_ei -T max_searches=2
```

Custom domains pass a `(TaskSpec, Evaluator)` pair to `run_episode(config, domain=...)`. `TaskSpec` supplies the candidate space and objective; `Evaluator` owns hidden outcomes.

Exports include `graph.html`, `graph.svg`, `actions.json`, `metrics.json`, and the `Episode.save` artifacts.
