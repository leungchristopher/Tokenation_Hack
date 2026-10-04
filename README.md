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
