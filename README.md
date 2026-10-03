# Tokenation_Hack

Inspect eval for LLM-driven experimental optimisation, using a Bayesian-optimisation tool and a legible reasoning graph.

- **Environments** (`bo_eval/env.py`): each task is a CSV of measured conditions (parameters, mean, SD) plus one entry in `ENVS`. Choose one with `-T env=<name>`. Each experiment snaps to the nearest measured condition and returns a draw from N(mean, SD).
  - `upo_abts`: UPO specific rate (`data/upo_abts.csv`).
  - `icfree_cole1`, `icfree_colm`, `icfree_colm_eu`: cell-free colicin yield from the active-learning data of [Borkowski et al., iScience 2025](https://doi.org/10.1016/j.isci.2025.113599) ([Zenodo](https://doi.org/10.5281/zenodo.14904992)). `data/icfree/convert.py` converts Echo transfer volumes to final concentrations with Table S1 (conc = max conc × volume / max volume) and pools the replicate yields.
  - `zimmer_a549` (minimise): A549 survival over an 8×8×8 taxol × cisplatin × doxorubicin dose grid from [Zimmer et al., PNAS 2016](https://doi.org/10.1073/pnas.1606301113), taken from the S2 ZIP of [Tendler et al., PLoS Comput Biol 2019](https://doi.org/10.1371/journal.pcbi.1006956.s002). `data/zimmer/convert.py` maps dose indices to the paper's 3-fold series (20 µM to 13.7 nM). There are no replicates, so the SD is a constant estimated from residuals against grid neighbours.
- **Tools** (`bo_eval/tools/`): `run_experiment`, `bayes_opt_suggest` (GP + EI), `add_reasoning`, `close_branch` (Hintikka-style: the node and its descendants are closed and can't be extended), `view_graph`, `submit`.
- **Reasoning graph** (`bo_eval/state.py`): nodes are experiments (inputs and output), and edges carry reasoning. It is exported to `logs/graphs/*.md` (Mermaid) and `*.json`, and shown in the score explanation.
- **Scorer** (`bo_eval/scorer.py`): `found_optimal`, `n_experiments`, `regret`.
- **Solvers** (`bo_eval/solvers.py`): `react`, `react_prior` (LLM states a domain-knowledge prior via `set_prior`; `bayes_opt_suggest` weights EI by π(x)^(β·trust/n), πBO), `react_research` (planner + a `researcher` sub-agent with web search that adds cited, trust-scored evidence nodes to the graph and can set a trust-weighted prior), `react_no_bo`, `react_no_graph`, plus the no-LLM baselines `bo` and `random`. Add more to `SOLVERS`.

```bash
pip install -e .
inspect eval bo_eval/task.py --model openai/gpt-4o -T solver=react -T budget=30 --epochs 5
inspect eval bo_eval/task.py -T solver=bo --model mockllm/model
```
