# Auditable experimental optimisation

A small, framework-independent core combines:

1. Bayesian optimisation over a finite set of feasible experiments.
2. Optional literature priors whose signed influence is learned from observations.
3. A reasoning graph containing every input, output, decision, citation and closure.

Plain GP-BO is the zero-prior limit of the same model used by the literature-guided
controller. Inspect is an adapter, not a dependency of the core.

## Put it in a new domain

A domain needs only parameter names, feasible candidates and an evaluator:

```python
import numpy as np

from bo_eval import Domain, Session, bo_loop

candidates = np.array([[20, 0.1], [30, 0.1], [20, 1.0], [30, 1.0]])

def evaluate(params, rng):
    # Replace with an instrument, simulator or remote API.
    return run_experiment(temperature=params["temperature"], salt=params["salt"])

domain = Domain(
    name="my_domain",
    description="reaction yield",
    params=["temperature", "salt"],
    X=candidates,
    evaluate=evaluate,
    goal="maximize",
    log=("salt",),
)

session = Session(domain=domain, budget=20)
bo_loop(session)
session.graph.export("results/reasoning")
```

This writes:

- `reasoning.svg`: a self-contained overview;
- `reasoning.md`: the complete, lossless audit trail and Mermaid diagram;
- `reasoning.json`: structured data for downstream analysis.

For the literature-guided loop, provide two async functions:

```python
from bo_eval import explore

gates = await explore(
    session,
    think=lambda prompt: my_llm(prompt),
    search=lambda query: my_literature_search(query),
)
```

`think` returns JSON requested by the prompt. `search` returns findings with DOI or
URL citations. `Session.max_searches` is enforced independently of either backend.

## Model

The shared posterior is

\[
y(x) = \sum_i g_i h_i(x) + f(x),
\]

where \(f\) is a Matérn GP and each \(h_i\) is a smooth basis function representing
a cited belief about the optimum. Literature trust initializes the signed coefficient
\(g_i\); observations update it conditional on the fitted residual kernel. With no
priors, the sum vanishes and this is ordinary GP-BO.

The coefficients are signed Gaussian variables, not bounded LSTM forget gates. Their
values are runtime model state. The graph records original evidence, later literature
recalibrations and experiment decisions—not repetitive gate-update nodes.

## Reasoning graph

- **Experiment nodes:** exact feasible inputs and observed outputs.
- **Solid edges:** experiment ancestry; these alone define a branch.
- **Dashed edges:** non-structural reasoning.
- **Evidence nodes:** claim, sources, applicability trust and justification.
- **Prior nodes:** the belief derived from that evidence.
- **Closures:** why a trajectory was not pursued, explicitly framed as a model
  judgement rather than a proof of optimality.
- **Final selection:** the chosen tested condition and its rationale.

## Bundled benchmarks and Inspect

Install optional integrations:

```bash
pip install -e '.[inspect,literature]'
```

The included CSV benchmarks are registered in `bo_eval/env.py`: `upo_abts`,
`icfree_cole1`, `icfree_colm`, `icfree_colm_eu` and `zimmer_a549`.

```bash
inspect eval bo_eval/task.py \
  --model anthropic/claude-sonnet-4-6 \
  -T env=zimmer_a549 \
  -T solver=dual \
  -T budget=30 \
  -T max_searches=6
```

Set `AMASS_API_KEY` to use Amass BiomedCore; otherwise the Inspect adapter uses
provider-side web search. `solver=bo`, `solver=random` and `solver=react` are retained
as controlled references.

`Session.score()` is intentionally benchmark-only because live domains do not know
their true optimum. Supply an external scorer for a simulator or physical system.
