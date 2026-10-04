# Minimal auditable optimisation

One loop, one decision record per experiment. No agent hierarchy, learned literature
trust scores, separate memory system, or claim that a deferred branch is disproved.

`loop.py` is the reusable system. It takes a finite numeric candidate matrix,
parameter names, an experiment callable, and optional async research/choice callbacks.
It has no robotics, Inspect, dataset or API imports. `lab.py` and `task.py` are adapters.

```python
from minimal_lab.loop import run, save

# Any finite numerical domain. Only settings are visible here, not unqueried outcomes.
episode = await run(
    candidates, parameter_names, execute,
    objective="reviewer-assessed design quality", goal="maximize", budget=12,
    research=search_evidence, choose=choose_experiment,
)
save(episode, "results/run-001")  # new directory, never overwrite an earlier run
```

`execute(params)` returns `{"value": float | None, "reported": [float, ...],
"report_sd": [float, ...], "ok": bool}` plus any auditable domain metadata.
`reported` is the observed input estimate, in parameter-column order. Use intended
parameters and zero SD only when execution is exact. Failure returns `value=None`,
`ok=False`, and a reason. The loop stops on failure rather than inventing a reward.
The numeric value is an observed proxy. Its relationship to the underlying objective
may remain unverified. No reward oracle is required by the loop.

`research(context)` returns source records with `title`, `abstract`, `url` and uncertainty.
`choose(context)` returns the JSON contract shown in `task.py`. It sees the objective,
shortlist, past observations and retrieved sources. It cannot query the response table.
Citation quotes must occur in retrieved abstracts. This validates attribution, not
the inference or reliability of the source. Citations can support selection or deferral.

Each round:

1. Fit a fixed Matern GP to observed rewards at reported inputs. Approximately propagate
   input-report uncertainty into response variance. These intervals are not calibrated.
2. Shortlist highest expected improvement, highest predictive uncertainty and an incumbent
   repeat. Previously measured conditions remain eligible. Deduplicate. The first round
   uses three seeded random candidates.
3. Optionally let an LLM choose within this shortlist, citing sources and explaining
   every deferred alternative. Invalid responses fall back to the numerical policy.
4. Record the decision, then execute. A result outside the pre-update 2-SD interval
   schedules a diagnostic repeat. It does not identify the cause or refute a mechanism.
5. Search at most twice: initially and after surprise. Reserve the last attempt for
   confirmation. Return the best mean observed proxy across intended settings and repeats.

Deferral is local to a decision, not permanent pruning. A later selection links back
to prior deferrals of that condition. Incumbent selection, final confirmation and final
recommendation all use mean observed response. No evidence edge propagates closure.

This bounds LLM influence, but it also limits literature guidance to shortlisted candidates.
The graph is an inspectable decision record, not proof that generated reasons are faithful
internal explanations. Callback providers are responsible for their own request timeouts.

## Robot demo

From the repository root:

```bash
pip install -e .
python -m inspect_ai eval minimal_lab/task.py --model mockllm/model -T budget=3

# Requires AMASS_API_KEY and credentials for the chosen Inspect model provider.
python -m inspect_ai eval minimal_lab/task.py --model "$INSPECT_MODEL" \
  -T budget=12 -T llm=true -T literature=true -T out=logs/literature-demo
```

Output: `graph.html` (standalone clickable audit), `graph.json`, and `result.json`
(best condition plus its simulation protocol), beneath the chosen output directory.
Use a fresh `out` for each invocation. Inspect also retains the episode and scores.

The adapter uses current upstream `LabBackend`: fresh tips, pipette motion, mixing,
and distinct wells. Successful transfers acquire a separate, seeded, mean-one lognormal
volume error. Realised concentrations, including dilution error, reach the dataset lookup.
Only a noisy continuous delivery estimate reaches the policy. Hidden mapped IDs do not.
Source replacement and tip-box refresh are simulated services, not robot actions.

**This is a simulation protocol.** The upstream scene does not simulate liquid volumes.
The CSV omits concentration units and stock recipes. Stock concentrations are explicitly
assumed to be five times each axis maximum. Scene reagent names denote physical slots,
not a chemically validated UPO recipe. pH/temperature are ideal external settings.
Nearest-row lookup on this sparse dataset can also move non-pipetted parameters.
The independent effect of enzyme-volume error is absent from the response table.

Inspect's `regret`/`found_optimal` use hidden dataset means only after the run, to score
the nominal recommended condition. They are benchmark-only metrics, not general
verification of reward quality or robustness to delivery noise. For an unverifiable
objective, retain observed feedback, repeats, provenance and failure counts instead.

## Repository integration

Based on upstream `KingL0ui5/Tokenation_Hack` at `5f28107` (2026-10-04).
Fork PRs #4–#7 are an overlapping stack. PR #8 (`985e98d`) replaces that stack with
an epistemic core, but its execution modes are perfect/perturbed, not MuJoCo.
This module combines the small-loop idea with upstream robotics without merging those
agent layers. No existing solver, branch or PR is replaced.

Run focused checks with `python -m pytest -q tests/test_minimal_lab.py`.
