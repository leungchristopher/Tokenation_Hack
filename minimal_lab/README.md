# Minimal auditable optimisation

One loop, one decision record per experiment. No agent hierarchy, learned literature
trust scores, separate memory system, or claim that a deferred branch is disproved.

`loop.py` is the reusable system. It takes a finite numeric candidate matrix,
parameter names, an experiment callable, and optional async research/choice/prior callbacks.
It has no robotics, Inspect, dataset or API imports. `lab.py` and `task.py` are adapters.

```python
from minimal_lab.loop import run, save

# Any finite numerical domain. Only settings are visible here, not unqueried outcomes.
episode = await run(
    candidates, parameter_names, execute,
    objective="reviewer-assessed design quality", goal="maximize", budget=12,
    research=search_evidence, choose=choose_experiment,
    make_prior=interpret_literature, replicates=3,
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
2. Shortlist highest expected improvement, highest latent predictive uncertainty and an incumbent
   repeat. Previously measured conditions remain eligible. Deduplicate. The first round
   uses seeded random candidates. A cited location prior adds a preferred candidate across
   the whole domain. Its acquisition preference is bounded to 4:1 initially and decays
   with observations; raw EI and exploration options remain eligible.
3. Optionally let an LLM choose within this shortlist, citing sources and explaining
   every deferred alternative. Invalid responses fall back to the numerical policy.
4. Record the decision, then execute. Planned replicates repeat independent preparations
   (fresh wells in the robot adapter), consume budget individually, and keep every outcome. A result outside the pre-update 2-SD interval
   schedules a diagnostic repeat. It does not identify the cause or refute a mechanism.
5. Search at most twice: initially and after surprise. Reserve the last attempt for
   confirmation. Return the best mean observed proxy across intended settings and repeats.

Deferral is local to a decision, not permanent pruning. A later selection links back
to prior deferrals of that condition. Incumbent selection, final confirmation and final
recommendation all use mean observed response. No evidence edge propagates closure.

Literature can guide the shortlist through a location prior and then guide the choice.
`evidence.py` validates cited parameter bounds and computes bounded prior weights.
`research.py` owns AMASS retrieval and optional Inspect interpretation. The search query
comes from the task objective/parameters or an explicit `research_query`; the core loop
has no dependency on AMASS, Inspect or a particular assay. An unjustified prior is omitted;
retrieval/interpretation failure is recorded and numerical search continues.
The graph is an inspectable decision record, not proof that generated reasons are faithful
internal explanations. Callback providers are responsible for their own request timeouts.

## Robot demo

From the repository root:

```bash
pip install -e .
python -m inspect_ai eval minimal_lab/task.py --model mockllm/model -T budget=6 -T replicates=2

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

## Replication and scope

The demo defaults to two independent preparations per selection (`--replicates` for
the live server, `-T replicates=` for Inspect); the reusable loop defaults to one.
Budget counts preparations, including diagnostic and final confirmation runs. A budget
ending mid-group is allowed in the core loop and recorded through `replication_complete`.
Recommendations include the mean, sample SD and standard error over all valid repeats.
These describe observed variability under an independence assumption, not systematic
bias, biological/batch variability or confidence that the selected condition is optimal.
For those experiments, implement the relevant batch/replicate structure in `execute`.

Use `python -m minimal_lab.live --budget 12 --replicates 3` for the linked robot/graph
demo. Add `--model YOUR_INSPECT_MODEL_ID` and credentials to enable AMASS-backed priors
and literature-guided decisions. The older `bo_eval/task.py` is a separate legacy
optimiser and still has its original branch-closing semantics; it is not this demo.

To use another task, supply its candidates, parameter names, objective, goal and an
execution callable to `run`. Supply research and prior callbacks if needed. The UPO
robot recipe in `lab.py` and the benchmark scorer in `task.py` remain assay adapters;
a different physical assay needs its own recipe and measurement implementation.

## Cancer assay video

The live server now defaults to `--env zimmer_a549`: taxol, cisplatin and
 doxorubicin are dispensed into the same fresh well, topped up with culture medium,
then mixed. Every replicate uses a fresh well. `--env upo_abts` retains the enzyme demo.
For Inspect, select `-T env=zimmer_a549` explicitly. The objective is minimum A549
survival, not maximum enzyme activity.

Each live run writes `assay.mp4` and `assay.timeline.json` next to its Inspect logs.
Frames come from the executing backend's MuJoCo data; overlays identify the current
graph decision, drug, volume and well. The timeline maps action boundaries to frame
numbers and decision IDs. Video is sampled during execution and is not a calibrated
real-time record. Cell seeding/incubation/readout and liquid dynamics are not animated
as if physically simulated. See `data/zimmer/README.md` for data and noise provenance.

## Iterations and recording

The live default is 24 preparations, including paired search replicates and a reserved
final confirmation pair. A preparation is not an optimisation iteration: graph nodes
record `iteration` and `phase` (proposal, replicate, diagnostic or confirmation).
The model is updated after observations; proposals use all preceding observations.
Recommendations report both optimisation iterations and distinct conditions.

Recordings are 2560 x 1440. The robot panel is rendered at 1280 x 960 from current
MuJoCo state, with the current decision rationale above it and the growing decision/
observation graph alongside it. Deferred options are compacted in the video; the
interactive graph retains complete reasons, source quotations and evidence links.
