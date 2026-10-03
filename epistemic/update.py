"""Deterministic epistemic-state updates. Model prose never becomes an observation."""

from __future__ import annotations

from epistemic.evidence import in_misleading_region
from epistemic.graph import Claim, EvidenceGraph, Observation, UncertaintyKind, UncertaintyRecord
from epistemic.surrogate import ResponseUncertainty
from epistemic.tasks import TaskSpec

BEST = "K_best"
CALIBRATION = "K_calibration"


def _better(task: TaskSpec, a: float, b: float) -> bool:
    return a > b if task.direction == "maximize" else a < b


def update_state(graph: EvidenceGraph, task: TaskSpec, observation: Observation,
                 prediction: ResponseUncertainty | None, round: int) -> None:
    if observation.value_shown is None:
        return
    uncertainty_ids = []
    descriptions: tuple[tuple[UncertaintyKind, str], ...] = (
        ("response", task.noise.description),
        ("execution", f"Reported execution model: {observation.execution.get('execution_model', 'unspecified')}. "
         "Delivery reports and hidden realised conditions must not be conflated."),
        ("model", "No fitted pre-experiment forecast was available." if prediction is None else
         f"Pre-experiment latent SD {prediction.latent_sd:g}, noise SD {prediction.noise_sd:g}, "
         f"interval [{prediction.interval[0]:g}, {prediction.interval[1]:g}]. "
         "These estimates do not establish calibration or model correctness."),
    )
    for category, statement in descriptions:
        item = graph.add_uncertainty(UncertaintyRecord(
            id=f"U_{observation.id}_{category}", category=category, statement=statement,
            scope=f"shown response and prediction for {observation.id} on {task.name}",
            evidence=(observation.id,), round=round,
        ))
        uncertainty_ids.append(item.id)
    value = observation.value_shown
    updated = []
    if BEST not in graph.claims:
        graph.add_claim(Claim(
            id=BEST,
            statement=f"The best shown objective so far is {value:g} {task.outcome_unit} at {observation.candidate_id}.",
            scope=f"executed experiments on {task.name}",
            source="measurement",
            evidence=[observation.id],
            status="supported",
            discriminating_result="Any executed candidate with a better objective value.",
            uncertainties=uncertainty_ids,
        ), round=round)
        graph.link(observation.id, BEST, "supports")
        updated.append(BEST)
    else:
        current = graph.claims[BEST]
        last_best = current.revisions[-1].evidence[0]
        best_value = graph.observations[last_best].value_shown
        assert best_value is not None
        if _better(task, value, best_value):
            graph.revise(current.id, round, "supported", "a better executed result was obtained",
                         [observation.id],
                         f"The best shown objective so far is {value:g} {task.outcome_unit} at {observation.candidate_id}.",
                         uncertainties=uncertainty_ids)
            graph.link(observation.id, BEST, "supports")
            updated.append(BEST)

    if prediction is not None:
        covered = prediction.interval[0] <= value <= prediction.interval[1]
        if CALIBRATION not in graph.claims:
            graph.add_claim(Claim(
                id=CALIBRATION,
                statement="The latest pre-experiment 95% interval covers its corresponding shown response.",
                scope=f"surrogate predictions on {task.name}",
                source="model_conjecture",
                evidence=[observation.id],
                status="supported" if covered else "contradicted",
                discriminating_result="An executed result outside the predicted interval.",
                uncertainties=uncertainty_ids,
            ), round=round)
        else:
            graph.revise(CALIBRATION, round, "supported" if covered else "contradicted",
                         "result inside the predicted interval" if covered else
                         "result outside the predicted interval; possible causes include noise, execution error "
                         "or model form, none of which this result alone establishes",
                         [observation.id], uncertainties=uncertainty_ids)
        graph.link(observation.id, CALIBRATION, "supports" if covered else "contradicts")
        updated.append(CALIBRATION)
    for claim_id in updated:
        for uncertainty_id in uncertainty_ids:
            graph.link(uncertainty_id, claim_id, "qualifies")

    for claim in list(graph.claims.values()):
        if claim.benchmark_generated and claim.status != "contradicted":
            params = task.params_of(observation.candidate_id)
            claimed_region = [o for o in graph.observations.values()
                              if in_misleading_region(task, o.intended_params) and o.value_shown is not None]
            best_inside = (max if task.direction == "maximize" else min)(
                claimed_region, key=lambda o: o.value_shown or 0.0, default=None)
            if not in_misleading_region(task, params) and best_inside is not None and _better(
                task, value, best_inside.value_shown or 0.0
            ):
                graph.revise(claim.id, round, "contradicted",
                             "a shown result outside the claimed region beats one inside it; "
                             "this noisy comparison is not proof about all unmeasured settings",
                             [observation.id, best_inside.id])
                graph.link(observation.id, claim.id, "contradicts")
