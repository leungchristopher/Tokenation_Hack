"""Small fixed evidence bundles. Nothing here may be invented: every entry is traceable or flagged.

Optional live retrieval is available through epistemic.amass when AMASS_API_KEY is configured.
"""

from __future__ import annotations

import json
from pathlib import Path

from epistemic.graph import Assumption, Claim, EvidenceGraph
from epistemic.tasks import TaskSpec


def _provenance_claim(task: TaskSpec) -> Claim:
    return Claim(
        id="K1",
        statement=f"The candidate outcomes come from {task.provenance}",
        scope=f"the {len(task.candidates)} measured candidates of {task.name}",
        source="literature",
        reference=task.provenance,
        discriminating_result="A recorded value that cannot be reproduced from the cited source file.",
    )


MISLEADING = {
    "enzyme": "The highest activity occurs at the maximum measured temperature.",
    "drug": "The lowest survival always occurs at the highest dose of all three drugs together.",
}


def in_misleading_region(task: TaskSpec, params: dict[str, float]) -> bool:
    """The region the benchmark-generated claim points at, used only to decide contradiction."""
    bounds = task.bounds()
    if task.name.startswith("drug"):
        return all(params[name] >= bounds[name][1] for name in task.names)
    return params["temperature"] >= bounds["temperature"][1]


def seed_graph(task: TaskSpec, misleading: bool = False, evidence_file: str | None = None) -> EvidenceGraph:
    graph = EvidenceGraph()
    graph.add_claim(_provenance_claim(task))
    if evidence_file:
        bundle = json.loads(Path(evidence_file).read_text())
        for record in bundle:
            claim = Claim.model_validate(record)
            if claim.benchmark_generated:
                raise ValueError("Benchmark-generated evidence uses the controlled misleading condition, not a source bundle.")
            graph.add_claim(claim)
    for i, text in enumerate(task.limitations, start=1):
        graph.add_assumption(Assumption(id=f"A{i}", statement=text, scope=task.name))
    graph.add_assumption(Assumption(
        id="A0",
        statement=task.noise.description,
        scope=f"observation noise for {task.name}",
    ))
    graph.add_assumption(Assumption(
        id="A_gp",
        statement="A stationary Matérn GP with fitted Gaussian noise is used in scaled intended coordinates. "
                  "Delivery reports are available to the LLM, not fitted as extra numerical covariates. "
                  "Small-sample fits are not guaranteed to be calibrated.",
        scope=f"numerical model for {task.name}",
    ))
    if misleading:
        key = "drug" if task.name.startswith("drug") else "enzyme"
        graph.add_claim(Claim(
            id="K9",
            statement=MISLEADING[key],
            scope=task.name,
            source="model_conjecture",
            reference="benchmark-generated evaluator metadata; not a publication",
            benchmark_generated=True,
            discriminating_result="A measured candidate outside that region with a better objective value.",
        ))
    return graph
