"""Small fixed evidence bundles. Nothing here may be invented: every entry is traceable or flagged.

Optional live retrieval is available through epistemic.amass when AMASS_API_KEY is configured.
"""

from __future__ import annotations

import json
from pathlib import Path

from epistemic.amass import records_to_claims
from epistemic.graph import Assumption, Claim, EvidenceGraph, EvidenceRecord, UncertaintyRecord
from epistemic.source_filter import accessible_result, excluded_source
from epistemic.tasks import TaskSpec


def _provenance_claim(task: TaskSpec, evidence_id: str) -> Claim:
    return Claim(
        id="K1",
        statement=f"The candidate outcomes come from {task.provenance}",
        scope=f"the {len(task.candidates)} measured candidates of {task.name}",
        source="literature",
        reference=task.provenance,
        evidence=[evidence_id],
        discriminating_result="A recorded value that cannot be reproduced from the cited source file.",
    )


def _legacy_bundle(claims: list[dict], records: list[dict], query: str) -> dict:
    bundle = EvidenceGraph()
    for payload in claims:
        if excluded_source(payload):
            continue
        claim = Claim.model_validate(payload)
        if claim.source != "literature" or claim.benchmark_generated:
            raise ValueError("Retrieval must return attributable literature, not model conjectures.")
        matching = next((r for r in records if (r.get("doi") and str(r["doi"]) in claim.reference)
                         or (r.get("pmid") and str(r["pmid"]) in claim.reference)), {})
        record = {**matching, "title": matching.get("title") or f"Source for {claim.id}"}
        if "doi.org/" in claim.reference:
            record["doi"] = claim.reference.split("doi.org/", 1)[1]
        elif "pubmed.ncbi.nlm.nih.gov/" in claim.reference:
            record["pmid"] = claim.reference.split("pubmed.ncbi.nlm.nih.gov/", 1)[1].strip("/")
        else:
            record["doi"] = claim.reference
        converted = EvidenceGraph.model_validate(records_to_claims([record], query))
        evidence = next(iter(converted.evidence_records.values()))
        evidence = evidence.model_copy(update={
            "id": f"E_{claim.id}", "reference": claim.reference,
            "metadata": {**record, "statement_as_received": claim.statement},
        })
        evidence = EvidenceRecord.model_validate(evidence.model_dump())
        bundle.add_evidence(evidence)
        uncertainty_ids = []
        for item in converted.uncertainties.values():
            item = item.model_copy(update={"id": f"U_{claim.id}_{item.category}", "evidence": (evidence.id,)})
            bundle.add_uncertainty(item)
            uncertainty_ids.append(item.id)
        if len(claim.statement) > 280 or "\n" in claim.statement:
            claim.statement = next(iter(converted.claims.values())).statement
        claim.evidence = [evidence.id]
        claim.uncertainties = uncertainty_ids
        claim.revisions = []
        claim.unresolved_transfer_assumptions = [
            "Matching assay, organism/cell line, dose range and timing must be checked."
        ]
        bundle.add_claim(claim)
        bundle.link(evidence.id, claim.id, "supports", "Source attribution does not establish transfer.")
        for uncertainty_id in uncertainty_ids:
            bundle.link(uncertainty_id, claim.id, "qualifies")
    return bundle.model_dump(mode="json")


def ingest_literature(graph: EvidenceGraph, payload: dict | list[dict], round: int = 0,
                      prefix: str = "", query: str = "", limit: int | None = None) -> tuple[list[dict], list[dict]]:
    if isinstance(payload, list):
        payload = {"claims": payload}
    clean = accessible_result(payload)
    if isinstance(clean.get("claims", []), list):
        clean = _legacy_bundle(clean.get("claims", []), clean.get("records", []), query)
    removed = set()
    for table in ("evidence_records", "uncertainties", "claims"):
        if isinstance(payload.get(table), dict):
            removed.update(set(payload[table]) - set(clean.get(table, {})))
    for table in ("uncertainties", "claims"):
        for record_id, record in list(clean.get(table, {}).items()):
            if set(record.get("evidence", [])) & removed or set(record.get("uncertainties", [])) & removed:
                del clean[table][record_id]
                removed.add(record_id)
    clean["edges"] = [
        edge for edge in clean.get("edges", []) if edge["source"] not in removed and edge["target"] not in removed
        and not excluded_source(edge)
    ]
    incoming = EvidenceGraph.model_validate(clean)
    if incoming.observations or incoming.decisions or incoming.assumptions:
        raise ValueError("External source bundles cannot contain experiments or agent decisions.")
    for claim in incoming.claims.values():
        if not set(claim.evidence) <= set(incoming.evidence_records):
            raise ValueError("Literature propositions must cite raw source records.")
    for item in incoming.uncertainties.values():
        if not set(item.evidence) <= set(incoming.evidence_records):
            raise ValueError("Source uncertainty must refer to raw source records.")
    known = {c.reference: c.id for c in graph.claims.values() if c.source == "literature"}
    added, duplicates = [], []
    selected = []
    for claim in list(incoming.claims.values())[:limit]:
        if claim.source != "literature" or claim.benchmark_generated:
            raise ValueError("Only external literature may be ingested.")
        if claim.reference in known:
            duplicates.append({"reference": claim.reference, "existing_claim_id": known[claim.reference]})
        else:
            selected.append(claim)
            known[claim.reference] = f"L{prefix}_{len(selected)}" if prefix else claim.id
    uncertainty_ids = {i for c in selected for i in c.uncertainties}
    source_ids = {i for c in selected for i in c.evidence if i in incoming.evidence_records}
    source_ids |= {i for u in incoming.uncertainties.values() if u.id in uncertainty_ids
                   for i in u.evidence if i in incoming.evidence_records}
    mapping = {
        c.id: f"L{prefix}_{index}" if prefix else c.id for index, c in enumerate(selected, 1)
    }
    for label, records_table, ids in (("E", incoming.evidence_records, source_ids),
                                      ("U", incoming.uncertainties, uncertainty_ids)):
        mapping.update({record_id: f"{label}{prefix}_{index}" if prefix else record_id
                        for index, record_id in enumerate((i for i in records_table if i in ids), 1)})
    if any(new_id in graph for new_id in mapping.values()):
        raise ValueError("Imported record ID already exists.")
    for record_id, evidence in incoming.evidence_records.items():
        if record_id in source_ids:
            graph.add_evidence(evidence.model_copy(update={"id": mapping[record_id]}))
    for record_id, uncertainty in incoming.uncertainties.items():
        if record_id in uncertainty_ids:
            graph.add_uncertainty(uncertainty.model_copy(update={
                "id": mapping[record_id], "round": round,
                "evidence": tuple(mapping[i] for i in uncertainty.evidence),
            }))
    for claim in selected:
        claim.id = mapping[claim.id]
        claim.evidence = [mapping[i] for i in claim.evidence]
        claim.uncertainties = [mapping[i] for i in claim.uncertainties]
        claim.revisions = []
        graph.add_claim(claim, round=round)
        added.append(claim.model_dump(mode="json"))
    for edge in incoming.edges:
        if edge.source in mapping and edge.target in mapping:
            graph.link(mapping[edge.source], mapping[edge.target], edge.kind, edge.note)
    return added, duplicates


def external_evidence(graph: EvidenceGraph) -> dict:
    claims = {i: c for i, c in graph.claims.items() if c.source == "literature" and i != "K1"}
    uncertainty_ids = {i for c in claims.values() for i in c.uncertainties}
    evidence_ids = {i for c in claims.values() for i in c.evidence if i in graph.evidence_records}
    selected = set(claims) | uncertainty_ids | evidence_ids
    return EvidenceGraph(
        claims=claims,
        evidence_records={i: e for i, e in graph.evidence_records.items() if i in evidence_ids},
        uncertainties={i: u for i, u in graph.uncertainties.items() if i in uncertainty_ids},
        edges=[e for e in graph.edges if e.source in selected and e.target in selected],
    ).model_dump(mode="json")


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
    source = graph.add_evidence(EvidenceRecord(
        id="E0", title=f"{task.name} dataset contract", reference=task.provenance, source="dataset",
        metadata={"candidates": len(task.candidates), "limitations": task.limitations},
    ))
    graph.add_claim(_provenance_claim(task, source.id))
    graph.link(source.id, "K1", "supports", "Dataset attribution, not mechanistic evidence.")
    uncertainty = graph.add_uncertainty(UncertaintyRecord(
        id="U0_source", category="source", scope=f"dataset attribution for {task.name}",
        statement="Dataset limitations and undocumented experimental details remain unresolved.",
        evidence=(source.id,),
    ))
    graph.claims["K1"].uncertainties.append(uncertainty.id)
    graph.link(uncertainty.id, "K1", "qualifies")
    if evidence_file:
        bundle = json.loads(Path(evidence_file).read_text())
        ingest_literature(graph, bundle)
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
