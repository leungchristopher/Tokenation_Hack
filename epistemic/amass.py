"""Amass BiomedCore literature search (https://amass.tech). Needs AMASS_API_KEY.

Bibliographic metadata supports attribution, not numerical probabilities that a claim is correct.
"""

import os
import re

from epistemic.graph import Claim, EvidenceGraph, EvidenceRecord, UncertaintyKind, UncertaintyRecord
from epistemic.source_filter import accessible_result, excluded_source

URL = "https://api.amass.tech/api/v1/cores/biomedcore/records"


async def search_result(query: str, limit: int = 3, timeout: float = 60.0) -> dict:
    import httpx

    if excluded_source(query):
        raise ValueError("This source is excluded from evaluation evidence.")
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.get(URL, params={"query": query, "limit": limit},
                        headers={"Authorization": f"Bearer {os.environ['AMASS_API_KEY']}"})
    r.raise_for_status()
    return accessible_result({"records": r.json()["data"], "credit_cost": r.headers.get("X-Amass-Credit-Cost")})


def _proposition(record: dict) -> str:
    abstract = str(record.get("abstract") or "")
    conclusion = re.search(r"\bconclusions?\s*:\s*(.+)", abstract, re.IGNORECASE | re.DOTALL)
    if conclusion:
        sentence = re.split(r"(?<=[.!?])\s+", conclusion.group(1).strip(), maxsplit=1)[0]
        if len(sentence) <= 240:
            return f"The source reports: {sentence}"
    title = " ".join(str(record.get("title") or "an unspecified assay").split())
    if len(title) > 200:
        title = title[:197].rsplit(" ", 1)[0] + "…"
    return f"The source investigates: {title}."


def records_to_claims(records: list[dict], query: str = "") -> dict:
    graph = EvidenceGraph()
    for index, record in enumerate(records):
        if excluded_source(record):
            continue
        reference = f"https://doi.org/{record['doi']}" if record.get("doi") else (
            f"https://pubmed.ncbi.nlm.nih.gov/{record['pmid']}/" if record.get("pmid") else "")
        if not reference:
            continue
        source = graph.add_evidence(EvidenceRecord(
            id=f"E{index + 1}", title=str(record.get("title") or "Untitled source"),
            abstract=str(record.get("abstract") or ""), reference=reference, query=query,
            metadata=record,
        ))
        uncertainties = []
        descriptions: tuple[tuple[UncertaintyKind, str], ...] = (
            ("source", "The source is flagged as retracted." if record.get("isRetracted") else
             "Retrieval establishes attribution, not the reliability of methods, controls or conclusions."),
            ("transfer", "Matching assay, organism or cell line, dose range and timing have not been established."),
            ("mechanistic", "A reported association alone does not establish a mechanism or higher-order interaction."),
        )
        for category, statement in descriptions:
            uncertainty = graph.add_uncertainty(UncertaintyRecord(
                id=f"U{index + 1}_{category}", category=category,
                statement=statement, scope="the source report and its proposed benchmark application",
                evidence=(source.id,),
            ))
            uncertainties.append(uncertainty.id)
        claim = graph.add_claim(Claim(
            id=f"L{index + 1}", statement=_proposition(record), source="literature",
            scope="source-reported context only; benchmark transfer is unverified",
            reference=reference, evidence=[source.id], uncertainties=uncertainties,
            discriminating_result="Run a matched-context dose or condition series with controls and replication; "
                                  "an inconsistent response challenges transfer of the reported effect.",
        ))
        graph.link(source.id, claim.id, "supports", "Supports source attribution, not benchmark applicability.")
        for uncertainty_id in uncertainties:
            graph.link(uncertainty_id, claim.id, "qualifies")
    return graph.model_dump(mode="json")
