"""Amass BiomedCore literature search (https://amass.tech). Needs AMASS_API_KEY.

Bibliographic metadata supports attribution, not numerical probabilities that a claim is correct.
"""

import os

import httpx

URL = "https://api.amass.tech/api/v1/cores/biomedcore/records"


def _fmt(p: dict) -> str:
    ref = f"doi:{p['doi']}" if p.get("doi") else f"PMID:{p.get('pmid')}"
    meta = [p.get("journal") or "?", (p.get("publicationDate") or "")[:4],
            f"{p.get('citationCount')} citations", f"JuFo {p.get('journalQualityJufo')}"]
    if p.get("isRetracted"):
        meta.append("RETRACTED")
    return f"- {p.get('title')} ({'; '.join(meta)}) {ref}\n  {(p.get('abstract') or '')[:600]}"


async def amass_search(query: str, limit: int = 6) -> str:
    papers = await search_records(query, limit)
    return "\n".join(_fmt(p) for p in papers) or "No records found."


async def search_records(query: str, limit: int = 6) -> list[dict]:
    return (await search_result(query, limit))["records"]


async def search_result(query: str, limit: int = 3) -> dict:
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.get(URL, params={"query": query, "limit": limit},
                        headers={"Authorization": f"Bearer {os.environ['AMASS_API_KEY']}"})
    r.raise_for_status()
    return {"records": r.json()["data"], "credit_cost": r.headers.get("X-Amass-Credit-Cost")}


async def evidence_bundle(query: str) -> list[dict]:
    return records_to_claims(await search_records(query))


def records_to_claims(records: list[dict]) -> list[dict]:
    from epistemic.graph import Claim

    claims = []
    for index, record in enumerate(records):
        reference = f"https://doi.org/{record['doi']}" if record.get("doi") else (
            f"https://pubmed.ncbi.nlm.nih.gov/{record['pmid']}/" if record.get("pmid") else "")
        if not reference:
            continue
        claim = Claim(
            id=f"L{index + 1}", statement=_fmt(record), source="literature",
            scope="the contexts described by the source; transfer to the benchmark assay is unresolved",
            reference=reference,
            discriminating_result="A matched-context experiment with an outcome inconsistent with the quoted report.",
            unresolved_transfer_assumptions=["Applicability to the benchmark assay must be checked, not assumed."],
        )
        claims.append(claim.model_dump())
    return claims
