"""Amass BiomedCore literature search (https://amass.tech). Needs AMASS_API_KEY.

Each record carries the signals needed to calibrate trust: citations, journal quality (JuFo) and retraction.
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
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.get(URL, params={"query": query, "limit": limit},
                        headers={"Authorization": f"Bearer {os.environ['AMASS_API_KEY']}"})
    r.raise_for_status()
    return "\n".join(_fmt(p) for p in r.json()["data"]) or "No records found."
