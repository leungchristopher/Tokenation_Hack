"""Source holdout enforced before evidence reaches the agent."""

import html
import json
import re
import unicodedata
from urllib.parse import unquote

EXCLUDED_SOURCE_IDENTIFIERS = (
    "Prediction of multidimensional drug dose responses based on measurements of drug pairs",
    "10.1073/pnas.1606301113",
    "27562164",
    "PMC5027409",
)


def _normalize(text: str) -> str:
    decoded = unicodedata.normalize("NFKC", html.unescape(unquote(text))).casefold()
    return re.sub(r"[^a-z0-9]", "", decoded)


def excluded_source(value: object) -> bool:
    text = _normalize(json.dumps(value, ensure_ascii=False, default=str))
    return any(_normalize(identifier) in text for identifier in EXCLUDED_SOURCE_IDENTIFIERS)


def accessible_result(result: dict) -> dict:
    """Keep allowed records, never retain blocked raw abstracts in tool/audit results."""
    clean = {}
    for key, value in result.items():
        if key in ("records", "claims", "evidence_records", "uncertainties"):
            clean[key] = ({i: record for i, record in value.items() if not excluded_source(record)}
                          if isinstance(value, dict) else
                          [record for record in value if not excluded_source(record)])
        elif not excluded_source(value):
            clean[key] = value
    return clean
