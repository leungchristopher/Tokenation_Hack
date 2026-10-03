"""Dependency-free SVG view of the graph. Full explanations remain in tooltips and Markdown."""

import textwrap
from html import escape
from urllib.parse import urlparse

from bo_eval.graph import Evidence, Prior, ReasoningGraph


def _url(source: str) -> str:
    if source.lower().startswith(("doi:", "10.")):
        source = "https://doi.org/" + source.removeprefix("doi:")
    return source if urlparse(source).scheme in {"https", "http"} else ""


def to_svg(graph: ReasoningGraph) -> str:
    experiments = graph.experiments
    knowledge: list[Evidence | Prior] = [*graph.evidence, *graph.priors]
    height = max(260 + len(experiments) * 126, 240 + len(knowledge) * 180)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}" role="img">',
        "<title>Experimental reasoning graph</title>",
        "<desc>Experiments in chronological order, linked to their parents and cited evidence. "
        "Hover over a node for its full explanation. Dashed arrows are reasoning, not ancestry.</desc>",
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
        '<path d="M0,0 L8,4 L0,8" fill="#94a3b8"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<g font-family="system-ui,sans-serif" fill="#0f172a">',
        '<text x="40" y="45" font-size="25" font-weight="700">Experimental reasoning graph</text>',
        f'<text x="40" y="73" font-size="13" fill="#64748b">{len(experiments)} experiments · '
        f'{len(graph.evidence)} evidence records · {len(graph.priors)} priors</text>',
        '<text x="40" y="97" font-size="12" fill="#64748b">'
        'Green: final choice · grey: not pursued · hover: full reasoning · complete audit: accompanying Markdown</text>',
        '<text x="40" y="131" font-size="12" fill="#64748b">EXPERIMENTS / PARENT LINKS</text>',
        '<text x="760" y="131" font-size="12" fill="#64748b">EVIDENCE / PRIORS</text>',
    ]
    positions = {"root": (52, 166)}
    for i, n in enumerate(experiments):
        positions[n.id] = (140 + (i % 2) * 46, 218 + i * 126)
    for i, item in enumerate(knowledge):
        positions[item.id] = (760, 218 + i * 180)

    def line(source, target, label="", dashed=False):
        if source not in positions or target not in positions:
            return
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        parts.append(f'<path d="M{x1},{y1} C{x1 - 80},{y1} {x2 - 80},{y2} {x2},{y2}" '
                     f'fill="none" stroke="#94a3b8" stroke-width="1.5"{dash} marker-end="url(#arrow)">'
                     f'<title>{escape(label)}</title></path>')

    for e in graph.edges:
        line(e.source, e.target, e.reasoning, e.kind == "reasoning")
    for p in graph.priors:
        line(p.after, p.id, p.reasoning, True)
    for v in graph.evidence:
        for target in v.about:
            line(v.id, target, v.claim, True)

    def box(nid, title, subtitle, detail, color, stroke, width=500):
        x, y = positions[nid]
        parts.append(f'<g><title>{escape(detail)}</title><rect x="{x}" y="{y - 34}" width="{width}" height="100" '
                     f'rx="10" fill="{color}" stroke="{stroke}" stroke-width="1.5"/>')
        parts.append(f'<text x="{x + 14}" y="{y - 11}" font-size="14" font-weight="650">{escape(title)}</text>')
        for j, text in enumerate(textwrap.wrap(subtitle, width=70 if width == 500 else 36)[:3]):
            parts.append(f'<text x="{x + 14}" y="{y + 12 + j * 17}" font-size="11">{escape(text)}</text>')
        parts.append("</g>")

    parts += ['<circle cx="52" cy="166" r="7" fill="#334155"/>',
              '<text x="68" y="171" font-size="12">start</text>']
    for n in experiments:
        reasons = "\n".join(f"From {e.source}: {e.reasoning}" for e in graph.edges if e.target == n.id)
        detail = f"{graph._label(n)}\n{reasons}"
        if n.closed:
            detail += f"\nNot pursued: {n.closed_reason}"
        selected = n.id == graph.selected
        if selected:
            detail += f"\nFinal selection: {graph.decision}"
        color, stroke = ("#dcfce7", "#15803d") if selected else (
            ("#f1f5f9", "#94a3b8") if n.closed else ("#fff", "#cbd5e1")
        )
        title = f"{n.id} · measured {n.result:.4g}" + (" · selected" if selected else " · not pursued" if n.closed else "")
        subtitle = ", ".join(f"{p}={v:g}" for p, v in n.inputs.items())
        box(n.id, title, subtitle, detail, color, stroke)
    for v in graph.evidence:
        detail = f"{v.claim}\nTrust {v.trust:g}: {v.trust_reason}\nSources: {'; '.join(v.sources)}"
        box(v.id, f"{v.id} · trust {v.trust:g}", v.claim, detail, "#fff7ed", "#fdba74", 300)
        x, y = positions[v.id]
        for j, source in enumerate(v.sources):
            href = _url(source)
            if href:
                parts.append(f'<a href="{escape(href, quote=True)}"><text x="{x + 14 + j * 55}" y="{y + 88}" '
                             f'font-size="11" fill="#0369a1">source {j + 1}</text></a>')
    for p in graph.priors:
        detail = f"{p.reasoning}\nBelief: {p.belief}\nInitial trust: {p.trust:g}"
        box(p.id, f"{p.id} · initial trust {p.trust:g}", p.reasoning, detail, "#eff6ff", "#93c5fd", 300)
    parts.append("</g></svg>")
    return "\n".join(parts)
