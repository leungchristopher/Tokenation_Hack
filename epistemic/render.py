"""Dependency-free, attributable evidence and experiment graph exports."""

from __future__ import annotations

import html
import json
import re
import textwrap
from importlib.resources import files
from pathlib import Path
from typing import Any, TypeAlias

from epistemic.graph import Assumption, Claim, Decision, EvidenceRecord, Observation, UncertaintyRecord
from epistemic.loop import Episode
from epistemic.metrics import episode_metrics

EVIDENCE_WIDTH = 1640
EXPERIMENT_WIDTH = 1900
Record: TypeAlias = Observation | Claim | Assumption | Decision | EvidenceRecord | UncertaintyRecord
EVIDENCE_EDGES = ("supports", "qualifies", "contradicts", "not_transferable")

STYLE = """
text{font-family:Inter,ui-sans-serif,system-ui,sans-serif;fill:#172033;font-size:13px}
.title{font-size:22px;font-weight:700}.section{font-size:15px;font-weight:650}
.node-title{font-size:13px;font-weight:650}.node-copy{font-size:12px}.muted{fill:#647184;font-size:11px}
.canvas{fill:#fff}.lane{fill:#f8fafc;stroke:#d7dde5}.record-box{fill:#fff;stroke:#8c98a8;stroke-width:1.25}
.record.source .record-box{stroke:#8a6422}.record.observation .record-box{stroke:#327454}
.record.claim .record-box{stroke:#315f91}.record.assumption .record-box{stroke:#677487;stroke-dasharray:4 3}
.record.contradicted .record-box{stroke:#a83b32;stroke-width:2}
.record:focus .record-box,.record:hover .record-box{stroke:#111827;stroke-width:2.5;outline:none}
.edge{fill:none;stroke-width:1.45}.supports{stroke:#327454}.contradicts{stroke:#a83b32;stroke-width:1.8}
.depends_on{stroke:#677487;stroke-dasharray:5 4}.tests{stroke:#315f91}
.qualifies{stroke:#8a6422;stroke-dasharray:3 3}.not_transferable{stroke:#a83b32;stroke-dasharray:8 3}
.record.uncertainty .record-box{stroke:#8a6422;stroke-dasharray:3 3}
.badge-box{fill:#f1f4f8;stroke:#cdd5df}.badge-text{font-size:10px;fill:#445368}
.edge-label{font-size:10px;font-weight:650}.divider{stroke:#d7dde5}
"""


def export(episode: Episode, directory: str | Path) -> Path:
    out = episode.save(directory)
    (out / "metrics.json").write_text(json.dumps(episode_metrics(episode), indent=2))
    (out / "evidence.svg").write_text(to_evidence_svg(episode))
    (out / "experiments.svg").write_text(to_experiment_svg(episode))
    (out / "actions.json").write_text(json.dumps(actions_graph(episode), indent=2))
    (out / "actions.svg").write_text(to_actions_svg(episode))
    (out / "graph.svg").write_text(to_svg(episode))
    (out / "graph.html").write_text(to_html(episode))
    (out / "audit.md").write_text(to_markdown(episode))
    return out


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _wrap(value: object, width: int, limit: int) -> list[str]:
    lines = textwrap.wrap(str(value), width, break_long_words=False, break_on_hyphens=False)
    if len(lines) > limit:
        lines = lines[:limit]
        lines[-1] = lines[-1].rstrip(" .") + "…"
    return lines or [""]


def _text(parts: list[str], x: float, y: float, value: object, css: str = "",
          width: int | None = None, limit: int = 1, step: int = 16) -> None:
    lines = _wrap(value, width, limit) if width else [str(value)]
    for index, line in enumerate(lines):
        parts.append(f'<text x="{x}" y="{y + index * step}" class="{css}">{_esc(line)}</text>')


def _detail(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _record(episode: Episode, record_id: str) -> Record:
    if record_id in episode.graph.evidence_records:
        return episode.graph.evidence_records[record_id]
    if record_id in episode.graph.uncertainties:
        return episode.graph.uncertainties[record_id]
    if record_id in episode.graph.observations:
        return episode.graph.observations[record_id]
    if record_id in episode.graph.claims:
        return episode.graph.claims[record_id]
    if record_id in episode.graph.assumptions:
        return episode.graph.assumptions[record_id]
    if record_id in episode.graph.decisions:
        return episode.graph.decisions[record_id]
    raise KeyError(record_id)


def _record_kind(record: Record) -> str:
    if isinstance(record, EvidenceRecord):
        return "source"
    if isinstance(record, UncertaintyRecord):
        return "uncertainty"
    if isinstance(record, Observation):
        return "observation"
    if isinstance(record, Assumption):
        return "assumption"
    if isinstance(record, Claim) and record.source == "literature":
        return "source"
    return "claim"


def _node(parts: list[str], *, record_id: str, dom_id: str, x: int, y: int, width: int, height: int,
          title: str, lines: list[str], footer: str, detail: str, css: str,
          badges: list[tuple[str, str]] | None = None, centered: bool = False) -> None:
    attrs = (f'id="{_esc(dom_id)}" class="a-node record {_esc(css)}" data-a-node="true" '
             f'data-record-id="{_esc(record_id)}" data-a-title="{_esc(title)}" '
             f'data-a-detail="{_esc(detail)}" tabindex="0" role="button"')
    if centered:
        attrs += ' text-anchor="middle"'
    parts.append(f"<g {attrs}><title>{_esc(detail)}</title>")
    parts.append(f'<rect class="record-box" x="{x}" y="{y}" width="{width}" height="{height}" rx="3"/>')
    text_x = x + width // 2 if centered else x + 13
    _text(parts, text_x, y + 22, title, "node-title", width=max(20, width // 7), limit=1)
    for index, line in enumerate(lines):
        _text(parts, text_x, y + 46 + index * 15, line, "node-copy")
    _text(parts, text_x, y + height - 13, footer, "muted", width=47, limit=1)
    badge_x = x + 13
    badge_y = y + height - 65
    for index, (category, status) in enumerate(badges or []):
        if index == 3:
            badge_x = x + 13
            badge_y += 22
        label = f"{category}: {status.replace('_', ' ')}"
        badge_width = int(len(label) * 5.5) + 12
        parts.append(
            f'<g data-uncertainty="{_esc(category)}"><rect class="badge-box" x="{badge_x}" '
            f'y="{badge_y}" width="{badge_width}" height="18" rx="3"/>'
            f'<text class="badge-text" x="{badge_x + 6}" y="{badge_y + 12}">{_esc(label)}</text></g>'
        )
        badge_x += badge_width + 6
    parts.append("</g>")


def _edge(parts: list[str], *, source: str, target: str, kind: str,
          start: tuple[float, float], end: tuple[float, float], note: str = "",
          via_y: float | None = None, label: str | None = None) -> None:
    sx, sy = start
    tx, ty = end
    middle = sx + (tx - sx) * 0.52
    title = f"{source} {kind} {target}"
    path = f"M{sx},{sy} C{middle},{sy} {middle},{ty} {tx},{ty}"
    detail = _detail({"kind": "edge", "source": source, "target": target, "relationship": kind,
                      "reason": note or label or kind})
    group = (
        f'<g data-a-edge="true" tabindex="0" role="button" data-a-title="{_esc(title)}" '
        f'data-a-detail="{_esc(detail)}" data-edge-source="{_esc(source)}" '
        f'data-edge-target="{_esc(target)}" data-edge-kind="{_esc(kind)}">'
    )
    parts.append(
        f'{group}<title>{_esc(title)}</title><path class="a-edge edge {kind}" d="{path}" '
        f'marker-end="url(#arrow-{kind})"/>'
    )
    label_x = sx + (tx - sx) * 0.44
    label_y = sy + (ty - sy) * 0.44
    if via_y is not None:
        path = f"M{sx},{sy} H{sx + 20} V{via_y} H{tx - 20} V{ty} H{tx}"
        label_x, label_y = (sx + tx) / 2 - 20, via_y - 4
        parts[-1] = (
            f'{group}<title>{_esc(title)}</title><path class="a-edge edge {kind}" d="{path}" '
            f'marker-end="url(#arrow-{kind})"/>'
        )
    lines = _wrap(label, 23, 4) if label else [kind]
    label_width = max(len(line) for line in lines) * 6 + 12
    if label:
        label_x = (sx + tx - label_width) / 2 + 6
    parts.append(f'<rect class="canvas" x="{label_x - 4}" y="{label_y - 12}" '
                 f'width="{label_width}" height="{len(lines) * 13 + 2}"/>')
    for index, line in enumerate(lines):
        _text(parts, label_x, label_y + index * 13, line, f"edge-label {kind}")
    if note:
        parts[-1] = parts[-1].replace("</text>", f"<title>{_esc(note)}</title></text>")
    parts.append("</g>")


def _definitions() -> str:
    colors = {"supports": "#327454", "contradicts": "#a83b32", "qualifies": "#8a6422",
              "not_transferable": "#a83b32",
              "depends_on": "#677487", "tests": "#315f91"}
    markers = []
    for kind, color in colors.items():
        markers.append(
            f'<marker id="arrow-{kind}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
            f'<path d="M0,0 L8,4 L0,8 Z" fill="{color}"/></marker>'
        )
    return f"<defs>{''.join(markers)}</defs>"


def _document(body: str, width: int, height: int, label: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(label)}" '
        f'data-a-diagram-canvas="true">'
        f"<style>{STYLE}</style><rect class=\"canvas\" width=\"100%\" height=\"100%\"/>"
        f"{_definitions()}<g data-a-diagram-viewport=\"true\">{body}</g></svg>"
    )


def _claim_title(claim: Claim) -> str:
    if claim.id == "K1":
        return f"{claim.id} · Dataset contract"
    labels = {"literature": "Literature evidence", "measurement": "Measured summary",
              "model_conjecture": "Working claim"}
    return f"{claim.id} · {labels[claim.source]}"


def _reference_label(reference: str) -> str:
    if not reference:
        return "No external reference"
    if "doi.org/" in reference:
        return "DOI " + reference.split("doi.org/", 1)[1].split()[0].rstrip(".,;")
    if "pubmed.ncbi.nlm.nih.gov/" in reference:
        return "PubMed " + reference.split("pubmed.ncbi.nlm.nih.gov/", 1)[1].split("/", 1)[0]
    return reference


def _evidence_parts(episode: Episode) -> tuple[str, int]:
    graph = episode.graph
    sources: list[Record] = [*graph.evidence_records.values(),
                             *sorted(graph.observations.values(), key=lambda item: item.round)]
    claims = list(graph.claims.values())
    height = max(185 + len(sources) * 138, 185 + len(claims) * 250, 400)
    header: list[str] = []
    edges: list[str] = []
    nodes: list[str] = []

    _text(header, 32, 38, "Claims, evidence and refutations", "title")
    _text(header, 32, 62, f"{episode.config.task} · immutable sources and observations → scoped propositions", "muted")
    _text(header, 32, 85, "Support is green; contradiction is red; qualification is dotted amber; rejected transfer is dashed red.", "muted")
    for x, width, title in ((32, 350, "Evidence and observations"), (490, 660, "Concise claims"),
                             (1270, 338, "Qualifications and uncertainty")):
        header.append(f'<rect class="lane" x="{x}" y="106" width="{width}" height="{height - 136}" rx="3"/>')
        _text(header, x + 12, 130, title, "section")

    positions: dict[str, tuple[int, int, int, int]] = {}
    used = {e.source for e in graph.edges if e.kind in EVIDENCE_EDGES}
    for index, record in enumerate(sources):
        y = 148 + index * 138
        positions[record.id] = (44, y, 326, 120)
        if isinstance(record, EvidenceRecord):
            title = f"{record.id} · {record.source} source"
            lines = _wrap(record.title, 46, 3)
            footer = _reference_label(record.reference) if record.id in used else "available, not linked"
        else:
            assert isinstance(record, Observation)
            title = f"{record.id} · Experimental result"
            shown = "not shown" if record.value_shown is None else f"{record.value_shown:.5g} {record.outcome_unit}"
            lines = [f"Observed: {shown}", *_wrap(
                " · ".join(f"{k}={v:g}" for k, v in record.intended_params.items()), 45, 3)]
            footer = f"Trial {record.round} · {record.candidate_id} · simulated"
        _node(nodes, record_id=record.id, dom_id=f"evidence-{record.id}", x=44, y=y, width=326, height=120,
              title=title, lines=lines, footer=footer, detail=_detail(record), css=_record_kind(record))

    qualifier_positions: dict[tuple[str, str], tuple[int, int]] = {}
    for index, claim in enumerate(claims):
        y = 148 + index * 250
        positions[claim.id] = (502, y, 636, 230)
        lines = [*_wrap(claim.statement, 88, 3), *_wrap(f"Scope: {claim.scope}", 88, 2),
                 *_wrap(f"Discriminating experiment: {claim.discriminating_result}", 88, 2)]
        _node(nodes, record_id=claim.id, dom_id=f"evidence-{claim.id}", x=502, y=y, width=636, height=230,
              title=_claim_title(claim), lines=lines, footer=f"{claim.status} · {_reference_label(claim.reference)}",
              detail=_detail(claim), css="claim" + (" contradicted" if claim.status == "contradicted" else ""),
              badges=_uncertainty_badges(episode, claim))
        qualifiers = list(dict.fromkeys(claim.uncertainties or [
            e.source for e in graph.edges if e.target == claim.id and e.kind in ("qualifies", "not_transferable")
            and e.source in graph.uncertainties
        ]))
        for offset, record_id in enumerate(qualifiers[:7]):
            uncertainty = graph.uncertainties[record_id]
            qualifier_y = y + offset * 30
            qualifier_positions[(record_id, claim.id)] = (1282, qualifier_y + 13)
            _node(nodes, record_id=record_id, dom_id=f"evidence-{claim.id}-{record_id}",
                  x=1282, y=qualifier_y, width=314, height=27,
                  title=f"{uncertainty.category} · {uncertainty.status}", lines=[], footer="",
                  detail=_detail(uncertainty), css="uncertainty")

    for relationship in graph.edges:
        if relationship.kind not in EVIDENCE_EDGES:
            continue
        key = (relationship.source, relationship.target)
        if key in qualifier_positions and relationship.target in positions:
            tx, ty, tw, th = positions[relationship.target]
            _edge(edges, source=relationship.source, target=relationship.target, kind=relationship.kind,
                  start=qualifier_positions[key], end=(tx + tw, ty + th / 2), note=relationship.note)
            continue
        if relationship.source not in positions or relationship.target not in positions:
            continue
        sx, sy, sw, sh = positions[relationship.source]
        tx, ty, _, th = positions[relationship.target]
        _edge(edges, source=relationship.source, target=relationship.target, kind=relationship.kind,
              start=(sx + sw, sy + sh / 2), end=(tx, ty + th / 2), note=relationship.note)

    return "".join(header + edges + nodes), height


def _uncertainty_badges(episode: Episode, claim: Claim) -> list[tuple[str, str]]:
    statuses = {episode.graph.uncertainties[i].category: episode.graph.uncertainties[i].status
                for i in claim.uncertainties if i in episode.graph.uncertainties}
    badges = [(category, statuses.get(category, "not assessed")) for category in ("source", "transfer", "mechanistic")]
    model = [statuses[k] for k in ("response", "model") if k in statuses]
    badges.append(("response/model", "unresolved" if "unresolved" in model else model[-1] if model else "not assessed"))
    badges.append(("execution", statuses.get("execution", "not assessed")))
    return badges


def _compact_summary(record: Record) -> str:
    if isinstance(record, Observation):
        value = "missing" if record.value_shown is None else f"{record.value_shown:.4g} {record.outcome_unit}"
        return f"result · {value}"
    if isinstance(record, Assumption):
        return f"assumption · {record.status}"
    if isinstance(record, Claim):
        labels = {"literature": "source", "measurement": "summary", "model_conjecture": "claim"}
        return f"{labels[record.source]} · {record.status}"
    return record.kind


def _experiment_parts(episode: Episode) -> tuple[str, int]:
    decisions = sorted(episode.graph.decisions.values(), key=lambda item: item.round)
    rows: list[tuple[Decision, list[str], list[str], list[str], int]] = []
    height = 155
    for decision in decisions:
        dependencies = list(dict.fromkeys(
            edge.target for edge in episode.graph.edges
            if edge.source == decision.id and edge.kind == "depends_on"
        ))
        tested = list(dict.fromkeys(
            edge.target for edge in episode.graph.edges
            if edge.source == decision.id and edge.kind == "tests"
            and edge.target in episode.graph.observations
        ))
        updates = list(dict.fromkeys(
            edge.target for edge in episode.graph.edges if edge.source in tested
            and edge.kind in EVIDENCE_EDGES and edge.target in episode.graph.claims
        ))
        row_height = max(260, 210 + 32 * len(dependencies), 60 + 200 * len(updates))
        rows.append((decision, dependencies, tested, updates, row_height))
        height += row_height

    header: list[str] = []
    edges: list[str] = []
    nodes: list[str] = []
    _text(header, 32, 38, "Experiments and reasoning", "title")
    _text(header, 32, 62, f"{episode.config.task} · {len(decisions)} experiment decisions in chronological order", "muted")
    _text(header, 32, 85, "Decision → experiment → observation → claim update. The next row is the next decision.", "muted")
    _text(header, 32, 108, "tests connects the decision to its observation; depends_on records declared inputs, not causal attribution.", "muted")
    for x, title in ((32, "Decision and reasoning"), (480, "Experiment"), (860, "Observation"),
                     (1320, "Claim update → next decision")):
        _text(header, x, 136, title, "section")

    y = 160
    for row_index, (decision, dependencies, tested, updates, row_height) in enumerate(rows):
        _text(nodes, 8, y + 19, f"{decision.round:02d}", "muted")
        nodes.append(f'<line class="divider" x1="32" y1="{y + row_height - 15}" '
                     f'x2="{EXPERIMENT_WIDTH - 32}" y2="{y + row_height - 15}"/>')
        decision_y = y + 32
        decision_height = 148
        prediction = "unavailable" if decision.prediction is None else f"{decision.prediction:.5g}"
        reason = _wrap(decision.justification or "No written rationale.", 49, 4)
        _node(
            nodes, record_id=decision.id, dom_id=f"experiment-{decision.id}", x=32, y=decision_y,
            width=370, height=decision_height,
            title=f"{decision.id} · {decision.policy} → {decision.candidate_id}",
            lines=reason,
            footer=(f"targets {decision.targeted_uncertainty} uncertainty · "
                    f"{decision.prediction_source} forecast {prediction}"),
            detail=_detail(decision), css="claim",
        )

        for index, record_id in enumerate(dependencies):
            record = _record(episode, record_id)
            dependency_y = decision_y + 166 + index * 32
            _node(
                nodes, record_id=record_id, dom_id=f"experiment-{decision.id}-dependency-{record_id}",
                x=32, y=dependency_y, width=370, height=27,
                title=f"{record_id} · {_compact_summary(record)}", lines=[], footer="",
                detail=_detail(record), css=_record_kind(record),
            )
            _edge(edges, source=decision.id, target=record_id, kind="depends_on",
                  start=(402, decision_y + 130),
                  end=(402, dependency_y + 13), note="Declared decision dependency.")

        for index, record_id in enumerate(tested):
            observation = episode.graph.observations[record_id]
            observation_y = decision_y + index * 138
            value = "not shown" if observation.value_shown is None else (
                f"{observation.value_shown:.5g} {observation.outcome_unit}"
            )
            params = " · ".join(f"{key}={value:g}" for key, value in observation.intended_params.items())
            _node(
                nodes, record_id=record_id, dom_id=f"experiment-{decision.id}-execution-{record_id}",
                x=480, y=observation_y, width=280, height=120,
                title=f"Experiment · {observation.candidate_id}",
                lines=_wrap(params, 37, 4),
                footer=f"execution: {observation.execution.get('execution_model', 'unspecified')}",
                detail=_detail({"intended_params": observation.intended_params, "execution": observation.execution}),
                css="assumption",
            )
            _node(
                nodes, record_id=record_id, dom_id=f"experiment-{decision.id}-test-{record_id}",
                x=860, y=observation_y, width=330, height=120,
                title=f"{record_id} · observed {value}",
                lines=["Accessible result, not evaluator truth.",
                       "Execution details remain separately recorded."],
                footer=f"candidate {observation.candidate_id} · simulated observation",
                detail=_detail(observation), css="observation",
            )
            _edge(edges, source=decision.id, target=record_id, kind="tests",
                  start=(402, decision_y + 30), end=(860, observation_y + 54),
                  via_y=decision_y - 14, note="The experiment this decision ran.")
            edges.append(
                f'<g><title>Recorded execution produces {record_id}; visual sequence, not an evidence claim.</title>'
                f'<path class="edge" stroke="#8c98a8" d="M760,{observation_y + 60} H850"/>'
                f'<text class="muted" x="777" y="{observation_y + 53}">result</text></g>'
            )
        for index, claim_id in enumerate(updates):
            claim = episode.graph.claims[claim_id]
            matching = [r for r in claim.revisions if r.round == decision.round or
                        (r.round == 0 and set(r.evidence) & set(tested))]
            revision = matching[-1] if matching else None
            update_y = decision_y + index * 200
            snapshot = claim.model_copy(update={
                "statement": revision.statement if revision else claim.statement,
                "status": revision.status if revision else claim.status,
                "uncertainties": revision.uncertainties if revision else claim.uncertainties,
            })
            _node(nodes, record_id=claim_id, dom_id=f"experiment-{decision.id}-update-{claim_id}",
                  x=1320, y=update_y, width=548, height=180,
                  title=f"{claim_id} · {snapshot.status}",
                  lines=_wrap(snapshot.statement, 73, 3),
                  footer=revision.reason if revision else "Recorded relationship; revision timing unavailable.",
                  detail=_detail({"claim_id": claim_id, "revision": revision.model_dump() if revision else None}),
                  css="claim" + (" contradicted" if snapshot.status == "contradicted" else ""),
                  badges=_uncertainty_badges(episode, snapshot))
            for edge in episode.graph.edges:
                if edge.source in tested and edge.target == claim_id and edge.kind in EVIDENCE_EDGES:
                    _edge(edges, source=edge.source, target=claim_id, kind=edge.kind,
                          start=(1190, decision_y + 60), end=(1320, update_y + 64), note=edge.note)
            if row_index + 1 < len(rows):
                next_decision = rows[row_index + 1][0]
                declared = any(e.source == next_decision.id and e.target == claim_id
                               and e.kind == "depends_on" for e in episode.graph.edges)
                if declared:
                    bottom = y + row_height - 28 - index * 14
                    next_y = y + row_height + 56
                    edges.append(
                        f'<g><title>{_esc(next_decision.id)} depends_on {_esc(claim_id)}</title>'
                        f'<path class="a-edge edge depends_on" marker-end="url(#arrow-depends_on)" '
                        f'd="M32,{next_y} H18 V{bottom} H1250 V{update_y + 144} H1320"/>'
                        f'<text class="edge-label depends_on" x="635" y="{bottom - 4}">depends_on · next decision</text></g>'
                    )
        y += row_height

    return "".join(header + edges + nodes), height


def to_evidence_svg(episode: Episode) -> str:
    body, height = _evidence_parts(episode)
    return _document(body, EVIDENCE_WIDTH, height,
                     "Claims and evidence graph with explicit support and refutation relationships.")


def to_experiment_svg(episode: Episode) -> str:
    body, height = _experiment_parts(episode)
    return _document(body, EXPERIMENT_WIDTH, height,
                     "Chronological experiment graph with decision reasoning, dependencies and observations.")


def actions_graph(episode: Episode) -> dict:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    observations = sorted(episode.graph.observations.values(), key=lambda o: o.round)
    for observation in observations:
        decision = episode.graph.decisions[f"D{observation.round}"]
        step = episode.trajectory[observation.round - 1]
        nodes.append({
            "id": decision.id, "kind": "action", "round": decision.round,
            "candidate_id": decision.candidate_id, "parameters": observation.intended_params,
            "observation": observation.model_dump(mode="json"),
            "forecast": step.get("pre_experiment_prediction"),
            "uncertainty_records": [
                u.model_dump(mode="json") for u in episode.graph.uncertainties.values()
                if observation.id in u.evidence
            ],
            "data_used": [o.id for o in observations if o.round < observation.round],
            "rationale": decision.justification,
            "prior_gates": decision.prior_gates,
        })
        if len(nodes) > 1:
            edges.append({
                "source": nodes[-2]["id"], "target": decision.id, "kind": "reasoning",
                "reason": decision.justification,
                "qualification": "Chronology, not causation. The GP is refitted using all prior observed results.",
            })
    result = episode.final_result()
    if result:
        nodes.append({"id": "Final", "kind": "selection", **result})
        source = next(n["id"] for n in nodes if n.get("observation", {}).get("id") == result["observation_id"])
        edges.append({"source": source, "target": "Final", "kind": "reasoning", "reason": result["rule"],
                      "qualification": result["uncertainty"]})
    return {"nodes": nodes, "edges": edges}


def _actions_parts(episode: Episode) -> tuple[str, int]:
    graph = actions_graph(episode)
    parts: list[str] = []
    _text(parts, 32, 40, "Actions and reasoning", "title")
    _text(parts, 32, 66, "Action nodes contain observed results. Edge labels explain the next choice. No hidden means.", "muted")
    positions = {}
    for index, node in enumerate(graph["nodes"]):
        y = 110 + index * 310
        positions[node["id"]] = y
        if node["kind"] == "selection":
            title = f"Final · {node['candidate_id']} · {node['value_shown']:.6g} {node['outcome_unit']}"
            lines = _wrap(node["rule"], 84, 2) + _wrap(node["uncertainty"], 84, 4)
            badges = []
        else:
            observation, forecast = node["observation"], node["forecast"]
            title = f"Trial {node['round']} · {node['candidate_id']}"
            lines = _wrap(", ".join(f"{k}={v:g}" for k, v in node["parameters"].items()), 84, 3)
            value = observation["value_shown"]
            lines += [f"Observed: {value:g} {observation['outcome_unit']}" if value is not None else "No observed result."]
            lines += [f"Execution: {observation['execution'].get('execution_model', 'unspecified')}; "
                      f"measurement noise {'enabled' if observation['measurement_noise'] else 'disabled'}."]
            lines += ([f"GP predictive interval [{forecast['interval'][0]:.5g}, {forecast['interval'][1]:.5g}]; "
                       "calibration not established."] if forecast else ["GP forecast: unavailable before the first experiment."])
            badges = [(u["category"], u["status"]) for u in node["uncertainty_records"]]
        _node(parts, record_id=node["id"], dom_id=f"action-{node['id']}", x=32, y=y, width=730, height=218,
              title=title, lines=lines, footer="Select for complete inputs, result and uncertainty.",
              detail=_detail(node), css="observation", badges=badges)
    for edge in graph["edges"]:
        sy, ty = positions[edge["source"]] + 218, positions[edge["target"]]
        _edge(parts, source=edge["source"], target=edge["target"], kind="depends_on",
              start=(770, sy), end=(770, ty), note=f"{edge['reason']} {edge['qualification']}")
        _text(parts, 32, ty - 50, edge["reason"], "node-copy", width=88, limit=2)
    if not graph["nodes"]:
        _text(parts, 32, 130, "No experiments or observed final answer.", "section")
    return "".join(parts), 150 + max(1, len(graph["nodes"])) * 310


def to_actions_svg(episode: Episode) -> str:
    body, height = _actions_parts(episode)
    return _document(body, 900, height, "Actions as nodes and numerical reasoning as labelled edges; final best-observed selection.")


def to_svg(episode: Episode) -> str:
    """Compact dependency tree. An EI edge identifies its incumbent, not its entire GP training set."""
    actions = actions_graph(episode)
    nodes = {n["id"]: n for n in actions["nodes"]}
    children: dict[str, list[str]] = {"Root": []}
    links = []
    previous: list[dict[str, Any]] = []
    for node in actions["nodes"]:
        parent = "Root"
        label = "Seeded initial exploration"
        note = node.get("rationale", "")
        if node["kind"] == "selection":
            parent = next(n["id"] for n in previous if n["observation"]["id"] == node["observation_id"])
            label, note = "Best observed result among tested candidates", node["uncertainty"]
        elif "Highest expected improvement" in node["rationale"] and previous:
            observed = [n for n in previous if n["observation"]["value_shown"] is not None]
            if observed:
                incumbent = (max if episode.task.direction == "maximize" else min)(
                    observed, key=lambda n: n["observation"]["value_shown"],
                )
                parent = incumbent["id"]
                node["ei_reference"] = incumbent["observation"]["id"]
            match = re.search(r"expected improvement \(([^)]+)\)", node["rationale"])
            label = f"Largest expected improvement: {match[1]}" if match else "Largest expected improvement"
            prediction = node["forecast"]
            if prediction:
                label += f" · GP {prediction['mean']:.3g} ± {prediction['latent_sd']:.2g}"
            note += " Parent supplies the best-observed EI reference; the GP uses all prior observations. " \
                    "The ± value is latent SD, not a calibrated confidence bound. No causal or pruning claim."
        elif node["kind"] == "action":
            label = node["rationale"]
        children.setdefault(parent, []).append(node["id"])
        children.setdefault(node["id"], [])
        links.append((parent, node["id"], label, note))
        if node["kind"] == "action":
            previous.append(node)

    claims = [c for c in episode.graph.claims.values() if not c.benchmark_generated
              and c.id not in ("K1", "K_best", "K_calibration")]
    sources = [e for e in episode.graph.evidence_records.values() if e.source == "literature"]
    evidence_lines: dict[str, list[str]] = {}
    heights = {record_id: 136 for record_id in [*nodes, "Root"]}
    evidence_nodes: list[EvidenceRecord | Claim] = [*sources, *claims]
    for record in evidence_nodes:
        lines = _wrap(record.title if isinstance(record, EvidenceRecord) else record.statement,
                      34, 2 if isinstance(record, Claim) and record.belief else 3)
        if isinstance(record, Claim) and record.belief:
            for name, (best, width_fraction) in record.belief.items():
                lines.extend(_wrap(f"{name}≈{best:g}; width {width_fraction:.2g}", 34, 2))
            lines.append(f"Initial trust: {record.trust:.2f}" if record.trust is not None else "Initial trust: unavailable")
        evidence_lines[record.id] = lines
        heights[record.id] = max(136, 76 + 15 * len(lines))
    left = 812 if claims or sources else 32
    positions: dict[str, tuple[int, int]] = {}
    leaf = 0

    def place(record_id: str, depth: int) -> int:
        nonlocal leaf
        rows = [place(child, depth + 1) for child in children[record_id]]
        y = (rows[0] + rows[-1]) // 2 if rows else 42 + leaf * 175
        if not rows:
            leaf += 1
        positions[record_id] = (left + depth * 390, y)
        return y

    place("Root", 0)
    for index, source in enumerate(sources):
        positions[source.id] = (32, 42 + index * 175)
    claim_y = 42
    for claim in claims:
        positions[claim.id] = (412, claim_y)
        claim_y += heights[claim.id] + 39
    height = max(y + heights[record_id] for record_id, (_, y) in positions.items()) + 44
    width = max(x for x, _ in positions.values()) + 280
    edges: list[str] = []
    parts: list[str] = []
    for parent, target, label, note in links:
        sx, sy = positions[parent]
        tx, ty = positions[target]
        _edge(edges, source=parent, target=target, kind="depends_on",
              start=(sx + 240, sy + 68), end=(tx, ty + 68), note=note, label=label)
    aliases = {o.id: f"D{o.round}" for o in episode.graph.observations.values()}
    for edge in episode.graph.edges:
        source_id, target_id = aliases.get(edge.source, edge.source), aliases.get(edge.target, edge.target)
        if source_id not in positions or target_id not in positions or edge.kind == "tests":
            continue
        sx, sy = positions[source_id]
        tx, ty = positions[target_id]
        label = f"{edge.kind.replace('_', ' ')}: {edge.note}" if edge.note else edge.kind.replace("_", " ")
        if edge.kind == "depends_on" and edge.note.startswith("gate "):
            label = f"Literature prior: {edge.note}"
        _edge(edges, source=source_id, target=target_id, kind=edge.kind,
              start=(sx + 240, sy + heights[source_id] / 2), end=(tx, ty + heights[target_id] / 2),
              note=edge.note, label=label)

    x, y = positions["Root"]
    _node(parts, record_id="Root", dom_id="reasoning-Root", x=x, y=y, width=240, height=136,
          title=episode.task.name, lines=[f"GP-BO · {episode.task.direction}",
                                        f"{len(previous)} experiments",
                                        f"Stop: {episode.stop_reason.replace('_', ' ')}"],
          footer="Select nodes for details",
          detail=_detail({"briefing": episode.task.briefing(episode.config.budget),
                          "limitations": episode.task.limitations,
                          "termination": {"reason": episode.stop_reason, "elapsed_s": episode.elapsed_s,
                                          "model_tokens": episode.provider_metadata.get("tokens", 0)},
                          "assumptions": [a.model_dump() for a in episode.graph.assumptions.values()]}),
          css="assumption", centered=True)
    for record_id, node in nodes.items():
        x, y = positions[record_id]
        if node["kind"] == "selection":
            title = f"Final · {node['candidate_id']}"
            lines = [f"Observed: {node['value_shown']:.5g}", "Greedy best observed"]
            footer = "Not a proven optimum"
        else:
            observation = node["observation"]
            title = f"Trial {node['round']} · {node['candidate_id']}"
            lines = _wrap(" · ".join(f"{k.replace('_conc', '')}={v:g}"
                                    for k, v in node["parameters"].items()), 34, 3)
            value = observation["value_shown"]
            lines += [f"Observed: {value:.5g}" if value is not None else "No observed result"]
            footer = "Simulated noise · model uncertain" if observation["measurement_noise"] else "Model uncertain"
        _node(parts, record_id=record_id, dom_id=f"reasoning-{record_id}", x=x, y=y, width=240, height=136,
              title=title, lines=lines, footer=footer, detail=_detail(node), css="observation", centered=True)
    for record in evidence_nodes:
        x, y = positions[record.id]
        is_source = isinstance(record, EvidenceRecord)
        title = f"{record.id} · {'Source' if is_source else 'Claim'}"
        lines = evidence_lines[record.id]
        dependency = any(e.kind == "depends_on" and e.source in nodes and e.target == record.id
                         for e in episode.graph.edges)
        if isinstance(record, EvidenceRecord):
            footer = "Source details"
        else:
            footer = ("Numerical prior used in acquisition" if record.belief and dependency
                      else "Annotation only" if dependency else "Not used in acquisition")
        detail = record.model_dump(mode="json")
        if isinstance(record, Claim):
            detail["uncertainty_records"] = [
                episode.graph.uncertainties[i].model_dump(mode="json") for i in record.uncertainties
            ]
        _node(parts, record_id=record.id, dom_id=f"reasoning-{record.id}", x=x, y=y, width=240, height=heights[record.id],
              title=title, lines=lines, footer=footer, detail=_detail(detail),
              css="source" if is_source else "claim", centered=True)
    if not previous:
        _text(parts, left, height - 12, "No experiments or observed final answer.", "muted")
    return _document("".join(edges + parts), width, height,
                     "Actions and reasoning. Branches identify the best-observed EI reference; the GP uses all prior data.")


def _namespace_svg(svg: str, prefix: str) -> str:
    for record_id in set(re.findall(r' id="([^"]+)"', svg)):
        svg = svg.replace(f' id="{record_id}"', f' id="{prefix}-{record_id}"')
        svg = svg.replace(f'url(#{record_id})', f'url(#{prefix}-{record_id})')
    return svg.replace('data-a-diagram-canvas="true"', 'class="a-diagram__canvas" data-a-diagram-canvas="true"')


HTML_STYLE = """
<style>
.a-shell{max-width:none;padding:12px 20px}
.a-header{position:fixed;z-index:9;right:12px;top:12px;padding:0;border:0}
.a-header>div,.a-header__subtitle,.a-footnote{display:none}
.a-header .a-btn{width:28px;height:28px}
.a-diagram{border:0;border-radius:0;background:transparent;box-shadow:none;padding:0}
.graph-toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin:8px 0}
.graph-toolbar .a-btn{padding:3px 9px;min-height:28px}
.graph-scroll{height:calc(100vh - 56px);min-height:340px;overflow:hidden;overscroll-behavior:contain}
.graph-scroll .a-diagram__canvas{width:100%;height:100%;touch-action:none}
.a-diagram .a-inspector{position:fixed;z-index:10;right:12px;top:70px;width:min(360px,calc(100vw - 24px));
 max-height:calc(100vh - 90px);overflow:auto;background:rgb(var(--bg-elevated));padding:16px;
 border:1px solid rgb(var(--border-primary));box-shadow:var(--shadow-l2);border-radius:3px}
.graph-detail-head{display:flex;justify-content:space-between;align-items:center;gap:12px}
.graph-detail-head p{margin:0}
.graph-record-detail{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;line-height:17px}
.graph-fields{font-size:12px;line-height:18px;margin:12px 0}
.graph-fields dt{font-weight:600;margin-top:12px}
.graph-fields dd{margin:3px 0 0;white-space:pre-wrap;overflow-wrap:anywhere}
.graph-related{display:flex;flex-wrap:wrap;gap:6px}
.a-diagram .canvas,.a-diagram .record-box{fill:rgb(var(--bg-elevated))}
.a-diagram .record.observation .record-box{fill:rgb(var(--text-green)/.045)}
.a-diagram .record.source .record-box{fill:rgb(var(--text-orange)/.055)}
.a-diagram .record.claim .record-box{fill:rgb(var(--text-blue)/.045)}
.a-diagram .record.assumption .record-box{fill:rgb(var(--bg-wash));stroke-dasharray:none}
.a-diagram svg text{fill:rgb(var(--text-primary))}
.a-diagram svg .muted,.a-diagram svg .badge-text{fill:rgb(var(--text-secondary))}
.a-diagram svg .node-title{font-size:13px;font-weight:600}
.a-diagram svg .node-copy{font-size:12px}
.a-diagram svg .muted{font-size:10px}
.a-diagram svg .edge-label{font-size:10px;font-weight:400}
.a-diagram svg .edge{stroke-width:1}
.a-diagram [data-a-edge]:hover .edge,.a-diagram [data-a-edge]:focus .edge{stroke-width:2.5}
.a-diagram svg .depends_on{stroke-dasharray:none}
.a-diagram .record-selected .record-box{stroke:rgb(var(--text-accent));stroke-width:2.8}
.a-diagram .record-related .record-box{stroke:rgb(var(--text-accent));stroke-width:1.8;stroke-dasharray:3 2}
.a-diagram .edge-unrelated{opacity:.2}
@media(max-width:500px){.a-shell{padding:8px}}
@media print{
 .graph-scroll{height:auto;min-height:0;overflow:visible}
 .graph-scroll .a-diagram__canvas{width:100%;height:auto}
 .a-diagram .a-inspector{position:static;width:auto;max-height:none;box-shadow:none}
 [data-a-diagram-viewport]{transform:none!important}
 .edge-unrelated{opacity:1!important}
}
</style>
"""

HTML_SCRIPT = """
<script>
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-episode]").forEach(episode => {
    const nodes = [...episode.querySelectorAll("[data-a-node]")];
    const edges = [...episode.querySelectorAll("[data-edge-source]")];
    const inspector = episode.querySelector("[data-a-inspector]");
    const number = value => typeof value === "number" ? Number(value.toPrecision(6)) : value;
    let selected = null;
    const close = () => {
      inspector.hidden = true;
      nodes.forEach(node => node.classList.remove("record-selected", "record-related"));
      edges.forEach(edge => edge.classList.remove("edge-unrelated"));
      if (selected) selected.focus({preventScroll:true});
    };
    inspector.querySelector("[data-close-detail]").addEventListener("click", close);
    episode.addEventListener("keydown", event => {
      if (event.key === "Escape") close();
    });
    const select = node => {
      selected = node;
      const id = node.dataset.recordId;
      const isEdge = node.hasAttribute("data-a-edge");
      const related = new Set();
      edges.forEach(edge => {
        const connected = isEdge ? edge === node : edge.dataset.edgeSource === id || edge.dataset.edgeTarget === id;
        edge.classList.toggle("edge-unrelated", !connected);
        if (connected) { related.add(edge.dataset.edgeSource); related.add(edge.dataset.edgeTarget); }
      });
      nodes.forEach(other => {
        other.classList.toggle("record-selected", other.dataset.recordId === id);
        other.classList.toggle("record-related", other.dataset.recordId !== id && related.has(other.dataset.recordId));
      });
      inspector.hidden = false;
      inspector.querySelector("[data-a-inspector-title]").textContent = node.dataset.aTitle;
      const record = JSON.parse(node.dataset.aDetail);
      const body = inspector.querySelector("[data-a-inspector-body]");
      body.replaceChildren();
      const fields = document.createElement("dl");
      fields.className = "graph-fields";
      const add = (label, value) => {
        if (value === undefined || value === null || value === "") return;
        const term = document.createElement("dt");
        term.textContent = label;
        const description = document.createElement("dd");
        description.textContent = typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
        fields.append(term, description);
      };
      if (record.kind === "action") {
        add("Intended settings", Object.entries(record.parameters).map(([k,v]) => `${k}: ${v}`).join("\\n"));
        add("Observed result", `${number(record.observation.value_shown) ?? "Unavailable"} ${record.observation.outcome_unit}`);
        add("Choice rationale", record.rationale);
        if (record.forecast) {
          add("GP mean", number(record.forecast.mean));
          add("Latent SD / noise SD", `${number(record.forecast.latent_sd)} / ${number(record.forecast.noise_sd)}`);
          add("95% predictive interval · calibration not established", record.forecast.interval.map(number).join(" – "));
        }
        add("EI reference observation", record.ei_reference);
        add("All observations used by GP", record.data_used.join(", ") || "None");
        add("Learned literature gates · signed coefficients, not probabilities", record.prior_gates);
      } else {
        Object.entries(record).filter(([key]) => !["metadata", "revisions", "uncertainty_records"].includes(key))
          .forEach(([key,value]) => add(key.replaceAll("_", " "), value));
      }
      (record.uncertainty_records || []).forEach(item => add(`${item.category} uncertainty`, item.statement));
      body.append(fields);
      const raw = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "Full record";
      const pre = document.createElement("pre");
      pre.className = "graph-record-detail";
      pre.textContent = node.dataset.aDetail;
      raw.append(summary, pre);
      body.append(raw);
      const links = inspector.querySelector("[data-related-records]");
      links.replaceChildren();
      [...related].filter(key => key !== id).forEach(key => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "a-btn a-btn--outline";
        button.textContent = key;
        button.addEventListener("click", () => {
          const match = nodes.find(other => other.dataset.recordId === key);
          if (match) { select(match); match.focus({preventScroll:true}); }
        });
        links.append(button);
      });
    };
    episode.addEventListener("click", event => {
      const node = event.target.closest("[data-a-node],[data-a-edge]");
      if (node) select(node);
    });
    episode.addEventListener("keydown", event => {
      if (event.key !== "Enter" && event.key !== " ") return;
      const node = event.target.closest("[data-a-node],[data-a-edge]");
      if (node) { event.preventDefault(); select(node); }
    });
  });
});
</script>
"""


def html_body(episode: Episode, prefix: str = "episode") -> str:
    return (
        f'<article data-episode="{_esc(prefix)}">'
        '<div class="a-diagram" data-a-diagram tabindex="0" aria-label="Actions and reasoning">'
        '<div class="graph-toolbar a-no-print">'
        '<button type="button" class="a-btn a-btn--outline" data-a-zoom="out" aria-label="Zoom out">−</button>'
        '<button type="button" class="a-btn a-btn--outline" data-a-zoom="in" aria-label="Zoom in">+</button>'
        '<button type="button" class="a-btn a-btn--outline" data-a-zoom="reset">Fit</button></div>'
        f'<div class="graph-scroll">{_namespace_svg(to_svg(episode), f"{prefix}-reasoning")}</div>'
        '<aside class="a-inspector" data-a-inspector aria-label="Selected record" aria-live="polite" hidden>'
        '<div class="graph-detail-head"><p class="a-inspector__title" data-a-inspector-title></p>'
        '<button type="button" class="a-btn a-btn--outline" data-close-detail aria-label="Close details">Close</button></div>'
        '<div data-a-inspector-body></div>'
        '<div class="graph-related" data-related-records aria-label="Connected records"></div>'
        '</aside></div></article>'
    )


def to_html(episode: Episode) -> str:
    template = files("epistemic").joinpath("viewer.html").read_text(encoding="utf-8")
    return template.replace("__EPISTEMIC_BODY__", HTML_STYLE + html_body(episode) + HTML_SCRIPT)


def to_markdown(episode: Episode) -> str:
    chunks = [
        f"# {episode.config.task}: experimental-design audit",
        "Simulated observations. Dependencies are not claims of causation.",
        "## Dataset contract", episode.task.briefing(episode.config.budget),
        f"Provenance: {episode.task.provenance}",
        *[f"- {limitation}" for limitation in episode.task.limitations],
        "## Greedy final answer", "```json", json.dumps(episode.final_result(), indent=2), "```",
        "## Experiment timeline",
    ]
    for step in episode.trajectory:
        chunks += [f"### Round {step['round']}", "```json", json.dumps(step, indent=2), "```"]
    chunks += [
        "## Evidence records and full revisions",
        "```json", episode.graph.model_dump_json(indent=2), "```",
    ]
    return "\n\n".join(chunks) + "\n"
