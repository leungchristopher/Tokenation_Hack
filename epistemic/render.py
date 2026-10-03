"""Dependency-free audit viewer: timeline, objective curve, decisions, evidence, uncertainty and revisions."""

from __future__ import annotations

import html
import json
import textwrap
from pathlib import Path

from epistemic.loop import Episode
from epistemic.metrics import episode_metrics


def export(episode: Episode, directory: str | Path) -> Path:
    out = episode.save(directory)
    (out / "metrics.json").write_text(json.dumps(episode_metrics(episode), indent=2))
    (out / "graph.svg").write_text(to_svg(episode))
    (out / "audit.md").write_text(to_markdown(episode))
    return out


def to_svg(episode: Episode) -> str:
    rows = len(episode.decisions)
    claims = list(episode.graph.claims.values())
    height = max(570 + rows * 128, 570 + sum(165 + 24 * len(c.revisions) for c in claims))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">',
        '<style>text{font-family:system-ui,sans-serif;fill:#203249;font-size:13px}'
        '.title{font-size:23px;font-weight:700}.head{font-size:16px;font-weight:650}'
        '.muted{fill:#63758a;font-size:12px}</style>',
        '<rect width="1200" height="100%" fill="#f4f7fb"/>',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
        '<path d="M0,0 L8,4 L0,8" fill="none" stroke="#7790ac"/></marker></defs>',
    ]

    def text(x, y, label, style="", tooltip=""):
        safe = html.escape(str(label))
        tip = f"<title>{html.escape(tooltip)}</title>" if tooltip else ""
        svg.append(f'<text x="{x}" y="{y}" class="{style}">{tip}{safe}</text>')

    text(40, 46, "Experimental design · attributable evidence graph", "title")
    text(40, 72, f"{episode.config.task} / {episode.config.policy} · seed {episode.config.seed} · "
         f"{episode.config.execution} execution · {episode.config.feedback} feedback", "muted")
    text(40, 94, "All observations are simulated. Edges express evidence relationships, not causal attribution.", "muted")

    svg.append('<rect x="40" y="114" width="1120" height="210" rx="12" fill="white" stroke="#dce4ef"/>')
    text(60, 142, "Accessible objective curve", "head")
    text(60, 164, f"{episode.task.direction} {episode.task.objective} · unqueried evaluator labels are not displayed", "muted")
    shown = [step["accessible_observation"]["value_shown"] for step in episode.trajectory
             if step.get("accessible_observation") and step["accessible_observation"]["value_shown"] is not None]
    if shown:
        best = [(max if episode.task.direction == "maximize" else min)(shown[:i + 1]) for i in range(len(shown))]
        low, high = min(shown), max(shown)
        span = max(high - low, 1.0)
        points = [(80 + i * 1000 / max(len(shown) - 1, 1), 288 - 100 * (v - low) / span)
                  for i, v in enumerate(best)]
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        svg.append(f'<polyline points="{poly}" fill="none" stroke="#315dcc" stroke-width="3"/>')
        for i, ((x, y), value) in enumerate(zip(points, shown), start=1):
            svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#315dcc"><title>'
                       f'Trial {i}; shown {value:.4g}; best shown {best[i - 1]:.4g}</title></circle>')
        text(60, 310, f"Best shown {best[-1]:.4g}. Full predictions and hidden-truth scoring are separate.", "muted")
    else:
        text(60, 250, "No outcome feedback was available.")

    text(40, 362, "Experiment timeline", "head")
    text(676, 362, "Claims · uncertainty · revisions", "head")
    positions = {}
    for i, decision in enumerate(episode.decisions):
        y = 388 + 128 * i
        observation = episode.graph.observations.get(f"O{decision.round}")
        positions[decision.id] = (40, y)
        step = next(s for s in episode.trajectory if s["round"] == decision.round)
        body = json.dumps({"decision": decision.model_dump(),
                           "accessible_observation": step.get("accessible_observation"),
                           "response_uncertainty": step.get("pre_experiment_prediction"),
                           "model_uncertainty": step.get("state_revision", {}).get("model_uncertainty")},
                          indent=2)
        svg.append(f'<g><title>{html.escape(body)}</title><rect x="40" y="{y}" width="578" height="112"'
                   ' rx="8" fill="white" stroke="#d1ddec"/></g>')
        text(56, y + 22, f"{decision.id} → {decision.candidate_id} · targeting {decision.targeted_uncertainty}", "head")
        for j, line in enumerate(textwrap.wrap(decision.justification, 77)[:2]):
            text(56, y + 44 + j * 17, line)
        outcome = "missing" if observation is None or observation.value_shown is None else f"{observation.value_shown:.5g}"
        forecast = f"{decision.prediction:.5g}" if decision.prediction is not None else "unavailable"
        forecast_label = "LLM point forecast" if decision.prediction_source == "llm" else "numerical forecast"
        text(56, y + 84, f"{observation.id if observation else 'no observation'} · shown {outcome} · "
             f"{forecast_label} {forecast}", "muted")
        text(56, y + 101, f"Evidence: {', '.join(decision.claims) or 'none cited'} · assumptions: "
             f"{', '.join(decision.assumptions) or 'none cited'}", "muted")
        if i:
            svg.append(f'<path d="M322,{y - 16} L322,{y}" stroke="#7790ac" marker-end="url(#arrow)"/>')

    y = 388
    for claim in claims:
        positions[claim.id] = (676, y)
        color = "#b83d47" if claim.status == "contradicted" else "#315dcc"
        h = 165 + 24 * len(claim.revisions)
        tooltip = json.dumps(claim.model_dump(), indent=2)
        svg.append(f'<g><title>{html.escape(tooltip)}</title><rect x="676" y="{y}" width="484" height="{h - 12}"'
                   f' rx="8" fill="white" stroke="{color}"/></g>')
        text(692, y + 23, f"{claim.id} · {claim.source} · {claim.status}", "head")
        for j, line in enumerate(textwrap.wrap(claim.statement, 63)[:3]):
            text(692, y + 46 + 17 * j, line)
        text(692, y + 105, f"Evidence: {', '.join(claim.evidence) or 'source reference'}", "muted")
        supporting = [e.source for e in episode.graph.edges if e.target == claim.id and e.kind == "supports"]
        contradicting = [e.source for e in episode.graph.edges if e.target == claim.id and e.kind == "contradicts"]
        text(692, y + 124, f"Supports: {', '.join(supporting) or 'none'} · contradicts: "
             f"{', '.join(contradicting) or 'none'}", "muted")
        for j, revision in enumerate(claim.revisions):
            label = f"r{j} · trial {revision.round}: {revision.status} · {revision.reason[:47]}"
            text(692, y + 147 + 24 * j, label, "muted", json.dumps(revision.model_dump()))
        y += h
    for edge in episode.graph.edges:
        if (edge.kind != "depends_on" or edge.source not in episode.graph.decisions
                or edge.target not in episode.graph.claims):
            continue
        _, dy = positions[edge.source]
        _, cy = positions[edge.target]
        label = html.escape(f"{edge.source} depends_on {edge.target}")
        svg.append(f'<path d="M618,{dy + 36} C646,{dy + 36} 650,{cy + 26} 676,{cy + 26}"'
                   ' fill="none" stroke="#97a8c0" stroke-dasharray="4 4" marker-end="url(#arrow)">'
                   f'<title>{label}</title></path>')
    text(40, height - 90, "Uncertainty remains separate: response distribution · delivery report · model diagnostics · evidence status.", "muted")
    text(40, height - 68, "Unresolved explanations for surprises: execution error, observation noise, or GP misspecification.", "muted")
    text(40, height - 46, "Hover a record for full scope, supporting evidence, alternatives, predictions and revisions.", "muted")
    text(40, height - 24, "The complete audit is also exported as Markdown and JSON; evaluator truth is a separate file.", "muted")
    svg.append("</svg>")
    return "\n".join(svg)


def to_markdown(episode: Episode) -> str:
    chunks = [f"# {episode.config.task}: experimental-design audit",
              "Simulated observations. Dependencies are not claims of causation.",
              "## Dataset contract", episode.task.briefing(episode.config.budget),
              f"Provenance: {episode.task.provenance}",
              *[f"- {limitation}" for limitation in episode.task.limitations],
              "## Experiment timeline"]
    for step in episode.trajectory:
        chunks += [f"### Round {step['round']}", "```json", json.dumps(step, indent=2), "```"]
    chunks += ["## Evidence records and full revisions", "```json", episode.graph.model_dump_json(indent=2), "```"]
    return "\n\n".join(chunks) + "\n"
