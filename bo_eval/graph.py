"""The auditable record: experiments, reasoning, cited priors, closures and final selection."""

from html import escape
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Node(BaseModel):
    id: str
    params: dict[str, float] | None = None
    result: float | None = None
    closed: bool = False
    closed_reason: str | None = None


class Prior(BaseModel):
    """LLM belief about where the optimum lies: param -> (best, width as a fraction of the range)."""

    id: str
    after: str
    belief: dict[str, tuple[float, float]]
    reasoning: str
    trust: float = 1.0


class Evidence(BaseModel):
    """A cited literature claim from the research agent, with how far it can be trusted for this system."""

    id: str
    claim: str
    sources: list[str]
    trust: float
    trust_reason: str
    about: list[str] = Field(default_factory=list)


class Edge(BaseModel):
    source: str
    target: str
    reasoning: str
    kind: Literal["experiment", "reasoning"] = "experiment"


def _root() -> dict[str, Node]:
    return {"root": Node(id="root")}


class ReasoningGraph(BaseModel):
    nodes: dict[str, Node] = Field(default_factory=_root)
    edges: list[Edge] = Field(default_factory=list)
    priors: list[Prior] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    selected: str | None = None
    decision: str = ""

    @property
    def experiments(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.params is not None]

    def open_leaves(self) -> list[str]:
        """Experiment nodes that were neither extended nor closed."""
        sources = {e.source for e in self.edges if e.kind == "experiment"}
        return [n.id for n in self.experiments if not n.closed and n.id not in sources]

    def close(self, node_id: str, reason: str) -> list[str]:
        """Close a branch (Hintikka-style): the node and all its descendants."""
        if not reason.strip():
            raise ValueError("A branch closure needs a reason.")
        closed, stack = [], [node_id]
        while stack:
            nid = stack.pop()
            if self.nodes[nid].closed:
                continue
            self.nodes[nid].closed, self.nodes[nid].closed_reason = True, reason
            closed.append(nid)
            stack += [e.target for e in self.edges if e.source == nid and e.kind == "experiment"]
        return closed

    def _label(self, n: Node) -> str:
        if n.params is None:
            return "root"
        p = ", ".join(f"{k}={v:g}" for k, v in n.params.items())
        return f"{n.id} [{p}] -> {n.result:.4g}" + (" (CLOSED)" if n.closed else "")

    def parent(self, node_id: str) -> str:
        return next((e.source for e in self.edges if e.target == node_id and e.kind == "experiment"), "root")

    def to_text(self) -> str:
        lines = [self._label(n) for n in self.nodes.values()]
        lines += [f"{e.source} -> {e.target}: {e.reasoning}" for e in self.edges]
        lines += [f"{p.id} (set after {p.after}, trust {p.trust:g}): {p.belief} because {p.reasoning}" for p in self.priors]
        lines += [f"{v.id} [trust {v.trust:g}] {v.claim} ({'; '.join(v.sources)}) re {v.about}: {v.trust_reason}" for v in self.evidence]
        closed = [f"{n.id}: {n.closed_reason}" for n in self.nodes.values() if n.closed]
        if closed:
            lines += ["Closed branches:"] + closed
        if self.decision:
            lines += [f"Final selection ({self.selected or 'untested'}): {self.decision}"]
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        def q(s: str) -> str:
            return escape(s, quote=True)

        lines = ["graph TD"]
        for n in self.nodes.values():
            if n.params is None:
                lines.append(f'  {n.id}(("start"))')
            else:
                p = "<br/>".join(f"{q(k)}={v:g}" for k, v in n.params.items())
                lines.append(f'  {n.id}["{n.id}<br/>{p}<br/><b>{n.result:.4g}</b>"]')
        lines += [
            f'  {e.source} {"-->" if e.kind == "experiment" else "-.->"}|"{q(e.reasoning)}"| {e.target}'
            for e in self.edges
        ]
        for p in self.priors:
            b = "<br/>".join(f"{q(k)}≈{v:g}, width={w:g}" for k, (v, w) in p.belief.items())
            lines.append(f'  {p.id}[/"{p.id}<br/>{b}"/]')
            lines.append(f'  {p.after} -.->|"{q(p.reasoning)}"| {p.id}')
        for v in self.evidence:
            lines.append(f'  {v.id}{{{{"{v.id} trust {v.trust:g}<br/>{q(v.claim)}<br/><i>{q("; ".join(v.sources))}</i>"}}}}')
            lines += [f"  {v.id} -.- {a}" for a in v.about if a in self.nodes or any(a == p.id for p in self.priors)]
        if self.evidence:
            lines.append("  classDef evidence fill:#fff4dd,stroke:#c80")
            lines.append(f"  class {','.join(v.id for v in self.evidence)} evidence")
        if self.priors:
            lines.append("  classDef prior fill:#e8f0ff,stroke:#36c")
            lines.append(f"  class {','.join(p.id for p in self.priors)} prior")
        closed = [n.id for n in self.nodes.values() if n.closed]
        if closed:
            lines.append("  classDef closed fill:#eee,stroke:#999,stroke-dasharray:4")
            lines.append(f"  class {','.join(closed)} closed")
        if self.selected:
            lines += ["  classDef selected fill:#dcfce7,stroke:#15803d,stroke-width:3",
                      f"  class {self.selected} selected"]
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """A diagram plus lossless explanations; no renderer-specific truncation of the audit trail."""
        lines = ["# Reasoning graph", "", f"```mermaid\n{self.to_mermaid()}\n```"]
        if self.decision:
            lines += ["", "## Final selection", "", self.decision]
        for n in self.experiments:
            lines += ["", f"## {n.id}", "", self._label(n)]
            lines += [f"\n**From {e.source} ({e.kind}):** {e.reasoning}"
                      for e in self.edges if e.target == n.id]
            if n.closed:
                lines += [f"\n**Not pursued:** {n.closed_reason}"]
        for p in self.priors:
            lines += ["", f"## {p.id}: prior", "", f"Set after {p.after}; initial trust {p.trust:g}.",
                      f"\nBelief: {p.belief}", f"\nReasoning: {p.reasoning}"]
        for v in self.evidence:
            lines += ["", f"## {v.id}: evidence", "", v.claim, f"\nTrust {v.trust:g}: {v.trust_reason}",
                      f"\nApplies to: {', '.join(v.about)}"]
            lines += [f"- {source}" for source in v.sources]
        return "\n".join(lines) + "\n"

    def export(self, path: str | Path) -> list[Path]:
        """Write portable JSON, readable Markdown and a self-contained SVG to a filename stem."""
        from bo_eval.render import to_svg

        stem = Path(path)
        stem.parent.mkdir(parents=True, exist_ok=True)
        outputs = {".json": self.model_dump_json(indent=2), ".md": self.to_markdown(), ".svg": to_svg(self)}
        paths = []
        for suffix, content in outputs.items():
            target = stem.with_suffix(suffix)
            target.write_text(content, encoding="utf-8")
            paths.append(target)
        return paths
