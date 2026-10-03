"""An append-only evidence graph: observations, claims, assumptions, decisions and four edge types."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EdgeKind = Literal["supports", "contradicts", "depends_on", "tests"]
Status = Literal["open", "supported", "contradicted", "retired"]


class FrozenDict(dict):
    """JSON-serialisable immutable tool metadata, including nested containers."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("Tool outputs are immutable.")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable

    def __deepcopy__(self, memo):
        return self


def _freeze(value):
    if isinstance(value, dict):
        return FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


class Observation(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    """An immutable tool output. Model-written text must never be stored here."""

    id: str
    round: int
    candidate_id: str
    intended_params: dict[str, float]
    execution: dict[str, Any]
    value_shown: float | None
    outcome_unit: str
    simulated: bool = True
    kind: Literal["observation"] = "observation"

    @field_validator("intended_params", "execution")
    @classmethod
    def freeze_metadata(cls, value):
        return _freeze(value)


class Revision(BaseModel):
    model_config = ConfigDict(frozen=True)
    round: int
    statement: str
    status: Status
    reason: str
    scope: str = ""
    discriminating_result: str = ""
    evidence: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    id: str
    statement: str
    scope: str
    source: Literal["literature", "model_conjecture", "measurement"]
    evidence: list[str] = Field(default_factory=list)
    status: Status = "open"
    discriminating_result: str = Field(min_length=1)
    reference: str = ""
    benchmark_generated: bool = False
    revisions: list[Revision] = Field(default_factory=list)
    unresolved_transfer_assumptions: list[str] = Field(default_factory=list)
    kind: Literal["claim"] = "claim"

    @model_validator(mode="after")
    def cited_literature(self):
        if self.source == "literature" and not self.reference:
            raise ValueError("Literature claims need an attributable source.")
        return self


class Assumption(BaseModel):
    id: str
    statement: str
    scope: str
    status: Status = "open"
    kind: Literal["assumption"] = "assumption"


class Decision(BaseModel):
    id: str
    round: int
    candidate_id: str
    claims: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    targeted_uncertainty: Literal["response", "execution", "model", "evidence"] = "response"
    justification: str = ""
    prediction: float | None = None
    prediction_source: Literal["llm", "numerical_model", "unavailable"] = "unavailable"
    search_query: str | None = None
    implications: str = ""
    policy: str = "random"
    valid_llm_output: bool | None = None
    kind: Literal["decision"] = "decision"


class Edge(BaseModel):
    model_config = ConfigDict(frozen=True)
    source: str
    target: str
    kind: EdgeKind
    note: str = ""


class EvidenceGraph(BaseModel):
    """References are validated; claims are revised by appending, never by overwriting."""

    observations: dict[str, Observation] = Field(default_factory=dict)
    claims: dict[str, Claim] = Field(default_factory=dict)
    assumptions: dict[str, Assumption] = Field(default_factory=dict)
    decisions: dict[str, Decision] = Field(default_factory=dict)
    edges: list[Edge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self):
        ids = [i for table in (self.observations, self.claims, self.assumptions, self.decisions) for i in table]
        if len(set(ids)) != len(ids):
            raise ValueError("Record IDs must be globally unique.")
        for claim in self.claims.values():
            self._require(*claim.evidence)
            for revision in claim.revisions:
                self._require(*revision.evidence)
        for decision in self.decisions.values():
            self._require(*decision.claims, *decision.assumptions, *decision.evidence)
        for edge in self.edges:
            self._require(edge.source, edge.target)
        return self

    def __contains__(self, record_id: str) -> bool:
        return any(record_id in table for table in (self.observations, self.claims, self.assumptions, self.decisions))

    def _require(self, *ids: str) -> None:
        unknown = [i for i in ids if i not in self]
        if unknown:
            raise KeyError(f"Unknown record reference(s): {unknown}")

    def add_observation(self, observation: Observation) -> Observation:
        if observation.id in self:
            raise ValueError(f"Observation {observation.id} already recorded; observations are immutable.")
        self.observations[observation.id] = observation
        return observation

    def add_claim(self, claim: Claim, round: int = 0) -> Claim:
        self._require(*claim.evidence)
        if claim.id in self:
            raise ValueError(f"Claim {claim.id} already exists; use revise().")
        claim.revisions.append(Revision(round=round, statement=claim.statement, status=claim.status,
                                        scope=claim.scope, discriminating_result=claim.discriminating_result,
                                        reason="initial record", evidence=list(claim.evidence)))
        self.claims[claim.id] = claim
        return claim

    def add_assumption(self, assumption: Assumption) -> Assumption:
        if assumption.id in self:
            raise ValueError(f"ID {assumption.id} already exists.")
        self.assumptions[assumption.id] = assumption
        return assumption

    def add_decision(self, decision: Decision) -> Decision:
        self._require(*decision.claims, *decision.assumptions, *decision.evidence)
        if decision.id in self:
            raise ValueError(f"ID {decision.id} already exists.")
        self.decisions[decision.id] = decision
        return decision

    def revise(self, claim_id: str, round: int, status: Status, reason: str,
               evidence: list[str] | None = None, statement: str | None = None,
               scope: str | None = None, discriminating_result: str | None = None) -> Claim:
        claim = self.claims[claim_id]
        evidence = evidence or []
        self._require(*evidence)
        claim.statement = statement or claim.statement
        claim.status = status
        claim.scope = scope or claim.scope
        claim.discriminating_result = discriminating_result or claim.discriminating_result
        claim.evidence = sorted(set(claim.evidence) | set(evidence))
        claim.revisions.append(Revision(round=round, statement=claim.statement, status=status,
                                        scope=claim.scope, discriminating_result=claim.discriminating_result,
                                        reason=reason, evidence=evidence))
        return claim

    def link(self, source: str, target: str, kind: EdgeKind, note: str = "") -> Edge:
        self._require(source, target)
        edge = Edge(source=source, target=target, kind=kind, note=note)
        self.edges.append(edge)
        return edge

    def contradictions(self) -> list[Edge]:
        return [e for e in self.edges if e.kind == "contradicts"]

    def neighbours(self, record_id: str) -> list[Edge]:
        return [e for e in self.edges if record_id in (e.source, e.target)]

    def view(self, round: int, recent: int = 4, with_edges: bool = True) -> dict:
        """A compact relevant slice; the whole graph is never resent each round."""
        observations = sorted(self.observations.values(), key=lambda o: o.round)[-recent:]
        active = [c for c in self.claims.values() if c.status != "retired" and c.id != "K1"]
        claims = [self.claims["K1"]] if "K1" in self.claims else []
        claims += sorted(active, key=lambda c: c.revisions[-1].round if c.revisions else 0, reverse=True)[:7]
        slice_: dict[str, Any] = {
            "round": round,
            "recent_observations": [
                {"id": o.id, "candidate_id": o.candidate_id, "value_shown": o.value_shown,
                 "execution": o.execution} for o in observations
            ],
            "evidence_uncertainty": [
                {"id": c.id, "statement": c.statement, "scope": c.scope, "source": c.source,
                 "status": c.status, "evidence": c.evidence, "discriminating_result": c.discriminating_result,
                 "revisions": len(c.revisions),
                 "reference": "Unverified working hypothesis; not a cited publication." if c.benchmark_generated else c.reference,
                 "unresolved_transfer_assumptions": c.unresolved_transfer_assumptions,
                 "supporting_evidence": [e.source for e in self.edges if e.target == c.id and e.kind == "supports"],
                 "contradicting_evidence": [e.source for e in self.edges if e.target == c.id and e.kind == "contradicts"]}
                for c in claims
            ],
            "assumptions": [
                {"id": a.id, "statement": a.statement, "status": a.status} for a in self.assumptions.values()
            ],
        }
        if with_edges:
            relevant = {o.id for o in observations} | {c.id for c in claims} | set(self.assumptions)
            slice_["relationships"] = [
                e.model_dump() for e in self.edges if e.source in relevant and e.target in relevant
            ]
        return slice_
