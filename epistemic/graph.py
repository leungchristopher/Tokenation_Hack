"""Raw evidence and uncertainty snapshots are immutable; claims have append-only revisions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EdgeKind = Literal["supports", "qualifies", "contradicts", "not_transferable", "depends_on", "tests"]
Status = Literal["open", "supported", "contradicted", "retired"]
UncertaintyKind = Literal["source", "transfer", "mechanistic", "response", "model", "execution"]


class FrozenDict(dict):
    """JSON-serialisable immutable tool metadata, including nested containers."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("Tool outputs are immutable.")

    __setitem__ = __delitem__ = __ior__ = clear = pop = popitem = setdefault = update = _immutable

    def __deepcopy__(self, memo):
        return self


def _freeze(value):
    if isinstance(value, dict):
        return FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False, validate_default=True)

    id: str
    title: str
    abstract: str = ""
    reference: str = Field(min_length=1)
    query: str = ""
    source: Literal["literature", "dataset"] = "literature"
    metadata: dict[str, Any] = Field(default_factory=dict)
    kind: Literal["evidence"] = "evidence"

    @field_validator("metadata")
    @classmethod
    def freeze_metadata(cls, value):
        return _freeze(value)


class UncertaintyRecord(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    id: str
    category: UncertaintyKind
    statement: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    status: Literal["unresolved", "bounded", "not_applicable"] = "unresolved"
    evidence: tuple[str, ...] = ()
    round: int = 0
    kind: Literal["uncertainty"] = "uncertainty"


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
    measurement_noise: bool = True
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
    uncertainties: list[str] = Field(default_factory=list)


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
    uncertainties: list[str] = Field(default_factory=list)
    belief: dict[str, tuple[float, float]] = Field(default_factory=dict)
    trust: float | None = Field(default=None, ge=0.0, le=1.0)
    trust_reason: str = ""
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
    prediction_source: Literal["numerical_model", "unavailable"] = "unavailable"
    search_query: str | None = None
    implications: str = ""
    policy: str = "gp_bo"
    prior_gates: dict[str, tuple[float, float]] = Field(default_factory=dict)
    kind: Literal["decision"] = "decision"


class Edge(BaseModel):
    model_config = ConfigDict(frozen=True)
    source: str
    target: str
    kind: EdgeKind
    note: str = ""


class EvidenceGraph(BaseModel):
    """References are validated; claims are revised by appending, never by overwriting."""

    evidence_records: dict[str, EvidenceRecord] = Field(default_factory=dict)
    uncertainties: dict[str, UncertaintyRecord] = Field(default_factory=dict)
    observations: dict[str, Observation] = Field(default_factory=dict)
    claims: dict[str, Claim] = Field(default_factory=dict)
    assumptions: dict[str, Assumption] = Field(default_factory=dict)
    decisions: dict[str, Decision] = Field(default_factory=dict)
    edges: list[Edge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self):
        ids = [i for table in self._tables() for i in table]
        if len(set(ids)) != len(ids):
            raise ValueError("Record IDs must be globally unique.")
        if any(record_id != record.id for table in self._tables() for record_id, record in table.items()):
            raise ValueError("Table keys must match their record IDs.")
        for claim in self.claims.values():
            self._require(*claim.evidence, *claim.uncertainties)
            if any(i not in self.uncertainties for i in claim.uncertainties):
                raise ValueError("Claim uncertainty IDs must refer to uncertainty records.")
            for revision in claim.revisions:
                self._require(*revision.evidence, *revision.uncertainties)
        for decision in self.decisions.values():
            self._require(*decision.claims, *decision.assumptions, *decision.evidence)
        for uncertainty in self.uncertainties.values():
            self._require(*uncertainty.evidence)
        for edge in self.edges:
            self._require(edge.source, edge.target)
            if edge.kind == "tests" and (edge.source not in self.decisions or edge.target not in self.observations):
                raise ValueError("tests links must connect a decision to its observation.")
        return self

    def _tables(self) -> tuple[dict, ...]:
        return (self.evidence_records, self.uncertainties, self.observations,
                self.claims, self.assumptions, self.decisions)

    def __contains__(self, record_id: str) -> bool:
        return any(record_id in table for table in self._tables())

    def _require(self, *ids: str) -> None:
        unknown = [i for i in ids if i not in self]
        if unknown:
            raise KeyError(f"Unknown record reference(s): {unknown}")

    def add_observation(self, observation: Observation) -> Observation:
        if observation.id in self:
            raise ValueError(f"Observation {observation.id} already recorded; observations are immutable.")
        self.observations[observation.id] = observation
        return observation

    def add_evidence(self, record: EvidenceRecord) -> EvidenceRecord:
        if record.id in self:
            raise ValueError(f"Evidence {record.id} already exists; raw source records are immutable.")
        self.evidence_records[record.id] = record
        return record

    def add_uncertainty(self, record: UncertaintyRecord) -> UncertaintyRecord:
        self._require(*record.evidence)
        if record.id in self:
            raise ValueError(f"Uncertainty {record.id} already exists; add a new snapshot instead.")
        self.uncertainties[record.id] = record
        return record

    def add_claim(self, claim: Claim, round: int = 0) -> Claim:
        self._require(*claim.evidence, *claim.uncertainties)
        if any(i not in self.uncertainties for i in claim.uncertainties):
            raise ValueError("Claim uncertainty IDs must refer to uncertainty records.")
        if claim.id in self:
            raise ValueError(f"Claim {claim.id} already exists; use revise().")
        claim.revisions.append(Revision(round=round, statement=claim.statement, status=claim.status,
                                        scope=claim.scope, discriminating_result=claim.discriminating_result,
                                        reason="initial record", evidence=list(claim.evidence),
                                        uncertainties=list(claim.uncertainties)))
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
               scope: str | None = None, discriminating_result: str | None = None,
               uncertainties: list[str] | None = None) -> Claim:
        claim = self.claims[claim_id]
        evidence = evidence or []
        self._require(*evidence)
        if uncertainties is not None:
            self._require(*uncertainties)
            if any(i not in self.uncertainties for i in uncertainties):
                raise ValueError("Claim uncertainty IDs must refer to uncertainty records.")
        claim.statement = statement or claim.statement
        claim.status = status
        claim.scope = scope or claim.scope
        claim.discriminating_result = discriminating_result or claim.discriminating_result
        claim.evidence = sorted(set(claim.evidence) | set(evidence))
        if uncertainties is not None:
            claim.uncertainties = list(uncertainties)
        claim.revisions.append(Revision(round=round, statement=claim.statement, status=status,
                                        scope=claim.scope, discriminating_result=claim.discriminating_result,
                                        reason=reason, evidence=evidence, uncertainties=list(claim.uncertainties)))
        return claim

    def link(self, source: str, target: str, kind: EdgeKind, note: str = "") -> Edge:
        self._require(source, target)
        if kind == "tests" and (source not in self.decisions or target not in self.observations):
            raise ValueError("tests links must connect a decision to its observation.")
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
                 "uncertainties": c.uncertainties,
                 "supporting_evidence": [e.source for e in self.edges if e.target == c.id and e.kind == "supports"],
                 "contradicting_evidence": [e.source for e in self.edges if e.target == c.id and e.kind == "contradicts"],
                 "qualifications": [e.source for e in self.edges if e.target == c.id and e.kind == "qualifies"],
                 "not_transferable": [e.source for e in self.edges if e.target == c.id and e.kind == "not_transferable"]}
                for c in claims
            ],
            "assumptions": [
                {"id": a.id, "statement": a.statement, "status": a.status} for a in self.assumptions.values()
            ],
        }
        evidence_ids = {i for c in claims for i in c.evidence if i in self.evidence_records}
        uncertainty_ids = {i for c in claims for i in c.uncertainties}
        slice_["source_evidence"] = [
            {"id": record.id, "title": record.title, "reference": record.reference, "query": record.query}
            for record in self.evidence_records.values() if record.id in evidence_ids
        ]
        slice_["uncertainty_records"] = [
            record.model_dump() for record in self.uncertainties.values() if record.id in uncertainty_ids
        ]
        if with_edges:
            relevant = ({o.id for o in observations} | {c.id for c in claims} | set(self.assumptions)
                        | evidence_ids | uncertainty_ids)
            slice_["relationships"] = [
                e.model_dump() for e in self.edges if e.source in relevant and e.target in relevant
            ]
        return slice_
