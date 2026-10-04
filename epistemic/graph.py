from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EdgeKind = Literal["supports", "qualifies", "depends_on", "tests"]
UncertaintyKind = Literal["source", "transfer", "mechanistic", "response"]


class FrozenDict(dict):
    def _immutable(self, *args, **kwargs):
        raise TypeError("Record metadata is immutable.")

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
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

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
    evidence: tuple[str, ...] = ()
    round: int = 0
    kind: Literal["uncertainty"] = "uncertainty"


class Observation(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    id: str
    round: int
    candidate_id: str
    intended_params: dict[str, float]
    value_shown: float | None
    outcome_unit: str
    simulated: bool = True
    measurement_noise: bool = True
    kind: Literal["observation"] = "observation"

    @field_validator("intended_params")
    @classmethod
    def freeze_params(cls, value):
        return _freeze(value)


class Claim(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    id: str
    statement: str
    scope: str
    source: Literal["literature", "model_conjecture"]
    evidence: list[str] = Field(default_factory=list)
    discriminating_result: str = Field(min_length=1)
    reference: str = ""
    uncertainties: list[str] = Field(default_factory=list)
    belief: dict[str, tuple[float, float]] = Field(default_factory=dict)
    trust: float | None = Field(default=None, ge=0.0, le=1.0)
    trust_reason: str = ""
    kind: Literal["claim"] = "claim"

    @model_validator(mode="after")
    def require_reference_for_literature(self):
        if self.source == "literature" and not self.reference:
            raise ValueError("Literature claims need an attributable source.")
        return self


class Assumption(BaseModel):
    id: str
    statement: str
    scope: str
    kind: Literal["assumption"] = "assumption"


class Decision(BaseModel):
    id: str
    round: int
    candidate_id: str
    claims: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    justification: str = ""
    prediction: float | None = None
    prediction_source: Literal["numerical_model", "unavailable"] = "unavailable"
    prior_gates: dict[str, tuple[float, float]] = Field(default_factory=dict)
    kind: Literal["decision"] = "decision"


class Edge(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    target: str
    kind: EdgeKind
    note: str = ""


class EvidenceGraph(BaseModel):
    evidence_records: dict[str, EvidenceRecord] = Field(default_factory=dict)
    uncertainties: dict[str, UncertaintyRecord] = Field(default_factory=dict)
    observations: dict[str, Observation] = Field(default_factory=dict)
    claims: dict[str, Claim] = Field(default_factory=dict)
    assumptions: dict[str, Assumption] = Field(default_factory=dict)
    decisions: dict[str, Decision] = Field(default_factory=dict)
    edges: list[Edge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self):
        tables = self._tables()
        ids = [record_id for table in tables for record_id in table]
        if len(set(ids)) != len(ids):
            raise ValueError("Record IDs must be globally unique.")
        if any(record_id != record.id for table in tables for record_id, record in table.items()):
            raise ValueError("Table keys must match their record IDs.")
        for claim in self.claims.values():
            self._require(*claim.evidence, *claim.uncertainties)
            if any(record_id not in self.uncertainties for record_id in claim.uncertainties):
                raise ValueError("Claim uncertainty IDs must refer to uncertainty records.")
        for decision in self.decisions.values():
            self._require(*decision.claims, *decision.assumptions, *decision.evidence)
            if any(record_id not in self.claims for record_id in decision.claims):
                raise ValueError("Decision claim IDs must refer to claims.")
        for uncertainty in self.uncertainties.values():
            self._require(*uncertainty.evidence)
        for edge in self.edges:
            self._require(edge.source, edge.target)
            if edge.kind == "tests" and (edge.source not in self.decisions or edge.target not in self.observations):
                raise ValueError("tests links must connect a decision to its observation.")
        return self

    def _tables(self) -> tuple[dict, ...]:
        return (
            self.evidence_records, self.uncertainties, self.observations,
            self.claims, self.assumptions, self.decisions,
        )

    def __contains__(self, record_id: str) -> bool:
        return any(record_id in table for table in self._tables())

    def _require(self, *ids: str) -> None:
        unknown = [record_id for record_id in ids if record_id not in self]
        if unknown:
            raise KeyError(f"Unknown record reference(s): {unknown}")

    def add_observation(self, observation: Observation) -> Observation:
        if observation.id in self:
            raise ValueError(f"Observation {observation.id} already recorded.")
        self.observations[observation.id] = observation
        return observation

    def add_evidence(self, record: EvidenceRecord) -> EvidenceRecord:
        if record.id in self:
            raise ValueError(f"Evidence {record.id} already exists.")
        self.evidence_records[record.id] = record
        return record

    def add_uncertainty(self, record: UncertaintyRecord) -> UncertaintyRecord:
        self._require(*record.evidence)
        if record.id in self:
            raise ValueError(f"Uncertainty {record.id} already exists.")
        self.uncertainties[record.id] = record
        return record

    def add_claim(self, claim: Claim) -> Claim:
        self._require(*claim.evidence, *claim.uncertainties)
        if any(record_id not in self.uncertainties for record_id in claim.uncertainties):
            raise ValueError("Claim uncertainty IDs must refer to uncertainty records.")
        if claim.id in self:
            raise ValueError(f"Claim {claim.id} already exists.")
        self.claims[claim.id] = claim
        return claim

    def add_assumption(self, assumption: Assumption) -> Assumption:
        if assumption.id in self:
            raise ValueError(f"ID {assumption.id} already exists.")
        self.assumptions[assumption.id] = assumption
        return assumption

    def add_decision(self, decision: Decision) -> Decision:
        self._require(*decision.claims, *decision.assumptions, *decision.evidence)
        if any(record_id not in self.claims for record_id in decision.claims):
            raise ValueError("Decision claim IDs must refer to claims.")
        if decision.id in self:
            raise ValueError(f"ID {decision.id} already exists.")
        self.decisions[decision.id] = decision
        return decision

    def link(self, source: str, target: str, kind: EdgeKind, note: str = "") -> Edge:
        self._require(source, target)
        if kind == "tests" and (source not in self.decisions or target not in self.observations):
            raise ValueError("tests links must connect a decision to its observation.")
        edge = Edge(source=source, target=target, kind=kind, note=note)
        self.edges.append(edge)
        return edge
