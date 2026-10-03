"""Long-term memory: cited claims whose trust is learned across runs (and across tasks sharing parameters).

Trust is the mean of Beta(confirmed, refuted). Gates, after LSTM:
  read   -- recall only claims about this task or its parameters;
  write  -- a claim enters only once it has been calibrated against the literature;
  update -- experiments add to its confirmed or refuted count, so contradicted claims fade.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from bo_eval.env import TabularEnv


class Claim(BaseModel):
    key: str
    claim: str
    env: str
    sources: list[str] = Field(default_factory=list)
    belief: dict[str, tuple[float, float]] = Field(default_factory=dict)
    confirmed: float = 0.5
    refuted: float = 0.5
    history: list[str] = Field(default_factory=list)

    @property
    def trust(self) -> float:
        return self.confirmed / (self.confirmed + self.refuted)

    def calibrate(self, literature_trust: float, strength: float = 4.0) -> None:
        """Literature acts as `strength` pseudo-observations; experiments add one each."""
        self.confirmed += strength * literature_trust
        self.refuted += strength * (1 - literature_trust)

    def update(self, confirmed: bool, why: str) -> None:
        if confirmed:
            self.confirmed += 1
        else:
            self.refuted += 1
        self.history.append(("confirmed: " if confirmed else "refuted: ") + why)


class Memory(BaseModel):
    path: str | None = None
    claims: dict[str, Claim] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "Memory":
        p = Path(path)
        return cls.model_validate_json(p.read_text()) if p.exists() else cls(path=path)

    def recall(self, env: TabularEnv) -> list[Claim]:
        return [c for c in self.claims.values() if c.env == env.name or set(c.belief) & set(env.params)]

    def write(self, c: Claim) -> None:
        self.claims[c.key] = c

    def save(self) -> None:
        if self.path:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.path).write_text(json.dumps(self.model_dump(), indent=1))
