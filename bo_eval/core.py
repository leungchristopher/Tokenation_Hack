"""Framework-agnostic optimisation session: environment + reasoning graph + BO. No Inspect dependency.

    s = Session(env_name="zimmer_a549", budget=30)
    s.cite("taxol/doxorubicin antagonistic in A549", ["doi:..."], trust=0.4, trust_reason="...")
    for _ in range(30):
        s.run(s.suggest()[0]["params"], reasoning="BO suggestion")
    print(s.score(), s.graph.to_mermaid())
"""

from typing import Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from bo_eval.domain import Domain
from bo_eval.graph import Edge, Evidence, Node, Prior, ReasoningGraph
from bo_eval.optimiser import Posterior, expected_improvement, fit_posterior, prior_weight


class Session(BaseModel):
    """One optimisation episode. Errors are raised as ValueError."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    env_name: str = ""
    domain: Domain | None = Field(default=None, exclude=True)
    budget: int = Field(default=30, ge=0)
    seed: int = Field(default=0, ge=0)
    max_searches: int = Field(default=6, ge=0)
    searches: int = 0
    graph: ReasoningGraph = Field(default_factory=ReasoningGraph)
    submission: dict[str, float] | None = None

    @property
    def env(self) -> Domain:
        if self.domain is not None:
            return self.domain
        from bo_eval.env import get_env

        return get_env(self.env_name)

    def _node(self, nid: str) -> Node:
        if nid not in self.graph.nodes:
            raise ValueError(f"Unknown node '{nid}'.")
        return self.graph.nodes[nid]

    def _after(self) -> str:
        return self.graph.experiments[-1].id if self.graph.experiments else "root"

    def run(self, params: dict, parent: str = "root", reasoning: str = "") -> Node:
        """Run one experiment (snapped to the nearest feasible condition) and add it to the graph."""
        if self._node(parent).closed:
            raise ValueError(f"Branch '{parent}' is closed and cannot be extended.")
        n = len(self.graph.experiments)
        if n >= self.budget:
            raise ValueError("Experiment budget exhausted. Submit your answer.")
        i = self.env.index(params)
        result = self.env.sample(i, np.random.default_rng([self.seed, n]))
        if not np.isfinite(result):
            raise ValueError("Evaluator returned a non-finite result; no experiment was recorded.")
        node = Node(id=f"E{n + 1}", params=self.env.condition(i), result=result)
        self.graph.nodes[node.id] = node
        self.graph.edges.append(Edge(source=parent, target=node.id, reasoning=reasoning))
        return node

    def reason(self, source: str, target: str, reasoning: str) -> None:
        self._node(source), self._node(target)
        self.graph.edges.append(Edge(source=source, target=target, reasoning=reasoning, kind="reasoning"))

    def close(self, node: str, reason: str) -> list[str]:
        self._node(node)
        return self.graph.close(node, reason)

    def set_prior(self, belief: dict, reasoning: str, trust: float = 1.0) -> Prior:
        """belief: param -> [best, width as a fraction of the (encoded) range]."""
        if (not belief or set(belief) - set(self.env.params) or not 0 <= trust <= 1
                or any(len(b) != 2 or not np.isfinite(b).all() or b[1] <= 0 for b in belief.values())):
            raise ValueError(f"belief must map parameters in {self.env.params} to [best, width>0]; got {belief}")
        p = Prior(id=f"P{len(self.graph.priors) + 1}", after=self._after(),
                  belief={k: tuple(b) for k, b in belief.items()}, reasoning=reasoning, trust=trust)
        self.graph.priors.append(p)
        return p

    def cite(self, claim: str, sources: list[str], trust: float, trust_reason: str,
             about: list[str] | None = None, belief: dict | None = None) -> Evidence:
        """Record a literature claim; an implied belief becomes a prior weighted by `trust`."""
        if not sources or not 0 <= trust <= 1:
            raise ValueError("Give at least one source and a trust in [0, 1].")
        v = Evidence(id=f"R{len(self.graph.evidence) + 1}", claim=claim, sources=sources, trust=trust,
                     trust_reason=trust_reason, about=list(about or []))
        if belief:
            v.about.append(self.set_prior(belief, f"from {v.id}: {claim}", trust).id)
        self.graph.evidence.append(v)
        return v

    def charge_search(self) -> None:
        if self.searches >= self.max_searches:
            raise ValueError(f"Search budget ({self.max_searches}) used up; record evidence from what you have.")
        self.searches += 1

    def search(self, query: str, fn: Callable[[str], str]) -> str:
        """Run a literature search with any backend, counted against max_searches."""
        self.charge_search()
        return fn(query)

    def submit(self, params: dict, require_closed: bool = False, reason: str = "") -> dict:
        cond = self.env.condition(self.env.index(params))
        if require_closed:
            dangling = [n for n in self.graph.open_leaves() if self.graph.nodes[n].params != cond]
            if dangling:
                raise ValueError(f"Unexplained open branches: {dangling}. Call close_branch on each with the "
                                 "reason it was not continued, then submit again.")
        self.submission = cond
        self.graph.selected = next((e.id for e in self.graph.experiments if e.params == cond), None)
        self.graph.decision = f"{cond}. {reason}".strip()
        return cond

    def best_observed(self) -> dict | None:
        exps = self.graph.experiments
        pick = max if self.env.goal == "maximize" else min
        return pick(exps, key=lambda e: e.result).params if exps else None

    def model(self, gated: bool = False, learn: bool = True) -> Posterior:
        priors = [
            p.model_copy(update={"trust": next(
                (v.trust for v in reversed(self.graph.evidence) if p.id in v.about), p.trust
            )})
            for p in self.graph.priors
        ] if gated else []
        return fit_posterior(
            self.env, self.graph.experiments, priors, learn, self.seed
        )

    def posterior(self) -> tuple[np.ndarray, np.ndarray]:
        """Plain GP posterior mean and sd, in the result's units."""
        model = self.model()
        return model.mean, model.sd

    def suggest(self, n: int = 1, xi: float = 0.01, avoid_closed: bool = True,
                gated: bool = False, learn: bool = True, model: Posterior | None = None) -> list[dict]:
        """Shared GP-EI acquisition. With no priors, gated=True reduces to plain GP-BO."""
        if n < 1:
            raise ValueError("n must be positive.")
        env, exps = self.env, self.graph.experiments
        C = env.encode(env.X)
        priors = self.graph.priors if gated else []
        run_idx = [env.index(e.params) for e in exps]
        mask = np.ones(len(C), bool)
        mask[run_idx] = False
        if avoid_closed and exps:
            nearest = ((C[:, None, :] - C[run_idx][None]) ** 2).sum(-1).argmin(1)
            mask &= ~np.array([exps[j].closed for j in nearest])
        cand = np.flatnonzero(mask)
        if not len(cand):
            return []

        if len(exps) < 2:
            rng = np.random.default_rng([self.seed, len(exps), 1])
            scores = sum((p.trust * prior_weight(env, C, p.belief) for p in priors), start=np.zeros(len(C)))
            weights = np.exp(scores[cand] - scores[cand].max())
            weights = np.maximum(weights, 1e-12)
            weights /= weights.sum()
            why = "random initial design" if not priors else "initial design sampled from cited priors"
            return [{"params": env.condition(i), "reason": why}
                    for i in rng.choice(cand, size=min(n, len(cand)), replace=False, p=weights)]

        model = model or self.model(gated=gated, learn=learn)
        mu, sd = model.mean, model.sd
        pick = max if env.goal == "maximize" else min
        best = pick(e.result for e in exps)
        ei = expected_improvement(mu, sd, best, env.goal, xi)
        acq = ei
        return [
            {"params": env.condition(i), "predicted_mean": float(mu[i]), "predicted_sd": float(sd[i]),
             "expected_improvement": float(ei[i]),
             "reason": f"EI {ei[i]:.3g}; predicted {mu[i]:.4g} ± {sd[i]:.3g}"
                       + (f"; priors {', '.join(model.gates)}" if model.gates else "")}
            for i in [i for i in np.argsort(-acq) if mask[i]][:n]
        ]

    def score(self, tolerance: float = 0.0) -> dict:
        """found_optimal (submitted, else best observed, within `tolerance` relative regret), n_experiments, regret."""
        from bo_eval.env import TabularEnv

        env, answer = self.env, self.submission or self.best_observed()
        if not isinstance(env, TabularEnv):
            raise ValueError("score() requires a benchmark with known truth; live domains need an external scorer.")
        best = env.true_value(env.optimum)
        regret = 1.0 if answer is None else abs(best - env.true_value(env.index(answer))) / abs(best)
        hits = [e.id for e in self.graph.experiments if env.index(e.params) == env.optimum]
        return {"found_optimal": float(regret <= tolerance), "n_experiments": len(self.graph.experiments),
                "regret": regret, "answer": answer, "first_experiment_at_optimum": hits[0] if hits else None}


# No-LLM baselines (also used by the Inspect `bo` / `random` solvers).
def bo_loop(s: Session, n_init: int = 3) -> None:
    """No LLM: random initial design, then GP-EI until the budget is spent; submit the best observed."""
    init = np.random.default_rng(s.seed).choice(len(s.env.X), size=min(n_init, len(s.env.X)), replace=False)
    for k in range(s.budget):
        sug = [{"params": s.env.condition(init[k])}] if k < len(init) else s.suggest(1)
        if not sug:
            break
        s.run(sug[0]["params"], reasoning="initial design" if k < n_init else "BO suggestion")
    if s.graph.experiments:
        s.submit(s.best_observed(), reason="Best observed result.")


def random_loop(s: Session) -> None:
    for i in np.random.default_rng(s.seed).permutation(len(s.env.X))[: s.budget]:
        s.run(s.env.condition(i), reasoning="random")
    if s.graph.experiments:
        s.submit(s.best_observed(), reason="Best observed result.")
