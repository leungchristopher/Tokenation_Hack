"""Framework-agnostic optimisation session: environment + reasoning graph + BO. No Inspect dependency.

    s = Session(env_name="zimmer_a549", budget=30)
    s.cite("taxol/doxorubicin antagonistic in A549", ["doi:..."], trust=0.4, trust_reason="...")
    for _ in range(30):
        s.run(s.suggest()[0]["params"], reasoning="BO suggestion")
    print(s.score(), s.graph.to_mermaid())
"""

from typing import Callable

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from bo_eval.env import TabularEnv, get_env
from bo_eval.graph import Edge, Evidence, Node, Prior, ReasoningGraph


def prior_weight(env: TabularEnv, C: np.ndarray, belief: dict) -> np.ndarray:
    """pi(x): product of Gaussians centred on the believed optimum, in encoded coordinates."""
    logp = np.zeros(len(C))
    for k, (best, width) in belief.items():
        j = env.params.index(k)
        ref = env.X[:1].copy()
        ref[0, j] = best
        logp -= 0.5 * ((C[:, j] - env.encode(ref)[0, j]) / max(width, 1e-3)) ** 2
    return np.exp(logp - logp.max())


class Session(BaseModel):
    """One optimisation episode. Errors are raised as ValueError."""

    env_name: str
    budget: int = 30
    seed: int = 0
    max_searches: int = 6
    searches: int = 0
    graph: ReasoningGraph = Field(default_factory=ReasoningGraph)
    submission: dict[str, float] | None = None

    @property
    def env(self) -> TabularEnv:
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
        node = Node(id=f"E{n + 1}", params=self.env.condition(i),
                    result=self.env.sample(i, np.random.default_rng([self.seed, n])))
        self.graph.nodes[node.id] = node
        self.graph.edges.append(Edge(source=parent, target=node.id, reasoning=reasoning))
        return node

    def reason(self, source: str, target: str, reasoning: str) -> None:
        self._node(source), self._node(target)
        self.graph.edges.append(Edge(source=source, target=target, reasoning=reasoning))

    def close(self, node: str, reason: str) -> list[str]:
        self._node(node)
        return self.graph.close(node, reason)

    def set_prior(self, belief: dict, reasoning: str, trust: float = 1.0) -> Prior:
        """belief: param -> [best, width as a fraction of the (encoded) range]."""
        if set(belief) - set(self.env.params) or any(len(b) != 2 or b[1] <= 0 for b in belief.values()):
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
                     trust_reason=trust_reason, about=about or [])
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

    def submit(self, params: dict, require_closed: bool = False) -> dict:
        cond = self.env.condition(self.env.index(params))
        if require_closed:
            dangling = [n for n in self.graph.open_leaves() if self.graph.nodes[n].params != cond]
            if dangling:
                raise ValueError(f"Unexplained open branches: {dangling}. Call close_branch on each with the "
                                 "reason it was not continued, then submit again.")
        self.submission = cond
        return cond

    def best_observed(self) -> dict | None:
        exps = self.graph.experiments
        pick = max if self.env.goal == "maximize" else min
        return pick(exps, key=lambda e: e.result).params if exps else None

    def suggest(self, n: int = 1, xi: float = 0.01, avoid_closed: bool = True, beta: float = 10.0) -> list[dict]:
        """GP + expected improvement over all feasible, not-yet-run conditions.

        The latest prior weights EI by pi(x)^(beta*trust/n) (piBO, Hvarfner et al. 2022), so domain knowledge
        dominates early and the GP takes over as data accumulates.
        """
        env, exps = self.env, self.graph.experiments
        C = env.encode(env.X)
        prior = self.graph.priors[-1] if self.graph.priors else None
        pi = prior_weight(env, C, prior.belief) if prior else None
        run_idx = [env.index(e.params) for e in exps]
        mask = np.ones(len(C), bool)
        mask[run_idx] = False

        if len(exps) < 2:
            rng = np.random.default_rng([self.seed, len(exps), 1])
            cand = np.flatnonzero(mask)
            w = None if pi is None else pi[cand] / pi[cand].sum()
            why = "random initial design" if pi is None else "initial design sampled from the prior"
            return [{"params": env.condition(i), "reason": why} for i in rng.choice(cand, size=n, replace=False, p=w)]

        Xo = C[run_idx]
        sign = 1.0 if env.goal == "maximize" else -1.0
        y = sign * np.array([e.result for e in exps])
        kernel = ConstantKernel() * Matern(nu=2.5, length_scale=np.ones(C.shape[1])) + WhiteKernel()
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2).fit(Xo, y)
        mu, sd = gp.predict(C, return_std=True)
        z = (mu - y.max() - xi) / np.maximum(sd, 1e-9)
        ei = (mu - y.max() - xi) * norm.cdf(z) + sd * norm.pdf(z)
        acq = ei if pi is None else ei * pi ** (beta * prior.trust / len(exps))
        if avoid_closed:
            nearest = ((C[:, None, :] - Xo[None]) ** 2).sum(-1).argmin(1)
            mask &= ~np.array([exps[j].closed for j in nearest])
        return [
            {"params": env.condition(i), "predicted_mean": float(sign * mu[i]), "predicted_sd": float(sd[i]),
             "expected_improvement": float(ei[i]), **({} if pi is None else {"prior_weight": float(pi[i])})}
            for i in [i for i in np.argsort(-acq) if mask[i]][:n]
        ]

    def score(self, tolerance: float = 0.0) -> dict:
        """found_optimal (submitted, else best observed, within `tolerance` relative regret), n_experiments, regret."""
        env, answer = self.env, self.submission or self.best_observed()
        best = env.true_value(env.optimum)
        regret = 1.0 if answer is None else abs(best - env.true_value(env.index(answer))) / abs(best)
        hits = [e.id for e in self.graph.experiments if env.index(e.params) == env.optimum]
        return {"found_optimal": float(regret <= tolerance), "n_experiments": len(self.graph.experiments),
                "regret": regret, "answer": answer, "first_experiment_at_optimum": hits[0] if hits else None}


# No-LLM baselines (also used by the Inspect `bo` / `random` solvers).
def bo_loop(s: Session, n_init: int = 3) -> None:
    """No LLM: random initial design, then GP-EI until the budget is spent; submit the best observed."""
    init = np.random.default_rng(s.seed).choice(len(s.env.df), size=n_init, replace=False)
    for k in range(s.budget):
        sug = [{"params": s.env.condition(init[k])}] if k < n_init else s.suggest(1)
        if not sug:
            break
        s.run(sug[0]["params"], reasoning="initial design" if k < n_init else "BO suggestion")
    if s.graph.experiments:
        s.submit(s.best_observed())


def random_loop(s: Session) -> None:
    for i in np.random.default_rng(s.seed).permutation(len(s.env.df))[: s.budget]:
        s.run(s.env.condition(i), reasoning="random")
    if s.graph.experiments:
        s.submit(s.best_observed())
