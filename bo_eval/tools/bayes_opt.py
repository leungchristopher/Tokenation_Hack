import json

import numpy as np
from inspect_ai.tool import tool
from inspect_ai.util import store_as
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from bo_eval.env import get_env
from bo_eval.state import BOState


def prior_weight(env, C: np.ndarray, belief: dict) -> np.ndarray:
    """pi(x): product of Gaussians centred on the believed optimum, in normalised coordinates."""
    logp = np.zeros(len(C))
    for k, (best, width) in belief.items():
        j = env.params.index(k)
        ref = env.X[:1].copy()
        ref[0, j] = best
        x = env.encode(ref)[0, j]
        logp -= 0.5 * ((C[:, j] - x) / max(width, 1e-3)) ** 2
    return np.exp(logp - logp.max())


def suggest(n: int = 1, xi: float = 0.01, avoid_closed: bool = True, beta: float = 10.0) -> list[dict]:
    """GP + expected improvement over all feasible, not-yet-run conditions.

    If the LLM has set a prior, EI is weighted by pi(x)^(beta*trust/n) (piBO, Hvarfner et al. 2022),
    so domain knowledge dominates early and the GP takes over as data accumulates.
    """
    s = store_as(BOState)
    env, exps = get_env(s.env), s.graph.experiments
    C = env.encode(env.X)
    prior = s.graph.priors[-1] if s.graph.priors else None
    pi = prior_weight(env, C, prior.belief) if prior else None
    run_idx = [env.index(e.params) for e in exps]
    mask = np.ones(len(C), bool)
    mask[run_idx] = False

    if len(exps) < 2:
        rng = np.random.default_rng([s.seed, len(exps), 1])
        cand = np.flatnonzero(mask)
        w = None if pi is None else pi[cand] / pi[cand].sum()
        pick = rng.choice(cand, size=n, replace=False, p=w)
        why = "random initial design" if pi is None else "initial design sampled from the prior"
        return [{"params": env.condition(i), "reason": why} for i in pick]

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
    order = [i for i in np.argsort(-acq) if mask[i]][:n]
    return [
        {"params": env.condition(i), "predicted_mean": float(sign * mu[i]),
         "predicted_sd": float(sd[i]), "expected_improvement": float(ei[i]),
         **({} if pi is None else {"prior_weight": float(pi[i])})}
        for i in order
    ]


@tool
def bayes_opt_suggest():
    async def execute(n: int = 1, avoid_closed: bool = True) -> str:
        """Suggest the next experiment(s) by Bayesian optimisation (GP + expected improvement) over all results so far. Does not consume budget.

        Args:
            n: Number of suggestions to return.
            avoid_closed: Skip candidates whose nearest observed experiment lies on a closed branch.
        """
        return json.dumps(suggest(n, avoid_closed=avoid_closed))

    return execute
