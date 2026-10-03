"""GP with gated prior knowledge: y = sum_i g_i h_i(x) + f(x).

Each prior P_i in the graph is a basis function h_i (a bump on its believed optimum, in [0, 1]); its gate g_i
has prior N(2 * trust_i, 1) in normalised-output units. The gates' posterior is closed form under the GP marginal
likelihood (Rasmussen & Williams 2006, sec. 2.7): data that a prior explains keeps its gate open, data it
contradicts closes it. This replaces a hand-set decay of the prior's weight.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from bo_eval.core import prior_weight
from bo_eval.env import TabularEnv
from bo_eval.graph import Node, Prior


def gated_posterior(env: TabularEnv, exps: list[Node], priors: list[Prior], learn: bool = True):
    """Returns (mean, sd) over every condition in result units, and each gate's (mean, sd) on the trust scale."""
    C, sign = env.encode(env.X), (1.0 if env.goal == "maximize" else -1.0)
    idx = [env.index(e.params) for e in exps]
    y = sign * np.array([e.result for e in exps])
    ym, ys = y.mean(), y.std() or 1.0
    z = (y - ym) / ys
    H = np.array([prior_weight(env, C, p.belief) for p in priors]).reshape(len(priors), len(C))
    b, Ho = 2.0 * np.array([p.trust for p in priors]), H[:, idx]

    kernel = ConstantKernel() * Matern(nu=2.5, length_scale=np.ones(C.shape[1])) + WhiteKernel()
    k = GaussianProcessRegressor(kernel, n_restarts_optimizer=2).fit(C[idx], z - Ho.T @ b).kernel_
    Ks, Kinv = k(C[idx], C), np.linalg.inv(k(C[idx]) + 1e-8 * np.eye(len(idx)))
    A = np.eye(len(b)) + Ho @ Kinv @ Ho.T
    g = np.linalg.solve(A, Ho @ Kinv @ z + b) if len(b) and learn else b
    R = H - Ho @ Kinv @ Ks
    mu = Ks.T @ Kinv @ (z - Ho.T @ g) + H.T @ g
    var = k.diag(C) - np.einsum("ij,ij->j", Ks, Kinv @ Ks)
    if len(b):
        var += np.einsum("ij,ij->j", R, np.linalg.solve(A, R))
        gate_sd = np.sqrt(np.diag(np.linalg.inv(A)))
    else:
        gate_sd = b
    return sign * (mu * ys + ym), np.sqrt(np.maximum(var, 1e-12)) * ys, g / 2, gate_sd / 2
