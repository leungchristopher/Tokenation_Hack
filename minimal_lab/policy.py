"""Noisy finite-domain search for reproducible intended protocols, independent of task APIs."""
from dataclasses import dataclass
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


def statistics(observations):
    groups = {}
    for o in observations:
        if o.get('ok', True) and o['value'] is not None:
            groups.setdefault(o['candidate'], []).append(o['value'])
    return {i: dict(n=len(v), mean=float(np.mean(v)),
                   sd=float(np.std(v, ddof=1)) if len(v)>1 else None)
            for i, v in groups.items()}


@dataclass
class Posterior:
    gp: object
    centre: float
    spread: float
    X: np.ndarray
    log_axes: tuple
    noise: float
    mean: np.ndarray
    sd: np.ndarray

    def encode(self, values):
        return encode(values, self.X, self.log_axes)

    def predict(self, values, return_std=False):
        return self.gp.predict(values, return_std=return_std)


def encode(values, domain, log_axes=()):
    values, domain = np.asarray(values, float).copy(), np.asarray(domain, float).copy()
    for j in log_axes:
        if (values[...,j] <= 0).any() or (domain[:,j] <= 0).any():
            raise ValueError('Log-scaled parameters must be strictly positive.')
        values[...,j], domain[:,j] = np.log10(values[...,j]), np.log10(domain[:,j])
    span = np.ptp(domain, axis=0)
    return (values-domain.min(0))/np.where(span>0, span, 1)


def fit(X, observations, direction=1, log_axes=()):
    stats = statistics(observations)
    ids = list(stats)
    y = direction*np.array([stats[i]['mean'] for i in ids])
    counts = np.array([stats[i]['n'] for i in ids])
    # Pool within-protocol scatter. Includes delivery + readout variability, not SE.
    variances = [v['sd']**2 for v in stats.values() if v['sd'] is not None]
    noise = float(np.median(variances)) if variances else max(float(np.var(y))*.05, 1.)
    noise = max(noise, 1e-6)
    variance = np.array([noise if stats[i]['sd'] is None else
                         (2*noise+(stats[i]['n']-1)*stats[i]['sd']**2)/(stats[i]['n']+1)
                         for i in ids])
    centre, spread = float(y.mean()), max(float(y.std()), np.sqrt(noise), 1.)
    kernel = ConstantKernel(1., (.1, 10.))*Matern(length_scale=np.full(X.shape[1], .35),
                                               length_scale_bounds=(.08, 3.), nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=variance/counts/spread**2+1e-8,
                                  optimizer='fmin_l_bfgs_b' if len(ids)>=6 else None)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        gp.fit(encode(X[ids], X, log_axes), (y-centre)/spread)
    mu, sd = gp.predict(encode(X, X, log_axes), return_std=True)
    return Posterior(gp, centre, spread, X, tuple(log_axes), noise,
                     direction*(mu*spread+centre), sd*spread)


def shortlist(X, observations, rng, direction=1, weights=None, log_axes=()):
    stats = statistics(observations)
    if not stats:
        ids = rng.choice(len(X), min(3,len(X)), replace=False).tolist()
        if weights is not None:
            first = int(np.argmax(weights))
            ids = [first]+[i for i in ids if i != first]
        return [dict(candidate=int(i), mean=None, sd=None, ei=None,
                     reason='Literature-initialised starting recipe.' if weights is not None and i==ids[0]
                            else 'Initial domain coverage; no response measurements yet.') for i in ids], None
    posterior = fit(X, observations, direction, log_axes)
    mu, sd = direction*posterior.mean, posterior.sd
    # A noisy lucky reading must not set the improvement threshold.
    incumbent = max(stats, key=lambda i: mu[i])
    delta = mu-mu[incumbent]
    z = delta/np.maximum(sd,1e-9)
    ei = delta*norm.cdf(z)+sd*norm.pdf(z)
    acquisition = ei if weights is None else ei*weights
    unseen = np.array([i for i in range(len(X)) if i not in stats], dtype=int)
    pool = unseen if len(unseen) else np.arange(len(X))
    # Explicitly preserve measured recipes as revisit options. Default search expands
    # coverage; repeats are scheduled separately instead of consuming every proposal.
    improvement = int(pool[np.argmax(acquisition[pool])])
    exploration = int(pool[np.argmax(sd[pool])])
    choices = [(improvement, 'Highest improvement among untested recipes; previous recipes remain revisitable.'),
               (exploration, 'Explore the largest uncertainty among untested recipes.'),
               (incumbent, 'Revisit the best posterior recipe; confirmation uses independent preparations.'),
               (max(stats, key=lambda i: direction*stats[i]['mean']),
                'Revisit the best measured mean despite surrogate disagreement.')]
    # Periodic coverage prevents a wrong prior/local model from monopolising search.
    if len(stats) % 4 == 0 and len(unseen):
        choices.insert(0, (exploration, 'Scheduled global exploration; no branch is permanently excluded.'))
    unique = dict(reversed(choices))
    return [dict(candidate=i, mean=float(posterior.mean[i]), sd=float(sd[i]), ei=float(ei[i]),
                 prior_weight=None if weights is None else float(weights[i]),
                 n_observed=stats.get(i,{}).get('n',0), reason=unique[i]) for i in dict(choices)], posterior
