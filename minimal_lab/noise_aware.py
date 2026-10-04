"""Noise-aware alternative to loop.shortlist, plus a paired offline benchmark against it.

The baseline GP adds a fixed WhiteKernel, so its predictive SD (used for EI and the
uncertainty pick) includes measurement noise, and EI improves on the best *noisy* mean.
Here replicates are aggregated per intended condition, noise is estimated from replicate
scatter only (shrunk to the baseline level, floored), acquisition uses latent epistemic SD,
and the incumbent is the posterior-best measured condition. Repeats then happen only when
latent EI values replication. Same inputs and options contract as loop.shortlist.

Offline CPU tabular benchmark, not a validation of MuJoCo delivery noise:
    python -m minimal_lab.noise_aware --envs upo_abts icfree_cole1 --seeds 0-9 --budgets 12 24 --out r.json
"""
import argparse
import asyncio
import json
import time
from typing import Any
from unittest import mock

import numpy as np
from scipy.stats import norm, t
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from minimal_lab import loop
from minimal_lab.loop import shortlist as baseline_shortlist

PRIOR_NOISE = 0.05  # Baseline WhiteKernel level (standardized units), used as a weak prior.
PRIOR_DOF = 2.0  # Pseudo-degrees of freedom of that prior. Fixed a priori, not tuned.
NOISE_FLOOR = 1e-4


class LatentGP:
    """Standardized exactly like loop.shortlist (centre=mean, spread=max(std, 1) of all
    observed responses). predict() returns the SD of ONE new measurement, as the loop's
    surprise check expects; latent() returns epistemic SD for acquisition."""

    def __init__(self, gp, noise):
        self.gp, self.noise = gp, noise

    def latent(self, Z):
        return self.gp.predict(Z, return_std=True)

    def predict(self, Z, return_std=False):
        mu, sd = self.latent(Z)
        return (mu, np.sqrt(sd**2 + self.noise)) if return_std else mu


def fit(X, observations, direction=1):
    """Observation-only model: uses candidate settings, reported inputs and responses."""
    X = np.asarray(X, float)
    good = [o for o in observations if o['value'] is not None]
    y = direction*np.array([o['value'] for o in good])
    centre, spread = y.mean(), max(y.std(), 1.0)
    lo, span = X.min(0), np.ptp(X, axis=0)
    scale = np.where(span > 0, span, 1)
    groups = {}
    for o in good:
        groups.setdefault(o['candidate'], []).append(o)
    ids = list(groups)
    n = np.array([len(groups[i]) for i in ids], float)
    rows = [(direction*np.array([o['value'] for o in groups[i]])-centre)/spread for i in ids]
    ybar = np.array([r.mean() for r in rows])
    within = sum(((r-r.mean())**2).sum() for r in rows)
    noise = max((PRIOR_DOF*PRIOR_NOISE+within)/(PRIOR_DOF+(n-1).sum()), NOISE_FLOOR)
    # Mean reported input per condition; independent report errors average down by sqrt(n).
    training = (np.array([np.mean([o['reported'] for o in groups[i]], 0) for i in ids])-lo)/scale
    input_sd = np.array([np.sqrt(np.mean(np.square([o.get('report_sd', np.zeros(X.shape[1]))
                                                    for o in groups[i]]), 0)/len(groups[i]))
                         for i in ids])/scale
    gp = GaussianProcessRegressor(kernel=Matern(length_scale=0.3, nu=2.5), alpha=noise/n,
                                  optimizer=None).fit(training, ybar)
    # Same first-order reported-input propagation as loop.shortlist.
    variance = np.zeros(len(ids))
    for j in range(X.shape[1]):
        shift = np.zeros_like(training)
        shift[:, j] = input_sd[:, j]
        variance += ((gp.predict(training+shift)-gp.predict(training-shift))/2)**2
    gp.alpha = noise/n+variance+1e-8
    gp.fit(training, ybar)
    return LatentGP(gp, noise), {'ids': ids, 'centre': centre, 'spread': spread, 'Z': (X-lo)/scale}


def shortlist(X, observations, rng, direction=1):
    if not any(o['value'] is not None for o in observations):
        return baseline_shortlist(X, observations, rng, direction)  # Identical seeded start.
    model, info = fit(X, observations, direction)
    mu, sd = model.latent(info['Z'])
    incumbent = info['ids'][int(np.argmax(mu[info['ids']]))]
    delta = mu-mu[incumbent]
    z = delta/np.maximum(sd, 1e-9)
    ei = delta*norm.cdf(z)+sd*norm.pdf(z)
    choices = [(int(np.argmax(ei)), ('Highest expected improvement of the latent mean over the posterior '
                'incumbent; a measured condition is repeated only when replication has the most value.')),
               (int(np.argmax(sd)), 'Largest latent (epistemic) uncertainty; measurement noise excluded.'),
               (incumbent, 'Repeat the posterior-best measured condition to resolve its latent mean.')]
    unique = {i: why for i, why in reversed(choices)}
    s, c = info['spread'], info['centre']
    return [{'candidate': i, 'mean': float(direction*(mu[i]*s+c)), 'sd': float(sd[i]*s), 'ei': float(ei[i]*s),
                 'reason': unique[i]} for i in dict(choices)], model


def recommend(X, observations, direction=1):
    """Posterior-best measured condition (noise-aware alternative to the best raw mean).

    Limitation: replicates are grouped by INTENDED candidate. Under reported-input noise each
    replicate's realised delivery differs, so a group's mean and scatter mix deliveries and
    the noise estimate absorbs delivery error. Exact here (offline report_sd=0)."""
    model, info = fit(X, observations, direction)
    return info['ids'][int(np.argmax(model.latent(info['Z'][info['ids']])[0]))]


# ---- Offline paired benchmark. Only black_box() and score() touch the table. ----

def black_box(env, seed):
    """execute(params) -> public fields. Paired noise: replicate k of condition i under seed s
    gets the same draw in every policy (common random numbers)."""
    counts = {}

    def execute(params):
        i = env.index(params)
        counts[i] = k = counts.get(i, -1)+1
        x = env.X[i].tolist()
        return {'value': env.sample(i, np.random.default_rng([seed, i, k])), 'reported': x,
                    'report_sd': [0.0]*len(x), 'ok': True}
    return execute


def random_shortlist(X, observations, rng, direction=1):
    """Same-loop random control (matches PR #11): one uniform unmeasured condition, no GP,
    so no surprise repeats; the loop still spends its last call on final confirmation."""
    seen = {o['candidate'] for o in observations}
    pool = [i for i in range(len(X)) if i not in seen] or list(range(len(X)))
    return [{'candidate': int(rng.choice(pool)), 'mean': None, 'sd': None, 'ei': None,
             'reason': 'Uniform random unmeasured condition.'}], None


POLICIES = {'noise_aware': shortlist, 'loop_random': random_shortlist}


def run_loop(env, budget, seed, policy):
    execute = black_box(env, seed)
    goal = getattr(env, 'goal', 'maximize')
    episode = loop.run(env.X.copy(), env.params, execute, budget=budget, seed=seed, goal=goal)
    if policy in POLICIES:
        with mock.patch.object(loop, 'shortlist', POLICIES[policy]):  # Scoped to this run only.
            episode = asyncio.run(episode)
    else:
        episode = asyncio.run(episode)
    kinds, seen = {'policy': 0, 'diagnostic': 0, 'final': 0}, set()
    for d in (n for n in episode['graph']['nodes'] if n['kind'] == 'decision'):
        if d['selected'] in seen:
            kinds['diagnostic' if d['reason'].startswith('Repeat after') else
                  'final' if d['reason'].startswith('Final') else 'policy'] += 1
        seen.add(d['selected'])
    surprises = sum('surprise' in n for n in episode['graph']['nodes'])
    return episode['observations'], episode['result']['candidate'], {'repeat_kinds': kinds, 'surprises': surprises}


def run_random(env, budget, seed):
    """Unconstrained random: B distinct conditions, no final confirmation (not the loop protocol)."""
    execute = black_box(env, seed)
    ids = np.random.default_rng(seed).choice(len(env.X), min(budget, len(env.X)), replace=False)
    observations = [dict(execute(dict(zip(env.params, env.X[i].tolist()))), candidate=int(i)) for i in ids]
    means = loop.observed_means(observations)
    return observations, max(means, key=means.get), {}


def score(env, candidate, observations):
    """Post-hoc only: hidden table means. Normalized latent regret, lower is better."""
    best = env.true_value(env.optimum)
    regret = lambda i: abs(best-env.true_value(i))/max(abs(best), 1e-12)
    means = loop.observed_means(observations)
    unique = len({o['candidate'] for o in observations})
    alt = recommend(env.X, observations, 1 if getattr(env, 'goal', 'maximize') == 'maximize' else -1)
    return {'regret': regret(candidate), 'found': float(candidate == env.optimum), 'calls': len(observations),
                'unique': unique, 'repeats': len(observations)-unique, 'rec_repeats': sum(o['candidate'] == candidate
                for o in observations), 'optimism': (means[candidate]-env.true_value(candidate))/max(abs(best), 1e-12),
                'posterior_rec_regret': regret(alt)}


def paired(a, b):
    d = np.array(a)-np.array(b)
    half = t.ppf(0.975, len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)) if len(d) > 1 else float('nan')
    return {'mean': float(d.mean()), 'ci95': [float(d.mean()-half), float(d.mean()+half)],
                'lower': int((d < -1e-12).sum()), 'ties': int((abs(d) <= 1e-12).sum()), 'higher': int((d > 1e-12).sum())}


NAMES = ('baseline', 'noise_aware', 'loop_random', 'random_unconstrained')


def benchmark(envs, seeds, budgets):
    from bo_eval.env import get_env
    rows: list[dict[str, Any]] = []
    for name in envs:
        env = get_env(name)
        for budget in budgets:
            for seed in seeds:
                for policy in NAMES:
                    obs, rec, extra = (run_random(env, budget, seed) if policy == 'random_unconstrained'
                                       else run_loop(env, budget, seed, policy))
                    rows.append(dict(env=name, budget=budget, seed=seed, policy=policy, recommended=int(rec),
                                     **score(env, rec, obs), **extra))
    summary: list[dict[str, Any]] = []
    for name in envs:
        for budget in budgets:
            cell = {p: [r for r in rows if (r['env'], r['budget'], r['policy']) == (name, budget, p)]
                    for p in NAMES}
            metric = lambda p, k, cell=cell: [r[k] for r in cell[p]]
            policies: dict[str, dict[str, Any]] = {p: {k: float(np.mean(np.array(metric(p, k), float))) for k in
                ('regret', 'found', 'unique', 'repeats', 'optimism', 'posterior_rec_regret')} for p in cell}
            entry: dict[str, Any] = {'env': name, 'budget': budget, 'n_seeds': len(seeds), 'policies': policies}
            for p in ('baseline', 'noise_aware', 'loop_random'):
                entry['policies'][p]['repeat_kinds'] = {k: float(np.mean([r['repeat_kinds'][k] for r in cell[p]]))
                                                        for k in ('policy', 'diagnostic', 'final')}
                entry['policies'][p]['surprises'] = float(np.mean(np.array(metric(p, 'surprises'), float)))
            entry['paired_regret'] = {f'{a}-{b}': paired(metric(a, 'regret'), metric(b, 'regret')) for a, b in
                                      (('noise_aware', 'baseline'), ('noise_aware', 'loop_random'),
                                       ('baseline', 'loop_random'), ('noise_aware', 'random_unconstrained'),
                                       ('baseline', 'random_unconstrained'))}
            summary.append(entry)
    return {'summary': summary, 'rows': rows}


def main():
    from threadpoolctl import threadpool_limits
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--envs', nargs='+', default=['upo_abts', 'icfree_cole1'])
    p.add_argument('--seeds', default='0-9', help='e.g. 0-9 or 0,3,5')
    p.add_argument('--budgets', nargs='+', type=int, default=[12, 24])
    p.add_argument('--out', help='write raw JSON results here')
    a = p.parse_args()
    seeds = (list(range(int(a.seeds.split('-')[0]), int(a.seeds.split('-')[1])+1)) if '-' in a.seeds
             else [int(s) for s in a.seeds.split(',')])
    start = time.time()
    with threadpool_limits(1):
        result = benchmark(a.envs, seeds, a.budgets)
    result['config'] = {'envs': a.envs, 'seeds': seeds, 'budgets': a.budgets, 'prior_noise': PRIOR_NOISE,
                            'prior_dof': PRIOR_DOF, 'noise_floor': NOISE_FLOOR, 'seconds': round(time.time()-start, 1)}
    print(f"{'env':13} {'B':>3} {'policy':20} {'regret':>7} {'found':>5} {'uniq':>5} {'reps':>5} "
          f"{'pol/diag/fin':>14} {'surpr':>5} {'optim':>7} {'postRec':>7}")
    for s in result['summary']:
        for name, m in s['policies'].items():
            k = m.get('repeat_kinds')
            kinds = '/'.join(f'{k[x]:.1f}' for x in ('policy', 'diagnostic', 'final')) if k else '-'
            print(f"{s['env']:13} {s['budget']:3} {name:20} {m['regret']:7.3f} {m['found']:5.2f} {m['unique']:5.1f} "
                  f"{m['repeats']:5.1f} {kinds:>14} {m.get('surprises', float('nan')):5.1f} {m['optimism']:7.3f} "
                  f"{m['posterior_rec_regret']:7.3f}")
        for pair, d in s['paired_regret'].items():
            print(f"  {pair:33} mean {d['mean']:+.3f} 95%CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] "
                  f"lower/tie/higher {d['lower']}/{d['ties']}/{d['higher']}")
    print(f"{result['config']['seconds']} s")
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(result, f, indent=1, allow_nan=False)


if __name__ == '__main__':
    main()
