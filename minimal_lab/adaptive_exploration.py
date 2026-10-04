"""Space-filling start + bounded adaptive exploration for the minimal loop (offline, additive).

`shortlist` is a drop-in for `minimal_lab.loop.shortlist`: same signature, same option records,
and it returns the loop's own fixed-kernel GP so surprise checks in `loop.run` are unchanged.
Only *selection* differs:

1. Space-filling start: the first `n_init(budget, d)` choices are greedy maximin points of the
   finite candidate set (seeded random first point, then the candidate farthest from every
   reported input). Uses candidate settings only, never responses.
2. Afterwards: expected improvement on the latent response of a GP whose ARD Matern length
   scales and noise are fitted by marginal likelihood (bounded), incumbent = best posterior mean
   at a measured condition. If EI is negligible (the model expects no improvement anywhere) the
   choice switches to the largest *epistemic* SD, but for at most EXPLORE_FRACTION of the budget.

Nothing is excluded: measured and deferred conditions stay eligible in every acquisition.
Predeclared settings are the module constants below; they were fixed before evaluation.

CLI (offline CPU benchmark, hidden means are read only by the scorer after each run):
    python -m minimal_lab.adaptive_exploration --tasks upo_abts icfree_cole1 \
        --seeds 0-9 --budgets 12 24 --out results.json
"""
import argparse
import asyncio
import json
import types
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from minimal_lab import loop

LENGTH_BOUNDS = (0.05, 20.0)    # normalised [0, 1] inputs
NOISE_BOUNDS = (1e-4, 1.0)      # standardised response units
RESTARTS = 2
EI_TOL = 1e-3                   # EI below this x response SD counts as negligible
EXPLORE_FRACTION = 0.25


def n_init(budget, dims):
    return int(min(dims + 1, max(1, budget // 3)))


def maximin(Z, measured, rng):
    """Candidate farthest from all measured inputs (seeded random choice if nothing measured)."""
    if not len(measured):
        return int(rng.integers(len(Z)))
    gap = np.min(((Z[:, None, :] - np.asarray(measured)[None]) ** 2).sum(-1), axis=1)
    return int(np.argmax(gap))


def fit_latent_gp(training, y, input_sd, seed=0):
    """ARD Matern + noise by bounded marginal likelihood; first-order input-report variance."""
    d = training.shape[1]
    kernel = (ConstantKernel(1.0, (1e-2, 1e2)) * Matern(np.full(d, 0.5), LENGTH_BOUNDS, nu=2.5)
              + WhiteKernel(0.1, NOISE_BOUNDS))
    gp = GaussianProcessRegressor(kernel, normalize_y=False, n_restarts_optimizer=RESTARTS,
                                  random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        gp.fit(training, y)
        if np.any(input_sd > 0):
            variance = np.zeros(len(y))
            for j in range(d):
                shift = np.zeros_like(training)
                shift[:, j] = input_sd[:, j]
                variance += ((gp.predict(training + shift) - gp.predict(training - shift)) / 2) ** 2
            gp = GaussianProcessRegressor(gp.kernel_, alpha=1e-8 + variance,
                                          n_restarts_optimizer=0, random_state=seed).fit(training, y)
    return gp


def shortlist(X, observations, rng, direction=1, *, budget=None, state=None):
    """Drop-in replacement for loop.shortlist; `state` counts exploration steps across rounds."""
    state = {} if state is None else state
    good = [o for o in observations if o['value'] is not None]
    span = np.ptp(X, axis=0)
    scale = np.where(span > 0, span, 1)
    Z = (X - X.min(0)) / scale
    budget = budget or len(X)
    training = (np.array([o['reported'] for o in good]).reshape(-1, X.shape[1]) - X.min(0)) / scale
    # Loop-contract GP (fixed kernel) only so loop.run's surprise check behaves as in baseline.
    _, contract_gp = loop.shortlist(X, observations, rng, direction) if good else (None, None)
    if len(good) < n_init(budget, X.shape[1]):
        i = maximin(Z, training, rng)
        return [dict(candidate=i, mean=None, sd=None, ei=None,
                     reason='Space-filling start: candidate farthest from every measured input.')], contract_gp
    y = direction * np.array([o['value'] for o in good])
    centre, spread = y.mean(), y.std() if y.std() > 0 else 1.0
    input_sd = np.array([o.get('report_sd', np.zeros(X.shape[1])) for o in good]) / scale
    gp = fit_latent_gp(training, (y - centre) / spread, input_sd, seed=len(good))
    mu, sd = np.asarray(gp.predict(Z, return_std=True))
    fitted: Any = gp.kernel_
    noise = fitted.k2.noise_level
    epistemic = np.sqrt(np.maximum(sd ** 2 - noise, 1e-12))
    measured = sorted({o['candidate'] for o in good})
    incumbent = measured[int(np.argmax(mu[measured]))]
    delta = mu - mu[incumbent]
    z = delta / epistemic
    ei = delta * norm.cdf(z) + epistemic * norm.pdf(z)
    best_ei = int(np.argmax(ei))
    explore = int(np.argmax(epistemic))
    allowed = int(EXPLORE_FRACTION * budget)
    if ei[best_ei] < EI_TOL and state.get('explored', 0) < allowed and explore != best_ei:
        state['explored'] = state.get('explored', 0) + 1
        choices = [(explore, 'Bounded exploration: EI negligible, largest epistemic SD.'),
                   (best_ei, 'Highest latent EI under a fitted ARD GP; repeats allowed.')]
    else:
        choices = [(best_ei, 'Highest latent EI under a fitted ARD GP; repeats allowed.'),
                   (explore, 'Largest epistemic SD across all conditions; repeats allowed.')]
    choices.append((incumbent, 'Repeat the best posterior-mean measured condition.'))
    unique = {i: why for i, why in reversed(choices)}
    m, s = mu * spread + centre, epistemic * spread
    return [dict(candidate=i, mean=float(direction * m[i]), sd=float(s[i]), ei=float(ei[i] * spread),
                 reason=unique[i]) for i in dict(choices)], contract_gp


def random_shortlist(X, observations, rng, direction=1):
    """Uniform random search over unmeasured conditions; no GP, so no surprise repeats."""
    seen = {o['candidate'] for o in observations}
    pool = [i for i in range(len(X)) if i not in seen] or list(range(len(X)))
    return [dict(candidate=int(rng.choice(pool)), mean=None, sd=None, ei=None,
                 reason='Uniform random unmeasured condition.')], None


def with_shortlist(policy):
    """Copy of loop.run whose global `shortlist` is `policy`. loop itself is never mutated."""
    run = loop.run
    scope = dict(run.__globals__, shortlist=policy)
    clone = types.FunctionType(run.__code__, scope, run.__name__, run.__defaults__, run.__closure__)
    clone.__kwdefaults__ = run.__kwdefaults__
    return clone


def black_box(env, seed):
    """env.sample with common random numbers keyed by (seed, condition, repeat): identical
    choices get identical draws across policies; different choices get different outcomes."""
    visits = {}

    def execute(params):
        i = env.index(params)
        k = visits[i] = visits.get(i, -1) + 1
        value = env.sample(i, np.random.default_rng([seed, i, k]))
        return dict(value=value, ok=True, reported=[params[p] for p in env.params],
                    report_sd=[0.0] * len(env.params))
    return execute


def episode(task, policy, seed, budget):
    from threadpoolctl import threadpool_limits

    from bo_eval.env import get_env
    env = get_env(task)
    state = {}
    if policy == 'baseline':
        runner = loop.run
    elif policy == 'random':
        runner = with_shortlist(random_shortlist)
    else:
        runner = with_shortlist(lambda X, obs, rng, direction=1:
                                shortlist(X, obs, rng, direction, budget=budget, state=state))
    with threadpool_limits(1):
        out = asyncio.run(runner(env.X.copy(), env.params, black_box(env, seed), goal=env.goal,
                                 budget=budget, seed=seed))
    # Scorer: the only place hidden means are read, after the run has finished.
    y = env.df[env.mean_col].to_numpy()
    opt = y[env.optimum]

    def regret(i):
        return abs(opt - y[i]) / max(abs(opt), 1e-12)
    chosen = [o['candidate'] for o in out['observations']]
    rec = out['result']['candidate']
    return dict(task=task, policy=policy, seed=seed, budget=budget, calls=len(chosen),
                unique=len(set(chosen)), repeats=len(chosen) - len(set(chosen)),
                recommended=rec, regret=float(regret(rec)), found_optimal=bool(y[rec] == opt),
                best_visited_regret=float(min(regret(i) for i in chosen)),
                diagnostic_repeats=sum(n['kind'] == 'decision' and n['reason'].startswith('Repeat after')
                                       for n in out['graph']['nodes']),
                explored=state.get('explored', 0) if policy == 'adaptive' else 0,
                chosen=chosen)


def summarise(rows):
    stats, rng = [], np.random.default_rng(12345)
    for task in dict.fromkeys(r['task'] for r in rows):
        for budget in dict.fromkeys(r['budget'] for r in rows):
            group = {p: sorted((r for r in rows if r['task'] == task and r['budget'] == budget
                                and r['policy'] == p), key=lambda r: r['seed'])
                     for p in ('baseline', 'random', 'adaptive')}
            entry = dict(task=task, budget=budget)
            for p, g in group.items():
                entry[p] = {k: float(np.mean([r[k] for r in g])) for k in
                            ('regret', 'found_optimal', 'best_visited_regret', 'unique', 'repeats', 'calls')}
            for other in ('baseline', 'random'):
                pairs = zip(group['adaptive'], group[other], strict=True)
                diff = np.array([a['regret'] - b['regret'] for a, b in pairs])
                boot = rng.choice(diff, (10000, len(diff))).mean(1)
                entry['adaptive_minus_' + other] = dict(
                    mean=float(diff.mean()), se=float(diff.std(ddof=1) / np.sqrt(len(diff))),
                    ci95=[float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                    wins=int((diff < 0).sum()), losses=int((diff > 0).sum()), ties=int((diff == 0).sum()),
                    per_seed=diff.round(4).tolist())
            stats.append(entry)
    return stats


def seeds(text):
    lo, _, hi = text.partition('-')
    return list(range(int(lo), int(hi or lo) + 1))


def main():
    parser = argparse.ArgumentParser(description='Offline space-filling/adaptive exploration benchmark.')
    parser.add_argument('--tasks', nargs='+', default=['upo_abts', 'icfree_cole1'])
    parser.add_argument('--seeds', type=seeds, default=seeds('0-9'))
    parser.add_argument('--budgets', nargs='+', type=int, default=[12, 24])
    parser.add_argument('--policies', nargs='+', default=['baseline', 'random', 'adaptive'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out')
    args = parser.parse_args()
    jobs = [(t, p, s, b) for t in args.tasks for b in args.budgets for s in args.seeds for p in args.policies]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(episode, *zip(*jobs, strict=True)))
    stats = summarise(rows) if set(args.policies) == {'baseline', 'random', 'adaptive'} else []
    for e in stats:
        cells = '  '.join(f"{p}: regret {e[p]['regret']:.3f} opt {e[p]['found_optimal']:.1f} "
                          f"uniq {e[p]['unique']:.1f}" for p in ('baseline', 'random', 'adaptive'))
        d = e['adaptive_minus_baseline']
        print(f"{e['task']:>13} B={e['budget']:<3} {cells}  | adaptive-baseline {d['mean']:+.3f} "
              f"CI[{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}] W/L/T {d['wins']}/{d['losses']}/{d['ties']}")
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(dict(settings=dict(LENGTH_BOUNDS=LENGTH_BOUNDS, NOISE_BOUNDS=NOISE_BOUNDS,
                                         RESTARTS=RESTARTS, EI_TOL=EI_TOL, EXPLORE_FRACTION=EXPLORE_FRACTION,
                                         n_init='min(d+1, budget//3)'), args=vars(args),
                           summary=stats, runs=rows), f, indent=1)


if __name__ == '__main__':
    main()
