"""Offline experiment: observation-gated literature priors on the unmodified minimal_lab loop.

The GP still selects every experiment. A prior pi(x) over candidate SETTINGS only reweights
expected improvement (piBO, Hvarfner et al. 2022, arXiv:2204.11051):

    acquisition = EI(x) * max(pi(x), eps) ** (gate * beta / n)

`beta / n` decays the prior with the number n of observations, `eps` bounds its log-influence,
and `gate` in [0, 1] is Phi(atanh(rho) * sqrt((m-3)/1.06)) for the Spearman rho between log pi
and observed means over m unique conditions. It is a heuristic agreement score, NOT a
calibrated probability: points are chosen adaptively and means are noisy. Disagreeing data
drives the gate to 0, so EI alone resumes. Literature is never a pseudo-observation and priors never see labels.

Within each worker process `loop.shortlist` is patched in scope only; `loop.run` is unchanged.
Hidden table means are read by `score` after a run finishes, never by the policy.

    OMP_NUM_THREADS=1 python -m minimal_lab.gated_prior_experiment run --out raw.jsonl
    python -m minimal_lab.gated_prior_experiment summary raw.jsonl
    python -m minimal_lab.gated_prior_experiment verify-citations
"""
import argparse
import asyncio
import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from unittest import mock

import numpy as np
from scipy.stats import norm, spearmanr
from threadpoolctl import threadpool_limits

from bo_eval.env import get_env
from minimal_lab import loop

EPS = 0.05
HYPOTHESIS_SEED = 991  # synthetic priors: separate from evaluation seeds, drawn before any run
N_SYNTHETIC = 8

# Cited knowledge only. Unlisted axes stay flat. Quotes are exact abstract substrings.
LITERATURE: dict[str, dict] = {
    'upo_abts': {
        'id': 'literature', 'kind': 'literature',
        'centre': {'ph': 6.0, 'cosubstrate_conc': 2.0}, 'width': {'ph': 0.18, 'cosubstrate_conc': 0.2},
        'sources': [
            {'doi': '10.1128/aem.70.8.4575-4581.2004',
             'quote': 'The optimum pH for the oxidation of aryl alcohols was found to be around 7',
             'transfer_limit': 'AaeUPO on aryl alcohols, not ABTS; enzyme source of the table is unstated.'},
            {'doi': '10.1128/aem.70.8.4575-4581.2004',
             'quote': 'the enzyme required relatively high concentrations of H(2)O(2) (2 mM) for optimum activity',
             'transfer_limit': 'Assumes cosubstrate_conc is H2O2 in mM; the CSV omits units.'},
            {'doi': '10.1186/2191-0855-1-31',
             'quote': 'The pH optimum of the M. rotula enzyme was found to vary between pH 5 and 6 for most reactions studied.',
             'transfer_limit': 'Different UPO (MroUPO) and mostly non-ABTS reactions.'},
        ]},
    # No abstract-level numeric optimum found for this proCFPS colicin E1 system: no literature prior.
}


def make_priors(env_name):
    env = get_env(env_name)
    lo, hi = env.X.min(0), env.X.max(0)
    priors = [{'id': 'vague', 'kind': 'uninformative',
               'centre': dict(zip(env.params, ((lo+hi)/2).tolist())), 'width': {p: 0.5 for p in env.params},
               'sources': [], 'note': 'Box centre, wide. Domain-agnostic default, not a claim.'}]
    if env_name in LITERATURE:
        priors.append(LITERATURE[env_name])
    rng = np.random.default_rng([HYPOTHESIS_SEED, len(env.params), len(env.X)])
    for k, i in enumerate(rng.choice(len(env.X), N_SYNTHETIC, replace=False)):
        priors.append({'id': f'synthetic{k}', 'kind': 'synthetic', 'centre': env.condition(int(i)),
                       'width': {p: 0.15 for p in env.params}, 'sources': [],
                       'note': 'SYNTHETIC hypothesis: a uniformly drawn candidate setting, independent of outcomes.'})
    return priors


def log_prior(X, names, prior):
    """log pi on candidate settings, normalised so max pi = 1, floored at log(EPS)."""
    span = np.ptp(X, axis=0)
    Z = (X - X.min(0)) / np.where(span > 0, span, 1)
    logp = np.zeros(len(X))
    for p, c in prior['centre'].items():
        j = names.index(p)
        z = (c - X[:, j].min()) / (span[j] or 1)
        logp -= 0.5 * ((Z[:, j] - z) / prior['width'][p]) ** 2
    return np.maximum(logp - np.max(logp), np.log(EPS))


def gate(logp, observations, direction):
    means = loop.observed_means(observations)
    m = len(means)
    if m < 4:
        return 0.5
    ids = list(means)
    rho = spearmanr(logp[ids], [direction*means[i] for i in ids]).statistic
    if not np.isfinite(rho):
        return 0.5
    return float(norm.cdf(np.arctanh(np.clip(rho, -0.999, 0.999)) * np.sqrt((m-3)/1.06)))


def prior_shortlist(logp, beta, gated, trace):
    """Wrap the unmodified shortlist; only its first (EI) option is re-ranked by the prior."""
    def shortlist(X, observations, rng, direction=1):
        options, gp = base_shortlist(X, observations, rng, direction)
        if gp is None:
            return options, gp
        good = [o for o in observations if o['value'] is not None]
        span = np.ptp(X, axis=0)
        mu, sd = gp.predict((X - X.min(0)) / np.where(span > 0, span, 1), return_std=True)
        y = direction*np.array([o['value'] for o in good])
        mu, sd = mu*max(y.std(), 1.0) + y.mean(), sd*max(y.std(), 1.0)
        means = loop.observed_means(good)
        delta = mu - max(direction*v for v in means.values())
        ei = delta*norm.cdf(delta/np.maximum(sd, 1e-9)) + sd*norm.pdf(delta/np.maximum(sd, 1e-9))
        g = gate(logp, good, direction) if gated else 1.0
        pick = int(np.argmax(np.log(np.maximum(ei, 1e-300)) + g*beta/len(good)*logp))
        trace.append({'n': len(good), 'gate': g, 'ei_pick': int(np.argmax(ei)), 'pick': pick})
        if pick == options[0]['candidate']:
            return options, gp
        first = {'candidate': pick, 'mean': float(direction*mu[pick]), 'sd': float(sd[pick]), 'ei': float(ei[pick]),
                     'reason': f'Prior-weighted EI (gate {g:.2f}); GP still ranks, prior only reweights.'}
        return [first] + [o for o in options if o['candidate'] != pick], gp
    return shortlist


base_shortlist = loop.shortlist


def black_box(env, seed, calls):
    """Common random numbers: the k-th visit to a condition gets the same noise in every policy."""
    visits = {}
    def execute(params):
        i = env.index(params)
        k = visits[i] = visits.get(i, -1) + 1
        calls.append(i)
        value = env.sample(i, np.random.default_rng([seed, i, k]))
        return {'value': value, 'reported': [params[p] for p in env.params],
                    'report_sd': [0.0]*len(env.params), 'ok': True}
    return execute


def run_one(job):
    env_name, budget, seed, policy, prior = job
    env = get_env(env_name)
    direction = 1 if env.goal == 'maximize' else -1
    calls, trace = [], []
    execute = black_box(env, seed, calls)
    with threadpool_limits(1):
        if policy == 'random':
            obs = []
            for i in np.random.default_rng(seed).choice(len(env.X), budget-1, replace=False):
                obs.append(dict(execute(env.condition(int(i))), candidate=int(i)))
            means = loop.observed_means(obs)
            best = max(means, key=lambda i: direction*means[i])
            execute(env.condition(best))
            rec = best
        else:
            patch = mock.patch.object(loop, 'shortlist', base_shortlist) if prior is None else mock.patch.object(
                loop, 'shortlist', prior_shortlist(log_prior(env.X, env.params, prior), budget/10,
                                                   policy == 'gated', trace))
            with patch:
                episode = asyncio.run(loop.run(env.X.copy(), env.params, execute, budget=budget,
                                               seed=seed, goal=env.goal))
            rec = episode['result']['candidate']
    return {'env': env_name, 'budget': budget, 'seed': seed, 'policy': policy,
                'prior': None if prior is None else prior['id'], 'calls': calls, 'recommended': rec, 'trace': trace} | score(env, rec, calls, prior)


def score(env, rec, calls, prior):
    """Post-hoc only: hidden means score the recommendation and, for analysis, the prior."""
    y = env.df[env.mean_col].to_numpy()
    best = env.true_value(env.optimum)
    out = {'regret': abs(best - y[rec]) / abs(best), 'found_optimal': rec == env.optimum,
               'n_calls': len(calls), 'unique': len(set(calls)), 'repeats': len(calls)-len(set(calls))}
    if prior is not None:
        mode = int(np.argmax(log_prior(env.X, env.params, prior)))
        out['prior_mode_regret'] = abs(best - y[mode]) / abs(best)
        out['prior_mode_percentile'] = float((y < y[mode]).mean())
    return out


def jobs(envs, budgets, seeds):
    for env_name in envs:
        priors = make_priors(env_name)
        for budget in budgets:
            for seed in seeds:
                yield env_name, budget, seed, 'random', None
                yield env_name, budget, seed, 'baseline', None
                for prior in priors:
                    for policy in ('bounded', 'gated'):
                        yield env_name, budget, seed, policy, prior


def quality(r):
    if r['prior'] == 'vague':
        return 'uninformative'
    return 'correct' if r['prior_mode_percentile'] >= 0.9 else 'misleading' if r['prior_mode_percentile'] < 0.5 else 'partial'


def paired(diffs, rng):
    d = np.asarray(diffs)
    boot = rng.choice(d, (10000, len(d))).mean(1)
    return d.mean(), d.std(ddof=1)/np.sqrt(len(d)) if len(d) > 1 else 0.0, *np.quantile(boot, [0.025, 0.975])


def cluster(values, seeds, rng):
    v, seeds = np.asarray(values), np.asarray(seeds)
    ids = np.unique(seeds)
    sums = np.array([v[seeds == i].sum() for i in ids])
    counts = np.array([(seeds == i).sum() for i in ids])
    pick = rng.integers(len(ids), size=(10000, len(ids)))
    boot = sums[pick].sum(1) / counts[pick].sum(1)
    return v.mean(), *np.quantile(boot, [0.025, 0.975])


def summary(rows):
    rng = np.random.default_rng(0)
    key = lambda r: (r['env'], r['budget'], r['seed'])
    base = {key(r): r['regret'] for r in rows if r['policy'] == 'baseline'}
    groups = {}
    for r in rows:
        label = r['policy'] if r['prior'] is None else f"{r['policy']}:{r['prior']}[{quality(r)}]"
        groups.setdefault((r['env'], r['budget'], label), []).append(r)
    print('env budget policy:prior[post-hoc quality] | regret mean+-se | found | unique | repeats | '
          'final gate | prior overrides | paired d(regret vs baseline) mean+-se [95% boot CI] W/T/L')
    for (env, budget, label), rs in sorted(groups.items()):
        reg = np.array([r['regret'] for r in rs])
        d = [r['regret'] - base[key(r)] for r in rs]
        m, se, lo, hi = paired(d, rng)
        gates = [r['trace'][-1]['gate'] for r in rs if r['trace']]
        over = [sum(t['pick'] != t['ei_pick'] for t in r['trace']) for r in rs]
        print(f'{env} {budget} {label} | {reg.mean():.3f}+-{reg.std(ddof=1)/np.sqrt(len(reg)):.3f} | '
              f'{np.mean([r["found_optimal"] for r in rs]):.1f} | {np.mean([r["unique"] for r in rs]):.1f} | '
              f'{np.mean([r["repeats"] for r in rs]):.1f} | {np.mean(gates) if gates else float("nan"):.2f} | '
              f'{np.mean(over):.1f} | {m:+.3f}+-{se:.3f} [{lo:+.3f},{hi:+.3f}] '
              f'{sum(x < 0 for x in d)}/{sum(x == 0 for x in d)}/{sum(x > 0 for x in d)}')
    print('\nPooled over priors by post-hoc quality, paired d(regret) vs baseline and gated-bounded on the same '
          'prior/seed; 95% CI = seed-cluster bootstrap (baseline and RNG are shared within a seed):')
    by = {}
    for r in rows:
        if r['prior'] is not None:
            by.setdefault((r['env'], r['budget'], quality(r), r['prior'], r['seed']), {})[r['policy']] = r['regret']
    pooled = {}
    for (env, budget, q, _, seed), v in by.items():
        pooled.setdefault((env, budget, q), []).append((v['bounded'] - base[(env, budget, seed)],
                                                        v['gated'] - base[(env, budget, seed)],
                                                        v['gated'] - v['bounded'], seed))
    for (env, budget, q), v in sorted(pooled.items()):
        a = np.array(v)
        s = [cluster(a[:, j], a[:, 3], rng) for j in range(3)]
        s = [(m, None, lo, hi) for m, lo, hi in s]
        print(f'{env} {budget} {q:13s} pairs={len(a):3d} seeds={len(set(a[:, 3]))} | bounded {s[0][0]:+.3f} [{s[0][2]:+.3f},{s[0][3]:+.3f}] | '
              f'gated {s[1][0]:+.3f} [{s[1][2]:+.3f},{s[1][3]:+.3f}] | gated-bounded {s[2][0]:+.3f} '
              f'[{s[2][2]:+.3f},{s[2][3]:+.3f}]')


def verify_citations():
    for prior in LITERATURE.values():
        for s in prior['sources']:
            url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode(
                {'query': f'DOI:"{s["doi"]}"', 'resultType': 'core', 'format': 'json'})
            abstract = json.load(urllib.request.urlopen(url, timeout=30))['resultList']['result'][0]['abstractText']
            print('OK ' if s['quote'] in abstract else 'MISSING', s['doi'], repr(s['quote'][:60]))


def main():
    cli = argparse.ArgumentParser()
    sub = cli.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--envs', nargs='+', default=['upo_abts', 'icfree_cole1'])
    r.add_argument('--budgets', nargs='+', type=int, default=[12, 24])
    r.add_argument('--seeds', nargs='+', type=int, default=list(range(10)))
    r.add_argument('--workers', type=int, default=8)
    r.add_argument('--out', required=True)
    sub.add_parser('summary').add_argument('raw')
    sub.add_parser('verify-citations')
    a = cli.parse_args()
    if a.cmd == 'verify-citations':
        return verify_citations()
    if a.cmd == 'summary':
        return summary(load(a.raw))
    todo = list(jobs(a.envs, a.budgets, a.seeds))
    with open(a.out, 'x') as f, ProcessPoolExecutor(a.workers) as pool:
        for k, row in enumerate(pool.map(run_one, todo, chunksize=4)):
            f.write(json.dumps(row) + '\n')
            print(f'\r{k+1}/{len(todo)}', end='', file=sys.stderr, flush=True)
    print(file=sys.stderr)
    summary(load(a.out))


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


if __name__ == '__main__':
    main()
