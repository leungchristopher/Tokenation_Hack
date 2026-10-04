"""One loop. The policy receives candidate settings and observations, never unseen labels."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


def shortlist(X, observations, attempted, rng, direction=1):
    """EI, response exploration and an incumbent repeat. No literature pseudo-observations."""
    good = [o for o in observations if o['value'] is not None]
    span = np.ptp(X, axis=0)
    scale = np.where(span > 0, span, 1)
    Z = (X - X.min(0)) / scale
    if not good:
        ids = rng.choice(len(X), min(3, len(X)), replace=False)
        return [dict(candidate=int(i), mean=None, sd=None, ei=None,
                     reason='Initial coverage: no response measurements yet.') for i in ids], None
    inputs = np.array([o['reported'] for o in good])
    # Regress on reported delivery, not hidden realised settings or intended settings.
    y = direction*np.array([o['value'] for o in good])
    centre, spread = y.mean(), max(y.std(), 1.0)
    gp = GaussianProcessRegressor(kernel=Matern(length_scale=0.3, nu=2.5)
                                  + WhiteKernel(noise_level=0.05), optimizer=None)
    gp.fit((inputs-X.min(0))/scale, (y-centre)/spread)
    # First-order propagation of reported input uncertainty into response variance.
    # This is an approximation, not a calibrated uncertain-input GP.
    input_sd = np.array([o.get('report_sd', np.zeros(X.shape[1])) for o in good])/scale
    training = (inputs-X.min(0))/scale
    variance = np.zeros(len(good))
    for j in range(X.shape[1]):
        shift = np.zeros_like(training)
        shift[:,j] = input_sd[:,j]
        variance += ((gp.predict(training+shift)-gp.predict(training-shift))/2)**2
    gp.alpha = 1e-8+variance
    gp.fit(training, (y-centre)/spread)
    mu, sd = gp.predict(Z, return_std=True)
    mu, sd = mu*spread+centre, sd*spread
    delta = mu-max(y)
    z = delta/np.maximum(sd, 1e-9)
    ei = delta*norm.cdf(z) + sd*norm.pdf(z)
    unseen = np.array([i for i in range(len(X)) if i not in attempted], dtype=int)
    incumbent = max(good, key=lambda o: direction*o['value'])['candidate']
    choices = []
    if len(unseen):
        choices += [(int(unseen[np.argmax(ei[unseen])]), 'Highest expected improvement.'),
                    (int(unseen[np.argmax(sd[unseen])]), 'Largest predictive uncertainty among untested conditions.')]
    choices += [(incumbent, 'Repeat the best observed condition to check noise and delivery.')]
    unique = {i: why for i, why in reversed(choices)}
    return [dict(candidate=i, mean=float(direction*mu[i]), sd=float(sd[i]), ei=float(ei[i]), reason=unique[i])
            for i in dict(choices)], gp


def validate_choice(reply, options, papers):
    ids = {o['candidate'] for o in options}
    if reply.get('selected') not in ids or not str(reply.get('reason', '')).strip():
        raise ValueError('Choose a shortlisted candidate and give a reason.')
    alternatives = reply.get('alternatives', {})
    if set(alternatives) != {str(i) for i in ids if i != reply['selected']}:
        raise ValueError('Every deferred candidate needs a reason.')
    if any(not isinstance(v, str) or not v.strip() for v in alternatives.values()):
        raise ValueError('Empty deferral reason.')
    sources = {p['id']: p for p in papers}
    for citation in reply.get('citations', []):
        paper = sources.get(citation.get('id'))
        quote = citation.get('quote', '')
        if not paper or not quote or quote not in paper['abstract']:
            raise ValueError('Citation must quote a retrieved abstract exactly.')
        if not citation.get('transfer_limit'):
            raise ValueError('State the limitation of transferring this evidence.')
        if citation.get('candidate', reply['selected']) not in ids:
            raise ValueError('A citation must qualify a shortlisted branch.')
    if not reply.get('uncertainty'):
        raise ValueError('State what remains unknown.')
    return {key:reply[key] for key in ('selected','reason','alternatives','uncertainty')} | {
        'citations':reply.get('citations', [])}


async def run(X, names, execute, *, objective='observed reward proxy', goal='maximize',
              budget=12, seed=0, research=None, choose=None):
    """execute(params) -> public observation. research/choose only receive public context."""
    if goal not in ('maximize', 'minimize'):
        raise ValueError('goal must be maximize or minimize.')
    if budget < 1 or not np.isfinite(X).all() or not len(X):
        raise ValueError('Use finite candidates and a positive budget.')
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError('One distinct parameter name per candidate column is required.')
    direction = 1 if goal == 'maximize' else -1
    rng = np.random.default_rng(seed)
    graph = {'nodes': [], 'edges': [], 'objective': objective, 'goal': goal, 'stop': 'budget exhausted'}
    observations, papers, attempted = [], [], set()
    searches, check_id = 0, None

    def node(kind, **data):
        ident = f'N{len(graph["nodes"])}'
        graph['nodes'].append(dict(id=ident, kind=kind, **data))
        return ident

    def edge(source, target, reason):
        graph['edges'].append(dict(source=source, target=target, reason=reason))

    root = node('goal', reason=f'{goal.capitalize()} {objective} within the experiment budget.',
                uncertainty='Optimisation of an observed proxy does not verify the underlying objective. '
                'Predictive intervals are approximate; source reliability and applicability require assessment.')
    for step in range(budget):
        options, gp = shortlist(X, observations, attempted, rng, direction)
        context = dict(objective=objective, goal=goal, parameters=names, options=[dict(o, params=dict(zip(names, X[o['candidate']].tolist())))
                                                  for o in options], history=observations)
        # Search once initially, once after a surprising result. A search must affect a future decision.
        if research and searches < 2 and (searches == 0 or check_id is not None):
            searches += 1
            try:
                found = await research(context)
                for p in found:
                    p = dict(p)
                    p['id'] = node('source', **{k:v for k,v in p.items() if k != 'id'})
                    papers.append(p)
            except Exception as error:
                nid = node('search_failure', reason=type(error).__name__)
                edge(root, nid, 'No evidence obtained; continue with measured feedback.')
        forced = check_id
        if forced is not None and forced not in {o['candidate'] for o in options}:
            options.append(dict(candidate=forced, mean=None, sd=None, ei=None, reason='Diagnostic repeat.'))
        good = [o for o in observations if o['value'] is not None]
        # Spend the final attempt confirming the observed incumbent, not searching a new condition.
        if step == budget-1 and good and forced is None:
            forced = max(good, key=lambda o:direction*o['value'])['candidate']
        default = forced if forced is not None else options[0]['candidate']
        reason = ('Repeat after a surprising result; response-model error and delivery error remain alternatives.'
                  if check_id is not None else 'Final confirmation of the best observed condition.'
                  if forced is not None else options[0]['reason'])
        decision = dict(selected=default, reason=reason, citations=[],
                        uncertainty='Predictive uncertainty mixes response and noise. Delivery is only estimated.',
                        alternatives={str(o['candidate']): 'Deferred under the numerical policy: '+o['reason']
                                      for o in options if o['candidate'] != default})
        context['options'] = [dict(o, params=dict(zip(names, X[o['candidate']].tolist()))) for o in options]
        if choose and forced is None:
            try:
                decision = validate_choice(await choose(dict(context, papers=papers)), options, papers)
            except Exception as error:
                decision['fallback'] = type(error).__name__
        selected = decision['selected']
        d = node('decision', round=step+1, **decision, options=context['options'])
        edge(root if not observations else observations[-1]['id'], d, 'Evidence available before selection.')
        branch_nodes = {selected:d}
        for i, why in decision['alternatives'].items():
            a = node('deferred', candidate=int(i), reason=why, status='Reconsiderable; not falsified.')
            branch_nodes[int(i)] = a
            edge(d, a, 'Not selected in this round.')
        for citation in decision['citations']:
            edge(citation['id'], branch_nodes[citation.get('candidate', selected)],
                 'Literature interpretation: '+citation['transfer_limit'])
        for paper in papers:
            if paper['id'] not in {c['id'] for c in decision['citations']}:
                edge(paper['id'], d, 'Retrieved but not cited as a selection reason.')
        # Decision is committed before execution. No result can rewrite its rationale.
        observation = execute(dict(zip(names, X[selected].tolist())))
        observation.update(candidate=selected)
        oid = node('observation', round=step+1, **observation)
        observation['id'] = oid
        edge(d, oid, 'Executed this decision.')
        observations.append(observation)
        attempted.add(selected)
        check_id = None
        if gp is not None and observation['value'] is not None and forced is None:
            train = [direction*o['value'] for o in observations[:-1] if o['value'] is not None]
            z = (np.array(observation['reported'])-X.min(0))/np.where(np.ptp(X,axis=0)>0,np.ptp(X,axis=0),1)
            mu, sd = gp.predict(z.reshape(1,-1), return_std=True)
            residual = abs(direction*observation['value']-(mu[0]*max(np.std(train),1)+np.mean(train)))
            if residual > 2*sd[0]*max(np.std(train),1):
                check_id = selected
                graph['nodes'][int(oid[1:])]['surprise'] = 'Outside pre-update 2-SD predictive interval; repeat before interpretation.'
        if not observation.get('ok', True):
            graph['stop'] = 'Execution failed; no reward fabricated.'
            break
    grouped = {}
    for o in observations:
        if o['value'] is not None:
            grouped.setdefault(o['candidate'], []).append(o['value'])
    best = max(grouped, key=lambda i:direction*np.mean(grouped[i])) if grouped else None
    result = dict(candidate=best, params=None if best is None else dict(zip(names, X[best].tolist())),
                  mean=None if best is None else float(np.mean(grouped[best])),
                  repeats=0 if best is None else len(grouped[best]),
                  rule=f'{goal.capitalize()} mean observed proxy by intended parameters, including repeats.',
                  uncertainty='Best observed proxy, not a verified objective or proven optimum. '
                  'One confirmation cannot establish reproducibility.')
    final = node('recommendation', **result)
    for o in observations:
        if o['candidate'] == best:
            edge(o['id'], final, 'Observed result used in the protocol mean.')
    return dict(result=result, graph=graph, observations=observations)


def save(episode, directory):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    (out/'graph.json').write_text(json.dumps(episode['graph'], indent=2, allow_nan=False))
    (out/'result.json').write_text(json.dumps(episode['result'], indent=2, allow_nan=False))
    template = Path(__file__).with_name('viewer.html').read_text()
    payload = json.dumps(episode['graph'], allow_nan=False).replace('<', '\\u003c')
    (out/'graph.html').write_text(template.replace('GRAPH_DATA', payload))
