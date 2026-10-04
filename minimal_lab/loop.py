"""One loop. The policy receives candidate settings and observations, never unseen labels."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


def observed_means(observations):
    grouped = {}
    for o in observations:
        if o['value'] is not None:
            grouped.setdefault(o['candidate'], []).append(o['value'])
    return {i:float(np.mean(values)) for i,values in grouped.items()}


def shortlist(X, observations, rng, direction=1):
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
    means = observed_means(good)
    incumbent = max(means, key=lambda i:direction*means[i])
    delta = mu-direction*means[incumbent]
    z = delta/np.maximum(sd, 1e-9)
    ei = delta*norm.cdf(z) + sd*norm.pdf(z)
    # An observation or deferral never proves a condition cannot improve.
    choices = [(int(np.argmax(ei)), 'Highest expected improvement across all conditions; repeats allowed.'),
               (int(np.argmax(sd)), 'Largest predictive uncertainty across all conditions; repeats allowed.'),
               (incumbent, 'Repeat the best observed mean to check noise and delivery.')]
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
              budget=12, seed=0, research=None, choose=None, on_event=lambda stage, graph: None):
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
    graph = {'nodes': [], 'edges': [], 'objective': objective, 'goal': goal, 'stop': 'running',
             'legend': dict(uncertainty=[
                'GP predictive SD is an approximation, not calibrated confidence.',
                'Delivered settings are simulated estimates with their own SD.',
                'Cited literature carries a stated transfer limit; retrieved-only sources are not evidence.',
                'A node without a stated confidence has none estimated; that is not certainty.'])}
    observations, papers, deferred = [], [], {}
    searches, check_id = 0, None

    def node(kind, **data):
        ident = f'N{len(graph["nodes"])}'
        graph['nodes'].append(dict(id=ident, kind=kind, **data))
        return ident

    def edge(source, target, reason, relation=None, **meta):
        graph['edges'].append(dict(source=source, target=target, reason=reason,
                                   **({'relation': relation} if relation else {}), **meta))

    def find(ident):
        return graph['nodes'][int(ident[1:])]

    def source_label(title):
        text = (title or 'Untitled source').strip()
        return text if len(text) <= 48 else text[:45].rstrip()+'…'

    root = node('goal', reason=f'{goal.capitalize()} {objective} within the experiment budget.',
                uncertainty='Optimisation of an observed proxy does not verify the underlying objective. '
                'Predictive intervals are approximate; source reliability and applicability require assessment.')
    for step in range(budget):
        on_event('planning', graph)
        options, gp = shortlist(X, observations, rng, direction)
        context = dict(objective=objective, goal=goal, parameters=names, options=[dict(o, params=dict(zip(names, X[o['candidate']].tolist())))
                                                  for o in options], history=observations)
        # Search once initially, once after a surprising result. A search must affect a future decision.
        if research and searches < 2 and (searches == 0 or check_id is not None):
            on_event('research', graph)
            searches += 1
            try:
                found = await research(context)
                for p in found:
                    p = dict(p)
                    p['id'] = node('source', round=step+1, label=source_label(p.get('title')),
                                   **{k:v for k,v in p.items() if k != 'id'})
                    papers.append(p)
            except Exception as error:
                nid = node('search_failure', reason=type(error).__name__)
                edge(root, nid, 'No evidence obtained; continue with measured feedback.',
                     relation='search_failed')
        forced = check_id
        if forced is not None and forced not in {o['candidate'] for o in options}:
            options.append(dict(candidate=forced, mean=None, sd=None, ei=None, reason='Diagnostic repeat.'))
        good = [o for o in observations if o['value'] is not None]
        # Spend the final attempt confirming the observed incumbent, not searching a new condition.
        if step == budget-1 and good and forced is None:
            means = observed_means(good)
            forced = max(means, key=lambda i:direction*means[i])
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
        edge(root if not observations else observations[-1]['id'], d,
             'Evidence available before selection.', relation='informs')
        for prior in deferred.pop(selected, []):
            edge(prior, d, 'Reconsidered and selected; the earlier deferral was not a refutation.',
                 relation='reconsidered')
        branch_nodes = {selected:d}
        for i, why in decision['alternatives'].items():
            a = node('deferred', candidate=int(i), reason=why, status='Reconsiderable; not falsified.')
            deferred.setdefault(int(i), []).append(a)
            branch_nodes[int(i)] = a
            edge(d, a, 'Not selected in this round.', relation='defers')
        cited = {c['id'] for c in decision['citations']}
        for citation in decision['citations']:
            branch = citation.get('candidate', selected)
            # Quote and limit stay scoped to this edge and this round's branch.
            edge(citation['id'], branch_nodes[branch],
                 'Literature interpretation: '+citation['transfer_limit'], relation='cites',
                 citation=dict(round=step+1, candidate=branch,
                               quote=citation['quote'], transfer_limit=citation['transfer_limit']))
            target = find(branch_nodes[branch])
            target['cited'] = sorted(set(target.get('cited', []) + [citation['id']]))
            paper_node = find(citation['id'])
            paper_node['cited_in'] = sorted(set(paper_node.get('cited_in', []) + [step+1]))
        for paper in papers:
            if paper['id'] not in cited:
                edge(paper['id'], d, 'Retrieved but not cited as a selection reason.',
                     relation='retrieved')
        # Decision is committed before execution. No result can rewrite its rationale.
        on_event('decision', graph)
        observation = execute(dict(zip(names, X[selected].tolist())))
        observation.update(candidate=selected)
        oid = node('observation', round=step+1, **observation)
        observation['id'] = oid
        edge(d, oid, 'Executed this decision.', relation='executes')
        observations.append(observation)
        check_id = None
        if gp is not None and observation['value'] is not None and forced is None:
            train = [direction*o['value'] for o in observations[:-1] if o['value'] is not None]
            z = (np.array(observation['reported'])-X.min(0))/np.where(np.ptp(X,axis=0)>0,np.ptp(X,axis=0),1)
            mu, sd = gp.predict(z.reshape(1,-1), return_std=True)
            residual = abs(direction*observation['value']-(mu[0]*max(np.std(train),1)+np.mean(train)))
            if residual > 2*sd[0]*max(np.std(train),1):
                check_id = selected
                graph['nodes'][int(oid[1:])]['surprise'] = 'Outside pre-update 2-SD predictive interval; repeat before interpretation.'
        on_event('observation', graph)
        if not observation.get('ok', True):
            graph['stop'] = 'Execution failed; no reward fabricated.'
            break
    if graph['stop'] == 'running':
        graph['stop'] = 'budget exhausted'
    means = observed_means(observations)
    best = max(means, key=lambda i:direction*means[i]) if means else None
    result = dict(candidate=best, params=None if best is None else dict(zip(names, X[best].tolist())),
                  mean=None if best is None else means[best],
                  repeats=sum(o['candidate']==best and o['value'] is not None for o in observations),
                  rule=f'{goal.capitalize()} mean observed proxy by intended parameters, including repeats.',
                  uncertainty='Best observed proxy, not a verified objective or proven optimum. '
                  'One confirmation cannot establish reproducibility.')
    final = node('recommendation', **result)
    for o in observations:
        if o['candidate'] == best:
            edge(o['id'], final, 'Observed result used in the protocol mean.', relation='aggregates')
    on_event('recommendation', graph)
    return dict(result=result, graph=graph, observations=observations)


def save(episode, directory):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    (out/'graph.json').write_text(json.dumps(episode['graph'], indent=2, allow_nan=False))
    (out/'result.json').write_text(json.dumps(episode['result'], indent=2, allow_nan=False))
    template = Path(__file__).with_name('viewer.html').read_text()
    payload = json.dumps(episode['graph'], allow_nan=False).replace('<', '\\u003c')
    (out/'graph.html').write_text(template.replace('GRAPH_DATA', payload))
