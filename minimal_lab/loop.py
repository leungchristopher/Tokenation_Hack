"""One loop. The policy receives candidate settings and observations, never unseen labels."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from minimal_lab.evidence import prior_weights, validate_prior


def observed_means(observations):
    grouped = {}
    for o in observations:
        if o.get('ok', True) and o['value'] is not None:
            grouped.setdefault(o['candidate'], []).append(o['value'])
    return {i:float(np.mean(values)) for i,values in grouped.items()}


def shortlist(X, observations, rng, direction=1, weights=None):
    """EI, response exploration and an incumbent repeat. No literature pseudo-observations."""
    good = [o for o in observations if o['value'] is not None]
    span = np.ptp(X, axis=0)
    scale = np.where(span > 0, span, 1)
    Z = (X - X.min(0)) / scale
    if not good:
        ids = rng.choice(len(X), min(3, len(X)), replace=False).tolist()
        if weights is not None:
            preferred = int(np.argmax(weights))
            ids = [preferred] + [i for i in ids if i != preferred]
        return [dict(candidate=int(i), mean=None, sd=None, ei=None,
                     reason=('Bounded literature prior initial candidate.' if weights is not None and i == ids[0]
                             else 'Initial coverage: no response measurements yet.')) for i in ids], None
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
    # Acquisition needs uncertainty in the latent response, not irreducible reading noise.
    # Otherwise a noisy incumbent can keep winning EI even after many repeats.
    noise = getattr(getattr(getattr(gp, 'kernel_', None), 'k2', None), 'noise_level', 0.0)
    sd = np.sqrt(np.maximum(sd**2 - noise, 0.0))
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
    if weights is not None:
        choices.insert(0, (int(np.argmax(ei * weights)),
                           'Literature-weighted expected improvement; bounded, decaying prior.'))
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
              budget=12, seed=0, research=None, choose=None, on_event=lambda stage, graph: None,
              make_prior=None, research_query=None, replicates=1, confirmation_replicates=1, previous=None):
    """execute(params) -> public observation. research/choose only receive public context."""
    if goal not in ('maximize', 'minimize'):
        raise ValueError('goal must be maximize or minimize.')
    X = np.asarray(X, float)
    if (not isinstance(budget, int) or budget < 1 or not isinstance(replicates, int)
            or replicates < 1 or not isinstance(confirmation_replicates, int)
            or not 1 <= confirmation_replicates <= budget or X.size == 0 or not np.isfinite(X).all()):
        raise ValueError('Use finite candidates and positive integer budget and replicates.')
    if X.ndim != 2 or X.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError('One distinct parameter name per candidate column is required.')
    if len(np.unique(X, axis=0)) != len(X):
        raise ValueError('Candidate settings must be distinct; replicates reuse a candidate.')
    direction = 1 if goal == 'maximize' else -1
    rng = np.random.default_rng(seed)
    graph = {'nodes': [], 'edges': [], 'objective': objective, 'goal': goal, 'stop': 'running',
             'parameters': list(names), 'candidates': X.tolist()}
    observations, papers, deferred = [], [], {}
    searches, check_id = 0, None
    prior, prior_id, pending = None, None, []
    iteration, confirmation_id = 0, None
    if previous is not None:
        graph = deepcopy(previous['graph'])
        if (graph.get('parameters') != list(names) or graph.get('candidates') != X.tolist()
                or graph['goal'] != goal or graph['objective'] != objective):
            raise ValueError('Continuation must use the same candidate domain and objective.')
        observations = deepcopy(previous['observations'])
        papers = [n for n in graph['nodes'] if n['kind'] == 'source']
        old_priors = [n for n in graph['nodes'] if n['kind'] == 'prior']
        if old_priors:
            prior = old_priors[-1]
            prior_id = prior['id']
        for n in graph['nodes']:
            if n['kind'] == 'deferred':
                deferred.setdefault(n['candidate'], []).append(n['id'])
        searches = 1 if papers else 0
        iteration = previous['result'].get('optimisation_iterations', 0)
        graph['stop'] = 'running'
    offset = len(observations)
    search_budget = budget - confirmation_replicates

    def node(kind, **data):
        ident = f'N{len(graph["nodes"])}'
        graph['nodes'].append(dict(id=ident, kind=kind, **data))
        return ident

    def edge(source, target, reason):
        graph['edges'].append(dict(source=source, target=target, reason=reason))

    root = node('goal', reason=f'{goal.capitalize()} {objective} within the experiment budget.',
                uncertainty='Optimisation of an observed proxy does not verify the underlying objective. '
                'Predictive intervals are approximate; source reliability and applicability require assessment.')
    if previous is not None:
        edge(previous['graph']['nodes'][-1]['id'], root, 'Continue with all earlier measurements and evidence.')
    for step in range(budget):
        on_event('planning', graph)
        options, gp = shortlist(X, observations, rng, direction)
        context = dict(objective=objective, goal=goal, parameters=names, research_query=research_query,
                       bounds={name:[float(X[:,j].min()), float(X[:,j].max())] for j,name in enumerate(names)}, options=[dict(o, params=dict(zip(names, X[o['candidate']].tolist())))
                                                  for o in options], history=observations)
        # Search once initially, once after a surprising result. A search must affect a future decision.
        if research and searches < 2 and (searches == 0 or check_id is not None):
            on_event('research', graph)
            searches += 1
            try:
                found = await research(context)
                for p in found:
                    p = dict(p)
                    p['id'] = node('source', **{k:v for k,v in p.items() if k != 'id'})
                    papers.append(p)
                if make_prior and papers:
                    proposal = await make_prior(dict(context, papers=papers))
                    if not proposal.get('belief'):
                        prior, prior_id = None, None
                    else:
                        prior = validate_prior(proposal, names, X, papers)
                        prior_id = node('prior', **prior)
                        edge(root, prior_id, 'Tentative literature location prior, not measured data.')
                        for citation in prior['citations']:
                            edge(citation['id'], prior_id, citation['transfer_limit'])
            except Exception as error:
                nid = node('search_failure', reason=type(error).__name__)
                edge(root, nid, 'Retrieval or prior interpretation failed; continue with the last valid policy.')
        if prior is not None:
            weights = prior_weights(X, names, prior['belief'], len(observations))
            options, gp = shortlist(X, observations, rng, direction, weights=weights)
        planned_repeat = bool(pending)
        forced = pending.pop(0) if pending else check_id
        if forced is not None and forced not in {o['candidate'] for o in options}:
            options.append(dict(candidate=forced, mean=None, sd=None, ei=None, reason='Planned or diagnostic repeat.'))
        good = [o for o in observations if o['value'] is not None]
        # Reserve a final replicate group; do not let a pending search consume it.
        confirming = step >= search_budget and bool(good)
        if confirming:
            pending.clear()
            if confirmation_id is None:
                means = observed_means(good)
                confirmation_id = max(means, key=lambda i:direction*means[i])
            forced = confirmation_id
            if forced not in {o['candidate'] for o in options}:
                options.append(dict(candidate=forced, mean=None, sd=None, ei=None,
                                    reason='Final confirmation group.'))
        default = forced if forced is not None else options[0]['candidate']
        reason = ('Final confirmation of the best observed condition.' if confirming else
                  'Planned independent preparation replicate; retain every result in the mean.'
                  if planned_repeat else 'Repeat after a surprising result; response-model error and delivery error remain alternatives.'
                  if check_id is not None else 'Final confirmation of the best observed condition.'
                  if forced is not None else options[0]['reason'])
        decision = dict(selected=default, reason=reason, citations=[],
                        uncertainty='Predictive uncertainty mixes response and noise. Delivery is only estimated.',
                        alternatives={str(o['candidate']): 'Deferred under the numerical policy: '+o['reason']
                                      for o in options if o['candidate'] != default})
        context['options'] = [dict(o, params=dict(zip(names, X[o['candidate']].tolist()))) for o in options]
        if choose and forced is None:
            try:
                decision = validate_choice(await choose(dict(context, papers=papers, prior=prior)), options, papers)
            except Exception as error:
                decision['fallback'] = type(error).__name__
        selected = decision['selected']
        if forced is None:
            iteration += 1
            pending = [selected] * max(0, min(replicates - 1, search_budget - step - 1))
        phase = 'confirmation' if confirming else 'replicate' if planned_repeat else 'diagnostic' if forced is not None else 'proposal'
        d = node('decision', round=offset+step+1, iteration=iteration, phase=phase, **decision, options=context['options'])
        edge(root if not observations else observations[-1]['id'], d, 'Evidence available before selection.')
        if prior_id is not None:
            edge(prior_id, d, 'Bounded prior informs acquisition; raw EI and uncertainty remain eligible.')
        for deferred_id in deferred.pop(selected, []):
            edge(deferred_id, d, 'Reconsidered and selected; the earlier deferral was not a refutation.')
        branch_nodes = {selected:d}
        for i, why in decision['alternatives'].items():
            a = node('deferred', candidate=int(i), reason=why, status='Reconsiderable; not falsified.')
            deferred.setdefault(int(i), []).append(a)
            branch_nodes[int(i)] = a
            edge(d, a, 'Not selected in this round.')
        for citation in decision['citations']:
            edge(citation['id'], branch_nodes[citation.get('candidate', selected)],
                 'Literature interpretation: '+citation['transfer_limit'])
        for paper in papers:
            if paper['id'] not in {c['id'] for c in decision['citations']}:
                edge(paper['id'], d, 'Retrieved but not cited as a selection reason.')
        # Decision is committed before execution. No result can rewrite its rationale.
        on_event('decision', graph)
        observation = dict(execute(dict(zip(names, X[selected].tolist()))))
        if observation.get('ok', True):
            value = observation.get('value')
            reported = np.asarray(observation.get('reported'), float)
            report_sd = np.asarray(observation.get('report_sd', np.zeros(X.shape[1])), float)
            if (value is None or not np.isscalar(value) or not np.isfinite(value)
                    or reported.shape != (X.shape[1],) or not np.isfinite(reported).all()
                    or report_sd.shape != reported.shape or not np.isfinite(report_sd).all()
                    or (report_sd < 0).any()):
                raise ValueError('Successful execution requires finite reward, reported inputs and nonnegative SDs.')
        else:
            observation['value'] = None
        observation.update(candidate=selected)
        oid = node('observation', round=offset+step+1, **observation)
        observation['id'] = oid
        edge(d, oid, 'Executed this decision.')
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
    values = [o['value'] for o in observations if o['candidate'] == best and o['value'] is not None]
    sample_sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
    result = dict(candidate=best, sample_sd=sample_sd,
                  optimisation_iterations=iteration, distinct_conditions=len(means),
                  confirmation_candidate=confirmation_id, confirmation_replicates=confirmation_replicates,
                  standard_error=None if sample_sd is None else sample_sd / np.sqrt(len(values)),
                  requested_replicates=replicates,
                  replication_complete=len(values) >= replicates,
                  params=None if best is None else dict(zip(names, X[best].tolist())),
                  mean=None if best is None else means[best],
                  repeats=sum(o['candidate']==best and o['value'] is not None for o in observations),
                  rule=f'{goal.capitalize()} mean observed proxy by intended parameters, including repeats.',
                  uncertainty='Best observed proxy, not a verified objective or proven optimum. '
                  'Scatter and standard error assume independent preparations; they exclude systematic bias '
                  'and selection uncertainty. One confirmation cannot establish reproducibility.')
    final = node('recommendation', **result)
    for o in observations:
        if o['candidate'] == best:
            edge(o['id'], final, 'Observed result used in the protocol mean.')
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
