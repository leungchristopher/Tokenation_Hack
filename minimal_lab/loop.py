"""One loop. The policy receives candidate settings and observations, never unseen labels."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from minimal_lab.evidence import prior_weights, validate_prior
from minimal_lab.policy import shortlist, statistics
from minimal_lab.context import graph_context


def observed_means(observations):
    grouped = {}
    for o in observations:
        if o.get('ok', True) and o['value'] is not None:
            grouped.setdefault(o['candidate'], []).append(o['value'])
    return {i:float(np.mean(values)) for i,values in grouped.items()}


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
              make_prior=None, research_query=None, replicates=1, confirmation_replicates=1, previous=None, log_parameters=(), finalist_count=1, max_choice_calls=4, explain=None, require_prior=False):
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
    if set(log_parameters) - set(names):
        raise ValueError('Unknown log-scaled parameter.')
    log_axes = tuple(names.index(p) for p in log_parameters)
    direction = 1 if goal == 'maximize' else -1
    rng = np.random.default_rng(seed)
    graph = {'nodes': [], 'edges': [], 'objective': objective, 'goal': goal, 'stop': 'running',
             'parameters': list(names), 'candidates': X.tolist()}
    observations, papers, deferred = [], [], {}
    searches, check_id = 0, None
    prior, prior_id, pending = None, None, []
    iteration, confirmation_id = 0, None
    finalists, choice_calls = [], 0
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
    graph['log_parameters'] = list(log_parameters)
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
        options, gp = shortlist(X, observations, rng, direction, log_axes=log_axes)
        context = dict(objective=objective, goal=goal, parameters=names, research_query=research_query,
                       log_parameters=list(log_parameters),
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
                    p['id'] = node('source', **{k:v for k,v in p.items() if k not in ('id','kind')})
                    papers.append(p)
                    edge(root, p['id'], 'Retrieved literature; applicability must be assessed.')
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
        if require_prior and not observations and prior is None:
            raise ValueError('Literature mode requires a validated prior before the first experiment.')
        if prior is not None:
            weights = prior_weights(X, names, prior['belief'], len(observations), log_axes=log_axes)
            options, gp = shortlist(X, observations, rng, direction, weights=weights, log_axes=log_axes)
        planned_repeat = bool(pending)
        forced = pending.pop(0) if pending else check_id
        if forced is not None and forced not in {o['candidate'] for o in options}:
            options.append(dict(candidate=forced, mean=None, sd=None, ei=None, reason='Planned or diagnostic repeat.'))
        good = [o for o in observations if o['value'] is not None]
        # Reserve a final replicate group; do not let a pending search consume it.
        confirming = step >= search_budget and bool(good)
        if confirming:
            pending.clear()
            if not finalists:
                means = observed_means(good)
                rank = (lambda i: direction*gp.mean[i]) if gp is not None else (lambda i: direction*means[i])
                finalists = sorted(means, key=rank, reverse=True)[:max(1, finalist_count)]
                confirmation_id = finalists[0]
            forced = finalists[(step-search_budget) % len(finalists)]
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
        if choose and forced is None and (observations or prior is None) and choice_calls < max_choice_calls:
            try:
                choice_calls += 1
                decision = validate_choice(await choose(dict(context, papers=papers, prior=prior)), options, papers)
            except Exception as error:
                decision['fallback'] = type(error).__name__
        selected = decision['selected']
        if forced is None:
            iteration += 1
            pending = [selected] * max(0, min(replicates - 1, search_budget - step - 1))
        phase = 'confirmation' if confirming else 'replicate' if planned_repeat else 'diagnostic' if forced is not None else 'proposal'
        selected_option = next(o for o in options if o['candidate'] == selected)
        source_ids = sorted({c['id'] for c in prior['citations']}) if prior else []
        decision.update(source_ids=source_ids, prior_id=prior_id,
                        predicted_mean=selected_option.get('mean'), latent_sd=selected_option.get('sd'),
                        uncertainty_kind='Uncalibrated model SD for the mean recipe response; not replicate scatter.')
        decision['policy_reason'] = decision['reason']
        if explain:
            try:
                cited_sources = [dict(id=p['id'], title=p.get('title'),
                    excerpts=[c['quote'] for c in (prior or {}).get('citations',[]) if c['id']==p['id']])
                    for p in papers if p['id'] in source_ids]
                note = await explain(dict(objective=objective, goal=goal, phase=phase,
                    iteration=iteration, params=dict(zip(names,X[selected].tolist())),
                    proposal=selected_option, policy_reason=decision['policy_reason'],
                    recent=[dict(candidate=o['candidate'],value=o['value']) for o in observations[-6:]],
                    candidate_statistics=statistics(observations).get(selected), sources=cited_sources,
                    graph=graph_context(graph, observations, selected)))
                if (not isinstance(note.get('reason'),str) or not note['reason'].strip()
                        or not isinstance(note.get('uncertainty'),str) or not note['uncertainty'].strip()
                        or not set(note.get('source_ids',[])).issubset(source_ids)):
                    raise ValueError('Rationale requires grounded source IDs and explicit uncertainty.')
                decision.update(reason=note['reason'], uncertainty=note['uncertainty'],
                                rationale_source_ids=note.get('source_ids',[]), rationale_provider='LLM pre-execution interpretation')
            except Exception as error:
                decision['rationale_failure'] = type(error).__name__
                decision['rationale_provider'] = 'Numerical policy fallback'
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
        for sid in decision.get('rationale_source_ids', []):
            edge(sid, d, 'Cited in the pre-execution LLM interpretation; transfer remains uncertain.')
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
            residual = abs(observation['value']-gp.mean[selected])
            predictive_sd = float(np.sqrt(gp.sd[selected]**2+gp.noise))
            if residual > 2*predictive_sd:
                # Already planned preparation repeats serve as the diagnostic.
                check_id = selected if not pending else None
                graph['nodes'][int(oid[1:])]['surprise'] = 'Outside approximate predictive band; check noise/model mismatch.'
        summary = statistics(observations).get(selected)
        if summary:
            graph['nodes'][int(oid[1:])]['replicate_summary'] = summary
        on_event('observation', graph)
        if not observation.get('ok', True):
            graph['stop'] = 'Execution failed; no reward fabricated.'
            break
    if graph['stop'] == 'running':
        graph['stop'] = 'budget exhausted'
    means = observed_means(observations)
    eligible = finalists or list(means)
    best = max(eligible, key=lambda i:direction*means[i]) if eligible else None
    values = [o['value'] for o in observations if o['candidate'] == best and o['value'] is not None]
    sample_sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
    result = dict(candidate=best, sample_sd=sample_sd,
                  optimisation_iterations=iteration, distinct_conditions=len(means),
                  confirmation_candidate=confirmation_id, confirmation_replicates=confirmation_replicates,
                  finalists=finalists, choice_model_calls=choice_calls,
                  finalist_statistics={i: statistics(observations)[i] for i in finalists},
                  rationale_calls=sum(n['kind']=='decision' and n.get('rationale_provider')=='LLM pre-execution interpretation' for n in graph['nodes']),
                  standard_error=None if sample_sd is None else sample_sd / np.sqrt(len(values)),
                  requested_replicates=replicates,
                  replication_complete=len(values) >= replicates,
                  params=None if best is None else dict(zip(names, X[best].tolist())),
                  mean=None if best is None else means[best],
                  repeats=sum(o['candidate']==best and o['value'] is not None for o in observations),
                  rule=f'{goal.capitalize()} observed mean among confirmed finalists, including every preparation.',
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
