"""Bounded graph context for step reasoning; only public observations and provenance."""
from minimal_lab.policy import statistics


def graph_context(graph, observations, selected, recent_decisions=4):
    decisions = [n for n in graph['nodes'] if n['kind']=='decision'][-recent_decisions:]
    # Include earlier decisions about the selected recipe, so revisits retain history.
    related = [n for n in graph['nodes'] if n['kind']=='decision' and n['selected']==selected][-2:]
    relevant = {n['id']: n for n in decisions+related}
    deferred = [n for n in graph['nodes'] if n['kind']=='deferred' and n['candidate']==selected][-2:]
    relevant.update({n['id']:n for n in deferred})
    nodes = [dict(id=n['id'],kind=n['kind'],candidate=n.get('selected',n.get('candidate')),
                  reason=n.get('reason','')[:300],uncertainty=n.get('uncertainty','')[:180],
                  source_ids=n.get('source_ids',[]),alternatives=n.get('alternatives',{}))
             for n in relevant.values()]
    edges = [e for e in graph['edges'] if e['target'] in relevant or e['source'] in relevant]
    # No motion arrays, hidden realised doses, oracle labels or full abstracts.
    return dict(objective=graph['objective'],goal=graph['goal'],nodes=nodes,edges=edges[-16:],
                observed_conditions=statistics(observations),
                scope='Recent reasoning plus selected-recipe history; all measured-condition summaries. '
                      'Absence from this compact view is not branch closure.')
