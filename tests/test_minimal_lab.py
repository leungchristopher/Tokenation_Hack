import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from minimal_lab.lab import Lab
from minimal_lab.loop import run, save, validate_choice


class Backend:
    def __init__(self, fail=False):
        self.fail = fail
        self.wells = ['well_B3', 'well_B4']
        self.contract = SimpleNamespace(reagents={p:'reagent_'+p for p in
            ('nacl','pnpp','glycerol','phosphate','enzyme','water')})
        self.skills = SimpleNamespace(tip_status=lambda:dict(tips_remaining=24))

    def change_tip(self):
        return dict(ok=True)

    def pipette(self, source, dest):
        return dict(ok=not self.fail, reason='injected failure', moves=[])

    def mix(self, well, cycles):
        return dict(ok=True)


class Table:
    params = ['ph','salt_conc','cosubstrate_conc','organic_solvent_conc','temperature']
    X = np.array([[6,1,1,1,20],[7,10,10,10,30]],float)

    def index(self, params):
        self.queried = params
        return 0

    def sample(self, index, rng):
        return 42.0


def test_realised_volumes_feed_black_box_but_not_policy():
    env = Table()
    params = dict(zip(env.params,env.X[1]))
    lab = Lab(env, seed=4, cv=0.8, backend=Backend())
    result = lab(params)
    assert result['value'] == 42
    assert env.queried != params
    assert env.queried == lab.hidden[0]['realised']
    assert 'mapped_candidate' not in json.dumps(result)
    assert 'realised' not in result
    assert len(result['report_sd']) == 5


def test_failed_motion_never_queries_response_table():
    env = Table()
    result = Lab(env, backend=Backend(fail=True))(dict(zip(env.params,env.X[0])))
    assert result['value'] is None and not result['ok']
    assert not hasattr(env,'queried')


def test_zero_volume_error_preserves_requested_concentrations():
    env = Table()
    params = dict(zip(env.params,env.X[1]))
    result = Lab(env, cv=0, report_cv=0, backend=Backend())(params)
    assert result['reported'] == list(params.values())


def test_literature_changes_selection_with_pre_execution_reasons(tmp_path):
    X = np.array([[0],[1],[2],[3]],float)
    calls = []
    async def research(context):
        return [dict(title='Test source',abstract='A reported effect.',url='https://example.org')]
    async def choose(context):
        assert len(calls) == len(context['history'])
        selected = context['options'][-1]['candidate']
        return dict(selected=selected, reason='Test the reported effect in this assay.',
                    alternatives={str(o['candidate']):'Lower priority under this hypothesis.'
                                  for o in context['options'] if o['candidate'] != selected},
                    citations=[dict(id=context['papers'][0]['id'],quote='reported effect',
                                    transfer_limit='Different assay; applicability untested.')],
                    uncertainty='Transfer remains uncertain.')
    def execute(params):
        calls.append(params)
        return dict(value=float(params['x']),reported=[params['x']],report_sd=[0.0],ok=True)
    episode = asyncio.run(run(X,['x'],execute,budget=3,research=research,choose=choose))
    decisions = [n for n in episode['graph']['nodes'] if n['kind']=='decision']
    assert decisions[0]['selected'] == decisions[0]['options'][-1]['candidate']
    assert decisions[0]['selected'] != decisions[0]['options'][0]['candidate']
    assert decisions[0]['citations']
    assert decisions[-1]['reason'].startswith('Final confirmation')
    assert episode['result']['repeats'] >= 2
    save(episode,tmp_path/'run')
    assert (tmp_path/'run'/'graph.html').exists()
    assert 'GRAPH_DATA' not in (tmp_path/'run'/'graph.html').read_text()


def test_fake_citations_and_missing_alternative_reasons_rejected():
    options=[dict(candidate=0),dict(candidate=1)]
    reply=dict(selected=0,reason='Why',alternatives={'1':'Later'},uncertainty='Unknown',
               citations=[dict(id='S1',quote='fabricated',transfer_limit='Unknown')])
    with pytest.raises(ValueError,match='quote'):
        validate_choice(reply,options,[dict(id='S1',abstract='Actual abstract')])
    reply.update(citations=[], alternatives={})
    with pytest.raises(ValueError,match='Every deferred'):
        validate_choice(reply,options,[])


def test_failure_stops_loop_and_preserves_failed_decision():
    episode=asyncio.run(run(np.array([[0],[1]]),['x'],
        lambda p:dict(value=None,reported=None,ok=False,reason='Motion failed'),budget=4))
    assert len(episode['observations']) == 1
    assert episode['result']['candidate'] is None
    assert episode['graph']['stop'].startswith('Execution failed')


def test_minimise_proxy_without_oracle():
    episode=asyncio.run(run(np.array([[0],[1],[2]]),['x'],
        lambda p:dict(value=p['x'],reported=[p['x']],ok=True),budget=5,goal='minimize',
        objective='unverified reviewer loss'))
    assert episode['result']['params'] == {'x':0.0}
    assert 'not a verified objective' in episode['result']['uncertainty']


def test_measured_non_incumbent_can_be_reconsidered(monkeypatch):
    from minimal_lab import loop
    class GP:
        def __init__(self, **kwargs): pass
        def fit(self, X, y): return self
        def predict(self, X, return_std=False):
            mu, sd = np.array([0., 2., 0.]), np.array([1., 10., 1.])
            return (mu, sd) if return_std else mu
    monkeypatch.setattr(loop, 'GaussianProcessRegressor', GP)
    observations = [dict(candidate=i, value=v, reported=[float(i)], report_sd=[0.])
                    for i,v in enumerate([10., 1., 2.])]
    options, _ = loop.shortlist(np.array([[0.],[1.],[2.]]), observations, np.random.default_rng(0))
    assert {o['candidate'] for o in options} >= {0, 1}


@pytest.mark.parametrize('goal,sign', [('maximize',1), ('minimize',-1)])
def test_deferral_revisited_and_confirmation_uses_mean(monkeypatch, goal, sign):
    from minimal_lab import loop
    options = [dict(candidate=i, mean=None, sd=None, ei=None, reason='Test option') for i in (0,1)]
    monkeypatch.setattr(loop, 'shortlist', lambda *args: (options.copy(), None))
    selections, results = iter([0,0,1]), iter([10.,2.,8.,8.])
    async def choose(context):
        pick = next(selections)
        return dict(selected=pick,reason='Test selection',uncertainty='Uncertain response',
                    alternatives={str(1-pick):'Deferred until further measurements.'},citations=[])
    def execute(params):
        return dict(value=sign*next(results),reported=[params['x']],ok=True)
    episode = asyncio.run(loop.run(np.array([[0.],[1.]]), ['x'], execute,
                                   budget=4, goal=goal, choose=choose))
    decisions = [n for n in episode['graph']['nodes'] if n['kind']=='decision']
    assert decisions[-1]['selected'] == 1  # mean 8 beats mean 6, despite a single reading of 10
    assert episode['result']['candidate'] == 1
    assert episode['result']['repeats'] == 2
    edges = episode['graph']['edges']
    assert any(e['target']==decisions[2]['id'] and e['reason'].startswith('Reconsidered') for e in edges)
    assert len(episode['observations']) == 4  # no prior observation was erased by a deferral


def test_planned_replicates_consume_budget_and_report_scatter():
    values = iter([1., 3., 5.])
    episode = asyncio.run(run(np.array([[0.]]), ['x'],
        lambda p: dict(value=next(values), reported=[0.], ok=True), budget=3, replicates=3))
    assert len(episode['observations']) == 3
    assert episode['result']['mean'] == 3.
    assert episode['result']['sample_sd'] == 2.
    assert episode['result']['replication_complete']
    decisions = [n for n in episode['graph']['nodes'] if n['kind'] == 'decision']
    assert decisions[1]['reason'].startswith('Planned independent')
    assert decisions[2]['phase'] == 'confirmation'


def test_cited_prior_reaches_candidates_outside_random_shortlist():
    from minimal_lab.evidence import prior_weights
    X = np.arange(20, dtype=float).reshape(-1, 1)
    async def research(context):
        return [dict(title='Source', abstract='Higher settings improved response.', url='https://example.org')]
    async def prior(context):
        return dict(belief={'x': [19., .1]}, reason='Tentative transfer', uncertainty='Different system',
                    citations=[dict(id=context['papers'][0]['id'], quote='Higher settings',
                                    parameters=['x'], transfer_limit='Different system')])
    episode = asyncio.run(run(X, ['x'],
        lambda p: dict(value=p['x'], reported=[p['x']], ok=True),
        budget=2, research=research, make_prior=prior))
    assert episode['observations'][0]['candidate'] == 19
    assert any(n['kind'] == 'prior' for n in episode['graph']['nodes'])
    weights = prior_weights(X, ['x'], {'x': [19., .1]}, 0)
    assert weights.min() >= .25 and weights.max() <= 1
    assert prior_weights(X, ['x'], {'x': [19., .1]}, 100).min() > weights.min()


def test_failed_result_is_never_included_in_recommendation():
    episode = asyncio.run(run(np.array([[0.]]), ['x'],
        lambda p: dict(value=999., reported=None, ok=False), budget=1))
    assert episode['result']['candidate'] is None
    assert episode['observations'][0]['value'] is None


def test_fenced_model_json_is_decoded():
    from minimal_lab.model_json import decode_reply
    assert decode_reply('```json\n{"selected": 2}\n```') == {"selected": 2}


def test_iteration_count_excludes_replicates_and_confirmation():
    episode = asyncio.run(run(np.arange(6.).reshape(-1, 1), ['x'],
        lambda p: dict(value=p['x'], reported=[p['x']], ok=True),
        budget=8, replicates=2, confirmation_replicates=2))
    decisions = [n for n in episode['graph']['nodes'] if n['kind'] == 'decision']
    assert [n['phase'] for n in decisions] == ['proposal', 'replicate'] * 3 + ['confirmation'] * 2
    assert episode['result']['optimisation_iterations'] == 3
    assert len(episode['observations']) == 8
