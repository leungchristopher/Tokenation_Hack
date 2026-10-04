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
