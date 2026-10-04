"""Paired-seed comparison: numerical BO policy vs Modal open-model chooser, equal budgets.

  python -m minimal_lab.modal_experiment --seeds 0 1 2 3 4 --budget 12 --out logs/modal.json
  python -m minimal_lab.modal_experiment --offline   # scripted replies: contract/fallback only

Robot motion is stubbed (always succeeds); Lab still applies delivery, report and assay noise.
The chooser sees only public context; hidden table means are used for regret scoring only.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bo_eval.env import get_env
from minimal_lab.lab import Lab
from minimal_lab.loop import run
from minimal_lab.task import regret, text_chooser

L40S_USD_PER_S = 0.000542  # https://modal.com/pricing


class Backend:
    def __init__(self, wells=96):
        self.wells = [f'well_{i}' for i in range(wells)]
        self.contract = SimpleNamespace(reagents={p:'reagent_'+p for p in
            ('nacl','pnpp','glycerol','phosphate','enzyme','water')})
        self.skills = SimpleNamespace(tip_status=lambda: dict(tips_remaining=96))

    def change_tip(self):
        return dict(ok=True)

    def pipette(self, source, dest):
        return dict(ok=True, moves=[])

    def mix(self, well, cycles):
        return dict(ok=True)


async def episode(seed, budget, cv, pick=None):
    env, calls = get_env('upo_abts'), []

    async def counted(context):
        calls.append(len(context['history']))
        return await pick(context)
    result = await run(env.X.copy(), env.params, Lab(env, seed=seed, cv=cv, backend=Backend()),
                       budget=budget, seed=seed, objective='simulated UPO-ABTS assay response',
                       choose=counted if pick else None)
    decisions = [n for n in result['graph']['nodes'] if n['kind'] == 'decision']
    asked = [d for d in decisions if d['round']-1 in calls]
    loss = regret(env, result['result']['params'])
    return dict(seed=seed, regret=loss, found_optimal=loss == 0, n_experiments=len(result['observations']),
                best_observed=result['result']['mean'], chooser_calls=len(calls),
                fallbacks=[d['fallback'] for d in asked if 'fallback' in d],
                model_picks=[next(o['reason'] for o in d['options'] if o['candidate'] == d['selected'])
                             for d in asked if 'fallback' not in d],
                agrees_with_numerical=sum(d['selected'] == d['options'][0]['candidate']
                                          for d in asked if 'fallback' not in d),
                cited=sum(bool(d['citations']) for d in asked))


def scripted():
    """Offline replies: valid non-default pick, hallucinated citation, then prose (no JSON)."""
    count = []

    async def generate(prompt):
        context = json.loads(prompt[prompt.index('{"objective"'):])
        count.append(1)
        options = [o['candidate'] for o in context['options']]
        pick = options[-1]
        reply = dict(selected=pick, reason='Offline scripted pick.', uncertainty='Scripted.',
                     alternatives={str(i):'Deferred by script.' for i in options if i != pick}, citations=[])
        if len(count) % 3 == 2:
            reply['citations'] = [dict(id='N999', quote='invented', transfer_limit='none')]
        text = 'No JSON here.' if len(count) % 3 == 0 else json.dumps(reply)
        return dict(text=text, prompt_tokens=None, completion_tokens=None, latency_s=0.0)
    return generate


def summary(base, model, calls, meta):
    asked = sum(r['chooser_calls'] for r in model)
    fallbacks = [f for r in model for f in r['fallbacks']]
    diff = [m['regret']-b['regret'] for b, m in zip(base, model)]
    lat = [c['latency_s'] for c in calls if c.get('latency_s')]
    return dict(meta, seeds=[r['seed'] for r in base],
                numerical_mean_regret=float(np.mean([r['regret'] for r in base])),
                model_mean_regret=float(np.mean([r['regret'] for r in model])),
                paired_regret_diff_model_minus_numerical=diff,
                model_better_tie_worse=[sum(d < 0 for d in diff), sum(d == 0 for d in diff), sum(d > 0 for d in diff)],
                numerical_found_optimal=sum(r['found_optimal'] for r in base),
                model_found_optimal=sum(r['found_optimal'] for r in model),
                chooser_calls=asked, valid_choices=asked-len(fallbacks),
                valid_rate=(asked-len(fallbacks))/asked if asked else None,
                fallback_rate=len(fallbacks)/asked if asked else None,
                fallback_types={f:fallbacks.count(f) for f in set(fallbacks)},
                prompt_tokens=sum(c.get('prompt_tokens') or 0 for c in calls),
                completion_tokens=sum(c.get('completion_tokens') or 0 for c in calls),
                latency_s=dict(mean=float(np.mean(lat)), p50=float(np.median(lat)), max=float(max(lat))) if lat else None,
                numerical=base, model=model)


async def main(args):
    base = [await episode(s, args.budget, args.cv) for s in args.seeds]
    if args.offline:
        model = [await episode(s, args.budget, args.cv, text_chooser(scripted())) for s in args.seeds]
        return summary(base, model, [], dict(mode='OFFLINE scripted replies; no Modal, no GPU, no model'))
    from minimal_lab import modal_model
    start = time.monotonic()
    async with modal_model.session() as client:
        warm = time.monotonic()-start
        model = [await episode(s, args.budget, args.cv, text_chooser(client)) for s in args.seeds]
    wall = time.monotonic()-start
    return summary(base, model, client.calls, dict(
        mode='LIVE Modal', model=modal_model.MODEL, revision=modal_model.REVISION, gpu=modal_model.GPU,
        app_id=getattr(client, 'app_id', None), startup_s=warm, session_wall_s=wall,
        est_gpu_cost_usd=wall*L40S_USD_PER_S, cost_note='Upper-bound estimate: session wall time x L40S '
        'list price; excludes image build and scaledown, not a billing record.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--budget', type=int, default=12)
    parser.add_argument('--cv', type=float, default=0.15)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--out')
    args = parser.parse_args()
    report = asyncio.run(main(args))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k:v for k,v in report.items() if k not in ('numerical', 'model')}, indent=2, default=float))
