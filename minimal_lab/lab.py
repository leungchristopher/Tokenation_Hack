"""Current upstream robot -> explicit simulated volume error -> dataset black box.

MuJoCo simulates motion, NOT liquid. Stock conversions below are dimensionless demo
assumptions because the CSV does not document concentration units. Never a wet-lab SOP.
"""
from dataclasses import asdict

import numpy as np


class Lab:
    def __init__(self, env, seed=0, cv=0.15, report_cv=0.05, backend=None):
        if not np.isfinite([cv, report_cv]).all() or min(cv, report_cv) < 0:
            raise ValueError('Volume error scales must be finite and nonnegative.')
        if backend is None:
            from harness.tools.lab_backend import LabBackend
            backend = LabBackend(seed=seed)
        self.backend, self.env = backend, env
        self.cv, self.report_cv = cv, report_cv
        self.delivery_rng, self.report_rng, self.assay_rng = [np.random.default_rng(s)
            for s in np.random.SeedSequence(seed).spawn(3)]
        self.hidden = []
        self.count = 0
        # Physical source slots only. Scene labels are NOT the chemical identity of a UPO assay.
        self.slots = {'salt_conc': 'nacl', 'cosubstrate_conc': 'pnpp',
                      'organic_solvent_conc': 'glycerol'}
        self.maxima = dict(zip(env.params, env.X.max(0)))

    def protocol(self, params):
        transfers = [dict(role=p, source=self.backend.contract.reagents[slot],
                          volume_ul=20*params[p]/self.maxima[p]) for p,slot in self.slots.items()]
        transfers += [dict(role='buffer', source=self.backend.contract.reagents['phosphate'], volume_ul=20),
                      dict(role='enzyme', source=self.backend.contract.reagents['enzyme'], volume_ul=10)]
        transfers.append(dict(role='water', source=self.backend.contract.reagents['water'],
                              volume_ul=100-sum(t['volume_ul'] for t in transfers)))
        return dict(transfers=[t for t in transfers if t['volume_ul'] > 0],
                    final_volume_ul=100, ideal_external_settings={p:params[p] for p in ('ph','temperature')},
                    assumptions='Simulation only: stock level = 5 × maximum tabulated concentration. '
                    'Source names identify scene slots, not assay chemicals. Buffer pH and temperature '
                    'are ideal external settings, not robot-controlled. Enzyme delivery has no independent '
                    'axis in the table. No liquid physics or calibrated dispenser model.')

    def __call__(self, params):
        plan = self.protocol(params)
        if self.count >= len(self.backend.wells):
            return dict(value=None, ok=False, reason='No unused wells remain.', reported=None)
        well = self.backend.wells[self.count]
        self.count += 1
        motion, volumes = [], {}
        for transfer in plan['transfers']:
            if self.backend.skills.tip_status()['tips_remaining'] == 0:
                # Automatic stock replacement is an upstream abstraction, not robot motion.
                self.backend.refresh_tips()
                motion.append(dict(action='tip_box_refresh', simulated_service=True))
            changed = self.backend.change_tip()
            if not changed['ok']:
                return dict(value=None, ok=False, reason=changed['reason'], reported=None, motion=motion)
            out = self.backend.pipette(transfer['source'], well)
            moves = [dict(action=a, site=s, **asdict(r)) for a,s,r in out.get('moves', [])]
            motion.append(dict(role=transfer['role'], moves=moves))
            if not out['ok']:
                return dict(value=None, ok=False, reason=out['reason'], reported=None, motion=motion)
            # Mean-one lognormal volume error, CV parameterised exactly. Independent of motion.
            sigma = np.sqrt(np.log1p(self.cv**2))
            volumes[transfer['role']] = transfer['volume_ul']*self.delivery_rng.lognormal(-sigma*sigma/2, sigma)
        mixed = self.backend.mix(well, cycles=2)
        motion.append(dict(action='mix', **mixed))
        if not mixed['ok']:
            return dict(value=None, ok=False, reason=mixed['reason'], reported=None, motion=motion)
        total = sum(volumes.values())
        realised = dict(params)
        for p in self.slots:
            realised[p] = 5*self.maxima[p]*volumes.get(p, 0)/total
        # A hypothetical noisy delivery sensor, NOT MuJoCo telemetry. Reports continuous values
        # before nearest-grid mapping, never the hidden candidate or hidden table mean.
        reported, report_sd = dict(realised), dict.fromkeys(params, 0.0)
        for p in self.slots:
            report_sd[p] = abs(realised[p])*self.report_cv
            reported[p] = max(0.0, self.report_rng.normal(realised[p], report_sd[p]))
        index = self.env.index(realised)
        value = self.env.sample(index, self.assay_rng)
        self.hidden.append(dict(realised=realised, mapped_candidate=index, volumes=volumes))
        return dict(value=value, ok=True, intended=params,
                    reported=[reported[p] for p in self.env.params],
                    report_sd=[report_sd[p] for p in self.env.params],
                    execution=dict(model='MuJoCo motion + assumed lognormal liquid-volume error',
                                   cv=self.cv, report_cv=self.report_cv, well=well),
                    protocol=plan, motion=motion,
                    uncertainty='Delivery estimates are simulated sensor reports. Grid snapping can change '
                    'several parameters in this sparse table. pH/temperature actuation is not simulated.')
