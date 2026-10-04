"""Assay recipes translate concentrations to simulated robot transfers."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CancerRecipe:
    name: str = 'A549: taxol + cisplatin + doxorubicin'

    @property
    def slots(self):
        return {'taxol_uM': 'nacl', 'cisplatin_uM': 'pnpp', 'doxorubicin_uM': 'glycerol'}

    def protocol(self, params, maxima, contract):
        transfers = [dict(role=drug, source=contract.reagents[slot],
                          stock_uM=5*maxima[drug], target_uM=params[drug],
                          volume_ul=20*params[drug]/maxima[drug])
                     for drug, slot in self.slots.items()]
        transfers.append(dict(role='cell culture medium', source=contract.reagents['water'],
                              volume_ul=100-sum(t['volume_ul'] for t in transfers)))
        return dict(assay=self.name, transfers=transfers, final_volume_ul=100,
                    cells='A549 cells assumed pre-seeded in each fresh well',
                    endpoint='Cell survival percent; lower is better',
                    assumptions='Simulation recipe: three 100 uM stocks, medium top-up to 100 uL. '
                    'Scene source slots are reassigned to the named drugs. Stock recipes, solvents, '
                    'cell seeding, incubation and survival readout are not physically simulated. '
                    'Response comes from the Zimmer three-drug table; SD is a neighbour-residual '
                    'estimate, not measured replicate scatter.')


def recipe_for(env):
    return CancerRecipe() if env.name == 'zimmer_a549' else None
