"""Build icfree_*.csv from the Zenodo AL data (doi:10.5281/zenodo.14904992).

Echo transfer volumes (nL) are converted to final concentrations with Table S1 of
Borkowski et al., iScience 2025 (doi:10.1016/j.isci.2025.113599): each stock is fixed,
so conc = max_conc * volume / max_volume. Replicate yields are pooled per condition.

    python data/icfree/convert.py <unzipped zenodo dir>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# system: (learner files, {volume column: (name, max conc, max volume nL)})
SYSTEMS = {
    "cole1_pro": (
        "ColE1 in proCFPS/4. LEARNER/top50/plate[0-4].csv",
        {
            "Mg-glutamate": ("mg_glutamate_mM", 1.25, 80),
            "K-glutamate": ("k_glutamate_mM", 275, 980),
            "Amino acid": ("amino_acids_mM", 1.875, 1100),
            "Spermidine": ("spermidine_mM", 1.25, 140),
            "Creatine Phosphate": ("creatine_phosphate_mM", 150, 940),
            "NTP": ("ntps_mM", 1.875, 140),
            "GFP11_Col-E1": ("gfp11_cole1_nM", 5, 860),
            "GFP1-10": ("gfp1_10_nM", 5, 1240),
            "PEG-8000": ("peg8000_pct", 2.5, 1320),
        },
    ),
    "colm_pro": (
        "ColM in proCFPS/4. LEARNER/top50/[Pp]late[0-4]*.csv",
        {
            "Mg-glutamate": ("mg_glutamate_mM", 2.5, 131.3),
            "K-glutamate": ("k_glutamate_mM", 100, 350),
            "Amino acid": ("amino_acids_mM", 1.875, 1094),
            "Spermidine": ("spermidine_mM", 1.25, 131.3),
            "3-PGA": ("pga_mM", 37.5, 281.3),
            "NTP": ("ntps_mM", 1.875, 126.2),
            "DNA 1": ("gfp11_colm_nM", 6, 1434),
            "DNA 2": ("gfp1_10_nM", 6, 1485),
            "PEG-8000": ("peg8000_pct", 2.5, 1312),
        },
    ),
    "colm_eu": (
        "ColM in euCFPS/4. LEARNER/top50/plate*.csv",
        {
            "Hela lysate": ("hela_lysate_X", 0.925, 940),
            "Access prot": ("accessory_proteins_X", 0.925, 200),
            "Reaction mix": ("reaction_mix_X", 0.925, 380),
            "DNA ColM": ("hibit_colm_X", 0.925, 480),
        },
    ),
}


def convert(root: Path, pattern: str, comps: dict) -> pd.DataFrame:
    files = sorted(f for f in root.glob(f"*/{pattern}") if "__MACOSX" not in str(f))
    rows = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        y = df.filter(regex=r"^Yield").to_numpy(float)
        X = pd.DataFrame({n: (df[c] * mc / mv).round(4) for c, (n, mc, mv) in comps.items()})
        for x, ys in zip(X.itertuples(index=False), y):
            rows += [(*x, v) for v in ys if np.isfinite(v)]
    params = [n for n, _, _ in comps.values()]
    long = pd.DataFrame(rows, columns=[*params, "yield"])
    out = long.groupby(params)["yield"].agg(yield_mean="mean", yield_sd="std", n="count").reset_index()
    return out[out.n > 1]


if __name__ == "__main__":
    root = Path(sys.argv[1])
    for name, (pattern, comps) in SYSTEMS.items():
        df = convert(root, pattern, comps)
        df.to_csv(Path(__file__).parent / f"{name}.csv", index=False)
        print(name, df.shape, "best:", df.loc[df.yield_mean.idxmax()].to_dict())
