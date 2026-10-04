"""A549 survival for taxol x cisplatin x doxorubicin (Zimmer et al., PNAS 2016), 8 doses each.

Source: TaxolCisDox20141211 in DatasetsFromEColiDBFolder.mat from the S2 ZIP of Tendler et al.,
PLoS Comput Biol 2019 (doi:10.1371/journal.pcbi.1006956.s002). Columns are s1, s2, s3, s12, s13, s23, s123
(% survival); rows are a full 8x8x8 grid with dose index 0 = 20 uM, descending by 3-fold dilutions
(Zimmer methods: 20 uM, 10 uM, 3.3 uM, 1.1 uM, 370 nM, 123 nM, 41 nM, 13.7 nM).
There are no replicates, so the SD is a constant estimated from residuals against grid neighbours (MAD).

    python data/zimmer/convert.py <unzipped S2 dir>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

DOSES_UM = [20, 10, 3.3, 1.1, 0.37, 0.123, 0.041, 0.0137]
DRUGS = ["taxol_uM", "cisplatin_uM", "doxorubicin_uM"]


def neighbour_sd(T: np.ndarray) -> float:
    nb, c = np.zeros_like(T), np.zeros_like(T)
    for ax in range(3):
        for s in (-1, 1):
            valid = np.ones_like(T, bool)
            valid[(slice(None),) * ax + (0 if s == 1 else -1,)] = False
            nb += np.where(valid, np.roll(T, s, ax), 0)
            c += valid
    r = T - nb / c
    return float(1.4826 * np.median(abs(r - np.median(r))))


def convert(root: Path) -> pd.DataFrame:
    a = sio.loadmat(root / "DatasetsFromEColiDBFolder.mat")["TaxolCisDox20141211"]
    T = a[:, 6].reshape(8, 8, 8)
    idx = np.array(np.unravel_index(np.arange(512), T.shape)).T
    df = pd.DataFrame(np.array(DOSES_UM)[idx], columns=DRUGS)
    df["survival_mean"] = a[:, 6].round(3)
    df["survival_sd"] = round(neighbour_sd(T), 3)
    return df


if __name__ == "__main__":
    out = Path(__file__).parent / "a549_taxol_cis_dox.csv"
    convert(Path(sys.argv[1])).to_csv(out, index=False)
    print(out)
