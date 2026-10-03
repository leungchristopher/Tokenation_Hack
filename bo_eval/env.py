"""Black-box experiment environments backed by measured data."""

import sys
from functools import cache, cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bo_eval.domain import Domain

DATA = Path(__file__).resolve().parent.parent / "data"
if not DATA.is_dir():
    DATA = Path(sys.prefix) / "share" / "bo-eval" / "data"


class TabularEnv(Domain):
    """Each run snaps to the nearest measured condition and returns a draw from N(mean, sd)."""

    def __init__(self, name, description, df, params, mean_col, sd_col, goal="maximize", log=()):
        self.df, self.mean_col, self.sd_col = df, mean_col, sd_col
        super().__init__(name, description, params, df[params].to_numpy(float), self._evaluate, goal, tuple(log))

    def _evaluate(self, params: dict, rng: np.random.Generator) -> float:
        i = self.index(params)
        row = self.df.iloc[i]
        return float(max(0.0, rng.normal(row[self.mean_col], row[self.sd_col])))

    def true_value(self, i: int) -> float:
        return float(self.df[self.mean_col].iloc[i])

    @cached_property
    def optimum(self) -> int:
        y = self.df[self.mean_col].to_numpy()
        return int(np.argmax(y) if self.goal == "maximize" else np.argmin(y))

    @classmethod
    def from_csv(cls, name: str, file: str, description: str, mean_col: str, sd_col: str, params=None, goal: str = "maximize", log=()):
        """Load a CSV of measured conditions; duplicate conditions are pooled."""
        df = pd.read_csv(DATA / file)
        params = params or [c for c in df.columns if c not in (mean_col, sd_col, "n")]
        df["_var"] = df[sd_col] ** 2
        df = df.groupby(params, as_index=False)[[mean_col, "_var"]].mean()
        df[sd_col] = np.sqrt(df.pop("_var"))
        return cls(name, description, df, params, mean_col, sd_col, goal, tuple(log))


_ICFREE = "split-GFP fluorescence yield (relative to the no-DNA control) of {} in an Echo-assembled cell-free reaction"

# name -> TabularEnv.from_csv kwargs. To add a task, drop a CSV in data/ and add a line here.
ENVS: dict[str, dict[str, Any]] = {
    "upo_abts": dict(
        file="upo_abts.csv",
        description="the mean specific rate [U/mg] of unspecific peroxygenase (UPO) oxidising ABTS",
        mean_col="rate_mean",
        sd_col="rate_sd",
        params=["ph", "salt_conc", "cosubstrate_conc", "organic_solvent_conc", "temperature"],
    ),
    "icfree_cole1": dict(
        file="icfree/cole1_pro.csv",
        description=_ICFREE.format("colicin E1 in E. coli lysate (proCFPS)"),
        mean_col="yield_mean",
        sd_col="yield_sd",
    ),
    "icfree_colm": dict(
        file="icfree/colm_pro.csv",
        description=_ICFREE.format("colicin M in E. coli lysate (proCFPS)"),
        mean_col="yield_mean",
        sd_col="yield_sd",
    ),
    "icfree_colm_eu": dict(
        file="icfree/colm_eu.csv",
        description="HiBiT luminescence yield of colicin M in HeLa lysate (euCFPS); concentrations are in X of the kit default",
        mean_col="yield_mean",
        sd_col="yield_sd",
    ),
    "zimmer_a549": dict(
        file="zimmer/a549_taxol_cis_dox.csv",
        description="the % survival of A549 lung cancer cells after 48 h with taxol, cisplatin and doxorubicin (doses in uM)",
        mean_col="survival_mean", sd_col="survival_sd", goal="minimize",
        log=("taxol_uM", "cisplatin_uM", "doxorubicin_uM"),
    ),
}


@cache
def get_env(name: str) -> TabularEnv:
    return TabularEnv.from_csv(name, **ENVS[name])
