"""Black-box experiment environments backed by measured data."""

from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


@dataclass
class TabularEnv:
    """Each run snaps to the nearest measured condition and returns a draw from N(mean, sd)."""

    name: str
    description: str
    df: pd.DataFrame
    params: list[str]
    mean_col: str
    sd_col: str
    goal: str = "maximize"
    log: tuple[str, ...] = ()

    @cached_property
    def X(self) -> np.ndarray:
        return self.df[self.params].to_numpy(float)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Scale each parameter to [0, 1]; parameters in `log` are scaled in log10."""
        lg = np.isin(self.params, self.log)
        X, ref = (np.where(lg, np.log10(np.maximum(a, 1e-12)), a) for a in (np.asarray(X, float), self.X))
        lo, hi = ref.min(0), ref.max(0)
        return (X - lo) / np.where(hi > lo, hi - lo, 1.0)

    def index(self, params: dict) -> int:
        missing = set(self.params) - set(params)
        if missing:
            raise ValueError(f"Missing parameters: {sorted(missing)}")
        x = np.array([[float(params[p]) for p in self.params]])
        return int(np.argmin(((self.encode(self.X) - self.encode(x)) ** 2).sum(1)))

    def condition(self, i: int) -> dict:
        return {p: float(self.df[p].iloc[i]) for p in self.params}

    def sample(self, i: int, rng: np.random.Generator) -> float:
        row = self.df.iloc[i]
        return float(max(0.0, rng.normal(row[self.mean_col], row[self.sd_col])))

    def true_value(self, i: int) -> float:
        return float(self.df[self.mean_col].iloc[i])

    @cached_property
    def optimum(self) -> int:
        y = self.df[self.mean_col].to_numpy()
        return int(np.argmax(y) if self.goal == "maximize" else np.argmin(y))

    def prompt(self, budget: int) -> str:
        def levels(p):
            v = sorted(self.df[p].unique().tolist())
            return v if len(v) <= 12 else f"{len(v)} levels in [{v[0]}, {v[-1]}]"

        lines = "\n".join(f"- {p}: {levels(p)}" for p in self.params)
        return (
            f"Goal: {self.goal} {self.description}.\n"
            f"Parameters and their available levels:\n{lines}\n"
            f"Only {len(self.df)} combinations are feasible; requested settings are snapped to the "
            f"nearest feasible condition. Measurements are noisy.\n"
            f"Experiment budget: {budget}. You are scored on finding the true optimum "
            f"using as few experiments as possible."
        )

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
ENVS = {
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
