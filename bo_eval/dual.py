"""Dual spotlight explorer: broad adaptive optimisation (System 1) + narrow literature search (System 2).

No Inspect dependency.

Broad:  every step, EI under a GP whose prior knowledge is gated (gated.py). Each gate is learned from the data,
        so the model decides how much of each prior to keep.
Narrow: the LLM is called only (1) at cold start, to state hypotheses from its own taste, each calibrated by one
        literature search into a cited prior; and (2) when the data pulls a gate far from the trust the
        literature gave it, for one focused search on that hypothesis.
The graph stays pristine: experiments, reasoning edges, closures, and cited evidence/priors.

    s = Session(env_name="zimmer_a549", budget=30, max_searches=6)
    gates = await explore(s, think=my_llm, search=amass_search)
"""

import json
import re
from typing import Awaitable, Callable

import numpy as np
from scipy.stats import norm

from bo_eval.core import Session
from bo_eval.gated import gated_posterior

Ask = Callable[[str], Awaitable[str]]

HYPOTHESISE = """You are planning experiments on {desc}. Goal: {goal}. Parameters and tested ranges: {ranges}.
From your own domain knowledge, give at most {k} hypotheses about where the optimum lies that should change where
to look first (optimal levels, interactions, antagonism, inhibition at high levels).
Include at least one hypothesis that challenges the naive "more is better" view (antagonism, interactions,
non-monotonic or narrow optima), and phrase its query to look for that evidence.
Answer only JSON: {{"hypotheses": [{{"claim": "...", "query": "literature search query",
"belief": {{"<param>": [best_value, width as a fraction of the range]}}}}]}}"""

CALIBRATE = """Hypothesis about {desc}: {claim}
{context}Literature search results:
{found}
How far should this hypothesis be trusted for this exact system? Weigh system match, replication, citations,
journal quality and retractions; contradicting evidence lowers trust. Answer only JSON:
{{"trust": <0-1>, "reason": "...", "sources": ["DOI or URL that appears above"]}}"""


async def ask(think: Ask, prompt: str) -> dict:
    m = re.search(r"\{.*\}", await think(prompt), re.S)
    try:
        return json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        return {}


async def calibrate(s: Session, think: Ask, search: Ask, claim: str, query: str, context: str = "") -> dict:
    s.charge_search()
    r = await ask(think, CALIBRATE.format(desc=s.env.description, claim=claim, context=context, found=await search(query)))
    return {"trust": float(np.clip(r.get("trust", 0.0), 0, 1)), "trust_reason": r.get("reason", ""),
            "sources": r.get("sources") or ["(no source found)"]}


async def hypothesise(s: Session, think: Ask, search: Ask) -> None:
    env = s.env
    ranges = {p: [float(env.X[:, j].min()), float(env.X[:, j].max())] for j, p in enumerate(env.params)}
    out = await ask(think, HYPOTHESISE.format(desc=env.description, goal=env.goal, ranges=ranges, k=max(s.max_searches // 2, 1)))
    for h in out.get("hypotheses", [])[: s.max_searches]:
        belief = {p: b for p, b in (h.get("belief") or {}).items() if p in env.params and len(b) == 2 and b[1] > 0}
        if belief:
            s.cite(h["claim"], **await calibrate(s, think, search, h["claim"], h["query"]), about=list(belief), belief=belief)


async def explore(s: Session, think: Ask, search: Ask, refocus: float = 0.3) -> dict[str, float]:
    """Run the budget; return each prior's learned gate (on the trust scale)."""
    await hypothesise(s, think, search)
    env, sign, gates, refocused = s.env, (1.0 if s.env.goal == "maximize" else -1.0), {}, set()
    while len(exps := s.graph.experiments) < s.budget:
        priors, open_ = s.graph.priors, [e for e in exps if not e.closed]
        tested = {env.index(e.params) for e in exps}
        if len(exps) < 2:
            sug = s.suggest(1)[0]
            i, why = env.index(sug["params"]), sug["reason"]
        else:
            mu, sd, g, gsd = gated_posterior(env, exps, priors)
            gates = {p.id: round(float(x), 2) for p, x in zip(priors, g)}
            best = max(sign * e.result for e in exps)
            u = (sign * mu - best) / sd
            ei = (sign * mu - best) * norm.cdf(u) + sd * norm.pdf(u)
            idx = {e.id: env.index(e.params) for e in exps}
            inc = max(exps, key=lambda e: sign * mu[idx[e.id]])
            for e in open_:  # Hintikka closure: the optimistic bound cannot beat the incumbent
                bound = mu[idx[e.id]] + sign * 2 * sd[idx[e.id]]
                if e.id != inc.id and e.id in s.graph.open_leaves() and sign * bound < sign * mu[idx[inc.id]]:
                    s.close(e.id, f"GP bound {bound:.3g} cannot beat {inc.id} (predicted {mu[idx[inc.id]]:.3g})")
            for p, gp, gs in zip(priors, g, gsd):  # narrow spotlight: data disagrees with the literature
                if p.id not in refocused and abs(gp - p.trust) > refocus and gs < refocus / 2 and s.searches < s.max_searches:
                    refocused.add(p.id)
                    ev = next(v for v in s.graph.evidence if p.id in v.about)
                    seen = f"Experiments so far gate it at {gp:.2f} against literature trust {p.trust:.2f}.\n"
                    r = await calibrate(s, think, search, ev.claim, f"{ev.claim} contradicting evidence", seen)
                    s.cite(ev.claim, r["sources"], r["trust"], r["trust_reason"], about=[p.id, inc.id])
                    p.trust = r["trust"]
            closed = {env.index(e.params) for e in exps if e.closed}
            order = [j for j in np.argsort(-ei) if j not in tested and j not in closed]
            if not order:
                break
            i = order[0]
            why = f"EI {ei[i]:.2g}: predicted {mu[i]:.3g} ± {sd[i]:.3g}" + "".join(f", gate {k} {v:g}" for k, v in gates.items())
        params = env.condition(i)
        open_ = [e for e in s.graph.experiments if not e.closed]
        C = env.encode(np.array([list(params.values())] + [list(e.params.values()) for e in open_]))
        parent = open_[int(((C[1:] - C[0]) ** 2).sum(1).argmin())].id if open_ else "root"
        s.run(params, parent, why)
    mu, *_ = gated_posterior(env, s.graph.experiments, s.graph.priors)
    s.submit(max(s.graph.experiments, key=lambda e: sign * mu[env.index(e.params)]).params)
    return gates
