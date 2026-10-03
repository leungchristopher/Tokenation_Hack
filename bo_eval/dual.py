"""Dual spotlight explorer: broad adaptive optimisation (System 1) + narrow literature search (System 2).

No Inspect dependency.

Broad:  every step, EI under the shared GP, whose optional prior basis functions are gated. Each gate is learned from the data,
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

from bo_eval.core import Session
import numpy as np

Ask = Callable[[str], Awaitable[str]]

CHALLENGE = """Include at least one hypothesis that challenges the naive "more is better" view (antagonism, interactions,
non-monotonic or narrow optima), and phrase its query to look for that evidence.
"""

HYPOTHESISE = """You are planning experiments on {desc}. Goal: {goal}. Parameters and tested ranges: {ranges}.
From your own domain knowledge, give at most {k} hypotheses about where the optimum lies that should change where
to look first (optimal levels, interactions, antagonism, inhibition at high levels).
{challenge}Answer only JSON: {{"hypotheses": [{{"claim": "...", "query": "literature search query",
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
    sources = [
        source for source in r.get("sources", [])
        if isinstance(source, str) and source.strip() and source not in {"(none)", "(no source found)"}
    ]
    return {"trust": float(np.clip(r.get("trust", 0.0), 0, 1)) if sources else 0.0,
            "trust_reason": r.get("reason", ""), "sources": sources}


async def hypothesise(s: Session, think: Ask, search: Ask, literature: bool = True, challenge: bool = True) -> None:
    env = s.env
    ranges = {p: [float(env.X[:, j].min()), float(env.X[:, j].max())] for j, p in enumerate(env.params)}
    out = await ask(think, HYPOTHESISE.format(desc=env.description, goal=env.goal, ranges=ranges, k=max(s.max_searches // 2, 1),
                                           challenge=CHALLENGE if challenge else ""))
    for h in out.get("hypotheses", [])[: s.max_searches]:
        belief = {p: b for p, b in (h.get("belief") or {}).items() if p in env.params and len(b) == 2 and b[1] > 0}
        if belief:
            if literature:
                r = await calibrate(s, think, search, h["claim"], h["query"])
                if r["sources"]:
                    s.cite(h["claim"], **r, about=list(belief), belief=belief)
            else:
                s.set_prior(belief, f"Uncited LLM hypothesis: {h['claim']}", trust=0.5)


async def explore(s: Session, think: Ask, search: Ask, refocus: float = 0.3, priors: bool = True, learn: bool = True,
                  literature: bool = True, challenge: bool = True) -> dict[str, float]:
    """Run the budget; return each prior's learned gate (on the trust scale). The flags switch parts off for ablation."""
    if priors:
        await hypothesise(s, think, search, literature, challenge)
    env, sign, gates, refocused = s.env, (1.0 if s.env.goal == "maximize" else -1.0), {}, set()
    while len(exps := s.graph.experiments) < s.budget:
        priors, open_ = s.graph.priors, [e for e in exps if not e.closed]
        if len(exps) < 2:
            suggestion = s.suggest(1, avoid_closed=False, gated=True, learn=learn)[0]
            why = suggestion["reason"]
        else:
            model = s.model(gated=True, learn=learn)
            mu, sd = model.mean, model.sd
            g, gsd = list(model.gates.values()), list(model.gate_sd.values())
            gates = {p.id: round(float(x), 2) for p, x in zip(priors, g)}
            idx = {e.id: env.index(e.params) for e in exps}
            inc = max(exps, key=lambda e: sign * mu[idx[e.id]])
            for e in open_:  # Hintikka closure: the optimistic bound cannot beat the incumbent
                bound = mu[idx[e.id]] + sign * 2 * sd[idx[e.id]]
                if e.id != inc.id and e.id in s.graph.open_leaves() and sign * bound < sign * mu[idx[inc.id]]:
                    s.close(e.id, f"Not pursued under the current GP: bound {bound:.3g} is below "
                              f"{inc.id}'s prediction {mu[idx[inc.id]]:.3g}. This is a model judgement, not proof.")
            for p, gp, gs in zip(priors, g, gsd):  # narrow spotlight: data disagrees with the literature
                if literature and p.id not in refocused and abs(gp - p.trust) > refocus and gs < refocus / 2 and s.searches < s.max_searches:
                    refocused.add(p.id)
                    ev = next((v for v in reversed(s.graph.evidence) if p.id in v.about), None)
                    if ev is None:
                        continue
                    seen = f"Experiments so far gate it at {gp:.2f} against literature trust {p.trust:.2f}.\n"
                    r = await calibrate(s, think, search, ev.claim, f"{ev.claim} contradicting evidence", seen)
                    if r["sources"]:
                        s.cite(ev.claim, r["sources"], r["trust"], r["trust_reason"], about=[p.id, inc.id])
            suggestions = s.suggest(
                1, xi=0.0, avoid_closed=False, gated=True, learn=learn, model=model
            )
            if not suggestions:
                break
            suggestion = suggestions[0]
            why = suggestion["reason"] + "".join(f"; gate {k}={v:g}" for k, v in gates.items())
        params = suggestion["params"]
        open_ = [e for e in s.graph.experiments if not e.closed]
        C = env.encode(np.array([list(params.values())] + [list(e.params.values()) for e in open_]))
        parent = open_[int(((C[1:] - C[0]) ** 2).sum(1).argmin())].id if open_ else "root"
        s.run(params, parent, why)
    mu = s.model(gated=True, learn=learn).mean
    chosen = max(s.graph.experiments, key=lambda e: sign * mu[env.index(e.params)])
    chosen_mean = mu[env.index(chosen.params)]
    for node_id in s.graph.open_leaves():
        if node_id != chosen.id:
            node = s.graph.nodes[node_id]
            prediction = mu[env.index(node.params)]
            s.close(node_id, f"Budget ended; posterior mean {prediction:.4g} was not preferred to "
                    f"{chosen.id} at {chosen_mean:.4g}.")
    s.submit(chosen.params, reason=f"Best posterior mean among tested conditions ({chosen_mean:.4g}).")
    return gates
