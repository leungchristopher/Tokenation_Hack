"""System 1 / System 2 explorer with short- and long-term memory. No Inspect dependency.

System 1 (fast, every step): the GP proposes each experiment, and closes any branch whose optimistic GP bound
cannot beat the incumbent, with those numbers as the reason.
System 2 (slow, gated): the LLM, backed by literature search, is called only
  1. at cold start, where BO is weakest: it proposes hypotheses from its own taste, and each is calibrated
     against the literature (or recalled from long-term memory, which costs no search);
  2. on surprise, when a result falls outside the GP's band: it judges which hypotheses the result bears on.
Hypotheses steer BO as one prior: the product of their beliefs, each raised to its trust (piBO).
Short-term memory is the Session (graph + GP). Long-term memory (memory.py) is updated at the end of the run:
a claim is confirmed if the GP's predicted optimum lies inside its belief.

    s = Session(env_name="zimmer_a549", budget=30, max_searches=4)
    await explore(s, think=my_llm, search=amass_search, memory=Memory.load("logs/memory.json"))
"""

import json
import re
from typing import Awaitable, Callable

import numpy as np

from bo_eval.core import Session
from bo_eval.graph import Edge
from bo_eval.memory import Claim, Memory

Ask = Callable[[str], Awaitable[str]]

HYPOTHESISE = """You are planning experiments on {desc}. Goal: {goal}. Parameters and tested ranges: {ranges}.
Claims already in long-term memory (key: claim [trust]):
{recalled}
Give at most {k} hypotheses from your own domain knowledge that should change where to look first (optimal
levels, interactions, antagonism, inhibition at high levels). Reuse a memory key when a claim already covers one.
Answer only JSON: {{"hypotheses": [{{"key": "short-slug", "claim": "...", "query": "literature search query",
"belief": {{"<param>": [best_value, width as a fraction of the range]}}}}]}}"""

CALIBRATE = """Hypothesis about {desc}: {claim}
Literature search results:
{found}
How far should this hypothesis be trusted for this exact system? Weigh system match, replication, citations,
journal quality and retractions; contradicting evidence lowers trust. Answer only JSON:
{{"trust": <0-1>, "reason": "...", "sources": ["DOI or URL that appears above"]}}"""

EXPLAIN = """Optimising {desc} ({goal}). Experiment {node} at {params} gave {y:.4g}; the GP predicted {mu:.4g} ± {sd:.4g}.
Hypotheses (key: claim [trust]):
{claims}
Is this noise, or evidence for or against a hypothesis? Answer only JSON:
{{"explanation": "...", "supports": ["key", ...], "refutes": ["key", ...]}}"""


async def ask(think: Ask, prompt: str) -> dict:
    m = re.search(r"\{.*\}", await think(prompt), re.S)
    try:
        return json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        return {}


def _listing(claims: list[Claim]) -> str:
    return "\n".join(f"{c.key}: {c.claim} [{c.trust:.2f}]" for c in claims) or "(none)"


def _enc(s: Session, p: str, v: float) -> float:
    ref = s.env.X[:1].copy()
    ref[0, s.env.params.index(p)] = v
    return s.env.encode(ref)[0, s.env.params.index(p)]


def _prior(s: Session, claims: list[Claim], why: str) -> None:
    """One prior = product of Gaussian beliefs, each to the power of its trust (a precision-weighted average)."""
    belief, lg = {}, set(s.env.log)
    for p in s.env.params:
        bs = [(max(c.trust, 1e-3) / c.belief[p][1] ** 2, c.belief[p][0]) for c in claims if p in c.belief]
        if bs:
            f = (lambda v: np.log10(max(v, 1e-12))) if p in lg else float
            m = sum(w * f(b) for w, b in bs) / sum(w for w, _ in bs)
            belief[p] = [10**m if p in lg else m, sum(w for w, _ in bs) ** -0.5]
    if belief:
        prior = s.set_prior(belief, why)
        for v in s.graph.evidence:
            v.about.append(prior.id)


async def hypothesise(s: Session, think: Ask, search: Ask, memory: Memory) -> list[Claim]:
    env, recalled = s.env, {c.key: c for c in memory.recall(s.env)}
    ranges = {p: [float(env.X[:, j].min()), float(env.X[:, j].max())] for j, p in enumerate(env.params)}
    out = await ask(think, HYPOTHESISE.format(desc=env.description, goal=env.goal, ranges=ranges,
                                              recalled=_listing(list(recalled.values())), k=s.max_searches + 2))
    claims = []
    for h in out.get("hypotheses", []):
        belief = {p: tuple(b) for p, b in (h.get("belief") or {}).items() if p in env.params and len(b) == 2 and b[1] > 0}
        if h.get("key") in recalled:
            c = recalled[h["key"]]
            why = f"long-term memory: {c.confirmed:.1f} confirmed vs {c.refuted:.1f} refuted"
        elif s.searches < s.max_searches:
            s.charge_search()
            r = await ask(think, CALIBRATE.format(desc=env.description, claim=h["claim"], found=await search(h["query"])))
            c = Claim(key=h["key"], claim=h["claim"], env=env.name, sources=r.get("sources", []), belief=belief)
            c.calibrate(float(np.clip(r.get("trust", 0.0), 0, 1)))
            why = r.get("reason", "")
        else:
            continue
        s.cite(c.claim, c.sources or ["(no source found)"], c.trust, why, about=list(c.belief))
        claims.append(c)
    _prior(s, [c for c in claims if c.belief], "trust-weighted product of " + ", ".join(v.id for v in s.graph.evidence))
    return claims


def prune(s: Session, z: float = 2.0) -> None:
    """Close open leaves whose optimistic GP bound cannot beat the incumbent's predicted value."""
    exps = s.graph.experiments
    if len(exps) < 5:
        return
    mu, sd = s.posterior()
    g = 1.0 if s.env.goal == "maximize" else -1.0
    idx = {e.id: s.env.index(e.params) for e in exps}
    inc = max(exps, key=lambda e: g * mu[idx[e.id]])
    for nid in s.graph.open_leaves():
        i = idx[nid]
        if nid != inc.id and g * (mu[i] + g * z * sd[i]) < g * mu[idx[inc.id]]:
            s.close(nid, f"GP: optimistic bound {mu[i] + g * z * sd[i]:.3g} cannot beat {inc.id} (predicted {mu[idx[inc.id]]:.3g})")


def _parent(s: Session, params: dict) -> str:
    """The nearest open experiment, so the graph shows which result each step refines."""
    open_ = [e for e in s.graph.experiments if not e.closed]
    if not open_:
        return "root"
    C = s.env.encode(np.array([list(params.values())] + [list(e.params.values()) for e in open_]))
    return open_[int(((C[1:] - C[0]) ** 2).sum(1).argmin())].id


async def explore(s: Session, think: Ask, search: Ask, memory: Memory, surprise: float = 2.5) -> None:
    claims = await hypothesise(s, think, search, memory)
    while len(s.graph.experiments) < s.budget and (sug := s.suggest(1)):
        x = sug[0]
        why = x.get("reason") or f"S1: predicted {x['predicted_mean']:.3g} ± {x['predicted_sd']:.3g}, EI {x['expected_improvement']:.2g}"
        node = s.run(x["params"], _parent(s, x["params"]), why)
        if claims and "predicted_sd" in x and abs(node.result - x["predicted_mean"]) > surprise * x["predicted_sd"]:
            r = await ask(think, EXPLAIN.format(desc=s.env.description, goal=s.env.goal, node=node.id, params=node.params,
                                                y=node.result, mu=x["predicted_mean"], sd=x["predicted_sd"], claims=_listing(claims)))
            for c in claims:
                if c.key in r.get("supports", []) + r.get("refutes", []):
                    c.update(c.key in r.get("supports", []), f"{node.id} in {s.env_name}: {r.get('explanation', '')}")
            s.graph.edges[-1].reasoning += " | S2 surprise: " + r.get("explanation", "?")
            _prior(s, [c for c in claims if c.belief], f"revised after surprise at {node.id}")
        prune(s)
    mu, _ = s.posterior()
    g = 1.0 if s.env.goal == "maximize" else -1.0
    best = max(s.graph.experiments, key=lambda e: g * mu[s.env.index(e.params)])
    s.submit(best.params)
    x_star = s.env.condition(int(np.argmax(g * mu)))
    for c, v in zip(claims, s.graph.evidence):
        if c.belief:
            ok = all(abs(_enc(s, p, x_star[p]) - _enc(s, p, b)) <= 2 * w for p, (b, w) in c.belief.items())
            c.update(ok, f"GP optimum {x_star} in {s.env_name} {'inside' if ok else 'outside'} belief")
            s.graph.edges.append(Edge(source=v.id, target=best.id, reasoning=c.history[-1]))
        memory.write(c)
    memory.save()
