"""Research agent: searches the literature and adds cited, trust-weighted evidence to the reasoning graph."""

from inspect_ai.agent import as_tool, react
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.tool import ToolError, tool, web_search
from inspect_ai.util import store_as

from bo_eval.env import get_env
from bo_eval.state import BOState, Evidence, Prior

RESEARCH = """You are a research agent supporting an experimentalist who is optimising a black-box system.
Search the literature with literature_search and record what you find with cite_evidence. Every claim needs sources
(DOI or URL of the paper you actually found). Score trust in [0, 1]: how far the finding should be believed
*for this exact system* (same cell line / enzyme / lysate, similar ranges and assay, replicated, effect size),
and explain the score. Prefer findings that change what to test (interactions, antagonism, non-monotonicity,
inhibition at high levels) over textbook generalities. When asked to critique a plan or a result, look for
literature that contradicts it as well as supports it. If the evidence implies where the optimum lies, pass a
belief; bayes_opt_suggest then weights its suggestions by it in proportion to the trust you give it.
Finish with a short summary that cites the evidence ids."""


@tool
def literature_search():
    async def execute(query: str) -> str:
        """Search the web/literature and return a summary of findings with DOIs/URLs.

        Args:
            query: What to look for, e.g. "taxol doxorubicin antagonism A549 combination index".
        """
        s = store_as(BOState)
        if s.searches >= s.max_searches:
            raise ToolError(f"Search budget ({s.max_searches}) used up; record evidence from what you have.")
        s.searches += 1
        # Isolated generate: provider-side search blocks never mix with client tool calls in one history.
        out = await get_model().generate(
            "Search the scientific literature for: " + query + "\nReport each relevant finding with its system "
            "(cell line/enzyme, doses), effect size and the DOI or URL, and note conflicting reports. "
            "Be concise: at most 5 findings, under 300 words.",
            tools=[web_search({"anthropic": {"max_uses": 4}})],
            config=GenerateConfig(max_tokens=2000),
        )
        return out.completion

    return execute


@tool
def cite_evidence():
    async def execute(
        claim: str,
        sources: list[str],
        trust: float,
        trust_reason: str,
        about: list[str] | None = None,
        belief: dict[str, list[float]] | None = None,
    ) -> str:
        """Record a literature-backed claim in the reasoning graph.

        Args:
            claim: The finding, stated concretely (e.g. "taxol and doxorubicin are antagonistic in A549 at >1 uM").
            sources: DOIs or URLs supporting the claim.
            trust: 0-1, how far this should be believed for this exact system and parameter range.
            trust_reason: Why that trust (system match, replication, effect size, conflicting reports).
            about: Graph ids (experiments like "E3", priors like "P1") or parameter names the claim bears on.
            belief: Optional map of parameter -> [best_value, width as a fraction of the range] implied by the
                evidence. Becomes the prior used by bayes_opt_suggest, weighted by trust.
        """
        if not sources or not 0 <= trust <= 1:
            raise ToolError("Give at least one source and a trust in [0, 1].")
        s = store_as(BOState)
        g, env = s.graph, get_env(s.env)
        v = Evidence(id=f"R{len(g.evidence) + 1}", claim=claim, sources=sources, trust=trust,
                     trust_reason=trust_reason, about=about or [])
        g.evidence.append(v)
        msg = f"Recorded {v.id}."
        if belief:
            bad = set(belief) - set(env.params)
            if bad or any(len(b) != 2 or b[1] <= 0 for b in belief.values()):
                raise ToolError(f"belief must map parameters in {env.params} to [best, width>0]; got {belief}")
            after = g.experiments[-1].id if g.experiments else "root"
            p = Prior(id=f"P{len(g.priors) + 1}", after=after, belief={k: tuple(b) for k, b in belief.items()},
                      reasoning=f"from {v.id}: {claim}", trust=trust)
            g.priors.append(p)
            v.about.append(p.id)
            msg += f" Prior {p.id} (trust {trust:g}) now weights bayes_opt_suggest."
        s.graph = g
        return msg

    return execute


def researcher():
    agent = react(
        name="researcher",
        description="Searches the literature and adds cited, trust-scored evidence (and optionally a prior) to the reasoning graph.",
        prompt=RESEARCH,
        tools=[literature_search(), cite_evidence()],
    )
    return as_tool(agent, description="Ask the research agent a literature question or to critique a plan/result. "
                   "It records cited evidence (R nodes) in the graph and returns a summary.")
