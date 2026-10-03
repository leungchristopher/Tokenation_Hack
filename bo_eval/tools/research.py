"""Research agent: searches the literature and adds cited, trust-weighted evidence to the reasoning graph."""

from inspect_ai.agent import as_tool, react
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.tool import tool, web_search

from bo_eval.state import use_session

RESEARCH = """You are a research agent supporting an experimentalist who is optimising a black-box system.
Search the literature with literature_search and record what you find with cite_evidence. Every claim needs sources
(DOI or URL of the paper you actually found). Score trust in [0, 1]: how far the finding should be believed
*for this exact system* (same cell line / enzyme / lysate, similar ranges and assay, replicated, effect size),
and explain the score. Prefer findings that change what to test (interactions, antagonism, non-monotonicity,
inhibition at high levels) over textbook generalities. When asked to critique a plan or a result, look for
literature that contradicts it as well as supports it. If the evidence implies where the optimum lies, pass a
belief; bayes_opt_suggest then weights its suggestions by it in proportion to the trust you give it.
Finish with a short summary that cites the evidence ids."""


async def web_literature(query: str) -> str:
    """Web literature search via the eval model's provider-side search tool."""
    # Isolated generate: provider-side search blocks never mix with client tool calls in one history.
    out = await get_model().generate(
        "Search the scientific literature for: " + query + "\nReport each relevant finding with its system "
        "(cell line/enzyme, doses), effect size and the DOI or URL, and note conflicting reports. "
        "Be concise: at most 5 findings, under 300 words.",
        tools=[web_search({"anthropic": {"max_uses": 4}})],
        config=GenerateConfig(max_tokens=2000),
    )
    return out.completion


@tool
def literature_search():
    async def execute(query: str) -> str:
        """Search the web/literature and return a summary of findings with DOIs/URLs.

        Args:
            query: What to look for, e.g. "taxol doxorubicin antagonism A549 combination index".
        """
        use_session(lambda s: s.charge_search())
        return await web_literature(query)

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
        v = use_session(lambda s: s.cite(claim, sources, trust, trust_reason, about, belief))
        prior = [a for a in v.about if a.startswith("P")]
        return f"Recorded {v.id}." + (f" Prior {prior[-1]} (trust {trust:g}) now weights bayes_opt_suggest." if belief else "")

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
