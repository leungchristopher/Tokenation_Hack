"""Provider-side web literature search used when Amass is unavailable."""

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.tool import web_search


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
