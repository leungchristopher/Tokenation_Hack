import json

from inspect_ai.tool import tool

from bo_eval.state import use_session


@tool
def submit():
    async def execute(params: dict[str, float], reasoning: str = "") -> str:
        """Submit the configuration you believe is optimal. Ends the episode. Every other unextended experiment must first be closed with close_branch.

        Args:
            params: Value for every parameter.
            reasoning: Why this configuration was chosen over the alternatives.
        """
        return json.dumps(use_session(lambda s: s.submit(params, require_closed=True, reason=reasoning)))

    return execute
