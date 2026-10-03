from inspect_ai.tool import tool

from bo_eval.state import session, use_session


@tool
def add_reasoning():
    async def execute(source: str, target: str, reasoning: str) -> str:
        """Add a reasoning edge between two existing nodes of the reasoning graph.

        Args:
            source: Id of the node the reasoning starts from.
            target: Id of the node the reasoning leads to.
            reasoning: The inference linking the two nodes.
        """
        use_session(lambda s: s.reason(source, target, reasoning))
        return f"Added edge {source} -> {target}."

    return execute


@tool
def close_branch():
    async def execute(node: str, reason: str) -> str:
        """Close a branch: record why it is no longer pursued. Closed branches cannot be extended.

        Args:
            node: Id of the node at the top of the branch.
            reason: Evidence for not pursuing it, with uncertainty; closure is not proof of optimality.
        """
        closed = use_session(lambda s: s.close(node, reason))
        return f"Closed: {', '.join(closed) or 'nothing (already closed)'}."

    return execute


@tool
def view_graph():
    async def execute() -> str:
        """View the reasoning graph: experiment nodes (inputs -> result), reasoning edges, priors, evidence and closed branches."""
        return session().graph.to_text()

    return execute
