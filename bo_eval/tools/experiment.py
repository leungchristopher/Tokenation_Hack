import json

from inspect_ai.tool import tool

from bo_eval.state import use_session


@tool
def run_experiment():
    async def execute(params: dict[str, float], parent: str = "root", reasoning: str = "") -> str:
        """Run one experiment (consumes one unit of budget) and add it to the reasoning graph.

        Args:
            params: Value for every parameter, e.g. {"ph": 3.5, ...}. Snapped to the nearest feasible condition.
            parent: Id of the graph node this experiment follows from ("root" or an experiment id like "E3").
            reasoning: Why this experiment follows from the parent node.
        """
        def go(s):
            n = s.run(params, parent, reasoning)
            return {"id": n.id, "params": n.params, "result": n.result,
                    "remaining_budget": s.budget - len(s.graph.experiments)}

        return json.dumps(use_session(go))

    return execute
