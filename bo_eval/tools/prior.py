from inspect_ai.tool import tool

from bo_eval.state import use_session


@tool
def set_prior():
    async def execute(belief: dict[str, list[float]], reasoning: str) -> str:
        """Set or revise your domain-knowledge prior on where the optimum lies. bayes_opt_suggest weights its acquisition by this prior, with the weight decaying as experiments accumulate. Does not consume budget.

        Args:
            belief: Map of parameter -> [best_value, width]. width is your uncertainty as a fraction of the parameter's range (e.g. 0.1 = confident, 0.5 = vague). Omit parameters you have no view on.
            reasoning: The domain knowledge or evidence behind this belief.
        """
        p = use_session(lambda s: s.set_prior(belief, reasoning))
        return f"Recorded prior {p.id} after {p.after}."

    return execute
