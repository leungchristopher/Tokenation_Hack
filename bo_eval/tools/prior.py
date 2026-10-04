from inspect_ai.tool import ToolError, tool
from inspect_ai.util import store_as

from bo_eval.env import get_env
from bo_eval.state import BOState, Prior


@tool
def set_prior():
    async def execute(belief: dict[str, list[float]], reasoning: str) -> str:
        """Set or revise your domain-knowledge prior on where the optimum lies. bayes_opt_suggest weights its acquisition by this prior, with the weight decaying as experiments accumulate. Does not consume budget.

        Args:
            belief: Map of parameter -> [best_value, width]. width is your uncertainty as a fraction of the parameter's range (e.g. 0.1 = confident, 0.5 = vague). Omit parameters you have no view on.
            reasoning: The domain knowledge or evidence behind this belief.
        """
        s = store_as(BOState)
        g, env = s.graph, get_env(s.env)
        bad = set(belief) - set(env.params)
        if bad or any(len(v) != 2 for v in belief.values()):
            raise ToolError(f"belief must map parameters in {env.params} to [best, width]; got {belief}")
        after = g.experiments[-1].id if g.experiments else "root"
        p = Prior(id=f"P{len(g.priors) + 1}", after=after, belief={k: tuple(v) for k, v in belief.items()}, reasoning=reasoning)
        g.priors.append(p)
        s.graph = g
        return f"Recorded prior {p.id} after {after}."

    return execute
