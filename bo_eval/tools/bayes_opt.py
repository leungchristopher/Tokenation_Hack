import json

from inspect_ai.tool import tool

from bo_eval.state import use_session


@tool
def bayes_opt_suggest():
    async def execute(n: int = 1, avoid_closed: bool = True) -> str:
        """Suggest the next experiment(s) by Bayesian optimisation (GP + expected improvement) over all results so far. Does not consume budget.

        Args:
            n: Number of suggestions to return.
            avoid_closed: Skip candidates whose nearest observed experiment lies on a closed branch.
        """
        return json.dumps(use_session(lambda s: s.suggest(n, avoid_closed=avoid_closed)))

    return execute
