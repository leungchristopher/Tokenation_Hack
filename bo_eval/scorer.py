import json

from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

from bo_eval.env import TabularEnv
from bo_eval.state import session

METRICS = {k: [mean(), stderr()] for k in ("found_optimal", "n_experiments", "regret")}


@scorer(metrics=METRICS)
def bo_scorer(tolerance: float = 0.0, graph_dir: str | None = "logs/graphs"):
    """Inspect wrapper around Session.score(); also writes the reasoning graph to graph_dir."""

    async def score(state: TaskState, target: Target) -> Score:
        s = session()
        r, g = s.score(tolerance), s.graph
        mermaid = g.to_mermaid()
        if graph_dir:
            g.export(f"{graph_dir}/{state.sample_id}_epoch{state.epoch}")
        env = s.env
        assert isinstance(env, TabularEnv)
        return Score(
            value={k: r[k] for k in METRICS},
            answer=json.dumps(r["answer"]),
            explanation=f"```mermaid\n{mermaid}\n```",
            metadata={"optimum": env.condition(env.optimum), "optimum_value": env.true_value(env.optimum),
                      "first_experiment_at_optimum": r["first_experiment_at_optimum"], "graph": g.model_dump()},
        )

    return score
