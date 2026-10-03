import json
from pathlib import Path

from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

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
            out = Path(graph_dir) / f"{state.sample_id}_epoch{state.epoch}"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.with_suffix(".md").write_text(f"```mermaid\n{mermaid}\n```\n\n```\n{g.to_text()}\n```\n")
            out.with_suffix(".json").write_text(g.model_dump_json(indent=2))
        env = s.env
        return Score(
            value={k: r[k] for k in METRICS},
            answer=json.dumps(r["answer"]),
            explanation=f"```mermaid\n{mermaid}\n```",
            metadata={"optimum": env.condition(env.optimum), "optimum_value": env.true_value(env.optimum),
                      "first_experiment_at_optimum": r["first_experiment_at_optimum"], "graph": g.model_dump()},
        )

    return score
