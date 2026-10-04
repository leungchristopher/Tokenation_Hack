"""Thin Inspect adapter. The exact same episode engine also runs without Inspect."""

import asyncio
import os
import time

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import Generate, TaskState, solver

from epistemic.loop import Config, run_episode
from epistemic.metrics import episode_metrics
from epistemic.provider import Provider, get_provider
from epistemic.render import export
from epistemic.tasks import load_task


@task
def experimental_design(
    task_name: str = "drug", budget: int = 1000, seed: int = 0,
    provider: str = "none", graph_dir: str = "logs/current-inspect", max_searches: int = 0,
    max_seconds: float = 300.0, max_model_tokens: int = 12_000,
    acquisition: str = "ei", patience: int = 30, noise_cv: float | None = None,
) -> Task:
    config = Config(task=task_name, budget=budget, max_seconds=max_seconds,
                    max_model_tokens=max_model_tokens, patience=patience, seed=seed,
                    provider=provider, max_searches=max_searches, acquisition=acquisition,
                    noise_cv=noise_cv)
    spec, _ = load_task(task_name, noise_cv=noise_cv)

    @solver
    def sequential_episode():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            settings = Config(**(vars(config) | {"seed": seed + state.epoch - 1}))
            if provider == "inspect":
                model = get_model()
                loop = asyncio.get_running_loop()
                generation = model._resolve_config(GenerateConfig(
                    temperature=0.0,
                    seed=settings.seed if os.getenv("EPISTEMIC_SEED_SUPPORTED", "").lower() == "true" else None,
                ))
                generation.max_tokens = generation.max_tokens or 1400

                class InspectProvider(Provider):
                    def metadata(self):
                        return super().metadata() | generation.model_dump(exclude_none=True)

                    def output_token_limit(self) -> int:
                        return generation.max_tokens or 1400

                    def complete(self, prompt: str) -> str:
                        self.calls += 1
                        start = time.perf_counter()
                        future = asyncio.run_coroutine_threadsafe(model.generate(prompt, config=generation), loop)
                        remaining = None if self.deadline is None else max(self.deadline - start, 0.001)
                        try:
                            output = future.result(timeout=remaining)
                        except TimeoutError:
                            future.cancel()
                            raise
                        finally:
                            self.latency_s += time.perf_counter() - start
                        if output.usage:
                            self.tokens += output.usage.total_tokens
                        return output.completion

                backend: Provider = InspectProvider(name="inspect", model=model.name, temperature=0.0)
            else:
                backend = get_provider(provider)
            episode = await asyncio.to_thread(run_episode, settings, backend)
            if graph_dir:
                export(episode, f"{graph_dir}/{task_name}-{state.sample_id}-epoch{state.epoch}")
            state.metadata["metrics"] = episode_metrics(episode)
            state.metadata["provider"] = episode.provider_metadata
            state.output.completion = episode.final_selection() or "no valid experiment"
            return state
        return solve

    @scorer(metrics={"found_optimum": [mean()], "experiments": [mean()]})
    def evaluator_truth():
        async def score(state: TaskState, target: Target) -> Score:
            metrics = state.metadata["metrics"]
            return Score(
                value={"found_optimum": float(metrics["found_optimum"]), "experiments": metrics["experiments"]},
                answer=state.output.completion, metadata=metrics,
                explanation="Finite-dataset evaluator truth; complete error/regret/cost metrics are in metadata.",
            )
        return score

    return Task(dataset=[Sample(id=task_name, input=spec.briefing(budget))],
                solver=sequential_episode(), scorer=evaluator_truth())
