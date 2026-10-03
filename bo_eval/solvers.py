"""Solver registry. Add new architectures to SOLVERS (or pass any solver via `inspect eval --solver`)."""

import os

from inspect_ai.agent import AgentSubmit, as_solver, react
from inspect_ai.model import get_model
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.util import store_as

from bo_eval.amass import amass_search
from bo_eval.core import Session, bo_loop, random_loop
from bo_eval.dual import explore
from bo_eval.state import BOState, use_session
from bo_eval.tools import add_reasoning, bayes_opt_suggest, close_branch, run_experiment, submit, view_graph
from bo_eval.tools.research import web_literature

INSTRUCTIONS = """You are an autonomous experimentalist searching an experimental space for the optimal configuration.
Every experiment is a node in a reasoning graph; link it to the node it follows from (its parent) with a concise,
specific reason. Close a branch when you decide not to pursue it, stating the evidence and uncertainty: closure
is an auditable decision, not proof that the region cannot contain the optimum. Closed branches cannot be extended.
Before submitting, close every experiment you did not continue from and explain the final selection. Keep the
number of experiments small. When confident, call submit()."""

@solver
def init_bo(budget: int, seed: int = 0, max_searches: int = 6):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        store_as(BOState).session = Session(env_name=state.metadata["env"], budget=budget,
                                            seed=seed * 1000 + state.epoch, max_searches=max_searches)
        return state

    return solve


def llm_agent():
    tools = [run_experiment(), bayes_opt_suggest(), add_reasoning(), close_branch(), view_graph()]
    agent = react(
        prompt=INSTRUCTIONS,
        tools=tools,
        submit=AgentSubmit(tool=submit(), answer_only=True),
    )
    return as_solver(agent)


@solver
def bo_baseline(n_init: int = 3):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        use_session(lambda s: bo_loop(s, n_init))
        return state

    return solve


@solver
def random_baseline():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        use_session(random_loop)
        return state

    return solve


@solver
def dual_solver(**ablate):
    """Dual spotlight: gated-prior GP (broad) + targeted literature search (narrow); see bo_eval/dual.py."""
    async def think(prompt: str) -> str:
        return (await get_model().generate(prompt)).completion

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        search = amass_search if os.environ.get("AMASS_API_KEY") else web_literature
        st = store_as(BOState)
        s = st.session
        state.metadata["gates"] = await explore(s, think, search, **ablate)
        st.session = s
        return state

    return solve


SOLVERS = {
    "react": llm_agent,
    "bo": bo_baseline,
    "random": random_baseline,
    "dual": dual_solver,
}
