"""Solver registry. Add new architectures to SOLVERS (or pass any solver via `inspect eval --solver`)."""

from inspect_ai.agent import AgentSubmit, as_solver, react
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.util import store_as

from bo_eval.core import Session, bo_loop, random_loop
from bo_eval.state import BOState, use_session
from bo_eval.tools import add_reasoning, bayes_opt_suggest, close_branch, researcher, run_experiment, set_prior, submit, view_graph

INSTRUCTIONS = """You are an autonomous experimentalist searching an experimental space for the optimal configuration.
Every experiment is a node in a reasoning graph; link it to the node it follows from (its parent) with a short
reasoning label. When the evidence shows a branch cannot contain the optimum, close it with close_branch;
closed branches cannot be extended. Before submitting, every experiment you did not continue from must be
closed with a reason explaining why it was not pursued. Keep the number of experiments small. When confident,
call submit()."""

PRIOR = """
Before the first experiment, use set_prior to state where you expect the optimum to lie from your own domain
knowledge of this system, with honest widths and the reasoning behind them. bayes_opt_suggest weights its
suggestions by this prior. Revise it with set_prior when the results contradict it."""

RESEARCH = """
You have a research agent (researcher). Before the first experiment, ask it to survey the literature on this
system and the parameters' effects and interactions; it records cited, trust-scored evidence (R nodes) and may set
a prior that biases bayes_opt_suggest in proportion to its trust. Critique your plan against that evidence. Ask it
again when a result surprises you or before closing a branch, and cite the evidence ids in your reasoning labels."""


@solver
def init_bo(budget: int, seed: int = 0, max_searches: int = 6):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        store_as(BOState).session = Session(env_name=state.metadata["env"], budget=budget,
                                            seed=seed * 1000 + state.epoch, max_searches=max_searches)
        return state

    return solve


def llm_agent(bo: bool = True, graph: bool = True, prior: bool = False, research: bool = False):
    tools = [run_experiment()]
    tools += [bayes_opt_suggest()] if bo else []
    tools += [set_prior()] if prior else []
    tools += [researcher()] if research else []
    tools += [add_reasoning(), close_branch(), view_graph()] if graph else []
    agent = react(
        prompt=INSTRUCTIONS + (PRIOR if prior else "") + (RESEARCH if research else ""),
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


SOLVERS = {
    "react": lambda: llm_agent(bo=True, graph=True),
    "react_prior": lambda: llm_agent(bo=True, graph=True, prior=True),
    "react_research": lambda: llm_agent(bo=True, graph=True, research=True),
    "react_no_bo": lambda: llm_agent(bo=False, graph=True),
    "react_no_graph": lambda: llm_agent(bo=True, graph=False),
    "bo": bo_baseline,
    "random": random_baseline,
}
