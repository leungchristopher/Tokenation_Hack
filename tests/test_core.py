import json
import asyncio
import sys

import numpy as np

from bo_eval import Domain, Session, bo_loop, explore
from bo_eval.graph import Edge, Node, ReasoningGraph


def quadratic_domain() -> Domain:
    return Domain(
        name="quadratic",
        description="negative squared distance",
        params=["x"],
        X=np.arange(6)[:, None],
        evaluate=lambda params, rng: -(params["x"] - 3) ** 2,
    )


def test_live_domain_optimises_and_exports_without_inspect(tmp_path):
    assert not any(module == "inspect_ai" or module.startswith("inspect_ai.") for module in sys.modules)
    session = Session(domain=quadratic_domain(), budget=6)
    bo_loop(session, n_init=2)

    assert len(session.graph.experiments) == 6
    assert session.submission == {"x": 3.0}
    assert session.graph.selected is not None
    paths = session.graph.export(tmp_path / "reasoning")
    assert {path.suffix for path in paths} == {".json", ".md", ".svg"}
    assert "Final selection" in paths[1].read_text()
    assert json.loads(paths[0].read_text())["decision"]


def test_gated_model_without_priors_is_plain_gp():
    session = Session(domain=quadratic_domain(), budget=3)
    session.run({"x": 0}, reasoning="initial")
    session.run({"x": 5}, reasoning="contrast")

    plain = session.model()
    empty_gated = session.model(gated=True)
    np.testing.assert_allclose(plain.mean, empty_gated.mean)
    np.testing.assert_allclose(plain.sd, empty_gated.sd)
    assert empty_gated.gates == {}


def test_reasoning_edges_do_not_change_branch_structure():
    graph = ReasoningGraph(
        nodes={
            "root": Node(id="root"),
            "E1": Node(id="E1", params={"x": 1}, result=1),
            "E2": Node(id="E2", params={"x": 2}, result=2),
        },
        edges=[
            Edge(source="root", target="E1", reasoning="try one"),
            Edge(source="root", target="E2", reasoning="try two"),
            Edge(source="E1", target="E2", reasoning="compare", kind="reasoning"),
        ],
    )

    assert set(graph.open_leaves()) == {"E1", "E2"}
    assert graph.close("E1", "not pursued") == ["E1"]
    assert not graph.nodes["E2"].closed


def test_live_domain_has_no_hidden_benchmark_score():
    session = Session(domain=quadratic_domain(), budget=1)
    session.run({"x": 0})
    try:
        session.score()
    except ValueError as error:
        assert "external scorer" in str(error)
    else:
        raise AssertionError("A live domain must not expose benchmark truth.")


def test_literature_guided_loop_uses_same_portable_core():
    async def think(prompt):
        if "give at most" in prompt:
            return json.dumps({
                "hypotheses": [{
                    "claim": "The optimum is near x=3.",
                    "query": "quadratic optimum x 3",
                    "belief": {"x": [3, 0.2]},
                }]
            })
        return json.dumps({
            "trust": 0.8,
            "reason": "Directly applicable synthetic evidence.",
            "sources": ["https://example.org/study"],
        })

    async def search(query):
        return "A directly applicable study: https://example.org/study"

    session = Session(domain=quadratic_domain(), budget=4, max_searches=2)
    gates = asyncio.run(explore(session, think, search))

    assert session.searches == 1
    assert len(session.graph.evidence) == len(session.graph.priors) == 1
    assert session.graph.selected
    assert set(gates) == {"P1"}
