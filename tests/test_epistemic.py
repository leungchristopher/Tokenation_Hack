import json
from dataclasses import asdict

import numpy as np
import pytest
from pydantic import ValidationError

from epistemic.evidence import seed_graph
from epistemic.execution import PerfectExecution, PerturbedExecution
from epistemic.graph import Claim, EvidenceGraph, Observation
from epistemic.interventions import closed_loop, graph_value, intervene, next_decision, paired_interventions, rebuild
from epistemic.loop import Config, run_episode
from epistemic.metrics import episode_metrics
from epistemic.policies import LLMPolicy, SelectionError
from epistemic.provider import MockProvider, OpenAICompatibleProvider
from epistemic.render import export
from epistemic.surrogate import Surrogate
from epistemic.tasks import TASKS, load_task


@pytest.mark.parametrize("name,count", [("enzyme", 814), ("drug", 512)])
def test_two_real_task_contracts_and_no_labels(name, count):
    task, evaluator = load_task(name)
    assert set(TASKS) == {"enzyme", "drug"}
    assert len(task.ids()) == len(set(task.ids())) == count
    assert len(task.candidates[task.names].drop_duplicates()) == count
    assert set(task.candidates) == {"candidate_id", *task.names}
    assert not hasattr(task, "_outcomes")
    serialized = json.dumps(asdict(task), default=str)
    assert "survival_mean" not in serialized and "rate_mean" not in serialized
    assert evaluator.truth(task.ids()[0]) >= 0
    assert task.noise.replicates is False
    if name == "drug":
        assert all(len(task.candidates[n].unique()) == 8 for n in task.names)
        assert all(p.unit == "micromolar" for p in task.parameters)
        assert all(p.log for p in task.parameters)
        assert "no synergy" in " ".join(task.limitations)


def test_clipping_mapping_and_perfect_control():
    task, _ = load_task("drug")
    rng = np.random.default_rng(2)
    cid = task.ids()[-1]
    perfect = PerfectExecution(task).run(cid, rng)
    assert perfect.realised_id == cid
    assert perfect.accessible()["reported_delivered_params"] == task.params_of(cid)
    perturbed = PerturbedExecution(task, relative_sd=8).run(cid, rng)
    assert perturbed.clipped
    assert perturbed.realised_id in task.ids()
    for n, (lo, hi) in task.bounds().items():
        assert lo <= perturbed.realised_params[n] <= hi
    assert not {"realised_id", "realised_params"} & set(perturbed.accessible())
    # Mapping uses parameter geometry in declared coordinates, independent of candidate row order.
    before = task.nearest(perturbed.realised_params)
    task.candidates = task.candidates.sample(frac=1, random_state=4)
    assert task.nearest(perturbed.realised_params) == before


def test_immutable_outputs_revision_preservation_and_references():
    graph = EvidenceGraph()
    observation = Observation(id="O1", round=1, candidate_id="C0", intended_params={"x": 1.},
                              execution={"report": {"x": 0.9}}, value_shown=4., outcome_unit="test")
    graph.add_observation(observation)
    with pytest.raises(ValidationError):
        observation.value_shown = 5.
    with pytest.raises(TypeError):
        observation.execution["report"]["x"] = 2.
    with pytest.raises(ValueError):
        graph.add_observation(observation)
    claim = graph.add_claim(Claim(id="K", statement="a model conjecture", scope="test", source="model_conjecture",
                                 evidence=["O1"], discriminating_result="an observable counterexample"))
    old = claim.revisions[0].model_dump()
    graph.revise("K", 2, "contradicted", "a counterexample", ["O1"], "revised conjecture")
    assert claim.revisions[0].model_dump() == old
    assert len(claim.revisions) == 2
    graph.link("O1", "K", "contradicts")
    with pytest.raises(KeyError):
        graph.link("fabricated", "K", "supports")
    with pytest.raises(ValidationError):
        graph.link("O1", "K", "causes")
    assert EvidenceGraph.model_validate_json(graph.model_dump_json()).claims["K"].revisions[0].statement == old["statement"]


def test_rebuild_removes_stale_state_and_preserves_frozen_experiment_metadata():
    episode = run_episode(Config(task="drug", policy="random", budget=3, seed=2))
    original = list(episode.graph.observations.values())
    for arm in ("removed", "permuted", "contradicted"):
        modified = intervene(original, episode.task, arm)
        assert [o.id for o in modified] == [o.id for o in original]
        assert [o.execution for o in modified] == [o.execution for o in original]
        graph, model, history = rebuild(episode.task, modified, 2)
        assert graph is not episode.graph
        if arm == "removed":
            assert len(graph.observations) == 3
            assert not history
            assert "K_best" not in graph
            assert "K_calibration" not in graph
            assert model.observations == 0
        else:
            altered = [o.value_shown for o in modified]
            assert [v for _, v in history] == altered
            best_id = graph.claims["K_best"].revisions[-1].evidence[0]
            assert graph.observations[best_id].value_shown == min(altered)


def test_direct_llm_arm_has_no_numerical_model_advice():
    episode = run_episode(Config(task="drug", policy="random", budget=2))
    config = Config(policy="llm", with_model_advice=False, budget=9)
    for arm in ("true", "contradicted"):
        _, audit = next_decision(config, list(episode.graph.observations.values()), arm, 1)
        prompt = audit["prompt"]
        assert "predicted_mean" not in prompt
        assert "K_calibration" not in prompt
        assert '"available": false' in prompt
        assert "Budget: 9" in prompt


@pytest.mark.parametrize("task_name", ["enzyme", "drug"])
@pytest.mark.parametrize("policy", ["random", "bo", "llm"])
def test_seeded_real_adapter_episodes_repeat(task_name, policy):
    config = Config(task=task_name, policy=policy, budget=4, seed=11, execution="perturbed")
    a, b = run_episode(config), run_episode(config)
    assert a.trajectory == b.trajectory
    assert a.hidden == b.hidden
    assert a.graph.model_dump() == b.graph.model_dump()
    assert all(o.simulated for o in a.graph.observations.values())
    for decision in a.decisions:
        expected = "unavailable" if decision.prediction is None else (
            "llm" if policy == "llm" else "numerical_model")
        assert decision.prediction_source == expected
        assert all(edge.kind == "depends_on" for edge in a.graph.edges
                   if edge.source == decision.id and edge.target in decision.claims)
    assert episode_metrics(a) == episode_metrics(b)


def test_scoring_uses_hidden_truth_not_displayed_feedback():
    episode = run_episode(Config(policy="random", budget=3, seed=8, feedback="corrupted"))
    before = episode_metrics(episode)
    for step in episode.hidden:
        step["shown_value"] = -9999.
    after = episode_metrics(episode)
    assert before["best_true_regret"] == after["best_true_regret"]
    assert before["best_true_value"] == after["best_true_value"]
    assert before["best_true_objective_curve"] == after["best_true_objective_curve"]
    assert after["best_true_value"] != -9999


class FixedProvider(MockProvider):
    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return self.text


def test_full_candidate_pool_and_bounded_invalid_output_without_fallback():
    task, _ = load_task("drug")
    graph = seed_graph(task)
    model = Surrogate(task).fit([])
    payload = {
        "candidate_id": task.ids()[-1], "evidence": ["K1"], "targeted_uncertainty": "evidence",
        "prediction": 1., "rationale": "select outside the numerical suggestions",
    }
    provider = FixedProvider(json.dumps(payload))
    policy = LLMPolicy(task, provider)
    assert policy.propose(graph, model, [], np.random.default_rng(0), 1).candidate_id == task.ids()[-1]
    pool = json.loads(policy.last_prompt.split("FULL_CANDIDATE_POOL=", 1)[1].splitlines()[0])
    assert [row[0] for row in pool] == task.ids()
    for broken in ("narrative", json.dumps(payload | {"evidence": ["missing"]}),
                   json.dumps(payload | {"prediction": float("inf")}),
                   json.dumps(payload | {"targeted_uncertainty": "confidence"}),
                   json.dumps(payload | {"candidate_id": "invented"})):
        invalid = FixedProvider(broken)
        with pytest.raises(SelectionError):
            LLMPolicy(task, invalid).propose(graph, model, [], np.random.default_rng(0), 1)
        assert invalid.calls == 3
    stopped = run_episode(Config(policy="llm", budget=3), provider=FixedProvider("narrative"))
    assert not stopped.hidden
    assert not stopped.decisions
    assert stopped.trajectory[0]["status"] == "stopped_after_invalid_output"
    assert episode_metrics(stopped)["invalid_llm_attempts"] == 3


def test_repeats_use_budget_and_agent_audit_has_no_hidden_execution_fields(tmp_path):
    task, _ = load_task("drug")
    candidate = task.ids()[0]
    episode = run_episode(Config(policy="llm", budget=3, execution="perturbed"), initial=[candidate] * 3)
    assert len(episode.hidden) == 3
    episode.save(tmp_path)
    trajectory = (tmp_path / "trajectory.jsonl").read_text()
    assert "realised_id" not in trajectory
    assert "realised_params" not in trajectory
    assert "true_value" not in trajectory
    assert len((tmp_path / "hidden_truth.jsonl").read_text().splitlines()) == 3
    with pytest.raises(FileExistsError):
        episode.save(tmp_path)


def test_separate_latent_and_noise_uncertainty():
    task, _ = load_task("drug")
    model = Surrogate(task).fit([(task.ids()[0], 10.), (task.ids()[-1], 30.)])
    uncertainty = model.response(task.ids()[0])
    half_interval = (uncertainty.interval[1] - uncertainty.interval[0]) / 2
    assert half_interval == pytest.approx(1.96 * np.hypot(uncertainty.latent_sd, uncertainty.noise_sd))


def test_openai_configuration_metadata_and_seed_opt_in(monkeypatch):
    monkeypatch.setenv("EPISTEMIC_API_KEY", "test-only")
    bodies = []

    class Reply:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "{}"}}], "usage": {"total_tokens": 12},
                    "system_fingerprint": "fake-server"}

    def post(url, **kwargs):
        bodies.append(kwargs["json"])
        assert url == "https://server.invalid/v1/chat/completions"
        return Reply()

    monkeypatch.setattr("httpx.post", post)
    provider = OpenAICompatibleProvider(model="configured-open-model", base_url="https://server.invalid/v1", seed=4)
    assert provider.complete("prompt") == "{}"
    assert "seed" not in bodies[-1]
    provider.seed_supported = True
    provider.complete("prompt")
    assert bodies[-1]["seed"] == 4
    assert provider.metadata()["server_fingerprint"] == "fake-server"
    assert provider.tokens == 24


def test_missing_feedback_serializes_as_null_and_has_no_derived_claims(tmp_path):
    episode = run_episode(Config(policy="llm", budget=3, feedback="missing"))
    assert all(o.value_shown is None for o in episode.graph.observations.values())
    assert "K_best" not in episode.graph
    assert "K_calibration" not in episode.graph
    episode.save(tmp_path)
    text = (tmp_path / "trajectory.jsonl").read_text()
    assert "NaN" not in text
    assert "null" in text
    EvidenceGraph.model_validate_json((tmp_path / "graph.json").read_text())


def test_llm_hypotheses_remain_distinct_from_tool_outputs():
    payload = {
        "candidate_id": "C0000", "evidence": ["K1"], "targeted_uncertainty": "evidence",
        "prediction": 20., "rationale": "A provisional interpretation, not a measurement.",
        "claim_updates": [{
            "statement": "This report may transfer to this assay.", "scope": "drug assay only",
            "evidence": ["K1"], "discriminating_result": "An observed counterexample in matched conditions.",
            "status": "open",
        }],
    }
    episode = run_episode(Config(policy="llm", budget=1), provider=FixedProvider(json.dumps(payload)))
    assert episode.graph.claims["H1_0"].source == "model_conjecture"
    assert episode.graph.claims["H1_0"].revisions[0].round == 1
    assert len(episode.graph.observations) == 1
    assert episode.graph.observations["O1"].value_shown != 20.
    assert "H1_0" not in json.dumps(episode.trajectory[0]["state_available_to_agent"])
    assert episode.trajectory[0]["state_revision"]["new_revisions"]["H1_0"] == 1


def test_flat_view_has_identical_records_and_uncertainty_but_no_edges():
    episode = run_episode(Config(policy="random", budget=3, seed=1))
    edges = episode.graph.view(4, with_edges=True)
    flat = episode.graph.view(4, with_edges=False)
    relationships = edges.pop("relationships")
    assert relationships
    assert edges == flat


def test_paired_arms_share_frozen_metadata_and_are_repeatable():
    config = Config(policy="llm", budget=4)
    a = paired_interventions(config, rounds=2, repeats=1, direct_llm_only=True)
    b = paired_interventions(config, rounds=2, repeats=1, direct_llm_only=True)
    assert a == b
    assert set(a["arms"]) == {"true", "removed", "permuted", "contradicted"}
    for arm, rows in a["arms"].items():
        assert rows[0]["valid"]
        assert "Budget: 4" in rows[0]["prompt"]
        assert "predicted_mean" not in rows[0]["prompt"]
    assert len(a["frozen_observations"]) == 2


def test_seeded_fault_and_matched_closed_loop_controls():
    rows = closed_loop(Config(policy="random", budget=4, fault_round=2), seeds=(1,), initial=3)
    assert {row["feedback"] for row in rows} == {"true", "missing", "corrupted"}
    assert all(row["experiments"] == 4 for row in rows)
    assert rows[0]["best_true_objective_curve"][:3] == rows[1]["best_true_objective_curve"][:3]
    episode = run_episode(Config(policy="bo", budget=4, fault_round=2))
    assert [s["round"] for s in episode.hidden if s["seeded_execution_fault"]] == [2]
    assert episode.trajectory[1]["accessible_observation"]["execution"]["execution_model"] == "perturbed"


def test_exports_are_valid_svg_and_complete_audit(tmp_path):
    import xml.etree.ElementTree as ET
    episode = run_episode(Config(policy="bo", budget=3))
    export(episode, tmp_path)
    assert ET.fromstring((tmp_path / "graph.svg").read_text()).tag.endswith("svg")
    audit = (tmp_path / "audit.md").read_text()
    for label in ("state_available_to_agent", "action", "pre_experiment_prediction",
                  "accessible_observation", "state_revision", "discriminating_result", "revisions"):
        assert label in audit


def test_inspect_llm_wrapper_uses_shared_loop_offline(tmp_path):
    pytest.importorskip("inspect_ai")
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput, get_model

    from epistemic.inspect_task import experimental_design

    payload = {"candidate_id": "C0000", "evidence": ["K1"], "targeted_uncertainty": "response",
               "prediction": 20., "rationale": "Offline integration fixture."}
    model = get_model(
        "mockllm/model",
        custom_outputs=lambda *_: ModelOutput.from_content("mockllm/model", json.dumps(payload)),
    )
    logs = inspect_eval(
        experimental_design(policy="llm", budget=2, graph_dir=str(tmp_path / "graphs")),
        model=model, max_tokens=512, log_dir=str(tmp_path / "inspect"),
    )
    assert logs[0].status == "success"
    metrics = logs[0].samples[0].metadata["metrics"]
    assert metrics["experiments"] == 2
    assert metrics["model_calls"] == 2
    assert logs[0].samples[0].metadata["provider"]["max_tokens"] == 512


def test_svg_renders_actual_dependencies_not_legacy_tests_claim_links(tmp_path):
    episode = run_episode(Config(policy="bo", budget=4))
    episode.graph.decisions["D1"].claims.append("K1")
    episode.graph.link("D1", "K1", "tests", "Legacy edge: negative renderer control.")
    export(episode, tmp_path)
    svg = (tmp_path / "graph.svg").read_text()
    assert "<title>D4 depends_on K_best</title>" in svg
    assert "<title>D1 depends_on K1</title>" not in svg
    assert "<title>D1 tests K1</title>" not in svg
    assert "<title>D4 tests K_best</title>" not in svg
    assert any(e.source == "D4" and e.target == "O4" and e.kind == "tests" for e in episode.graph.edges)


def test_live_search_is_deferred_bounded_and_attributable():
    class SearchingMock(MockProvider):
        def complete(self, prompt):
            payload = json.loads(super().complete(prompt))
            payload["search_query"] = f"ABTS peroxygenase question {self.calls}"
            return json.dumps(payload)

    queries = []

    def retrieve(query):
        queries.append(query)
        claim = Claim(id="source", statement="Quoted paper evidence.", source="literature",
                      scope="Reported assay only.", reference="https://doi.org/10.1234/fixture",
                      discriminating_result="A matched assay contradicts the quoted result.")
        return {"claims": [claim.model_dump()], "records": [{"title": "Paper fixture"}], "credit_cost": "0"}

    episode = run_episode(Config(task="enzyme", policy="llm", budget=4, max_searches=2),
                          provider=SearchingMock(), literature_search=retrieve)
    assert len(queries) == 2
    assert episode.provider_metadata["calls"] == 4
    assert episode.trajectory[0]["literature_search"]["status"] == "success"
    assert "L1_1_1" not in episode.trajectory[0]["prompt"]
    assert "L1_1_1" in episode.trajectory[1]["prompt"]
    assert "Literature searches remaining: 0" in episode.trajectory[2]["prompt"]
    assert episode.trajectory[2]["literature_search"]["status"] == "budget_exhausted"
    assert episode.trajectory[3]["literature_search"]["status"] == "no_future_round"
    assert episode_metrics(episode)["literature_searches"] == 2
    assert episode.graph.claims["L1_1_1"].reference == "https://doi.org/10.1234/fixture"
    assert "L2_2_1" not in episode.graph.claims
    assert episode.trajectory[1]["literature_search"]["duplicate_sources"] == [{
        "reference": "https://doi.org/10.1234/fixture",
        "existing_claim_id": "L1_1_1",
    }]
    frozen = [c for c in episode.graph.claims.values() if c.source == "literature"]
    paired = paired_interventions(episode.config, rounds=4, repeats=1,
                                  frozen=list(episode.graph.observations.values()), frozen_evidence=frozen)
    for arm in paired["arms"].values():
        assert "L1_1_1" in arm[0]["prompt"]


def test_failed_search_consumes_budget_without_fabricating_evidence():
    class SearchingMock(MockProvider):
        def complete(self, prompt):
            payload = json.loads(super().complete(prompt))
            payload["search_query"] = "ABTS pH"
            return json.dumps(payload)

    def failing_search(query):
        raise RuntimeError("Do not propagate provider secrets or response bodies.")

    episode = run_episode(Config(policy="llm", budget=3, max_searches=1),
                          provider=SearchingMock(), literature_search=failing_search)
    assert episode.trajectory[0]["literature_search"]["error_type"] == "RuntimeError"
    assert episode.trajectory[1]["literature_search"]["status"] == "budget_exhausted"
    assert episode_metrics(episode)["literature_searches"] == 1
    assert not any(c.id.startswith("L1_") for c in episode.graph.claims.values())
    assert "secrets" not in json.dumps(episode.trajectory)
    assert episode.provider_metadata["calls"] == 3
    with pytest.raises(ValueError, match="matched fixed evidence"):
        graph_value(Config(max_searches=1))


def test_fenced_json_is_accepted_without_extra_model_calls():
    class FencedMock(MockProvider):
        def complete(self, prompt):
            return "```json\n" + super().complete(prompt) + "\n```"

    episode = run_episode(Config(policy="llm", budget=2), provider=FencedMock())
    assert len(episode.hidden) == 2
    assert episode_metrics(episode)["invalid_llm_attempts"] == 0
    assert episode.provider_metadata["calls"] == 2


def test_invalid_optional_claim_revision_does_not_discard_valid_action():
    class BadRevisionMock(MockProvider):
        def complete(self, prompt):
            payload = json.loads(super().complete(prompt))
            payload["claim_updates"] = [{
                "claim_id": "K1",
                "statement": "An invalid attempted rewrite.",
                "scope": "Invalid fixture.",
                "evidence": ["K1"],
                "discriminating_result": "No result.",
                "status": "open",
            }]
            return json.dumps(payload)

    episode = run_episode(Config(policy="llm", budget=1), provider=BadRevisionMock())
    assert len(episode.hidden) == 1
    assert episode.provider_metadata["calls"] == 1
    assert episode.trajectory[0]["llm"]["valid_output"] is True
    assert "claim_update 1 rejected" in episode.trajectory[0]["llm"]["failures"][0]
    assert episode.graph.claims["K1"].statement.startswith("The candidate outcomes")
