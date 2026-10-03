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
from epistemic.source_filter import excluded_source
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
    for name in ("graph.svg", "evidence.svg", "experiments.svg"):
        assert ET.fromstring((tmp_path / name).read_text()).tag.endswith("svg")
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
    from epistemic.graph import Edge

    episode = run_episode(Config(policy="bo", budget=4))
    episode.graph.decisions["D1"].claims.append("K1")
    episode.graph.edges.append(Edge(source="D1", target="K1", kind="tests",
                                   note="Legacy edge: negative renderer control."))
    export(episode, tmp_path)
    svg = (tmp_path / "experiments.svg").read_text()
    assert "<title>D4 depends_on K_best</title>" in svg
    assert "<title>D1 depends_on K1</title>" not in svg
    assert "<title>D1 tests K1</title>" not in svg
    assert "<title>D4 tests K_best</title>" not in svg
    evidence = (tmp_path / "evidence.svg").read_text()
    assert "supports" in evidence
    assert "contradicts" in evidence
    assert any(e.source == "D4" and e.target == "O4" and e.kind == "tests" for e in episode.graph.edges)


def test_raw_evidence_and_uncertainty_are_immutable_and_globally_identified():
    from epistemic.graph import Assumption, EvidenceRecord, UncertaintyRecord

    graph = EvidenceGraph()
    source = graph.add_evidence(EvidenceRecord(
        id="E1", title="Raw title", abstract="Full abstract", query="Exact query", reference="doi:fixture",
        metadata={"authors": [{"name": "A"}], "abstract": "Full abstract"},
    ))
    for field in ("title", "abstract", "reference", "query"):
        with pytest.raises(ValueError):
            setattr(source, field, "Changed")
    with pytest.raises(TypeError):
        source.metadata["abstract"] = "Changed"
    with pytest.raises(TypeError):
        source.metadata["authors"][0]["name"] = "Changed"
    default_source = EvidenceRecord(id="empty", title="No metadata", reference="doi:empty")
    with pytest.raises(TypeError):
        default_source.metadata["changed"] = True
    uncertainty = graph.add_uncertainty(UncertaintyRecord(
        id="U1", category="transfer", statement="Context transfer is unresolved.", scope="target assay",
        evidence=("E1",), round=2,
    ))
    with pytest.raises(ValueError):
        uncertainty.status = "bounded"
    with pytest.raises(ValueError, match="already exists"):
        graph.add_evidence(source)
    with pytest.raises(ValueError, match="already exists"):
        graph.add_uncertainty(uncertainty)
    with pytest.raises(ValueError, match="already exists"):
        graph.add_assumption(Assumption(id="E1", statement="Collision", scope="test"))
    roundtrip = EvidenceGraph.model_validate_json(graph.model_dump_json())
    assert roundtrip.evidence_records["E1"].abstract == "Full abstract"
    assert roundtrip.uncertainties["U1"].round == 2


def test_literature_propositions_keep_full_abstracts_only_in_source_metadata():
    from epistemic.amass import records_to_claims

    abstract = "RAW_ABSTRACT_SENTINEL " * 90 + " Conclusions: Matched treatment reduced response. More details."
    graph = EvidenceGraph.model_validate(records_to_claims([
        {"title": "Reported assay response", "doi": "10.1234/scoped", "abstract": abstract},
        {"title": "An exploratory mechanism", "pmid": "123456", "abstract": "LONG_BACKGROUND " * 90},
    ], query="specific timing and assay"))
    for claim in graph.claims.values():
        assert len(claim.statement) <= 280
        assert "RAW_ABSTRACT_SENTINEL" not in claim.statement
        assert "LONG_BACKGROUND" not in claim.statement
        assert claim.scope and claim.discriminating_result
        categories = {graph.uncertainties[i].category for i in claim.uncertainties}
        assert categories == {"source", "transfer", "mechanistic"}
        assert claim.unresolved_transfer_assumptions
        assert all(i in graph.evidence_records for i in claim.evidence)
    source = graph.evidence_records["E1"]
    assert source.abstract == abstract
    assert source.metadata["abstract"] == abstract
    assert source.query == "specific timing and assay"
    assert graph.claims["L1"].statement == "The source reports: Matched treatment reduced response."
    assert graph.claims["L2"].statement.startswith("The source investigates:")
    assert "RAW_ABSTRACT_SENTINEL" not in json.dumps(graph.view(1))


def test_bundle_ingestion_preserves_raw_sources_remaps_ids_and_deduplicates():
    from epistemic.amass import records_to_claims
    from epistemic.evidence import external_evidence, ingest_literature, seed_graph

    task, _ = load_task("enzyme")
    graph = seed_graph(task)
    bundle = records_to_claims([{"title": "Source", "abstract": "Full raw text", "doi": "10.1234/raw"}],
                              query="a precise question")
    added, duplicates = ingest_literature(graph, bundle, prefix="4_2", round=4)
    assert not duplicates and len(added) == 1
    claim = graph.claims["L4_2_1"]
    assert claim.revisions[0].round == 4
    assert graph.evidence_records[claim.evidence[0]].abstract == "Full raw text"
    assert {graph.uncertainties[i].category for i in claim.uncertainties} == {"source", "transfer", "mechanistic"}
    before = graph.model_dump_json()
    added, duplicates = ingest_literature(graph, bundle, prefix="5_3", round=5)
    assert not added and duplicates[0]["existing_claim_id"] == "L4_2_1"
    assert before == graph.model_dump_json()
    frozen = external_evidence(graph)
    restored = seed_graph(task)
    ingest_literature(restored, frozen)
    assert restored.evidence_records["E4_2_1"].query == "a precise question"
    assert restored.evidence_records["E4_2_1"].abstract == "Full raw text"


def test_bundle_holdout_filter_cascades_through_claims_and_uncertainty(tmp_path):
    from epistemic.amass import records_to_claims
    from epistemic.evidence import ingest_literature

    bundle = records_to_claims([
        {"title": "First source", "doi": "10.1234/first", "abstract": "First abstract"},
        {"title": "Allowed source", "doi": "10.1234/allowed", "abstract": "Allowed abstract"},
    ])
    bundle["evidence_records"]["E1"]["metadata"]["hidden_reference"] = "10.1073/pnas.1606301113"
    episode = run_episode(Config(budget=2))
    ingest_literature(episode.graph, bundle)
    assert "E1" not in episode.graph and "L1" not in episode.graph
    assert "U1_transfer" not in episode.graph
    assert "E2" in episode.graph and "L2" in episode.graph
    export(episode, tmp_path)
    for name in ("graph.json", "graph.svg", "graph.html", "evidence.svg", "experiments.svg", "audit.md"):
        assert "10.1073/pnas.1606301113" not in (tmp_path / name).read_text()


def test_all_evidence_relationships_render_with_distinct_semantics(tmp_path):
    import xml.etree.ElementTree as ET

    from epistemic.amass import records_to_claims
    from epistemic.evidence import ingest_literature

    episode = run_episode(Config(policy="bo", budget=4))
    abstract = "RAW_ABSTRACT_ONLY_IN_METADATA " * 30
    ingest_literature(episode.graph, records_to_claims([
        {"title": "Timing changes the response", "doi": "10.1234/timing", "abstract": abstract},
    ]))
    episode.graph.link("O1", "L1", "contradicts", "Test fixture: contrasting observation.")
    episode.graph.link("E1", "L1", "not_transferable", "Test fixture: incompatible timing.")
    export(episode, tmp_path)
    for name in ("graph.svg", "evidence.svg", "experiments.svg"):
        assert ET.fromstring((tmp_path / name).read_text()).tag.endswith("svg")
    root = ET.fromstring((tmp_path / "evidence.svg").read_text())
    ns = {"s": "http://www.w3.org/2000/svg"}
    visible = " ".join("".join(node.itertext()) for node in root.findall(".//s:text", ns))
    assert "RAW_ABSTRACT_ONLY_IN_METADATA" not in visible
    assert abstract in episode.graph.evidence_records["E1"].metadata["abstract"]
    for kind in ("supports", "qualifies", "contradicts", "not_transferable"):
        assert any(kind in element.get("class", "").split() for element in root.findall(".//s:path", ns))
        assert f"arrow-{kind}" in (tmp_path / "evidence.svg").read_text()
    for category in ("source", "transfer", "mechanistic", "response/model", "execution"):
        assert root.findall(f'.//s:g[@data-uncertainty="{category}"]', ns)
    experiment = (tmp_path / "experiments.svg").read_text()
    assert "Experiment ·" in experiment
    assert "Claim update" in experiment
    assert "experiment-D1-update-K_best" in experiment
    assert "<title>D4 depends_on K_best</title>" in experiment
    assert "<title>D4 tests O4</title>" in experiment
    assert episode.graph.claims["K_best"].revisions[0].round == 1
    for claim_id in ("K_best", "K_calibration"):
        assert {episode.graph.uncertainties[i].category
                for i in episode.graph.claims[claim_id].uncertainties} == {"response", "model", "execution"}
    combined = (tmp_path / "graph.svg").read_text()
    assert "Claims, evidence and refutations" in combined and "Experiments and reasoning" in combined


def test_html_export_is_self_contained_and_keeps_ids_separate(tmp_path):
    import re

    episode = run_episode(Config(policy="bo", budget=4))
    export(episode, tmp_path)
    content = (tmp_path / "graph.html").read_text()
    assert "__EPISTEMIC_BODY__" not in content
    assert "data-a-inspector" in content and "data-related-records" in content
    assert "data-a-zoom" in content and "data-record-id" in content
    assert "data-episode" in content
    ids = re.findall(r'(?<![\w-])id="([^"]+)"', content)
    assert len(ids) == len(set(ids))
    assert not re.search(r'<(?:script|img|iframe)[^>]+\bsrc\s*=', content, re.I)
    assert not re.search(r'<link[^>]+rel=["\']stylesheet', content, re.I)
    assert not re.search(r'\b(?:fetch|XMLHttpRequest|WebSocket|EventSource)\s*\(', content)
    with pytest.raises(ValueError, match="tests links"):
        episode.graph.link("D1", "K1", "tests")


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


@pytest.mark.parametrize("task_name", ["enzyme", "drug"])
@pytest.mark.parametrize("failure", ["malformed", "unknown_evidence", "protected_claim", "override", "provider"])
def test_evidence_failures_cannot_change_or_stop_bo(task_name, failure):
    class BrokenEvidence(MockProvider):
        def complete(self, prompt):
            self.calls += 1
            assert prompt.startswith("EVIDENCE_ONLY:")
            assert "FULL_CANDIDATE_POOL" not in prompt
            if failure == "provider":
                raise RuntimeError("fixture provider failure")
            if failure == "malformed":
                return "not JSON"
            if failure == "override":
                return json.dumps({"candidate_id": "invented", "claim_updates": []})
            update = {
                "statement": "An unsupported interpretation.",
                "scope": "Fixture assay.",
                "evidence": ["missing"] if failure == "unknown_evidence" else ["K1"],
                "discriminating_result": "A matched counterexample.",
            }
            if failure == "protected_claim":
                update["claim_id"] = "K1"
            return json.dumps({"claim_updates": [update]})

    common = {"task": task_name, "budget": 8, "seed": 11, "execution": "perturbed"}
    reference = run_episode(Config(policy="bo", **common))
    assisted = run_episode(Config(policy="bo_evidence", max_searches=2, **common), provider=BrokenEvidence())
    assert assisted.hidden == reference.hidden
    assert [d.candidate_id for d in assisted.decisions] == [d.candidate_id for d in reference.decisions]
    assert assisted.graph.model_dump() == reference.graph.model_dump() | {
        "decisions": {key: value.model_dump() for key, value in assisted.graph.decisions.items()},
    }
    assert all(d.prediction_source != "llm" for d in assisted.decisions)
    metrics = episode_metrics(assisted)
    assert metrics["completed_budget"]
    assert metrics["experiments"] == 8
    assert metrics["invalid_llm_selections"] == 0
    assert 1 <= metrics["rejected_evidence_outputs"] <= 3
    assert 1 <= metrics["model_calls"] <= 3
    assert assisted.final_selection() == reference.final_selection()


def test_evidence_bo_uses_retrieved_sources_without_claiming_they_select_actions():
    class EvidenceMock(MockProvider):
        def complete(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return json.dumps({"search_query": "Assay-specific mechanism", "claim_updates": []})
            return json.dumps({"claim_updates": [{
                "statement": "This source may transfer, pending a matched assay.",
                "scope": "The cited assay only; benchmark transfer is unresolved.",
                "evidence": ["L1_1_1"],
                "discriminating_result": "A matched assay with an inconsistent result.",
            }]})

    def retrieve(query):
        source = Claim(id="source", statement="A quoted assay result.", source="literature",
                       reference="https://doi.org/10.1234/fixture", scope="The reported assay only.",
                       discriminating_result="An inconsistent result in matched conditions.")
        return {"claims": [source.model_dump()], "records": [{"title": "Assay fixture"}]}

    reference = run_episode(Config(policy="bo", budget=8, seed=3))
    assisted = run_episode(Config(policy="bo_evidence", budget=8, seed=3, max_searches=2),
                           provider=EvidenceMock(), literature_search=retrieve)
    assert assisted.hidden == reference.hidden
    assert episode_metrics(assisted)["completed_budget"]
    assert "L1_1_1" in assisted.graph.claims
    assert assisted.graph.claims["H2_0"].source == "model_conjecture"
    assert any(e.source == "L1_1_1" and e.target == "H2_0" and e.kind == "supports"
               for e in assisted.graph.edges)
    assert not any(e.source in assisted.graph.decisions and e.target.startswith(("H", "L"))
                   and e.kind == "depends_on" for e in assisted.graph.edges)
    assert all("FULL_CANDIDATE_POOL" not in (step["prompt"] or "") for step in assisted.trajectory)


def test_inspect_evidence_bo_completes_both_tasks_despite_bad_model_output(tmp_path):
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput, get_model

    from epistemic.inspect_task import experimental_design

    model = get_model("mockllm/model", custom_outputs=lambda *_: ModelOutput.from_content(
        "mockllm/model", "invalid evidence fixture"))
    logs = inspect_eval(
        [experimental_design(task_name=name, budget=8, graph_dir=str(tmp_path / "graphs"))
         for name in ("enzyme", "drug")],
        model=model, max_tokens=512, max_tasks=2, log_dir=str(tmp_path / "inspect"),
    )
    assert len(logs) == 2
    for log in logs:
        assert log.status == "success"
        metadata = log.samples[0].metadata
        assert metadata["metrics"]["policy"] == "bo_evidence"
        assert metadata["metrics"]["completed_budget"]
        assert metadata["metrics"]["experiments"] == 8
        assert metadata["metrics"]["invalid_llm_selections"] == 0
        assert 1 <= metadata["metrics"]["model_calls"] <= 3
        assert metadata["provider"]["max_tokens"] == 512
    for path in (tmp_path / "graphs").glob("*/trajectory.jsonl"):
        text = path.read_text()
        assert "FULL_CANDIDATE_POOL" not in text
        assert "true_value" not in text
        assert "realised_params" not in text


@pytest.mark.parametrize("identifier", [
    "Prediction of multidimensional drug dose responses based on measurements of drug pairs",
    "PREDICTION OF MULTIDIMENSIONAL DRUG DOSE-RESPONSES BASED ON MEASUREMENTS OF DRUG PAIRS",
    "doi:10.1073/PNAS.1606301113",
    "https://doi.org/10.1073%2Fpnas.1606301113",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC5027409/",
    "https://pubmed.ncbi.nlm.nih.gov/27562164/",
    {"pmid": 27562164, "title": "No DOI supplied"},
    {"pmcid": "PMC5027409", "title": "No DOI supplied"},
])
def test_source_paper_holdout_matches_title_and_identifier_aliases(identifier):
    assert excluded_source(identifier)
    assert not excluded_source("A separate study of taxol-cisplatin antagonism")


def test_amass_filters_the_holdout_before_results_or_claims_reach_the_agent(monkeypatch):
    import asyncio

    from epistemic.amass import records_to_claims, search_result

    blocked = {"title": "Prediction of multidimensional drug dose responses based on measurements of drug pairs",
               "doi": "10.1073/pnas.1606301113", "abstract": "PRIVATE_HELD_OUT_RESULT"}
    allowed = {"title": "A different assay", "doi": "10.1234/allowed", "abstract": "Permitted fixture"}
    calls = []

    class Reply:
        headers = {"X-Amass-Credit-Cost": "5"}

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [blocked, allowed]}

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, **kwargs):
            calls.append(kwargs["params"])
            return Reply()

    monkeypatch.setenv("AMASS_API_KEY", "fixture-only")
    monkeypatch.setattr("httpx.AsyncClient", Client)
    result = asyncio.run(search_result("a targeted assay question"))
    assert result["records"] == [allowed]
    assert result["credit_cost"] == "5"
    assert "PRIVATE_HELD_OUT_RESULT" not in json.dumps(result)
    claims = records_to_claims([blocked, allowed])
    assert len(claims["claims"]) == 1
    assert len(claims["evidence_records"]) == 1
    assert not excluded_source(claims)
    with pytest.raises(ValueError, match="excluded"):
        asyncio.run(search_result("https://doi.org/10.1073/pnas.1606301113"))
    assert len(calls) == 1


def test_holdout_is_absent_from_public_task_fixed_and_frozen_evidence(tmp_path):
    task, evaluator = load_task("drug")
    assert "10.1073/pnas.1606301113" in evaluator.provenance
    public = json.dumps(asdict(task), default=str) + task.briefing(8)
    assert not excluded_source(public)
    assert "Zimmer" not in public and "Tendler" not in public
    blocked = Claim(id="excluded", statement="A result quoted from the source paper.", source="literature",
                    scope="The source assay.", reference="https://doi.org/10.1073/pnas.1606301113",
                    discriminating_result="A matched counterexample.")
    allowed = Claim(id="allowed", statement="An independent assay result.", source="literature",
                    scope="The independent assay.", reference="https://doi.org/10.1234/allowed",
                    discriminating_result="A matched counterexample.")
    bundle = tmp_path / "evidence.json"
    bundle.write_text(json.dumps([blocked.model_dump(), allowed.model_dump()]))
    graph = seed_graph(task, evidence_file=str(bundle))
    assert "allowed" in graph and "excluded" not in graph
    assert not excluded_source(graph.model_dump())
    rebuilt, _, _ = rebuild(task, [], 0, frozen_evidence=[blocked, allowed])
    assert "allowed" in rebuilt and "excluded" not in rebuilt
    episode = run_episode(Config(policy="bo_evidence", budget=4, evidence_file=str(bundle)))
    episode.save(tmp_path / "episode")
    assert "10.1073/pnas.1606301113" in (tmp_path / "episode/hidden_provenance.json").read_text()
    for filename in ("trajectory.jsonl", "graph.json"):
        assert not excluded_source((tmp_path / "episode" / filename).read_text())


def test_live_holdout_filter_removes_raw_records_and_preserves_allowed_sources():
    class SearchingEvidence(MockProvider):
        def complete(self, prompt):
            self.calls += 1
            assert not excluded_source(prompt)
            return json.dumps({"search_query": "Independent dose mechanism"})

    def retrieve(query):
        blocked = Claim(id="blocked", statement="PRIVATE_HELD_OUT_RESULT", source="literature",
                        scope="Source assay.", reference="https://doi.org/10.1073/pnas.1606301113",
                        discriminating_result="Counterexample.")
        allowed = Claim(id="allowed", statement="An independent result.", source="literature",
                        scope="Independent assay.", reference="https://doi.org/10.1234/allowed",
                        discriminating_result="Counterexample.")
        return {"claims": [blocked.model_dump(), allowed.model_dump()],
                "records": [{"pmcid": "PMC5027409", "abstract": "PRIVATE_HELD_OUT_RESULT"},
                            {"title": "Independent result"}]}

    episode = run_episode(Config(policy="bo_evidence", budget=8, max_searches=2),
                          provider=SearchingEvidence(), literature_search=retrieve)
    assert episode_metrics(episode)["completed_budget"]
    assert any(c.reference == "https://doi.org/10.1234/allowed" for c in episode.graph.claims.values())
    assert not excluded_source(episode.graph.model_dump())
    audit = json.dumps(episode.trajectory)
    assert not excluded_source(audit)
    assert "PRIVATE_HELD_OUT_RESULT" not in audit


def test_model_cannot_add_the_excluded_paper_back_to_the_evidence_graph():
    class RecallingEvidence(MockProvider):
        def complete(self, prompt):
            self.calls += 1
            return json.dumps({"claim_updates": [{
                "statement": "The source 10.1073/pnas.1606301113 supplies PRIVATE_HELD_OUT_RESULT.",
                "scope": "Source assay.", "evidence": ["K1"], "discriminating_result": "Counterexample.",
            }]})

    episode = run_episode(Config(policy="bo_evidence", budget=8, max_searches=2), provider=RecallingEvidence())
    assert episode_metrics(episode)["completed_budget"]
    assert not any(c.id.startswith("H") for c in episode.graph.claims.values())
    assert not excluded_source(episode.graph.model_dump())
    audit = json.dumps(episode.trajectory)
    assert not excluded_source(audit)
    assert "PRIVATE_HELD_OUT_RESULT" not in audit
