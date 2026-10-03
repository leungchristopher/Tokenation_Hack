import asyncio
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from epistemic.amass import records_to_claims, search_result
from epistemic.evidence import external_evidence, ingest_literature, seed_graph
from epistemic.execution import PerfectExecution, PerturbedExecution
from epistemic.graph import Assumption, Claim, EvidenceGraph, EvidenceRecord, Observation, UncertaintyRecord
from epistemic.loop import Config, Episode, run_episode
from epistemic.metrics import episode_metrics
from epistemic.provider import MockProvider, get_provider
from epistemic.render import actions_graph, export, to_html
from epistemic.source_filter import excluded_source
from epistemic.surrogate import Surrogate
from epistemic.tasks import TASKS, Evaluator, load_task


@pytest.mark.parametrize("name,count", [("enzyme", 814), ("drug", 512)])
def test_task_contract_and_hidden_labels(name, count):
    task, evaluator = load_task(name)
    assert set(TASKS) == {"enzyme", "drug"}
    assert len(task.ids()) == len(set(task.ids())) == count
    assert set(task.candidates) == {"candidate_id", *task.names}
    assert not hasattr(task, "_outcomes")
    assert not excluded_source(json.dumps(asdict(task), default=str) + task.briefing(8))
    assert evaluator.truth(task.ids()[0]) >= 0
    assert not task.noise.replicates
    if name == "drug":
        assert all(len(task.candidates[n].unique()) == 8 for n in task.names)
        assert all(p.unit == "micromolar" and p.log for p in task.parameters)
        assert "no synergy" in " ".join(task.limitations)


@pytest.mark.parametrize("task_name,values,index", [
    ("enzyme", [0.0, 8.0, 5.0], 1), ("drug", [4.0, 0.0, 3.0], 1), ("drug", [2.0, 2.0, 9.0], 0),
])
def test_greedy_final_selection_never_reads_hidden_truth(task_name, values, index):
    task, _ = load_task(task_name)
    graph = EvidenceGraph()
    for i, value in enumerate(values):
        graph.add_observation(Observation(
            id=f"O{i}", round=i + 1, candidate_id=task.ids()[i],
            intended_params=task.params_of(task.ids()[i]), execution={},
            value_shown=value, outcome_unit=task.outcome_unit,
        ))
    episode = Episode(Config(task=task_name), task, graph, hidden=[{"true_value": -100.0}])
    assert episode.final_selection() == task.ids()[index]
    result = episode.final_result()
    assert result["observation_id"] == f"O{index}" and result["value_shown"] == values[index]
    assert "not a proven optimum" in result["uncertainty"]
    assert "No hidden dataset means or GP forecasts" in result["uncertainty"]


def test_no_observed_result_means_no_final_answer():
    task, _ = load_task("drug")
    graph = EvidenceGraph()
    graph.add_observation(Observation(id="O1", round=1, candidate_id=task.ids()[0],
                                     intended_params={}, execution={}, value_shown=None, outcome_unit="test"))
    episode = Episode(Config(), task, graph)
    assert episode.final_selection() is None and episode.final_result() is None


def test_execution_reports_do_not_reveal_realised_conditions():
    task, _ = load_task("drug")
    cid = task.ids()[-1]
    assert PerfectExecution(task).run(cid, np.random.default_rng(2)).realised_id == cid
    perturbed = PerturbedExecution(task, relative_sd=8).run(cid, np.random.default_rng(2))
    assert perturbed.clipped and perturbed.realised_id in task.ids()
    assert not {"realised_id", "realised_params"} & set(perturbed.accessible())
    for name, (lo, hi) in task.bounds().items():
        assert lo <= perturbed.realised_params[name] <= hi
    before = task.nearest(perturbed.realised_params)
    task.candidates = task.candidates.sample(frac=1, random_state=4)
    assert task.nearest(perturbed.realised_params) == before


def test_records_are_deeply_immutable_and_ids_are_global():
    graph = EvidenceGraph()
    source = graph.add_evidence(EvidenceRecord(
        id="E1", title="Raw title", abstract="Full abstract", query="Exact query", reference="doi:fixture",
        metadata={"authors": [{"name": "A"}]},
    ))
    for field in ("title", "abstract", "reference", "query"):
        with pytest.raises(ValidationError):
            setattr(source, field, "Changed")
    with pytest.raises(TypeError):
        source.metadata["authors"][0]["name"] = "Changed"
    uncertainty = graph.add_uncertainty(UncertaintyRecord(
        id="U1", category="transfer", statement="Context transfer is unresolved.", scope="target assay",
        evidence=("E1",),
    ))
    with pytest.raises(ValidationError):
        uncertainty.status = "bounded"
    with pytest.raises(ValueError):
        graph.add_assumption(Assumption(id="E1", statement="Collision", scope="test"))
    assert EvidenceGraph.model_validate_json(graph.model_dump_json()).evidence_records["E1"].abstract == "Full abstract"


def test_observations_and_claim_revisions_preserve_originals():
    graph = EvidenceGraph()
    observation = graph.add_observation(Observation(
        id="O1", round=1, candidate_id="C0", intended_params={"x": 1.0},
        execution={"report": {"x": 0.9}}, value_shown=4.0, outcome_unit="test",
    ))
    with pytest.raises(ValidationError):
        observation.value_shown = 5.0
    with pytest.raises(TypeError):
        observation.execution["report"]["x"] = 2.0
    claim = graph.add_claim(Claim(id="K", statement="A conjecture", scope="test", source="model_conjecture",
                                 evidence=["O1"], discriminating_result="A counterexample"))
    original = claim.revisions[0].model_dump()
    graph.revise("K", 2, "contradicted", "A counterexample", ["O1"], "Revised conjecture")
    assert claim.revisions[0].model_dump() == original and len(claim.revisions) == 2
    with pytest.raises(KeyError):
        graph.link("invented", "K", "supports")


@pytest.mark.parametrize("task_name", ["enzyme", "drug"])
@pytest.mark.parametrize("acquisition", ["ei", "random"])
def test_same_loop_is_seeded_and_predictions_remain_numerical(task_name, acquisition):
    config = Config(task=task_name, acquisition=acquisition, budget=4, seed=11, execution="perturbed")
    a, b = run_episode(config), run_episode(config)
    assert a.trajectory == b.trajectory and a.hidden == b.hidden
    assert a.graph.model_dump() == b.graph.model_dump()
    assert all(d.prediction_source in ("numerical_model", "unavailable") for d in a.decisions)
    assert episode_metrics(a) == episode_metrics(b)
    assert a.provider_metadata["calls"] == 0


@pytest.fixture(scope="module")
def episode():
    return run_episode(Config(budget=5, seed=11, execution="perturbed"))


@pytest.mark.parametrize("failure", ["malformed", "unknown_evidence", "protected_claim", "override", "provider"])
def test_bad_evidence_cannot_change_or_stop_gp_actions(episode, failure):
    class BrokenEvidence(MockProvider):
        def complete(self, prompt):
            self.calls += 1
            assert prompt.startswith("EVIDENCE_ONLY:") and "FULL_CANDIDATE_POOL" not in prompt
            if failure == "provider":
                raise RuntimeError("DO_NOT_LOG_PROVIDER_SECRETS")
            if failure == "malformed":
                return "not JSON"
            if failure == "override":
                return json.dumps({"candidate_id": "invented"})
            update = {"statement": "An interpretation.", "scope": "Fixture assay",
                      "evidence": ["missing"] if failure == "unknown_evidence" else ["K1"],
                      "discriminating_result": "A matched counterexample."}
            if failure == "protected_claim":
                update["claim_id"] = "K1"
            return json.dumps({"claim_updates": [update]})

    assisted = run_episode(replace(episode.config, max_searches=1), provider=BrokenEvidence())
    assert assisted.hidden == episode.hidden
    assert [d.candidate_id for d in assisted.decisions] == [d.candidate_id for d in episode.decisions]
    assert assisted.final_result() == episode.final_result()
    metrics = episode_metrics(assisted)
    assert metrics["completed_budget"] and 1 <= metrics["model_calls"] <= 3
    assert 1 <= metrics["rejected_evidence_outputs"] <= 3
    assert "DO_NOT_LOG_PROVIDER_SECRETS" not in json.dumps(assisted.trajectory)


def test_successful_evidence_updates_are_annotations_not_acquisition_inputs(episode):
    class EvidenceFixture(MockProvider):
        def complete(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return json.dumps({"search_query": "Assay-specific mechanism"})
            return json.dumps({"claim_updates": [{
                "statement": "Transfer remains unresolved pending a matched assay.",
                "scope": "The cited assay only", "evidence": ["L1_1_1"],
                "discriminating_result": "A matched assay with an inconsistent result.",
            }]})

    assisted = run_episode(replace(episode.config, max_searches=1), provider=EvidenceFixture(),
                           literature_search=lambda query: records_to_claims(
                               [{"title": "Independent assay", "doi": "10.1234/fixture", "abstract": "Assay text"}],
                               query=query))
    assert assisted.hidden == episode.hidden
    assert assisted.graph.claims["H2_0"].source == "model_conjecture"
    assert any(e.source == "L1_1_1" and e.target == "H2_0" and e.kind == "supports"
               for e in assisted.graph.edges)
    assert not any(e.source in assisted.graph.decisions and e.target.startswith(("H", "L"))
                   and e.kind == "depends_on" for e in assisted.graph.edges)
    assert episode_metrics(assisted)["literature_searches"] == 1


def test_search_failures_are_bounded_and_nonfatal():
    class SearchingFixture(MockProvider):
        def complete(self, prompt):
            self.calls += 1
            return json.dumps({"search_query": "Assay discrepancy"})

    def failing_search(query):
        raise RuntimeError("DO_NOT_LOG_BACKEND_SECRET")

    episode = run_episode(Config(budget=5, max_searches=1), provider=SearchingFixture(),
                          literature_search=failing_search)
    assert episode_metrics(episode)["literature_searches"] == 1
    assert episode_metrics(episode)["completed_budget"]
    assert episode.trajectory[0]["literature_search"]["error_type"] == "RuntimeError"
    assert "DO_NOT_LOG_BACKEND_SECRET" not in json.dumps(episode.trajectory)
    assert not any(c.id.startswith("L1_") for c in episode.graph.claims.values())


def test_evidence_and_flat_switches_keep_identical_experiments(episode):
    for config in (replace(episode.config, with_evidence=False), replace(episode.config, with_edges=False)):
        controlled = run_episode(config, provider=MockProvider())
        assert controlled.hidden == episode.hidden and controlled.final_result() == episode.final_result()
        if not config.with_edges:
            assert all("relationships" not in step["state_available_to_agent"] for step in controlled.trajectory)
        if not config.with_evidence:
            assert controlled.provider_metadata["calls"] == 0


def test_custom_domain_uses_the_same_loop_and_its_own_evaluator():
    task, _ = load_task("drug")
    task = replace(task, name="custom", candidates=task.candidates.head(3))
    evaluator = Evaluator(task, pd.DataFrame({"candidate_id": task.ids(), "outcome": [4., 2., 3.], "spread": [0.] * 3}))
    episode = run_episode(Config(task="unregistered", budget=3, observation_noise=False), domain=(task, evaluator))
    assert episode.final_result()["value_shown"] == 2.0
    assert episode_metrics(episode)["final_selection_regret"] == 0.0
    assert all("disabled" in u.statement for u in episode.graph.uncertainties.values() if u.category == "response")


def test_gp_separates_latent_noise_and_predictive_interval():
    task, _ = load_task("drug")
    model = Surrogate(task).fit([(task.ids()[0], 10.), (task.ids()[-1], 30.)])
    prediction = model.response(task.ids()[0])
    assert (prediction.interval[1] - prediction.interval[0]) / 2 == pytest.approx(
        1.96 * np.hypot(prediction.latent_sd, prediction.noise_sd))


def test_source_metadata_is_separate_from_concise_claims():
    abstract = "RAW_ABSTRACT_SENTINEL " * 90 + " Conclusions: Matched treatment reduced response. More details."
    graph = EvidenceGraph.model_validate(records_to_claims([
        {"title": "Reported response", "doi": "10.1234/scoped", "abstract": abstract},
    ], query="specific assay timing"))
    claim = graph.claims["L1"]
    assert len(claim.statement) <= 280 and "RAW_ABSTRACT_SENTINEL" not in claim.statement
    assert claim.scope and claim.discriminating_result and claim.unresolved_transfer_assumptions
    assert {graph.uncertainties[i].category for i in claim.uncertainties} == {"source", "transfer", "mechanistic"}
    assert graph.evidence_records["E1"].abstract == graph.evidence_records["E1"].metadata["abstract"] == abstract
    assert graph.evidence_records["E1"].query == "specific assay timing"
    assert "RAW_ABSTRACT_SENTINEL" not in json.dumps(graph.view(1))
    restored = seed_graph(load_task("enzyme")[0])
    ingest_literature(restored, graph.model_dump(), prefix="4_2", round=4)
    before = restored.model_dump_json()
    added, duplicates = ingest_literature(restored, graph.model_dump(), prefix="5_3", round=5)
    assert not added and duplicates and before == restored.model_dump_json()
    assert external_evidence(restored)["evidence_records"]["E4_2_1"]["abstract"] == abstract


@pytest.mark.parametrize("identifier", [
    "Prediction of multidimensional drug dose responses based on measurements of drug pairs",
    "PREDICTION OF MULTIDIMENSIONAL DRUG DOSE-RESPONSES BASED ON MEASUREMENTS OF DRUG PAIRS",
    "doi:10.1073/PNAS.1606301113", "https://doi.org/10.1073%2Fpnas.1606301113",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC5027409/", "https://pubmed.ncbi.nlm.nih.gov/27562164/",
    {"pmid": 27562164}, {"pmcid": "PMC5027409"},
])
def test_holdout_normalization(identifier):
    assert excluded_source(identifier)
    assert not excluded_source("An independent study of taxol-cisplatin antagonism")


def test_holdout_filter_removes_sources_and_dependent_claims(episode, tmp_path):
    episode = replace(episode, graph=EvidenceGraph.model_validate_json(episode.graph.model_dump_json()))
    bundle = records_to_claims([
        {"title": "First source", "doi": "10.1234/first", "abstract": "First text"},
        {"title": "Allowed source", "doi": "10.1234/allowed", "abstract": "Second text"},
    ])
    bundle["evidence_records"]["E1"]["metadata"]["hidden_reference"] = "10.1073/pnas.1606301113"
    ingest_literature(episode.graph, bundle)
    assert "E1" not in episode.graph and "L1" not in episode.graph and "U1_transfer" not in episode.graph
    assert "E2" in episode.graph and "L2" in episode.graph
    export(episode, tmp_path)
    for name in ("graph.json", "actions.json", "actions.svg", "graph.html", "audit.md", "final.json"):
        assert not excluded_source((tmp_path / name).read_text())
    assert excluded_source((tmp_path / "hidden_provenance.json").read_text())


def test_retrieval_filters_before_returning_agent_results(monkeypatch):
    allowed = {"title": "Independent assay", "doi": "10.1234/allowed", "abstract": "Permitted text"}
    blocked = {"title": "Held out", "doi": "10.1073/pnas.1606301113", "abstract": "PRIVATE_HELD_OUT_RESULT"}

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
            return Reply()

    monkeypatch.setenv("AMASS_API_KEY", "fixture-only")
    monkeypatch.setattr("httpx.AsyncClient", Client)
    result = asyncio.run(search_result("Assay discrepancy"))
    assert result["records"] == [allowed] and result["credit_cost"] == "5"
    assert "PRIVATE_HELD_OUT_RESULT" not in json.dumps(result)
    assert len(records_to_claims([blocked, allowed])["evidence_records"]) == 1
    with pytest.raises(ValueError):
        asyncio.run(search_result("doi:10.1073/pnas.1606301113"))


def test_all_graph_views_and_greedy_final_selection_are_exported(episode, tmp_path):
    episode = replace(episode, graph=EvidenceGraph.model_validate_json(episode.graph.model_dump_json()))
    abstract = "RAW_ABSTRACT_ONLY_IN_METADATA " * 30
    ingest_literature(episode.graph, records_to_claims([
        {"title": "Assay timing", "doi": "10.1234/timing", "abstract": abstract},
    ]))
    episode.graph.link("O1", "L1", "contradicts", "Contrasting observation")
    episode.graph.link("E1", "L1", "not_transferable", "Incompatible timing")
    export(episode, tmp_path)
    for name in ("graph.svg", "evidence.svg", "experiments.svg", "actions.svg"):
        assert ET.fromstring((tmp_path / name).read_text()).tag.endswith("svg")
    root = ET.fromstring((tmp_path / "evidence.svg").read_text())
    ns = {"s": "http://www.w3.org/2000/svg"}
    visible = " ".join("".join(n.itertext()) for n in root.findall(".//s:text", ns))
    assert "RAW_ABSTRACT_ONLY_IN_METADATA" not in visible
    for kind in ("supports", "qualifies", "contradicts", "not_transferable"):
        assert any(kind in n.get("class", "").split() for n in root.findall(".//s:path", ns))
    experiments = (tmp_path / "experiments.svg").read_text()
    assert "<title>D4 depends_on K_best</title>" in experiments
    assert "<title>D4 tests K_best</title>" not in experiments
    assert "<title>D4 tests O4</title>" in experiments
    with pytest.raises(ValueError, match="tests links"):
        episode.graph.link("D1", "K1", "tests")
    actions = actions_graph(episode)
    assert len(actions["nodes"]) == len(episode.decisions) + 1
    assert actions["nodes"][-1]["kind"] == "selection"
    assert actions["nodes"][-1]["candidate_id"] == episode.final_selection()
    assert actions["nodes"][1]["forecast"] is not None
    assert any(e["target"] == "Final" and "Greedy" in e["reason"] for e in actions["edges"])
    assert json.loads((tmp_path / "final.json").read_text()) == episode.final_result()
    assert "true_value" not in (tmp_path / "trajectory.jsonl").read_text()
    with pytest.raises(FileExistsError):
        episode.save(tmp_path)


def test_html_is_self_contained_accessible_and_ids_are_unique(episode):
    content = to_html(episode)
    assert "__EPISTEMIC_BODY__" not in content
    for marker in ("data-a-inspector", "data-related-records", "data-a-zoom", "Actions and reasoning",
                   "not a proven optimum", "data-record-id", "role=\"button\""):
        assert marker in content
    ids = re.findall(r'(?<![\w-])id="([^"]+)"', content)
    assert len(ids) == len(set(ids))
    assert not re.search(r'<(?:script|img|iframe)[^>]+\bsrc\s*=', content, re.I)
    assert not re.search(r'<link[^>]+rel=["\']stylesheet', content, re.I)
    assert not re.search(r'\b(?:fetch|XMLHttpRequest|WebSocket|EventSource)\s*\(', content)
    assert not excluded_source(content)
    empty = Episode(Config(), episode.task, EvidenceGraph())
    assert "No experiments or observed final answer." in to_html(empty)


def test_optional_inspect_adapter_imports_without_running_evaluations():
    from epistemic.inspect_task import experimental_design
    assert experimental_design(budget=1).solver


def test_provider_default_never_calls_a_model():
    assert get_provider().name == "none"
    assert MockProvider().complete("EVIDENCE_ONLY:") == '{"claim_updates": []}'
    with pytest.raises(KeyError):
        get_provider("old_selection_controller")
