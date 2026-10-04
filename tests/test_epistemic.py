import asyncio
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from scipy.stats import norm

from epistemic.amass import records_to_claims, search_result
from epistemic.graph import Assumption, EvidenceGraph, EvidenceRecord, Observation, UncertaintyRecord
from epistemic.literature import Calibrations, Hypotheses, ingest_literature, seed_graph
from epistemic.loop import Config, Episode, run_episode
from epistemic.metrics import episode_metrics
from epistemic.provider import MockProvider, get_provider
from epistemic.render import actions_graph, export, to_html, to_svg
from epistemic.source_filter import excluded_source
from epistemic.surrogate import NumericalPrior, Surrogate
from epistemic.tasks import DRUG_NOISE_CV, TASKS, Evaluator, load_task


@pytest.fixture(scope="module")
def episode():
    return run_episode(Config(budget=5, seed=11))


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
        assert all(parameter.unit == "micromolar" and parameter.log for parameter in task.parameters)
        assert "no synergy" in " ".join(task.limitations)


def test_drug_noise_uses_configurable_relative_cv():
    assert DRUG_NOISE_CV == 0.05
    task, evaluator = load_task("drug", noise_cv=0.05)
    assert task.noise.kind == "relative_cv"
    for candidate_id in (task.ids()[0], task.ids()[100], task.ids()[-1]):
        assert evaluator.spread(candidate_id) == pytest.approx(0.05 * evaluator.truth(candidate_id))

    default_task, default_evaluator = load_task("drug")
    candidate_id = default_task.ids()[100]
    assert default_evaluator.spread(candidate_id) == pytest.approx(
        DRUG_NOISE_CV * default_evaluator.truth(candidate_id)
    )


def test_noise_cv_is_only_configurable_for_drug():
    with pytest.raises(ValueError):
        load_task("enzyme", noise_cv=0.1)


@pytest.mark.parametrize("task_name,values,index", [
    ("enzyme", [0.0, 8.0, 5.0], 1),
    ("drug", [4.0, 0.0, 3.0], 1),
    ("drug", [2.0, 2.0, 9.0], 0),
])
def test_greedy_final_selection_never_reads_hidden_truth(task_name, values, index):
    task, _ = load_task(task_name)
    graph = EvidenceGraph()
    for round_number, value in enumerate(values, 1):
        graph.add_observation(Observation(
            id=f"O{round_number}", round=round_number, candidate_id=task.ids()[round_number - 1],
            intended_params=task.params_of(task.ids()[round_number - 1]), value_shown=value,
            outcome_unit=task.outcome_unit,
        ))
    episode = Episode(Config(task=task_name), task, graph, hidden=[{"true_value": -100.0}])
    assert episode.final_selection() == task.ids()[index]
    result = episode.final_result()
    assert result["observation_id"] == f"O{index + 1}" and result["value_shown"] == values[index]
    assert "not a proven optimum" in result["uncertainty"]
    assert "predictions do not determine final selection" in result["uncertainty"]


def test_no_observed_result_means_no_final_answer():
    task, _ = load_task("drug")
    graph = EvidenceGraph()
    graph.add_observation(Observation(
        id="O1", round=1, candidate_id=task.ids()[0], intended_params={},
        value_shown=None, outcome_unit=task.outcome_unit,
    ))
    episode = Episode(Config(), task, graph)
    assert episode.final_selection() is None and episode.final_result() is None


def test_records_are_deeply_immutable_and_ids_are_global():
    graph = EvidenceGraph()
    source = graph.add_evidence(EvidenceRecord(
        id="E1", title="Raw title", abstract="Full abstract", query="Exact query",
        reference="doi:fixture", metadata={"authors": [{"name": "A"}]},
    ))
    for field in ("title", "abstract", "reference", "query"):
        with pytest.raises(ValidationError):
            setattr(source, field, "Changed")
    with pytest.raises(TypeError):
        source.metadata["authors"][0]["name"] = "Changed"
    uncertainty = graph.add_uncertainty(UncertaintyRecord(
        id="U1", category="transfer", statement="Context transfer is unresolved.",
        scope="target assay", evidence=("E1",),
    ))
    with pytest.raises(ValidationError):
        uncertainty.statement = "Changed"
    with pytest.raises(ValueError):
        graph.add_assumption(Assumption(id="E1", statement="Collision", scope="test"))
    observation = graph.add_observation(Observation(
        id="O1", round=1, candidate_id="C0", intended_params={"x": 1.0},
        value_shown=4.0, outcome_unit="test",
    ))
    with pytest.raises(ValidationError):
        observation.value_shown = 5.0
    restored = EvidenceGraph.model_validate_json(graph.model_dump_json())
    assert restored.evidence_records["E1"].abstract == "Full abstract"


@pytest.mark.parametrize("task_name,acquisition", [
    ("enzyme", "ei"), ("drug", "ei"), ("drug", "gated_ei"),
])
def test_same_loop_is_seeded_and_predictions_remain_numerical(task_name, acquisition):
    config = Config(task=task_name, acquisition=acquisition, budget=4, seed=11)
    first, second = run_episode(config), run_episode(config)
    assert first.trajectory == second.trajectory and first.hidden == second.hidden
    assert first.graph.model_dump() == second.graph.model_dump()
    assert all(
        decision.prediction_source in ("numerical_model", "unavailable")
        for decision in first.graph.decisions.values()
    )
    first_metrics, second_metrics = episode_metrics(first), episode_metrics(second)
    first_metrics.pop("elapsed_s")
    second_metrics.pop("elapsed_s")
    assert first_metrics == second_metrics
    assert first.provider_metadata["calls"] == 0
    assert all("candidate_id" in row for row in first.hidden)


def test_removed_random_acquisition_is_rejected():
    with pytest.raises(ValueError, match="ei or gated_ei"):
        Config(acquisition="random")


def test_time_and_model_token_limits_stop_before_experiments():
    timed = run_episode(Config(budget=100, max_seconds=0))
    assert timed.stop_reason == "time_limit" and not timed.hidden

    provider = MockProvider()
    provider.tokens = 1
    token_limited = run_episode(
        Config(budget=100, max_model_tokens=1, provider="mock"), provider=provider,
    )
    assert token_limited.stop_reason == "model_token_limit" and not token_limited.hidden
    assert token_limited.provider_metadata["calls"] == 0


def test_stagnation_stops_after_patience_non_improving_observations():
    episode = run_episode(Config(task="drug", budget=1000, patience=5, max_seconds=60))
    assert episode.stop_reason == "stagnation"
    assert len(episode.hidden) < 1000
    observations = sorted(episode.graph.observations.values(), key=lambda item: item.round)
    values = [item.value_shown for item in observations if item.value_shown is not None]
    assert len(values) >= 6
    best_before = (max if episode.task.direction == "maximize" else min)(values[:-5])
    for value in values[-5:]:
        assert not (value > best_before if episode.task.direction == "maximize" else value < best_before)


def test_zero_patience_disables_stagnation_stopping():
    episode = run_episode(Config(budget=6, patience=0))
    assert len(episode.hidden) == 6
    assert episode.stop_reason == "experiment_limit"


def test_hypothesis_text_fields_are_trimmed():
    long_text = "x" * 311
    payload = json.dumps({"hypotheses": [{
        "statement": long_text,
        "scope": "s" * 201,
        "query": long_text,
        "belief": {"x": {"best": 0.5, "width_fraction": 0.2}},
        "discriminating_result": long_text,
    }]})
    hypothesis = Hypotheses.model_validate_json(payload).hypotheses[0]
    assert len(hypothesis.statement) == 300
    assert len(hypothesis.scope) == 200
    assert len(hypothesis.query) == 300
    assert len(hypothesis.discriminating_result) == 300


def test_calibration_reason_is_trimmed():
    payload = json.dumps({"calibrations": [{
        "index": 0, "trust": 0.5, "reason": "r" * 747, "evidence_ids": [],
    }]})
    calibration = Calibrations.model_validate_json(payload).calibrations[0]
    assert len(calibration.reason) == 300


def test_custom_domain_uses_the_same_loop_and_its_own_evaluator():
    task, _ = load_task("drug")
    task = replace(task, name="custom", candidates=task.candidates.head(3))
    evaluator = Evaluator(
        task,
        pd.DataFrame({
            "candidate_id": task.ids(), "outcome": [4.0, 2.0, 3.0], "spread": [0.0] * 3,
        }),
    )
    episode = run_episode(Config(task="unregistered", budget=3, observation_noise=False),
                          domain=(task, evaluator))
    assert episode.final_result()["value_shown"] == 2.0
    assert episode_metrics(episode)["final_selection_regret"] == 0.0
    assert len(episode.graph.uncertainties) == 3
    assert all(uncertainty.category == "response" for uncertainty in episode.graph.uncertainties.values())


def test_each_observation_has_a_response_uncertainty_and_qualifies_edge():
    episode = run_episode(Config(budget=3, patience=0))
    for observation in episode.graph.observations.values():
        records = [
            uncertainty for uncertainty in episode.graph.uncertainties.values()
            if observation.id in uncertainty.evidence
        ]
        assert len(records) == 1 and records[0].category == "response"
        assert any(
            edge.kind == "qualifies" and edge.source == records[0].id and edge.target == observation.id
            for edge in episode.graph.edges
        )
    assert all("candidate_id" in row for row in episode.hidden)
    metrics = episode_metrics(episode)
    expected = any(row["candidate_id"] == episode.evaluator.optimum_id for row in episode.hidden)
    assert metrics["found_optimum"] is expected
    assert "final_selected_mean" in metrics


def test_gp_separates_latent_noise_and_predictive_interval():
    task, _ = load_task("drug")
    model = Surrogate(task).fit([(task.ids()[0], 10.0), (task.ids()[-1], 30.0)])
    prediction = model.response(task.ids()[0])
    assert (prediction.interval[1] - prediction.interval[0]) / 2 == pytest.approx(
        1.96 * np.hypot(prediction.latent_sd, prediction.noise_sd)
    )


def test_expected_improvement_uses_total_predictive_sd():
    task, evaluator = load_task("drug")
    history = [(task.ids()[index], evaluator.truth(task.ids()[index])) for index in (0, 100, 300)]
    surrogate = Surrogate(task, seed=0).fit(history)
    surrogate._noise = 0.5
    assert surrogate._noise > 0

    incumbent = min(value for _, value in history)
    candidate_id, score = surrogate.ranked(incumbent, top=1)[0]
    index = task.ids().index(candidate_id)
    spread = np.hypot(surrogate._sd[index], surrogate._noise)
    improvement = -(surrogate._mean[index] - incumbent) - 0.01
    z = improvement / max(spread, 1e-12)
    expected = improvement * norm.cdf(z) + spread * norm.pdf(z)
    assert score == pytest.approx(expected)

    candidate_id, score = surrogate.ranked(None, top=1)[0]
    index = task.ids().index(candidate_id)
    assert score == pytest.approx(np.hypot(surrogate._sd[index], surrogate._noise))


def test_literature_graph_ingestion_keeps_full_sources_and_rejects_legacy_payloads():
    abstract = "RAW_ABSTRACT_SENTINEL " * 90 + " Conclusions: Matched treatment reduced response. More details."
    bundle = records_to_claims([{
        "title": "Reported response", "doi": "10.1234/scoped", "abstract": abstract,
    }], query="specific assay timing")
    incoming = EvidenceGraph.model_validate(bundle)
    claim = incoming.claims["L1"]
    assert len(claim.statement) <= 280 and "RAW_ABSTRACT_SENTINEL" not in claim.statement
    assert claim.scope and claim.discriminating_result
    assert {incoming.uncertainties[key].category for key in claim.uncertainties} == {
        "source", "transfer", "mechanistic",
    }
    source = incoming.evidence_records["E1"]
    assert source.abstract == source.metadata["abstract"] == abstract
    assert source.query == "specific assay timing"

    task, _ = load_task("enzyme")
    graph = seed_graph(task)
    assert set(graph.evidence_records) == {"E0"}
    assert set(graph.assumptions) == {"A0", "A_gp"}
    added, duplicates = ingest_literature(graph, bundle, prefix="X", limit=1)
    assert len(added) == 1 and not duplicates
    assert "EX_1" in graph.evidence_records and "LX_1" in graph.claims
    before = graph.model_dump_json()
    added, duplicates = ingest_literature(graph, bundle, prefix="Y")
    assert not added and duplicates and graph.model_dump_json() == before
    with pytest.raises(TypeError, match="graph dictionary"):
        ingest_literature(graph, list(bundle["claims"].values()))
    with pytest.raises(ValueError, match="cannot contain experiments"):
        ingest_literature(graph, bundle | {"observations": {"O1": {}}})


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
    graph = EvidenceGraph.model_validate_json(episode.graph.model_dump_json())
    bundle = records_to_claims([
        {"title": "First source", "doi": "10.1234/first", "abstract": "First text"},
        {"title": "Allowed source", "doi": "10.1234/allowed", "abstract": "Second text"},
    ])
    bundle["evidence_records"]["E1"]["metadata"]["hidden_reference"] = "10.1073/pnas.1606301113"
    ingest_literature(graph, bundle)
    assert "E1" not in graph and "L1" not in graph and "U1_transfer" not in graph
    assert "E2" in graph and "L2" in graph
    copied = replace(episode, graph=graph)
    export(copied, tmp_path)
    for name in ("graph.json", "actions.json", "graph.html", "graph.svg", "metrics.json"):
        assert not excluded_source((tmp_path / name).read_text())


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
    assert len(records_to_claims(result["records"])["evidence_records"]) == 1
    with pytest.raises(ValueError):
        asyncio.run(search_result("doi:10.1073/pnas.1606301113"))


def test_export_writes_only_compact_graph_views(episode, tmp_path):
    out = export(episode, tmp_path)
    expected = {
        "graph.json", "trajectory.jsonl", "hidden_truth.jsonl", "hidden_provenance.json",
        "config.json", "termination.json", "literature_setup.json", "final.json",
        "graph.html", "graph.svg", "actions.json", "metrics.json",
    }
    assert expected <= {path.name for path in out.iterdir()}
    assert not {"evidence.svg", "experiments.svg", "actions.svg", "audit.md", "final.md"} & {
        path.name for path in out.iterdir()
    }
    assert ET.fromstring((out / "graph.svg").read_text()).tag.endswith("svg")
    actions = json.loads((out / "actions.json").read_text())
    assert actions == actions_graph(episode)
    assert len(actions["nodes"]) == len(episode.graph.decisions) + 1
    assert actions["nodes"][-1]["kind"] == "selection"
    assert actions["nodes"][-1]["candidate_id"] == episode.final_selection()
    assert "true_value" not in (out / "trajectory.jsonl").read_text()
    hidden = [json.loads(line) for line in (out / "hidden_truth.jsonl").read_text().splitlines()]
    assert all("candidate_id" in row and "intended_id" not in row for row in hidden)
    with pytest.raises(FileExistsError):
        episode.save(tmp_path)


def test_html_is_self_contained_accessible_and_ids_are_unique(episode):
    content = to_html(episode)
    assert "__EPISTEMIC_BODY__" not in content
    for marker in ("data-a-inspector", "data-related-records", "data-a-zoom",
                   "Actions and reasoning", "data-record-id", 'role="button"'):
        assert marker in content
    ids = re.findall(r'(?<![\w-])id="([^"]+)"', content)
    assert len(ids) == len(set(ids))
    assert not re.search(r'<(?:script|img|iframe)[^>]+\bsrc\s*=', content, re.I)
    assert not re.search(r'<link[^>]+rel=["\']stylesheet', content, re.I)
    assert not re.search(r'\b(?:fetch|XMLHttpRequest|WebSocket|EventSource)\s*\(', content)
    assert not excluded_source(content)
    empty = Episode(Config(), episode.task, EvidenceGraph())
    assert "No experiments or observed final answer." in to_html(empty)


def test_interactive_graph_has_one_canvas_and_incumbent_dependencies(episode):
    content = to_html(episode)
    assert content.count('data-a-diagram-canvas="true"') == 1
    assert content.count('data-a-node="true"') == len(episode.graph.decisions) + 2
    assert "data-close-detail" in content
    for decision in sorted(episode.graph.decisions.values(), key=lambda item: item.round)[3:]:
        earlier = [
            observation for observation in episode.graph.observations.values()
            if observation.round < decision.round and observation.value_shown is not None
        ]
        incumbent = (max if episode.task.direction == "maximize" else min)(
            earlier, key=lambda item: item.value_shown,
        )
        assert f'data-edge-source="D{incumbent.round}" data-edge-target="{decision.id}"' in content
        assert f'&quot;ei_reference&quot;: &quot;{incumbent.id}&quot;' in content


def test_compact_graph_does_not_invent_literature_influence(episode):
    graph = EvidenceGraph.model_validate_json(episode.graph.model_dump_json())
    ingest_literature(graph, records_to_claims([
        {"title": "Independent assay", "doi": "10.1234/fixture", "abstract": "FULL_RAW_ABSTRACT_SENTINEL"},
    ]))
    copy = replace(episode, graph=graph)
    content = to_html(copy)
    assert 'data-edge-source="E1" data-edge-target="L1" data-edge-kind="supports"' in content
    assert "Not used in acquisition" in content
    assert 'data-edge-source="L1" data-edge-target="D' not in content
    svg = ET.fromstring(to_svg(copy))
    ns = {"s": "http://www.w3.org/2000/svg"}
    visible = " ".join("".join(node.itertext()) for node in svg.findall(".//s:text", ns))
    assert "FULL_RAW_ABSTRACT_SENTINEL" not in visible
    assert "FULL_RAW_ABSTRACT_SENTINEL" in content


def test_optional_inspect_adapter_imports_without_running_evaluations():
    from epistemic.inspect_task import experimental_design

    assert experimental_design(budget=1, noise_cv=0.05).solver


def test_provider_default_never_calls_a_model():
    assert get_provider().name == "none"
    assert MockProvider().complete("fixture prompt") == "{}"
    with pytest.raises(KeyError):
        get_provider("old_selection_controller")


class LiteratureFixture(MockProvider):
    def complete(self, prompt):
        self.calls += 1
        self.tokens += len(prompt) // 4
        assert "at most 300 characters" in prompt
        if self.calls == 1:
            assert "scope at most 200 characters" in prompt
            return json.dumps({"hypotheses": [{
                "statement": "Higher taxol dose may reduce survival in this assay.",
                "scope": "A549 cells under the task exposure conditions",
                "query": "A549 taxol dose response assay",
                "belief": {"taxol_uM": {"best": 0.1, "width_fraction": 0.3}},
                "discriminating_result": "Compare high and low taxol with other doses held fixed.",
            }]})
        assert "PRIVATE_HELD_OUT_RESULT" not in prompt
        return json.dumps({"calibrations": [{
            "index": 0, "trust": 0.6, "reason": "Cell-line and timing transfer remain uncertain.",
            "evidence_ids": ["EP1_1"],
        }]})


def literature_fixture_search(query):
    return {"records": [
        {"title": "Held out", "doi": "10.1073/pnas.1606301113", "abstract": "PRIVATE_HELD_OUT_RESULT"},
        {"title": "Independent assay", "doi": "10.1234/independent", "abstract": "Reported assay details."},
    ]}


def test_gated_prior_changes_the_shared_posterior_and_learns_uncertain_weights():
    task, evaluator = load_task("drug")
    history = [(task.ids()[index], evaluator.truth(task.ids()[index])) for index in (31, 220, 400)]
    plain = Surrogate(task, seed=0).fit(history)
    empty = Surrogate(task, seed=0).with_priors([]).fit(history)
    np.testing.assert_array_equal(plain._mean, empty._mean)
    np.testing.assert_array_equal(plain._sd, empty._sd)
    prior = NumericalPrior("P1", {"taxol_uM": (0.1, 0.3)}, 0.6)
    gated = Surrogate(task, seed=0).with_priors([prior]).fit(history)
    assert not np.allclose(plain._mean, gated._mean)
    assert gated.ranked(min(value for _, value in history), top=10) != plain.ranked(
        min(value for _, value in history), top=10,
    )
    mean, sd = gated.gates["P1"]
    assert np.isfinite(mean) and 0 < sd < 0.5
    assert mean != prior.trust
    assert np.isfinite(gated._sd).all()


def test_bounded_literature_calibrates_prior_and_exports_gate_provenance(tmp_path):
    provider = LiteratureFixture()
    episode = run_episode(
        Config(acquisition="gated_ei", budget=5, max_searches=1),
        provider=provider, literature_search=literature_fixture_search,
    )
    assert len(episode.hidden) == 5 and provider.calls == 2
    prior = episode.graph.claims["P1"]
    assert prior.trust == 0.6 and prior.evidence == ["EP1_1"]
    assert prior.belief == {"taxol_uM": (0.1, 0.3)}
    assert {episode.graph.uncertainties[key].category for key in prior.uncertainties} == {
        "source", "transfer", "mechanistic",
    }
    dependencies = [
        edge for edge in episode.graph.edges if edge.target == "P1" and edge.kind == "depends_on"
    ]
    assert dependencies and all(edge.note.startswith("gate ") for edge in dependencies)
    decision = episode.graph.decisions["D4"]
    assert "P1" in decision.claims and "P1" in decision.prior_gates
    assert "Learned literature gates" in decision.justification
    metrics = episode_metrics(episode)
    assert metrics["literature_searches"] == 1 and metrics["active_literature_priors"] == 1
    html = to_html(episode)
    assert "Initial trust: 0.60" in html and "Numerical prior used in acquisition" in html
    assert "Cell-line and timing" in html and "depends_on" in html
    assert "&quot;prior_gates&quot;" in html and not excluded_source(html)
    out = export(episode, tmp_path)
    termination = json.loads((out / "termination.json").read_text())
    assert termination["reason"] == "experiment_limit" and termination["experiments"] == 5
    assert json.loads((out / "literature_setup.json").read_text())["priors"]


@pytest.mark.parametrize("settings,reason", [
    ({"max_seconds": 0}, "time_limit"),
    ({"max_model_tokens": 1}, "model_token_limit"),
])
def test_literature_setup_obeys_resource_limits_before_spending(settings, reason):
    provider = LiteratureFixture()
    episode = run_episode(
        Config(acquisition="gated_ei", max_searches=1, **settings),
        provider=provider, literature_search=literature_fixture_search,
    )
    assert episode.stop_reason == reason and not episode.hidden
    assert provider.calls == 0 and not episode.initial_evidence["searches"]


def test_optional_literature_failures_fall_back_to_plain_gp():
    def unavailable(query):
        raise RuntimeError("fixture-only retrieval failure")

    provider = LiteratureFixture()
    episode = run_episode(
        Config(acquisition="gated_ei", budget=4, max_searches=1),
        provider=provider, literature_search=unavailable,
    )
    baseline = run_episode(Config(budget=4))
    assert len(episode.hidden) == 4 and provider.calls == 1
    assert episode.final_selection() == baseline.final_selection()
    assert [row["candidate_id"] for row in episode.hidden] == [
        row["candidate_id"] for row in baseline.hidden
    ]
    assert episode.initial_evidence["searches"][0]["status"] == "failed"
    assert not episode.initial_evidence["priors"]
