"""Small CLI: direct episodes, feedback interventions and matched controls."""

import argparse
import asyncio
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from epistemic.interventions import closed_loop, graph_value, paired_interventions
from epistemic.loop import Config, run_episode
from epistemic.provider import get_provider
from epistemic.render import export
from epistemic.tasks import load_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "feedback", "closed-loop", "graph-control", "dataset-info", "literature"))
    parser.add_argument("--task", choices=("enzyme", "drug"), default="drug")
    parser.add_argument("--policy", choices=("random", "bo", "llm"), default="llm")
    parser.add_argument("--provider", default="mock", help="mock or openai:<model>")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--execution", choices=("perfect", "perturbed"), default="perfect")
    parser.add_argument("--feedback", choices=("true", "missing", "corrupted"), default="true")
    parser.add_argument("--flat", action="store_true")
    parser.add_argument("--misleading", action="store_true")
    parser.add_argument("--fault-round", type=int)
    parser.add_argument("--prefix", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=("total", "direct", "both"), default="both")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out", default="logs/epistemic")
    parser.add_argument("--evidence-file", help="A fixed, source-backed JSON bundle of claim records")
    parser.add_argument("--max-searches", type=int, default=0, help="Hard per-episode Amass request cap; sources arrive next round")
    parser.add_argument("--query", help="Amass literature query; only used by the literature subcommand")
    args = parser.parse_args()
    config = Config(task=args.task, policy=args.policy, budget=args.budget, seed=args.seed,
                    execution=args.execution, feedback=args.feedback, with_edges=not args.flat,
                    misleading_evidence=args.misleading, provider=args.provider, fault_round=args.fault_round,
                    temperature=args.temperature, evidence_file=args.evidence_file, max_searches=args.max_searches)
    if args.command == "literature":
        from epistemic.amass import evidence_bundle
        if not args.query:
            parser.error("literature requires --query and AMASS_API_KEY")
        bundle = asyncio.run(evidence_bundle(args.query))
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "evidence.json").open("x") as handle:
            json.dump(bundle, handle, indent=2)
        print(f"Saved attributable extracts to {out / 'evidence.json'}; transfer applicability remains unresolved.")
        return
    if args.command == "dataset-info":
        task, _ = load_task(args.task)
        print(json.dumps({
            "task": task.name, "candidates": len(task.candidates), "parameters": [asdict(p) for p in task.parameters],
            "objective": task.objective, "direction": task.direction, "provenance": task.provenance,
            "limitations": task.limitations, "noise": asdict(task.noise),
        }, indent=2))
        return
    out = Path(args.out)
    if args.command == "run":
        provider = get_provider(args.provider, args.seed, args.temperature)
        episode = run_episode(config, provider=provider)
        export(episode, out)
        print((out / "metrics.json").read_text())
        return
    results: Any
    if args.command == "feedback":
        results = {}
        base = run_episode(replace(config, feedback="true"), stop_after=args.prefix)
        frozen = list(base.graph.observations.values())
        frozen_evidence = [c for c in base.graph.claims.values() if c.source == "literature"]
        for mode in ("total", "direct"):
            if args.mode not in ("both", mode):
                continue
            results[mode] = paired_interventions(config, args.prefix, args.repeats,
                                                 direct_llm_only=mode == "direct", frozen=frozen,
                                                 frozen_evidence=frozen_evidence)
    elif args.command == "closed-loop":
        results = closed_loop(config, seeds=tuple(args.seeds))
    else:
        results = graph_value(config, rounds=args.budget, seeds=tuple(args.seeds))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.json").open("x") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved {args.command} results to {out / 'results.json'}")


if __name__ == "__main__":
    main()
