"""Run GP-BO and export its reasoning graph."""

import argparse
import asyncio
import json
from pathlib import Path

from epistemic.loop import Config, run_episode
from epistemic.render import export


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "literature"))
    parser.add_argument("--task", choices=("enzyme", "drug"), default="drug")
    parser.add_argument("--acquisition", choices=("ei", "gated_ei", "random"), default="ei")
    parser.add_argument("--budget", type=int, default=1000, help="Safety cap; time is the primary stopping limit")
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument("--max-model-tokens", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--provider", default="none", help="none or openai:<model>; mock is a test fixture")
    parser.add_argument("--execution", choices=("perfect", "perturbed"), default="perfect")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--evidence-file")
    parser.add_argument("--max-searches", type=int, default=0)
    parser.add_argument("--out", default="logs/current")
    parser.add_argument("--query", help="Targeted Amass query for the literature command")
    args = parser.parse_args()
    out = Path(args.out)
    if args.command == "literature":
        from epistemic.amass import evidence_bundle
        if not args.query:
            parser.error("literature requires --query")
        result = asyncio.run(evidence_bundle(args.query))
        out.mkdir(parents=True, exist_ok=True)
        with (out / "evidence.json").open("x") as handle:
            json.dump(result, handle, indent=2)
        return
    config = Config(
        task=args.task, budget=args.budget, max_seconds=args.max_seconds,
        max_model_tokens=args.max_model_tokens, seed=args.seed, execution=args.execution,
        acquisition=args.acquisition,
        provider=args.provider, temperature=args.temperature,
        evidence_file=args.evidence_file, max_searches=args.max_searches,
    )
    episode = run_episode(config)
    export(episode, out)
    print(json.dumps(episode.final_result(), indent=2))


if __name__ == "__main__":
    main()
