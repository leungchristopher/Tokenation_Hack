"""Run GP-BO and export its reasoning graph."""

import argparse
import json
from pathlib import Path

from epistemic.loop import Config, run_episode
from epistemic.render import export


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--task", choices=("enzyme", "drug"), default="drug")
    parser.add_argument("--acquisition", choices=("ei", "gated_ei"), default="ei")
    parser.add_argument("--budget", type=int, default=1000, help="Safety cap; time is the primary stopping limit")
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument("--max-model-tokens", type=int, default=12_000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--provider", default="none", help="none or openai:<model>; mock is a test fixture")
    parser.add_argument("--max-searches", type=int, default=0)
    parser.add_argument("--noise-cv", type=float)
    parser.add_argument("--out", default="logs/current")
    args = parser.parse_args()
    out = Path(args.out)
    config = Config(
        task=args.task, budget=args.budget, max_seconds=args.max_seconds,
        max_model_tokens=args.max_model_tokens, patience=args.patience, seed=args.seed,
        acquisition=args.acquisition, provider=args.provider, max_searches=args.max_searches,
        noise_cv=args.noise_cv,
    )
    episode = run_episode(config)
    export(episode, out)
    print(json.dumps(episode.final_result(), indent=2))


if __name__ == "__main__":
    main()
