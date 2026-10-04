"""Headless recorded run. Example:
python -m minimal_lab.record --budget 80 --replicates 1 --papers data/zimmer/literature_pubmed.json
Reads ANTHROPIC_API_KEY and AMASS_API_KEY from the environment; --papers replaces AMASS retrieval."""
import argparse
import json
import os
from pathlib import Path

from minimal_lab.live import Demo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env', default='zimmer_a549')
    p.add_argument('--budget', type=int, default=80)
    p.add_argument('--replicates', type=int, default=1)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--cv', type=float, default=.15)
    p.add_argument('--model', default='anthropic/claude-sonnet-4-6')
    p.add_argument('--papers', help='JSON list of {title, abstract, url} used instead of AMASS retrieval')
    p.add_argument('--out', default='logs/recorded')
    args = p.parse_args()
    if args.papers:
        os.environ['AMASS_CACHE_FILE'] = str(Path(args.papers).resolve())
        os.environ.setdefault('AMASS_API_KEY', 'unused')
    demo = Demo(args)
    event = demo.event

    def report(stage, graph):
        event(stage, graph)
        if stage in ('observation', 'saved'):
            n = sum(x['kind'] == 'observation' for x in graph['nodes'])
            print(f'{stage}: {n}/{args.budget} preparations', flush=True)
    demo.event = report
    demo.work()
    print(json.dumps({k: demo.state.get(k) for k in ('status', 'error', 'scores')}))
    print('Video:', demo.output_dir/'assay.mp4')


if __name__ == '__main__':
    main()
