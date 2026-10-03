"""A minimal benchmark for epistemically aware experimental design.

Two measured tasks, one sequential loop, three selection policies, and an
append-only evidence graph that separates tool outputs from model interpretation.
"""

from epistemic.graph import EvidenceGraph
from epistemic.loop import Config, Episode, run_episode
from epistemic.tasks import TASKS, Evaluator, TaskSpec, load_task

__all__ = ["TASKS", "Config", "Episode", "Evaluator", "EvidenceGraph", "TaskSpec", "load_task", "run_episode"]
