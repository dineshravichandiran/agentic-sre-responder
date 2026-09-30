"""The fixed vocabulary of moves a planner is allowed to propose.

Both HeuristicPlanner and LLMPlanner emit the same Action type, which is
what lets the agent loop, the guardrails, and the tests treat them
interchangeably — an LLM-backed planner is a drop-in replacement for the
rule-based one, not a special case.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class ActionType(Enum):
    GET_LOGS = auto()
    RESTART = auto()
    SCALE = auto()
    ESCALATE = auto()
    RESOLVED = auto()


@dataclass(frozen=True)
class Action:
    type: ActionType
    replicas: int | None = None
    reason: str | None = None

    def __str__(self) -> str:
        if self.type is ActionType.SCALE:
            return f"SCALE(replicas={self.replicas})"
        if self.reason:
            return f"{self.type.name}({self.reason})"
        return self.type.name
