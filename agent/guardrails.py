"""Safety policy the agent cannot reason its way around.

The planner (rule-based or LLM-backed) decides what it *wants* to do.
GuardrailViolation is what stops it from doing something unsafe even if the
planner's reasoning is wrong or, in the LLM case, hallucinated. This layer
knows nothing about incidents or root causes — it only counts and caps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .actions import Action, ActionType

MAX_REPLICAS = 10
MAX_RESTARTS_PER_INCIDENT = 1
MAX_TOTAL_ACTIONS_PER_INCIDENT = 6


class GuardrailViolation(Exception):
    pass


@dataclass
class GuardrailState:
    total_actions: int = 0
    restart_count: int = 0
    history: list[Action] = field(default_factory=list)


class Guardrails:
    def __init__(self) -> None:
        self.state = GuardrailState()

    def check(self, action: Action) -> None:
        """Raise GuardrailViolation if `action` is not allowed to execute."""
        if self.state.total_actions >= MAX_TOTAL_ACTIONS_PER_INCIDENT:
            raise GuardrailViolation(
                f"exceeded {MAX_TOTAL_ACTIONS_PER_INCIDENT} actions for this incident "
                "without resolving it — forcing escalation instead of looping forever"
            )

        if action.type is ActionType.RESTART and self.state.restart_count >= MAX_RESTARTS_PER_INCIDENT:
            raise GuardrailViolation(
                f"already restarted {self.state.restart_count} time(s) with no improvement — "
                "a repeat restart is not a new remediation, it's the agent repeating itself"
            )

        if action.type is ActionType.SCALE and (action.replicas is None or action.replicas > MAX_REPLICAS):
            raise GuardrailViolation(
                f"refusing to scale to {action.replicas} replicas, cap is {MAX_REPLICAS}"
            )

    def record(self, action: Action) -> None:
        self.state.total_actions += 1
        self.state.history.append(action)
        if action.type is ActionType.RESTART:
            self.state.restart_count += 1
