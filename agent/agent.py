"""The ReAct-style loop: observe, plan, guardrail-check, act, repeat."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from . import tools
from .actions import Action, ActionType
from .environment import Environment
from .guardrails import Guardrails, GuardrailViolation
from .planner import Planner


class Verdict(Enum):
    RESOLVED = auto()
    ESCALATED = auto()
    TIMED_OUT = auto()


@dataclass
class Step:
    action: Action
    result: str


@dataclass
class Transcript:
    steps: list[Step] = field(default_factory=list)
    verdict: Verdict | None = None
    guardrail_blocks: int = 0

    def print(self) -> None:
        for i, step in enumerate(self.steps, start=1):
            print(f"  {i}. {step.action} -> {step.result}")
        print(f"  verdict: {self.verdict.name}")


class IncidentResponder:
    def __init__(self, env: Environment, planner: Planner, guardrails: Guardrails | None = None) -> None:
        self.env = env
        self.planner = planner
        self.guardrails = guardrails or Guardrails()

    def run(self, max_steps: int = 10) -> Transcript:
        transcript = Transcript()
        history: list[Action] = []
        logs: list[str] = []

        for _ in range(max_steps):
            obs = tools.get_metrics(self.env)
            action = self.planner.decide(obs, logs, history)

            try:
                self.guardrails.check(action)
            except GuardrailViolation as violation:
                transcript.guardrail_blocks += 1
                result = tools.escalate(str(violation))
                transcript.steps.append(Step(Action(ActionType.ESCALATE, reason=str(violation)), result))
                transcript.verdict = Verdict.ESCALATED
                return transcript

            self.guardrails.record(action)
            history.append(action)

            if action.type is ActionType.RESOLVED:
                transcript.steps.append(Step(action, "metrics healthy, closing incident"))
                transcript.verdict = Verdict.RESOLVED
                return transcript

            if action.type is ActionType.GET_LOGS:
                logs = tools.get_logs(self.env)
                transcript.steps.append(Step(action, f"{len(logs)} recent log line(s) retrieved"))
                continue

            if action.type is ActionType.RESTART:
                result = tools.restart_service(self.env)
                transcript.steps.append(Step(action, result))
                continue

            if action.type is ActionType.SCALE:
                result = tools.scale_service(self.env, action.replicas or 1)
                transcript.steps.append(Step(action, result))
                continue

            if action.type is ActionType.ESCALATE:
                result = tools.escalate(action.reason or "unspecified")
                transcript.steps.append(Step(action, result))
                transcript.verdict = Verdict.ESCALATED
                return transcript

        transcript.verdict = Verdict.TIMED_OUT
        return transcript
