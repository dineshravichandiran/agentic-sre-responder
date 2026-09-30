"""Two interchangeable planners: one rule-based, one LLM-backed.

HeuristicPlanner is deterministic and needs no external API — it's what
the tests and the default demo run against, so the agent's control flow
(observe -> plan -> guardrail-check -> act) is provable without any network
dependency or API key.

LLMPlanner calls the Anthropic API to make the same kind of decision from
the same observation. It's a real, working integration, not a stub — but it
needs ANTHROPIC_API_KEY in the environment to actually call out, since this
agent doesn't ship with anyone's key. It's included so the planner boundary
is a genuine swap point, not just a hypothetical one.
"""
from __future__ import annotations

import json
import os
from typing import Protocol

from .actions import Action, ActionType
from .tools import Observation


class Planner(Protocol):
    def decide(self, obs: Observation, logs: list[str], history: list[Action]) -> Action: ...


class HeuristicPlanner:
    """Fixed rules an SRE would recognize: look before you act, don't retry
    a remediation that already failed, declare victory only when metrics
    say so."""

    def decide(self, obs: Observation, logs: list[str], history: list[Action]) -> Action:
        if obs.is_healthy():
            return Action(ActionType.RESOLVED)

        have_logs = any(a.type is ActionType.GET_LOGS for a in history)
        if not have_logs:
            return Action(ActionType.GET_LOGS)

        already_restarted = any(a.type is ActionType.RESTART for a in history)
        log_text = " ".join(logs)

        if already_restarted:
            # We already tried the standard fix and the metrics are still
            # bad — restarting again is not a new idea, it's a loop.
            return Action(
                ActionType.ESCALATE,
                reason="restart did not resolve the incident; likely root cause is not restart-fixable",
            )

        if "OutOfMemoryError" in log_text and obs.memory_pct > 0.85:
            return Action(ActionType.RESTART, reason="heap exhaustion, clear it with a clean restart")

        if "no space left on device" in log_text and obs.disk_pct > 0.9:
            return Action(ActionType.RESTART, reason="disk pressure, try a restart before escalating")

        return Action(
            ActionType.ESCALATE,
            reason="metrics unhealthy but logs don't match a known automated remediation",
        )


class LLMPlanner:
    """Same decision, made by an LLM instead of fixed rules.

    Requires the `anthropic` package and ANTHROPIC_API_KEY. Constructing
    this without a key raises immediately rather than silently falling back,
    so a misconfigured deployment fails loud, not quiet.
    """

    SYSTEM_PROMPT = (
        "You are an SRE incident-response planner. Given service metrics and "
        "recent logs, choose exactly one next action. Respond with ONLY a "
        'JSON object: {"action": "GET_LOGS"|"RESTART"|"SCALE"|"ESCALATE"|"RESOLVED", '
        '"replicas": <int, only for SCALE>, "reason": "<short reason>"}. '
        "Do not restart a service that has already been restarted this "
        "incident without improvement — escalate instead."
    )

    def __init__(self, model: str = "claude-sonnet-5", client=None) -> None:
        if client is not None:
            self._client = client
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "LLMPlanner requires ANTHROPIC_API_KEY to be set. "
                    "Use HeuristicPlanner if you don't have one configured."
                )
            import anthropic  # imported lazily so the package is optional

            self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def decide(self, obs: Observation, logs: list[str], history: list[Action]) -> Action:
        user_prompt = json.dumps(
            {
                "metrics": {
                    "memory_pct": obs.memory_pct,
                    "disk_pct": obs.disk_pct,
                    "error_rate": obs.error_rate,
                    "restart_count": obs.restart_count,
                    "replicas": obs.replicas,
                },
                "recent_logs": logs,
                "actions_taken_so_far": [str(a) for a in history],
            }
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=200,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = response.content[0].text
        return self._parse(text)

    @staticmethod
    def _parse(text: str) -> Action:
        data = json.loads(text)
        action_type = ActionType[data["action"]]
        return Action(
            type=action_type,
            replicas=data.get("replicas"),
            reason=data.get("reason"),
        )
