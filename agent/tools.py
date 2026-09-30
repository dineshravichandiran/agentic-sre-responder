"""Tools the agent is allowed to call against a live (or simulated) service.

Each tool is a thin, auditable wrapper around Environment. Keeping them
separate from the planner means every side-effecting action the agent can
take is visible in one file, independent of which planner (heuristic or
LLM-backed) decided to call it.
"""
from __future__ import annotations

from dataclasses import dataclass

from .environment import Environment


@dataclass(frozen=True)
class Observation:
    memory_pct: float
    disk_pct: float
    error_rate: float
    restart_count: int
    replicas: int

    @classmethod
    def from_env(cls, env: Environment) -> "Observation":
        s = env.service
        return cls(
            memory_pct=round(s.memory_mb / s.memory_limit_mb, 4),
            disk_pct=round(s.disk_used_mb / s.disk_limit_mb, 4),
            error_rate=s.error_rate,
            restart_count=s.restart_count,
            replicas=s.replicas,
        )

    def is_healthy(self) -> bool:
        return self.memory_pct < 0.75 and self.disk_pct < 0.85 and self.error_rate < 0.01


def get_metrics(env: Environment) -> Observation:
    return Observation.from_env(env)


def get_logs(env: Environment, n: int = 5) -> list[str]:
    return env.service.log_buffer[-n:]


def restart_service(env: Environment) -> str:
    env.apply_restart()
    return f"restarted {env.service.name} (restart_count={env.service.restart_count})"


def scale_service(env: Environment, replicas: int) -> str:
    env.apply_scale(replicas)
    return f"scaled {env.service.name} to {replicas} replicas"


def escalate(reason: str) -> str:
    # No real paging integration here — this simulates the handoff point
    # where an autonomous agent should stop acting and hand off to a human,
    # which is the behavior the tests care about, not the transport.
    return f"paged on-call: {reason}"
