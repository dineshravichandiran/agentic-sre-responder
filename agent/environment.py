"""A tiny synthetic production environment the agent investigates and acts on.

Not a real service. State is deterministic and seeded so tests are
reproducible, the same pattern used across the other self-directed
projects in this portfolio (real fixtures over mocks, no hidden randomness).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ServiceState:
    name: str
    memory_mb: float
    memory_limit_mb: float
    disk_used_mb: float
    disk_limit_mb: float
    error_rate: float
    restart_count: int = 0
    replicas: int = 1
    log_buffer: list[str] = field(default_factory=list)
    # If True, restarting does NOT fix the root cause (e.g. a runaway log
    # writer keeps refilling disk). Used to test that the agent escalates
    # instead of declaring false success.
    remediation_ineffective: bool = False


class Environment:
    """Holds one service's state and applies incident scenarios to it."""

    def __init__(self) -> None:
        self.service = ServiceState(
            name="checkout-api",
            memory_mb=512,
            memory_limit_mb=2048,
            disk_used_mb=4000,
            disk_limit_mb=20000,
            error_rate=0.001,
        )

    def seed_memory_leak(self) -> None:
        self.service.memory_mb = 1900  # near the 2048 MB limit
        self.service.error_rate = 0.18
        self.service.log_buffer = [
            "WARN  gc: full GC took 4200ms, heap usage at 92%",
            "ERROR OutOfMemoryError: Java heap space at checkout.CartCache.put",
            "ERROR OutOfMemoryError: Java heap space at checkout.CartCache.put",
            "WARN  request queue depth climbing: 340 pending",
        ]

    def seed_disk_full_unfixable(self) -> None:
        self.service.disk_used_mb = 19500  # near the 20000 MB limit
        self.service.error_rate = 0.09
        self.service.remediation_ineffective = True
        self.service.log_buffer = [
            "ERROR write failed: no space left on device (/var/log/checkout/debug.log)",
            "ERROR write failed: no space left on device (/var/log/checkout/debug.log)",
            "WARN  debug logging left enabled at DEBUG level in production config",
        ]

    # --- actions the agent can take -------------------------------------

    def apply_restart(self) -> None:
        self.service.restart_count += 1
        if self.service.remediation_ineffective:
            # Root cause (e.g. a misconfigured debug logger) keeps regrowing
            # disk usage within moments of the restart.
            self.service.disk_used_mb = 19600
            self.service.error_rate = 0.09
            self.service.log_buffer.append(
                "ERROR write failed: no space left on device (/var/log/checkout/debug.log)"
            )
            return
        # Normal case: restart actually clears the leaked heap.
        self.service.memory_mb = 480
        self.service.error_rate = 0.001
        self.service.log_buffer = ["INFO  service restarted cleanly, heap reset"]

    def apply_scale(self, replicas: int) -> None:
        self.service.replicas = replicas
