#!/usr/bin/env python3
"""Run the agent against both seeded incident scenarios and print its transcript.

    python scripts/demo.py

Uses HeuristicPlanner by default (no API key needed). Pass --llm to use
LLMPlanner instead, which requires ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent import IncidentResponder
from agent.environment import Environment
from agent.planner import HeuristicPlanner, LLMPlanner


def run_scenario(title: str, seed_fn_name: str, planner) -> None:
    print(f"\n=== {title} ===")
    env = Environment()
    getattr(env, seed_fn_name)()
    responder = IncidentResponder(env, planner)
    transcript = responder.run()
    transcript.print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="use LLMPlanner (requires ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    planner = LLMPlanner() if args.llm else HeuristicPlanner()
    print(f"planner: {type(planner).__name__}")

    run_scenario("Scenario 1: memory leak (restart fixes it)", "seed_memory_leak", planner)
    run_scenario(
        "Scenario 2: disk full from a misconfigured logger (restart does NOT fix it)",
        "seed_disk_full_unfixable",
        planner,
    )


if __name__ == "__main__":
    main()
