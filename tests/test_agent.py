import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from agent.actions import Action, ActionType
from agent.agent import IncidentResponder, Verdict
from agent.environment import Environment
from agent.guardrails import Guardrails, GuardrailViolation, MAX_REPLICAS
from agent.planner import HeuristicPlanner, LLMPlanner


def test_memory_leak_resolves_via_single_restart():
    env = Environment()
    env.seed_memory_leak()
    responder = IncidentResponder(env, HeuristicPlanner())

    transcript = responder.run()

    assert transcript.verdict is Verdict.RESOLVED
    assert env.service.restart_count == 1
    assert env.service.error_rate < 0.01


def test_disk_full_unfixable_escalates_instead_of_looping():
    """The core safety property: a remediation that doesn't work must lead
    to escalation, not the agent trying the same restart forever."""
    env = Environment()
    env.seed_disk_full_unfixable()
    responder = IncidentResponder(env, HeuristicPlanner())

    transcript = responder.run()

    assert transcript.verdict is Verdict.ESCALATED
    # Exactly one restart was attempted before the planner recognized it
    # didn't help and escalated — not zero (never tried), not many (looping).
    assert env.service.restart_count == 1


def test_guardrail_blocks_second_restart_even_if_planner_asks_for_it():
    guardrails = Guardrails()
    guardrails.record(Action(ActionType.RESTART))

    with pytest.raises(GuardrailViolation):
        guardrails.check(Action(ActionType.RESTART))


def test_guardrail_caps_replica_count():
    guardrails = Guardrails()
    with pytest.raises(GuardrailViolation):
        guardrails.check(Action(ActionType.SCALE, replicas=MAX_REPLICAS + 1))

    # Exactly at the cap is fine.
    guardrails.check(Action(ActionType.SCALE, replicas=MAX_REPLICAS))


def test_guardrail_stops_a_buggy_planner_that_never_resolves():
    """Simulates a hallucinating/broken planner that always asks to keep
    fetching logs forever. The agent must still terminate via escalation,
    not hang or loop past the action budget."""

    class StuckPlanner:
        def decide(self, obs, logs, history):
            return Action(ActionType.GET_LOGS)

    env = Environment()
    env.seed_memory_leak()
    responder = IncidentResponder(env, StuckPlanner())

    transcript = responder.run(max_steps=20)

    assert transcript.verdict is Verdict.ESCALATED
    assert len(transcript.steps) <= 7  # 6 allowed actions + the forced escalation step


def test_heuristic_planner_reads_logs_before_acting():
    planner = HeuristicPlanner()
    env = Environment()
    env.seed_memory_leak()
    obs = __import__("agent.tools", fromlist=["Observation"]).Observation.from_env(env)

    first_action = planner.decide(obs, logs=[], history=[])
    assert first_action.type is ActionType.GET_LOGS


def test_llm_planner_requires_api_key_when_no_client_given(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        LLMPlanner()


def test_llm_planner_parses_a_valid_model_response():
    """Exercises the real parsing path with a fake client so this test needs
    no network access or API key, while still proving the LLM integration's
    request/response contract is correct end to end."""

    class FakeContentBlock:
        def __init__(self, text):
            self.text = text

    class FakeResponse:
        def __init__(self, text):
            self.content = [FakeContentBlock(text)]

    class FakeMessages:
        def create(self, **kwargs):
            payload = json.loads(kwargs["messages"][0]["content"])
            assert "metrics" in payload
            return FakeResponse(json.dumps({"action": "RESTART", "reason": "heap exhaustion"}))

    class FakeClient:
        def __init__(self):
            self.messages = FakeMessages()

    planner = LLMPlanner(client=FakeClient())
    env = Environment()
    env.seed_memory_leak()
    from agent.tools import Observation

    obs = Observation.from_env(env)
    action = planner.decide(obs, logs=["ERROR OutOfMemoryError"], history=[])

    assert action.type is ActionType.RESTART
    assert action.reason == "heap exhaustion"
