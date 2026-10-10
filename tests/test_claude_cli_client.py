import json

from agent.claude_cli_client import ClaudeCLIClient, ISOLATION_FLAGS, build_prompt, build_schema
from agent.llm_agent import TOOLS, LLMToolAgent
from agent.scenarios import BY_NAME, build_environment
from evals import run as evals


def test_schema_only_allows_the_agents_own_tools():
    schema = build_schema(TOOLS)
    assert schema["properties"]["tool"]["enum"] == [t["name"] for t in TOOLS]
    assert "note" in schema["required"]


def test_prompt_carries_the_whole_exchange_and_marks_errors():
    messages = [
        {"role": "user", "content": "alert fired"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "restart_service", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "BLOCKED by guardrail", "is_error": True}]},
    ]
    prompt = build_prompt(TOOLS, messages)
    assert "[TASK] alert fired" in prompt
    assert "[YOU CALLED] restart_service {}" in prompt
    assert "[TOOL ERROR] BLOCKED by guardrail" in prompt


def test_each_decision_runs_isolated_from_the_users_files_and_settings():
    seen = []

    def fake_run(argv, cwd):
        seen.append((argv, cwd))
        return json.dumps({"is_error": False, "structured_output": {"tool": "get_metrics", "arguments": {}, "note": "look first"}})

    client = ClaudeCLIClient(run=fake_run)
    client.create(model="claude-sonnet-5", system="sys", tools=TOOLS, messages=[{"role": "user", "content": "x"}])
    argv, cwd = seen[0]
    for flag in ("--tools", "--setting-sources", "--no-session-persistence"):
        assert flag in argv
    assert argv[argv.index("--tools") + 1] == "" and argv[argv.index("--setting-sources") + 1] == ""
    assert "sre-agent-empty" in cwd  # not the project or home directory
    assert set(ISOLATION_FLAGS) <= set(argv)


def test_the_agent_runs_end_to_end_through_the_cli_adapter():
    script = iter([
        {"tool": "get_logs", "arguments": {}, "note": "read logs"},
        {"tool": "restart_service", "arguments": {}, "note": "heap exhausted"},
        {"tool": "close_incident", "arguments": {"summary": "restart fixed it"}, "note": "healthy"},
    ])

    def fake_run(argv, cwd):
        return json.dumps({"is_error": False, "structured_output": next(script)})

    scenario = BY_NAME["memory_leak"]
    env = build_environment(scenario)
    result = LLMToolAgent(ClaudeCLIClient(run=fake_run), model="claude-sonnet-5").run(env)
    outcome = evals.judge(scenario, result.verdict, env, result.guardrail_blocks, result.false_close_attempts, len(result.tool_calls))
    assert outcome.passed, outcome.failures


def test_a_cli_error_is_raised_not_swallowed():
    import pytest

    client = ClaudeCLIClient(run=lambda argv, cwd: json.dumps({"is_error": True, "result": "rate limited"}))
    with pytest.raises(RuntimeError, match="rate limited"):
        client.create(model="m", tools=TOOLS, messages=[{"role": "user", "content": "x"}])
