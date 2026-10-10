"""Run the agent on Claude through the Claude Code command line, with no API key.

LLMToolAgent talks to a client that has `messages.create(...)`. This adapter
implements that call by asking a fresh, isolated Claude (`claude -p`) for the
agent's next tool call, one decision per invocation:

  * each call sees ONLY the system prompt, the incident task, and the tool
    results so far. No tools, no project files, no user settings or memory
    (--tools "", --setting-sources "", run from an empty directory), so it
    behaves like an agent that knows nothing but what its tools told it
  * the choice is constrained with --json-schema, so it returns exactly one
    of the agent's tools plus arguments. This is structured-output tool
    selection through the CLI, not the native tool-use API; the scores below
    should be described that way

    python3 -m evals.run --agent claude
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
from typing import Any, Callable, Optional

ISOLATION_FLAGS = [
    "--tools", "",
    "--setting-sources", "",
    "--disable-slash-commands",
    "--strict-mcp-config",
    "--no-session-persistence",
    "--output-format", "json",
]


def build_schema(tools: list) -> dict:
    return {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": [t["name"] for t in tools]},
            "arguments": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "replicas": {"type": "integer"},
                    "reason": {"type": "string"},
                    "summary": {"type": "string"},
                },
            },
            "note": {"type": "string", "description": "One sentence: why this tool, based on the evidence so far."},
        },
        "required": ["tool", "arguments", "note"],
    }


def build_prompt(tools: list, messages: list) -> str:
    lines = ["Available tools:"]
    for t in tools:
        props = ", ".join(t["input_schema"].get("properties", {})) or "no arguments"
        lines.append(f"- {t['name']}({props}): {t.get('description', '')}")
    lines.append("\nConversation so far:")
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            lines.append(f"[TASK] {content}")
            continue
        for b in content:
            if b["type"] == "text":
                lines.append(f"[YOUR NOTE] {b['text']}")
            elif b["type"] == "tool_use":
                lines.append(f"[YOU CALLED] {b['name']} {json.dumps(b['input'])}")
            elif b["type"] == "tool_result":
                tag = "TOOL ERROR" if b.get("is_error") else "TOOL RESULT"
                lines.append(f"[{tag}] {b['content']}")
    lines.append("\nChoose your next tool call.")
    return "\n".join(lines)


def _run_cli(argv: list, cwd: str) -> str:
    done = subprocess.run(argv, capture_output=True, text=True, cwd=cwd, timeout=180)
    if done.returncode != 0 and not done.stdout:
        raise RuntimeError(f"claude CLI failed: {done.stderr.strip()[:300]}")
    return done.stdout


class ClaudeCLIClient:
    def __init__(self, run: Optional[Callable[[list, str], str]] = None, claude_bin: Optional[str] = None) -> None:
        self._run = run or _run_cli
        self._bin = claude_bin or shutil.which("claude") or "claude"
        self._calls = 0
        self._workdir = tempfile.mkdtemp(prefix="sre-agent-empty-")
        self.messages = self  # client.messages.create(...)

    def create(self, *, model: str, system: Optional[str] = None, tools: Optional[list] = None, messages: list, **_: Any) -> Any:
        self._calls += 1
        tools = tools or []
        argv = [
            self._bin, "-p", build_prompt(tools, messages),
            "--model", model,
            "--system-prompt", (system or "") + "\nReply only with your next tool call as JSON.",
            "--json-schema", json.dumps(build_schema(tools)),
            *ISOLATION_FLAGS,
        ]
        data = json.loads(self._run(argv, self._workdir))
        if data.get("is_error"):
            raise RuntimeError(f"claude CLI error: {str(data.get('result'))[:300]}")
        out = data.get("structured_output") or json.loads(data.get("result") or "{}")
        blocks = []
        if out.get("note"):
            blocks.append(SimpleNamespace(type="text", text=out["note"]))
        blocks.append(
            SimpleNamespace(type="tool_use", id=f"call_{self._calls}", name=out["tool"], input=out.get("arguments") or {})
        )
        return SimpleNamespace(content=blocks, stop_reason="tool_use")
