# agentic-sre-responder

A small, guardrailed autonomous incident-response agent, built to explore a
question that comes up constantly in SRE work: **when is it actually safe to
let an agent act on an incident by itself, and what stops it when it's
wrong?**

This is a local demo and test harness, not a hosted service — it runs
against a synthetic, deterministic service model (no real infrastructure,
no real API calls unless you opt into the LLM planner with your own key).

## The idea

The agent runs a plain observe → plan → act loop:

1. **Observe** — pull metrics (memory/disk/error-rate) from the service.
2. **Plan** — a planner decides the next move: check logs, restart, scale,
   escalate to a human, or declare the incident resolved.
3. **Guardrail-check** — a policy layer that knows nothing about the
   incident, only about limits, checks the proposed action before it's
   allowed to run.
4. **Act** — the action executes, and the loop repeats until the incident
   resolves or the agent escalates.

Two planners are included, and both produce the same `Action` type, so
either can drive the loop:

- **`HeuristicPlanner`** — fixed rules (read logs before acting, don't repeat
  a remediation that already failed, only declare victory when metrics say
  so). Deterministic, needs no API key, and is what the tests and the demo
  script run by default.
- **`LLMPlanner`** — the same decision made by calling the Anthropic API
  instead of following fixed rules. It's a real, working integration, not a
  stub, but it needs your own `ANTHROPIC_API_KEY` to actually call out — this
  repo doesn't ship with one.

## Why the guardrails are a separate layer

The interesting failure mode for an autonomous agent isn't "it does nothing"
— it's "it keeps trying the same fix forever" or "it takes an action bigger
than anyone intended." `guardrails.py` enforces that independently of
whatever planner is running:

- a service only gets restarted once per incident — a second restart
  request is treated as the agent repeating itself, not a new idea, and
  forces an escalation instead
- scaling is capped at a fixed replica count
- the whole incident has a hard action budget, so even a planner that never
  converges (rule bug, bad LLM output, anything) still terminates in an
  escalation rather than looping

`tests/test_agent.py` has a planner stub that always asks to re-fetch logs
forever, specifically to prove the agent still stops and hands off to a
human instead of hanging.

## Two seeded scenarios

- **Memory leak** — heap usage near its limit, `OutOfMemoryError`s in the
  logs. A restart genuinely fixes this one. The agent restarts once and
  resolves the incident.
- **Disk full from a misconfigured logger** — disk usage near its limit,
  restart does *not* fix it (the log volume regrows immediately). The agent
  tries the standard remediation once, sees it didn't help, and escalates
  instead of retrying.

Run both scenarios and see the actual decision trace:

```bash
python3 scripts/demo.py
```

```
=== Scenario 1: memory leak (restart fixes it) ===
  1. GET_LOGS -> 4 recent log line(s) retrieved
  2. RESTART(heap exhaustion, clear it with a clean restart) -> restarted checkout-api (restart_count=1)
  3. RESOLVED -> metrics healthy, closing incident
  verdict: RESOLVED

=== Scenario 2: disk full from a misconfigured logger (restart does NOT fix it) ===
  1. GET_LOGS -> 3 recent log line(s) retrieved
  2. RESTART(disk pressure, try a restart before escalating) -> restarted checkout-api (restart_count=1)
  3. ESCALATE(restart did not resolve the incident; likely root cause is not restart-fixable) -> paged on-call: restart did not resolve the incident; likely root cause is not restart-fixable
  verdict: ESCALATED
```

To try the LLM-backed planner instead (requires `pip install anthropic` and
`export ANTHROPIC_API_KEY=...`):

```bash
python3 scripts/demo.py --llm
```

## Running the tests

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -v
```

8 tests, all deterministic — including one that directly proves the
guardrail stops a runaway/hallucinating planner rather than trusting it to
self-correct.

## Layout

```
agent/
  environment.py   synthetic service state + the two seeded incident scenarios
  tools.py         the actions the agent can call (metrics, logs, restart, scale, escalate)
  actions.py       the shared Action type both planners emit
  planner.py       HeuristicPlanner + LLMPlanner
  guardrails.py    the safety layer, independent of which planner is running
  agent.py         the observe/plan/check/act loop
scripts/demo.py    runs both scenarios and prints the transcript
tests/test_agent.py
```
