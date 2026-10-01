# Project 5 — LLM Layer: Scenarios & Interpretation

**Status:** In review
**Depends on:** 0, 1   **Provides:** `generate_scenario(summary) -> Scenario`, `interpret(text, summary) -> Interpretation`, `check_feasibility(actions, state, role) -> FeasibilityReport`

## Goal
Natural language in and out, structured data in the middle. The LLM never touches WorldState: it reads a summary and returns Pydantic-validated objects.

## Scope
- In: state summary, scenario generator, interpreter, engine-side feasibility check, LLM client with retries, caching/recording, cost and latency logging, prompt versioning, eval cases.
- Out: outcome candidates (Project 6), news-seeded scenarios (Project 7), institutions as actors (Project 8).

## Design
- **One path for every call.** `llm/client.py: structured_call` sends a request, parses the JSON reply into the output model, runs a semantic check (node ids exist, list sizes, consistency), and on failure sends the reply and the problems back for another attempt (3 by default). Nothing unvalidated is returned.
- **Clients behind a protocol** (`LLMClient.complete(LLMRequest) -> LLMResponse`):
  - `AnthropicClient`: Claude API with structured outputs (`output_config.format`), adaptive thinking with per-call `effort`, server-side refusal fallback. Needs `uv sync --extra llm` and `ANTHROPIC_API_KEY`.
  - `FakeClient`: canned replies or a responder function. Used by the tests.
  - `RecordingClient`: on-disk cache keyed by request. Wrapping a live client records a cassette; alone it replays, so CI never needs a key. Keys include the prompt version and a schema fingerprint, so changed prompts never replay stale answers.
- **Models.** `ModelConfig` holds model and effort per call type. Both default to `claude-opus-5-5`; scenarios at medium effort, interpretation at low. A cheaper model can be swapped in per call type once there is cost data.
- **Cost/latency.** Every client keeps a `UsageLog` of `CallRecord`s (tokens, latency, USD estimate, cached flag) and logs each call.
- **Prompts** live in `llm/prompts/` with a `VERSION` string each.
- **Summary** (`llm/summary.py`): indicators with recent relative change, stressed indicators, angry groups, tense foreign countries, institutions, recent events, and the node catalogue the model may reference.
- **Feasibility** (`policy/feasibility.py`) is deterministic. Tax, spend and legislate need a legislative majority (Commons or Congress) for both roles; other kinds are executive. Spending rises and tax cuts need fiscal headroom (warning over 5% deficit, blocked over 10%). Independent institutions, and indicators only they drive (Bank Rate), cannot be directed; communicating with them is fine. Each action gets blockers, warnings and a 0..1 resistance score for later projects.

## Interface
```python
summarise_state(state, role="prime_minister", recent_events=None) -> StateSummary
generate_scenario(summary, client, config=None) -> GeneratedScenario   # Scenario + stakeholder_positions
interpret(text, summary, client, scenario=None, config=None) -> Interpretation
    # Interpretation(actions: list[PolicyAction], unmapped: list[str], clarifying_question: str | None)
check_feasibility(actions, state, role="prime_minister") -> FeasibilityReport
    # .checks[i]: feasible, requirements, blockers, warnings, resistance; .feasible_actions
```
Differences from §6a: `interpret` takes a `StateSummary` (not a raw state) and returns an `Interpretation` (actions plus an optional clarifying question). `GeneratedScenario` subclasses the core `Scenario` to add stakeholder positions, so core models are unchanged.

## Game loop adapters
`llm/adapters.py` has `LLMScenarioSource` and `LLMInterpreter`, which satisfy the `ScenarioSource` and `Interpreter` protocols in `game/interfaces.py` (Project 8 branch). The interpreter adapter drops infeasible actions and keeps the interpretation and feasibility report of the last call for the interface to show. Checked by playing turns of the Project 8 `Game` with these adapters and a fake client.

Follow-ups once Projects 3 and 8 merge: have generated scenarios emit `shocks` (the `Shock` model lives on the Project 3 branch), and persist `stakeholder_positions` (a `TurnRecord` currently saves the scenario as a plain `Scenario`).

## Tasks
- [x] Client protocol, fake and recording clients, Anthropic client
- [x] Validate-and-retry loop with error feedback
- [x] State summary
- [x] Scenario generator
- [x] Interpreter
- [x] Feasibility check
- [x] Game loop adapters
- [x] 20 eval cases (`llm/eval_cases.py`) with a scorer and live runner (`python -m hog_sim.llm.evaluate`)
- [ ] Live run of the eval with an API key; commit `tests/cassettes/interpreter.json`

## Done when
20 varied player responses map to sensible action lists; invalid outputs are caught and retried.
- Retry behaviour: covered by `tests/test_llm.py`.
- The 20 responses: the cases and scorer exist and CI replays the cassette once it is recorded. Needs one live run:
  `ANTHROPIC_API_KEY=... uv run --extra llm python -m hog_sim.llm.evaluate`

## Open questions
- Should the cheaper model handle interpretation? Decide from the first live eval's cost and pass rate.
- Feasibility thresholds (majority at 0.5 support, 5%/10% deficit) are placeholders until Project 8 models institutions properly.
