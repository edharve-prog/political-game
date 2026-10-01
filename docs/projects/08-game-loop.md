# Project 8 — Game Loop, Institutions & Persistence

**Status:** In progress (first pass in review: the loop, saves and replay; institutions not started)
**Depends on:** 3, 4, plus 5 and 6 through interfaces   **Provides:** `Game.play_turn()`, `Game.resume()`, `replay()`, `SaveStore`

## Design
- **Plug-in interfaces** (`game/interfaces.py`): `ScenarioSource.next_scenario(state, history)`, `Interpreter.interpret(text, state, scenario)` and `Forecaster.forecast(state, scenario, actions, engine)`. The loop only talks to these, so the LLM versions from Projects 5 and 6 can drop in.
- **Stubs** (`game/stubs.py`) make the game playable offline. `CannedScenarios` has five seeded scenarios, each carrying exogenous shocks. `KeywordInterpreter` turns text into a PolicyAction by keyword rules. `EngineForecaster` produces three candidates: as expected (60%), backlash (25%) and welcomed (15%), with approval events hitting the groups exposed to the action.
- **decide / resolve split.** `play_turn` runs the parts that may call an LLM: interpret, Monte Carlo forecast, candidates, then selection. `resolve` is pure and deterministic, and uses only logged data. `replay(start, config, records)` rebuilds a game without any LLM calls.
- **Effects across turns.** `resolve` runs the expected engine trajectory for the actions, the scenario's shocks and the outcome's shocks. It then writes per-turn increments into `WorldState.pending` (absolute turn → node → native change). Each turn applies its slot. This answers the "carrying effects across turns" question from Project 3.
- **Approval reference** is the start state.
- **Selection** (`forecasting/selection.py`): `argmax` or `sample` (the default), seeded per turn.
- **Saves** (`game/persistence.py`) use SQLite. `games` holds the config and start state; `turns` holds one full `TurnRecord` JSON per turn (scenario, response, actions, all candidates, the chosen index, state after, election).
- **CLI** (`ui/cli.py`, `hog-sim`): a dashboard, the scenario briefing with suggested options, and a free-text response. `--resume` continues the last save. Ctrl-D saves and quits.
- **Election** at `GameConfig.election_turn` (default 24), using Project 4's seat model.

## Model changes
`Shock` moved into `core/models.py`. `Scenario.shocks`, `Outcome.approval_events`, `Outcome.shocks`, `WorldState.pending` and `GameConfig.horizon/k_draws` were added.

## Done when
A full 12-turn game can be played, saved mid-way, resumed and replayed deterministically from the log. This is met by `tests/test_game.py`.

## Not yet done
- **Institutions as actors and constraints**: legislature votes, central bank reaction, courts. The feasibility check is planned in Project 5, so this waits to see how that lands.
- **Scenario source mix**: news vs generated vs scheduled events such as budgets. This needs Project 7.
