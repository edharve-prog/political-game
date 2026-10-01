# Project 6 — Outcome Forecasting & Selection

**Status:** In progress (first pass in review)
**Depends on:** 3, 4, 5   **Provides:** `LLMForecaster.forecast(state, scenario, actions, engine) -> [Outcome]`, `select(candidates, mode, rng)`

## Design
1. **Engine first.** The game loop runs Monte Carlo propagation for the actions and the scenario's shocks. `engine_text` summarises the expected change and the 10–90% range for each moving indicator, three turns out.
2. **Candidates** (`forecasting/candidates.py`, prompt `llm/prompts/outcomes.py`): one structured call proposes `n_candidates` (4 by default). It is told to include the expected path, a worse one and a better one. Each `CandidateDraft` has:
   - a title and narrative
   - claimed indicator shifts three turns out
   - one-off group approval effects (±0.1)
   - knock-on shocks (±1 standard step)
   - event tags from a fixed list
   - a self-reported probability

   Semantic checks reject unknown ids, the wrong count, duplicate titles and empty tags. Rejected output is retried with feedback through Project 5's `structured_call`.
3. **Judge** (`llm/prompts/judge.py`): a separate call estimates each candidate's probability with forecaster reasoning (base rates first, penalise unexplained departures from the engine, ignore vivid writing). It can be turned off (`use_judge=False`) to save a call; the self-reported probability is then used instead.
4. **Ensemble** (`forecasting/scoring.py`) blends three scores log-linearly, with weights 0.4 / 0.4 / 0.2 by default, then normalises:
   - **Consistency:** the geometric mean of exp(−z²/2) over the claimed shifts, where z is measured against a normal fitted to the engine's p10 and p90 (with a small floor on the spread).
   - **Judge:** the judge's probability.
   - **Base rate:** the geometric mean of `BASE_RATES` for the candidate's tags. These are placeholders until calibration in Project 11.
5. **Convert:** group effects become an `ApprovalEvent`, knock-on events become `Shock`s, tags become `Outcome.events`, and the score breakdown goes in `Outcome.scores`, which is logged in every TurnRecord. The engine's numbers stay authoritative; claimed shifts are kept for display only.
6. **Select:** `argmax` or `sample` (the default), as in Project 8.

## Game wiring
- `game/llm_plugins.py: llm_plugins(client)` returns the scenario source, interpreter and forecaster, all LLM-backed.
- `ClarifyingInterpreter` raises `NeedsClarification` when the response is too vague. This happens before any state changes, so the CLI shows the question and asks again.
- `dropped()` lists actions that feasibility blocked.
- `uv run hog-sim --llm` plays with Claude. It needs `uv sync --extra llm` and `ANTHROPIC_API_KEY`.

## Cost per turn (LLM mode)
Four calls per turn: scenario, interpretation, candidates and judge. A turn that needs clarification adds another interpretation call.

## Done when
Candidates are diverse and the scores are calibrated well enough that selected outcomes look sensible on manual review of 30 turns. Scoring, conversion, retries and a full fake-client game with replay are covered by `tests/test_forecasting.py`. **The 30-turn manual review still needs a live run with an API key.**

## Open questions
- Should weights be tuned against the backtests in Project 11?
- Should candidates claim shifts at more than one horizon?
- Show the "what nearly happened" view (unselected candidates) in the interface (Project 10)?
