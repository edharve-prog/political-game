# Political Game (Head of Government Simulator)

A turn-based simulation where you play Prime Minister or President. Each turn a scenario arrives
(from real news or generated from the world state), you respond, the game forecasts possible
outcomes and selects one, and the result reshapes a world of countries, economic sectors and
population groups that feeds the next scenario.

The full plan is in [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md). Each numbered project
gets its own file under `docs/projects/` when it is picked up.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run hog-sim            # play in the terminal (offline, rule-based stand-ins)
uv run hog-sim --resume   # continue the last save
uv run --extra llm hog-sim --check-llm   # one small call to confirm Claude is reachable
uv run --extra llm hog-sim --llm         # play with Claude (needs ANTHROPIC_API_KEY)
```

The game prints its mode at startup. Without `--llm` it says OFFLINE PRACTICE: five built-in
scenarios and keyword matching, no Claude. With `--llm` it names the Claude model, and each turn
ends with a line counting the Claude calls, tokens and cost so far.

## Layout

- `src/hog_sim/core/` — shared schemas (`models.py`), world state container (`state.py`), config and seeded RNG.
- `tests/` — pytest suite.
