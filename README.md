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
uv sync --extra llm && uv run hog-sim --llm   # play with Claude (needs ANTHROPIC_API_KEY)
```

## Layout

- `src/hog_sim/core/` — shared schemas (`models.py`), world state container (`state.py`), config and seeded RNG.
- `tests/` — pytest suite.
