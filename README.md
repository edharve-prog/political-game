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
uv run hog-sim              # play with Claude (see "Connecting to Claude")
uv run hog-sim --resume     # continue the last save
uv run hog-sim --check-llm  # one small call to confirm Claude is reachable
uv run hog-sim --offline    # offline practice: built-in scenarios, keyword matching
```

The game prints its mode at startup. It plays with Claude whenever it can reach it: the banner
names the Claude model, and each turn ends with a line counting the Claude calls, tokens and cost
so far. When it can't reach Claude it says why, explains how to connect, and starts OFFLINE
PRACTICE (67 built-in scenarios across 12 categories, keyword matching). `--offline` picks practice mode on
purpose; `--llm` stops with the explanation instead of falling back.

## Connecting to Claude

The game reaches Claude in one of two ways, chosen with `--provider` (default `claude-code`):

- **`claude-code`: your Claude subscription, no key.** Install [Claude Code](https://claude.com/claude-code),
  run `claude` once and sign in. The game then runs Claude Code in headless mode (`claude -p`),
  so calls count against your plan's usage limits. This is for playing on your own machine;
  anyone else needs their own sign-in.
- **`api`: the Anthropic API** (install the SDK first with `uv sync --extra llm`), billed per token to a [Console](https://console.anthropic.com)
  account. Either put `ANTHROPIC_API_KEY=sk-ant-...` in a `.env` file next to `pyproject.toml`
  (see `.env.example`; the file is git-ignored, so no environment variables to set on Windows),
  or sign in through the browser with the `ant` CLI: `ant auth login`.

When Claude Code isn't installed or isn't signed in and an API key or `ant` sign-in exists,
the game falls back to the API and says so when it starts. Check the connection with
`hog-sim --check-llm`; it says which route it used.

## Layout

- `src/hog_sim/core/` — shared schemas (`models.py`), world state container (`state.py`), config and seeded RNG.
- `tests/` — pytest suite.
