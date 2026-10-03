# Political Game (Head of Government Simulator)

A turn-based simulation where you play Prime Minister or President. Each turn a scenario arrives
(from real news or generated from the world state), you respond, the game forecasts possible
outcomes and selects one, and the result reshapes a world of countries, economic sectors and
population groups that feeds the next scenario.

How the engine decides what happens is explained in plain English in
[docs/ENGINE_RULES.md](docs/ENGINE_RULES.md).

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

## What the game keeps from Claude

Every Claude turn is also filed in a knowledge store inside the save file: the scenario, how
your response was read, the candidate outcomes, and any lasting changes to the world graph
Claude proposed (checked and capped by the engine before they apply). Later Claude turns see
similar past situations as precedents, and offline practice reuses Claude's scenarios,
interpretations and outcomes when they fit.

```
uv run hog-sim knowledge stats                      # what has been kept
uv run hog-sim knowledge export knowledge.jsonl     # move it to another install
uv run hog-sim knowledge import knowledge.jsonl
uv run hog-sim knowledge export-scenarios s.jsonl   # scenario-library format
```

Add `--db <file>` after `knowledge` for a save file other than `saves/game.db`.

## Reading what was sent to the model

Every request the game sends to the model (system prompt, prompt, retries) and every reply
or error is logged to `saves/llm-log.db`, next to the save file. Repeated text is stored
once and each prompt is compressed against an earlier one of the same kind, so a full game
takes well under a megabyte.

```
uv run hog-sim logs list                  # one line per call, most recent last
uv run hog-sim logs list --turn 3         # also --game <id>, --kind ScenarioDraft, --errors
uv run hog-sim logs show last             # the full request and reply of one call (or an id)
uv run hog-sim logs export calls.jsonl    # everything as plain JSON lines
uv run hog-sim logs stats                 # how much space the log takes
```

Play with `--llm-log off` (or `HOG_SIM_LLM_LOG=off` in `.env`) to keep no log, or
`--llm-log <file>` to write it elsewhere. Deleting the file loses nothing the game needs.

## Connecting to Claude

The game reaches a model in one of three ways, chosen with `--provider` (default `claude-code`):

- **`claude-code`: your Claude subscription, no key.** Install [Claude Code](https://claude.com/claude-code),
  run `claude` once and sign in. The game then runs Claude Code in headless mode (`claude -p`),
  so calls count against your plan's usage limits. This is for playing on your own machine;
  anyone else needs their own sign-in.
- **`api`: the Anthropic API** (install the SDK first with `uv sync --extra llm`), billed per token to a [Console](https://console.anthropic.com)
  account. Either put `ANTHROPIC_API_KEY=sk-ant-...` in a `.env` file next to `pyproject.toml`
  (see `.env.example`; the file is git-ignored, so no environment variables to set on Windows),
  or sign in through the browser with the `ant` CLI: `ant auth login`.
- **`codex`: your ChatGPT plan through OpenAI Codex, no key.** Install the
  [Codex CLI](https://developers.openai.com/codex/cli) (`npm install -g @openai/codex`), run
  `codex login` and sign in with ChatGPT in the browser. Then play with
  `hog-sim --provider codex` (or put `HOG_SIM_PROVIDER=codex` in `.env`). The game runs
  `codex exec` read-only in an empty scratch folder, so calls count against your plan's Codex
  limits. It uses `gpt-6.1-sol`; set `HOG_SIM_CODEX_MODEL` in `.env` to pick another.
  Like `claude-code`, this is for playing on your own machine. Codex is only used when you
  ask for it, and an `OPENAI_API_KEY` in your environment is hidden from it so your ChatGPT
  sign-in is what gets used.

When Claude Code isn't installed or isn't signed in and an API key or `ant` sign-in exists,
the game falls back to the API and says so when it starts. Check the connection with
`hog-sim --check-llm` (or `hog-sim --provider codex --check-llm`); it says which route it used.

## Layout

- `src/hog_sim/core/` — shared schemas (`models.py`), world state container (`state.py`), config and seeded RNG.
- `src/hog_sim/llm/calllog.py` — the log of model requests and replies, read with `hog-sim logs`.
- `src/hog_sim/knowledge/` — the knowledge store (`store.py`), precedents for prompts (`recall.py`) and offline reuse (`offline.py`); graph-change rules are in `world/changes.py`.
- `tests/` — pytest suite.
