"""Minimal text interface: ``uv run hog-sim`` (add ``--resume`` to continue the last save)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hog_sim.core.config import GameConfig
from hog_sim.core.state import WorldState
from hog_sim.game.interfaces import NeedsClarification
from hog_sim.game.loop import Game
from hog_sim.game.persistence import SaveStore
from hog_sim.game.records import TurnRecord
from hog_sim.game.stubs import CannedScenarios, EngineForecaster, KeywordInterpreter
from hog_sim.population.popularity import national_approval, vote_intention
from hog_sim.world.seed.toy import toy_world


def _dashboard(start: WorldState, state: WorldState) -> str:
    lines = [f"Turn {state.turn}"]
    for ind in state.indicators.values():
        delta = ind.value - start.indicators[ind.id].value
        lines.append(f"  {ind.name:<26}{ind.value:>8.2f} {ind.unit:<6} ({delta:+.2f})")
    lines.append("  Approval")
    for g in state.groups.values():
        lines.append(f"    {g.name:<24}{g.approval * 100:>6.1f}%")
    lines.append(
        f"  National {national_approval(state) * 100:.1f}%  ·  "
        f"Vote intention {vote_intention(state) * 100:.1f}%"
    )
    return "\n".join(lines)


def _report(record: TurnRecord) -> str:
    lines = [record.outcome.narrative]
    lines += [f"  (action: {a.kind} {a.target} {a.magnitude:+.2f})" for a in record.actions]
    if record.election:
        e = record.election
        verdict = "You win a majority." if e.majority else "You lose your majority."
        lines.append(
            f"ELECTION: {e.vote_share * 100:.1f}% of the vote, "
            f"{e.seats}/{e.total_seats} seats. {verdict}"
        )
    return "\n".join(lines)


OFFLINE_BANNER = (
    "Mode: OFFLINE PRACTICE. Scenarios come from a short built-in list and responses are\n"
    "matched by keywords; Claude is not used. Run with --llm to play with Claude."
)


def _llm_client():
    """An AnthropicClient, or exit with a plain explanation of what is missing."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        raise SystemExit(
            "--llm needs the Claude SDK. Install it with: uv sync --extra llm"
        ) from None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "--llm needs an API key. Set ANTHROPIC_API_KEY first, for example\n"
            '  PowerShell:  $env:ANTHROPIC_API_KEY = "sk-ant-..."\n'
            "  cmd:         set ANTHROPIC_API_KEY=sk-ant-...\n"
            "  bash/zsh:    export ANTHROPIC_API_KEY=sk-ant-..."
        )
    from hog_sim.llm.client import AnthropicClient

    return AnthropicClient()


def _llm_banner(model_config, forecast_config) -> str:
    models = sorted(
        {
            model_config.scenario_model,
            model_config.interpret_model,
            forecast_config.outcome_model,
            forecast_config.judge_model,
        }
    )
    return (
        f"Mode: CLAUDE ({', '.join(models)}). Scenarios, your responses and outcomes are\n"
        "written by Claude. Each turn ends with a line counting the calls it made."
    )


def check_llm(client) -> str:
    """Make one tiny call and say which model answered. Raises if the API is unreachable."""
    from hog_sim.core.models import Model
    from hog_sim.llm.client import structured_call

    class Ping(Model):
        reply: str

    result = structured_call(
        client,
        output_type=Ping,
        system="You are a connectivity check.",
        prompt="Reply with the single word: connected",
        model="claude-opus-5-5",
        prompt_version="ping-1",
        effort="low",
        max_tokens=2000,
        max_attempts=1,
    )
    record = client.usage.records[-1]
    served = record.served_model or record.model
    return (
        f"Claude is connected. {served} replied {result.reply!r} "
        f"({record.input_tokens} tokens in, {record.output_tokens} out, "
        f"{record.latency_s:.1f}s)."
    )


def _usage_line(client, since: int) -> str:
    calls = client.usage.records[since:]
    served = sorted({r.served_model or r.model for r in calls})
    tokens_in = sum(r.input_tokens for r in calls)
    tokens_out = sum(r.output_tokens for r in calls)
    return (
        f"  [Claude: {len(calls)} calls this turn via {', '.join(served) or 'none'}, "
        f"{tokens_in} tokens in, {tokens_out} out; "
        f"${client.usage.total_cost_usd:.2f} so far]"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Head of Government Simulator")
    parser.add_argument("--save", default="saves/game.db")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--llm",
        action="store_true",
        help="use Claude for scenarios, interpretation and outcomes (needs ANTHROPIC_API_KEY "
        "and `uv sync --extra llm`)",
    )
    parser.add_argument(
        "--check-llm",
        action="store_true",
        help="make one small call to Claude to confirm the key works, then exit",
    )
    args = parser.parse_args(argv)

    if args.check_llm:
        print(check_llm(_llm_client()))
        return

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    store = SaveStore(args.save)
    client = None
    if args.llm:
        from hog_sim.forecasting.candidates import ForecastConfig
        from hog_sim.game.llm_plugins import llm_plugins
        from hog_sim.llm.client import ModelConfig

        client = _llm_client()
        model_config, forecast_config = ModelConfig(), ForecastConfig()
        plugins = llm_plugins(client, model_config=model_config, forecast_config=forecast_config)
        print(_llm_banner(model_config, forecast_config))
    else:
        plugins = (CannedScenarios(args.seed), KeywordInterpreter(), EngineForecaster())
        print(OFFLINE_BANNER)
    game_id = store.latest_game() if args.resume else None
    if game_id:
        config, start, records = store.load(game_id)
        game = Game.resume(config, start, records, *plugins)
        print(f"Resumed game {game_id} at turn {game.state.turn}.")
    else:
        config, start = GameConfig(seed=args.seed), toy_world()
        game_id = store.new_game(config, start)
        game = Game(config, start, *plugins)
        print(f"New game {game_id}. Election at turn {config.election_turn}. Ctrl-D to quit.")

    try:
        while not game.over:
            print("\n" + _dashboard(game.start, game.state))
            s = game.scenario
            print(f"\n== {s.title} ==\n{s.briefing}")
            for option in s.suggested_options:
                print(f"  - {option}")
            calls_before = len(client.usage.records) if client else 0
            while True:
                response = input("\nYour response> ")
                try:
                    record = game.play_turn(response)
                    break
                except NeedsClarification as ask:
                    print(f"\nYour advisers ask: {ask.question}")
            store.save_turn(game_id, record)
            for line in getattr(game.interpreter, "dropped", lambda: [])():
                print(f"  (not possible: {line})")
            print("\n" + _report(record))
            if client:
                print(_usage_line(client, calls_before))
        print("\n" + _dashboard(game.start, game.state))
    except (EOFError, KeyboardInterrupt):
        print(f"\nSaved. Resume with: hog-sim --resume --save {args.save}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
