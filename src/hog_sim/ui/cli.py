"""Minimal text interface: ``uv run hog-sim`` (add ``--resume`` to continue the last save)."""

from __future__ import annotations

import argparse
from pathlib import Path

from hog_sim.core.config import GameConfig
from hog_sim.core.state import WorldState
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Head of Government Simulator")
    parser.add_argument("--save", default="saves/game.db")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    store = SaveStore(args.save)
    plugins = (CannedScenarios(args.seed), KeywordInterpreter(), EngineForecaster())
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
            response = input("\nYour response> ")
            record = game.play_turn(response)
            store.save_turn(game_id, record)
            print("\n" + _report(record))
        print("\n" + _dashboard(game.start, game.state))
    except (EOFError, KeyboardInterrupt):
        print(f"\nSaved. Resume with: hog-sim --resume --save {args.save}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
