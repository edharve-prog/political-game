"""Balance check: scripted strategy bots play full offline games (Project 16, EB-1).

Each bot plays the same response every turn, either as fixed engine actions (to probe one
lever at a time) or as text through the keyword interpreter (to play like a person). Games
use only the offline plug-ins, so the run is deterministic and needs no Claude.

``uv run python -m hog_sim.game.balance`` prints the results table; ``tests/test_balance.py``
asserts the limits in ``LIMITS``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from statistics import fmean

from hog_sim.content.library import ScenarioLibrary
from hog_sim.core.config import GameConfig
from hog_sim.core.models import Model, PolicyAction, Scenario
from hog_sim.core.state import WorldState
from hog_sim.game.loop import Game
from hog_sim.game.stubs import EngineForecaster, KeywordInterpreter
from hog_sim.world.seed.toy import toy_world

# Pass marks for the balance test, in one place.
LIMITS = {
    "max_single_lever_win_share": 0.34,  # a bot pulling one lever every turn
    "max_do_nothing_win_share": 0.0,
    "max_share_of_turns_at_a_bound": 0.1,  # bounds are a backstop, not what keeps things sane
    # Acting must not be punished on average: the lever bots' mean vote may trail doing
    # nothing by at most this much (EB-7).
    "max_acting_penalty": 0.005,
}


def act(kind: str, target: str, magnitude: float, turns: int = 1) -> PolicyAction:
    return PolicyAction(kind=kind, target=target, magnitude=magnitude, duration_turns=turns)


Strategy = Callable[[WorldState, Scenario], list[PolicyAction]]

# One lever, pulled as hard as possible every turn.
LEVER_BOTS: dict[str, list[PolicyAction]] = {
    "deregulate energy": [act("deregulate", "sector:energy", 1.0)],
    "deregulate housing": [act("deregulate", "sector:housing", 1.0)],
    "spend on public services": [act("spend", "sector:public", 1.0)],
    "spend on housebuilding": [act("spend", "sector:housing", 1.0)],
    "cut taxes on industry": [act("tax", "sector:manufacturing", -1.0)],
    "tax the banks": [act("tax", "sector:finance", 1.0)],
    "raise pensions": [act("spend", "group:pensioners", 1.0)],
    "cap energy bills": [act("regulate", "indicator:energy_prices", -1.0)],
    "court the EU": [act("diplomatic", "country:eu", 1.0)],
    "reassure young renters": [act("communicate", "group:young_renters", 1.0)],
    "spend on everything": [
        act("spend", "sector:public", 1.0),
        act("spend", "sector:housing", 1.0),
        act("spend", "group:pensioners", 1.0),
    ],
    "year-long energy deregulation": [act("deregulate", "sector:energy", 1.0, turns=12)],
}

# Typed responses, through the keyword interpreter.
TEXT_BOTS: dict[str, Callable[[Scenario], str]] = {
    "do nothing": lambda sc: "do nothing",
    "first suggested option": lambda sc: (
        sc.suggested_options[0] if sc.suggested_options else "do nothing"
    ),
}


class _Scripted:
    def __init__(self, actions: list[PolicyAction]) -> None:
        self.actions = actions

    def interpret(self, text: str, state: WorldState, scenario: Scenario) -> list[PolicyAction]:
        return [a.model_copy(deep=True) for a in self.actions]


class GameResult(Model):
    vote_share: float
    seats: int
    majority: bool
    turns_at_bound: int
    indicator_turns: int
    final: dict[str, float]


class BotResult(Model):
    name: str
    lever: bool
    games: list[GameResult]

    @property
    def win_share(self) -> float:
        return sum(g.majority for g in self.games) / len(self.games)

    @property
    def bound_share(self) -> float:
        return sum(g.turns_at_bound for g in self.games) / sum(
            g.indicator_turns for g in self.games
        )


def play(name: str, seed: int, turns: int = 24) -> GameResult:
    config = GameConfig(seed=seed, election_turn=turns, k_draws=20)
    if name in LEVER_BOTS:
        interpreter, respond = _Scripted(LEVER_BOTS[name]), (lambda sc: name)
    else:
        interpreter, respond = KeywordInterpreter(), TEXT_BOTS[name]
    game = Game(
        config, toy_world(), ScenarioLibrary.load(seed=seed), interpreter, EngineForecaster()
    )
    at_bound = checked = 0
    record = None
    while not game.over:
        record = game.play_turn(respond(game.scenario))
        for ind in game.state.indicators.values():
            checked += 1
            at_bound += ind.value in (ind.low, ind.high)
    assert record is not None and record.election is not None
    return GameResult(
        vote_share=record.election.vote_share,
        seats=record.election.seats,
        majority=record.election.majority,
        turns_at_bound=at_bound,
        indicator_turns=checked,
        final={i.id.split(":", 1)[1]: round(i.value, 2) for i in game.state.indicators.values()},
    )


def run(seeds: int = 3, turns: int = 24) -> list[BotResult]:
    return [
        BotResult(
            name=name,
            lever=name in LEVER_BOTS,
            games=[play(name, seed, turns) for seed in range(seeds)],
        )
        for name in [*TEXT_BOTS, *LEVER_BOTS]
    ]


def table(results: list[BotResult]) -> str:
    lines = [
        f"{'Strategy':<30}{'Vote':>7}{'Seats':>12}{'Wins':>7}{'At bound':>10}  Last game's end",
    ]
    for r in results:
        seats = [g.seats for g in r.games]
        lines.append(
            f"{r.name:<30}{fmean(g.vote_share for g in r.games):>7.1%}"
            f"{f'{min(seats)}-{max(seats)}':>12}"
            f"{f'{sum(g.majority for g in r.games)}/{len(r.games)}':>7}"
            f"{r.bound_share:>10.0%}  {r.games[-1].final}"
        )
    return "\n".join(lines)


def problems(results: list[BotResult]) -> list[str]:
    """Every way the results break ``LIMITS``; empty when the game is balanced enough."""
    found = []
    nothing = next((r for r in results if r.name == "do nothing"), None)
    levers = [r for r in results if r.lever]
    if nothing and levers:
        acting = fmean(fmean(g.vote_share for g in r.games) for r in levers)
        idle = fmean(g.vote_share for g in nothing.games)
        if idle - acting > LIMITS["max_acting_penalty"]:
            found.append(
                f"acting averages {acting:.1%} of the vote against {idle:.1%} for doing nothing"
            )
    for r in results:
        if r.lever and r.win_share > LIMITS["max_single_lever_win_share"]:
            found.append(f"{r.name!r} wins {r.win_share:.0%} of games")
        if r.name == "do nothing" and r.win_share > LIMITS["max_do_nothing_win_share"]:
            found.append(f"doing nothing wins {r.win_share:.0%} of games")
        if r.bound_share > LIMITS["max_share_of_turns_at_a_bound"]:
            found.append(
                f"{r.name!r} leaves indicators at a hard bound {r.bound_share:.0%} of turns"
            )
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Play scripted strategies and report balance")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--turns", type=int, default=24)
    args = parser.parse_args()
    results = run(args.seeds, args.turns)
    print(table(results))
    print("\n".join(problems(results)) or "Within limits.")


if __name__ == "__main__":
    main()
