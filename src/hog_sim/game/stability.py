"""Long-run stability: 120 offline turns should neither explode nor flatline (Project 11, CA-2).

Each player plays one offline game far past the usual election, so the world has time to drift.
The run fails if any number turns non-finite, an indicator wanders too far or keeps growing,
indicators sit at a hard bound too often, or the world goes still while scenarios keep coming.

Players are the balance bots (``game/balance.py``) plus a seeded random player who picks one of
each scenario's suggested options. ``uv run python -m hog_sim.game.stability`` prints the table;
``tests/test_stability.py`` asserts ``LIMITS``. The nightly 120-turn run planned in CI-4 can
call ``run`` with more seeds.
"""

from __future__ import annotations

import argparse
import math
import random
from collections.abc import Callable
from statistics import fmean, pstdev

from hog_sim.content.library import ScenarioLibrary
from hog_sim.core.config import GameConfig
from hog_sim.core.models import Model, Scenario
from hog_sim.game.balance import LEVER_BOTS, TEXT_BOTS, _Scripted
from hog_sim.game.loop import Game
from hog_sim.game.stubs import EngineForecaster, KeywordInterpreter
from hog_sim.population.popularity import vote_intention
from hog_sim.world.propagation import scale
from hog_sim.world.seed.toy import toy_world

TURNS = 120
WINDOW = 48  # the closing stretch checked for flatlining

# Pass marks, in one place. Sizes are in standard steps (see world/propagation.py).
LIMITS = {
    "max_steps_from_start": 8.0,  # no indicator wanders further than this
    # An indicator's average distance from its start over the second half may not be much more
    # than over the first: a trend that keeps growing is a runaway even while it is still
    # inside the bound above. Averages, not peaks, so one late shock that fades is fine.
    "max_late_growth": 1.5,
    "late_growth_slack": 0.5,
    # ...and only once that late average is big enough to matter. A policy held for the whole
    # game through a long-lag link (housebuilding moves prices 12 turns later) settles slowly
    # and can look like growth while it is still well inside normal swings.
    "min_late_level_for_growth": 2.0,
    "max_share_of_turns_at_a_bound": 0.1,
    # Still moving at the end: vote intention's range and the share of indicators with any
    # spread over the closing window.
    "min_late_vote_range": 0.01,
    "min_late_indicator_spread": 0.02,
    "min_share_of_moving_indicators": 0.5,
}

PLAYERS = ["do nothing", "first suggested option", "random option", *LEVER_BOTS]
TEST_PLAYERS = ["do nothing", "random option", "spend on everything", "deregulate energy"]


class Run(Model):
    player: str
    seed: int
    steps: dict[str, list[float]]  # each indicator's distance from its start, per turn
    votes: list[float]
    turns_at_bound: int
    indicator_turns: int


def _respond(name: str, seed: int) -> Callable[[Scenario], str]:
    if name in LEVER_BOTS:
        return lambda sc: name
    if name == "random option":
        rng = random.Random(seed)
        return lambda sc: rng.choice(sc.suggested_options) if sc.suggested_options else "wait"
    return TEXT_BOTS[name]


def play(name: str, seed: int, turns: int = TURNS) -> Run:
    config = GameConfig(seed=seed, election_turn=turns, k_draws=20)
    interpreter = _Scripted(LEVER_BOTS[name]) if name in LEVER_BOTS else KeywordInterpreter()
    game = Game(
        config, toy_world(), ScenarioLibrary.load(seed=seed), interpreter, EngineForecaster()
    )
    respond = _respond(name, seed)
    start = game.state
    steps: dict[str, list[float]] = {i: [] for i in start.indicators}
    votes = []
    at_bound = checked = 0
    while not game.over:
        game.play_turn(respond(game.scenario))
        state = game.state
        for node, ind in state.indicators.items():
            steps[node].append((ind.value - start.indicators[node].value) / scale(state, node))
            checked += 1
            at_bound += ind.value in (ind.low, ind.high)
        votes.append(vote_intention(state))
    return Run(
        player=name,
        seed=seed,
        steps=steps,
        votes=votes,
        turns_at_bound=at_bound,
        indicator_turns=checked,
    )


def run(players: list[str] = PLAYERS, seeds: int = 1, turns: int = TURNS) -> list[Run]:
    return [play(name, seed, turns) for name in players for seed in range(seeds)]


def _peak(values: list[float]) -> float:
    return max((abs(v) for v in values), default=0.0)


def _level(values: list[float]) -> float:
    return fmean(abs(v) for v in values) if values else 0.0


def problems(runs: list[Run]) -> list[str]:
    """Every way the runs break ``LIMITS``; empty when the engine is stable enough."""
    found = []
    for r in runs:
        who = f"{r.player!r} (seed {r.seed})"
        numbers = [*r.votes, *(v for vs in r.steps.values() for v in vs)]
        if not all(math.isfinite(v) for v in numbers):
            found.append(f"{who} produced a non-finite number")
            continue
        moving = 0
        for node, values in r.steps.items():
            name = node.split(":", 1)[1]
            half = len(values) // 2
            early, late = _level(values[:half]), _level(values[half:])
            if _peak(values) > LIMITS["max_steps_from_start"]:
                found.append(f"{who}: {name} moved {_peak(values):.1f} steps from its start")
            growing = late > LIMITS["max_late_growth"] * early + LIMITS["late_growth_slack"]
            if growing and late >= LIMITS["min_late_level_for_growth"]:
                found.append(
                    f"{who}: {name} keeps growing (average {early:.2f} then {late:.2f} steps)"
                )
            moving += pstdev(values[-WINDOW:]) >= LIMITS["min_late_indicator_spread"]
        if moving / len(r.steps) < LIMITS["min_share_of_moving_indicators"]:
            found.append(f"{who}: only {moving} of {len(r.steps)} indicators still move")
        tail = r.votes[-WINDOW:]
        if max(tail) - min(tail) < LIMITS["min_late_vote_range"]:
            found.append(f"{who}: vote intention has flatlined at {tail[-1]:.1%}")
        if r.turns_at_bound / r.indicator_turns > LIMITS["max_share_of_turns_at_a_bound"]:
            share = r.turns_at_bound / r.indicator_turns
            found.append(f"{who} leaves indicators at a hard bound {share:.0%} of turns")
    return found


def table(runs: list[Run]) -> str:
    header = f"{'Player':<30}{'Seed':>5}{'Peak':>7}{'Early avg':>11}{'Late avg':>10}"
    lines = [f"{header}{'Vote range':>15}"]
    for r in runs:
        peaks = {n: _peak(v) for n, v in r.steps.items()}
        worst = max(peaks, key=lambda n: peaks[n])
        values = r.steps[worst]
        half = len(values) // 2
        lines.append(
            f"{r.player:<30}{r.seed:>5}{peaks[worst]:>7.2f}"
            f"{_level(values[:half]):>11.2f}{_level(values[half:]):>10.2f}"
            f"{f'{min(r.votes):.1%}-{max(r.votes):.1%}':>15}  worst: {worst.split(':', 1)[1]}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play long offline games and check stability")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--turns", type=int, default=TURNS)
    args = parser.parse_args()
    runs = run(seeds=args.seeds, turns=args.turns)
    print(table(runs))
    print("\n".join(problems(runs)) or "Stable.")


if __name__ == "__main__":
    main()
