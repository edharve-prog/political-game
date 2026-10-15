"""Balance check: scripted strategy bots play full offline games (Project 16, EB-1).

Each bot plays the same response every turn, either as fixed engine actions (to probe one
lever at a time) or as text through the keyword interpreter (to play like a person), or
follows a schedule of actions by turn (``SCHEDULE_BOTS``: act early, then coast). Games use
only the offline plug-ins, so the run is deterministic and needs no Claude.

The exploit search (D4, ``search``) looks for the next dominant strategy before a player does:
it plays seeded random and greedy packages (kind x target x size x turns) once on turn 0 of a
quiet game with no outcomes, and flags any package that costs nothing in deficit yet beats
doing nothing at the election by more than ``LIMITS["max_free_package_gain"]``.

``uv run python -m hog_sim.game.balance`` prints the results table (``--search N`` runs the
exploit search with N packages); ``tests/test_balance.py`` asserts the limits in ``LIMITS``.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from statistics import fmean
from typing import get_args

from hog_sim.content.library import ScenarioLibrary
from hog_sim.core.config import GameConfig
from hog_sim.core.models import ActionKind, Model, Outcome, PolicyAction, Scenario
from hog_sim.core.state import WorldState
from hog_sim.game.loop import Game, resolve
from hog_sim.game.stubs import EngineForecaster, KeywordInterpreter
from hog_sim.policy.capital import available
from hog_sim.policy.feasibility import compatibility_problem
from hog_sim.policy.limits import constrain, lifetime_cost
from hog_sim.population.popularity import run_election
from hog_sim.world.seed.toy import toy_world

# Pass marks for the balance test, in one place.
LIMITS = {
    "max_single_lever_win_share": 0.34,  # a bot pulling one lever every turn
    "max_do_nothing_win_share": 0.0,
    "max_share_of_turns_at_a_bound": 0.1,  # bounds are a backstop, not what keeps things sane
    # Acting must not be punished on average: the lever bots' mean vote may trail doing
    # nothing by at most this much (EB-7). It was 0.005 until the fuller electorate (O8): with
    # every voter counted, pensioners are 23% of the vote rather than 38%, and the levers that
    # buy one group with borrowing now cost more voters than they win, so the gap is about 0.8
    # points. Group effects are retuned on top of the fuller electorate (review 2026-10-13, E6),
    # which should bring this back down. The Bank's rule and gilt yields (O4) also lean against
    # fiscal stimulus, and the opposition (O9), which goes after the groups a lever leaves
    # disaffected, widens the gap a little (0.47 to 0.55 points on its own).
    "max_acting_penalty": 0.012,
    # A package that costs nothing in deficit, enacted on turn 0 within one turn's capital,
    # may beat doing nothing at the election by at most this share of the vote (D4).
    "max_free_package_gain": 0.005,
}


def act(kind: str, target: str, magnitude: float, turns: int = 1) -> PolicyAction:
    return PolicyAction(kind=kind, target=target, magnitude=magnitude, duration_turns=turns)


Strategy = Callable[[WorldState, Scenario], list[PolicyAction]]

# One lever, pulled as hard as possible every turn.
LEVER_BOTS: dict[str, list[PolicyAction]] = {
    "deregulate energy": [act("deregulate", "sector:energy", 1.0)],
    "deregulate housing": [act("deregulate", "sector:housing", 1.0)],
    "spend on public services": [act("spend", "sector:public", 1.0)],
    "legislate for public services": [act("legislate", "sector:public", 1.0)],
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

# Act early, then coast: actions by turn, nothing on any other turn.
_GROUPS = ["group:pensioners", "group:young_renters", "group:public_workers", "group:business"]
SCHEDULE_BOTS: dict[str, dict[int, list[PolicyAction]]] = {
    # The review of 13 October's dominant strategy (E2): on turns 0 to 3 pass a 24-turn law and
    # give a 24-turn speech for one group each, then do nothing. It won 5 of 5 games.
    "four-turn laws": {
        turn: [act("legislate", group, 1.0, turns=24), act("communicate", group, 1.0, turns=24)]
        for turn, group in enumerate(_GROUPS)
    },
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


class _Scheduled:
    def __init__(self, schedule: Mapping[int, list[PolicyAction]]) -> None:
        self.schedule = schedule

    def interpret(self, text: str, state: WorldState, scenario: Scenario) -> list[PolicyAction]:
        return [a.model_copy(deep=True) for a in self.schedule.get(state.turn, [])]


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
    elif name in SCHEDULE_BOTS:
        interpreter, respond = _Scheduled(SCHEDULE_BOTS[name]), (lambda sc: name)
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
            lever=name in LEVER_BOTS or name in SCHEDULE_BOTS,
            games=[play(name, seed, turns) for seed in range(seeds)],
        )
        for name in [*TEXT_BOTS, *LEVER_BOTS, *SCHEDULE_BOTS]
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


# The exploit search (D4). Games are quiet (no scenario shocks, no outcomes) and skip the
# forecast, so a package plays out in a fraction of a second; the limits and resolve are the
# game's own.
_QUIET = Scenario(title="A quiet month", briefing="", affected_nodes=[], urgency=0.0)
_NO_OUTCOME = Outcome(narrative="Nothing else happens.", probability=1.0)
SEARCH_SIZES = (-1.0, -0.5, 0.5, 1.0)
SEARCH_TURNS = (1, 6, 24)
SEARCH_MAX_MEASURES = 3


def quiet_game(schedule: Mapping[int, Sequence[PolicyAction]], seed: int = 0, turns: int = 24):
    """Play ``schedule`` (actions by turn) through the game's limits and resolve, with quiet
    scenarios and no outcomes, and hold the election."""
    config = GameConfig(seed=seed, election_turn=turns)
    start = toy_world()
    state, past = start, []
    for turn in range(turns):
        limited = constrain(
            list(schedule.get(turn, [])), state, past, config.role, available(state, config)
        )
        past.append(limited.actions)
        state = resolve(start, state, _QUIET, limited.actions, _NO_OUTCOME, config)
    return run_election(state)


class Probe(Model):
    """One package the search played: what it gained over doing nothing, and what it cost."""

    actions: list[PolicyAction]
    gain: float  # vote share against doing nothing, at the election
    deficit_pts: float  # deficit the package adds, points of GDP summed over the turns paid
    free: bool  # no measure in it costs anything in deficit

    def describe(self) -> str:
        parts = [
            f"{a.kind} {a.target} {a.magnitude:+.2f} for {a.duration_turns}" for a in self.actions
        ]
        return "; ".join(parts)


def search_space(state: WorldState | None = None) -> list[PolicyAction]:
    """Every single measure the search may try: each kind on each target it fits, at each of
    ``SEARCH_SIZES`` and ``SEARCH_TURNS``."""
    state = state or toy_world()
    kinds = [k for k in get_args(ActionKind) if k != "do_nothing"]
    return [
        act(kind, target, size, turns)
        for kind in kinds
        for target in sorted(state.node_ids())
        if compatibility_problem(act(kind, target, 1.0), state) is None
        for size in SEARCH_SIZES
        for turns in SEARCH_TURNS
    ]


def search(budget: int = 60, seed: int = 0, turns: int = 24) -> list[Probe]:
    """Play ``budget`` packages on turn 0, best gain first: half are single measures drawn at
    random from ``search_space``, the rest add a random measure to one of the best packages so
    far that cost nothing in deficit (greedy). Seeded, so a run is reproducible."""
    rng = random.Random(f"exploit-search:{seed}")
    start = toy_world()
    space = search_space(start)
    idle = quiet_game({}, seed, turns).vote_share
    probes: dict[str, Probe] = {}

    def play_package(actions: list[PolicyAction]) -> None:
        applied = constrain(actions, start, []).actions
        costs = [lifetime_cost(a, start).deficit_turn_pts for a in applied]
        probe = Probe(
            actions=actions,
            gain=quiet_game({0: actions}, seed, turns).vote_share - idle,
            deficit_pts=sum(costs),
            free=all(abs(c) < 1e-9 for c in costs),
        )
        probes.setdefault(probe.describe(), probe)

    singles = budget // 2
    for action in rng.sample(space, min(singles, len(space))):
        play_package([action])
    while len(probes) < budget:
        # Build on the best packages that cost nothing in deficit and have room for more.
        open_ = [p for p in probes.values() if len(p.actions) < SEARCH_MAX_MEASURES]
        free = [p for p in open_ if p.deficit_pts <= 1e-9] or open_
        ranked = sorted(free, key=lambda p: -p.gain)[:5]
        base = rng.choice(ranked).actions if ranked else []
        extra = rng.choice(space)
        if any(a.target == extra.target and a.kind == extra.kind for a in base):
            continue
        play_package([*base, extra])
    return sorted(probes.values(), key=lambda p: -p.gain)


def exploits(probes: Sequence[Probe]) -> list[str]:
    """Packages that cost nothing in deficit (no measure in them, or the package overall) yet
    beat doing nothing by more than ``LIMITS["max_free_package_gain"]``."""
    margin = LIMITS["max_free_package_gain"]
    return [
        f"{p.describe()}: {p.gain:+.1%} of the vote over doing nothing, "
        f"deficit {p.deficit_pts:+.2f} pts of GDP x turns"
        for p in probes
        if p.deficit_pts <= 1e-9 and p.gain > margin
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Play scripted strategies and report balance")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--turns", type=int, default=24)
    parser.add_argument(
        "--search", type=int, default=0, metavar="N", help="also search N packages for exploits"
    )
    parser.add_argument("--search-seed", type=int, default=0, help="seed for the search")
    args = parser.parse_args()
    results = run(args.seeds, args.turns)
    print(table(results))
    found = problems(results)
    if args.search:
        probes = search(args.search, seed=args.search_seed, turns=args.turns)
        print(f"\nExploit search: {len(probes)} packages, best five:")
        print("\n".join(f"  {p.gain:+.2%}  {p.describe()}" for p in probes[:5]))
        found += exploits(probes)
    print("\n".join(found) or "Within limits.")
    if found:
        sys.exit(1)


if __name__ == "__main__":
    main()
