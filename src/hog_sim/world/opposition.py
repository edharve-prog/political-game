"""An opposition that moves (review 2026-10-13, O9).

Every quarter (``QUARTER`` turns) the opposition picks a stance aimed at some groups. A stance
pulls those groups' lean (where their approval of the government drifts when nothing happens)
towards the opposition by ``STANCES[...].pull``; the previous stance's pull is taken back
first, so only one stance acts at a time. The opposition goes after the voters most
disaffected with the government: it picks the stance whose groups' approval sits furthest
below their own lean, weighted by population share and turnout. On a tie it keeps its stance.

Each quarter there may also be a by-election (``BYELECTION_CHANCE``, seeded). The government
holds the seat with a chance that rises with its vote intention (``hold_chance``); a loss costs
Commons support ``BYELECTION_LOSS`` and a hold steadies it by ``BYELECTION_HOLD``. Seats won and
lost are kept in the record; at the general election every seat is fought again, so they
change the Commons now and not the final count.

Everything here is deterministic given the game's seed, so replays match.
"""

from __future__ import annotations

from hog_sim.core.config import make_rng
from hog_sim.core.models import ByElection, Model, Opposition
from hog_sim.core.state import WorldState

QUARTER = 3  # turns between the opposition's moves
BYELECTION_CHANCE = 0.5  # chance of a by-election in any quarter
BYELECTION_LOSS = 0.03  # Commons support lost when the government loses a by-election
BYELECTION_HOLD = 0.01  # Commons support gained when it holds the seat
LEGISLATURE = "institution:legislature"


class Stance(Model):
    name: str
    pitch: str  # what the opposition is saying, for prompts and the dashboard
    pull: dict[str, float]  # group id -> change to its lean (negative: away from government)


STANCES: dict[str, Stance] = {
    s.name: s
    for s in [
        Stance(
            name="cost of living",
            pitch="the government has lost control of prices and bills",
            pull={"group:pensioners": -0.015, "group:private_workers": -0.015},
        ),
        Stance(
            name="housing and youth",
            pitch="a generation locked out of a home of its own",
            pull={"group:young_renters": -0.02, "group:students": -0.02},
        ),
        Stance(
            name="public services",
            pitch="schools and hospitals starved of money and staff",
            pull={"group:public_workers": -0.02, "group:pensioners": -0.02},
        ),
        Stance(
            name="sound money",
            pitch="reckless borrowing will push up mortgages and taxes",
            pull={"group:business": -0.02, "group:homeowners": -0.015},
        ),
    ]
}


def pitch(opposition: Opposition) -> str:
    """The opposition's line, for prompts and the dashboard."""
    stance = STANCES.get(opposition.stance or "")
    return f"{stance.name}: {stance.pitch}" if stance else "no clear line yet"


def opposition_line(state: WorldState) -> str:
    """One line for prompts: the opposition's line and the by-election record so far."""
    record = state.opposition.byelections
    line = f"campaigning on {pitch(state.opposition)}"
    if record:
        lost = sum(1 for b in record if not b.held)
        last = record[-1]
        line += (
            f"; by-elections: {len(record) - lost} held, {lost} lost "
            f"(last on turn {last.turn}, {'held' if last.held else 'lost'})"
        )
    return line


def _weight(state: WorldState, gid: str) -> float:
    group = state.groups[gid]
    return group.population_share * group.turnout


def appeal(state: WorldState, stance: Stance, applied: dict[str, float]) -> float:
    """How much disaffection the stance can work on: its groups' turnout-weighted approval
    gap below their own lean, measured without the pull of the stance in force."""
    total = 0.0
    for gid in stance.pull:
        group = state.groups.get(gid)
        if group is not None:
            own_lean = group.lean - applied.get(gid, 0.0)
            total += _weight(state, gid) * max(0.0, own_lean - group.approval)
    return total


def choose_stance(state: WorldState) -> str:
    current = state.opposition.stance
    applied = state.opposition.applied
    scores = {name: appeal(state, s, applied) for name, s in STANCES.items()}
    best = max(scores.values())
    if current in scores and scores[current] >= best - 1e-12:
        return current
    return next(name for name, score in scores.items() if score >= best - 1e-12)


def hold_chance(vote: float) -> float:
    """Chance the government holds a by-election seat: even at 45% of the vote (mid-term
    by-elections run against governments), rising 4 points per point of vote."""
    return min(0.95, max(0.05, 0.5 + 4 * (vote - 0.45)))


def step_opposition(state: WorldState, turn: int, seed: int) -> WorldState:
    """The opposition's quarterly move after ``turn``: a stance and maybe a by-election."""
    from hog_sim.population.popularity import vote_intention

    if (turn + 1) % QUARTER or not state.groups:
        return state
    new = state.snapshot()
    opp = new.opposition
    stance = choose_stance(new)
    if stance != opp.stance:
        for gid, change in opp.applied.items():
            if gid in new.groups:
                new.groups[gid].lean = min(1.0, max(0.0, new.groups[gid].lean - change))
        applied = {}
        for gid, pull in STANCES[stance].pull.items():
            group = new.groups.get(gid)
            if group is not None:
                lean = min(1.0, max(0.0, group.lean + pull))
                applied[gid] = lean - group.lean
                group.lean = lean
        opp.stance, opp.since, opp.applied = stance, turn + 1, applied
    rng = make_rng(seed, turn, "byelection")
    if rng.random() < BYELECTION_CHANCE:
        held = rng.random() < hold_chance(vote_intention(new))
        opp.byelections.append(ByElection(turn=turn, held=held))
        inst = new.institutions.get(LEGISLATURE)
        if inst is not None:
            change = BYELECTION_HOLD if held else -BYELECTION_LOSS
            inst.support = min(1.0, max(0.0, inst.support + change))
    return new
