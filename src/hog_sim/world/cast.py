"""The cast: fictional people the leader deals with (backlog story SD-4).

Each character cares about a few nodes and wants them helped (+1) or squeezed (-1). In
``resolve()``, after the turn's actions:

- an action on a node they care about backs them (loyalty up) or crosses them (down);
- consulting one of their nodes counts in their favour;
- a character the scenario involved, whom the leader neither acted for nor consulted, feels
  ignored;
- a sacked minister leaves, costs the government some Commons support and is replaced;
- a minister whose loyalty falls below ``RESIGN_BELOW`` resigns, with the same cost.

Loyalty is shown in every prompt, so scenarios and outcomes can bring in resignations, leaks
or support. No character is a real person, and foreign leaders stay role-titled.
"""

from __future__ import annotations

from collections.abc import Sequence

from hog_sim.core.models import Character, Delivery, PolicyAction
from hog_sim.core.state import WorldState

BACKED = 0.08
CROSSED = -0.08
CONSULTED = 0.04
IGNORED = -0.03
RESIGN_BELOW = 0.25
DEPARTURE_COST = 0.05  # Commons support lost when a minister is sacked or resigns
MEMORY = 5
LEGISLATURE = "institution:legislature"

# Action kinds whose positive magnitude hurts the target rather than helping it.
_HARMFUL = {"tax", "regulate", "military"}


def push(action: PolicyAction) -> int:
    """+1 when the action helps its target, -1 when it squeezes it, 0 when neither.

    On an indicator the sign is the direction the leader wants it to move, whatever the kind.
    """
    if action.kind == "do_nothing" or action.magnitude == 0:
        return 0
    sign = 1 if action.magnitude > 0 else -1
    if action.kind in _HARMFUL and not action.target.startswith("indicator:"):
        sign = -sign
    return sign


def active_cast(state: WorldState) -> list[Character]:
    return [c for c in state.characters.values() if c.active]


def sackable(state: WorldState) -> list[str]:
    """Ids of the ministers in office, whom the leader may sack."""
    return [c.id for c in active_cast(state) if c.minister]


def _remember(character: Character, line: str) -> None:
    character.memory = [*character.memory, line][-MEMORY:]


def _leave(state: WorldState, character: Character, why: str, turn: int) -> None:
    character.active = False
    _remember(character, f"{why} on turn {turn}")
    if LEGISLATURE in state.institutions:
        inst = state.institutions[LEGISLATURE]
        inst.support = max(0.0, inst.support - DEPARTURE_COST)
    successor = _successor(state, character, turn)
    state.characters[successor.id] = successor


def _successor(state: WorldState, gone: Character, turn: int) -> Character:
    used = {c.name for c in state.characters.values()}
    name = next((n for n in SUCCESSORS if n not in used), f"New {gone.role}")
    slug = gone.id.split(":", 1)[1].split("-")[0]
    return gone.model_copy(
        update={
            "id": f"person:{slug}-{turn}",
            "name": name,
            "loyalty": 0.6,
            "active": True,
            "memory": [f"took over as {gone.role} on turn {turn}"],
        }
    )


def apply_cast(
    state: WorldState,
    turn: int,
    actions: Sequence[PolicyAction],
    delivery: Delivery | None = None,
    involved: Sequence[str] = (),
    sacked: Sequence[str] = (),
) -> WorldState:
    """Move the cast's loyalty for one turn and handle departures. Deterministic."""
    new = state.snapshot()
    consulted = set(delivery.consulted) if delivery else set()
    for character in active_cast(new):
        if character.id in sacked and character.minister:
            _leave(new, character, "sacked", turn)
            continue
        score = sum(
            push(a) * character.cares[a.target] for a in actions if a.target in character.cares
        )
        change = 0.0
        if score > 0:
            change += BACKED
            _remember(character, f"backed on turn {turn}")
        elif score < 0:
            change += CROSSED
            _remember(character, f"crossed on turn {turn}")
        if consulted & character.cares.keys():
            change += CONSULTED
            _remember(character, f"consulted on turn {turn}")
        elif score == 0 and character.id in involved:
            change += IGNORED
            _remember(character, f"ignored on turn {turn}")
        character.loyalty = min(1.0, max(0.0, character.loyalty + change))
        if character.minister and character.loyalty < RESIGN_BELOW:
            _leave(new, character, "resigned", turn)
    return new


def cast_line(character: Character) -> str:
    cares = ", ".join(f"{n} {'+' if w > 0 else '-'}" for n, w in character.cares.items())
    low = "  [LOW LOYALTY: may leak, rebel or resign]" if character.loyalty < 0.35 else ""
    recent = f" Recently: {'; '.join(character.memory[-3:])}." if character.memory else ""
    minister = " (minister)" if character.minister else ""
    return (
        f"{character.id} {character.name}, {character.role}{minister}: {character.agenda} "
        f"(cares about {cares}). Loyalty {character.loyalty:.2f}.{recent}{low}"
    )


def default_cast() -> dict[str, Character]:
    """The starting cast for the UK world. All names are invented."""
    cast = [
        Character(
            id="person:chancellor",
            name="Ruth Calder",
            role="Chancellor of the Exchequer",
            minister=True,
            agenda="keep borrowing down and the markets calm",
            cares={"indicator:deficit": -1, "sector:public": -1},
            loyalty=0.65,
        ),
        Character(
            id="person:housing_secretary",
            name="Owen Pryce",
            role="Housing Secretary",
            minister=True,
            agenda="get homes built and rents under control",
            cares={"sector:housing": 1, "group:young_renters": 1, "indicator:house_prices": -1},
        ),
        Character(
            id="person:union_leader",
            name="Dele Afolabi",
            role="General Secretary of the Public Services Union",
            agenda="win a fair pay deal for public sector workers",
            cares={"sector:public": 1, "group:public_workers": 1},
            loyalty=0.45,
        ),
        Character(
            id="person:governor",
            name="Martin Vane",
            role="Governor of the Bank of England",
            agenda="bring inflation back to target without political interference",
            cares={"institution:central_bank": 1, "indicator:inflation": -1},
            loyalty=0.5,
        ),
        Character(
            id="person:editor",
            name="Gail Torrance",
            role="Editor of the Daily Beacon tabloid",
            agenda="champion pensioners and households hit by bills",
            cares={"group:pensioners": 1, "indicator:energy_prices": -1},
            loyalty=0.5,
        ),
        Character(
            id="person:opposition_leader",
            name="Harriet Molyneux",
            role="Leader of the Opposition",
            agenda="paint the government as out of touch with young voters",
            cares={"group:young_renters": 1, "sector:housing": 1},
            loyalty=0.2,
        ),
        Character(
            id="person:growth_group",
            name="Gareth Lowther",
            role="Chair of the backbench Northern Growth Group",
            agenda="protect factory jobs and stand up to Chinese competition",
            cares={"sector:manufacturing": 1, "country:china": -1},
        ),
        Character(
            id="person:business_lobby",
            name="Priya Nandakumar",
            role="Director-General of the Federation of British Enterprise",
            agenda="lower taxes and lighter rules for business",
            cares={"group:business": 1, "sector:finance": 1},
            loyalty=0.5,
        ),
    ]
    return {c.id: c for c in cast}


# Invented names for ministers who replace one who left, used in order.
SUCCESSORS = ["Alan Mercer", "Fiona Strachan", "Tom Hadley", "Nadia Rahman", "Colin Ashby"]
