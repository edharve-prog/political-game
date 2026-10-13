"""The political calendar: events that come round on fixed turns (backlog story SD-6).

On a calendar turn the lead issue is the event, with options set by its kind: the Budget
asks for tax and spending choices. Claude still writes the briefing and the rest of the
in-tray; offline, the event's own briefing is used. Calendar leads don't open storylines.

Turns are months. In a 24-turn game the Budget falls on turns 3 and 15, the summit on 6 and
18, conference on 9 and 21, and a by-election on turn 12.
"""

from __future__ import annotations

from pydantic import Field

from hog_sim.core.models import Category, Model, PolicyAction, Scenario
from hog_sim.core.state import WorldState


class CalendarEvent(Model):
    kind: str
    title: str
    category: Category
    briefing: str = Field(description="Offline briefing; {group} is the by-election's group")
    options: list[str]
    packages: list[list[PolicyAction]] = Field(
        default_factory=list,
        description="The actions each option stands for, in order (E16). Targets may be "
        "{group} (the by-election's group), {country} (the player's country, for a speech) "
        "or {households} (shared over the population groups, see ``spread``).",
    )
    affected_nodes: list[str]
    every: int = Field(12, ge=1)
    offset: int = Field(ge=0)
    urgency: float = 0.7

    def falls_on(self, turn: int) -> bool:
        return turn > 0 and turn % self.every == self.offset % self.every


# Targets standing for the whole population and the whole economy (see ``spread``).
HOUSEHOLDS = "{households}"
ECONOMY = "{economy}"


def spread(action: PolicyAction, state: WorldState) -> list[PolicyAction]:
    """``action``, or one action per node when it is aimed at the whole population
    (``HOUSEHOLDS``: every group but business) or the whole economy (``ECONOMY``: every
    sector), with its size shared out evenly. A speech or doing nothing goes to the player's
    country instead: tax, spending and rules reach the country only through its sectors and
    groups."""
    if action.target not in (HOUSEHOLDS, ECONOMY):
        return [action]
    if action.kind in ("communicate", "do_nothing"):
        return [action.model_copy(update={"target": state.player_country})]
    if action.target == HOUSEHOLDS:
        nodes = [g for g in state.groups if g != "group:business"]
    else:
        nodes = list(state.sectors)
    if not nodes:
        return [action.model_copy(update={"target": state.player_country})]
    size = round(action.magnitude / len(nodes), 3)
    return [action.model_copy(update={"target": n, "magnitude": size}) for n in nodes]


def _act(kind: str, target: str, size: float) -> PolicyAction:
    return PolicyAction(kind=kind, target=target, magnitude=size)  # type: ignore[arg-type]


CALENDAR = [
    CalendarEvent(
        kind="budget",
        title="The Budget",
        category="economy",
        briefing=(
            "The Chancellor presents the Budget to the Commons this month. Markets, unions and "
            "business are waiting to see the government's tax and spending choices, and the "
            "deficit will be the headline."
        ),
        options=[
            "Raise taxes to fund public services",
            "Cut spending to bring borrowing down",
            "Cut taxes for business to boost growth",
            "A steady Budget with no big changes",
        ],
        packages=[
            [_act("tax", HOUSEHOLDS, 0.5), _act("spend", "sector:public", 0.5)],
            [_act("spend", "sector:public", -0.5)],
            [_act("tax", "group:business", -0.5)],
            [_act("do_nothing", "{country}", 0)],
        ],
        affected_nodes=["indicator:deficit", "sector:public", "group:business"],
        offset=3,
        urgency=0.8,
    ),
    CalendarEvent(
        kind="summit",
        title="International summit",
        category="foreign_affairs",
        briefing=(
            "World leaders gather for a summit this month. Allies want commitments on trade "
            "and security, and every move will be read at home and abroad."
        ),
        options=[
            "Seek a closer trade deal with the EU",
            "Take a hard line with China",
            "Pledge more aid and keep the summit calm",
        ],
        packages=[
            [_act("diplomatic", "country:eu", 0.5)],
            [_act("diplomatic", "country:china", -0.5)],
            [_act("diplomatic", "country:eu", 0.25)],
        ],
        affected_nodes=["country:eu", "country:china", "sector:manufacturing"],
        offset=6,
    ),
    CalendarEvent(
        kind="conference",
        title="Party conference",
        category="party_scandal",
        briefing=(
            "Party conference opens this month. Activists want red meat, backbenchers want "
            "their causes heard, and the leader's speech will set the tone for the year."
        ),
        options=[
            "Rally the party with a bold new promise",
            "Reach out to the centre ground",
            "Give backbenchers a policy concession",
        ],
        packages=[
            [_act("communicate", "institution:legislature", 0.5)],
            [_act("communicate", "{country}", 0.5)],
            [_act("spend", "institution:legislature", 0.25)],
        ],
        affected_nodes=["institution:legislature", "group:business", "group:public_workers"],
        offset=9,
    ),
    CalendarEvent(
        kind="by_election",
        title="By-election called",
        category="party_scandal",
        briefing=(
            "A by-election has been called in a seat full of {group}. It will be read as a "
            "verdict on the government."
        ),
        options=[
            "Campaign on the government's record",
            "Announce help aimed at {group}",
            "Keep the leader away and let the candidate run a local campaign",
        ],
        packages=[
            [_act("communicate", "{country}", 0.5)],
            [_act("spend", "{group}", 0.5)],
            [_act("do_nothing", "{country}", 0)],
        ],
        affected_nodes=["institution:legislature"],
        offset=0,
    ),
]


def calendar_event(turn: int) -> CalendarEvent | None:
    return next((e for e in CALENDAR if e.falls_on(turn)), None)


def upcoming(turn: int, within: int = 6) -> list[tuple[int, CalendarEvent]]:
    """Calendar events due after ``turn`` and within ``within`` turns, soonest first."""
    return [
        (t, e) for t in range(turn + 1, turn + within + 1) if (e := calendar_event(t)) is not None
    ]


def _by_election_group(state: WorldState) -> tuple[str | None, str]:
    groups = sorted(state.groups.values(), key=lambda g: (g.approval, g.id))
    return (groups[0].id, groups[0].name.lower()) if groups else (None, "voters")


def fill(event: CalendarEvent, state: WorldState) -> CalendarEvent:
    """``event`` with its by-election group filled in and nodes the world lacks left out."""
    group_id, group = _by_election_group(state)
    nodes = [
        *event.affected_nodes,
        *([group_id] if event.kind == "by_election" and group_id else []),
    ]
    known = set(state.node_ids())
    places = {"{group}": group_id, "{country}": state.player_country}
    packages = [
        [
            b
            for a in package
            for b in spread(a.model_copy(update={"target": places.get(a.target, a.target)}), state)
        ]
        for package in event.packages
    ]
    return event.model_copy(
        update={
            "briefing": event.briefing.format(group=group),
            "options": [o.format(group=group) for o in event.options],
            "packages": [p if all(a.target in known for a in p) else [] for p in packages],
            "affected_nodes": [n for n in nodes if n in known] or [state.player_country],
        }
    )


def scheduled_scenario(event: CalendarEvent, state: WorldState) -> Scenario:
    """The offline scenario for a calendar turn."""
    event = fill(event, state)
    return Scenario(
        title=event.title,
        briefing=event.briefing,
        affected_nodes=event.affected_nodes,
        urgency=event.urgency,
        source="scheduled",
        suggested_options=event.options,
        category=event.category,
    )


def calendar_lines(turn: int) -> list[str]:
    return [f"Turn {t}: {e.title} (in {t - turn} turns)" for t, e in upcoming(turn)]
