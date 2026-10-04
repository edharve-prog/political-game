"""The political calendar: events that come round on fixed turns (backlog story SD-6).

On a calendar turn the lead issue is the event, with options set by its kind: the Budget
asks for tax and spending choices. Claude still writes the briefing and the rest of the
in-tray; offline, the event's own briefing is used. Calendar leads don't open storylines.

Turns are months. In a 24-turn game the Budget falls on turns 3 and 15, the summit on 6 and
18, conference on 9 and 21, and a by-election on turn 12.
"""

from __future__ import annotations

from pydantic import Field

from hog_sim.core.models import Category, Model, Scenario
from hog_sim.core.state import WorldState


class CalendarEvent(Model):
    kind: str
    title: str
    category: Category
    briefing: str = Field(description="Offline briefing; {group} is the by-election's group")
    options: list[str]
    affected_nodes: list[str]
    every: int = Field(12, ge=1)
    offset: int = Field(ge=0)
    urgency: float = 0.7

    def falls_on(self, turn: int) -> bool:
        return turn > 0 and turn % self.every == self.offset % self.every


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
    return event.model_copy(
        update={
            "briefing": event.briefing.format(group=group),
            "options": [o.format(group=group) for o in event.options],
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
