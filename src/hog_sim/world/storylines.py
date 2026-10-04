"""Storylines: issues that run across turns (backlog story SD-1).

A scenario names the storyline it continues, or opens a new one. ``advance_storylines``
runs inside ``resolve()``, so it is deterministic and replay reproduces it:

* the scenario's storyline moves up a stage (or opens at stage 1) and records the turn;
* the chosen outcome closes it only when it says so (``Outcome.resolves_storyline``);
* every other open storyline left alone for ``ESCALATE_AFTER`` turns escalates a stage and
  gains pressure, so ignoring an issue never makes it quietly go away.

The other items in the turn's in-tray (``Scenario.secondary``, story SD-2) open or continue
storylines too. One counts as acted on when an action targets one of its nodes; one left alone
carries over and keeps escalating, unless it was minor (urgency below ``FADE_BELOW``), in
which case it fades. The chosen outcome only ever settles the lead.

Storylines also come to a head (story SD-9). One that would reach ``FINAL_STAGE`` this turn,
or is ``MAX_AGE`` turns old, is in its final stage: the scenario writer is told, it can only
return as the lead, and when it does that turn's outcome is how it ends, whatever the outcome
says. Left alone instead, it comes to a head without the leader on its next escalation.
``must_open_new`` asks for a fresh lead when the last ``NEW_LEAD_EVERY - 1`` leads all
continued old storylines, so a few sagas can't fill the whole premiership.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from hog_sim.core.models import Outcome, PolicyAction, Scenario, SideIssue, Storyline
from hog_sim.core.state import WorldState

ESCALATE_AFTER = 3
ESCALATION_PRESSURE = 0.15
FADE_BELOW = 0.3  # a minor in-tray item left alone below this urgency resolves itself
HISTORY_LINES = 6
MAX_LINE = 160
FINAL_STAGE = 6  # a storyline reaching this stage comes to a head
MAX_AGE = 10  # ... as does one open this many turns
NEW_LEAD_EVERY = 3  # at least one lead in this many opens a new storyline


def new_storyline_id(title: str, turn: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40].strip("-")
    return f"{slug or 'story'}-t{turn}"


def open_storylines(state: WorldState) -> list[Storyline]:
    return [s for s in state.storylines.values() if s.open]


def is_final(story: Storyline, turn: int) -> bool:
    """True when ``story`` comes to a head if it is the lead on ``turn``."""
    return story.stage + 1 >= FINAL_STAGE or turn - story.opened_turn >= MAX_AGE


def final_storylines(state: WorldState) -> list[str]:
    return [s.id for s in open_storylines(state) if is_final(s, state.turn)]


def must_open_new(state: WorldState, history: Sequence[Any]) -> bool:
    """True when each of the last ``NEW_LEAD_EVERY - 1`` turns' leads continued a storyline
    opened on an earlier turn. Calendar turns (SD-6) are skipped: their leads open none."""
    ordinary = [r for r in history if r.scenario.source != "scheduled"]
    recent = ordinary[-(NEW_LEAD_EVERY - 1) :]
    if len(recent) < NEW_LEAD_EVERY - 1:
        return False

    def continued(record: Any) -> bool:
        story = state.storylines.get(record.scenario.storyline or "")
        return story is not None and story.opened_turn < record.turn

    return all(continued(r) for r in recent)


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_LINE else text[: MAX_LINE - 3].rstrip() + "..."


def _add(story: Storyline, line: str) -> None:
    story.history = [*story.history, line][-HISTORY_LINES:]


def addressed(item: SideIssue, actions: list[PolicyAction]) -> bool:
    """An in-tray item counts as acted on when an action targets one of its nodes."""
    nodes = set(item.affected_nodes)
    return any(a.kind != "do_nothing" and a.target in nodes for a in actions)


def advance_storylines(
    state: WorldState,
    turn: int,
    scenario: Scenario,
    outcome: Outcome,
    actions: list[PolicyAction] = (),
) -> WorldState:
    """Return ``state`` with this turn's storyline moves applied. ``turn`` is the turn played."""
    if not scenario.storyline and not scenario.secondary and not open_storylines(state):
        return state
    stories = {k: v.model_copy(deep=True) for k, v in state.storylines.items()}

    def touch(item: Scenario | SideIssue, acted: bool) -> Storyline:
        story = stories.get(item.storyline)
        if story is None:
            story = Storyline(
                id=item.storyline,
                title=item.title,
                category=item.category,
                nodes=list(item.affected_nodes),
                opened_turn=turn,
                last_turn=turn,
                last_addressed=turn,
                pressure=item.urgency,
            )
            stories[story.id] = story
        elif acted:
            story.stage += 1
            story.open = True
            story.last_turn = story.last_addressed = turn
            story.nodes = list(dict.fromkeys([*story.nodes, *item.affected_nodes]))
            story.pressure = item.urgency
        else:
            # Seen but left alone: it keeps its idle clock, so it still escalates.
            story.last_addressed = turn
            story.pressure = max(story.pressure, item.urgency)
        return story

    if scenario.storyline:
        before = stories.get(scenario.storyline)
        final = before is not None and before.open and is_final(before, turn)
        story = touch(scenario, acted=True)
        _add(
            story, _clip(f"Turn {turn}, stage {story.stage}: {scenario.title}. {outcome.narrative}")
        )
        if outcome.resolves_storyline:
            story.open = False
            _add(story, f"Turn {turn}: resolved")
        elif final:
            story.open = False
            _add(story, f"Turn {turn}: came to a head")

    for item in scenario.secondary:
        if not item.storyline:
            continue
        acted = addressed(item, actions)
        story = touch(item, acted)
        if acted:
            _add(story, f"Turn {turn}, stage {story.stage}: {item.title} (acted on)")
        elif item.urgency < FADE_BELOW:
            story.open = False
            _add(story, f"Turn {turn}: {item.title}, left alone and faded away")
        else:
            _add(story, f"Turn {turn}: {item.title}, left in the in-tray")

    for story in stories.values():
        if story.open and turn - story.last_turn >= ESCALATE_AFTER and is_final(story, turn):
            # Its climax happens without the leader: it ends rather than escalating forever.
            story.open = False
            story.last_turn = turn
            story.pressure = min(1.0, story.pressure + ESCALATION_PRESSURE)
            _add(story, f"Turn {turn}: came to a head while left alone")
        elif story.open and turn - story.last_turn >= ESCALATE_AFTER:
            story.stage += 1
            story.last_turn = turn
            story.pressure = min(1.0, story.pressure + ESCALATION_PRESSURE)
            _add(story, f"Turn {turn}, stage {story.stage}: escalated while left alone")

    new = state.model_copy()
    new.storylines = stories
    return new


def storylines_text(state: WorldState) -> str:
    """The open storylines with their stage history, for the scenario prompt."""
    stories = sorted(open_storylines(state), key=lambda s: -s.pressure)
    if not stories:
        return ""
    lines = ["Open storylines (id, stage, pressure 0-1, turns since the leader last faced it):"]
    for s in stories:
        idle = state.turn - s.last_addressed
        final = (
            " FINAL STAGE: it can only return as the lead, and this turn would be how it ends"
            if is_final(s, state.turn)
            else ""
        )
        lines.append(
            f"- {s.id} [{s.category or 'uncategorised'}] {s.title}: stage {s.stage}, "
            f"pressure {s.pressure:.2f}, {idle} turn(s) since faced.{final}"
        )
        lines += [f"    {h}" for h in s.history]
    return "\n".join(lines)
