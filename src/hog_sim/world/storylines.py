"""Storylines: issues that run across turns (backlog story SD-1).

A scenario names the storyline it continues, or opens a new one. ``advance_storylines``
runs inside ``resolve()``, so it is deterministic and replay reproduces it:

* the scenario's storyline moves up a stage (or opens at stage 1) and records the turn;
* the chosen outcome closes it only when it says so (``Outcome.resolves_storyline``);
* every other open storyline left alone for ``ESCALATE_AFTER`` turns escalates a stage and
  gains pressure, so ignoring an issue never makes it quietly go away.
"""

from __future__ import annotations

import re

from hog_sim.core.models import Outcome, Scenario, Storyline
from hog_sim.core.state import WorldState

ESCALATE_AFTER = 3
ESCALATION_PRESSURE = 0.15
HISTORY_LINES = 6
MAX_LINE = 160


def new_storyline_id(title: str, turn: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40].strip("-")
    return f"{slug or 'story'}-t{turn}"


def open_storylines(state: WorldState) -> list[Storyline]:
    return [s for s in state.storylines.values() if s.open]


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_LINE else text[: MAX_LINE - 3].rstrip() + "..."


def _add(story: Storyline, line: str) -> None:
    story.history = [*story.history, line][-HISTORY_LINES:]


def advance_storylines(
    state: WorldState, turn: int, scenario: Scenario, outcome: Outcome
) -> WorldState:
    """Return ``state`` with this turn's storyline moves applied. ``turn`` is the turn played."""
    if not scenario.storyline and not open_storylines(state):
        return state
    stories = {k: v.model_copy(deep=True) for k, v in state.storylines.items()}

    if scenario.storyline:
        story = stories.get(scenario.storyline)
        if story is None:
            story = Storyline(
                id=scenario.storyline,
                title=scenario.title,
                category=scenario.category,
                nodes=list(scenario.affected_nodes),
                opened_turn=turn,
                last_turn=turn,
                last_addressed=turn,
            )
        else:
            story.stage += 1
            story.open = True
            story.last_turn = story.last_addressed = turn
            story.nodes = list(dict.fromkeys([*story.nodes, *scenario.affected_nodes]))
        story.pressure = scenario.urgency
        _add(
            story, _clip(f"Turn {turn}, stage {story.stage}: {scenario.title}. {outcome.narrative}")
        )
        if outcome.resolves_storyline:
            story.open = False
            _add(story, f"Turn {turn}: resolved")
        stories[story.id] = story

    for story in stories.values():
        if (
            story.open
            and story.id != scenario.storyline
            and turn - story.last_turn >= ESCALATE_AFTER
        ):
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
        lines.append(
            f"- {s.id} [{s.category or 'uncategorised'}] {s.title}: stage {s.stage}, "
            f"pressure {s.pressure:.2f}, {idle} turn(s) since faced"
        )
        lines += [f"    {h}" for h in s.history]
    return "\n".join(lines)
