"""Storylines: issues that run across turns (backlog story SD-1).

A scenario names the storyline it continues, or opens a new one. ``advance_storylines``
runs inside ``resolve()``, so it is deterministic and replay reproduces it:

* the scenario's storyline moves up a stage (or opens at stage 1) and records the turn;
* the chosen outcome closes it only when it says so (``Outcome.resolves_storyline``);
* every other open storyline left alone for ``ESCALATE_AFTER`` turns escalates a stage and
  gains pressure, so ignoring an issue never makes it quietly go away.

The other items in the turn's in-tray (``Scenario.secondary``, story SD-2) open or continue
storylines too. What happens to one depends on what this turn's decisions do to its nodes
(``effect_on``: the engine's effect, signed by the way voters want each node to go), not on
whether an action merely names one (E11):

* helped: it eases (loses ``RELIEF`` pressure) without moving up a stage, its idle clock
  restarts, and once its pressure is below ``FADE_BELOW`` it is dealt with and closes;
* made worse (a cut to the sector it is about): it escalates a stage and gains pressure;
* left alone: it carries over and keeps escalating on its idle clock, unless it was minor
  (urgency below ``FADE_BELOW``), in which case it fades.

So acting on an item never keeps it open longer than ignoring it. The chosen outcome only
ever settles the lead.

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

from hog_sim.core.models import EdgeKind, Outcome, PolicyAction, Scenario, SideIssue, Storyline
from hog_sim.core.state import WorldState
from hog_sim.world.effects import decision_effects

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


HELP_MIN = 0.05  # weighted standard steps a decision must move an item's nodes to count
RELIEF = 0.25  # pressure an in-tray item loses on a turn the decisions help it


def good_way(state: WorldState, node_id: str) -> float:
    """The way voters want ``node_id`` to move: the sign of how the groups feel it (each
    group's ``CARES_ABOUT`` or ``EMPLOYS`` weight times its share of the population). Output,
    growth, support and approval are wanted up when no group feels them; an indicator no group
    feels has no wanted way (0)."""
    felt = 0.0
    for edge in state.edges:
        if edge.kind == EdgeKind.CARES_ABOUT and edge.target == node_id:
            group = state.groups.get(edge.source)
        elif edge.kind == EdgeKind.EMPLOYS and edge.source == node_id:
            group = state.groups.get(edge.target)
        else:
            continue
        if group is not None:
            felt += edge.weight * group.population_share
    if felt:
        return 1.0 if felt > 0 else -1.0
    return 0.0 if node_id in state.indicators else 1.0


def effect_on(item: SideIssue, actions: Sequence[PolicyAction], state: WorldState) -> float:
    """How much this turn's decisions help ``item``: what they alone do to each of its nodes
    (``decision_effects``, standard steps), signed by the way voters want that node to go
    (``good_way``) and summed. Positive helps, negative makes it worse. ``state`` is the
    world the turn was played in."""
    moves = decision_effects(state, actions, item.affected_nodes)
    return sum(good_way(state, node_id) * move for node_id, move in moves.items())


def addressed(item: SideIssue, actions: Sequence[PolicyAction], state: WorldState) -> bool:
    """An in-tray item counts as acted on when this turn's decisions help it (``effect_on``),
    not merely when an action names one of its nodes: a cut to the sector it is about is no
    help."""
    return effect_on(item, actions, state) > HELP_MIN


def harmed(item: SideIssue, actions: Sequence[PolicyAction], state: WorldState) -> bool:
    """This turn's decisions make ``item`` worse."""
    return effect_on(item, actions, state) < -HELP_MIN


def advance_storylines(
    state: WorldState,
    turn: int,
    scenario: Scenario,
    outcome: Outcome,
    actions: list[PolicyAction] = (),
    played: WorldState | None = None,
) -> WorldState:
    """Return ``state`` with this turn's storyline moves applied. ``turn`` is the turn played,
    and ``played`` the world it was played in (default ``state``), which judges what the
    actions do to each in-tray item."""
    played = state if played is None else played
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
        effect = effect_on(item, actions, played)
        known = stories.get(item.storyline)
        eased_from = known.pressure if known is not None else item.urgency
        story = touch(item, acted=False)
        story.nodes = list(dict.fromkeys([*story.nodes, *item.affected_nodes]))
        if effect > HELP_MIN:
            # Eased, not escalated: it loses pressure from where it stood (the writer's urgency
            # for it does not undo the easing) and its idle clock starts again.
            story.pressure = max(0.0, eased_from - RELIEF)
            story.last_turn = turn
            if story.pressure < FADE_BELOW:
                story.open = False
                _add(story, f"Turn {turn}: {item.title}, acted on and dealt with")
            else:
                _add(story, f"Turn {turn}, stage {story.stage}: {item.title} (acted on, eased)")
        elif effect < -HELP_MIN:
            story.stage += 1
            story.last_turn = turn
            story.pressure = min(1.0, story.pressure + ESCALATION_PRESSURE)
            _add(story, f"Turn {turn}, stage {story.stage}: {item.title}, made worse")
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
