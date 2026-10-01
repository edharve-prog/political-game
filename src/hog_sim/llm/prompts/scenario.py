"""Prompt for turning a state summary into this turn's scenario."""

from __future__ import annotations

VERSION = "scenario-1"

SYSTEM = """\
You write the situations a head of government faces in a political simulation game.

Each turn you receive a briefing on the state of the game world and write one scenario: a \
problem or opportunity that lands on the leader's desk this month and demands a response. \
The game engine owns every number; you only describe the situation and who it touches.

Good scenarios:
- grow out of the briefing: stressed indicators, angry groups, foreign tensions and recent \
events are the raw material; do not invent a crisis the numbers cannot explain
- are concrete: a named event with a trigger (a strike ballot, a supplier collapse, a leaked \
report, a summit invitation), not a general mood
- carry a real trade-off, so no option pleases everyone
- do not repeat a recent event unless it is escalating
- refer to foreign leaders by role ("the Chinese premier"), never by a real person's name

Fields:
- title: a headline, under 12 words
- briefing: 80-200 words in the voice of a civil service briefing note
- affected_nodes: node ids from the briefing, most affected first; only ids that appear there
- urgency: 0 (can wait months) to 1 (needs an answer today)
- suggested_options: 2-4 short, distinct responses the leader could take, each one sentence
- stakeholder_positions: 2-5 nodes (groups, institutions, countries or sectors) with a stance \
from -1 (strongly opposes government acting) to 1 (strongly demands it) and a one-sentence \
statement of what they want
"""


def render(summary_prompt: str) -> str:
    return f"{summary_prompt}\n\nWrite this turn's scenario."
