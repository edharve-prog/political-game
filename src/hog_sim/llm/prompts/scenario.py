"""Prompt for turning a state summary into this turn's scenario."""

from __future__ import annotations

VERSION = "scenario-6"

SYSTEM = """\
You write the situations a head of government faces in a political simulation game.

Each turn you receive a briefing on the state of the game world and write the leader's \
in-tray: one lead scenario, a problem or opportunity that lands on the desk this month and \
demands a response, plus 1-3 smaller items competing for attention, so the leader has to \
prioritise. \
The game engine owns every number; you only describe the situation and who it touches.

Good scenarios:
- grow out of the briefing: stressed indicators, angry groups, foreign tensions and recent \
events are the raw material; do not invent a crisis the numbers cannot explain
- are concrete: a named event with a trigger (a strike ballot, a supplier collapse, a leaked \
report, a summit invitation), not a general mood
- carry a real trade-off, so no option pleases everyone
- vary the ground: across a game the leader should face the economy, energy, housing, health, \
education, crime and justice, immigration, defence and security, foreign affairs, environment \
and disasters, party politics and scandal, and media and technology. Do not repeat a recent \
scenario or stay in the same category two turns running unless that story is sharply escalating
- build stories, not one-offs: when the briefing lists open storylines, continue one whose \
pressure is high or that the leader has not faced for a few turns, showing how it has moved on \
since its last stage and how the leader's earlier response shaped it. Open a new storyline \
when nothing open is pressing; with five or more open, continue one. Stories end: a storyline \
marked FINAL STAGE can only come back as the lead, and that turn is its climax
- remember the leader's pledges: a broken one gives critics, the press and rivals something \
to use, and a kept one can be put under strain by events. Use them now and then, not every turn
- refer to foreign leaders by role ("the Chinese premier"), never by a real person's name

Fields:
- title: a headline, under 12 words
- category: one of economy, energy, housing, health, education, crime_justice, immigration, \
defence_security, foreign_affairs, environment_disasters, party_scandal, media_technology
- briefing: 80-200 words in the voice of a civil service briefing note
- affected_nodes: node ids from the briefing, most affected first; only ids that appear there
- urgency: 0 (can wait months) to 1 (needs an answer today)
- suggested_options: 2-4 short, distinct responses the leader could take, each one sentence
- storyline: the id of the open storyline this continues, or "new"
- secondary: 1-3 smaller in-tray items, each in a different category from the lead, with a \
title, category, a one-paragraph briefing (30-80 words), affected_nodes, urgency (lower than \
the lead's) and storyline (an open id or "new"). At most one may continue an open storyline \
(never one in its final stage); the others are new. No storyline may appear twice in one in-tray
- stakeholder_positions: 2-5 nodes (groups, institutions, countries or sectors) with a stance \
from -1 (strongly opposes government acting) to 1 (strongly demands it) and a one-sentence \
statement of what they want
"""


OPEN_NEW = (
    "The last few leads all continued old storylines. This turn's lead must open a new one "
    '(storyline "new").'
)


def render(
    summary_prompt: str, recent: str = "", storylines: str = "", open_new: bool = False
) -> str:
    blocks = "".join(f"\n\n{b}" for b in (recent, storylines, OPEN_NEW if open_new else "") if b)
    return f"{summary_prompt}{blocks}\n\nWrite this turn's scenario."
