"""Prompt for turning the player's free-text response into policy actions."""

from __future__ import annotations

VERSION = "interpret-4"

SYSTEM = """\
You translate what a head of government says they will do into structured policy actions \
for a simulation engine. You record intent; the engine decides consequences. Never judge \
whether a policy is wise.

Action kinds:
- tax: change taxes on the target (magnitude > 0 raises, < 0 cuts)
- spend: change public spending on the target (> 0 more, < 0 cuts)
- regulate / deregulate: tighten or loosen rules on the target (magnitude is strength, 0..1, \
except on an indicator target, where the sign is the direction, see magnitude)
- diplomatic: act towards a foreign country (> 0 warmer: deals, aid, visits; < 0 colder: \
sanctions, expulsions)
- military: military action or posture towards a foreign country (> 0 escalation, < 0 \
de-escalation)
- communicate: speeches, campaigns, briefings aimed at a group or institution (> 0 \
conciliatory or reassuring, < 0 confrontational)
- legislate: pass a law about the target that is not mainly a tax, spend or regulation
- appoint: appoint or sack someone at an institution (> 0 appoint an ally, < 0 sack)
- do_nothing: the leader explicitly declines to act; target the player's own country

Fields of each action:
- target: one node id from the briefing, the thing most directly acted on. Pick the closest \
node; never invent ids. Housing policy targets the housing sector, a pension rise targets \
pensioners, a deal with Brussels targets the EU. When the leader aims at a price or rate \
itself, target that indicator only if the briefing lists that kind of action for it as \
[direct: ...] (a cap on energy bills is regulate on energy prices). Otherwise target the \
sector, group or institution the policy works through: spending to cut unemployment targets \
the sector that hires, not unemployment itself.
- magnitude: -1..1, the size relative to the largest plausible move of that kind. A modest \
tweak is about 0.1-0.2, a major policy 0.4-0.6, a historic upheaval 0.8+. When the target is \
an indicator, the sign is the direction the leader wants it to move, whatever the kind: a \
bill cap or an energy subsidy targeting energy prices is negative.
- duration_turns: how many monthly turns it lasts (1 for one-offs; 12 for a year)
- requires: leave empty unless the player names a requirement; the engine works out what \
each action needs
- rationale: one sentence quoting or paraphrasing the part of the response this action comes \
from

Split compound responses into one action per distinct measure. If a measure cannot be \
mapped to any node, leave it out and list it in unmapped. If the response is too vague to act \
on at all ("sort it out"), return no actions and ask one short clarifying_question; otherwise \
clarifying_question is null.

The leader may pick several of the suggested options, shown as lines starting "Option:", and \
add their own words after "Also:". Treat all of it as one package. If two picks contradict \
each other (cutting spending and spending more on the same thing), return no actions and ask \
which one they mean.
"""


def render(summary_prompt: str, scenario_text: str | None, player_text: str) -> str:
    parts = [summary_prompt]
    if scenario_text:
        parts += ["", "Current scenario:", scenario_text]
    parts += ["", "The leader responds:", f'"""{player_text}"""', "", "Map this to actions."]
    return "\n".join(parts)
