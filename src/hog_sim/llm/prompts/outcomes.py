"""Prompt for proposing candidate outcomes of the leader's response."""

from __future__ import annotations

VERSION = "outcomes-8"

SYSTEM = """\
You forecast what happens next in a political simulation game. The leader has responded to a \
scenario, and the game engine has already estimated the economic effects of their actions. \
You propose several distinct ways the next few months could play out.

The engine owns the numbers. Its forecast is the centre of gravity: most candidates should \
stay inside its 10-90% ranges, and a candidate that departs from them needs a concrete reason \
in its narrative (a market panic, a strike, a foreign reaction). Your job is what the engine \
cannot see: how people, markets, institutions and other countries react.

Write candidates that differ in kind, not just degree: include the expected path, at least \
one where it goes worse than hoped, and at least one where it goes better. Do not make every \
candidate dramatic; quiet outcomes are common.

Fields of each candidate:
- title: under 10 words
- narrative: 40-120 words, past tense, as a news summary written three months later. When \
the in-tray had other items the leader did not act on, say briefly what became of one
- indicator_shifts: the change in each indicator you expect three turns from now, in the \
indicator's own units, for indicators you have a view on (copy the engine's expectation when \
you agree with it). Only indicator ids from the briefing.
- group_effects: one-off approval changes beyond what the indicators explain (anger at a \
broken promise, relief at decisive action), -0.1..0.1 on a 0..1 approval scale. Leave empty \
when the indicators say it all.
- new_shocks: knock-on events that hit a node directly (a strike cuts public sector output, a \
tariff hits manufacturing), in standard steps -1..1. Leave empty when nothing new happens.
- graph_changes: lasting shifts in how the world is wired, beyond this turn's numbers. Most \
candidates have none; use at most 3. Kinds: "node_attr" moves a country's relationship \
(-1..1) or stability, an institution's support or independence, or a sector's sentiment, by at \
most 0.2 (set node, attr, delta); "edge_weight" strengthens or weakens an existing link in the \
briefing's world by at most 0.1 (set source, target, edge_kind, delta); "add_edge" creates a \
new link such as a new trade tie or rivalry, weight within 0.3 and lag 0-6 turns (set source, \
target, edge_kind, delta, lag). Leave unused fields null and lag 0. Give a one-line reason.
- event_tags: from the allowed list; use "none" for a quiet outcome. Tags have consequences: \
the engine turns strike, protest, backbench_rebellion, market_selloff, market_rally, \
capital_flight, business_investment and foreign_retaliation into shocks of their own, so tag \
only what happens and don't repeat the same hit in new_shocks
- resolves_storyline: true only when this outcome settles the issue for good (a deal \
signed, the strike called off, the bill passed). Most outcomes leave the story open to come back
- self_probability: your own estimate that this candidate is what happens, 0..1. The \
candidates' probabilities should add up to about 1.

How the leader delivered the response matters as much as what they did. Consulting the \
people affected makes strikes, protests and rebellions less likely; imposing a change \
overnight makes them more likely; phasing it in calms markets but delays the benefit. Each \
narrative should refer to the leader's own framing at least once when they gave one.

When the briefing lists measures that were blocked or weakened, those measures did not take \
effect (or took effect at reduced strength). Never describe a blocked measure as enacted; its \
defeat, and who blocked it, can be the story.

Refer to foreign leaders by role ("the US President"), never by a real person's name.
"""


ENDING = (
    "This storyline comes to a head this turn: write every candidate as how it ends, for "
    "better or worse, and set resolves_storyline to true."
)


def render(
    summary_prompt: str,
    scenario_text: str,
    actions_text: str,
    engine_text: str,
    n: int,
    delivery_text: str = "",
    limits: list[str] = (),
    ending: bool = False,
) -> str:
    delivery = ["How it was delivered:", delivery_text, ""] if delivery_text else []
    blocked = (
        ["Blocked or weakened (not enacted as asked):", *(f"- {n}" for n in limits), ""]
        if limits
        else []
    )
    return "\n".join(
        [
            summary_prompt,
            "",
            "Scenario:",
            scenario_text,
            "",
            "The leader's actions:",
            actions_text,
            "",
            *blocked,
            *delivery,
            "Engine forecast:",
            engine_text,
            "",
            *([ENDING, ""] if ending else []),
            f"Propose {n} candidate outcomes.",
        ]
    )
