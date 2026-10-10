"""How standing and mood feed the economy (Project 16, EB-8).

Three fields that used to change only the briefing text now have effects:

* A sector's ``sentiment`` (-1..1, business confidence) pushes its output every turn by
  ``SENTIMENT_STEPS`` per unit, and settles back toward neutral by ``SENTIMENT_SETTLE`` of
  the gap each turn, so a confidence knock costs output for a while and then passes.
* A foreign country's ``stability`` (0..1) sets the odds of a crisis there. Below
  ``CRISIS_STABILITY`` each turn has a chance, rising to ``CRISIS_RATE`` at zero stability,
  of a ``CRISIS_STEPS`` hit to its growth, which reaches the player through trade. The draw is
  seeded by the game seed and turn, so replay reproduces it.
* A country's ``relationship`` with the player scales trade: see
  ``world/changes.py`` (``TRADE_PER_RELATIONSHIP``).

At the starting world (sentiment 0, stability 0.5) none of this moves anything.
"""

from __future__ import annotations

from hog_sim.core.config import make_rng
from hog_sim.core.models import Shock
from hog_sim.core.state import WorldState

SENTIMENT_STEPS = 0.2  # output push per unit of sentiment, standard steps a turn
SENTIMENT_SETTLE = 0.2  # share of the gap to neutral sentiment closed each turn

CRISIS_STABILITY = 0.5  # below this a country can fall into crisis
CRISIS_RATE = 0.3  # chance of a crisis a turn at zero stability
CRISIS_STEPS = -1.0  # hit to the country's growth, standard steps (1pp)


def sentiment_shocks(state: WorldState) -> list[Shock]:
    """This turn's output push from each sector's sentiment."""
    return [
        Shock(node=sid, delta=SENTIMENT_STEPS * sector.sentiment)
        for sid, sector in state.sectors.items()
        if sector.sentiment
    ]


def settle_sentiment(state: WorldState) -> WorldState:
    """Move every sector's sentiment ``SENTIMENT_SETTLE`` of the way back to neutral."""
    new = state.snapshot()
    for sector in new.sectors.values():
        sector.sentiment *= 1 - SENTIMENT_SETTLE
    return new


def crisis_chance(stability: float) -> float:
    if stability >= CRISIS_STABILITY:
        return 0.0
    return CRISIS_RATE * (CRISIS_STABILITY - stability) / CRISIS_STABILITY


def crisis_shocks(state: WorldState, seed: int) -> list[Shock]:
    """Crises this turn in unstable foreign countries, drawn from the game seed."""
    shocks = []
    for cid in sorted(state.countries):
        if cid == state.player_country:
            continue
        chance = crisis_chance(state.countries[cid].stability)
        if chance and make_rng(seed, state.turn, f"crisis:{cid}").random() < chance:
            shocks.append(Shock(node=cid, delta=CRISIS_STEPS))
    return shocks
