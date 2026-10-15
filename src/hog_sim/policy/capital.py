"""Political capital as a stock (O5).

Capital is no longer a fixed allowance a turn. At the end of each turn ``resolve()`` refills
it, and part of what the turn left unspent is banked:

    next = refill + BANK_SHARE * unspent,   at most MAX_STOCK times the base

The refill is ``GameConfig.capital_per_turn`` (the base) scaled by four things, each 1 at
the starting world:

* **approval**: ``APPROVAL_GAIN`` per point of vote intention above or below
  ``NEUTRAL_VOTE``;
* **majority**: ``MAJORITY_GAIN`` per point of the legislature's support above or below
  ``NEUTRAL_SUPPORT``;
* **the electoral cycle**: a honeymoon worth ``HONEYMOON`` of the base just after the
  election, halving every ``HONEYMOON_HALF_LIFE`` turns, and a mid-term slump of up to
  ``MIDTERM_SLUMP``, deepest halfway to the next election and gone by polling day;
* **credibility** (D9, ``policy.credibility.capital_factor``).

The refill never falls below ``MIN_REFILL`` of the base. Before the first turn is played the
state holds no capital yet, and the base is available: the first turn is the one every
earlier rule was tuned on, and the honeymoon shows from the second.
"""

from __future__ import annotations

import math

from hog_sim.core.config import GameConfig
from hog_sim.core.state import WorldState
from hog_sim.policy.credibility import capital_factor
from hog_sim.policy.feasibility import LEGISLATURE_ID
from hog_sim.population.popularity import vote_intention

BANK_SHARE = 0.5
MAX_STOCK = 2.0  # times the base
MIN_REFILL = 0.25  # times the base
APPROVAL_GAIN = 4.0
NEUTRAL_VOTE = 0.48
MAJORITY_GAIN = 2.0
NEUTRAL_SUPPORT = 0.6
HONEYMOON = 0.3
HONEYMOON_HALF_LIFE = 3.0
MIDTERM_SLUMP = 0.2


def cycle_factor(turn: int, election_turn: int) -> float:
    """The honeymoon and mid-term slump: 1.3 at turn 0, about 0.8 halfway, 1 at the end."""
    honeymoon = HONEYMOON * 0.5 ** (turn / HONEYMOON_HALF_LIFE)
    slump = MIDTERM_SLUMP * math.sin(math.pi * min(1.0, turn / max(1, election_turn))) ** 2
    return 1 + honeymoon - slump


def refill_factors(state: WorldState, config: GameConfig) -> dict[str, float]:
    """Each factor scaling this turn's refill, by name."""
    legislature = state.institutions.get(LEGISLATURE_ID)
    support = legislature.support if legislature is not None else NEUTRAL_SUPPORT
    return {
        "approval": max(0.0, 1 + APPROVAL_GAIN * (vote_intention(state) - NEUTRAL_VOTE)),
        "majority": max(0.0, 1 + MAJORITY_GAIN * (support - NEUTRAL_SUPPORT)),
        "cycle": cycle_factor(state.turn, config.election_turn),
        "credibility": capital_factor(state.credibility),
    }


def refill(state: WorldState, config: GameConfig) -> float:
    """Capital that comes in for ``state``'s turn."""
    base = config.capital_per_turn
    total = math.prod(refill_factors(state, config).values())
    return base * min(MAX_STOCK, max(MIN_REFILL, total))


def available(state: WorldState, config: GameConfig) -> float:
    """Capital the package for ``state``'s turn may spend."""
    return state.capital if state.capital is not None else config.capital_per_turn


def next_capital(new: WorldState, unspent: float, config: GameConfig) -> float:
    """Capital for ``new``'s turn: its refill plus the banked share of ``unspent``."""
    stock = refill(new, config) + BANK_SHARE * max(0.0, unspent)
    return min(MAX_STOCK * config.capital_per_turn, stock)
