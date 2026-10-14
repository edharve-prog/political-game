"""Choose one outcome from scored candidates.

Four difficulty profiles (``GameConfig.selection_mode``, D10):

* ``argmax``: always the most likely outcome.
* ``sample``: a draw by probability (the default, so the game can't be solved).
* ``fair``: a draw by probability whose luck is stratified over blocks of ``FAIR_BLOCK``
  turns. Candidates are ranked best first for the government, and each block takes one draw
  from each quarter of the luck range in a seeded order, so bad outcomes can't cluster: in
  any four aligned turns the player gets one draw from the unluckiest quarter, not three.
* ``hard``: the worse, for the government, of the two likeliest outcomes.

"Better for the government" is ``goodness``: the turnout-weighted approval change the
outcome's approval events bring, which is what moves the vote.
"""

from __future__ import annotations

import random
from typing import Literal

from hog_sim.core.config import make_rng
from hog_sim.core.models import Outcome

SelectionMode = Literal["argmax", "sample", "fair", "hard"]

FAIR_BLOCK = 4  # turns per stratified block; each takes one draw from each quarter of luck


def goodness(outcome: Outcome, weights: dict[str, float] | None = None) -> float:
    """Turnout-weighted approval change the outcome's events bring (higher is better for
    the government). ``weights`` maps group id to its vote weight; missing groups
    count 1. Only the order between outcomes matters."""
    return sum(
        (weights or {}).get(group, 1.0) * change
        for event in outcome.approval_events
        for group, change in event.group_effects.items()
    )


def fair_luck(seed: int, turn: int) -> float:
    """This turn's luck in [0, 1): 0 is the best draw, near 1 the worst. Within each block
    of ``FAIR_BLOCK`` turns every quarter of the range comes up exactly once."""
    block, slot = divmod(turn, FAIR_BLOCK)
    order = list(range(FAIR_BLOCK))
    make_rng(seed, block, "fair-order").shuffle(order)
    jitter = make_rng(seed, turn, "fair-jitter").random()
    return (order[slot] + jitter) / FAIR_BLOCK


def _weights(candidates: list[Outcome]) -> list[float]:
    weights = [max(c.probability, 0.0) for c in candidates]
    return weights if sum(weights) > 0 else [1.0] * len(candidates)


def select(
    candidates: list[Outcome],
    mode: SelectionMode,
    rng: random.Random,
    *,
    seed: int = 0,
    turn: int = 0,
    group_weights: dict[str, float] | None = None,
) -> int:
    """Index of the chosen candidate. ``seed`` and ``turn`` drive ``fair``'s stratified
    luck; ``group_weights`` (turnout-weighted population) rank outcomes for ``fair`` and
    ``hard``."""
    if not candidates:
        raise ValueError("no candidate outcomes")
    n = len(candidates)
    if mode == "argmax":
        return max(range(n), key=lambda i: candidates[i].probability)
    if mode == "hard":
        likeliest = sorted(range(n), key=lambda i: (-candidates[i].probability, i))[:2]
        # The worse of the two; on a tie, the likelier.
        return min(
            likeliest,
            key=lambda i: (goodness(candidates[i], group_weights), -candidates[i].probability, i),
        )
    weights = _weights(candidates)
    if mode == "fair":
        ranked = sorted(range(n), key=lambda i: (-goodness(candidates[i], group_weights), i))
        point = fair_luck(seed, turn) * sum(weights)
        cumulative = 0.0
        for i in ranked:
            cumulative += weights[i]
            if point < cumulative:
                return i
        return ranked[-1]
    return rng.choices(range(n), weights=weights)[0]
