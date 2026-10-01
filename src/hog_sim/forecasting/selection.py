"""Choose one outcome from scored candidates."""

from __future__ import annotations

import random
from typing import Literal

from hog_sim.core.models import Outcome


def select(candidates: list[Outcome], mode: Literal["argmax", "sample"], rng: random.Random) -> int:
    """Index of the chosen candidate. ``argmax`` picks the most likely, ``sample`` draws by
    probability (the default difficulty, so the game can't be solved)."""
    if not candidates:
        raise ValueError("no candidate outcomes")
    if mode == "argmax":
        return max(range(len(candidates)), key=lambda i: candidates[i].probability)
    weights = [max(c.probability, 0.0) for c in candidates]
    if sum(weights) == 0:
        weights = [1.0] * len(candidates)
    return rng.choices(range(len(candidates)), weights=weights)[0]
