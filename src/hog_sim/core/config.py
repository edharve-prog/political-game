"""Game configuration and seeded randomness."""

from __future__ import annotations

import random
from typing import Literal

from pydantic import Field

from hog_sim.core.models import Model


class GameConfig(Model):
    seed: int = 0
    role: Literal["prime_minister", "president"] = "prime_minister"
    turn_length_months: int = Field(1, ge=1)
    selection_mode: Literal["argmax", "sample"] = "sample"
    election_turn: int = Field(24, ge=1)
    horizon: int = Field(24, ge=1)  # turns the engine looks ahead
    k_draws: int = Field(100, ge=1)  # Monte Carlo draws per forecast
    # Political capital a turn's actions may use (policy/limits). Negative would flip the
    # sign of every scaled action.
    capital_per_turn: float = Field(1.5, ge=0)


def make_rng(seed: int, turn: int = 0, stream: str = "") -> random.Random:
    """Deterministic RNG per (seed, turn, stream) so replays reproduce exactly."""
    return random.Random(f"{seed}:{turn}:{stream}")
