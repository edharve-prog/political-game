"""Game configuration and seeded randomness."""

from __future__ import annotations

import random
from typing import Literal

from hog_sim.core.models import Model


class GameConfig(Model):
    seed: int = 0
    role: Literal["prime_minister", "president"] = "prime_minister"
    turn_length_months: int = 1
    selection_mode: Literal["argmax", "sample"] = "sample"
    election_turn: int = 24
    horizon: int = 24  # turns the engine looks ahead
    k_draws: int = 100  # Monte Carlo draws per forecast


def make_rng(seed: int, turn: int = 0, stream: str = "") -> random.Random:
    """Deterministic RNG per (seed, turn, stream) so replays reproduce exactly."""
    return random.Random(f"{seed}:{turn}:{stream}")
