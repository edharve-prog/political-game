"""Game configuration and seeded randomness."""

from __future__ import annotations

import random
from typing import Literal

from pydantic import Field

from hog_sim.core.models import Model


class GameConfig(Model):
    seed: int = 0
    role: Literal["prime_minister", "president"] = "prime_minister"
    # A turn is one month everywhere: the calendar (Budget on turns 3 and 15), effect lags,
    # half-lives and the calibrated backtests all count months. Nothing reads this field, so
    # any other value would claim a turn length the game does not play. It stays so saved
    # configs still load, and is fixed at 1 until the engine can scale time.
    turn_length_months: int = Field(1, ge=1, le=1)
    # Difficulty profile for picking the outcome: argmax, sample, fair or hard
    # (forecasting/selection.py).
    selection_mode: Literal["argmax", "sample", "fair", "hard"] = "sample"
    election_turn: int = Field(24, ge=1)
    horizon: int = Field(24, ge=1)  # turns the engine looks ahead
    k_draws: int = Field(100, ge=1)  # Monte Carlo draws per forecast
    # Political capital a turn's actions may use (policy/limits). Negative would flip the
    # sign of every scaled action.
    capital_per_turn: float = Field(1.5, ge=0)
    # D1: resolve plays on edge weights drawn once per game from their uncertainty, while the
    # forecast still shows the central weights and their spread. Off, the world behaves
    # exactly as the central forecast says (the calibration harnesses and older saves).
    hidden_world: bool = False


def make_rng(seed: int, turn: int = 0, stream: str = "") -> random.Random:
    """Deterministic RNG per (seed, turn, stream) so replays reproduce exactly."""
    return random.Random(f"{seed}:{turn}:{stream}")
