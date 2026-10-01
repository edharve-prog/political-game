"""No single lever wins elections, doing nothing loses, and the economy stays sane (EB-1)."""

import pytest

from hog_sim.game.balance import play, problems, run, table


@pytest.fixture(scope="module")
def results():
    out = run(seeds=3)
    print("\n" + table(out))
    return out


def test_balance_within_limits(results) -> None:
    assert problems(results) == []


def test_a_long_game_stays_inside_plausible_bounds() -> None:
    game = play("spend on everything", seed=0, turns=120)
    assert game.turns_at_bound == 0
