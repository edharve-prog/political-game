"""Long-run stability (Project 11, CA-2): 120 offline turns neither explode nor flatline."""

import math

import pytest

from hog_sim.game.stability import LIMITS, TEST_PLAYERS, TURNS, WINDOW, Run, problems, run


@pytest.fixture(scope="module")
def runs() -> list[Run]:
    return run(TEST_PLAYERS)


def test_long_games_are_stable(runs) -> None:
    assert problems(runs) == []


def test_games_run_the_full_length(runs) -> None:
    assert all(len(r.votes) == TURNS for r in runs)


def _run(steps: list[float], votes: list[float] | None = None, at_bound: int = 0) -> Run:
    moving = [0.1 * math.sin(t) for t in range(len(steps))]
    return Run(
        player="probe",
        seed=0,
        steps={"indicator:a": steps, "indicator:b": moving},
        votes=votes or [0.45 + 0.02 * math.sin(t) for t in range(len(steps))],
        turns_at_bound=at_bound,
        indicator_turns=2 * len(steps),
    )


def test_a_steady_trend_counts_as_a_runaway() -> None:
    trend = [0.05 * t for t in range(TURNS)]  # 6 steps by the end, inside the hard limit
    assert any("keeps growing" in p for p in problems([_run(trend)]))


def test_one_late_shock_that_fades_is_fine() -> None:
    quiet = [0.0] * 90
    shock = [1.5 * 0.85**t for t in range(TURNS - 90)]
    noisy = [v + 0.05 * math.sin(t) for t, v in enumerate(quiet + shock)]
    assert problems([_run(noisy)]) == []


def test_wandering_too_far_is_caught() -> None:
    far = [LIMITS["max_steps_from_start"] + 1] * TURNS
    assert any("steps from its start" in p for p in problems([_run(far)]))


def test_a_flat_vote_is_caught() -> None:
    wiggle = [0.1 * math.sin(t) for t in range(TURNS)]
    flat = [0.47] * TURNS
    assert any("flatlined" in p for p in problems([_run(wiggle, votes=flat)]))


def test_still_indicators_are_caught() -> None:
    r = _run([0.0] * TURNS)
    r.steps["indicator:b"] = [0.0] * TURNS
    assert any("still move" in p for p in problems([r]))


def test_non_finite_numbers_are_caught() -> None:
    r = _run([0.0] * TURNS)
    # Models refuse NaN, so build one past validation, as a broken engine would.
    broken = Run.model_construct(**{**r.__dict__, "votes": [math.nan] * TURNS})
    assert any("non-finite" in p for p in problems([broken]))


def test_sitting_at_a_bound_is_caught() -> None:
    wiggle = [0.1 * math.sin(t) for t in range(TURNS)]
    assert any("hard bound" in p for p in problems([_run(wiggle, at_bound=TURNS)]))


def test_window_fits_inside_a_game() -> None:
    assert WINDOW < TURNS
