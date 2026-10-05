"""Historical backtests (Project 11, CA-1): the engine moves the way real episodes did.

Checks with a ``gap`` note are known to fail; they must keep failing until the gap is closed,
so whoever closes one also updates its check.
"""

import pytest

from hog_sim.game.backtests import (
    EPISODES,
    VOTE,
    Check,
    Episode,
    act,
    evaluate,
    problems,
    report,
    run,
)
from hog_sim.world.seed.toy import toy_world

RESULTS = run()


@pytest.mark.parametrize("result", RESULTS, ids=lambda r: f"{r.episode}: {r.check.describe()}")
def test_check_matches_its_record(result) -> None:
    if result.check.gap is None:
        assert result.passed, f"moved {result.peak:+.2f} steps"
    else:
        assert not result.passed, "this gap is closed now; remove its gap note"


def test_no_problems() -> None:
    assert problems(RESULTS) == []


def test_episodes_name_real_nodes() -> None:
    world = toy_world()
    for episode in EPISODES:
        for check in episode.checks:
            if check.node != VOTE:
                world.node(check.node)
        for action in episode.actions:
            world.node(action.target)
        for shock in episode.shocks:
            world.node(shock.node)


def test_most_checks_pass() -> None:
    assert sum(r.passed for r in RESULTS) / len(RESULTS) >= 0.7


def test_tiny_moves_do_not_count() -> None:
    episode = Episode(
        name="whisper",
        happened="",
        actions=[act("communicate", "sector:energy", 0.01)],
        checks=[Check(node="indicator:energy_prices", direction=-1)],
    )
    assert not evaluate(episode)[0].passed


def test_report_counts_gaps() -> None:
    text = report(RESULTS)
    gaps = sum(r.check.gap is not None for r in RESULTS)
    assert f"{gaps} known gaps" in text
