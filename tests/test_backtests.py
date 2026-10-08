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
    play,
    problems,
    report,
    run,
    sizes,
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


def test_real_ranges_agree_with_their_direction() -> None:
    for episode in EPISODES:
        for check in episode.checks:
            if check.real is not None:
                low, high = check.real
                assert low < high, f"{episode.name}: {check.describe()}"
                assert low * check.direction >= 0 and high * check.direction > 0


def test_sector_sizes_are_per_cent_of_output() -> None:
    episode = Episode(
        name="cuts",
        happened="",
        actions=[act("spend", "sector:public", -0.6, 12)],
        checks=[Check(node="sector:public", direction=-1, real=(-4, -1))],
    )
    result = evaluate(episode)[0]
    lowest = min(s.node("sector:public").output_bn for s in play(episode))
    start = toy_world().node("sector:public").output_bn
    assert result.change == pytest.approx(100 * (lowest - start) / start)
    assert result.size == pytest.approx(result.change / -2.5)


def test_size_report_lists_every_sized_check() -> None:
    text = sizes(RESULTS)
    sized = [r for r in RESULTS if r.check.real is not None]
    assert f"of {len(sized)} sized checks" in text
    assert all(r.check.describe() in text for r in sized)
