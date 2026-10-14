"""Difficulty profiles for picking the outcome (review 2026-10-13, D10)."""

from hog_sim.core.config import GameConfig, make_rng
from hog_sim.core.models import ApprovalEvent, Outcome
from hog_sim.forecasting.selection import FAIR_BLOCK, fair_luck, goodness, select


def _outcome(name: str, probability: float, change: float) -> Outcome:
    event = ApprovalEvent(name=name, group_effects={"group:a": change})
    return Outcome(narrative=name, probability=probability, approval_events=[event])


GOOD = _outcome("good", 0.25, 0.05)
MIDDLE = _outcome("middle", 0.25, 0.0)
POOR = _outcome("poor", 0.25, -0.03)
AWFUL = _outcome("awful", 0.25, -0.08)


def test_goodness_weights_groups() -> None:
    both = Outcome(
        narrative="x",
        probability=1,
        approval_events=[ApprovalEvent(name="x", group_effects={"a": 0.1, "b": -0.1})],
    )
    assert goodness(both) == 0
    assert goodness(both, {"a": 3.0, "b": 1.0}) > 0


def test_fair_luck_covers_every_quarter_in_each_block() -> None:
    for seed in range(5):
        for block in range(6):
            luck = [fair_luck(seed, block * FAIR_BLOCK + s) for s in range(FAIR_BLOCK)]
            assert sorted(int(u * FAIR_BLOCK) for u in luck) == list(range(FAIR_BLOCK))


def test_fair_never_clusters_the_worst_outcome() -> None:
    # Four equally likely outcomes: fair gives each exactly once in every aligned block.
    cands = [POOR, GOOD, AWFUL, MIDDLE]
    for seed in range(5):
        picks = [select(cands, "fair", make_rng(seed, t), seed=seed, turn=t) for t in range(24)]
        for start in range(0, 24, FAIR_BLOCK):
            assert sorted(picks[start : start + FAIR_BLOCK]) == [0, 1, 2, 3]
    # Plain sampling does cluster for some seed.
    clustered = False
    for seed in range(20):
        picks = [select(cands, "sample", make_rng(seed, t)) for t in range(24)]
        if any(picks[s : s + FAIR_BLOCK].count(2) >= 2 for s in range(0, 24, FAIR_BLOCK)):
            clustered = True
    assert clustered


def test_fair_still_follows_probability() -> None:
    likely = _outcome("likely", 0.9, 0.0)
    rare = _outcome("rare", 0.1, -0.05)
    picks = [select([likely, rare], "fair", make_rng(1, t), seed=1, turn=t) for t in range(200)]
    assert 0.03 < picks.count(1) / 200 < 0.2


def test_hard_takes_the_worse_of_the_two_likeliest() -> None:
    cands = [
        _outcome("best", 0.4, 0.05),
        _outcome("worse", 0.35, -0.02),
        _outcome("worst but rare", 0.25, -0.1),
    ]
    assert select(cands, "hard", make_rng(0)) == 1
    assert select(cands, "argmax", make_rng(0)) == 0
    assert select(cands[:1], "hard", make_rng(0)) == 0


def test_config_accepts_every_profile() -> None:
    for mode in ("argmax", "sample", "fair", "hard"):
        assert GameConfig(selection_mode=mode).selection_mode == mode


def test_game_plays_and_replays_under_each_profile() -> None:
    from hog_sim.game.loop import Game, replay
    from hog_sim.game.stubs import CannedScenarios, EngineForecaster, KeywordInterpreter
    from hog_sim.world.seed.toy import toy_world

    for mode in ("fair", "hard"):
        config = GameConfig(seed=3, selection_mode=mode)
        start = toy_world()
        game = Game(config, start, CannedScenarios(), KeywordInterpreter(), EngineForecaster())
        for _ in range(3):
            game.play_turn("do nothing")
        assert replay(start, config, game.history) == game.state
