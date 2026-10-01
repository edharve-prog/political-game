import pytest

from hog_sim.core.config import GameConfig, make_rng
from hog_sim.core.models import Outcome, PolicyAction
from hog_sim.forecasting.selection import select
from hog_sim.game.loop import Game, replay
from hog_sim.game.persistence import SaveStore
from hog_sim.game.stubs import CannedScenarios, EngineForecaster, KeywordInterpreter
from hog_sim.world.seed.toy import toy_world

RESPONSES = [
    "Put a windfall tax on energy companies",
    "Invest in housebuilding",
    "Do nothing",
    "Fund a pay rise for nurses and teachers",
    "Negotiate with China",
    "Cut public spending",
    "Major investment in manufacturing",
    "Small tax cut for banks",
    "Regulate energy prices with a cap",
    "Wait",
    "Deregulate planning",
    "Support exporters",
]


def new_game(seed=0, election_turn=12):
    config = GameConfig(seed=seed, election_turn=election_turn, k_draws=20)
    plugins = (CannedScenarios(seed), KeywordInterpreter(), EngineForecaster())
    return Game(config, toy_world(), *plugins), plugins


def test_full_game_runs_to_election() -> None:
    game, _ = new_game()
    for response in RESPONSES:
        record = game.play_turn(response)
    assert game.over
    assert game.state.turn == 12
    assert record.election is not None
    assert all(r.election is None for r in game.history[:-1])
    with pytest.raises(RuntimeError):
        game.play_turn("one more")


def test_lagged_effects_keep_arriving() -> None:
    game, _ = new_game()
    game.play_turn("Put a windfall tax on energy companies")
    assert any(t > game.state.turn - 1 for t in game.state.pending)
    before = game.state.indicators["indicator:house_prices"].value
    for _ in range(6):
        game.play_turn("Do nothing")
    assert game.state.indicators["indicator:house_prices"].value != before


def test_same_seed_same_game() -> None:
    a, _ = new_game(seed=3)
    b, _ = new_game(seed=3)
    for response in RESPONSES[:6]:
        a.play_turn(response)
        b.play_turn(response)
    assert a.state == b.state
    assert [r.scenario.title for r in a.history] == [r.scenario.title for r in b.history]


def test_save_resume_and_replay(tmp_path) -> None:
    store = SaveStore(tmp_path / "game.db")
    game, plugins = new_game()
    game_id = store.new_game(game.config, game.start)
    for response in RESPONSES[:5]:
        store.save_turn(game_id, game.play_turn(response))
    store.close()

    store = SaveStore(tmp_path / "game.db")
    assert store.latest_game() == game_id
    config, start, records = store.load(game_id)
    resumed = Game.resume(config, start, records, *plugins)
    assert resumed.state == game.state
    assert resumed.scenario == game.scenario

    for response in RESPONSES[5:]:
        game.play_turn(response)
        resumed.play_turn(response)
    assert resumed.state == game.state
    assert replay(start, config, resumed.history) == game.state


def test_keyword_interpreter() -> None:
    world = toy_world()
    scenario = CannedScenarios().next_scenario(world, [])
    interpret = KeywordInterpreter().interpret
    [a] = interpret("Put a big windfall tax on energy firms", world, scenario)
    assert (a.kind, a.target, a.magnitude) == ("tax", "sector:energy", 0.9)
    [a] = interpret("cut taxes for banks", world, scenario)
    assert (a.kind, a.target) == ("tax", "sector:finance") and a.magnitude < 0
    [a] = interpret("Retaliate against China", world, scenario)
    assert (a.kind, a.target) == ("diplomatic", "country:china") and a.magnitude < 0
    [a] = interpret("do nothing", world, scenario)
    assert a.kind == "do_nothing"


def test_selection_modes() -> None:
    cands = [
        Outcome(narrative="a", probability=0.2),
        Outcome(narrative="b", probability=0.7),
        Outcome(narrative="c", probability=0.1),
    ]
    assert select(cands, "argmax", make_rng(0)) == 1
    picks = [select(cands, "sample", make_rng(0, t)) for t in range(300)]
    assert set(picks) == {0, 1, 2}
    assert picks.count(1) > picks.count(0) > picks.count(2)


def test_backlash_hits_exposed_groups() -> None:
    world = toy_world()
    scenario = CannedScenarios().next_scenario(world, [])
    from hog_sim.world.propagation import propagate

    action = PolicyAction(kind="tax", target="sector:energy", magnitude=0.5)
    candidates = EngineForecaster().forecast(world, scenario, [action], propagate(world, [], 2, 1))
    assert sum(c.probability for c in candidates) == pytest.approx(1.0)
    backlash = candidates[1].approval_events[0].group_effects
    assert "group:pensioners" in backlash and all(v < 0 for v in backlash.values())
