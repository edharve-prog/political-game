"""RB-5: save an action package and reuse it on a later turn."""

import builtins

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import PolicyAction
from hog_sim.game.loop import Game
from hog_sim.game.packages import PolicyLibrary
from hog_sim.game.persistence import SaveStore
from hog_sim.game.stubs import CannedScenarios, EngineForecaster, KeywordInterpreter
from hog_sim.ui import cli
from hog_sim.world.seed.toy import toy_world

TAX = PolicyAction(kind="tax", target="sector:energy", magnitude=0.4, duration_turns=3)
HOMES = PolicyAction(kind="spend", target="sector:housing", magnitude=0.3)


@pytest.fixture
def store(tmp_path):
    store = SaveStore(tmp_path / "game.db")
    yield store
    store.close()


def test_packages_save_load_and_list_per_game(store) -> None:
    mine = PolicyLibrary(store.conn, "game-a")
    other = PolicyLibrary(store.conn, "game-b")
    assert mine.save("Windfall", [TAX, HOMES], 2) == "windfall"
    assert mine.get("windfall") == [TAX, HOMES]
    assert mine.names() == ["windfall"] and other.names() == []
    assert "windfall: tax sector:energy +0.40; spend sector:housing +0.30" in mine.listing()
    mine.save("windfall", [HOMES], 3)  # saving again replaces it
    assert mine.get("windfall") == [HOMES]


@pytest.mark.parametrize("name", ["two words", "", "x" * 41, "semi;colon"])
def test_a_package_name_is_one_short_word(store, name) -> None:
    with pytest.raises(ValueError, match="one word"):
        PolicyLibrary(store.conn, "g").save(name, [TAX], 0)


def test_empty_and_unknown_packages_are_refused(store) -> None:
    library = PolicyLibrary(store.conn, "g")
    with pytest.raises(ValueError, match="no actions"):
        library.save("nothing", [], 0)
    with pytest.raises(ValueError, match="no package called 'missing'"):
        library.get("missing")


def test_export_and_import_move_packages_between_games(store) -> None:
    source = PolicyLibrary(store.conn, "game-a")
    source.save("windfall", [TAX], 2)
    target = PolicyLibrary(store.conn, "game-b")
    assert target.import_(source.export()) == 1
    assert target.get("windfall") == [TAX]


def test_the_cli_saves_a_package_and_loads_it_next_turn(store, monkeypatch) -> None:
    config = GameConfig(seed=0, election_turn=6, k_draws=20)
    plugins = (CannedScenarios(0, calendar=False), KeywordInterpreter(), EngineForecaster())
    game = Game(config, toy_world(), *plugins)
    library = PolicyLibrary(store.conn, store.new_game(config, game.start))
    answers = iter(
        [
            "Put a windfall tax on energy companies",
            "save windfall",
            "",
            "packages",
            "use windfall",
            "",
        ]
    )
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))

    first = game.commit(cli._ask_for_turn(game, library=library))
    assert library.get("windfall") == first.actions
    second = cli._ask_for_turn(game, library=library)
    assert second.response == "(package windfall)"
    assert [(a.kind, a.target) for a in second.actions] == [("tax", "sector:energy")]


def test_hog_sim_packages_exports_and_imports(tmp_path, capsys) -> None:
    save = tmp_path / "game.db"
    store = SaveStore(save)
    game_id = store.new_game(GameConfig(), toy_world())
    PolicyLibrary(store.conn, game_id).save("windfall", [TAX], 1)
    store.close()

    out = tmp_path / "packages.json"
    cli.main(["packages", "--save", str(save), "export", str(out)])
    cli.main(["packages", "--save", str(save), "list"])
    assert "windfall: tax sector:energy +0.40" in capsys.readouterr().out

    store = SaveStore(save)
    newer = store.new_game(GameConfig(), toy_world())
    store.close()
    cli.main(["packages", "--save", str(save), "--game", newer, "import", str(out)])
    store = SaveStore(save)
    assert PolicyLibrary(store.conn, newer).get("windfall") == [TAX]
    store.close()
