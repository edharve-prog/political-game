import json

import pytest
from pydantic import ValidationError

from hog_sim.content.library import (
    CATEGORIES,
    Condition,
    LibraryScenario,
    Provenance,
    ScenarioLibrary,
    slugify,
)
from hog_sim.core.config import GameConfig
from hog_sim.core.models import Scenario
from hog_sim.game.loop import Game, replay
from hog_sim.game.stubs import EngineForecaster, KeywordInterpreter
from hog_sim.world.seed.toy import toy_world


def entry(title, category="economy", **kw):
    return LibraryScenario(
        id=kw.pop("id", slugify(title)),
        category=category,
        title=title,
        briefing=f"{title}.",
        affected_nodes=kw.pop("affected_nodes", ["indicator:inflation"]),
        urgency=0.5,
        **kw,
    )


@pytest.fixture
def world():
    return toy_world()


def test_categories_and_slug() -> None:
    assert len(CATEGORIES) == 12 and "energy" in CATEGORIES
    assert slugify("Gas prices spike!") == "gas-prices-spike"
    with pytest.raises(ValidationError):
        entry("Bad", category="sport")


def test_library_scenario_is_a_scenario_and_keeps_core_source() -> None:
    s = entry("Rates rise", origin="llm", provenance=Provenance(model="m", turn=3))
    assert isinstance(s, Scenario) and s.source == "generated" and s.origin == "llm"


def test_builtin_files_load_and_fit_the_toy_world(world) -> None:
    library = ScenarioLibrary.load()
    assert len(library.scenarios) >= 5
    assert library.check(world) == []


def test_dedupes_by_id_then_title(tmp_path) -> None:
    export = tmp_path / "harvest.jsonl"
    rows = [entry("Rates rise", id="llm-rates-rise-1a2b3c4d", origin="llm"), entry("Rates  RISE")]
    export.write_text("\n".join(r.model_dump_json() for r in rows))
    listed = tmp_path / "more.json"
    listed.write_text(json.dumps([entry("Rates rise", id="other").model_dump(mode="json")]))
    library = ScenarioLibrary.load(
        export,
        listed,
        [entry("Bond wobble"), entry("x", id="llm-rates-rise-1a2b3c4d")],
        builtin=False,
    )
    assert [s.id for s in library.scenarios] == ["llm-rates-rise-1a2b3c4d", "bond-wobble"]


def test_conditions_filter_and_unknown_nodes_are_skipped(world) -> None:
    hot = entry("Inflation bites", conditions={"indicator:inflation": Condition(op=">", value=9)})
    alien = entry("Mars crisis", affected_nodes=["country:mars"])
    calm = entry("Quiet month")
    library = ScenarioLibrary([hot, alien, calm])
    assert library.check(world) == ["mars-crisis: unknown node ids ['country:mars']"]
    assert library.next_scenario(world, []).title == "Quiet month"
    world.indicators["indicator:inflation"].value = 10
    assert hot.applies_to(world)


def test_no_repeat_within_window_and_replays(world) -> None:
    library = ScenarioLibrary([entry(f"Issue {i}") for i in range(20)], seed=3)
    config = GameConfig(election_turn=16, k_draws=10)
    game = Game(config, world, library, KeywordInterpreter(), EngineForecaster())
    for _ in range(16):
        game.play_turn("do nothing")
    titles = [r.scenario.title for r in game.history]
    assert len(set(titles)) == 16
    assert replay(world, config, game.history) == game.state


def test_builtin_library_meets_sd3(world) -> None:
    library = ScenarioLibrary.load()
    counts = {c: 0 for c in CATEGORIES}
    for s in library.scenarios:
        counts[s.category] += 1
    assert len(library.scenarios) >= 60 and min(counts.values()) >= 5
    config = GameConfig(election_turn=30, k_draws=10)
    game = Game(config, world, library, KeywordInterpreter(), EngineForecaster())
    for i in range(30):
        game.play_turn(["tax energy profits", "spend on the NHS", "do nothing"][i % 3])
    titles = [r.scenario.title for r in game.history]
    assert all(t not in titles[max(0, i - 15) : i] for i, t in enumerate(titles))
