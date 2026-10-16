"""The final review's offline reading fixes (FR-7 to FR-12): several measures in a sentence,
cutting bad things, mirror-image requests, possessives, holding firm, tax wording, and
stored readings that can be flagged."""

import pytest

from hog_sim.core.models import LOW_CONFIDENCE, PolicyAction, Scenario
from hog_sim.game.loop import not_possible
from hog_sim.game.stubs import KeywordInterpreter, keyword_pledges, measures
from hog_sim.knowledge.entries import (
    NOT_POSSIBLE,
    InterpretationEntry,
    Provenance,
    reuse_context,
    scenario_id,
)
from hog_sim.knowledge.offline import StoredInterpreter, reuse_confidence, same_request
from hog_sim.knowledge.store import KnowledgeStore
from hog_sim.world.seed.toy import toy_world

QUIET = Scenario(title="Quiet", briefing="", affected_nodes=["sector:housing"], urgency=0)
HOUSEHOLDS = sorted(g for g in toy_world().groups if g != "group:business")


def read(text: str):
    interpreter = KeywordInterpreter()
    actions = interpreter.interpret(text, toy_world(), QUIET)
    return interpreter, actions


def summary(actions: list[PolicyAction]) -> list[tuple[str, str, int]]:
    """(kind, target, sign of magnitude), households' shares folded into one."""
    out = []
    for a in actions:
        target = "households" if a.target in HOUSEHOLDS else a.target
        item = (a.kind, target, (a.magnitude > 0) - (a.magnitude < 0))
        if item not in out:
            out.append(item)
    return out


# FR-7: one action per measure, each with its own target and direction.


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Raise taxes on banks and cut taxes for families",
            [("tax", "sector:finance", 1), ("tax", "households", -1)],
        ),
        (
            "Cut taxes for families and raise spending on schools",
            [("tax", "households", -1), ("spend", "sector:public", 1)],
        ),
        (
            "Spend more on schools, cut taxes for business and regulate banks",
            [
                ("spend", "sector:public", 1),
                ("tax", "group:business", -1),
                ("regulate", "sector:finance", 1),
            ],
        ),
        (
            "Raise taxes on banks while cutting taxes for families",
            [("tax", "sector:finance", 1), ("tax", "households", -1)],
        ),
    ],
)
def test_each_measure_in_a_sentence_is_its_own_action(text, expected) -> None:
    interpreter, actions = read(text)
    assert summary(actions) == expected
    assert all(not r.doubtful for r in interpreter.last_readings())


@pytest.mark.parametrize(
    "text",
    ["Help households and families", "Tax banks and building societies", "Cut taxes and spending"],
)
def test_nouns_after_and_are_not_new_measures(text) -> None:
    assert measures(text) == [text]


def test_a_rise_and_a_cut_that_cannot_be_split_ask_the_player_to_check() -> None:
    interpreter, _ = read("Raise taxes on banks, families get tax cuts")
    assert any(r.doubtful for r in interpreter.last_readings())


# FR-8: cutting a bad thing means acting to reduce it.


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Cut hospital waiting lists", ("spend", "sector:public", 1)),
        ("Reduce NHS waiting lists", ("spend", "sector:public", 1)),
        ("Slash hospital waiting times", ("spend", "sector:public", 1)),
        ("Waiting lists must come down", ("spend", "sector:public", 1)),
        ("Cut crime", ("spend", "sector:public", 1)),
        ("Cut child poverty", ("spend", "households", 1)),
        ("Cut emissions", ("regulate", "sector:energy", 1)),
        ("Cut red tape for business", ("deregulate", "group:business", 1)),
        # About money, so still a spending cut.
        ("Cut spending on hospitals", ("spend", "sector:public", -1)),
        ("Cut funding for crime prevention", ("spend", "sector:public", -1)),
    ],
)
def test_cutting_a_bad_thing_acts_against_it(text, expected) -> None:
    assert summary(read(text)[1]) == [expected]


# FR-9: "X to fund Y" mirror images are different requests.


@pytest.mark.parametrize(
    "a, b",
    [
        ("tax business to fund pensioners", "tax pensioners to fund business"),
        ("tax business to help pensioners", "tax pensioners to help business"),
        ("tax the banks to pay for nurses", "tax the nurses to pay for banks"),
        ("fund nurses by taxing banks", "fund banks by taxing nurses"),
    ],
)
def test_mirror_images_never_match(a, b) -> None:
    assert same_request(a, b) == 0.0


def test_a_purpose_reworded_with_filler_still_matches() -> None:
    assert same_request("Tax the banks to fund the nurses", "tax banks to fund nurses") >= 0.6


# FR-10: a possessive names something of the minister's, not the minister.


@pytest.mark.parametrize(
    "text",
    [
        "Replace the Chancellor's fiscal rule with a new one",
        "Back the Chancellor's plan",
        "Fire the Chancellor's adviser",
        "Replace the Chancellor’s budget rule",
    ],
)
def test_a_possessive_sacks_nobody(text) -> None:
    interpreter, _ = read(text)
    assert interpreter.last_sacked() == []


def test_sacking_the_chancellor_still_sacks() -> None:
    assert read("Sack the Chancellor and cut taxes for business")[0].last_sacked() == [
        "person:chancellor"
    ]


# FR-11: holding firm or waiting doesn't cancel the rest; tax and red tape wording.


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hold firm on pay but fund a small pay rise for nurses", ("spend", "sector:public", 1)),
        ("Wait for the figures then cut taxes for business", ("tax", "group:business", -1)),
        ("Hold firm and build more homes", ("spend", "sector:housing", 1)),
    ],
)
def test_holding_firm_keeps_the_other_measures(text, expected) -> None:
    assert summary(read(text)[1]) == [expected]


def test_holding_firm_alone_still_does_nothing() -> None:
    assert summary(read("Hold firm on pay")[1]) == [("do_nothing", "country:uk", 0)]


def test_vat_on_bills_is_a_tax_on_energy_customers() -> None:
    _, [action] = read("Cut VAT on energy bills")
    assert (action.kind, action.target, action.borne_by) == ("tax", "sector:energy", "consumers")
    assert action.magnitude < 0


@pytest.mark.parametrize(
    "text", ["Cut income tax to boost growth", "Lower income tax", "Cut national insurance"]
)
def test_income_tax_falls_on_households(text) -> None:
    assert summary(read(text)[1]) == [("tax", "households", -1)]


# FR-12: pledges, stored readings and the reuse context.


@pytest.mark.parametrize(
    "text, direction",
    [
        ("No new tax cuts", "down"),
        ("We will not cut taxes", "down"),
        ("We will not raise taxes", "up"),
        ("No new taxes", "up"),
        ("No tax rises", "up"),
    ],
)
def test_tax_pledges_point_the_right_way(text, direction) -> None:
    [pledge] = keyword_pledges(text)[0]
    assert (pledge.kind, pledge.direction) == ("tax", direction)


SCENARIO = Scenario(title="Waiting lists", briefing="", affected_nodes=["sector:public"], urgency=0)
SPEND = PolicyAction(kind="spend", target="sector:public", magnitude=0.5)
LOST = PolicyAction(kind="spend", target="sector:nowhere", magnitude=0.5)


def stored(tmp_path, response: str, actions: list[PolicyAction]) -> StoredInterpreter:
    store = KnowledgeStore(tmp_path / "k.db")
    sid = scenario_id(SCENARIO)
    entry = InterpretationEntry(
        scenario_id=sid,
        scenario_title=SCENARIO.title,
        response=response,
        actions=actions,
        provenance=Provenance(game_id="g", turn=0),
    )
    store.put("interpretation", f"{sid}|{response}", entry, sid)
    return StoredInterpreter(store, KeywordInterpreter())


def test_reused_readings_can_be_flagged(tmp_path) -> None:
    interpreter = stored(tmp_path, "Increase spending on hospitals by 5% this year", [SPEND])
    interpreter.interpret("Increase hospital spending by 5% this year", toy_world(), SCENARIO)
    assert interpreter.last_source == "stored"
    [reading] = interpreter.last_readings()
    assert reading.doubtful
    # The same words are as sure as when they were first read.
    interpreter.interpret("Increase spending on hospitals by 5% this year", toy_world(), SCENARIO)
    assert not interpreter.last_readings()[0].doubtful


def test_a_reused_reading_that_lost_a_measure_is_flagged(tmp_path) -> None:
    interpreter = stored(tmp_path, "Spend on hospitals and nowhere", [SPEND, LOST])
    interpreter.interpret("Spend on hospitals and nowhere", toy_world(), SCENARIO)
    assert interpreter.last_source == "stored"
    assert all(r.doubtful for r in interpreter.last_readings())


def test_reuse_confidence_scale() -> None:
    assert reuse_confidence(1.0) == 1.0
    assert reuse_confidence(0.95) >= LOW_CONFIDENCE
    assert reuse_confidence(0.7) < LOW_CONFIDENCE
    assert reuse_confidence(1.0, dropped=True) < LOW_CONFIDENCE


def test_unmapped_measures_are_part_of_the_reuse_context() -> None:
    assert not_possible(["x"]) == [f"x{NOT_POSSIBLE}"]
    plain = reuse_context(None, [], [])
    unmapped = reuse_context(None, [], not_possible(["Nationalise the moon"]))
    assert unmapped != plain and not unmapped.plain
    assert unmapped == reuse_context(None, [], not_possible(["nationalise the moon!"]))


def test_harvest_files_unmapped_measures_in_the_context(tmp_path) -> None:
    from hog_sim.core.config import GameConfig
    from hog_sim.game.loop import Game
    from hog_sim.game.stubs import CannedScenarios, EngineForecaster

    config = GameConfig(seed=0, election_turn=6, k_draws=5)
    plugins = (CannedScenarios(0, calendar=False), KeywordInterpreter(), EngineForecaster())
    game = Game(config, toy_world(), *plugins)
    before = game.state
    record = game.play_turn("Build more homes")
    record = record.model_copy(update={"unmapped": ["Nationalise the moon"]})
    store = KnowledgeStore(tmp_path / "k.db")
    store.harvest_turn("g1", record, before)
    (entry,) = store.outcomes(scenario_id(record.scenario))
    assert entry.context.blocked == ["Not possible: nationalise the moon"]
