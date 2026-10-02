"""Project 15: graph changes, the knowledge store, precedents and offline reuse."""

import json
import sqlite3

import pytest

from hog_sim.content.library import ScenarioLibrary
from hog_sim.core.config import GameConfig
from hog_sim.core.models import EdgeKind, GraphChange, PolicyAction
from hog_sim.forecasting.candidates import (
    CandidateDraft,
    CandidateSet,
    GraphChangeDraft,
    Judgement,
    LLMForecaster,
    Shift,
    Verdict,
)
from hog_sim.game.llm_plugins import llm_plugins
from hog_sim.game.loop import Game, replay
from hog_sim.game.stubs import EngineForecaster, KeywordInterpreter
from hog_sim.knowledge.entries import action_signature, guess_category, scenario_id
from hog_sim.knowledge.offline import StoredForecaster, StoredInterpreter
from hog_sim.knowledge.recall import Recaller, recall
from hog_sim.knowledge.store import KnowledgeStore
from hog_sim.llm.client import FakeClient
from hog_sim.llm.interpreter import Interpretation
from hog_sim.llm.scenario_gen import GeneratedScenario, ScenarioDraft, StakeholderPosition
from hog_sim.ui.cli import main
from hog_sim.world.changes import apply_graph_changes, validate_graph_changes
from hog_sim.world.propagation import actions_to_shocks, propagate
from hog_sim.world.seed.toy import toy_world

TAX = PolicyAction(kind="tax", target="sector:energy", magnitude=0.5, rationale="windfall tax")
RESPONSE = "Put a windfall tax on energy companies"

CHINA_COOLS = GraphChange(
    kind="node_attr", node="country:china", attr="relationship", delta=-0.1, reason="row"
)
EU_TIES = GraphChange(
    kind="edge_weight",
    source="country:uk",
    target="country:eu",
    edge_kind=EdgeKind.TRADES_WITH,
    delta=0.05,
)
NEW_LINK = GraphChange(
    kind="add_edge",
    source="sector:finance",
    target="sector:public",
    edge_kind=EdgeKind.SUPPLIES,
    delta=0.2,
    lag=2,
)


@pytest.fixture
def world():
    return toy_world()


# --- graph changes ---------------------------------------------------------


def test_valid_graph_changes_apply(world) -> None:
    assert validate_graph_changes(world, [CHINA_COOLS, EU_TIES, NEW_LINK]) == []
    new = apply_graph_changes(world, [CHINA_COOLS, EU_TIES, NEW_LINK])
    assert new.countries["country:china"].relationship == pytest.approx(
        world.countries["country:china"].relationship - 0.1
    )
    weights = {(e.source, e.target, e.kind): e.weight for e in new.edges}
    assert weights[("country:uk", "country:eu", EdgeKind.TRADES_WITH)] == pytest.approx(0.5)
    assert weights[("sector:finance", "sector:public", EdgeKind.SUPPLIES)] == 0.2
    assert len(world.edges) == len(new.edges) - 1  # the input state is untouched


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            GraphChange(kind="node_attr", node="country:mars", attr="stability", delta=0.1),
            "unknown",
        ),
        (GraphChange(kind="node_attr", node="country:china", attr="gdp_bn", delta=0.1), "cannot"),
        (CHINA_COOLS.model_copy(update={"delta": -0.5}), "<= 0.2"),
        (EU_TIES.model_copy(update={"delta": 0.4}), "at most"),
        (EU_TIES.model_copy(update={"kind": "add_edge"}), "already exists"),
        (NEW_LINK.model_copy(update={"kind": "edge_weight"}), "no edge"),
        (NEW_LINK.model_copy(update={"edge_kind": EdgeKind.CARES_ABOUT}), "cannot be created"),
        (NEW_LINK.model_copy(update={"lag": 12}), "lag"),
    ],
)
def test_invalid_graph_changes_are_rejected_and_skipped(world, change, message) -> None:
    problems = validate_graph_changes(world, [change])
    assert problems and message in problems[0]
    assert apply_graph_changes(world, [change]) == world


def test_too_many_graph_changes(world) -> None:
    extra = CHINA_COOLS.model_copy(update={"node": "country:eu"})
    problems = validate_graph_changes(world, [CHINA_COOLS, EU_TIES, NEW_LINK, extra])
    assert any("at most 3" in p for p in problems)


# --- a Claude game on a fake client ------------------------------------------


def change_draft(change: GraphChange) -> GraphChangeDraft:
    data = change.model_dump()
    return GraphChangeDraft(**{k: data[k] for k in GraphChangeDraft.model_fields})


def candidates(engine, changes=(CHINA_COOLS,)):
    expected = engine.nodes["indicator:inflation"].mean[2]

    def draft(title, p):
        return CandidateDraft(
            title=title,
            narrative=f"{title} after the windfall tax.",
            indicator_shifts=[Shift(node="indicator:inflation", change=expected)],
            group_effects=[],
            new_shocks=[],
            graph_changes=[change_draft(c) for c in changes],
            event_tags=["none"],
            self_probability=p,
        )

    return CandidateSet(candidates=[draft("Tax lands", 0.5), draft("Firms grumble", 0.5)])


SCENARIO = ScenarioDraft(
    title="Energy bills soar",
    category="energy",
    briefing="Bills are up and pensioners are worried.",
    affected_nodes=["indicator:energy_prices", "group:pensioners", "sector:energy"],
    urgency=0.7,
    suggested_options=["Tax windfall profits", "Do nothing"],
    stakeholder_positions=[
        StakeholderPosition(node="group:pensioners", stance=0.8, statement="Help us."),
        StakeholderPosition(node="sector:energy", stance=-0.6, statement="Hands off."),
    ],
)


def responder(engine):
    def respond(request):
        if request.schema_name == "ScenarioDraft":
            return SCENARIO
        if request.schema_name == "Interpretation":
            return Interpretation(actions=[TAX])
        if request.schema_name == "CandidateSet":
            return candidates(engine)
        if request.schema_name == "Verdict":
            return Verdict(
                judgements=[Judgement(index=i, probability=0.5, reason="r") for i in range(2)]
            )
        raise AssertionError(request.schema_name)

    return respond


@pytest.fixture
def engine(world):
    return propagate(world, actions_to_shocks([TAX]), horizon=6, k_draws=50)


def play_claude_game(world, engine, store, game_id, turns=3):
    from hog_sim.forecasting.candidates import ForecastConfig
    from hog_sim.knowledge.entries import Provenance

    client = FakeClient(responder=responder(engine))
    recaller = Recaller(store, game_id)
    config = GameConfig(election_turn=turns, k_draws=20)
    plugins = llm_plugins(client, forecast_config=ForecastConfig(n_candidates=2), recaller=recaller)
    game = Game(config, world, *plugins)
    while not game.over:
        before = game.state
        record = game.play_turn(RESPONSE)
        store.harvest_turn(game_id, record, before, Provenance(model="fake", prompt_version="t"))
    return game, client


def test_claude_game_fills_store_and_graph_changes_replay(world, engine, tmp_path) -> None:
    store = KnowledgeStore(tmp_path / "game.db")
    game, client = play_claude_game(world, engine, store, "g1")

    # The chosen outcome's graph change reached the world each turn, and replay agrees.
    start = world.countries["country:china"].relationship
    assert game.state.countries["country:china"].relationship == pytest.approx(start - 0.3)
    assert replay(world, game.config, game.history) == game.state

    stats = store.stats()
    assert stats["scenario"] == 1 and stats["interpretation"] == 1 and stats["outcome"] == 1
    assert stats["graph_change"] == 6  # 3 turns x 2 candidates
    assert sum(e.status == "applied" for e in store.graph_changes()) == 3

    [scenario] = store.library_scenarios()
    assert scenario.origin == "llm" and scenario.id == scenario_id(game.history[0].scenario)
    assert scenario.id.startswith("llm-energy-bills-soar-")
    assert scenario.category == "energy"
    assert scenario.stakeholder_positions  # kept, though the save drops them
    assert scenario.provenance.game_id == "g1" and scenario.provenance.model == "fake"

    # From turn 1 on, the outcome prompt cites this game's earlier turn as a precedent.
    outcome_prompts = [
        r.messages[0].content for r in client.requests if r.schema_name == "CandidateSet"
    ]
    assert "Precedents" not in outcome_prompts[0]
    assert 'Turn 0 of this game: "Energy bills soar"' in outcome_prompts[1]
    assert "Links in the world graph" in outcome_prompts[0]


def test_bad_graph_change_is_sent_back(world, engine) -> None:
    bad = candidates(engine, changes=[CHINA_COOLS.model_copy(update={"attr": "gdp_bn"})])
    client = FakeClient([bad, candidates(engine)])
    from hog_sim.forecasting.candidates import ForecastConfig

    forecaster = LLMForecaster(client, config=ForecastConfig(n_candidates=2, use_judge=False))
    scenario = GeneratedScenario(source="generated", **SCENARIO.model_dump())
    outcomes = forecaster.forecast(world, scenario, [TAX], engine)
    assert "gdp_bn" in client.requests[1].messages[-1].content
    assert outcomes[0].graph_changes == [CHINA_COOLS]


def test_second_game_scenario_prompt_gets_one_precedent(world, engine, tmp_path) -> None:
    store = KnowledgeStore(tmp_path / "game.db")
    play_claude_game(world, engine, store, "g1")
    stressed = world.model_copy(deep=True)
    stressed.indicators["indicator:energy_prices"].history = [80.0, 80.0, 80.0]
    lines = recall(store, stressed, game_id="g2")
    assert len(lines) == 1 and lines[0].startswith('Another game: "Energy bills soar"')
    assert recall(KnowledgeStore(tmp_path / "empty.db"), stressed) == []


# --- offline reuse -------------------------------------------------------------


def test_offline_game_reuses_claude_work(world, engine, tmp_path) -> None:
    store = KnowledgeStore(tmp_path / "game.db")
    claude_game, _ = play_claude_game(world, engine, store, "g1")
    claude_narratives = {c.narrative for c in claude_game.history[0].candidates}

    interpreter = StoredInterpreter(store, KeywordInterpreter())
    forecaster = StoredForecaster(store, EngineForecaster())
    library = ScenarioLibrary.load(store.library_scenarios(), builtin=False)
    game = Game(GameConfig(election_turn=2, k_draws=20), world, library, interpreter, forecaster)
    assert game.scenario.title == "Energy bills soar"

    record = game.play_turn("put a windfall tax on the energy companies")
    assert interpreter.last_source == "stored" and record.actions == [TAX]
    assert forecaster.last_source == "stored"
    assert {c.narrative for c in record.candidates} == claude_narratives
    assert all(c.scores["reused"] == 1.0 for c in record.candidates)
    assert sum(c.probability for c in record.candidates) == pytest.approx(1.0)
    assert replay(world, game.config, game.history) == game.state

    # Something Claude never saw falls back to keyword matching and the stub forecaster.
    game.play_turn("build more houses")
    assert interpreter.last_source == "fallback" and forecaster.last_source == "fallback"


def test_action_signature_ignores_size() -> None:
    bigger = TAX.model_copy(update={"magnitude": 0.9})
    cut = TAX.model_copy(update={"magnitude": -0.5})
    assert action_signature([TAX]) == action_signature([bigger]) != action_signature([cut])
    assert action_signature([]) == "none"


def test_guess_category_reads_text_then_nodes() -> None:
    scenario = GeneratedScenario(source="generated", **SCENARIO.model_dump())
    assert guess_category(scenario) == "energy"  # its own category
    scenario = scenario.model_copy(update={"category": None})
    nurses = scenario.model_copy(update={"title": "Nurses walk out", "briefing": "NHS pay."})
    assert guess_category(nurses) == "health"
    quiet = scenario.model_copy(update={"title": "A quiet month", "briefing": "Little news."})
    assert guess_category(quiet) == "energy"  # from the affected nodes


# --- moving knowledge ------------------------------------------------------------


def test_export_import_and_library_export(world, engine, tmp_path) -> None:
    store = KnowledgeStore(tmp_path / "game.db")
    play_claude_game(world, engine, store, "g1")
    n = store.export_jsonl(tmp_path / "k.jsonl")
    assert n == len(store)

    fresh = KnowledgeStore(tmp_path / "fresh.db")
    assert fresh.import_jsonl(tmp_path / "k.jsonl") == n
    assert fresh.stats() == store.stats()
    assert fresh.outcomes() == store.outcomes()

    assert store.export_scenarios(tmp_path / "s.jsonl") == 1
    library = ScenarioLibrary.load(tmp_path / "s.jsonl", builtin=False)
    assert library.scenarios == store.library_scenarios()
    assert library.check(world) == []


def test_rows_from_an_old_schema_are_skipped(tmp_path) -> None:
    store = KnowledgeStore(tmp_path / "game.db")
    with sqlite3.connect(tmp_path / "game.db") as conn:
        conn.execute(
            "INSERT INTO knowledge (kind, key, payload) VALUES ('outcome', 'x', ?)",
            (json.dumps({"something": "else"}),),
        )
    assert store.outcomes() == [] and store.stats()["outcome"] == 1


def test_knowledge_cli(world, engine, tmp_path, capsys) -> None:
    db = tmp_path / "game.db"
    store = KnowledgeStore(db)
    play_claude_game(world, engine, store, "g1")
    store.close()
    main(["knowledge", "--db", str(db), "stats"])
    out = capsys.readouterr().out
    assert "scenario" in out and "3 graph changes were applied" in out
    main(["knowledge", "--db", str(db), "export-scenarios", str(tmp_path / "s.jsonl")])
    assert "Wrote 1 scenarios" in capsys.readouterr().out
