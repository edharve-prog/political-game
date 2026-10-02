import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hog_sim.core.models import PolicyAction
from hog_sim.core.state import WorldState
from hog_sim.llm import (
    AnthropicClient,
    FakeClient,
    GeneratedScenario,
    Interpretation,
    LLMOutputError,
    LLMRefusal,
    RecordingClient,
    generate_scenario,
    interpret,
    structured_call,
    summarise_state,
)
from hog_sim.llm.client import CassetteMiss, LLMRequest, Message
from hog_sim.llm.eval_cases import CASES, score
from hog_sim.llm.evaluate import DEFAULT_CASSETTE, run
from hog_sim.llm.scenario_gen import ScenarioDraft
from hog_sim.world.seed.toy import toy_world


@pytest.fixture
def world() -> WorldState:
    return toy_world()


@pytest.fixture
def summary(world: WorldState):
    return summarise_state(world, recent_events=["Energy price cap review announced"])


def draft(**overrides) -> dict:
    data = {
        "title": "Energy firms warn of winter blackouts",
        "category": "energy",
        "storyline": "new",
        "briefing": "Three suppliers say they cannot buy enough gas for winter.",
        "affected_nodes": ["sector:energy", "indicator:energy_prices", "group:pensioners"],
        "urgency": 0.8,
        "suggested_options": ["Cap bills", "Bail out suppliers", "Do nothing"],
        "stakeholder_positions": [
            {"node": "group:pensioners", "stance": 0.9, "statement": "Protect our bills."},
            {"node": "sector:energy", "stance": 0.6, "statement": "We need support."},
        ],
    }
    return data | overrides


def action(**overrides) -> dict:
    data = {"kind": "tax", "target": "sector:energy", "magnitude": 0.3, "rationale": "windfall"}
    return data | overrides


# --- structured_call -------------------------------------------------------


def call(client, check=None, max_attempts=3):
    return structured_call(
        client,
        output_type=ScenarioDraft,
        system="sys",
        prompt="write",
        model="m",
        prompt_version="v1",
        check=check,
        max_attempts=max_attempts,
    )


def test_structured_call_returns_validated_model() -> None:
    client = FakeClient([draft()])
    result = call(client)
    assert isinstance(result, ScenarioDraft)
    assert len(client.requests) == 1
    assert client.requests[0].schema_name == "ScenarioDraft"


def test_invalid_output_is_retried_with_feedback() -> None:
    client = FakeClient(["not json", draft(urgency=3), draft()])
    result = call(client)
    assert result.urgency == 0.8
    assert len(client.requests) == 3
    last = client.requests[-1].messages
    assert [m.role for m in last] == ["user", "assistant", "user", "assistant", "user"]
    assert "urgency" in last[-1].content


def test_semantic_check_failures_are_retried() -> None:
    client = FakeClient([draft(title="x"), draft()])
    result = call(client, check=lambda d: ["bad title"] if d.title == "x" else [])
    assert result.title != "x"
    assert "bad title" in client.requests[1].messages[-1].content


def test_gives_up_after_max_attempts() -> None:
    client = FakeClient(["{}"] * 5)
    with pytest.raises(LLMOutputError) as exc:
        call(client, max_attempts=3)
    assert len(exc.value.errors) == 3
    assert len(client.requests) == 3


def test_usage_is_logged(summary) -> None:
    client = FakeClient([draft()])
    generate_scenario(summary, client)
    record = client.usage.records[0]
    assert record.schema_name == "ScenarioDraft"
    assert record.prompt_version == "scenario-3"


# --- Recording -------------------------------------------------------------


def test_recording_client_records_then_replays(tmp_path: Path, summary) -> None:
    cassette = tmp_path / "c.json"
    live = FakeClient([draft()])
    first = generate_scenario(summary, RecordingClient(cassette, inner=live))
    replay = RecordingClient(cassette)
    second = generate_scenario(summary, replay)
    assert first == second
    assert replay.usage.records[0].cached
    assert replay.usage.total_cost_usd == 0


def test_replay_miss_raises(tmp_path: Path, summary) -> None:
    with pytest.raises(CassetteMiss):
        generate_scenario(summary, RecordingClient(tmp_path / "empty.json"))


def test_cache_key_ignores_wire_schema_but_not_prompt() -> None:
    base = dict(
        model="m",
        system="s",
        messages=[Message(role="user", content="hi")],
        schema_name="X",
        schema_fingerprint="abc",
        prompt_version="v1",
    )
    a = LLMRequest(output_schema={"type": "object"}, **base)
    b = LLMRequest(output_schema={"type": "object", "title": "X"}, **base)
    c = LLMRequest(output_schema={"type": "object"}, **(base | {"prompt_version": "v2"}))
    assert a.cache_key() == b.cache_key()
    assert a.cache_key() != c.cache_key()


# --- Anthropic client (SDK stubbed) ----------------------------------------


class StubMessages:
    def __init__(self, response) -> None:
        self.response = response
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def stub_sdk(text: str = "{}", stop_reason: str = "end_turn"):
    response = SimpleNamespace(
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
    )
    messages = StubMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def test_anthropic_client_sends_structured_output_request(summary) -> None:
    sdk, messages = stub_sdk(json.dumps(draft()))
    client = AnthropicClient(sdk)
    scenario = generate_scenario(summary, client)
    assert scenario.title.startswith("Energy")
    kwargs = messages.kwargs
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert kwargs["output_config"]["effort"] == "medium"
    assert kwargs["fallbacks"] == "default"
    assert "thinking" not in kwargs  # adaptive by default; disabling is a 400
    assert client.usage.total_cost_usd == pytest.approx((1000 * 4 + 200 * 20) / 1e6)


def test_anthropic_client_raises_on_refusal(summary) -> None:
    sdk, _ = stub_sdk(stop_reason="refusal")
    with pytest.raises(LLMRefusal):
        generate_scenario(summary, AnthropicClient(sdk))


# --- Summary ---------------------------------------------------------------


def test_summary_flags_stress_anger_and_tension(world: WorldState) -> None:
    world.indicators["indicator:inflation"].history = [2.0, 2.5, 3.0]
    world.groups["group:young_renters"].approval = 0.3
    s = summarise_state(world)
    assert s.stressed_indicators == ["indicator:inflation"]
    assert s.angry_groups == ["group:young_renters"]
    assert s.foreign_tensions == ["country:china"]
    text = s.to_prompt()
    assert "[STRESSED]" in text and "[ANGRY]" in text and "[TENSE]" in text
    assert set(s.catalogue) == set(world.node_ids())


def test_summary_keeps_recent_events_only(world: WorldState) -> None:
    s = summarise_state(world, recent_events=[f"e{i}" for i in range(20)], max_events=3)
    assert s.recent_events == ["e17", "e18", "e19"]


# --- Scenario generator ----------------------------------------------------


def test_generate_scenario(summary) -> None:
    client = FakeClient([draft()])
    scenario = generate_scenario(summary, client)
    assert isinstance(scenario, GeneratedScenario)
    assert scenario.source == "generated"
    assert scenario.stakeholder_positions[0].node == "group:pensioners"
    prompt = client.requests[0].messages[0].content
    assert "indicator:energy_prices" in prompt
    assert "Energy price cap review announced" in prompt


def test_generate_scenario_rejects_unknown_nodes(summary) -> None:
    client = FakeClient([draft(affected_nodes=["sector:fishing"]), draft()])
    scenario = generate_scenario(summary, client)
    assert scenario.affected_nodes[0] == "sector:energy"
    assert "sector:fishing" in client.requests[1].messages[-1].content


# --- Interpreter -----------------------------------------------------------


def test_interpret_maps_text_to_actions(summary) -> None:
    client = FakeClient([{"actions": [action()]}])
    result = interpret("Windfall tax on energy firms", summary, client)
    assert result.actions == [PolicyAction(**action())]
    assert '"""Windfall tax on energy firms"""' in client.requests[0].messages[0].content


def test_interpret_includes_scenario(summary) -> None:
    scenario = GeneratedScenario(**draft())
    client = FakeClient([{"actions": [action()]}])
    interpret("Tax them", summary, client, scenario=scenario)
    assert "Energy firms warn of winter blackouts" in client.requests[0].messages[0].content


@pytest.mark.parametrize(
    "bad",
    [
        {"actions": [action(target="sector:fishing")]},
        {"actions": [action(kind="do_nothing", target="sector:energy")]},
        {"actions": [action(kind="regulate", magnitude=-0.5)]},
        {"actions": []},
        {"actions": [action()], "clarifying_question": "Which firms?"},
    ],
)
def test_interpret_retries_bad_interpretations(summary, bad) -> None:
    client = FakeClient([bad, {"actions": [action()]}])
    result = interpret("Windfall tax", summary, client)
    assert len(client.requests) == 2
    assert result.actions[0].target == "sector:energy"


def test_interpret_can_ask_a_question(summary) -> None:
    client = FakeClient([{"actions": [], "clarifying_question": "Sort out what?"}])
    result = interpret("Sort it out.", summary, client)
    assert result == Interpretation(clarifying_question="Sort out what?")


# --- Eval cases ------------------------------------------------------------


def test_there_are_twenty_varied_cases(world: WorldState) -> None:
    assert len(CASES) == 20
    ids = set(world.node_ids())
    for case in CASES:
        for exp in case.expect:
            assert exp.targets <= ids, case.text
    kinds = {k for case in CASES for exp in case.expect for k in exp.kinds}
    assert {"tax", "spend", "regulate", "diplomatic", "military", "communicate"} <= kinds


def test_score_checks_kind_target_and_sign() -> None:
    case = CASES[0]  # windfall tax on energy
    good = PolicyAction(**action())
    assert score(case, [good], None) == []
    assert score(case, [good.model_copy(update={"magnitude": -0.3})], None)
    assert score(case, [good.model_copy(update={"target": "sector:finance"})], None)


def test_eval_harness_runs_end_to_end() -> None:
    def oracle(request: LLMRequest) -> dict:
        text = request.messages[0].content.split('"""')[1]
        case = next(c for c in CASES if c.text == text)
        actions = [
            action(
                kind=sorted(e.kinds)[0], target=sorted(e.targets)[0], magnitude=0.4 * (e.sign or 1)
            )
            for e in case.expect
        ]
        question = "Sort out what?" if case.expects_question else None
        return {"actions": actions, "clarifying_question": question}

    results = run(FakeClient(responder=oracle))
    assert all(not problems for _, problems in results)


@pytest.mark.skipif(not DEFAULT_CASSETTE.exists(), reason="no recorded live run yet")
def test_recorded_live_interpretations_pass() -> None:
    results = run(RecordingClient(DEFAULT_CASSETTE))
    failures = [(text, p) for text, p in results if p]
    assert len(failures) <= 2, failures


@pytest.mark.parametrize(
    "bad",
    [
        draft(affected_nodes=[]),
        draft(suggested_options=["Only one"]),
        draft(stakeholder_positions=[]),
    ],
)
def test_generate_scenario_checks_list_sizes(summary, bad) -> None:
    client = FakeClient([bad, draft()])
    generate_scenario(summary, client)
    assert len(client.requests) == 2


# --- Game loop adapters ----------------------------------------------------


def test_scenario_source_feeds_history_into_prompt(world: WorldState) -> None:
    from hog_sim.core.models import Outcome
    from hog_sim.llm.adapters import LLMScenarioSource

    past = SimpleNamespace(
        turn=3,
        scenario=GeneratedScenario(**draft()),
        outcome=Outcome(narrative="Bills capped; suppliers bailed out", probability=0.6),
    )
    client = FakeClient([draft()])
    scenario = LLMScenarioSource(client).next_scenario(world, [past])
    assert scenario.source == "generated"
    assert "Bills capped; suppliers bailed out" in client.requests[0].messages[0].content


def test_interpreter_adapter_drops_infeasible_actions(world: WorldState) -> None:
    from hog_sim.llm.adapters import LLMInterpreter

    world.institutions["institution:legislature"].support = 0.3
    reply = {
        "actions": [
            action(),
            action(kind="diplomatic", target="country:eu", magnitude=0.4),
        ]
    }
    adapter = LLMInterpreter(FakeClient([reply]))
    actions = adapter.interpret(
        "Tax energy, warm up to the EU", world, GeneratedScenario(**draft())
    )
    assert [a.kind for a in actions] == ["diplomatic"]
    assert not adapter.last_feasibility.checks[0].feasible


def test_adapters_drive_the_game_loop(world: WorldState) -> None:
    from hog_sim.core.config import GameConfig
    from hog_sim.game.loop import Game
    from hog_sim.game.stubs import EngineForecaster
    from hog_sim.llm.adapters import LLMInterpreter, LLMScenarioSource

    def responder(request: LLMRequest) -> dict:
        if request.schema_name == "ScenarioDraft":
            return draft()
        return {"actions": [action()]}

    client = FakeClient(responder=responder)
    game = Game(
        GameConfig(), world, LLMScenarioSource(client), LLMInterpreter(client), EngineForecaster()
    )
    for _ in range(2):
        record = game.play_turn("Windfall tax on energy firms")
        assert [a.kind for a in record.actions] == ["tax"]
    assert game.state.turn == 2


def test_scenario_prompt_lists_recent_scenarios_and_keeps_category(world: WorldState) -> None:
    from hog_sim.core.config import GameConfig
    from hog_sim.game.loop import Game
    from hog_sim.game.stubs import EngineForecaster
    from hog_sim.llm.adapters import LLMInterpreter, LLMScenarioSource

    def responder(request: LLMRequest) -> dict:
        if request.schema_name == "ScenarioDraft":
            return draft()
        return {"actions": [action()]}

    client = FakeClient(responder=responder)
    game = Game(
        GameConfig(), world, LLMScenarioSource(client), LLMInterpreter(client), EngineForecaster()
    )
    game.play_turn("Windfall tax on energy firms")
    assert game.history[0].scenario.category == "energy"
    prompts = [r.messages[0].content for r in client.requests if r.schema_name == "ScenarioDraft"]
    assert "Recent scenarios" not in prompts[0]
    assert "- [energy] Energy firms warn of winter blackouts" in prompts[-1]
