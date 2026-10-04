"""RB-9: slips with one obvious fix are repaired instead of costing a retry."""

import pytest

from hog_sim.llm.client import FakeClient
from hog_sim.llm.interpreter import interpret
from hog_sim.llm.summary import resolve_id, summarise_state
from hog_sim.world.seed.toy import toy_world


@pytest.fixture
def world():
    return toy_world()


@pytest.fixture
def summary(world):
    return summarise_state(world)


def tax(target="sector:energy", kind="tax"):
    return {"kind": kind, "target": target, "magnitude": 0.3, "rationale": "r"}


def test_resolve_id_only_maps_a_unique_bare_id() -> None:
    ids = ["group:business", "sector:energy", "indicator:energy", "group:pensioners"]
    assert resolve_id("business", ids) == "group:business"
    assert resolve_id("group:business", ids) == "group:business"
    assert resolve_id("energy", ids) == "energy"
    assert resolve_id("fishing", ids) == "fishing"
    assert resolve_id("group:fishing", ids) == "group:fishing"


def test_bare_ids_in_an_interpretation_are_prefixed(summary) -> None:
    reply = {"actions": [tax(target="energy")], "delivery": {"consulted": ["business"]}}
    client = FakeClient([reply])
    result = interpret("Windfall tax", summary, client)
    assert len(client.requests) == 1
    assert result.actions[0].target == "sector:energy"
    assert result.delivery.consulted == ["group:business"]


def test_question_beside_actions_is_dropped_not_retried(summary) -> None:
    client = FakeClient([{"actions": [tax()], "clarifying_question": "Which firms?"}])
    result = interpret("Windfall tax", summary, client)
    assert len(client.requests) == 1
    assert result.clarifying_question is None and len(result.actions) == 1


def test_do_nothing_beside_real_actions_is_dropped(summary) -> None:
    reply = {"actions": [tax(), tax(target="country:uk", kind="do_nothing")]}
    client = FakeClient([reply])
    result = interpret("Tax them, otherwise sit tight", summary, client)
    assert len(client.requests) == 1
    assert [a.kind for a in result.actions] == ["tax"]


def test_do_nothing_alone_is_kept(summary) -> None:
    client = FakeClient([{"actions": [tax(target="country:uk", kind="do_nothing")]}])
    result = interpret("Wait and see", summary, client)
    assert [a.kind for a in result.actions] == ["do_nothing"]
