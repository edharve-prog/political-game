"""The fuller electorate: every voter counted and turnout that moves (O8, M5)."""

import pytest

from hog_sim.population.popularity import (
    MAX_TURNOUT,
    effective_turnout,
    vote_intention,
)
from hog_sim.world.seed.toy import toy_world


def test_groups_cover_the_whole_population_and_pensioners_no_longer_dominate() -> None:
    world = toy_world()
    assert sum(g.population_share for g in world.groups.values()) == pytest.approx(1.0)
    weights = {gid: g.population_share * g.turnout for gid, g in world.groups.items()}
    assert weights["group:pensioners"] / sum(weights.values()) < 0.25


def test_enthusiasm_and_anger_both_raise_turnout() -> None:
    group = toy_world().groups["group:pensioners"]
    calm = effective_turnout(group)
    assert calm == group.turnout
    happy = group.model_copy(update={"approval": group.lean + 0.1})
    angry = group.model_copy(update={"approval": group.lean - 0.1})
    assert effective_turnout(happy) == pytest.approx(effective_turnout(angry))
    assert effective_turnout(happy) > calm
    assert effective_turnout(group.model_copy(update={"approval": 0.0, "lean": 1.0})) == (
        MAX_TURNOUT
    )


def test_an_angry_group_weighs_more_at_the_ballot() -> None:
    world = toy_world()
    before = vote_intention(world)
    world.groups["group:young_renters"].approval -= 0.1
    fixed = world.model_copy(deep=True)
    fixed.groups["group:young_renters"].lean -= 0.1  # same approval, no anger
    assert vote_intention(world) < vote_intention(fixed) < before
