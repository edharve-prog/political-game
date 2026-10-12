"""An outcome's graph changes apply in full or not at all (Ed's review, 2026-10)."""

from hog_sim.core.config import GameConfig
from hog_sim.core.models import EdgeKind, GraphChange, Outcome, PolicyAction, Scenario
from hog_sim.game.loop import resolve
from hog_sim.world.changes import (
    MAX_CHANGES,
    apply_graph_changes,
    max_weight_step,
    validate_graph_changes,
)
from hog_sim.world.seed.toy import toy_world

EU, UK = "country:eu", "country:uk"


def eu_trade(state):
    return next(
        e for e in state.edges if (e.source, e.target, e.kind) == (EU, UK, EdgeKind.TRADES_WITH)
    )


def row_and_tariffs(state) -> list[GraphChange]:
    """A trade row with the EU, then tariffs cutting EU->UK trade by as much as allowed today.

    The row scales trade down first (TRADE_PER_RELATIONSHIP), which shrinks how far the
    edge may then move, so the second change no longer fits once the first has landed."""
    weight = eu_trade(state).weight
    return [
        GraphChange(kind="node_attr", node=EU, attr="relationship", delta=-0.2, reason="row"),
        GraphChange(
            kind="edge_weight",
            source=EU,
            target=UK,
            edge_kind=EdgeKind.TRADES_WITH,
            delta=-max_weight_step(weight),
            reason="tariffs",
        ),
    ]


def test_validation_checks_each_change_after_the_earlier_ones() -> None:
    world = toy_world()
    problems = validate_graph_changes(world, row_and_tariffs(world))
    assert len(problems) == 1 and problems[0].startswith("graph_changes[1]")


def test_a_set_that_does_not_fit_applies_none_of_it() -> None:
    """It used to apply the row and silently skip the tariffs."""
    world = toy_world()
    after = apply_graph_changes(world, row_and_tariffs(world))
    assert after.countries[EU].relationship == world.countries[EU].relationship
    assert eu_trade(after).weight == eu_trade(world).weight


def test_a_set_that_validates_applies_in_full() -> None:
    world = toy_world()
    changes = row_and_tariffs(world)
    changes[1] = changes[1].model_copy(update={"delta": -0.05})
    assert validate_graph_changes(world, changes) == []
    after = apply_graph_changes(world, changes)
    assert after.countries[EU].relationship < world.countries[EU].relationship
    scaled = eu_trade(apply_graph_changes(world, changes[:1])).weight
    assert abs(eu_trade(after).weight - (scaled - 0.05)) < 1e-12


def test_too_many_changes_apply_none() -> None:
    world = toy_world()
    changes = [
        GraphChange(kind="node_attr", node=s, attr="sentiment", delta=-0.01, reason="x")
        for s in world.sectors
    ][: MAX_CHANGES + 1]
    assert len(changes) == MAX_CHANGES + 1
    assert apply_graph_changes(world, changes) == world


def test_engine_changes_still_apply_one_by_one() -> None:
    world = toy_world()
    changes = row_and_tariffs(world)
    after = apply_graph_changes(world, changes, trusted=True)
    assert after.countries[EU].relationship < world.countries[EU].relationship


def test_resolve_applies_the_outcome_whole() -> None:
    world = toy_world()
    scenario = Scenario(title="t", briefing="b", affected_nodes=[], urgency=0)
    outcome = Outcome(narrative="x", probability=1, graph_changes=row_and_tariffs(world))
    after = resolve(world, world, scenario, [], outcome, GameConfig())
    assert after.countries[EU].relationship == world.countries[EU].relationship


def test_candidates_are_checked_after_the_players_own_moves() -> None:
    """A diplomatic row by the player shrinks EU trade before the outcome's changes land."""
    from hog_sim.world.changes import action_changes

    world = toy_world()
    row = PolicyAction(kind="diplomatic", target=EU, magnitude=-1.0)
    staged = apply_graph_changes(world, action_changes(world, [row]), trusted=True)
    tariffs = row_and_tariffs(world)[1:]
    assert validate_graph_changes(world, tariffs) == []
    assert validate_graph_changes(staged, tariffs) != []
