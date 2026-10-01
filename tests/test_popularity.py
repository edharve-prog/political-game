import pytest

from hog_sim.core.models import ApprovalEvent, PolicyAction
from hog_sim.population.popularity import (
    national_approval,
    run_election,
    seat_share,
    step_approval,
    target_approval,
    vote_intention,
)
from hog_sim.world.propagation import actions_to_shocks, apply_deltas, propagate
from hog_sim.world.seed.toy import toy_world


@pytest.fixture
def world():
    return toy_world()


def play(world, actions, turns):
    """Apply the engine's expected trajectory turn by turn and update approval."""
    dist = propagate(world, actions_to_shocks(actions), horizon=turns, k_draws=50)
    state = world
    history = []
    for t in range(turns):
        # dist.at(t) is the cumulative change since the start, so apply it to the start state
        state = apply_deltas(world, dist.at(t)).model_copy(
            update={"groups": state.groups, "events": state.events}
        )
        state = step_approval(state, reference=world)
        history.append(state)
    return history


def test_no_action_is_stable(world) -> None:
    history = play(world, [], 24)
    for gid, group in world.groups.items():
        series = [s.groups[gid].approval for s in history]
        assert abs(series[-1] - series[-2]) < 1e-3
        assert abs(series[-1] - group.lean) < 0.01


def test_energy_tax_hurts_the_expected_groups(world) -> None:
    baseline = play(world, [], 12)[-1]
    taxed = play(world, [PolicyAction(kind="tax", target="sector:energy", magnitude=0.8)], 12)[-1]
    for gid in ("group:pensioners", "group:public_workers", "group:business"):
        assert taxed.groups[gid].approval < baseline.groups[gid].approval
    assert national_approval(taxed) < national_approval(baseline)


def test_approval_adjusts_gradually(world) -> None:
    world.indicators["indicator:inflation"].value += 2  # sudden +2pp
    target = target_approval(world, reference=toy_world())["group:pensioners"]
    after = step_approval(world, reference=toy_world()).groups["group:pensioners"].approval
    assert target < after < 0.55


def test_events_fade(world) -> None:
    world.events.append(
        ApprovalEvent(name="Scandal", group_effects={"group:business": -0.2}, half_life_turns=1)
    )
    state = world
    drops = []
    for _ in range(15):
        state = step_approval(state, reference=world)
        drops.append(0.5 - state.groups["group:business"].approval)
    assert drops[1] > 0
    assert drops[-1] < drops[2]
    assert state.events == []


def test_vote_intention_weights_turnout(world) -> None:
    # Pensioners turn out more than young renters, so they pull vote above plain approval
    assert vote_intention(world) > national_approval(world)


def test_seat_model() -> None:
    assert seat_share(0.5) == pytest.approx(0.5)
    assert seat_share(0.52) > 0.55
    assert seat_share(0.48) < 0.45


def test_election(world) -> None:
    result = run_election(world)
    assert result.total_seats == 650
    assert result.majority == (result.vote_share > 0.5)


def test_high_deficit_costs_every_group(world) -> None:
    from hog_sim.population.popularity import DEFICIT_TOLERANCE

    base = target_approval(world, reference=toy_world())
    world.indicators["indicator:deficit"].value = DEFICIT_TOLERANCE  # at tolerance: no penalty
    at = target_approval(world, reference=toy_world())
    world.indicators["indicator:deficit"].value = DEFICIT_TOLERANCE + 5
    above = target_approval(world, reference=toy_world())
    for gid in world.groups:
        assert above[gid] < at[gid]
    assert at["group:young_renters"] == base["group:young_renters"]  # renters ignore deficit
