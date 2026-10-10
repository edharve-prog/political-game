import pytest

from hog_sim.core.models import ApprovalEvent, PolicyAction
from hog_sim.population.popularity import (
    HABIT_HALF_LIFE,
    _event_weight,
    national_approval,
    policy_events,
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
            update={"groups": state.groups, "events": state.events, "baselines": state.baselines}
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


def test_stacked_events_are_capped() -> None:
    """However many boosts pile up, together they move a target by at most EVENT_CAP (EB-6)."""
    from hog_sim.core.models import ApprovalEvent
    from hog_sim.population.popularity import EVENT_CAP, target_approval

    world = toy_world()
    lean = world.groups["group:pensioners"].lean
    world.events = [
        ApprovalEvent(name=f"boost {i}", group_effects={"group:pensioners": 0.1}) for i in range(6)
    ]
    assert target_approval(world, toy_world())["group:pensioners"] == pytest.approx(
        lean + EVENT_CAP
    )


@pytest.mark.parametrize("turns", [1, 3, 12])
def test_a_policy_is_felt_in_full_for_exactly_its_duration(turns) -> None:
    """Finding 6: a 12-turn policy used to get 13 full-strength approval updates."""
    action = PolicyAction(
        kind="spend", target="group:pensioners", magnitude=0.5, duration_turns=turns
    )
    (event,) = policy_events([action])
    weights = [
        _event_weight(age, event.half_life_turns, event.hold_turns) for age in range(turns + 2)
    ]
    assert weights[:turns] == [1.0] * turns
    assert weights[turns] < 1.0


def test_voters_get_used_to_a_lasting_change(world) -> None:
    """EB-11: a lasting price rise hurts most when new and fades towards the new normal."""
    world.indicators["indicator:inflation"].value += 2
    state, gaps = world, []
    for _ in range(4 * HABIT_HALF_LIFE):
        state = step_approval(state, reference=toy_world())
        gaps.append(
            state.groups["group:pensioners"].approval - world.groups["group:pensioners"].lean
        )
    worst = min(gaps)
    assert worst < -0.02
    # After one half-life most of the hit is still felt; after four it has nearly gone.
    assert gaps[HABIT_HALF_LIFE] < worst / 3
    assert abs(gaps[-1]) < abs(worst) / 8
    base = state.baselines["indicator:inflation"]
    assert world.indicators["indicator:inflation"].value - base < 0.2


def test_baselines_only_track_what_groups_feel(world) -> None:
    new = step_approval(world, reference=world)
    assert "indicator:inflation" in new.baselines
    assert all(n.startswith(("indicator:", "sector:")) for n in new.baselines)
    assert new.baselines["indicator:inflation"] == world.indicators["indicator:inflation"].value
