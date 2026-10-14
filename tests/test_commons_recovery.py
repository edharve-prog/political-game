"""A rebellion costs the Commons majority for a while, not for the rest of the game (EB-14)."""

import pytest

import hog_sim.world.opposition as opposition
from hog_sim.core.config import GameConfig
from hog_sim.core.models import GraphChange, Outcome, PolicyAction, Scenario
from hog_sim.game.loop import INSTITUTION_SETTLE, resolve
from hog_sim.game.stubs import CannedScenarios
from hog_sim.world.seed.toy import toy_world

QUIET = Outcome(narrative="A quiet month.", probability=1.0)
# As in the review: the outcome writer also docks the Commons' support directly.
REBELLION = Outcome(
    narrative="Backbenchers rebel.",
    probability=1.0,
    events=["backbench_rebellion"],
    graph_changes=[
        GraphChange(kind="node_attr", node="institution:legislature", attr="support", delta=-0.15)
    ],
)


def support(state) -> float:
    return state.institutions["institution:legislature"].support


def play(outcomes: list[Outcome]) -> list[float]:
    start = toy_world()
    state, config = start, GameConfig(k_draws=10)
    scenario = CannedScenarios().next_scenario(start, [])
    seen = []
    for outcome in outcomes:
        state = resolve(start, state, scenario, [], outcome, config)
        seen.append(support(state))
    return seen


def test_support_recovers_a_majority_after_two_rebellions(monkeypatch) -> None:
    # By-elections are lost while the polls are down after a rebellion; this test is about
    # how the Commons settles, so it holds them off (O9).
    monkeypatch.setattr(opposition, "BYELECTION_CHANCE", 0.0)
    seen = play([REBELLION, REBELLION, *[QUIET] * 10])
    assert min(seen) < 0.5  # the rebellions do cost the majority
    lost = next(i for i, s in enumerate(seen) if s < 0.5)
    back = next(i for i, s in enumerate(seen) if i > lost and s >= 0.5)
    assert back - lost <= 6
    assert abs(seen[-1] - support(toy_world())) < 0.05


def test_settling_closes_a_fixed_share_of_the_gap() -> None:
    start = toy_world()
    state = start.snapshot()
    state.institutions["institution:legislature"].support = 0.2
    after = resolve(
        start, state, CannedScenarios().next_scenario(start, []), [], QUIET, GameConfig()
    )
    gap = support(start) - 0.2
    assert support(after) - 0.2 == pytest.approx(gap * INSTITUTION_SETTLE)


# E7: institutions revert once. A whip or a rebellion fades back to the quiet path; it never
# overshoots it, which it did when the engine faded the push and settling took the same
# ground back a second time.
LEGISLATURE = "institution:legislature"
QUIET_SCENARIO = Scenario(
    title="A quiet month", briefing="Nothing much.", affected_nodes=[], urgency=0.1
)


def path(actions: list[PolicyAction], first: Outcome = QUIET, turns: int = 20) -> list[float]:
    start = toy_world()
    state, config = start, GameConfig(k_draws=10)
    seen = []
    for turn in range(turns):
        state = resolve(
            start,
            state,
            QUIET_SCENARIO,
            actions if turn == 0 else [],
            first if turn == 0 else QUIET,
            config,
        )
        seen.append(support(state))
    return seen


@pytest.mark.parametrize("magnitude", [0.3, 0.8])
@pytest.mark.parametrize("turns", [1, 4, 12])
def test_a_whip_never_leaves_support_below_the_quiet_path(magnitude: float, turns: int) -> None:
    whip = PolicyAction(
        kind="communicate", target=LEGISLATURE, magnitude=magnitude, duration_turns=turns
    )
    quiet, whipped = path([]), path([whip])
    assert all(w >= q - 1e-9 for w, q in zip(whipped, quiet, strict=True))
    assert whipped[0] > quiet[0]
    assert whipped[-1] - quiet[-1] < 0.01  # and it does fade


@pytest.mark.parametrize("events", [["backbench_rebellion"], ["protest"]])
def test_a_rebellion_never_leaves_support_above_the_quiet_path(events: list[str]) -> None:
    rebellion = Outcome(narrative="Backbenchers rebel.", probability=1.0, events=events)
    quiet, rebelled = path([]), path([], rebellion)
    assert all(r <= q + 1e-9 for r, q in zip(rebelled, quiet, strict=True))
    assert rebelled[0] < quiet[0]
    assert quiet[-1] - rebelled[-1] < 0.01


def test_a_direct_cut_to_support_never_recovers_past_the_quiet_path() -> None:
    quiet, rebelled = path([]), path([], REBELLION)
    assert all(r <= q + 1e-9 for r, q in zip(rebelled, quiet, strict=True))
