"""A rebellion costs the Commons majority for a while, not for the rest of the game (EB-14)."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import GraphChange, Outcome
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


def test_support_recovers_a_majority_after_two_rebellions() -> None:
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
