"""The player's own country takes only a speech, and no action sets its growth (review of
13 October 2026, E3 and property test 3).

Aimed at "the country", tax, spending and deregulation used to push UK growth two steps a
unit; growth then cut the deficit and swelled the debt ratio's denominator, so spending on
the UK left debt lower than doing nothing.
"""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Outcome, PolicyAction, Scenario
from hog_sim.game.loop import resolve
from hog_sim.policy.feasibility import check_feasibility
from hog_sim.policy.limits import constrain
from hog_sim.world.propagation import actions_to_shocks
from hog_sim.world.seed.toy import toy_world

UK = "country:uk"
QUIET = Scenario(title="Quiet month", briefing="", affected_nodes=[], urgency=0)
NOTHING = Outcome(narrative="Nothing happens", probability=1)


def act(kind: str, magnitude: float, turns: int = 24) -> PolicyAction:
    return PolicyAction(kind=kind, target=UK, magnitude=magnitude, duration_turns=turns)


def play(actions: list[PolicyAction], turns: int = 24):
    """The world after ``turns`` quiet turns, with ``actions`` put through this turn's
    limits on turn 0, as the game does."""
    start = state = toy_world()
    for t in range(turns):
        applied = constrain(actions, state, []).actions if t == 0 else []
        state = resolve(start, state, QUIET, applied, NOTHING, GameConfig())
    return state


@pytest.mark.parametrize("kind", ["tax", "spend", "regulate", "deregulate", "legislate"])
def test_only_a_speech_can_be_aimed_at_the_country(kind) -> None:
    check = check_feasibility([act(kind, 0.5)], toy_world()).checks[0]
    assert not check.feasible
    assert any("as a whole" in b for b in check.blockers)
    assert check_feasibility([act("communicate", 0.5)], toy_world()).checks[0].feasible


def test_no_action_pushes_the_players_growth() -> None:
    world = toy_world()
    for kind in ("communicate", "tax", "spend", "deregulate"):
        assert not [s for s in actions_to_shocks([act(kind, 1.0)], world) if s.node == UK]


@pytest.mark.parametrize(
    "action",
    [act("spend", 1.0), act("tax", -1.0), act("deregulate", 1.0), act("communicate", 1.0)],
    ids=["spend", "tax cut", "deregulate", "speech"],
)
def test_loosening_aimed_at_the_country_never_lowers_debt(action) -> None:
    """Property test 3: spending (or any loosening) aimed at the player's own country never
    leaves debt lower than doing nothing."""
    idle = play([])
    after = play([action])
    assert after.countries[UK].debt_pct >= idle.countries[UK].debt_pct - 1e-9
    assert after.countries[UK].growth_pct == pytest.approx(idle.countries[UK].growth_pct)
