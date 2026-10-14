"""A tax rise pays for borrowing only while it lasts (review of 13 October 2026, E5 and
property test 7).

The markets' test, feasibility past the deficit limit and the deficit ceiling all compared
costs a turn: a one-turn bank tax rise read as paying for a tax cut that ran for two years.
"""

import pytest

from hog_sim.core.models import PolicyAction
from hog_sim.policy.feasibility import check_feasibility
from hog_sim.policy.limits import fit_deficit
from hog_sim.world.propagation import (
    MIN_FISCAL_TURNS,
    fiscal_cost,
    fiscal_turns,
    unfunded_tax_cut,
)
from hog_sim.world.seed.toy import toy_world

DEFICIT = "indicator:deficit"


def act(kind: str, target: str, magnitude: float, turns: int = 1) -> PolicyAction:
    return PolicyAction(kind=kind, target=target, magnitude=magnitude, duration_turns=turns)


def cut(turns: int) -> PolicyAction:
    return act("tax", "group:business", -0.8, turns)


def rise(turns: int) -> PolicyAction:
    # Half of a bank tax comes out of profits, not output, so it raises twice the output it
    # removes (O3's SECTOR_PASS_THROUGH): 0.5 raises what 1.0 used to.
    return act("tax", "sector:finance", 0.5, turns)


def test_a_one_turn_rise_does_not_fund_a_two_year_cut() -> None:
    """The review's case: the rise a turn nearly matches the cut a turn, but runs for 6 turns
    (a one-off's minimum) against 24."""
    world = toy_world()
    per_turn = fiscal_cost(cut(24), world) + fiscal_cost(rise(1), world)
    assert per_turn < 0.75  # what the old per-turn test saw, so it let the cut through
    life = fiscal_cost(cut(24), world) * 24 + fiscal_cost(rise(1), world) * MIN_FISCAL_TURNS
    assert unfunded_tax_cut([cut(24), rise(1)], world) == pytest.approx(
        fiscal_cost(cut(24), world) * life / (fiscal_cost(cut(24), world) * 24)
    )
    assert unfunded_tax_cut([cut(24), rise(24)], world) == 0


@pytest.mark.parametrize(("cut_turns", "rise_turns"), [(24, 1), (24, 12), (12, 6), (10, 6)])
def test_a_shorter_tax_rise_cannot_fund_a_longer_cut(cut_turns, rise_turns) -> None:
    """Property test 7: an unfunded tax cut is judged on lifetime cost. Savings that pay for
    the cuts in full while they run fund them when they last as long, and not when shorter."""
    world = toy_world()
    cuts = [act("tax", g, -1.0, cut_turns) for g in ("group:business", "group:pensioners")]
    savings = [act("spend", "sector:public", -1.0), act("tax", "sector:finance", 1.0)]
    share = sum(fiscal_cost(a, world) for a in cuts) / -sum(fiscal_cost(a, world) for a in savings)
    assert share <= 1

    def paid(turns: int) -> list[PolicyAction]:
        return [
            a.model_copy(update={"magnitude": a.magnitude * share, "duration_turns": turns})
            for a in savings
        ]

    assert unfunded_tax_cut([*cuts, *paid(cut_turns)], world) == pytest.approx(0, abs=1e-9)
    assert unfunded_tax_cut([*cuts, *paid(rise_turns)], world) > 0


def test_equal_lives_compare_as_before() -> None:
    world = toy_world()
    package = [cut(12), act("spend", "sector:public", -0.1, 12)]
    costs = [fiscal_cost(a, world) for a in package]
    assert unfunded_tax_cut(package, world) == pytest.approx(costs[0] + costs[1])


def test_past_the_limit_only_a_saving_that_lasts_funds_borrowing() -> None:
    world = toy_world()
    world.indicators[DEFICIT].value = 12
    short = check_feasibility([cut(24), rise(1)], world).checks[0]
    assert not short.feasible
    assert any("end before it does" in b for b in short.blockers)
    assert check_feasibility([cut(24), rise(24)], world).checks[0].feasible


def test_the_deficit_ceiling_counts_every_turn_the_package_runs() -> None:
    """Below the limit, a cut funded only for its first turns is scaled to what the deficit
    can take once the rise has ended."""
    world = toy_world()
    world.indicators[DEFICIT].value = 9.5
    package = [act("tax", "group:business", -1.0, 24), act("tax", "sector:finance", 1.0, 1)]
    fitted, note = fit_deficit(package, world)
    room = 10.0 - 9.5
    assert fitted[0].magnitude == pytest.approx(-room / fiscal_cost(package[0], world))
    assert note is not None and "past the 10% limit" in note
    # Paid for to the end, the same cut fits as it is.
    lasting = [package[0], package[1].model_copy(update={"duration_turns": 24})]
    assert fit_deficit(lasting, world) == (lasting, None)


def test_fiscal_turns_has_a_floor_for_one_offs() -> None:
    assert fiscal_turns(rise(1)) == MIN_FISCAL_TURNS
    assert fiscal_turns(rise(24)) == 24
