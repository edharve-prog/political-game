"""Who carries a tax or a price cap (M4, Ed's review of 13 October 2026): a windfall tax falls
on profits rather than output, VAT on customers, and a cap on bills is paid for by the
suppliers or by the Treasury, never by nobody."""

import pytest

from hog_sim.core.models import PolicyAction, Scenario
from hog_sim.game.stubs import KeywordInterpreter
from hog_sim.world.propagation import (
    ACTION_STEPS,
    CONSUMERS_INFLATION,
    CONSUMERS_OUTPUT_SHARE,
    FISCAL_NODE,
    PROFITS_OUTPUT_SHARE,
    SECTOR_PASS_THROUGH,
    actions_to_shocks,
    fiscal_cost,
    price_route,
    scale,
)
from hog_sim.world.seed.toy import toy_world

ENERGY, BILLS, INFLATION = "sector:energy", "indicator:energy_prices", "indicator:inflation"


def tax(borne_by: str | None = None, target: str = ENERGY) -> PolicyAction:
    return PolicyAction(
        kind="tax", target=target, magnitude=0.5, duration_turns=6, borne_by=borne_by
    )


def push(shocks, node: str) -> float:
    return sum(s.delta for s in shocks if s.node == node)


def test_a_sector_tax_falls_on_output_unless_told_otherwise() -> None:
    assert tax().incidence == "output"
    assert tax("treasury").incidence == "output"  # not a choice a tax has
    assert tax("profits", target="group:business").incidence is None
    cap = PolicyAction(kind="regulate", target=BILLS, magnitude=-1.0)
    assert cap.incidence == "suppliers"


def output_per_pound(action: PolicyAction, world) -> float:
    """Output the tax takes off its sector (bn) for each bn it raises."""
    cut = -push(actions_to_shocks([action], world), action.target) * scale(world, action.target)
    gdp = world.countries[world.player_country].gdp_bn
    raised = -fiscal_cost(action, world) * scale(world, FISCAL_NODE) / 100 * gdp
    return cut / raised


@pytest.mark.parametrize(
    ("borne_by", "share"),
    [
        ("output", SECTOR_PASS_THROUGH),
        ("profits", PROFITS_OUTPUT_SHARE),
        ("consumers", CONSUMERS_OUTPUT_SHARE),
    ],
)
def test_each_incidence_raises_as_much_and_says_how_much_output_it_costs(borne_by, share):
    # FR-28: the profits and customers shares used to multiply the 0.5 pass-through that
    # every sector tax already had, so a windfall tax cut an eighth of a pound of output for
    # each pound raised rather than a quarter.
    world = toy_world()
    action = tax(borne_by)
    assert fiscal_cost(action, world) == pytest.approx(fiscal_cost(tax(), world))
    assert output_per_pound(action, world) == pytest.approx(share)


def test_a_windfall_tax_cuts_half_the_output_a_levy_does() -> None:
    world = toy_world()
    usual = push(actions_to_shocks([tax()], world), ENERGY)
    windfall = push(actions_to_shocks([tax("profits")], world), ENERGY)
    assert windfall == pytest.approx(usual * PROFITS_OUTPUT_SHARE / SECTOR_PASS_THROUGH)
    # Public services are bought outright, so a levy on them cuts a pound for a pound.
    levy = tax(target="sector:public")
    assert output_per_pound(levy, world) == pytest.approx(1.0)
    assert output_per_pound(tax("profits", "sector:public"), world) == pytest.approx(
        PROFITS_OUTPUT_SHARE
    )


def test_a_tax_on_customers_lifts_prices_for_a_year() -> None:
    world = toy_world()
    action = tax("consumers", target="sector:manufacturing")
    shocks = actions_to_shocks([action], world)
    (prices,) = [s for s in shocks if s.node == INFLATION]
    raised = -fiscal_cost(action, world) * scale(world, FISCAL_NODE)
    assert raised > 0
    # Manufacturing sets no price of its own, so the whole rise lands on inflation.
    assert price_route(world, "sector:manufacturing") == 0
    assert prices.delta == pytest.approx(raised * CONSUMERS_INFLATION) and prices.hold
    assert prices.duration_turns == 12
    cut = action.model_copy(update={"magnitude": -0.5})
    assert push(actions_to_shocks([cut], world), INFLATION) < 0


def test_a_tax_on_energy_customers_does_not_count_dearer_bills_twice() -> None:
    # FR-28: less energy output already lifts bills, and bills lift inflation; the customers'
    # price rise used to come on top of that in full.
    world = toy_world()
    action = tax("consumers")
    shocks = actions_to_shocks([action], world)
    (prices,) = [s for s in shocks if s.node == INFLATION]
    raised = -fiscal_cost(action, world) * scale(world, FISCAL_NODE)
    whole = raised * CONSUMERS_INFLATION
    through_bills = push(shocks, ENERGY) * price_route(world, ENERGY)
    assert through_bills > 0
    assert prices.delta == pytest.approx(whole - through_bills)
    assert prices.delta + through_bills == pytest.approx(whole)
    assert ACTION_STEPS * 0.5 * CONSUMERS_OUTPUT_SHARE / SECTOR_PASS_THROUGH == pytest.approx(
        -push(shocks, ENERGY)
    )


def test_suppliers_carry_a_cap_unless_the_treasury_pays() -> None:
    world = toy_world()
    cap = PolicyAction(kind="regulate", target=BILLS, magnitude=-1.0, duration_turns=6)
    paid = cap.model_copy(update={"borne_by": "treasury"})
    on_suppliers, on_treasury = actions_to_shocks([cap], world), actions_to_shocks([paid], world)
    assert push(on_suppliers, BILLS) == push(on_treasury, BILLS) < 0
    assert push(on_suppliers, ENERGY) < 0 and push(on_treasury, ENERGY) == 0
    assert fiscal_cost(cap, world) == 0
    # The Treasury pays what a subsidy that moves bills as far costs.
    subsidy = PolicyAction(kind="spend", target=BILLS, magnitude=-0.5, duration_turns=6)
    assert push(actions_to_shocks([subsidy], world), BILLS) == pytest.approx(
        push(on_treasury, BILLS)
    )
    assert fiscal_cost(paid, world) == pytest.approx(fiscal_cost(subsidy, world))
    assert fiscal_cost(paid, world) > 0
    lifted = paid.model_copy(update={"magnitude": 1.0})
    assert fiscal_cost(lifted, world) > 0  # holding prices up is paid for too


@pytest.mark.parametrize(
    ("text", "borne_by"),
    [
        ("Put a windfall tax on energy profits", "profits"),
        ("Put a duty on energy companies", "consumers"),
        ("Tax energy companies", None),
        ("Cap energy bills and pay suppliers the difference", "treasury"),
        ("Cap energy bills", None),
    ],
)
def test_the_keyword_reader_says_who_pays(text, borne_by) -> None:
    world = toy_world()
    scenario = Scenario(
        title="Bills", briefing="", affected_nodes=[BILLS, ENERGY], urgency=0.5, shocks=[]
    )
    (action,) = KeywordInterpreter().interpret(text, world, scenario)
    assert action.borne_by == borne_by
