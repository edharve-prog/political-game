import pytest

from hog_sim.core.models import Edge, EdgeKind, PolicyAction
from hog_sim.core.state import WorldState
from hog_sim.policy.feasibility import check_feasibility, compatibility_problem, requirements_for
from hog_sim.world.seed.toy import toy_world


@pytest.fixture
def world() -> WorldState:
    return toy_world()


def act(kind: str, target: str, magnitude: float = 0.3, **kw) -> PolicyAction:
    return PolicyAction(kind=kind, target=target, magnitude=magnitude, **kw)


def only(report):
    assert len(report.checks) == 1
    return report.checks[0]


def test_tax_with_a_majority_is_feasible(world: WorldState) -> None:
    check = only(check_feasibility([act("tax", "sector:energy")], world))
    assert check.feasible
    assert check.requirements == ["legislature_majority"]
    assert check.resistance == pytest.approx(0.4 * 0.8)


def test_no_majority_blocks_legislation(world: WorldState) -> None:
    world.institutions["institution:legislature"].support = 0.4
    report = check_feasibility(
        [act("tax", "sector:energy"), act("diplomatic", "country:eu")], world
    )
    tax, diplomacy = report.checks
    assert not tax.feasible
    assert "House of Commons" in tax.blockers[0]
    assert diplomacy.feasible
    assert report.feasible_actions == [diplomacy.action]
    assert not report.all_feasible


def test_president_needs_congress_for_budget_not_for_regulation(world: WorldState) -> None:
    world.institutions["institution:legislature"].support = 0.3
    report = check_feasibility(
        [act("spend", "sector:public", -0.2), act("regulate", "sector:finance")],
        world,
        role="president",
    )
    assert [c.feasible for c in report.checks] == [False, True]
    assert report.checks[1].requirements == ["executive_authority"]


def test_central_bank_is_independent(world: WorldState) -> None:
    report = check_feasibility(
        [
            act("regulate", "indicator:interest_rate"),
            act("appoint", "institution:central_bank"),
            act("communicate", "institution:central_bank"),
        ],
        world,
    )
    assert [c.feasible for c in report.checks] == [False, False, True]
    assert "Bank of England" in report.checks[0].blockers[0]


def test_indicator_drivers_do_not_hide_the_central_bank(world: WorldState) -> None:
    # The bank's reaction to inflation is modelled as inflation DRIVES Bank Rate.
    world.edges.append(
        Edge(
            source="indicator:inflation",
            target="indicator:interest_rate",
            kind=EdgeKind.DRIVES,
            weight=0.5,
        )
    )
    check = only(check_feasibility([act("regulate", "indicator:interest_rate")], world))
    assert not check.feasible


def test_spending_needs_fiscal_headroom(world: WorldState) -> None:
    deficit = world.indicators["indicator:deficit"]
    deficit.value = 7.0
    rise = only(check_feasibility([act("spend", "sector:public", 0.3)], world))
    assert rise.feasible and "budget_headroom" in rise.requirements
    assert rise.warnings
    deficit.value = 12.0
    assert not only(check_feasibility([act("spend", "sector:public", 0.3)], world)).feasible
    assert not only(check_feasibility([act("tax", "group:business", -0.3)], world)).feasible
    # Cuts and tax rises do not need headroom.
    assert only(check_feasibility([act("spend", "sector:public", -0.3)], world)).feasible
    assert only(check_feasibility([act("tax", "group:business", 0.3)], world)).feasible


def test_unknown_target_and_requirement(world: WorldState) -> None:
    check = only(
        check_feasibility([act("spend", "sector:fishing", requires=["royal_assent"])], world)
    )
    assert not check.feasible
    assert any("royal_assent" in w for w in check.warnings)


def test_do_nothing_needs_nothing(world: WorldState) -> None:
    a = act("do_nothing", "country:uk", 0.0)
    assert requirements_for(a, "prime_minister") == []
    assert only(check_feasibility([a], world)).feasible


@pytest.mark.parametrize(
    ("kind", "target", "magnitude", "fits"),
    [
        ("regulate", "indicator:energy_prices", -0.5, True),
        ("spend", "indicator:energy_prices", -0.5, True),
        ("regulate", "indicator:inflation", -0.5, False),
        ("spend", "indicator:unemployment", -0.5, True),
        ("tax", "indicator:unemployment", -0.5, False),
        ("tax", "indicator:deficit", -0.5, False),
        ("military", "country:china", 0.5, True),
        ("military", "sector:energy", 0.5, False),
        ("diplomatic", "country:uk", 0.5, False),
        ("diplomatic", "group:pensioners", 0.5, False),
        ("appoint", "sector:finance", 0.5, False),
        ("spend", "sector:public", 0.5, True),
        ("tax", "country:china", 0.5, False),
        ("do_nothing", "country:uk", 0.0, True),
    ],
)
def test_actions_must_fit_their_target(kind, target, magnitude, fits) -> None:
    """Finding 4: indicators take only the interventions they declare, and each kind of
    action needs the right kind of target."""
    world = toy_world()
    action = PolicyAction(kind=kind, target=target, magnitude=magnitude)
    check = check_feasibility([action], world).checks[0]
    assert (compatibility_problem(action, world) is None) is fits
    if not fits:
        assert not check.feasible
