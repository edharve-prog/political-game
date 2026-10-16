"""Gilt yields and a Bank Rate policy rule (O4, Ed's review of 13 October 2026): markets price
fiscal credibility into gilts, the debt reprices toward them, and the independent Bank answers
inflation by more than one for one."""

from hog_sim.core.config import GameConfig
from hog_sim.core.models import EdgeKind, Outcome, PolicyAction, Scenario, Shock
from hog_sim.game.loop import resolve
from hog_sim.world.debt import RESERVES_SHARE, debt_rate, step_debt
from hog_sim.world.propagation import DAMPING
from hog_sim.world.seed.toy import toy_world

NOTHING = Outcome(narrative="Nothing else happens", probability=1)
RATE, GILT = "indicator:interest_rate", "indicator:gilt_yield"
INFLATION, DEFICIT = "indicator:inflation", "indicator:deficit"


def scenario(*shocks: Shock) -> Scenario:
    return Scenario(title="Quiet", briefing="", affected_nodes=[], urgency=0, shocks=list(shocks))


def play(turns: int, first: Scenario | None = None, actions=()):
    start = state = toy_world()
    states = []
    for t in range(turns):
        sc = first if t == 0 and first is not None else scenario()
        state = resolve(start, state, sc, list(actions) if t == 0 else [], NOTHING, GameConfig())
        states.append(state)
    return states


def weight(source: str, target: str) -> float:
    return sum(e.weight for e in toy_world().edges if e.source == source and e.target == target)


def test_the_rule_answers_inflation_more_than_one_for_one_in_the_long_run():
    smoothing = weight(RATE, RATE) * DAMPING
    assert 0 < smoothing < 1
    assert weight(INFLATION, RATE) * DAMPING / (1 - smoothing) > 1


def test_lasting_inflation_raises_real_rates():
    """Held inflation pressure: on average Bank Rate rises by more than inflation does, so the
    real rate goes up and leans against it (the rule's own tightening pulls inflation down,
    so the two swing around each other along the way)."""
    start = toy_world()
    push = Shock(node=INFLATION, delta=1.0, duration_turns=36, hold=True)
    later = play(36, scenario(push))[6:]
    rise = [s.indicators[RATE].value - start.indicators[RATE].value for s in later]
    inflation = [s.indicators[INFLATION].value - start.indicators[INFLATION].value for s in later]
    assert min(inflation) > 0
    assert sum(rise) > sum(inflation)


def test_borrowing_moves_gilts_not_bank_rate():
    world = toy_world()
    assert not any(
        e.source == DEFICIT and e.target == RATE and e.kind == EdgeKind.DRIVES for e in world.edges
    )
    assert weight(DEFICIT, GILT) > 0


def test_an_unfunded_tax_cut_shows_in_gilts_first():
    start = toy_world()
    cut = PolicyAction(kind="tax", target="country:uk", magnitude=-0.8, duration_turns=6)
    first = play(1, actions=[cut])[0]
    assert first.indicators[GILT].value - start.indicators[GILT].value > 0.5
    assert first.indicators[RATE].value == start.indicators[RATE].value


def test_the_debt_reprices_toward_gilts():
    start = toy_world()
    assert debt_rate(start) == start.countries["country:uk"].debt_rate_pct
    dear = start.snapshot()
    dear.indicators[GILT].value += 2.0
    gilts = step_debt(dear, start).countries["country:uk"].debt_rate_pct
    bank = start.snapshot()
    bank.indicators[RATE].value += 2.0
    reserves = step_debt(bank, start).countries["country:uk"].debt_rate_pct
    base = start.countries["country:uk"].debt_rate_pct
    assert gilts - base > reserves - base > 0
    assert abs((reserves - base) / (gilts - base) - RESERVES_SHARE / (1 - RESERVES_SHARE)) < 1e-9


def test_a_quiet_world_leaves_the_debt_rate_alone():
    start = toy_world()
    late = play(12)[-1]
    assert abs(debt_rate(late) - debt_rate(start)) < 1e-9
    assert late.indicators[GILT].value == start.indicators[GILT].value


GAP, JOBS, GROWTH = "indicator:output_gap", "indicator:unemployment", "country:uk"


def test_bank_rate_works_through_the_output_gap():
    # FR-27: Bank Rate used to move inflation and unemployment directly, so tightening never
    # lowered growth or widened the cyclical deficit.
    assert weight(RATE, GAP) < 0
    assert weight(RATE, INFLATION) == 0 and weight(RATE, JOBS) == 0
    hike = play(12, scenario(Shock(node=RATE, delta=1.0)))
    quiet = play(12)
    gap = [
        s.indicators[GAP].value - q.indicators[GAP].value for s, q in zip(hike, quiet, strict=True)
    ]
    assert min(gap) < -0.3
    later = hike[-1], quiet[-1]
    assert later[0].countries[GROWTH].growth_pct < later[1].countries[GROWTH].growth_pct
    assert later[0].indicators[JOBS].value > later[1].indicators[JOBS].value
    assert later[0].indicators[INFLATION].value < later[1].indicators[INFLATION].value
    assert later[0].indicators[DEFICIT].value > later[1].indicators[DEFICIT].value


def test_factory_jobs_are_counted_once_through_the_gap():
    # FR-27: manufacturing reached unemployment directly and again through the gap (Okun).
    assert weight("sector:manufacturing", JOBS) == 0
    assert weight("sector:manufacturing", GAP) > 0 and weight(GAP, JOBS) < 0
