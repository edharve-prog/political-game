"""The debt stock and its interest bill (Project 16, EB-9).

The player's country carries public debt (``Country.debt_pct``, % of GDP) and pays an average
rate on it (``Country.debt_rate_pct``). Each turn (a month):

* the debt grows by a twelfth of the deficit and is worn down by a twelfth of nominal growth
  (real growth plus inflation) times the debt, so borrowing compounds;
* the average rate closes ``REPRICE`` of the gap to the market rate on new borrowing, as old
  debt matures and is refinanced, so that half of a rate rise reaches the debt in
  ``REPRICE_HALF_LIFE`` turns. That market rate is the gilt yield, blended with Bank Rate for
  the ``RESERVES_SHARE`` of the debt that is reserves created by quantitative easing (O4); a
  world with no gilt yield uses Bank Rate alone;
* the interest bill, debt × average rate, adds to the deficit whatever it has risen since
  the game began. The start's own bill is already in the starting deficit, so the starting
  world is unchanged; higher rates or a bigger debt cost money, and lower ones save it.

The extra bill reaches the deficit as a push that keeps the deficit that much higher for as
long as the bill stays up, so it also feeds gilt yields and the pound through the usual links.
"""

from __future__ import annotations

from hog_sim.core.models import Shock
from hog_sim.core.state import WorldState
from hog_sim.world.propagation import FISCAL_NODE, persistence, scale

RATE_NODE = "indicator:interest_rate"
GILT_NODE = "indicator:gilt_yield"
RESERVES_SHARE = 0.25  # of the debt, paying Bank Rate rather than a gilt yield
INFLATION_NODE = "indicator:inflation"
TURNS_PER_YEAR = 12
REPRICE_HALF_LIFE = 24  # turns for half of a change in the market rate to reach the debt
REPRICE = 1 - 0.5 ** (1 / REPRICE_HALF_LIFE)


def _value(state: WorldState, node: str, default: float = 0.0) -> float:
    indicator = state.indicators.get(node)
    return indicator.value if indicator is not None else default


def _tracks(state: WorldState) -> bool:
    return state.countries[state.player_country].tracks_debt


def debt_rate(state: WorldState, start: WorldState | None = None) -> float:
    """Average rate paid on the player's debt. Until set, it is the market rate at the start
    (``start``, or ``state`` itself when that is the start)."""
    country = state.countries[state.player_country]
    if country.debt_rate_pct is not None:
        return country.debt_rate_pct
    return market_rate(_rates(start if start is not None else state))


def market_rate(values: dict[str, float]) -> float:
    """The rate new borrowing pays, from native values by node id: gilts, plus Bank Rate on
    the reserves share; Bank Rate alone when the world has no gilt yield."""
    bank = values.get(RATE_NODE, 0.0)
    if GILT_NODE not in values:
        return bank
    return (1 - RESERVES_SHARE) * values[GILT_NODE] + RESERVES_SHARE * bank


def _rates(state: WorldState) -> dict[str, float]:
    return {n: state.indicators[n].value for n in (RATE_NODE, GILT_NODE) if n in state.indicators}


def interest_bill(state: WorldState, start: WorldState | None = None) -> float:
    """Yearly interest on the player's debt, % of GDP."""
    return state.countries[state.player_country].debt_pct * debt_rate(state, start) / 100


def debt_line(state: WorldState) -> str:
    """One line on the debt for briefings; empty when debt is not tracked."""
    country = state.countries[state.player_country]
    if not country.tracks_debt:
        return ""
    debt = country.debt_pct
    return (
        f"Public debt: {debt:.0f}% of GDP at an average {debt_rate(state):.1f}% interest, "
        f"a bill of {interest_bill(state):.1f}% of GDP a year"
    )


def interest_shocks(state: WorldState, start: WorldState) -> list[Shock]:
    """This turn's push on the deficit from the interest bill's rise since the start.

    A push of ``(1 - persistence)`` times the extra bill, renewed each turn, holds the
    deficit that much above where it would be (the same top-up a held policy uses)."""
    if FISCAL_NODE not in state.indicators or not _tracks(state):
        return []
    return _interest_push(state, interest_bill(state, start) - interest_bill(start))


def _interest_push(state: WorldState, extra: float) -> list[Shock]:
    if not extra:
        return []
    push = extra / scale(state, FISCAL_NODE) * (1 - persistence(state, FISCAL_NODE))
    return [Shock(node=FISCAL_NODE, delta=push)]


def _next_debt(
    debt: float, rate: float, deficit: float, nominal: float, market: float
) -> tuple[float, float]:
    """The debt stock and its average rate a turn later."""
    debt = max(0.0, debt + (deficit - debt * nominal / 100) / TURNS_PER_YEAR)
    return debt, rate + REPRICE * (market - rate)


class DebtOutlook:
    """The debt and its interest bill projected turn by turn for the forecast, by the same
    rules as ``step_debt`` and ``interest_shocks``, from the forecast's own projected values
    (native units by node id). Call it once per turn ahead, after the turn before has moved."""

    def __init__(self, state: WorldState, start: WorldState) -> None:
        self.state = state
        self.on = FISCAL_NODE in state.indicators and _tracks(state)
        self.debt = state.countries[state.player_country].debt_pct
        self.rate = debt_rate(state, start)
        self.start_bill = interest_bill(start)

    def __call__(self, values: dict[str, float]) -> list[Shock]:
        if not self.on:
            return []
        self.debt, self.rate = _next_debt(
            self.debt,
            self.rate,
            values.get(FISCAL_NODE, 0.0),
            values[self.state.player_country] + values.get(INFLATION_NODE, 0.0),
            market_rate(values),
        )
        return _interest_push(self.state, self.debt * self.rate / 100 - self.start_bill)


def step_debt(state: WorldState, start: WorldState | None = None) -> WorldState:
    """Advance the debt stock and its average rate by one turn. Returns a new state.

    ``start`` supplies the opening rate when the state has none yet."""
    if not _tracks(state):
        return state
    new = state.snapshot()
    home = new.countries[new.player_country]
    home.debt_tracked = True  # a debt paid down to 0 is still tracked, and grows again
    home.debt_pct, home.debt_rate_pct = _next_debt(
        home.debt_pct,
        debt_rate(state, start),
        _value(state, FISCAL_NODE),
        home.growth_pct + _value(state, INFLATION_NODE),
        market_rate(_rates(state)),
    )
    return new
