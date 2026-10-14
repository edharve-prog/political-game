"""A loser or a risk for every lever (O2; review of 13 October, E2 and E4).

Levers that cost no money used to have no downside beyond one turn's political capital, so a
player could buy a majority early with laws, speeches and deregulation and coast to the
election. Each such lever now carries a cost that falls on someone or something:

* **A law for a group** (or any rule aimed at one: ``RULE_KINDS``) has a losing side
  (``LAW_COUNTERPARTIES``): rights for renters cost landlords and employers, protections for
  pensioners are paid for by the young. The
  counterparty loses ``LAW_COUNTERPARTY_SHARE`` of what the group gains. Unless the same
  package delivers for the group too (spending on it or a tax cut for it), the law's appeal
  wears thin: it is felt in full for at most ``LAW_HOLD`` turns, then fades.
* **A speech** to a sector or a country lifts confidence once and fades, instead of holding
  output up for as long as it "runs" (see ``world.propagation.actions_to_shocks``); a speech
  to a group is felt for its few turns and fades (``policy.limits.MAX_TURNS``).
* **Deregulation** costs workers protections (``DEREGULATION_LOSERS``), and some sectors
  carry a side effect on an indicator (``DEREGULATION_SIDE``: energy bills grow more
  volatile, banks lend into the housing market) or a risk (``DEREGULATION_RISKS``: a lighter
  touch on finance makes a sell-off or capital flight likelier).

The groups named here are the toy UK world's; in a world without them the entries do nothing.
"""

from __future__ import annotations

from collections.abc import Sequence

from hog_sim.core.models import PolicyAction

# Who loses when a law favours a group, with each one's share of the loss.
LAW_COUNTERPARTIES: dict[str, dict[str, float]] = {
    # Protections for pensioners (the triple lock, care) are paid for by working-age renters.
    "group:pensioners": {"group:young_renters": 1.0},
    # Renters' rights cost landlords and letting businesses.
    "group:young_renters": {"group:business": 1.0},
    # Pay and union rights for public workers are resented by employers who compete for staff.
    "group:public_workers": {"group:business": 1.0},
    # Laws for business (flexible contracts, planning freedom) come out of workers' rights.
    "group:business": {"group:public_workers": 0.5, "group:young_renters": 0.5},
}
LAW_COUNTERPARTY_SHARE = 1.0
# Turns, after the first, that a law for a group is felt in full when nothing in the same
# package delivers for that group; then it fades with the usual half-life.
LAW_HOLD = 2

# Only loosening carries these side effects; tightening rules is ``regulate``'s business.
# Approval lost per unit of deregulation, in standard steps of approval (GROUP_STEP each), by
# the groups whose protections it loosens. Private-sector workers are not in the toy world
# yet (review M5); young renters stand in for them.
DEREGULATION_LOSERS: dict[str, dict[str, float]] = {
    "sector:public": {"group:public_workers": 0.6},
}
DEREGULATION_DEFAULT_LOSERS = {"group:young_renters": 0.3, "group:public_workers": 0.3}
# Indicator pushes per unit of deregulation, standard steps, held while it runs.
DEREGULATION_SIDE: dict[str, dict[str, float]] = {
    # More volatile, less regulated supply passes risk on to bills.
    "sector:energy": {"indicator:energy_prices": 0.2},
    # Laxer lending feeds house prices.
    "sector:finance": {"indicator:house_prices": 0.5},
    # Laxer standards let landlords and builders charge more for less.
    "sector:housing": {"indicator:house_prices": 0.4},
}
# Extra chance, per unit of deregulation (tightening carries none), of these outcome events:
# the base rate is multiplied by 1 + risk * magnitude.
DEREGULATION_RISKS: dict[str, dict[str, float]] = {
    "sector:finance": {"market_selloff": 1.0, "capital_flight": 1.0},
}


def delivers_for(group: str, actions: Sequence[PolicyAction]) -> bool:
    """Whether the package puts money behind a group: spending on it or cutting its taxes."""
    return any(
        a.target == group and a.kind in ("spend", "tax") and a.magnitude * _favour(a) > 0
        for a in actions
    )


def _favour(action: PolicyAction) -> float:
    """+1 when a tax or spend favours its target (spending, or a tax cut), else -1."""
    return -1.0 if action.kind == "tax" else 1.0


# Kinds that set rules. Aimed at a group (a law for renters, regulating landlords, freeing
# business from red tape) each is a law for or against it, with a counterparty.
RULE_KINDS = frozenset({"legislate", "regulate", "deregulate"})


def law_counterparties(action: PolicyAction) -> dict[str, float]:
    """Groups that lose when a rule favours ``action.target`` (and gain when it goes against
    it), with each one's share of the target's effect, negated."""
    if action.kind not in RULE_KINDS:
        return {}
    losers = LAW_COUNTERPARTIES.get(action.target, {})
    return {g: -LAW_COUNTERPARTY_SHARE * share for g, share in losers.items()}


def deregulation_losers(action: PolicyAction) -> dict[str, float]:
    """Approval each group loses per unit of ``action``, in standard steps of approval."""
    if action.kind != "deregulate" or action.magnitude <= 0:
        return {}
    if not action.target.startswith("sector:"):
        return {}
    losers = DEREGULATION_LOSERS.get(action.target, DEREGULATION_DEFAULT_LOSERS)
    return {g: -steps for g, steps in losers.items()}


def deregulation_side(action: PolicyAction) -> dict[str, float]:
    """Indicator pushes per unit of ``action``, standard steps."""
    if action.kind != "deregulate" or action.magnitude <= 0:
        return {}
    return DEREGULATION_SIDE.get(action.target, {})


def risk_factor(tag: str, actions: Sequence[PolicyAction]) -> float:
    """Multiplier on an outcome event's base rate from the risks this turn's levers carry."""
    factor = 1.0
    for action in actions:
        if action.kind == "deregulate" and action.magnitude > 0:
            factor *= 1 + DEREGULATION_RISKS.get(action.target, {}).get(tag, 0.0) * abs(
                action.magnitude
            )
    return factor
