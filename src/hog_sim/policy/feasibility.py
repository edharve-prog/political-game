"""Feasibility: can the leader actually do what the interpreter says they want to?

Deterministic and engine-side. Each action gets the requirements its kind implies for the
player's role, merged with any the interpreter listed, and each requirement is checked
against the WorldState. Blockers make an action infeasible; warnings and ``resistance``
(0..1) feed later projects (forecasting, institutions) without stopping the action.

Rules for the MVP:
- Tax, spend and legislation need a legislative majority, for a Prime Minister (Commons)
  and a President (Congress) alike. A legislature with support under 0.5 blocks them.
- Everything else is within executive power for both roles.
- Spending rises and tax cuts need fiscal headroom: a warning above ``deficit_warn`` and a
  blocker above ``deficit_limit`` (deficit in % of GDP).
- Independent institutions (independence >= 0.7) cannot be directed, and nor can the
  indicators only they drive (Bank Rate). Communicating with them is always allowed.
- Each kind of action fits only some kinds of target (``COMPATIBLE``): diplomacy and military
  action need a foreign country, appointments an institution. An indicator can be acted on
  directly only by the interventions it declares (``Indicator.interventions``, such as a
  price cap on energy bills); everything else has to work through a sector, group or
  institution, so the world graph carries the trade-offs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from hog_sim.core.models import EdgeKind, Institution, Model, NodeKind, PolicyAction
from hog_sim.core.state import WorldState
from hog_sim.world.propagation import fiscal_size

Role = Literal["prime_minister", "president"]

LEGISLATURE_MAJORITY = "legislature_majority"
BUDGET_HEADROOM = "budget_headroom"
EXECUTIVE_AUTHORITY = "executive_authority"
KNOWN_REQUIREMENTS = {LEGISLATURE_MAJORITY, BUDGET_HEADROOM, EXECUTIVE_AUTHORITY}

LEGISLATIVE_KINDS = {"tax", "spend", "legislate"}
INDEPENDENCE_THRESHOLD = 0.7
LEGISLATURE_ID = "institution:legislature"
DEFICIT_ID = "indicator:deficit"


# Action kinds each kind of target accepts (``do_nothing`` fits anything). Indicators take
# only their own ``interventions``; the player's own country takes domestic policy only.
_DOMESTIC = {"tax", "spend", "regulate", "deregulate", "legislate", "communicate"}
COMPATIBLE: dict[NodeKind, set[str]] = {
    NodeKind.COUNTRY: {"diplomatic", "military", "communicate"},
    NodeKind.SECTOR: _DOMESTIC,
    NodeKind.GROUP: _DOMESTIC,
    NodeKind.INSTITUTION: {
        "spend",
        "regulate",
        "deregulate",
        "legislate",
        "communicate",
        "appoint",
    },
}


def compatibility_problem(action: PolicyAction, state: WorldState) -> str | None:
    """Why ``action`` cannot be aimed at its target, or None when it can."""
    if action.kind == "do_nothing" or action.target not in set(state.node_ids()):
        return None
    node = state.node(action.target)
    if node.kind == NodeKind.INDICATOR:
        if action.kind in state.indicators[action.target].interventions:
            return None
        return (
            f"{action.kind} cannot set {node.name} directly; act on a sector, group or "
            "institution that drives it"
        )
    if action.target == state.player_country:
        allowed = _DOMESTIC
    else:
        allowed = COMPATIBLE[node.kind]
    if action.kind in allowed:
        return None
    return f"{action.kind} does not apply to {node.name}"


class ActionCheck(Model):
    action: PolicyAction
    feasible: bool
    requirements: list[str]
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    resistance: float = Field(0.0, ge=0, le=1)


class FeasibilityReport(Model):
    checks: list[ActionCheck]

    @property
    def feasible_actions(self) -> list[PolicyAction]:
        return [c.action for c in self.checks if c.feasible]

    @property
    def all_feasible(self) -> bool:
        return all(c.feasible for c in self.checks)


def requirements_for(action: PolicyAction, role: Role) -> list[str]:
    reqs = set(action.requires)
    if action.kind in LEGISLATIVE_KINDS:
        reqs.add(LEGISLATURE_MAJORITY)
    elif action.kind != "do_nothing":
        reqs.add(EXECUTIVE_AUTHORITY)
    if _loosens_budget(action):
        reqs.add(BUDGET_HEADROOM)
    return sorted(reqs)


def check_feasibility(
    actions: list[PolicyAction],
    state: WorldState,
    role: Role = "prime_minister",
    *,
    deficit_warn: float = 5.0,
    deficit_limit: float = 10.0,
) -> FeasibilityReport:
    return FeasibilityReport(
        checks=[
            _check(a, state, role, deficit_warn=deficit_warn, deficit_limit=deficit_limit)
            for a in actions
        ]
    )


def _check(
    action: PolicyAction,
    state: WorldState,
    role: Role,
    *,
    deficit_warn: float,
    deficit_limit: float,
) -> ActionCheck:
    reqs = requirements_for(action, role)
    blockers: list[str] = []
    warnings: list[str] = []
    resistance = 0.0
    node_ids = set(state.node_ids())

    if action.target not in node_ids:
        blockers.append(f"unknown target {action.target!r}")

    for req in reqs:
        if req == LEGISLATURE_MAJORITY:
            legislature = _legislature(state)
            name = _legislature_name(role, legislature)
            if legislature is None:
                warnings.append("no legislature in the world model; majority assumed")
            elif legislature.support < 0.5:
                blockers.append(f"{name} support is {legislature.support:.2f}, short of a majority")
            else:
                resistance = max(resistance, (1 - legislature.support) * legislature.power)
        elif req == BUDGET_HEADROOM:
            deficit = state.indicators.get(DEFICIT_ID)
            if deficit is not None and deficit.value > deficit_limit:
                blockers.append(
                    f"deficit is {deficit.value:g}% of GDP, above the {deficit_limit:g}% limit"
                )
            elif deficit is not None and deficit.value > deficit_warn:
                warnings.append(f"deficit is already {deficit.value:g}% of GDP")
                resistance = max(resistance, min(1.0, (deficit.value - deficit_warn) / 10))
        elif req not in KNOWN_REQUIREMENTS:
            warnings.append(f"unrecognised requirement {req!r} not checked")

    if action.kind not in ("communicate", "do_nothing"):
        controller = _independent_controller(action.target, state)
        if controller is not None:
            blockers.append(f"{controller.name} is independent of government")

    if (problem := compatibility_problem(action, state)) is not None:
        blockers.append(problem)

    return ActionCheck(
        action=action,
        feasible=not blockers,
        requirements=reqs,
        blockers=blockers,
        warnings=warnings,
        resistance=round(resistance, 3),
    )


def _loosens_budget(action: PolicyAction) -> bool:
    size = fiscal_size(action)
    return (action.kind == "spend" and size > 0) or (action.kind == "tax" and size < 0)


def _legislature(state: WorldState) -> Institution | None:
    if LEGISLATURE_ID in state.institutions:
        return state.institutions[LEGISLATURE_ID]
    return None


def _legislature_name(role: Role, legislature: Institution | None) -> str:
    if legislature is not None:
        return legislature.name
    return "Congress" if role == "president" else "The Commons"


def _independent_controller(target: str, state: WorldState) -> Institution | None:
    """The independent institution that owns ``target``, if any.

    Either the target is that institution, or it is an indicator the institution controls
    (``Indicator.controlled_by``) or is the only non-indicator driver of.
    """
    inst = state.institutions.get(target)
    if inst is not None:
        return inst if inst.independence >= INDEPENDENCE_THRESHOLD else None
    if target not in state.indicators:
        return None
    owner = state.institutions.get(state.indicators[target].controlled_by or "")
    if owner is not None:
        return owner if owner.independence >= INDEPENDENCE_THRESHOLD else None
    # Indicators that feed the target (inflation -> Bank Rate) are not levers, so ignore them.
    drivers = [
        e.source
        for e in state.edges
        if e.kind == EdgeKind.DRIVES and e.target == target and e.source not in state.indicators
    ]
    if len(drivers) == 1 and drivers[0] in state.institutions:
        owner = state.institutions[drivers[0]]
        if owner.independence >= INDEPENDENCE_THRESHOLD:
            return owner
    return None
