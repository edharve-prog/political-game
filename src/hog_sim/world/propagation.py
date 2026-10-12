"""Shock propagation through the world graph.

The engine is a linear dynamic system on the graph:

* Every node has one primary metric (see ``METRICS``) and a ``scale``: the size of one
  "standard step" in that metric's native units. Propagation runs in standard steps so
  edge weights are comparable: weight 0.5 means a one-step move at the source moves the
  target half a step.
* Each node's deviation from where it would otherwise be is its own push plus what its
  drivers pass on: ``d(n, t) = own(n, t) + sum(weight * DAMPING * d(source, t - lag))``.
  Lag-0 edges are solved within the turn by iterating to a fixed point.
* A node's own push fades: each turn it keeps ``persistence`` of it (``PERSISTENCE`` by node
  kind, or the indicator's own ``persistence``). A shock adds ``delta`` to the push every
  turn for ``duration_turns``; a held shock (``hold``) keeps the push at ``delta`` for
  ``duration_turns``, which is how a running policy works. Once nothing sustains a push,
  the node and everything it drives drift back.
* Population groups are not propagated here: approval is Project 4's job, so ``EMPLOYS``,
  ``CARES_ABOUT`` and any edge into a group are skipped.
* The engine carries its state between turns (``WorldState.engine``: fading pushes, recent
  deviations, shocks still running) and ``step`` moves it on one turn. ``resolve`` and the
  forecast (``propagate``) both use that one step, so the forecast starts from what is already
  in flight. Nodes stay within their bounds as they move: a bound drops the push that would
  have carried a node past it, so the node fades back from the bound, not from beyond it.

Monte Carlo draws perturb each edge weight by its ``uncertainty`` (a normal std dev) once
per draw, seeded from the game seed so results reproduce exactly.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from statistics import fmean, quantiles

from pydantic import Field

from hog_sim.core.config import make_rng
from hog_sim.core.models import Edge, EdgeKind, Model, NodeKind, PolicyAction, Shock
from hog_sim.core.state import EngineState, Push, WorldState

DAMPING = 0.9
MAX_ITERATIONS = 200  # lag-0 fixed point; a stable graph settles well within this
TOLERANCE = 1e-12
NEGLIGIBLE = 1e-12  # pushes and deviations smaller than this (standard steps) are dropped
# Highest loop gain allowed among lag-0 edges (see ``lag0_gain``). Below 1 the within-turn
# fixed point exists and the iteration converges for any signs; the margin keeps Monte
# Carlo draws stable too.
MAX_LAG0_GAIN = 0.9
GAIN_SIGMAS = 3.0  # edge weights are taken at |weight| + 3 std devs when bounding the gain


class PropagationError(RuntimeError):
    """The within-turn (lag-0) feedback did not settle: the graph's loops are unstable."""


PROPAGATING = {EdgeKind.DRIVES, EdgeKind.SUPPLIES, EdgeKind.TRADES_WITH, EdgeKind.INFLUENCES}

# Primary metric per node kind. Groups are listed so deltas can be applied to them later.
METRICS = {
    NodeKind.COUNTRY: "growth_pct",
    NodeKind.SECTOR: "output_bn",
    NodeKind.GROUP: "approval",
    NodeKind.INSTITUTION: "support",
    NodeKind.INDICATOR: "value",
}

# Share of a node's own push kept each turn (half-lives: indicators ~4 turns, sectors and
# institutions ~6.5, countries ~3). Groups keep theirs: approval drift is Project 4's job.
PERSISTENCE = {
    NodeKind.COUNTRY: 0.8,
    NodeKind.SECTOR: 0.9,
    NodeKind.GROUP: 1.0,
    NodeKind.INSTITUTION: 0.9,
    NodeKind.INDICATOR: 0.85,
}


def persistence(state: WorldState, node_id: str) -> float:
    node = state.node(node_id)
    if node.kind == NodeKind.INDICATOR and node.persistence is not None:
        return node.persistence
    return PERSISTENCE[node.kind]


# Native size of one standard step for each indicator unit.
INDICATOR_UNIT_SCALE = {"%": 1.0, "% GDP": 1.0, "index": 10.0}


def scale(state: WorldState, node_id: str) -> float:
    """Native units per standard step for a node's primary metric."""
    node = state.node(node_id)
    match node.kind:
        case NodeKind.SECTOR:
            return 0.05 * node.output_bn  # 5% of output
        case NodeKind.COUNTRY:
            return 1.0  # 1pp of growth
        case NodeKind.GROUP:
            return 0.05
        case NodeKind.INSTITUTION:
            return 0.1
        case _:
            return INDICATOR_UNIT_SCALE.get(node.unit, 1.0)


class NodeForecast(Model):
    """Cumulative change in the node's primary metric, native units, one entry per turn."""

    mean: list[float]
    p10: list[float]
    p90: list[float]


class DeltaDistribution(Model):
    horizon: int
    k_draws: int
    nodes: dict[str, NodeForecast]
    baseline: dict[str, list[float]] = Field(
        default_factory=dict,
        description="Cumulative native change per node per turn from what happens whatever "
        "the player decides (effects in flight, background pushes); empty when not run",
    )

    def at(self, turn: int) -> dict[str, float]:
        """Expected cumulative native delta for every node at ``turn`` (0-based)."""
        return {n: f.mean[turn] for n, f in self.nodes.items()}

    def response_at(self, turn: int) -> dict[str, float]:
        """The part of ``at(turn)`` that this turn's decisions and events make: the expected
        change less the baseline."""
        return {
            n: f.mean[turn] - self.baseline.get(n, [0.0] * (turn + 1))[turn]
            for n, f in self.nodes.items()
        }


# Sign of the first-order effect of each action kind on its target's primary metric. When
# the target is an indicator, the action's magnitude already says which way the player wants
# it to go ("cap bills" is energy prices, negative), so only the size of the factor is used.
# Diplomatic and military actions do not push their target's growth this way: they move its
# standing (``world/changes.action_changes``) and, where it makes sense, trade (``FOREIGN``).
_ACTION_SIGN = {
    "tax": -1.0,
    "spend": 1.0,
    "regulate": -0.5,
    "deregulate": 0.5,
    "diplomatic": 0.0,
    "military": 0.0,
    "communicate": 0.2,
    "legislate": 1.0,
    "appoint": 0.5,
    "do_nothing": 0.0,
}
ACTION_STEPS = 2.0  # magnitude 1.0 means a two-step push


# Whipping (EB-14): communicating with the legislature (whips, a meeting with backbenchers,
# concessions on the floor) shores up its support by this many steps per unit, far more than a
# speech moves anyone else.
WHIP_TARGET = "institution:legislature"
WHIP_STEPS = 1.0


def action_factor(action: PolicyAction) -> float:
    """Signed standard steps per unit of magnitude that the action pushes its target."""
    if action.kind == "communicate" and action.target == WHIP_TARGET:
        return WHIP_STEPS
    sign = _ACTION_SIGN[action.kind]
    if action.target.startswith("indicator:"):
        sign = abs(sign)
    return sign * ACTION_STEPS


# Spending and tax also move the budget, as a flow: magnitude 1.0 of spending (or a tax cut)
# adds this many standard steps to the deficit for every turn it runs, and the same size of
# tax rise takes it off. A one-off is paid for over MIN_FISCAL_TURNS.
FISCAL_NODE = "indicator:deficit"
_FISCAL_SIGN = {"spend": 1.0, "tax": -1.0}
FISCAL_STEPS = 1.25
MIN_FISCAL_TURNS = 6

# Markets and unfunded tax cuts (CA-4): a tax cut that this turn's other decisions do not pay
# for, worth at least CONFIDENCE_THRESHOLD steps of deficit a turn, makes markets doubt the
# government's plans, as after the 2022 mini-budget. Each step of unfunded cut pushes Bank
# Rate up and the pound down once, and costs every group approval (see
# ``population.popularity.policy_events``). Spending is not judged this way: markets lent
# freely for the 2008 bailout and the 2020 lockdown.
CONFIDENCE_THRESHOLD = 0.75
CONFIDENCE_SHOCKS = {"indicator:interest_rate": 0.4, "indicator:exchange_rate": -0.8}


def unfunded_tax_cut(actions: list[PolicyAction], state: WorldState | None = None) -> float:
    """Steps of deficit a turn from this turn's tax cuts that nothing else this turn pays for,
    or 0 when that is under ``CONFIDENCE_THRESHOLD``."""
    costs = [fiscal_cost(a, state) for a in actions]
    cuts = sum(c for a, c in zip(actions, costs, strict=True) if a.kind == "tax" and c > 0)
    savings = sum(-c for c in costs if c < 0)
    unfunded = cuts - savings
    return unfunded if unfunded >= CONFIDENCE_THRESHOLD else 0.0


# Job protection (CA-4): spending aimed at unemployment is a wage subsidy, such as the 2020
# furlough scheme. It does not push unemployment down. While it runs it holds back
# JOB_PROTECTION of any rise in unemployment its drivers would cause, per unit of magnitude,
# so firms keep their staff through a slump; in good times it only costs money. It covers any
# rise while it runs, including one from a slump already under way, and costs the same as any
# other spending.
JOB_NODE = "indicator:unemployment"
JOB_PROTECTION = 0.8


def protects_jobs(action: PolicyAction) -> bool:
    return action.kind == "spend" and action.target == JOB_NODE


def job_protection(state: WorldState, actions: list[PolicyAction]) -> dict[int, float]:
    """Share of job losses held back on each absolute turn: what earlier schemes still cover,
    plus this turn's spending on unemployment. Overlapping schemes cover the larger share."""
    cover = {t: share for t, share in state.job_protection.items() if t >= state.turn}
    for action in actions:
        if protects_jobs(action):
            share = min(1.0, JOB_PROTECTION * abs(action.magnitude))
            for t in range(state.turn, state.turn + action.duration_turns):
                cover[t] = max(cover.get(t, 0.0), share)
    return cover


def _held(node_id: str, push: float, share: float) -> float:
    """What drivers pass on to a node once job protection holds back its share of a rise."""
    if node_id == JOB_NODE and share and push > 0:
        return push * (1 - share)
    return push


# Regulating an indicator (a price cap, a rent control) trims the output of the sectors that
# drive it, by this share of the push.
REGULATION_OUTPUT_COST = 0.25


# Foreign policy towards another country, in standard steps of its growth per unit of
# magnitude. A deal or sanctions move trade only with a trading partner, in the direction of
# the magnitude. Military escalation disrupts the target's economy (and, through trade, the
# player's); de-escalation moves only the relationship.
FOREIGN = {"diplomatic", "military"}
DIPLOMATIC_TRADE_STEPS = 0.5
MILITARY_DISRUPTION_STEPS = 1.0


def _trades(state: WorldState, a: str, b: str) -> bool:
    return any(
        e.kind == EdgeKind.TRADES_WITH and {e.source, e.target} == {a, b} for e in state.edges
    )


def foreign_shocks(action: PolicyAction, state: WorldState | None) -> list[Shock]:
    """Economic shocks from a diplomatic or military action towards a foreign country."""
    if (
        state is None
        or action.target not in state.countries
        or action.target == state.player_country
    ):
        return []
    if action.kind == "diplomatic" and _trades(state, state.player_country, action.target):
        delta = DIPLOMATIC_TRADE_STEPS * action.magnitude
    elif action.kind == "military" and action.magnitude > 0:
        delta = -MILITARY_DISRUPTION_STEPS * action.magnitude
    else:
        return []
    return [Shock(node=action.target, delta=delta, duration_turns=action.duration_turns, hold=True)]


def fiscal_size(action: PolicyAction) -> float:
    """Signed size of the action's budget move: positive spends more or taxes more.

    On an indicator target the sign is the direction the player wants the indicator to go,
    so a subsidy to push prices down still costs money and a tax to cool them still raises it.
    """
    if action.target.startswith("indicator:"):
        return abs(action.magnitude)
    return action.magnitude


def fiscal_steps(action: PolicyAction, state: WorldState | None = None) -> float:
    """Steps of deficit a turn per unit of magnitude. Aimed at a sector, the cost is the
    output the action pushes, as a share of the player's GDP (CA-4): the money spent on, or
    taxed from, the sector is the output it adds or removes. Anything else costs
    ``FISCAL_STEPS``."""
    if state is None or action.target not in state.sectors:
        return FISCAL_STEPS
    gdp = state.countries[state.player_country].gdp_bn
    output = ACTION_STEPS * scale(state, action.target)
    return output / gdp * 100 / scale(state, FISCAL_NODE)


def fiscal_cost(action: PolicyAction, state: WorldState | None = None) -> float:
    """Standard steps a turn that the action adds to the deficit while it runs (negative
    when it takes them off). Actions aimed at the deficit itself have no separate cost."""
    if action.target == FISCAL_NODE:
        return 0.0
    return _FISCAL_SIGN.get(action.kind, 0.0) * fiscal_size(action) * fiscal_steps(action, state)


def actions_to_shocks(actions: list[PolicyAction], state: WorldState | None = None) -> list[Shock]:
    """Map interpreted actions to engine shocks.

    Each action holds its push on the target for its duration. Actions aimed at a population
    group are not shocks: they become approval events (``population.popularity.policy_events``).
    When ``state`` is given and has a deficit indicator, spending and tax also hold a cost on
    the deficit, so nothing is free and a longer programme costs more.
    """
    fiscal = state is not None and FISCAL_NODE in state.indicators
    shocks = []
    for action in actions:
        turns = action.duration_turns
        if action.kind in FOREIGN:
            shocks += foreign_shocks(action, state)
            continue
        push = 0.0 if protects_jobs(action) else action_factor(action) * action.magnitude
        if push and not action.target.startswith("group:"):
            shocks.append(Shock(node=action.target, delta=push, duration_turns=turns, hold=True))
            if (
                state is not None
                and action.kind == "regulate"
                and action.target in state.indicators
            ):
                cost = -REGULATION_OUTPUT_COST * abs(push)
                shocks += [
                    Shock(node=e.source, delta=cost, duration_turns=turns, hold=True)
                    for e in state.edges
                    if e.kind == EdgeKind.DRIVES
                    and e.target == action.target
                    and e.source in state.sectors
                ]
        cost = fiscal_cost(action, state)
        if fiscal and cost:
            shocks.append(
                Shock(
                    node=FISCAL_NODE,
                    delta=cost,
                    duration_turns=max(turns, MIN_FISCAL_TURNS),
                    hold=True,
                )
            )
    loss = unfunded_tax_cut(actions, state)
    if state is not None and loss:
        shocks += [
            Shock(node=node, delta=size * loss)
            for node, size in CONFIDENCE_SHOCKS.items()
            if node in state.indicators
        ]
    return shocks


def _propagating_edges(state: WorldState) -> list[Edge]:
    return [e for e in state.edges if e.kind in PROPAGATING and not e.target.startswith("group:")]


def lag0_gain(state: WorldState, edges: list[Edge] | None = None) -> float:
    """Upper bound on how much the lag-0 loops amplify a push within one turn.

    The spectral radius of the damped lag-0 matrix with every weight at ``|weight| +
    GAIN_SIGMAS * uncertainty``. Below 1 the fixed point in ``simulate`` converges whatever
    the signs and whatever a Monte Carlo draw picks within that range. Found by power
    iteration, which suits the small non-negative matrix.
    """
    matrix: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for e in edges if edges is not None else _propagating_edges(state):
        if not e.lag:
            matrix[e.target][e.source] += (abs(e.weight) + GAIN_SIGMAS * e.uncertainty) * DAMPING
    nodes = set(matrix) | {src for row in matrix.values() for src in row}
    if not nodes:
        return 0.0
    # Collatz-Wielandt: for any positive vector v, max_i (Mv)_i / v_i bounds the spectral
    # radius from above. Shifted power iteration tightens the bound.
    vector = dict.fromkeys(nodes, 1.0)
    bound = math.inf
    for _ in range(500):
        product = {n: sum(w * vector[src] for src, w in matrix[n].items()) for n in nodes}
        bound = min(bound, max(product[n] / vector[n] for n in nodes))
        if bound == 0:
            break
        top = max(product.values()) or 1.0
        vector = {n: vector[n] + product[n] / top for n in nodes}
        peak = max(vector.values())
        vector = {n: v / peak for n, v in vector.items()}
    return bound


def _bounds(state: WorldState, node_id: str) -> tuple[float | None, float | None]:
    """Hard limits on a node's primary metric, native units (``apply_deltas`` keeps them)."""
    node = state.node(node_id)
    if node.kind in (NodeKind.GROUP, NodeKind.INSTITUTION):
        return 0.0, 1.0
    if node.kind == NodeKind.SECTOR:
        return 0.0, None
    if node.kind == NodeKind.INDICATOR:
        return node.low, node.high
    return None, None


def _value(state: WorldState, node_id: str) -> float:
    node = state.node(node_id)
    return getattr(node, METRICS[node.kind])


def pushes_for(shocks: list[Shock], state: WorldState, turn: int | None = None) -> list[Push]:
    """Shocks made on ``turn`` (default: the state's) as pushes in absolute turns."""
    turn = state.turn if turn is None else turn
    pushes = []
    for shock in shocks:
        state.node(shock.node)  # raises KeyError on an unknown node
        start = turn + shock.start_turn
        pushes.append(
            Push(
                node=shock.node,
                delta=shock.delta,
                start=start,
                end=start + shock.duration_turns,
                hold=shock.hold,
            )
        )
    return pushes


# Makes, for one projected path, a function called once per turn after the first with the
# projected values (native units by node id) that returns the background pushes due that turn.
Outlook = Callable[[], Callable[[dict[str, float]], list[Shock]]]


class _Engine:
    """The graph's dynamics for one set of edge weights. ``step`` moves an ``EngineState``
    on by one turn: the one transition that real turns, forecasts and ``simulate`` share."""

    def __init__(self, state: WorldState, weights: dict[int, float] | None = None) -> None:
        edges = _propagating_edges(state)
        self.lagged: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
        self.instant: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for i, e in enumerate(edges):
            factor = (weights[i] if weights is not None else e.weight) * DAMPING
            if e.lag:
                self.lagged[e.target].append((e.source, factor, e.lag))
            else:
                self.instant[e.target].append((e.source, factor))
        self.memory = max([1, *(e.lag for e in edges)])
        self.instant_items = list(self.instant.items())
        self.nodes = list(state.node_ids())
        self.keep = {n: persistence(state, n) for n in self.nodes}
        self.sectors = set(state.sectors)
        self.scales = {n: scale(state, n) for n in self.nodes}
        self.bounds = {n: _bounds(state, n) for n in self.nodes}

    def scale(self, node_id: str, values: dict[str, float], level: float) -> float:
        """Native units per step. A sector's step is 5% of the output it would have without
        the deviation it carries (``level`` steps), which stays put while that deviation
        comes and goes, so a shock that fades leaves output where it found it."""
        if node_id in self.sectors and 1 + 0.05 * level > 0:
            return 0.05 * values[node_id] / (1 + 0.05 * level)
        return self.scales[node_id]

    def step(
        self,
        engine: _Carry,
        turn: int,
        share: float = 0.0,
        values: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Advance ``engine`` (in place) through absolute ``turn``. ``share`` is the job
        protection in force. With ``values`` (native levels, updated in place), every node
        stays within its bounds; without, the system is linear and unbounded.

        Returns this turn's deviation per node (standard steps) and its change in native
        units (empty without ``values``)."""
        keep, nodes, past = self.keep, self.nodes, engine.levels
        own = {n: v * keep[n] for n, v in engine.own.items()}
        for push in engine.pushes:
            if push.start <= turn < push.end:
                # A held push tops itself back up to delta after each turn's fade.
                held = push.hold and turn > push.start
                delta = push.delta * (1 - keep[push.node]) if held else push.delta
                own[push.node] = own.get(push.node, 0.0) + delta
        engine.pushes = [p for p in engine.pushes if p.end > turn + 1]

        prev = {n: series[-1] for n, series in past.items()}
        limits: dict[str, tuple[float, float]] = {}
        sizes: dict[str, float] = {}
        if values is not None:
            for n in nodes:
                size = sizes[n] = self.scale(n, values, prev.get(n, 0.0))
                low, high = self.bounds[n]
                if size and (low is not None or high is not None):
                    before = prev.get(n, 0.0)
                    limits[n] = (
                        -math.inf if low is None else before + (low - values[n]) / size,
                        math.inf if high is None else before + (high - values[n]) / size,
                    )

        driven: dict[str, float] = {}
        for n, inputs in self.lagged.items():
            push = 0.0
            for src, f, lag in inputs:
                series = past.get(src)
                if series is not None and len(series) >= lag:
                    push += f * series[-lag]
            if push:
                driven[n] = push

        def settle(n: str, push: float) -> tuple[float, float]:
            """The node's held push and its level, within its bounds."""
            if n == JOB_NODE and share and push > 0:
                push *= 1 - share
            level = own.get(n, 0.0) + push
            limit = limits.get(n)
            if limit is not None:
                level = limit[0] if level < limit[0] else limit[1] if level > limit[1] else level
            return push, level

        held: dict[str, float] = {}
        level: dict[str, float] = {}
        for n in nodes:
            held[n], level[n] = settle(n, driven.get(n, 0.0))
        instant = self.instant_items
        for _ in range(MAX_ITERATIONS):
            change = 0.0
            for n, inputs in instant:
                push = driven.get(n, 0.0)
                for src, f in inputs:
                    push += f * level[src]
                held[n], value = settle(n, push)
                gap = abs(value - level[n]) / max(1.0, abs(value))
                if gap > change:
                    change = gap
                level[n] = value
            if change < TOLERANCE:
                break
        else:
            raise PropagationError(
                f"lag-0 feedback did not settle within {MAX_ITERATIONS} iterations on turn "
                f"{turn} (last relative change {change:.2g}); the graph's lag-0 loops are unstable"
            )

        moves = {}
        kept_own = {}
        levels = {}
        for n in nodes:
            mine = own.get(n, 0.0)
            if n in limits and level[n] != mine + held[n]:
                # A bound stopped the node: what was cut off is gone, so the push left is the
                # one that holds it at the bound, and it fades back from there, not from past it.
                mine = level[n] - held[n]
            if abs(mine) > NEGLIGIBLE:
                kept_own[n] = mine
            if values is not None:
                move = (level[n] - prev.get(n, 0.0)) * sizes[n]
                if move:
                    moves[n] = move
                    values[n] += move
            series = past.get(n)
            series = [*series[len(series) + 1 - self.memory :], level[n]] if series else [level[n]]
            if any(abs(v) > NEGLIGIBLE for v in series):
                levels[n] = series
        engine.own = kept_own
        engine.levels = levels
        return level, moves


class _Carry:
    """``EngineState`` as plain containers, for the engine's inner loop."""

    def __init__(self, state: EngineState, pushes: list[Push] = ()) -> None:
        self.own = dict(state.own)
        self.levels = {n: list(v) for n, v in state.levels.items()}
        self.pushes = [*state.pushes, *pushes]

    def freeze(self) -> EngineState:
        return EngineState(own=self.own, levels=self.levels, pushes=self.pushes)


def step(
    state: WorldState, shocks: list[Shock], protection: dict[int, float] | None = None
) -> tuple[EngineState, dict[str, float]]:
    """One turn of the engine from what ``state`` carries, plus this turn's ``shocks``.

    Returns the engine to carry into the next turn and this turn's change to each node's
    primary metric, in native units, within its bounds. ``protection`` is the job protection
    in force by absolute turn (see ``job_protection``); None uses the state's own."""
    if protection is None:
        protection = state.job_protection
    engine = _Carry(state.engine, pushes_for(shocks, state))
    values = {n: _value(state, n) for n in state.node_ids()}
    _, moves = _Engine(state).step(engine, state.turn, protection.get(state.turn, 0.0), values)
    return engine.freeze(), moves


def in_flight(state: WorldState, node_id: str) -> float:
    """What effects already in flight move ``node_id`` this turn, native units, before
    anything new is decided."""
    queued = state.pending.get(state.turn, {}).get(node_id, 0.0)
    return queued + step(state, [])[1].get(node_id, 0.0)


def simulate(
    state: WorldState,
    shocks: list[Shock],
    horizon: int,
    weights: dict[int, float] | None = None,
    protection: dict[int, float] | None = None,
) -> dict[str, list[float]]:
    """What ``shocks`` alone do: standard-step deviations per node per turn, starting from a
    quiet world (nothing in flight) and ignoring bounds. ``explain`` uses it to split a turn
    between its causes.

    ``weights`` overrides edge weights by index into the propagating edge list.
    ``protection`` is the job protection in force by absolute turn (see
    ``job_protection``); None uses the state's own.
    """
    if protection is None:
        protection = state.job_protection
    model = _Engine(state, weights)
    engine = _Carry(EngineState(), pushes_for(shocks, state, 0))
    trajectory: dict[str, list[float]] = {n: [] for n in model.nodes}
    for t in range(horizon):
        level, _ = model.step(engine, t, protection.get(state.turn + t, 0.0))
        for n in model.nodes:
            trajectory[n].append(level[n])
    return trajectory


def propagate(
    state: WorldState,
    shocks: list[Shock],
    horizon: int = 12,
    k_draws: int = 200,
    seed: int = 0,
    protection: dict[int, float] | None = None,
    baseline: list[Shock] | None = None,
    outlook: Outlook | None = None,
) -> DeltaDistribution:
    """Monte Carlo forecast: the change in every node over the next ``horizon`` turns, mean
    and 10th/90th percentiles in native units.

    It runs the same turn as ``resolve`` (``step``): from what the state already has in
    flight, plus ``shocks`` made this turn, within every node's bounds. Each draw perturbs the
    edge weights for the turns ahead. With ``baseline`` (the shocks among ``shocks`` that
    come whatever the player decides), one more run at the central weights, with only those
    and what is in flight, fills ``DeltaDistribution.baseline``. ``outlook`` makes the pushes
    that later turns' background mechanisms add, from the projected values (see ``Outlook``).
    """
    if protection is None:
        protection = state.job_protection
    edges = _propagating_edges(state)
    draws: list[dict[str, list[float]]] = []
    for k in range(k_draws):
        rng = make_rng(seed, state.turn, f"mc:{k}")
        weights = {i: rng.gauss(e.weight, e.uncertainty) for i, e in enumerate(edges)}
        draws.append(_path(state, shocks, horizon, _Engine(state, weights), protection, outlook))
    start = list(draws[0]) if draws else list(state.node_ids())
    carried = (
        _path(state, baseline, horizon, _Engine(state), protection, outlook)
        if baseline is not None
        else {}
    )

    nodes = {}
    for node_id in start:
        mean, p10, p90 = [], [], []
        for t in range(horizon):
            values = [d[node_id][t] for d in draws]
            mean.append(fmean(values))
            if len(values) > 1:
                cuts = quantiles(values, n=10, method="inclusive")
                p10.append(cuts[0])
                p90.append(cuts[-1])
            else:
                p10.append(values[0])
                p90.append(values[0])
        nodes[node_id] = NodeForecast(mean=mean, p10=p10, p90=p90)
    return DeltaDistribution(horizon=horizon, k_draws=k_draws, nodes=nodes, baseline=carried)


def _path(
    state: WorldState,
    shocks: list[Shock],
    horizon: int,
    model: _Engine,
    protection: dict[int, float],
    outlook: Outlook | None = None,
) -> dict[str, list[float]]:
    """Cumulative native change per node per turn, from what ``state`` has in flight plus
    ``shocks`` made this turn, within bounds."""
    engine = _Carry(state.engine, pushes_for(shocks, state))
    start = {n: _value(state, n) for n in state.node_ids()}
    values = dict(start)
    path: dict[str, list[float]] = {n: [] for n in start}
    later = outlook() if outlook is not None else None
    for t in range(state.turn, state.turn + horizon):
        if later is not None and t > state.turn:
            engine.pushes += pushes_for(later(values), state, t)
        model.step(engine, t, protection.get(t, 0.0), values)
        for n, queued in state.pending.get(t, {}).items():
            low, high = model.bounds[n]
            v = values[n] + queued
            v = min(high, v) if high is not None else v
            values[n] = max(low, v) if low is not None else v
        for n in start:
            path[n].append(values[n] - start[n])
    return path


def apply_deltas(state: WorldState, deltas: dict[str, float]) -> WorldState:
    """Return a new state with native-unit deltas added to each node's primary metric."""
    new = state.snapshot()
    for node_id, delta in deltas.items():
        if not delta:
            continue
        node = new.node(node_id)
        field = METRICS[node.kind]
        value = getattr(node, field) + delta
        if field in ("approval", "support"):
            value = min(1.0, max(0.0, value))
        elif node.kind == NodeKind.SECTOR:
            value = max(0.0, value)
        elif node.kind == NodeKind.INDICATOR:
            if node.low is not None:
                value = max(node.low, value)
            if node.high is not None:
                value = min(node.high, value)
        setattr(node, field, value)
    return new
