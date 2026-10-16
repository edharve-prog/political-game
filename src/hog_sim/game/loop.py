"""Turn orchestration.

A turn has two halves:

* **decide** (may call an LLM): interpret the response, run the engine forecast, generate
  candidate outcomes and select one. Everything it produces is logged.
* **resolve** (pure, deterministic): apply the chosen outcome to the world. It uses only
  logged data, so a game can be replayed exactly from its log without calling any LLM.

Graph changes in the chosen outcome (validated LLM proposals, see ``world/changes.py``) are
applied in resolve too, after the turn's effects land, so they shape the next turn's
propagation and replay reproduces them.

Before anything is forecast, ``policy.limits.constrain`` applies feasibility, diminishing
returns and the political-capital budget in every mode; the record keeps the actions that
were actually applied and notes saying why any were cut.

Storylines (issues that run across turns) move in resolve too: see ``world/storylines.py``.
So do pledges (SD-5): an action a kept pledge rules out breaks it and costs approval, and the
turn's new pledges are added afterwards. See ``policy/pledges.py``.
The cast's loyalty moves there as well (SD-4): see ``world/cast.py``. So does the opposition
(O9), which picks a stance and may fight a by-election every quarter: see
``world/opposition.py``.

Institutions settle back toward where they started the game: each turn closes
``INSTITUTION_SETTLE`` of the gap in their support. A rebellion still costs the government its
majority for a few turns, but no longer for the rest of the game (EB-14: in the 30-turn
review the Commons fell below a majority at turn 13 and blocked every spend and tax after).
The legislature settles toward its starting support plus ``COMMONS_POLLS`` times the change in
national approval since the start: backbenchers grow restless when the polls fall (EB-8).
Settling moves only the part of support the engine does not carry (direct edits, ministers
leaving, the polls' drift); the engine fades its own deviation (a whip, a rebellion's shock)
by itself. Settling both reverted the same ground twice, so a whip ended below never whipping
and a rebellion recovered past the quiet path (E7).

The engine carries its own state between turns (``WorldState.engine``: each node's fading
push, its recent deviations and the shocks still running), and each turn ``resolve`` moves it
on by one step (``world.propagation.step``). So a policy's lagged tail keeps arriving after the
turn it was made, a policy keeps pushing for as long as it runs however long that is, later
graph changes and job protection act on effects already in flight, and a node held at a bound
fades back from the bound. The forecast (``forecast``) runs the same step from the same state,
with the same shocks for this turn and the same rules after it (ministers leaving, institutions
settling, the debt, the actions' graph changes), so what it shows is what resolve would do with
no outcome; what it cannot project is listed in ``UNPROJECTED``. With ``hidden_world`` on (D1)
resolve plays on the game's hidden edge weights (``world.propagation.true_weights``) while the
forecast keeps the central ones, so the forecast is the government's best guess, not the truth.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

from pydantic import Field

from hog_sim.core.config import GameConfig, make_rng
from hog_sim.core.models import (
    Delivery,
    Model,
    Outcome,
    Pledge,
    PolicyAction,
    Reading,
    Scenario,
    Shock,
)
from hog_sim.core.state import WorldState
from hog_sim.forecasting.selection import select
from hog_sim.game.interfaces import Forecaster, Interpreter, ScenarioSource
from hog_sim.game.records import TurnRecord
from hog_sim.policy.capital import available, next_capital
from hog_sim.policy.credibility import (
    CredibilityChange,
    is_crisis,
    meets_crisis,
    step_credibility,
)
from hog_sim.policy.feasibility import LEGISLATURE_ID, forcing_shocks
from hog_sim.policy.limits import (
    Constrained,
    capital_cost,
    constrain,
    reverses,
    still_running,
)
from hog_sim.policy.pledges import apply_pledges
from hog_sim.population.popularity import (
    national_approval,
    policy_events,
    run_election,
    step_approval,
)
from hog_sim.world.cast import DEPARTURE_COST, apply_cast, sackable
from hog_sim.world.changes import action_changes, apply_graph_changes
from hog_sim.world.debt import DebtOutlook, interest_shocks, step_debt
from hog_sim.world.events import outcome_shocks
from hog_sim.world.opposition import step_opposition
from hog_sim.world.propagation import (
    DeltaDistribution,
    Projection,
    actions_to_shocks,
    apply_deltas,
    job_protection,
    propagate,
    scale,
    step,
    true_weights,
)
from hog_sim.world.standing import (
    SENTIMENT_SETTLE,
    crisis_shocks,
    sentiment_shocks,
    settle_sentiment,
)
from hog_sim.world.storylines import advance_storylines

INSTITUTION_SETTLE = 0.2  # share of the gap to the starting support closed each turn
COMMONS_POLLS = 0.5  # legislature support gained or lost per point of national approval


def _homes(state: WorldState, start: WorldState) -> dict[str, float]:
    """Where each institution's support settles toward this turn."""
    polls = national_approval(state) - national_approval(start) if state.groups else 0.0
    homes = {}
    for inst_id in state.institutions:
        origin = start.institutions.get(inst_id)
        if origin is not None:
            home = origin.support + (COMMONS_POLLS * polls if inst_id == LEGISLATURE_ID else 0.0)
            homes[inst_id] = min(1.0, max(0.0, home))
    return homes


def _settled(support: float, home: float, deviation: float) -> float:
    """``support`` after settling toward ``home``: only the part the engine does not carry
    (``deviation``, in native units, which the engine fades itself) closes its gap."""
    own = support - deviation
    return support + (home - own) * INSTITUTION_SETTLE


def _deviation(state: WorldState, node_id: str) -> float:
    """What the engine carries on ``node_id`` this turn, in native units."""
    levels = state.engine.levels.get(node_id)
    return levels[-1] * scale(state, node_id) if levels else 0.0


def _settle_institutions(state: WorldState, start: WorldState) -> WorldState:
    new = state.snapshot()
    for inst_id, home in _homes(state, start).items():
        inst = new.institutions[inst_id]
        settled = _settled(inst.support, home, _deviation(state, inst_id))
        inst.support = min(1.0, max(0.0, settled))
    return new


def _record_history(state: WorldState) -> WorldState:
    """Append this turn's starting indicator values and group approvals to their histories.

    Scenario generation reads these to see what is moving (stressed indicators, groups
    turning against the government), so the next scenario can grow out of the last outcome.
    """
    new = state.snapshot()
    for ind in new.indicators.values():
        ind.history.append(ind.value)
    for group in new.groups.values():
        group.history.append(group.approval)
    return new


def turn_shocks(
    start: WorldState,
    state: WorldState,
    scenario: Scenario,
    actions: list[PolicyAction],
    config: GameConfig,
) -> list[Shock]:
    """Everything that pushes the world this turn except the chosen outcome: the actions, the
    scenario, and the background mechanisms (business confidence, crises abroad, the interest
    bill, bills forced through). ``resolve`` applies them; the forecast starts from them."""
    return (
        actions_to_shocks(actions, state)
        + scenario.shocks
        + forcing_shocks(actions, state, config.role)
        + background_shocks(start, state, config)
    )


def background_shocks(start: WorldState, state: WorldState, config: GameConfig) -> list[Shock]:
    """This turn's pushes that come whatever the player decides: business confidence, crises
    abroad and the interest bill."""
    return (
        sentiment_shocks(state) + crisis_shocks(state, config.seed) + interest_shocks(state, start)
    )


def outlook_shocks(state: WorldState, horizon: int) -> list[Shock]:
    """Business confidence's pushes in the turns after this one, as it settles back, for the
    forecast. The interest bill is projected turn by turn (``DebtOutlook``); crises abroad are
    left out, since they are drawn turn by turn."""
    return [
        shock.model_copy(
            update={"delta": shock.delta * (1 - SENTIMENT_SETTLE) ** k, "start_turn": k}
        )
        for shock in sentiment_shocks(state)
        for k in range(1, horizon)
    ]


# What the forecast does not project, for anyone reading it (``DeltaDistribution.unprojected``).
UNPROJECTED = [
    "group approval: the forecast shows no change, and approval is judged from the indicators",
    "crises abroad in later turns, which are drawn turn by turn",
    "the Commons' drift with the polls after this turn: approval is held where it is",
    "the opposition's quarterly stance and by-elections, which turn on approval",
    "the chosen outcome, which is not known yet",
]
# With a hidden world (D1) the forecast is a guess about the economy, not a copy of it.
HIDDEN_WORLD_NOTE = (
    "how this economy really responds: the forecast spreads its guesses over each link's "
    "uncertainty, and the world plays out one of them"
)


def _departures(
    state: WorldState,
    scenario: Scenario,
    actions: list[PolicyAction],
    delivery: Delivery | None,
    sacked: Sequence[str],
) -> int:
    """Ministers who leave this turn (sacked, or resigning over it), as ``resolve`` finds."""
    after = apply_cast(state, state.turn, actions, delivery, scenario.characters, sacked)
    return sum(
        1
        for cid, character in state.characters.items()
        if character.active and not after.characters[cid].active
    )


def _projection(
    start: WorldState,
    state: WorldState,
    scenario: Scenario,
    actions: list[PolicyAction],
    config: GameConfig,
    delivery: Delivery | None,
    sacked: Sequence[str],
    shocks: list[Shock],
) -> Projection:
    """``resolve``'s turn as a course for the forecast: the job protection the actions give,
    ministers leaving, institutions settling, the debt repricing and the actions' lasting
    graph changes, which carry the effects from the next turn on."""
    left = _departures(state, scenario, actions, delivery, sacked)
    homes = _homes(state, start)

    sizes = {inst_id: scale(state, inst_id) for inst_id in homes}

    def settle(turn: int, values: dict[str, float], levels: dict[str, float]) -> None:
        if turn == state.turn and left and LEGISLATURE_ID in values:
            values[LEGISLATURE_ID] = max(0.0, values[LEGISLATURE_ID] - left * DEPARTURE_COST)
        for inst_id, home in homes.items():
            deviation = levels.get(inst_id, 0.0) * sizes[inst_id]
            settled = _settled(values[inst_id], home, deviation)
            values[inst_id] = min(1.0, max(0.0, settled))

    changes = action_changes(state, actions)
    return Projection(
        shocks,
        job_protection(state, actions),
        outlook=lambda: DebtOutlook(state, start),
        settle=settle,
        graph=apply_graph_changes(state, changes, trusted=True) if changes else None,
    )


def forecast(
    start: WorldState,
    state: WorldState,
    scenario: Scenario,
    actions: list[PolicyAction],
    config: GameConfig,
    delivery: Delivery | None = None,
    sacked: Sequence[str] = (),
) -> DeltaDistribution:
    """The engine's forecast for ``actions``: the same turn ``resolve`` runs, before the
    outcome is known, then ``config.horizon`` turns of what is already in flight, with the
    same rules after each step (ministers leaving, institutions settling, the debt, the
    actions' graph changes). What it leaves out is listed in ``unprojected``.

    Its ``baseline`` is the same crisis with none of this turn's decisions (no actions, no
    sackings, only the job protection already running), each draw run twice on the same
    edge weights, so ``response_at`` is what the decisions alone do."""
    later = outlook_shocks(state, config.horizon)
    background = background_shocks(start, state, config) + later
    return propagate(
        state,
        _projection(
            start,
            state,
            scenario,
            actions,
            config,
            delivery,
            sacked,
            turn_shocks(start, state, scenario, actions, config) + later,
        ),
        config.horizon,
        config.k_draws,
        config.seed,
        baseline=_projection(
            start, state, scenario, [], config, None, (), scenario.shocks + background
        ),
        unprojected=[*UNPROJECTED, HIDDEN_WORLD_NOTE] if config.hidden_world else UNPROJECTED,
    )


def resolve(
    start: WorldState,
    state: WorldState,
    scenario: Scenario,
    actions: list[PolicyAction],
    outcome: Outcome,
    config: GameConfig,
    pledges: Sequence[Pledge] = (),
    delivery: Delivery | None = None,
    sacked: Sequence[str] = (),
) -> WorldState:
    """Apply one turn's chosen outcome and advance the clock. Deterministic.

    ``pledges`` are the promises made this turn (SD-5); ``delivery`` and ``sacked`` (cast
    ids) move the cast's loyalty (SD-4)."""
    shocks = turn_shocks(start, state, scenario, actions, config)
    shocks += outcome_shocks(outcome, state)
    cover = job_protection(state, actions)
    weights = (
        true_weights(state, config.seed, config.hidden_world_draw, start)
        if config.hidden_world
        else None
    )
    engine, moves = step(state, shocks, protection=cover, weights=weights)

    # Saves from before the engine carried its own state queued effects here; drain them.
    pending = {t: dict(d) for t, d in state.pending.items() if t > state.turn}
    now = dict(moves)
    for node_id, change in state.pending.get(state.turn, {}).items():
        now[node_id] = now.get(node_id, 0.0) + change
    new = apply_deltas(_record_history(state), now)
    new.engine = engine
    new.pending = pending
    new.job_protection = {t: share for t, share in cover.items() if t > state.turn}
    new.events = [*new.events, *policy_events(actions, state), *outcome.approval_events]
    new = apply_pledges(new, state.turn, actions, pledges)
    new = apply_cast(new, state.turn, actions, delivery, scenario.characters, sacked, played=state)
    new = settle_sentiment(new)  # before this turn's changes, so a new mood is felt in full
    new = apply_graph_changes(new, action_changes(new, actions), trusted=True)
    new = apply_graph_changes(new, outcome.graph_changes)
    new = _settle_institutions(new, start)
    new = step_debt(new, start)
    new = step_approval(new, reference=start)
    new = advance_storylines(new, state.turn, scenario, outcome, actions, played=state)
    new = step_opposition(new, state.turn, config.seed)
    new.credibility = step_credibility(
        state.credibility, credibility_change(new, state, scenario, actions)
    )
    new.measures = still_running(state.measures, actions)
    new.turn = state.turn + 1
    unspent = available(state, config) - capital_cost(
        actions, state, config.role, running=state.measures
    )
    new.capital = next_capital(new, unspent, config)
    return new


def credibility_change(
    new: WorldState, state: WorldState, scenario: Scenario, actions: Sequence[PolicyAction]
) -> CredibilityChange:
    """What moved credibility on ``state``'s turn (D9): its U-turns, the pledges it broke,
    and whether a crisis was met with a real measure."""
    return CredibilityChange(
        u_turns=sum(reverses(a, state.measures) for a in actions),
        broken_pledges=sum(p.broken_turn == state.turn for p in new.pledges),
        crisis=is_crisis(scenario),
        handled=meets_crisis(scenario, actions),
    )


def not_possible(unmapped: Sequence[str]) -> list[str]:
    """Limit notes for measures the interpreter could not map to anything in the game, so the
    outcome prompt treats them like blocked measures rather than as enacted (E21)."""
    return [f"{m}: not possible, nothing in the game can carry it out" for m in unmapped]


class StaleProposal(RuntimeError):
    """A proposal no longer fits the game: it was made on another turn or against a world that
    has changed since, or its actions are not what this turn's limits allow. Propose again."""


def state_key(state: WorldState) -> str:
    """A fingerprint of everything in ``state``, to tell whether it has changed."""
    return hashlib.sha256(state.to_json().encode()).hexdigest()


class Proposal(Model):
    """A turn's interpreted actions before they are committed.

    A proposal belongs to the turn and the world it was checked against (``turn``,
    ``state_key``): ``Game.commit`` refuses it anywhere else, so it cannot be committed twice
    or after the world has moved, and it re-runs the limits on ``submitted`` to make sure
    ``actions`` is still what they allow."""

    response: str
    requested: list[PolicyAction]
    actions: list[PolicyAction]
    submitted: list[PolicyAction] = Field(
        default_factory=list, description="The actions the limits were applied to"
    )
    turn: int = -1
    state_key: str = ""
    notes: list[str] = Field(default_factory=list)
    delivery: Delivery = Field(default_factory=Delivery)
    pledges: list[Pledge] = Field(default_factory=list, description="Promises made (SD-5)")
    sacked: list[str] = Field(default_factory=list, description="Cast ids sacked (SD-4)")
    unmapped: list[str] = Field(
        default_factory=list, description="Measures the interpreter could not map (E21)"
    )
    readings: list[Reading] = Field(
        default_factory=list, description="How each requested action was read (D6)"
    )

    def reading(self, index: int) -> Reading | None:
        """The reading for ``actions[index]``: the first unused one of the same kind and
        target, or None (the player edited it, or the interpreter gave none)."""
        used: set[int] = set()
        for i, action in enumerate(self.actions):
            j = next(
                (j for j, r in enumerate(self.readings) if j not in used and r.fits(action)),
                None,
            )
            if i == index:
                return self.readings[j] if j is not None else None
            if j is not None:
                used.add(j)
        return None


class Game:
    """One game in play.

    The next scenario is fetched only when something asks for it (``game.scenario``), never
    inside ``commit``: a finished turn can be saved before the scenario source is called, so a
    failed or interrupted scenario request cannot lose it. ``on_scenario(turn, scenario)`` is
    called whenever this turn's scenario is fetched or changed, so it can be saved and a
    resumed game shows the same scenario (pass it back to ``resume`` as ``scenario``).
    """

    def __init__(
        self,
        config: GameConfig,
        start: WorldState,
        scenarios: ScenarioSource,
        interpreter: Interpreter,
        forecaster: Forecaster,
        on_scenario: Callable[[int, Scenario], None] | None = None,
    ) -> None:
        if start.turn >= config.election_turn:
            # The game would be over before it began, and the election never held.
            raise ValueError(
                f"the world starts at turn {start.turn}, at or past the election "
                f"(turn {config.election_turn})"
            )
        self.config = config
        self.start = start
        self.state = start
        self.scenarios = scenarios
        self.interpreter = interpreter
        self.forecaster = forecaster
        self.on_scenario = on_scenario
        self.history: list[TurnRecord] = []
        self._scenario: Scenario | None = None

    @classmethod
    def resume(
        cls,
        config: GameConfig,
        start: WorldState,
        records: list[TurnRecord],
        scenarios: ScenarioSource,
        interpreter: Interpreter,
        forecaster: Forecaster,
        scenario: Scenario | None = None,
        on_scenario: Callable[[int, Scenario], None] | None = None,
    ) -> Game:
        """Continue a saved game. ``scenario`` is the scenario saved for the turn being
        resumed, if any; without it the source is asked for a new one."""
        game = cls(config, start, scenarios, interpreter, forecaster, on_scenario)
        game.history = list(records)
        if records:
            game.state = records[-1].state_after
        game._scenario = scenario
        return game

    @property
    def scenario(self) -> Scenario:
        """This turn's scenario, fetched from the source the first time it is needed."""
        if self._scenario is None:
            if self.over:
                raise RuntimeError("the game is over")
            self.scenario = self.scenarios.next_scenario(self.state, self.history)
        assert self._scenario is not None
        return self._scenario

    @scenario.setter
    def scenario(self, scenario: Scenario) -> None:
        self._scenario = scenario
        if self.on_scenario is not None:
            self.on_scenario(self.state.turn, scenario)

    @property
    def over(self) -> bool:
        return self.state.turn >= self.config.election_turn

    def add_options(self, options: list[str]) -> None:
        """Add options to this turn's numbered list (advisers, story RB-3). The turn doesn't
        advance, and resolve never reads options, so replay is unaffected."""
        merged = list(dict.fromkeys([*self.scenario.suggested_options, *options]))
        self.scenario = self.scenario.model_copy(update={"suggested_options": merged})

    def propose(self, response: str) -> Proposal:
        """Interpret ``response`` and apply this turn's limits, without changing anything.

        May raise ``NeedsClarification``. Show the result, let the player edit it, then
        ``commit`` it (backlog story RB-2).
        """
        if self.over:
            raise RuntimeError("the game is over")
        requested = self.interpreter.interpret(response, self.state, self.scenario)
        delivery = getattr(self.interpreter, "last_delivery", lambda: None)() or Delivery()
        pledges = getattr(self.interpreter, "last_pledges", lambda: [])()
        sacked = getattr(self.interpreter, "last_sacked", lambda: [])()
        readings = getattr(self.interpreter, "last_readings", lambda: [])()
        unmapped = getattr(self.interpreter, "last_unmapped", lambda: [])()
        proposal = self._limit(
            response,
            requested,
            delivery=delivery,
            pledges=pledges,
            sacked=sacked,
            unmapped=unmapped,
        )
        return proposal.model_copy(update={"readings": list(readings)})

    def propose_actions(self, response: str, actions: list[PolicyAction]) -> Proposal:
        """A proposal from a ready-made action list (a saved package, story RB-5), checked
        against this turn's limits without calling the interpreter."""
        if self.over:
            raise RuntimeError("the game is over")
        return self._limit(response, actions)

    def revise(
        self,
        proposal: Proposal,
        actions: list[PolicyAction],
        sacked: Sequence[str] | None = None,
    ) -> Proposal:
        """The same proposal with the player's edited actions, re-checked against the limits.
        ``sacked`` replaces its sackings, so the player can call one off (E18)."""
        revised = self._limit(
            proposal.response,
            actions,
            proposal.requested,
            proposal.delivery,
            proposal.pledges,
            proposal.sacked if sacked is None else sacked,
            proposal.unmapped,
        )
        return revised.model_copy(update={"readings": proposal.readings})

    def _limit(
        self,
        response: str,
        actions: list[PolicyAction],
        requested: list[PolicyAction] | None = None,
        delivery: Delivery | None = None,
        pledges: Sequence[Pledge] = (),
        sacked: Sequence[str] = (),
        unmapped: Sequence[str] = (),
    ) -> Proposal:
        limited = self._constrain(actions)
        return Proposal(
            response=response,
            requested=list(requested if requested is not None else actions),
            actions=limited.actions,
            submitted=list(actions),
            turn=self.state.turn,
            state_key=state_key(self.state),
            notes=limited.notes,
            delivery=delivery or Delivery(),
            pledges=list(pledges),
            sacked=[c for c in sacked if c in sackable(self.state)],
            unmapped=list(unmapped),
        )

    def _constrain(self, actions: list[PolicyAction]) -> Constrained:
        cfg = self.config
        return constrain(
            actions,
            self.state,
            [r.actions for r in self.history],
            cfg.role,
            available(self.state, cfg),
        )

    def check(self, proposal: Proposal) -> None:
        """Raise ``StaleProposal`` unless ``proposal`` was made for this turn, against the
        world as it is now, and its actions are exactly what this turn's limits allow."""
        if proposal.turn != self.state.turn:
            raise StaleProposal(
                f"this proposal was made on turn {proposal.turn}, and it is now turn "
                f"{self.state.turn}"
            )
        if proposal.state_key != state_key(self.state):
            raise StaleProposal("the world has changed since this proposal was checked")
        if self._constrain(proposal.submitted).actions != proposal.actions:
            raise StaleProposal("this proposal's actions are not what this turn's limits allow")

    def play_turn(self, response: str) -> TurnRecord:
        return self.commit(self.propose(response))

    def commit(self, proposal: Proposal) -> TurnRecord:
        """Forecast, pick an outcome and resolve the turn with the proposal's actions.

        Raises ``StaleProposal`` if the proposal does not belong to this turn and this world
        (see ``check``): propose again and show the player the result."""
        if self.over:
            raise RuntimeError("the game is over")
        self.check(proposal)
        state, scenario, cfg = self.state, self.scenario, self.config
        actions = proposal.actions
        engine = forecast(
            self.start, state, scenario, actions, cfg, proposal.delivery, proposal.sacked
        )
        candidates = self.forecaster.forecast(
            state,
            scenario,
            actions,
            engine,
            delivery=proposal.delivery,
            limits=proposal.notes + not_possible(proposal.unmapped),
            sacked=proposal.sacked,
        )
        chosen = select(
            candidates,
            cfg.selection_mode,
            make_rng(cfg.seed, state.turn, "select"),
            seed=cfg.seed,
            turn=state.turn,
            group_weights={gid: g.population_share * g.turnout for gid, g in state.groups.items()},
        )

        new = resolve(
            self.start,
            state,
            scenario,
            actions,
            candidates[chosen],
            cfg,
            proposal.pledges,
            proposal.delivery,
            proposal.sacked,
        )
        election = run_election(new) if new.turn == cfg.election_turn else None
        record = TurnRecord(
            turn=state.turn,
            scenario=scenario,
            response=proposal.response,
            actions=actions,
            candidates=candidates,
            chosen=chosen,
            state_after=new,
            election=election,
            requested_actions=proposal.requested,
            notes=proposal.notes,
            delivery=proposal.delivery,
            pledges=proposal.pledges,
            sacked=proposal.sacked,
            unmapped=proposal.unmapped,
        )
        self.state = new
        self.history.append(record)
        self._scenario = None  # fetched when next needed, after the caller has saved the record
        return record


def replay(start: WorldState, config: GameConfig, records: list[TurnRecord]) -> WorldState:
    """Rebuild the final state from a turn log, without calling any LLM."""
    state = start
    for record in records:
        state = resolve(
            start,
            state,
            record.scenario,
            record.actions,
            record.outcome,
            config,
            record.pledges,
            record.delivery,
            record.sacked,
        )
    return state
