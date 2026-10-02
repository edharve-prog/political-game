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

Effects that land in later turns are kept in ``WorldState.pending`` (absolute turn ->
node -> native change), so a policy's lagged tail keeps arriving after the turn it was made.
"""

from __future__ import annotations

from pydantic import Field

from hog_sim.core.config import GameConfig, make_rng
from hog_sim.core.models import Model, Outcome, PolicyAction, Scenario
from hog_sim.core.state import WorldState
from hog_sim.forecasting.selection import select
from hog_sim.game.interfaces import Forecaster, Interpreter, ScenarioSource
from hog_sim.game.records import TurnRecord
from hog_sim.policy.limits import constrain
from hog_sim.population.popularity import policy_events, run_election, step_approval
from hog_sim.world.changes import action_changes, apply_graph_changes
from hog_sim.world.events import event_shocks
from hog_sim.world.propagation import actions_to_shocks, apply_deltas, propagate, scale, simulate
from hog_sim.world.storylines import advance_storylines


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


def resolve(
    start: WorldState,
    state: WorldState,
    scenario: Scenario,
    actions: list[PolicyAction],
    outcome: Outcome,
    config: GameConfig,
) -> WorldState:
    """Apply one turn's chosen outcome and advance the clock. Deterministic."""
    shocks = (
        actions_to_shocks(actions, state)
        + scenario.shocks
        + outcome.shocks
        + event_shocks(outcome.events, state)
    )
    trajectory = simulate(state, shocks, config.horizon)

    pending = {t: dict(d) for t, d in state.pending.items()}
    for node_id, series in trajectory.items():
        s = scale(state, node_id)
        previous = 0.0
        for offset, level in enumerate(series):
            step = (level - previous) * s
            previous = level
            if step:
                slot = pending.setdefault(state.turn + offset, {})
                slot[node_id] = slot.get(node_id, 0.0) + step

    now = pending.pop(state.turn, {})
    new = apply_deltas(_record_history(state), now)
    new.pending = pending
    new.events = [*new.events, *policy_events(actions), *outcome.approval_events]
    new = apply_graph_changes(new, action_changes(new, actions))
    new = apply_graph_changes(new, outcome.graph_changes)
    new = step_approval(new, reference=start)
    new = advance_storylines(new, state.turn, scenario, outcome)
    new.turn = state.turn + 1
    return new


class Proposal(Model):
    """A turn's interpreted actions before they are committed."""

    response: str
    requested: list[PolicyAction]
    actions: list[PolicyAction]
    notes: list[str] = Field(default_factory=list)


class Game:
    def __init__(
        self,
        config: GameConfig,
        start: WorldState,
        scenarios: ScenarioSource,
        interpreter: Interpreter,
        forecaster: Forecaster,
    ) -> None:
        self.config = config
        self.start = start
        self.state = start
        self.scenarios = scenarios
        self.interpreter = interpreter
        self.forecaster = forecaster
        self.history: list[TurnRecord] = []
        self.scenario = scenarios.next_scenario(start, [])

    @classmethod
    def resume(
        cls,
        config: GameConfig,
        start: WorldState,
        records: list[TurnRecord],
        scenarios: ScenarioSource,
        interpreter: Interpreter,
        forecaster: Forecaster,
    ) -> Game:
        game = cls(config, start, scenarios, interpreter, forecaster)
        if records:
            game.history = list(records)
            game.state = records[-1].state_after
            if not game.over:
                game.scenario = scenarios.next_scenario(game.state, game.history)
        return game

    @property
    def over(self) -> bool:
        return self.state.turn >= self.config.election_turn

    def propose(self, response: str) -> Proposal:
        """Interpret ``response`` and apply this turn's limits, without changing anything.

        May raise ``NeedsClarification``. Show the result, let the player edit it, then
        ``commit`` it (backlog story RB-2).
        """
        if self.over:
            raise RuntimeError("the game is over")
        requested = self.interpreter.interpret(response, self.state, self.scenario)
        return self._limit(response, requested)

    def revise(self, proposal: Proposal, actions: list[PolicyAction]) -> Proposal:
        """The same proposal with the player's edited actions, re-checked against the limits."""
        return self._limit(proposal.response, actions, proposal.requested)

    def _limit(
        self,
        response: str,
        actions: list[PolicyAction],
        requested: list[PolicyAction] | None = None,
    ) -> Proposal:
        cfg = self.config
        limited = constrain(
            actions,
            self.state,
            [r.actions for r in self.history],
            cfg.role,
            cfg.capital_per_turn,
        )
        return Proposal(
            response=response,
            requested=list(requested if requested is not None else actions),
            actions=limited.actions,
            notes=limited.notes,
        )

    def play_turn(self, response: str) -> TurnRecord:
        return self.commit(self.propose(response))

    def commit(self, proposal: Proposal) -> TurnRecord:
        """Forecast, pick an outcome and resolve the turn with the proposal's actions."""
        if self.over:
            raise RuntimeError("the game is over")
        state, scenario, cfg = self.state, self.scenario, self.config
        actions = proposal.actions
        shocks = actions_to_shocks(actions, state) + scenario.shocks
        engine = propagate(state, shocks, cfg.horizon, cfg.k_draws, cfg.seed)
        candidates = self.forecaster.forecast(state, scenario, actions, engine)
        chosen = select(candidates, cfg.selection_mode, make_rng(cfg.seed, state.turn, "select"))

        new = resolve(self.start, state, scenario, actions, candidates[chosen], cfg)
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
        )
        self.state = new
        self.history.append(record)
        if not self.over:
            self.scenario = self.scenarios.next_scenario(new, self.history)
        return record


def replay(start: WorldState, config: GameConfig, records: list[TurnRecord]) -> WorldState:
    """Rebuild the final state from a turn log, without calling any LLM."""
    state = start
    for record in records:
        state = resolve(start, state, record.scenario, record.actions, record.outcome, config)
    return state
