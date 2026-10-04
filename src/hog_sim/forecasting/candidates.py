"""LLM-backed outcome forecasting: propose candidates, judge them, score and convert.

``LLMForecaster`` satisfies the game loop's ``Forecaster`` protocol:

1. The engine's Monte Carlo forecast for the actions is summarised for the prompt.
2. One call proposes ``n_candidates`` outcomes (``CandidateDraft``).
3. A separate judge call estimates each candidate's probability (optional, for cost).
4. ``scoring.combine`` blends engine consistency, the judge and base rates.
5. Each candidate becomes an ``Outcome``: approval effects become ``ApprovalEvent``s and
   knock-on events become ``Shock``s; lasting graph changes are checked against the state
   (``world/changes.py``) before the candidate is accepted. The engine's numbers stay
   authoritative; a candidate's claimed indicator shifts are only kept for display and logs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import Field

from hog_sim.core.models import (
    ApprovalEvent,
    Delivery,
    EdgeKind,
    GraphChange,
    Model,
    Outcome,
    PolicyAction,
    Scenario,
    Shock,
)
from hog_sim.core.state import WorldState
from hog_sim.forecasting.scoring import (
    CandidateScores,
    EventTag,
    ScoreWeights,
    base_rate,
    combine,
    consistency,
)
from hog_sim.llm.client import Effort, LLMClient, structured_call
from hog_sim.llm.prompts import judge as judge_prompt
from hog_sim.llm.prompts import outcomes as outcomes_prompt
from hog_sim.llm.scenario_gen import scenario_text
from hog_sim.llm.summary import StateSummary, relevant_links, resolve_id, summarise_state
from hog_sim.policy.feasibility import Role
from hog_sim.policy.pledges import broken_by
from hog_sim.world.changes import validate_graph_changes
from hog_sim.world.propagation import DeltaDistribution
from hog_sim.world.storylines import final_storylines

CLAIM_TURN = 2  # candidates claim indicator changes three turns out (index 2)
MAX_GROUP_EFFECT = 0.1
MAX_SHOCK_STEPS = 1.0


class Shift(Model):
    node: str
    change: float


class GroupEffect(Model):
    group: str
    change: float = Field(ge=-MAX_GROUP_EFFECT, le=MAX_GROUP_EFFECT)


class NewShock(Model):
    node: str
    steps: float = Field(ge=-MAX_SHOCK_STEPS, le=MAX_SHOCK_STEPS)


class GraphChangeDraft(Model):
    """A lasting change to the world graph. Every field is required for structured outputs;
    the ones a kind doesn't use are null."""

    kind: Literal["edge_weight", "add_edge", "node_attr"]
    source: str | None
    target: str | None
    edge_kind: EdgeKind | None
    node: str | None
    attr: str | None
    delta: float
    lag: int
    reason: str

    def to_change(self) -> GraphChange:
        return GraphChange(**{**self.model_dump(), "lag": max(0, self.lag)})


class CandidateDraft(Model):
    title: str
    narrative: str
    indicator_shifts: list[Shift]
    group_effects: list[GroupEffect]
    new_shocks: list[NewShock]
    graph_changes: list[GraphChangeDraft]
    event_tags: list[EventTag]
    resolves_storyline: bool
    self_probability: float = Field(ge=0, le=1)


class CandidateSet(Model):
    candidates: list[CandidateDraft]


class Judgement(Model):
    index: int
    probability: float = Field(ge=0, le=1)
    reason: str


class Verdict(Model):
    judgements: list[Judgement]


# (state, scenario, actions) -> precedent lines for the prompt. See knowledge/recall.py.
Recall = Callable[[WorldState, Scenario, list[PolicyAction]], list[str]]


class ForecastConfig(Model):
    outcome_model: str = "claude-opus-5-5"
    outcome_effort: Effort = "medium"
    judge_model: str = "claude-opus-5-5"
    judge_effort: Effort = "medium"
    n_candidates: int = Field(4, ge=2, le=8)
    use_judge: bool = True
    weights: ScoreWeights = Field(default_factory=ScoreWeights)
    max_tokens: int = 16000
    max_attempts: int = Field(3, ge=1)


# --- Prompt text -----------------------------------------------------------


def actions_text(actions: list[PolicyAction]) -> str:
    if not actions:
        return "- none"
    return "\n".join(
        f"- {a.kind} {a.target} magnitude {a.magnitude:+.2f} for {a.duration_turns} turn(s)"
        + (f": {a.rationale}" if a.rationale else "")
        for a in actions
    )


def engine_text(state: WorldState, engine: DeltaDistribution, turn: int = CLAIM_TURN) -> str:
    turn = min(turn, engine.horizon - 1)
    lines = [f"Expected change by {turn + 1} turns from now (10-90% range):"]
    for ind in state.indicators.values():
        f = engine.nodes[ind.id]
        if abs(f.mean[turn]) < 1e-3 and abs(f.p90[turn] - f.p10[turn]) < 1e-3:
            continue
        lines.append(
            f"- {ind.id}: {f.mean[turn]:+.2f} {ind.unit} ({f.p10[turn]:+.2f} to {f.p90[turn]:+.2f})"
        )
    if len(lines) == 1:
        lines.append("- no material change in any indicator")
    return "\n".join(lines)


def candidates_text(candidates: list[CandidateDraft]) -> str:
    blocks = []
    for i, c in enumerate(candidates):
        shifts = ", ".join(f"{s.node} {s.change:+.2f}" for s in c.indicator_shifts) or "none"
        blocks.append(
            f"[{i}] {c.title}\n{c.narrative}\nClaimed shifts: {shifts}\n"
            f"Events: {', '.join(c.event_tags)}"
        )
    return "\n\n".join(blocks)


# --- Semantic checks -------------------------------------------------------


def repair_candidates(result: CandidateSet, summary: StateSummary) -> CandidateSet:
    """Give bare ids (``business``) their prefix (``group:business``) instead of retrying."""
    indicators = [i.id for i in summary.indicators]
    groups = [g.id for g in summary.groups]
    for c in result.candidates:
        for s in c.indicator_shifts:
            s.node = resolve_id(s.node, indicators)
        for g in c.group_effects:
            g.group = resolve_id(g.group, groups)
        for n in c.new_shocks:
            n.node = resolve_id(n.node, summary.catalogue)
    return result


def check_candidates(
    result: CandidateSet, summary: StateSummary, n: int, state: WorldState | None = None
) -> list[str]:
    problems = []
    if len(result.candidates) != n:
        problems.append(f"give exactly {n} candidates")
    catalogue = summary.catalogue
    indicators = {i.id for i in summary.indicators}
    groups = {g.id for g in summary.groups}
    for i, c in enumerate(result.candidates):
        bad = [s.node for s in c.indicator_shifts if s.node not in indicators]
        bad += [g.group for g in c.group_effects if g.group not in groups]
        bad += [s.node for s in c.new_shocks if s.node not in catalogue]
        if bad:
            problems.append(f"candidates[{i}]: unknown ids {sorted(set(bad))}")
        if not c.event_tags:
            problems.append(f"candidates[{i}]: event_tags is empty; use ['none']")
        elif "none" in c.event_tags and len(c.event_tags) > 1:
            problems.append(f"candidates[{i}]: 'none' cannot be combined with other event tags")
        if state is not None and c.graph_changes:
            changes = [g.to_change() for g in c.graph_changes]
            problems += [f"candidates[{i}].{p}" for p in validate_graph_changes(state, changes)]
    titles = [c.title.strip().lower() for c in result.candidates]
    if len(set(titles)) != len(titles):
        problems.append("candidates must be distinct")
    return problems


def check_verdict(result: Verdict, n: int) -> list[str]:
    indices = sorted(j.index for j in result.judgements)
    if indices != list(range(n)):
        return [f"judge every candidate exactly once, indices 0..{n - 1}"]
    return []


# --- Forecaster ------------------------------------------------------------


class LLMForecaster:
    def __init__(
        self,
        client: LLMClient,
        role: Role = "prime_minister",
        config: ForecastConfig | None = None,
        recall: Recall | None = None,
    ) -> None:
        self.client = client
        self.role = role
        self.config = config or ForecastConfig()
        self.recall = recall

    def forecast(
        self,
        state: WorldState,
        scenario: Scenario,
        actions: list[PolicyAction],
        engine: DeltaDistribution,
        delivery: Delivery | None = None,
        limits: list[str] | None = None,
        sacked: list[str] | None = None,
    ) -> list[Outcome]:
        cfg = self.config
        summary = summarise_state(state, self.role)
        summary.links = relevant_links(
            state, [*scenario.affected_nodes, *(a.target for a in actions)]
        )
        if self.recall is not None:
            summary.precedents = self.recall(state, scenario, actions)
        context = outcomes_prompt.render(
            summary.to_prompt(),
            scenario_text(scenario),
            actions_text(actions),
            engine_text(state, engine),
            cfg.n_candidates,
            delivery.describe() if delivery else "",
            limits or [],
            ending=scenario.storyline in final_storylines(state),
            broken_pledges=[p.text for p in broken_by(state, actions)],
            sacked=[
                f"{state.characters[c].name}, {state.characters[c].role}"
                for c in sacked or []
                if c in state.characters
            ],
        )
        drafts = structured_call(
            self.client,
            output_type=CandidateSet,
            system=outcomes_prompt.SYSTEM,
            prompt=context,
            model=cfg.outcome_model,
            prompt_version=outcomes_prompt.VERSION,
            effort=cfg.outcome_effort,
            max_tokens=cfg.max_tokens,
            max_attempts=cfg.max_attempts,
            check=lambda r: check_candidates(r, summary, cfg.n_candidates, state),
            repair=lambda r: repair_candidates(r, summary),
        ).candidates

        judged = self._judge(context, drafts) if cfg.use_judge else None
        turn = min(CLAIM_TURN, engine.horizon - 1)
        scores = combine(
            [
                CandidateScores(
                    consistency=consistency(
                        {s.node: s.change for s in d.indicator_shifts}, engine, turn
                    ),
                    judge=judged[i] if judged else d.self_probability,
                    base_rate=base_rate(list(d.event_tags), delivery),
                    self_reported=d.self_probability,
                )
                for i, d in enumerate(drafts)
            ],
            cfg.weights,
        )
        return [to_outcome(d, s) for d, s in zip(drafts, scores, strict=True)]

    def _judge(self, context: str, drafts: list[CandidateDraft]) -> list[float]:
        cfg = self.config
        verdict = structured_call(
            self.client,
            output_type=Verdict,
            system=judge_prompt.SYSTEM,
            prompt=judge_prompt.render(context, candidates_text(drafts)),
            model=cfg.judge_model,
            prompt_version=judge_prompt.VERSION,
            effort=cfg.judge_effort,
            max_tokens=cfg.max_tokens,
            max_attempts=cfg.max_attempts,
            check=lambda r: check_verdict(r, len(drafts)),
        )
        by_index = {j.index: j.probability for j in verdict.judgements}
        return [by_index[i] for i in range(len(drafts))]


def to_outcome(draft: CandidateDraft, scores: CandidateScores) -> Outcome:
    effects = {g.group: g.change for g in draft.group_effects if g.change}
    return Outcome(
        narrative=f"{draft.title}. {draft.narrative}",
        indicator_deltas={s.node: s.change for s in draft.indicator_shifts},
        events=[t for t in draft.event_tags if t != "none"],
        approval_events=[ApprovalEvent(name=draft.title, group_effects=effects)] if effects else [],
        shocks=[Shock(node=s.node, delta=s.steps) for s in draft.new_shocks if s.steps],
        graph_changes=[g.to_change() for g in draft.graph_changes],
        resolves_storyline=draft.resolves_storyline,
        probability=scores.probability,
        scores=scores.model_dump(exclude={"probability"}),
    )
