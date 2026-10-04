"""Interpreter: the player's free text -> PolicyActions (or a clarifying question)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from hog_sim.core.models import ActionKind, Delivery, Model, Pledge, PolicyAction, Scenario
from hog_sim.llm.client import LLMClient, ModelConfig, structured_call
from hog_sim.llm.prompts import interpreter as prompt
from hog_sim.llm.scenario_gen import scenario_text
from hog_sim.llm.summary import StateSummary, resolve_id


class PledgeDraft(Model):
    """A promise in the response (SD-5): it rules out ``kind`` in ``direction`` on
    ``target``, or on anything when ``target`` is null."""

    text: str = Field(description="The promise, a few words in the leader's voice")
    kind: ActionKind
    direction: Literal["up", "down", "any"]
    target: str | None = None
    groups: list[str] = Field(default_factory=list, description="Group ids who care most")

    def to_pledge(self) -> Pledge:
        return Pledge(**self.model_dump())


class Interpretation(Model):
    actions: list[PolicyAction] = Field(default_factory=list)
    unmapped: list[str] = Field(
        default_factory=list, description="Measures in the response that fit no node"
    )
    clarifying_question: str | None = None
    delivery: Delivery = Field(default_factory=Delivery)
    pledges: list[PledgeDraft] = Field(
        default_factory=list, description="Promises about future turns (SD-5)"
    )


def repair_interpretation(result: Interpretation, summary: StateSummary) -> Interpretation:
    """Fix slips with one obvious correction instead of retrying (RB-9): bare ids get their
    prefix, ``do_nothing`` beside real actions is dropped, and so is a clarifying question
    asked alongside actions."""
    ids = summary.catalogue
    for action in result.actions:
        action.target = resolve_id(action.target, ids)
    result.delivery.consulted = [resolve_id(c, ids) for c in result.delivery.consulted]
    for pledge in result.pledges:
        pledge.target = resolve_id(pledge.target, ids) if pledge.target else None
        pledge.groups = [resolve_id(g, ids) for g in pledge.groups]
    if any(a.kind != "do_nothing" for a in result.actions):
        result.actions = [a for a in result.actions if a.kind != "do_nothing"]
    if result.actions:
        result.clarifying_question = None
    return result


def check_interpretation(result: Interpretation, summary: StateSummary) -> list[str]:
    problems = []
    unknown = sorted({a.target for a in result.actions if a.target not in summary.catalogue})
    if unknown:
        problems.append(f"unknown target ids {unknown}; use ids from the briefing")
    for i, action in enumerate(result.actions):
        if action.kind == "do_nothing" and action.target != summary.player_country:
            problems.append(f"actions[{i}]: do_nothing must target {summary.player_country}")
        indicator = action.target.startswith("indicator:")
        if action.kind in ("regulate", "deregulate") and action.magnitude < 0 and not indicator:
            problems.append(f"actions[{i}]: {action.kind} magnitude must be 0..1")
    if not result.actions and not result.clarifying_question:
        problems.append("return at least one action, or a clarifying_question")
    unknown_consulted = sorted(set(result.delivery.consulted) - set(summary.catalogue))
    if unknown_consulted:
        problems.append(f"delivery.consulted has unknown ids {unknown_consulted}")
    if result.actions and result.clarifying_question:
        problems.append("ask a clarifying_question only when returning no actions")
    for i, pledge in enumerate(result.pledges):
        if pledge.target is not None and pledge.target not in summary.catalogue:
            problems.append(f"pledges[{i}]: unknown target {pledge.target!r}")
        bad = [g for g in pledge.groups if not g.startswith("group:") or g not in summary.catalogue]
        if bad:
            problems.append(f"pledges[{i}]: groups must be group ids from the briefing, not {bad}")
    return problems


def interpret(
    text: str,
    summary: StateSummary,
    client: LLMClient,
    scenario: Scenario | None = None,
    config: ModelConfig | None = None,
) -> Interpretation:
    config = config or ModelConfig()
    return structured_call(
        client,
        output_type=Interpretation,
        system=prompt.SYSTEM,
        prompt=prompt.render(
            summary.to_prompt(), scenario_text(scenario) if scenario else None, text
        ),
        model=config.interpret_model,
        prompt_version=prompt.VERSION,
        effort=config.interpret_effort,
        max_tokens=config.max_tokens,
        max_attempts=config.max_attempts,
        check=lambda r: check_interpretation(r, summary),
        repair=lambda r: repair_interpretation(r, summary),
    )
