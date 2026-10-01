"""Interpreter: the player's free text -> PolicyActions (or a clarifying question)."""

from __future__ import annotations

from pydantic import Field

from hog_sim.core.models import Model, PolicyAction, Scenario
from hog_sim.llm.client import LLMClient, ModelConfig, structured_call
from hog_sim.llm.prompts import interpreter as prompt
from hog_sim.llm.scenario_gen import scenario_text
from hog_sim.llm.summary import StateSummary


class Interpretation(Model):
    actions: list[PolicyAction] = Field(default_factory=list)
    unmapped: list[str] = Field(
        default_factory=list, description="Measures in the response that fit no node"
    )
    clarifying_question: str | None = None


def check_interpretation(result: Interpretation, summary: StateSummary) -> list[str]:
    problems = []
    unknown = sorted({a.target for a in result.actions if a.target not in summary.catalogue})
    if unknown:
        problems.append(f"unknown target ids {unknown}; use ids from the briefing")
    for i, action in enumerate(result.actions):
        if action.kind == "do_nothing" and action.target != summary.player_country:
            problems.append(f"actions[{i}]: do_nothing must target {summary.player_country}")
        if action.kind in ("regulate", "deregulate") and action.magnitude < 0:
            problems.append(f"actions[{i}]: {action.kind} magnitude must be 0..1")
    if not result.actions and not result.clarifying_question:
        problems.append("return at least one action, or a clarifying_question")
    if result.actions and result.clarifying_question:
        problems.append("ask a clarifying_question only when returning no actions")
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
    )
