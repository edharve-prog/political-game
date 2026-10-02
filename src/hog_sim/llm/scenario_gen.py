"""Scenario generator: state summary -> this turn's Scenario."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from hog_sim.core.models import Category, Model, Scenario, Storyline
from hog_sim.llm.client import LLMClient, ModelConfig, structured_call
from hog_sim.llm.prompts import scenario as prompt
from hog_sim.llm.summary import StateSummary
from hog_sim.world.storylines import new_storyline_id


class StakeholderPosition(Model):
    node: str
    stance: float = Field(ge=-1, le=1, description="-1 opposes government acting, 1 demands it")
    statement: str


class GeneratedScenario(Scenario):
    stakeholder_positions: list[StakeholderPosition] = Field(default_factory=list)


class ScenarioDraft(Model):
    """What the model writes. ``source`` is not the model's to choose, so it is set here."""

    title: str
    category: Category
    briefing: str
    affected_nodes: list[str]
    urgency: float = Field(ge=0, le=1)
    # List lengths are checked in check_draft: structured outputs only accept minItems 0 or 1.
    suggested_options: list[str]
    stakeholder_positions: list[StakeholderPosition]
    storyline: str = Field(description="Id of the open storyline this continues, or 'new'")


# Variety is asked for in the prompt, not enforced here: a failed check would stop the turn.
RECENT_TURNS = 8


def check_draft(
    draft: ScenarioDraft, catalogue: dict[str, str], storylines: Sequence[Storyline] = ()
) -> list[str]:
    problems = []
    ids = [s.id for s in storylines]
    if draft.storyline != "new" and draft.storyline not in ids:
        problems.append(f"storyline must be 'new' or one of {ids}")
    if not draft.affected_nodes:
        problems.append("affected_nodes is empty")
    if not 2 <= len(draft.suggested_options) <= 4:
        problems.append("give 2-4 suggested_options")
    if not 2 <= len(draft.stakeholder_positions) <= 5:
        problems.append("give 2-5 stakeholder_positions")
    unknown = [n for n in draft.affected_nodes if n not in catalogue]
    unknown += [s.node for s in draft.stakeholder_positions if s.node not in catalogue]
    if unknown:
        problems.append(f"unknown node ids {sorted(set(unknown))}; use ids from the briefing")
    if len(set(draft.affected_nodes)) != len(draft.affected_nodes):
        problems.append("affected_nodes has duplicates")
    if len(draft.title.split()) > 15:
        problems.append("title is too long; keep it under 12 words")
    return problems


def generate_scenario(
    summary: StateSummary,
    client: LLMClient,
    config: ModelConfig | None = None,
    recent: list[Scenario] | None = None,
    storylines: list[Storyline] | None = None,
    storylines_prompt: str = "",
) -> GeneratedScenario:
    """``storylines`` are the open ones the draft may continue; ``storylines_prompt`` shows
    them with their history (``world.storylines.storylines_text``)."""
    config = config or ModelConfig()
    recent = list(recent or [])[-RECENT_TURNS:]
    storylines = list(storylines or [])
    draft = structured_call(
        client,
        output_type=ScenarioDraft,
        system=prompt.SYSTEM,
        prompt=prompt.render(summary.to_prompt(), recent_text(recent), storylines_prompt),
        model=config.scenario_model,
        prompt_version=prompt.VERSION,
        effort=config.scenario_effort,
        max_tokens=config.max_tokens,
        max_attempts=config.max_attempts,
        check=lambda d: check_draft(d, summary.catalogue, storylines),
    )
    fields = draft.model_dump()
    if draft.storyline == "new":
        fields["storyline"] = new_storyline_id(draft.title, summary.turn)
    return GeneratedScenario(source="generated", **fields)


def recent_text(recent: list[Scenario]) -> str:
    if not recent:
        return ""
    lines = ["Recent scenarios (most recent last):"]
    lines += [f"- [{s.category or 'uncategorised'}] {s.title}" for s in recent]
    return "\n".join(lines)


def scenario_text(scenario: Scenario) -> str:
    """Plain-text rendering used when a scenario is passed back into a prompt."""
    lines = [scenario.title, "", scenario.briefing]
    if scenario.storyline:
        lines += ["", "This is part of an ongoing storyline."]
    if scenario.suggested_options:
        lines += ["", "Options on the table:"] + [f"- {o}" for o in scenario.suggested_options]
    return "\n".join(lines)
