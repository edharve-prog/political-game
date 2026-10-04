"""Scenario generator: state summary -> this turn's Scenario."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from pydantic import Field

from hog_sim.core.models import Category, Model, Scenario, SideIssue, Storyline
from hog_sim.llm.client import LLMClient, ModelConfig, structured_call
from hog_sim.llm.prompts import scenario as prompt
from hog_sim.llm.summary import StateSummary, resolve_id
from hog_sim.world.storylines import new_storyline_id


class StakeholderPosition(Model):
    node: str
    stance: float = Field(ge=-1, le=1, description="-1 opposes government acting, 1 demands it")
    statement: str


class GeneratedScenario(Scenario):
    stakeholder_positions: list[StakeholderPosition] = Field(default_factory=list)


class SideIssueDraft(Model):
    title: str
    category: Category
    briefing: str
    affected_nodes: list[str]
    urgency: float = Field(ge=0, le=1)
    storyline: str = Field(description="Id of the open storyline this continues, or 'new'")


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
    secondary: list[SideIssueDraft]
    characters: list[str] = Field(
        default_factory=list, description="Ids of the people from the briefing involved"
    )


# Variety is asked for in the prompt, not enforced here: a failed check would stop the turn.
RECENT_TURNS = 8


def check_draft(
    draft: ScenarioDraft,
    catalogue: dict[str, str],
    storylines: Sequence[Storyline] = (),
    final: Sequence[str] = (),
    open_new: bool = False,
    cast: Collection[str] = (),
) -> list[str]:
    """``final`` are the storylines in their final stage; ``open_new`` asks for a new lead
    storyline (story SD-9). ``cast`` are the ids of the people it may involve (SD-4)."""
    problems = []
    ids = [s.id for s in storylines]
    items = [draft, *draft.secondary]
    if any(i.storyline != "new" and i.storyline not in ids for i in items):
        problems.append(f"each storyline must be 'new' or one of {ids}")
    continued = [i.storyline for i in items if i.storyline != "new"]
    if len(set(continued)) != len(continued):
        problems.append("each storyline may appear only once in the in-tray")
    side = [i.storyline for i in draft.secondary if i.storyline != "new"]
    if len(side) > 1:
        problems.append("at most one secondary item may continue an open storyline")
    ending = sorted(set(side) & set(final))
    if ending:
        problems.append(f"{ending} are in their final stage and can only return as the lead")
    if open_new and draft.storyline != "new":
        problems.append("the last leads all continued old storylines: open a 'new' one this turn")
    if not 1 <= len(draft.secondary) <= 3:
        problems.append("give 1-3 secondary items")
    titles = [i.title.strip().lower() for i in items]
    if len(set(titles)) != len(titles):
        problems.append("in-tray items need distinct titles")
    if not draft.affected_nodes:
        problems.append("affected_nodes is empty")
    if not 2 <= len(draft.suggested_options) <= 4:
        problems.append("give 2-4 suggested_options")
    if not 2 <= len(draft.stakeholder_positions) <= 5:
        problems.append("give 2-5 stakeholder_positions")
    unknown = [n for n in draft.affected_nodes if n not in catalogue]
    unknown += [s.node for s in draft.stakeholder_positions if s.node not in catalogue]
    unknown += [n for i in draft.secondary for n in i.affected_nodes if n not in catalogue]
    if any(not i.affected_nodes for i in draft.secondary):
        problems.append("every secondary item needs affected_nodes")
    if unknown:
        problems.append(f"unknown node ids {sorted(set(unknown))}; use ids from the briefing")
    if len(set(draft.affected_nodes)) != len(draft.affected_nodes):
        problems.append("affected_nodes has duplicates")
    strangers = sorted(set(draft.characters) - set(cast))
    if strangers:
        problems.append(f"unknown people {strangers}; use person ids from the briefing")
    if len(draft.characters) > 3:
        problems.append("involve at most 3 people")
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
    final: Sequence[str] = (),
    open_new: bool = False,
) -> GeneratedScenario:
    """``storylines`` are the open ones the draft may continue; ``storylines_prompt`` shows
    them with their history (``world.storylines.storylines_text``). ``final`` and
    ``open_new`` apply the SD-9 lifespan rules (see ``check_draft``)."""
    config = config or ModelConfig()
    recent = list(recent or [])[-RECENT_TURNS:]
    storylines = list(storylines or [])
    draft = structured_call(
        client,
        output_type=ScenarioDraft,
        system=prompt.SYSTEM,
        prompt=prompt.render(
            summary.to_prompt(), recent_text(recent), storylines_prompt, open_new=open_new
        ),
        model=config.scenario_model,
        prompt_version=prompt.VERSION,
        effort=config.scenario_effort,
        max_tokens=config.max_tokens,
        max_attempts=config.max_attempts,
        check=lambda d: check_draft(
            d, summary.catalogue, storylines, final, open_new, summary.cast
        ),
        repair=lambda d: _repair_people(d, summary.cast),
    )
    fields = draft.model_dump(exclude={"secondary"})
    fields["storyline"] = _storyline_id(draft.storyline, draft.title, summary.turn)
    secondary = [
        SideIssue(
            **{
                **item.model_dump(),
                "storyline": _storyline_id(item.storyline, item.title, summary.turn),
            }
        )
        for item in draft.secondary
    ]
    return GeneratedScenario(source="generated", secondary=secondary, **fields)


def _repair_people(draft: ScenarioDraft, cast: Collection[str]) -> ScenarioDraft:
    draft.characters = [resolve_id(c, cast) for c in draft.characters]
    return draft


def _storyline_id(storyline: str, title: str, turn: int) -> str:
    return new_storyline_id(title, turn) if storyline == "new" else storyline


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
    if scenario.secondary:
        lines += ["", "Also in the in-tray this turn (the leader may act on any of them):"]
        lines += [f"- {i.title}: {i.briefing}" for i in scenario.secondary]
    if scenario.suggested_options:
        lines += ["", "Options on the table:"] + [f"- {o}" for o in scenario.suggested_options]
    return "\n".join(lines)
