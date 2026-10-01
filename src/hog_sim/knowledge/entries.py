"""What the knowledge store keeps: typed records of validated LLM output.

Scenarios are stored as SD-3's ``LibraryScenario`` (``origin="llm"``), so the offline
scenario library can use them as they are. The other kinds are defined here.
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from hog_sim.content.library import CATEGORIES, Provenance, slugify
from hog_sim.core.models import GraphChange, Model, Outcome, PolicyAction, Scenario

__all__ = [
    "GraphChangeEntry",
    "InterpretationEntry",
    "OutcomeEntry",
    "Provenance",
    "action_signature",
    "guess_category",
    "normalise_text",
    "scenario_id",
]


class InterpretationEntry(Model):
    """How Claude read one player response to one scenario."""

    scenario_id: str
    scenario_title: str
    response: str
    actions: list[PolicyAction]
    provenance: Provenance


class OutcomeEntry(Model):
    """The candidate outcomes Claude proposed for one response, and which was chosen."""

    scenario_id: str
    scenario_title: str
    affected_nodes: list[str]
    response: str
    actions: list[PolicyAction]
    candidates: list[Outcome]
    chosen: int
    provenance: Provenance

    @property
    def outcome(self) -> Outcome:
        return self.candidates[self.chosen]


class GraphChangeEntry(Model):
    """A graph change Claude proposed. ``applied`` when it was in the chosen outcome."""

    change: GraphChange
    status: Literal["applied", "proposed"]
    scenario_title: str
    provenance: Provenance


def normalise_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def scenario_id(scenario: Scenario) -> str:
    """The library id of a scenario, or a stable one derived from its text.

    Derived ids look like ``llm-<title-slug>-<8 hex>``, so the same generated scenario gets
    the same id when it is harvested and when it later comes back from the library.
    """
    existing = getattr(scenario, "id", None)
    if existing:
        return existing
    digest = hashlib.sha256(f"{scenario.title}\n{scenario.briefing}".encode()).hexdigest()[:8]
    return f"llm-{slugify(scenario.title)[:48].strip('-')}-{digest}"


def action_signature(actions: list[PolicyAction]) -> str:
    """What was done, ignoring how much: sorted ``kind:target:sign`` items."""
    items = set()
    for a in actions:
        sign = "+" if a.magnitude > 0 else "-" if a.magnitude < 0 else "0"
        items.add(f"{a.kind}:{a.target}:{sign}")
    return "|".join(sorted(items)) or "none"


# Checked in order against title and briefing; first match wins.
_CATEGORY_WORDS = [
    ("health", r"nhs|hospital|health|doctor|nurse|gp|pandemic|vaccin"),
    ("education", r"school|universit|teacher|pupil|student|exam"),
    ("crime_justice", r"crime|police|prison|court|judge|knife|riot"),
    ("immigration", r"migra|asylum|border|visa|small boats|refugee"),
    ("defence_security", r"defence|military|army|navy|troops|terror|security|nato|missile"),
    ("environment_disasters", r"flood|storm|climate|emission|drought|wildfire|heatwave|pollution"),
    ("party_scandal", r"scandal|leak|resign|backbench|rebel|whip|leadership|sleaze|donor"),
    ("media_technology", r"media|tabloid|press|broadcast|\bai\b|cyber|tech|online|social media"),
    ("energy", r"energy|\bgas\b|electric|\boil\b|nuclear|power station|bills"),
    ("housing", r"hous|rent|mortgage|landlord|homeless|planning"),
    ("foreign_affairs", r"trade|tariff|summit|ambassador|sanction|treaty|diplomat|foreign"),
]
_CATEGORY_NODES = [
    ("energy", ("indicator:energy_prices", "sector:energy")),
    ("housing", ("indicator:house_prices", "sector:housing", "group:young_renters")),
    ("foreign_affairs", ("country:",)),
]


def guess_category(scenario: Scenario) -> str:
    """The scenario's own category if it has one, else a deterministic guess from its text
    and affected nodes. Falls back to ``economy``."""
    own = getattr(scenario, "category", None)
    if own in CATEGORIES:
        return own
    text = f"{scenario.title} {scenario.briefing}".lower()
    for category, pattern in _CATEGORY_WORDS:
        if re.search(pattern, text):
            return category
    for category, prefixes in _CATEGORY_NODES:
        if any(n.startswith(p) for n in scenario.affected_nodes for p in prefixes):
            return category
    return "economy"
