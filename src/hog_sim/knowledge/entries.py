"""What the knowledge store keeps: typed records of validated LLM output.

Scenarios are stored as SD-3's ``LibraryScenario`` (``origin="llm"``), so the offline
scenario library can use them as they are. The other kinds are defined here.
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import Field

from hog_sim.content.library import CATEGORIES, Provenance, slugify
from hog_sim.core.models import Delivery, GraphChange, Model, Outcome, PolicyAction, Scenario

__all__ = [
    "GraphChangeEntry",
    "InterpretationEntry",
    "OutcomeEntry",
    "Provenance",
    "ReuseContext",
    "action_signature",
    "guess_category",
    "normalise_text",
    "reuse_context",
    "scenario_id",
]


class InterpretationEntry(Model):
    """How Claude read one player response to one scenario."""

    scenario_id: str
    scenario_title: str
    response: str
    actions: list[PolicyAction]
    provenance: Provenance


class ReuseContext(Model):
    """What an outcome's narrative may describe beyond the actions and sackings (E23): how the
    package was delivered, the pledges it broke and the measures blocked or forced through.
    A stored outcome is reused only on a turn with the same context."""

    consulted: list[str] = Field(default_factory=list)
    speed: str = "immediate"
    broken_pledges: list[str] = Field(default_factory=list)
    blocked: list[str] = Field(default_factory=list)

    @property
    def plain(self) -> bool:
        return self == ReuseContext()

    def key(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]


LIMIT_PREFIXES = ("Blocked:", "Forced through:")


def reuse_context(
    delivery: Delivery | None, broken_pledges: list[str], limits: list[str] | None
) -> ReuseContext:
    """The context to match stored outcomes on. Limit notes count only when a measure was
    blocked or forced through, and only by kind and target: the reasons carry numbers that
    change from turn to turn."""
    delivery = delivery or Delivery()
    blocked = {note.split(" (", 1)[0] for note in limits or [] if note.startswith(LIMIT_PREFIXES)}
    return ReuseContext(
        consulted=sorted(set(delivery.consulted)),
        speed=delivery.speed,
        broken_pledges=sorted(set(broken_pledges)),
        blocked=sorted(blocked),
    )


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
    sacked: list[str] = Field(
        default_factory=list,
        description="Cast members sacked that turn; candidates' narratives may mention them, so "
        "they are reused only on a turn that sacks the same people",
    )
    context: ReuseContext | None = Field(
        None,
        description="Delivery, broken pledges and blocked measures that turn (E23); entries "
        "from before it are reused only on a plain turn",
    )

    def fits(self, context: ReuseContext) -> bool:
        return self.context == context if self.context is not None else context.plain

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


# Size buckets for stored-outcome reuse (EB-13): a stored outcome is reused only for actions of
# roughly the same size, so a small spend does not reuse outcomes written for a huge one. An
# action's bucket is the largest bound its |magnitude| does not exceed.
SIZE_BUCKETS = (("small", 0.3), ("medium", 0.7), ("large", float("inf")))


def size_bucket(magnitude: float) -> str:
    size = abs(magnitude)
    return next(name for name, bound in SIZE_BUCKETS if size <= bound)


# Duration buckets (finding 9 of the 2026-10-12 review): a one-turn measure never reuses
# outcomes written for a policy that runs all parliament. Bucketed like size, by the largest
# bound the duration does not exceed.
DURATION_BUCKETS = (("once", 1), ("months", 6), ("year", 12), ("long", float("inf")))


def duration_bucket(turns: int) -> str:
    return next(name for name, bound in DURATION_BUCKETS if turns <= bound)


def action_signature(actions: list[PolicyAction]) -> str:
    """What was done, for how long and roughly how much: sorted
    ``kind:target:sign:instrument:duration:size`` items, where size is the bucket of the
    summed magnitude of actions sharing the rest. ``instrument`` is whether the tax or
    spending itself went up or down (``+``/``-``, ``0`` for other kinds); aimed at an
    indicator it can differ from the magnitude's sign (see ``PolicyAction.instrument``)."""
    totals: dict[str, float] = {}
    for a in actions:
        sign = "+" if a.magnitude > 0 else "-" if a.magnitude < 0 else "0"
        instrument = "0"
        if a.kind in ("tax", "spend"):
            instrument = {1: "+", -1: "-"}.get(int(a.instrument_sign), "0")
        key = f"{a.kind}:{a.target}:{sign}:{instrument}:{duration_bucket(a.duration_turns)}"
        totals[key] = totals.get(key, 0.0) + abs(a.magnitude)
    items = [f"{key}:{size_bucket(total)}" for key, total in totals.items()]
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
