"""What gets logged each turn."""

from __future__ import annotations

from pydantic import Field

from hog_sim.core.models import Delivery, Model, Outcome, PolicyAction, Scenario
from hog_sim.core.state import WorldState
from hog_sim.population.popularity import ElectionResult


class TurnRecord(Model):
    turn: int
    scenario: Scenario
    response: str
    actions: list[PolicyAction]
    candidates: list[Outcome]
    chosen: int
    state_after: WorldState
    election: ElectionResult | None = None
    requested_actions: list[PolicyAction] = Field(
        default_factory=list,
        description="The interpretation before limits (empty in saves from before Project 16)",
    )
    notes: list[str] = Field(
        default_factory=list, description="What limited the player's actions this turn"
    )
    delivery: Delivery = Field(
        default_factory=Delivery, description="How the response was delivered (RB-4)"
    )

    @property
    def outcome(self) -> Outcome:
        return self.candidates[self.chosen]
