"""Credibility: one score for how far voters and markets believe the government (D9).

Credibility runs from 0 to 1 and starts at ``NEUTRAL``. Each turn, in ``resolve()``:

* every U-turn (reversing a measure still running, E12) costs ``UTURN_LOSS``;
* every pledge broken (SD-5) costs ``BROKEN_PLEDGE_LOSS``;
* a crisis (an unscheduled scenario with urgency at least ``CRISIS_URGENCY``) met with a
  real measure on something it affects earns ``CRISIS_HANDLED``, and one met with words
  alone, nothing, or measures elsewhere costs ``CRISIS_DUCKED``;
* then the score drifts ``SETTLE`` of the way back toward ``NEUTRAL``, so old sins fade.

What it does:

* speeches carry ``speech_weight`` of their stated size (half at 0, one and a half at 1),
  applied by the limits so the logged actions are what ran;
* every group's target approval moves by ``APPROVAL_PER_POINT`` times its distance from
  ``NEUTRAL``;
* the capital refill is scaled by ``capital_factor`` (``policy.capital``).

At the starting world (credibility 0.5) none of this moves anything.
"""

from __future__ import annotations

from collections.abc import Sequence

from hog_sim.core.models import Model, PolicyAction, Scenario

NEUTRAL = 0.5
UTURN_LOSS = 0.06
BROKEN_PLEDGE_LOSS = 0.08
CRISIS_URGENCY = 0.75
CRISIS_HANDLED = 0.04
CRISIS_DUCKED = 0.05
SETTLE = 0.05  # share of the gap to NEUTRAL closed each turn
APPROVAL_PER_POINT = 0.1  # target approval per unit of credibility away from NEUTRAL
# Kinds that are words, not measures: they do not count as meeting a crisis.
_WORDS = {"communicate", "do_nothing"}


def speech_weight(credibility: float) -> float:
    """Share of its stated size a speech carries at this credibility."""
    return 0.5 + credibility


def capital_factor(credibility: float) -> float:
    """How credibility scales the capital refill: 1 at ``NEUTRAL``."""
    return 0.5 + credibility


def approval_shift(credibility: float) -> float:
    """What credibility adds to every group's target approval."""
    return APPROVAL_PER_POINT * (credibility - NEUTRAL)


def is_crisis(scenario: Scenario) -> bool:
    """An urgent turn the calendar did not bring: the Budget is not a crisis."""
    return scenario.urgency >= CRISIS_URGENCY and scenario.source != "scheduled"


def meets_crisis(scenario: Scenario, actions: Sequence[PolicyAction]) -> bool:
    """Whether the package does something beyond talking about what the crisis affects
    (anything at all, when the scenario names nothing)."""
    affected = set(scenario.affected_nodes)
    return any(
        a.kind not in _WORDS and a.magnitude and (not affected or a.target in affected)
        for a in actions
    )


class CredibilityChange(Model):
    """What moved credibility this turn, for the step and for the player."""

    u_turns: int = 0
    broken_pledges: int = 0
    crisis: bool = False
    handled: bool = False

    @property
    def delta(self) -> float:
        change = -UTURN_LOSS * self.u_turns - BROKEN_PLEDGE_LOSS * self.broken_pledges
        if self.crisis:
            change += CRISIS_HANDLED if self.handled else -CRISIS_DUCKED
        return change

    def reasons(self) -> list[str]:
        out = []
        if self.u_turns:
            out.append(f"{self.u_turns} U-turn(s) {-UTURN_LOSS * self.u_turns * 100:+.0f}")
        if self.broken_pledges:
            loss = -BROKEN_PLEDGE_LOSS * self.broken_pledges * 100
            out.append(f"{self.broken_pledges} broken pledge(s) {loss:+.0f}")
        if self.crisis:
            out.append(
                f"crisis met with action {CRISIS_HANDLED * 100:+.0f}"
                if self.handled
                else f"crisis met with words or nothing {-CRISIS_DUCKED * 100:+.0f}"
            )
        return out


def step_credibility(credibility: float, change: CredibilityChange) -> float:
    """This turn's credibility after ``change``, settled a little toward ``NEUTRAL``."""
    moved = min(1.0, max(0.0, credibility + change.delta))
    return moved + SETTLE * (NEUTRAL - moved)
