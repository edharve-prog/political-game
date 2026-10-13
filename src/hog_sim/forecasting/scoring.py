"""Plausibility scoring for candidate outcomes.

Each candidate gets three scores, each in (0, 1]:

* **consistency**: how well its claimed indicator shifts fit the engine's Monte Carlo
  distribution. Each claim is treated as a draw from a normal fitted to the engine's
  10th/90th percentiles; the score is the geometric mean of exp(-z^2 / 2) over its claims,
  with each |z| capped at ``Z_CAP``. Without the cap one far-off claim scored like 1e-25
  and consistency alone picked the outcome, overriding the judge (the 30-turn review,
  EB-13); with it the worst score is about 0.011, a strong but not decisive penalty.
  An indicator the engine expects to move materially but the candidate leaves out counts as
  a claim of no change, so saying nothing is not a free pass.
* **judge**: an independent LLM estimate of its probability (``forecasting/candidates.py``).
* **base_rate**: how common its tagged events are in practice (``FITTED_RATES``: the
  ``BASE_RATES`` placeholders updated with historical episodes, ``forecasting/base_rates.py``).
  Several tags all have to happen: the rarest bounds the chance, and each further tag lowers
  it a little, since events cluster but are not certain to come together. A quiet outcome is
  less likely when the scenario is urgent or the actions meet resistance.

``combine`` blends them log-linearly with ``ScoreWeights`` and normalises to probabilities.
"""

from __future__ import annotations

import math
from typing import Literal

from hog_sim.core.models import Delivery, Model
from hog_sim.forecasting.base_rates import HISTORICAL, fit_base_rates
from hog_sim.world.propagation import DeltaDistribution

EventTag = Literal[
    "none",
    "market_selloff",
    "market_rally",
    "strike",
    "protest",
    "backbench_rebellion",
    "legal_challenge",
    "foreign_retaliation",
    "foreign_praise",
    "media_backlash",
    "media_praise",
    "business_investment",
    "capital_flight",
]

# Rough chance that a policy response of this game's scale sets off each kind of event
# within a few months. Hand-set placeholders: the prior that FITTED_RATES updates.
BASE_RATES: dict[str, float] = {
    "none": 0.5,
    "market_selloff": 0.08,
    "market_rally": 0.08,
    "strike": 0.15,
    "protest": 0.2,
    "backbench_rebellion": 0.15,
    "legal_challenge": 0.07,
    "foreign_retaliation": 0.1,
    "foreign_praise": 0.15,
    "media_backlash": 0.35,
    "media_praise": 0.25,
    "business_investment": 0.12,
    "capital_flight": 0.04,
}

# The rates the game uses: placeholders updated with historical episodes (D7).
FITTED_RATES: dict[str, float] = fit_base_rates(BASE_RATES, HISTORICAL)

# 10th to 90th percentile of a normal spans 2 * 1.2816 standard deviations.
_P10_P90_SPAN = 2 * 1.2816
_FLOOR = 1e-6
Z_CAP = 3.0  # beyond 3 spreads a claim is simply "far off"; see the module docstring


class ScoreWeights(Model):
    consistency: float = 0.4
    judge: float = 0.4
    base_rate: float = 0.2


class CandidateScores(Model):
    consistency: float
    judge: float
    base_rate: float
    self_reported: float
    probability: float = 0.0


def _spread(mean: float, p10: float, p90: float) -> float:
    # Nodes the engine is certain about still tolerate small claims: at least 5% of the
    # expected move, or a tenth of a unit.
    return max((p90 - p10) / _P10_P90_SPAN, 0.05 * abs(mean), 0.1)


def consistency(claims: dict[str, float], engine: DeltaDistribution, turn: int) -> float:
    """Fit of claimed native-unit changes at ``turn`` to the engine's distribution.

    Indicators the engine moves by more than their spread count as a claim of zero when the
    candidate leaves them out (EC-8). With no claims and no material moves the fit is 1.0.
    """
    implied = dict(claims)
    for node, forecast in engine.nodes.items():
        if node in implied or not node.startswith("indicator:"):
            continue
        mean = forecast.mean[turn]
        if abs(mean) > _spread(mean, forecast.p10[turn], forecast.p90[turn]):
            implied[node] = 0.0
    logs = []
    for node, claimed in implied.items():
        forecast = engine.nodes.get(node)
        if forecast is None:
            continue
        mean = forecast.mean[turn]
        z = (claimed - mean) / _spread(mean, forecast.p10[turn], forecast.p90[turn])
        logs.append(-(min(abs(z), Z_CAP) ** 2) / 2)
    return math.exp(sum(logs) / len(logs)) if logs else 1.0


# How delivery (story RB-4) shifts the base rate of some events. Placeholders for Project 11.
CONSULTED = {"strike": 0.6, "protest": 0.6, "backbench_rebellion": 0.6, "legal_challenge": 0.7}
IMPOSED = {"strike": 1.3, "protest": 1.3, "backbench_rebellion": 1.2, "media_backlash": 1.1}
PHASED = {"market_selloff": 0.75, "capital_flight": 0.75}


# Consulting calms an event only when someone who could start it was consulted (M7):
# workers or employers for a strike, any group for a protest or a legal challenge, MPs for a
# rebellion. Courts and other institutions can also bring or settle a legal challenge.
CONSULT_PREFIXES = {
    "strike": ("group:", "sector:"),
    "protest": ("group:",),
    "backbench_rebellion": ("institution:legislature",),
    "legal_challenge": ("group:", "institution:"),
}


def consulted_for(tag: str, delivery: Delivery) -> bool:
    """Whether the leader consulted someone who could start (or stop) this event. For an
    event no stakeholder starts, such as a media backlash, consulting anyone counts."""
    prefixes = CONSULT_PREFIXES.get(tag)
    if prefixes is None:
        return bool(delivery.consulted)
    return any(c.startswith(prefixes) for c in delivery.consulted)


def delivery_factor(tag: str, delivery: Delivery | None) -> float:
    """Multiplier on a tag's base rate for how the response was delivered."""
    if delivery is None:
        return 1.0
    factor = CONSULTED.get(tag, 1.0) if consulted_for(tag, delivery) else 1.0
    if delivery.speed == "phased":
        factor *= PHASED.get(tag, 1.0)
    elif not consulted_for(tag, delivery):
        factor *= IMPOSED.get(tag, 1.0)
    return factor


# Resistance (``policy.feasibility``: a thin or reluctant majority, or a deficit already past
# the warning level) makes a rebellion likelier: the base rate is multiplied by
# 1 + RESISTANCE_FACTOR * resistance (EB-8).
RESISTANCE_FACTOR = {"backbench_rebellion": 2.0}


def resistance_factor(tag: str, resistance: float) -> float:
    return 1 + RESISTANCE_FACTOR.get(tag, 0.0) * resistance


# A quiet outcome gets rarer as the scenario grows urgent or the actions meet resistance:
# its rate is multiplied by (1 - QUIET_URGENCY * urgency) * (1 - QUIET_RESISTANCE * resistance).
QUIET_URGENCY = 0.5
QUIET_RESISTANCE = 0.5
# How strongly co-tagged events come together: each tag beyond the rarest multiplies the
# chance by its own rate ** (1 - TAG_CORRELATION). 1 would be "the rarest bounds it" (they
# always come together), 0 independent events.
TAG_CORRELATION = 0.7


def tag_rate(
    tag: str, delivery: Delivery | None = None, resistance: float = 0.0, urgency: float = 0.0
) -> float:
    """One tag's base rate, adjusted for delivery, resistance and, for a quiet outcome,
    urgency."""
    rate = FITTED_RATES.get(tag, 0.1) * delivery_factor(tag, delivery)
    if tag == "none":
        rate *= (1 - QUIET_URGENCY * urgency) * (1 - QUIET_RESISTANCE * resistance)
    else:
        rate *= resistance_factor(tag, resistance)
    return min(1.0, rate)


def base_rate(
    tags: list[str],
    delivery: Delivery | None = None,
    resistance: float = 0.0,
    urgency: float = 0.0,
) -> float:
    """How common the tagged events are, adjusted for delivery, the resistance the actions
    meet and the scenario's urgency; untagged counts as a quiet outcome.

    The rarest tag bounds the chance (P(A and B) <= min), and each further tag lowers it by
    its own rate ** (1 - TAG_CORRELATION). Adding a tag can therefore never make an outcome
    look more common (EC-9).
    """
    tags = list(dict.fromkeys(t for t in tags if t != "none")) or ["none"]
    rates = sorted(tag_rate(t, delivery, resistance, urgency) for t in tags)
    chance = rates[0]
    for rate in rates[1:]:
        chance *= rate ** (1 - TAG_CORRELATION)
    return chance


def combine(scores: list[CandidateScores], weights: ScoreWeights) -> list[CandidateScores]:
    """Blend each candidate's scores and normalise to probabilities that sum to 1."""
    logits = [
        weights.consistency * math.log(max(s.consistency, _FLOOR))
        + weights.judge * math.log(max(s.judge, _FLOOR))
        + weights.base_rate * math.log(max(s.base_rate, _FLOOR))
        for s in scores
    ]
    top = max(logits)
    raw = [math.exp(x - top) for x in logits]
    total = sum(raw)
    return [
        s.model_copy(update={"probability": r / total}) for s, r in zip(scores, raw, strict=True)
    ]
