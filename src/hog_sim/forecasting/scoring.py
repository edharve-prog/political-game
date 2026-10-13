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
* **base_rate**: how common its tagged events are in practice (``BASE_RATES``). Several
  tags all have to happen, so the rarest one bounds the score.

Candidates fill forecast slots (D3, ``slots``): one per equal-probability band of the
engine's distribution, ordered by luck for the government, plus a wildcard. A slot candidate
is scored for consistency against its own slot's target (``slot_target``) rather than the
engine's mean, so the unlucky and lucky paths are not penalised for being what they were asked
to be, and each band carries the same engine mass. The wildcard is scored against the mean.

``combine`` blends them log-linearly with ``ScoreWeights`` and normalises to probabilities.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Literal

from hog_sim.core.models import Delivery, Model
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
# within a few months. Placeholders to be calibrated in Project 11.
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


WILDCARD = "wildcard"


def slots(n: int) -> list[str]:
    """Slot names for ``n`` candidates: ``n - 1`` luck bands of equal engine probability,
    unluckiest first, named by the percentile at each band's middle (``p17``, ``p50``,
    ``p83`` for four candidates), then the wildcard."""
    bands = max(n - 1, 1)
    names = [f"p{round(100 * (i + 0.5) / bands)}" for i in range(bands)]
    return [*names, WILDCARD] if n > 1 else names


def slot_luck(slot: str) -> float | None:
    """The luck percentile (0..1) a slot sits at; None for the wildcard or an unknown slot."""
    if slot.startswith("p") and slot[1:].isdigit():
        return min(max(int(slot[1:]) / 100, 0.01), 0.99)
    return None


def slot_target(
    forecast_mean: float, p10: float, p90: float, luck: float | None, favour: float
) -> float:
    """Where an indicator sits at ``luck`` for the government: the engine's normal at that
    percentile, read on the side ``favour`` (+1 when a rise helps the government, -1 when it
    hurts) says is lucky. ``luck`` None is the mean."""
    if luck is None:
        return forecast_mean
    z = NormalDist().inv_cdf(luck)
    return forecast_mean + (favour or 1.0) * z * (p90 - p10) / _P10_P90_SPAN


def consistency(
    claims: dict[str, float],
    engine: DeltaDistribution,
    turn: int,
    luck: float | None = None,
    favour: dict[str, float] | None = None,
) -> float:
    """Fit of claimed native-unit changes at ``turn`` to the engine's distribution, centred
    on the slot target for ``luck`` (the mean when None; see ``slot_target``). ``favour``
    gives each indicator's lucky direction for the government (default +1).

    Indicators the engine moves by more than their spread count as a claim of zero when the
    candidate leaves them out (EC-8). With no claims and no material moves the fit is 1.0.
    """
    favour = favour or {}

    def centre(node: str) -> float:
        f = engine.nodes[node]
        return slot_target(f.mean[turn], f.p10[turn], f.p90[turn], luck, favour.get(node, 1.0))

    implied = dict(claims)
    for node, forecast in engine.nodes.items():
        if node in implied or not node.startswith("indicator:"):
            continue
        target = centre(node)
        if abs(target) > _spread(target, forecast.p10[turn], forecast.p90[turn]):
            implied[node] = 0.0
    logs = []
    for node, claimed in implied.items():
        forecast = engine.nodes.get(node)
        if forecast is None:
            continue
        target = centre(node)
        mean = forecast.mean[turn]
        z = (claimed - target) / _spread(mean, forecast.p10[turn], forecast.p90[turn])
        logs.append(-(min(abs(z), Z_CAP) ** 2) / 2)
    return math.exp(sum(logs) / len(logs)) if logs else 1.0


# How delivery (story RB-4) shifts the base rate of some events. Placeholders for Project 11.
CONSULTED = {"strike": 0.6, "protest": 0.6, "backbench_rebellion": 0.6, "legal_challenge": 0.7}
IMPOSED = {"strike": 1.3, "protest": 1.3, "backbench_rebellion": 1.2, "media_backlash": 1.1}
PHASED = {"market_selloff": 0.75, "capital_flight": 0.75}


def delivery_factor(tag: str, delivery: Delivery | None) -> float:
    """Multiplier on a tag's base rate for how the response was delivered."""
    if delivery is None:
        return 1.0
    factor = CONSULTED.get(tag, 1.0) if delivery.consulted else 1.0
    if delivery.speed == "phased":
        factor *= PHASED.get(tag, 1.0)
    elif not delivery.consulted:
        factor *= IMPOSED.get(tag, 1.0)
    return factor


# Resistance (``policy.feasibility``: a thin or reluctant majority, or a deficit already past
# the warning level) makes a rebellion likelier: the base rate is multiplied by
# 1 + RESISTANCE_FACTOR * resistance (EB-8).
RESISTANCE_FACTOR = {"backbench_rebellion": 2.0}


def resistance_factor(tag: str, resistance: float) -> float:
    return 1 + RESISTANCE_FACTOR.get(tag, 0.0) * resistance


def base_rate(tags: list[str], delivery: Delivery | None = None, resistance: float = 0.0) -> float:
    """How common the tagged events are, adjusted for delivery and the resistance the
    actions meet; untagged counts as a quiet outcome.

    All the tags have to happen, so the rarest bounds the chance (P(A and B) <= min). Adding
    a tag can therefore never make an outcome look more common (EC-9).
    """
    tags = [t for t in tags if t != "none"] or ["none"]
    return min(
        min(
            1.0,
            BASE_RATES.get(t, 0.1)
            * delivery_factor(t, delivery)
            * resistance_factor(t, resistance),
        )
        for t in tags
    )


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
