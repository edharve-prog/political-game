"""Event base rates fitted to what really happened (review 2026-10-13, D7 and M7).

``scoring.BASE_RATES`` holds the hand-set placeholders. They stay as the prior, and
``fit_base_rates`` updates them with observations: each observation is a policy response and
the event tags that followed it within a few months. The update is a weighted Beta-binomial
mean, so a tag's rate is

    (PRIOR_STRENGTH * placeholder + sum(weight * happened)) / (PRIOR_STRENGTH + sum(weight))

and a quiet response (no tags) counts towards ``none``.

``HISTORICAL`` is a small set of UK episodes with the events that followed them. Famous
episodes are famous because something happened, so the set over-represents eventful
responses; ``PRIOR_STRENGTH`` is set high enough that it moves the placeholders rather than
replacing them. Outcomes chosen in logged games are written by the model, so feeding them
back would only teach the game its own guesses; ``hog-sim base-rates`` instead compares them
with the fitted rates, to show where the model's outcomes drift from history.

Until held-out episodes confirm them the rates are relative, and the game labels outcome
probabilities "relative likelihood".
"""

from __future__ import annotations

from collections.abc import Iterable

from hog_sim.core.models import Model

PRIOR_STRENGTH = 20.0  # observations' worth of weight the placeholder carries


FOREIGN_TAGS = ("foreign_retaliation", "foreign_praise")


class Observation(Model):
    episode: str
    tags: list[str]  # events that followed; empty means a quiet response
    weight: float = 1.0
    # Whether the response faced other countries. A purely domestic response says nothing
    # about how often foreign reactions follow, so it does not count towards those tags.
    foreign: bool = False

    def informs(self, tag: str) -> bool:
        return self.foreign or tag not in FOREIGN_TAGS


def _obs(episode: str, *tags: str, foreign: bool = False) -> Observation:
    return Observation(episode=episode, tags=list(tags), foreign=foreign)


# UK policy responses and the event tags that followed within a few months.
HISTORICAL: list[Observation] = [
    _obs("1990 poll tax", "protest", "backbench_rebellion", "media_backlash"),
    _obs("2008 bank recapitalisation", "market_rally", "foreign_praise"),
    _obs("2008 temporary VAT cut"),
    _obs("2015 National Living Wage", "media_praise"),
    _obs("2016 soft drinks levy"),
    _obs(
        "2018 expulsion of Russian diplomats", "foreign_retaliation", "foreign_praise", foreign=True
    ),
    _obs("2019 net zero target in law"),
    _obs("2023 Windsor Framework", "foreign_praise", "backbench_rebellion", foreign=True),
    _obs("2010 tuition fee rise", "protest", "backbench_rebellion"),
    _obs("2011 public sector pension reform", "strike", "protest"),
    _obs("2012 Budget (pasty tax)", "media_backlash"),
    _obs("2013 spare room subsidy cut", "protest", "legal_challenge", "media_backlash"),
    _obs("2015 tax credit cuts", "media_backlash", "backbench_rebellion"),
    _obs("2016 junior doctors' contract imposed", "strike", "media_backlash"),
    _obs("2016 Article 50 without a Commons vote", "legal_challenge"),
    _obs("2017 social care plan in the manifesto", "media_backlash"),
    _obs("2019 prorogation of Parliament", "legal_challenge", "protest"),
    _obs("2020 furlough scheme", "media_praise"),
    _obs("2021 National Insurance rise for health and care", "backbench_rebellion"),
    _obs("2022 mini-budget", "market_selloff", "media_backlash", "backbench_rebellion"),
    _obs("2022 Energy Price Guarantee"),
    _obs("2022 public sector pay below inflation", "strike"),
    _obs("2022 Rwanda asylum plan", "legal_challenge", "protest"),
]


def fit_base_rates(
    prior: dict[str, float],
    observations: Iterable[Observation],
    strength: float = PRIOR_STRENGTH,
) -> dict[str, float]:
    """Each tag's rate: the prior updated with how often the tag followed the observed
    responses. ``none`` counts the responses after which nothing happened."""
    observations = list(observations)
    rates = {}
    for tag, rate in prior.items():
        relevant = [o for o in observations if o.informs(tag)]
        total = sum(o.weight for o in relevant)
        if tag == "none":
            seen = sum(o.weight for o in relevant if not o.tags)
        else:
            seen = sum(o.weight for o in relevant if tag in o.tags)
        rates[tag] = (strength * rate + seen) / (strength + total)
    return rates


def tag_frequencies(observations: Iterable[Observation]) -> dict[str, float]:
    """Share of the observations that inform each tag which it followed (``none`` for quiet
    ones)."""
    observations = list(observations)
    shares = {}
    for tag in dict.fromkeys(t for o in observations for t in [*o.tags, "none", *FOREIGN_TAGS]):
        relevant = [o for o in observations if o.informs(tag)]
        total = sum(o.weight for o in relevant)
        seen = sum(o.weight for o in relevant if (tag in o.tags if tag != "none" else not o.tags))
        if total:
            shares[tag] = seen / total
    return shares
