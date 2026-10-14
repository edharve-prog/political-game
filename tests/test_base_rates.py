"""Event base rates fitted to history, and the M7 fixes (review 2026-10-13, D7 and M7)."""

import pytest

from hog_sim.core.models import Delivery
from hog_sim.forecasting.base_rates import HISTORICAL, Observation, fit_base_rates
from hog_sim.forecasting.scoring import (
    BASE_RATES,
    FITTED_RATES,
    TAG_CORRELATION,
    base_rate,
    tag_rate,
)
from hog_sim.ui.cli import main


def test_fit_moves_the_prior_towards_what_happened() -> None:
    prior = {"none": 0.5, "strike": 0.1, "foreign_praise": 0.2}
    seen = [Observation(episode=str(i), tags=["strike"]) for i in range(20)]
    fitted = fit_base_rates(prior, seen, strength=20)
    assert fitted["strike"] == pytest.approx((20 * 0.1 + 20) / 40)
    assert fitted["none"] == pytest.approx(0.25)
    # Domestic episodes say nothing about foreign reactions.
    assert fitted["foreign_praise"] == pytest.approx(0.2)
    assert fit_base_rates(prior, []) == prior


def test_fitted_rates_cover_every_tag_and_stay_probabilities() -> None:
    assert set(FITTED_RATES) == set(BASE_RATES)
    assert all(0 < r < 1 for r in FITTED_RATES.values())
    assert len(HISTORICAL) >= 20
    # History has more strikes after imposed pay deals and legal challenges than the guess.
    assert FITTED_RATES["legal_challenge"] > BASE_RATES["legal_challenge"]


def test_quiet_outcomes_get_rarer_with_urgency_and_resistance() -> None:
    calm = base_rate(["none"])
    assert base_rate(["none"], urgency=0.9) < calm
    assert base_rate(["none"], resistance=0.5) < calm
    assert base_rate(["strike"], urgency=0.9) == base_rate(["strike"])


def test_consulting_only_calms_events_the_consulted_could_start() -> None:
    bank = Delivery(consulted=["institution:central_bank"])
    unions = Delivery(consulted=["group:public_workers"])
    mps = Delivery(consulted=["institution:legislature"])
    imposed = Delivery()
    assert tag_rate("strike", bank) == tag_rate("strike", imposed)
    assert tag_rate("strike", unions) < tag_rate("strike", imposed)
    assert tag_rate("backbench_rebellion", unions) == tag_rate("backbench_rebellion", imposed)
    assert tag_rate("backbench_rebellion", mps) < tag_rate("backbench_rebellion", imposed)


def test_a_second_tag_lowers_the_chance_but_less_than_independence() -> None:
    selloff, backlash = FITTED_RATES["market_selloff"], FITTED_RATES["media_backlash"]
    both = base_rate(["market_selloff", "media_backlash"])
    assert both == pytest.approx(selloff * backlash ** (1 - TAG_CORRELATION))
    assert selloff * backlash < both < selloff
    assert base_rate(["market_selloff", "market_selloff"]) == base_rate(["market_selloff"])


def test_base_rates_report(capsys, tmp_path) -> None:
    main(["base-rates", "--save", str(tmp_path / "none.db")])
    out = capsys.readouterr().out
    assert "fitted" in out and "legal_challenge" in out and "Games: 0 turns" in out
