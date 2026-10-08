"""A small hand-built UK world for tests and early development.

Figures are illustrative round numbers, not sourced data. Real start states come from
Project 2 (seed data).
"""

from __future__ import annotations

from hog_sim.core.models import (
    Country,
    Edge,
    EdgeKind,
    Group,
    Indicator,
    Institution,
    Sector,
)
from hog_sim.core.state import WorldState
from hog_sim.world.cast import default_cast

E = EdgeKind


def _edge(
    source: str, target: str, kind: EdgeKind, weight: float, lag: int = 0, unc: float = 0.1
) -> Edge:
    return Edge(source=source, target=target, kind=kind, weight=weight, lag=lag, uncertainty=unc)


def toy_world() -> WorldState:
    countries = [
        Country(id="country:uk", name="United Kingdom", gdp_bn=2700, growth_pct=1.0),
        Country(
            id="country:eu", name="European Union", gdp_bn=17000, growth_pct=1.0, relationship=0.3
        ),
        Country(id="country:china", name="China", gdp_bn=18000, growth_pct=4.5, relationship=-0.2),
    ]
    sectors = [
        Sector(id="sector:energy", name="Energy", output_bn=80, employment_k=170),
        Sector(id="sector:finance", name="Finance", output_bn=230, employment_k=1100),
        Sector(id="sector:manufacturing", name="Manufacturing", output_bn=220, employment_k=2600),
        Sector(
            id="sector:housing", name="Housing & Construction", output_bn=150, employment_k=2100
        ),
        Sector(id="sector:public", name="Public Sector", output_bn=500, employment_k=5900),
    ]
    groups = [
        Group(
            id="group:pensioners",
            name="Pensioners",
            population_share=0.19,
            turnout=0.80,
            approval=0.55,
            lean=0.55,
        ),
        Group(
            id="group:young_renters",
            name="Young renters",
            population_share=0.15,
            turnout=0.45,
            approval=0.35,
            lean=0.35,
        ),
        Group(
            id="group:public_workers",
            name="Public sector workers",
            population_share=0.17,
            turnout=0.70,
            approval=0.45,
            lean=0.45,
        ),
        Group(id="group:business", name="Business owners", population_share=0.08, turnout=0.75),
    ]
    institutions = [
        Institution(id="institution:legislature", name="House of Commons", support=0.6, power=0.8),
        Institution(
            id="institution:central_bank",
            name="Bank of England",
            support=0.5,
            power=0.7,
            independence=0.9,
        ),
    ]
    # Bounds are hard limits the engine never crosses; the deficit's own pushes fade fast
    # because spending and tax hold theirs for as long as the policy runs.
    indicators = [
        Indicator(
            id="indicator:inflation", name="CPI inflation", value=3.5, unit="%", low=-3, high=30
        ),
        Indicator(
            id="indicator:unemployment", name="Unemployment", value=4.5, unit="%", low=2, high=25
        ),
        Indicator(
            id="indicator:interest_rate",
            name="Bank Rate",
            value=4.0,
            unit="%",
            low=0,
            high=20,
            controlled_by="institution:central_bank",
        ),
        Indicator(
            id="indicator:energy_prices",
            name="Household energy prices",
            value=100,
            unit="index",
            low=30,
            high=400,
            interventions=["regulate", "spend"],  # a price cap, a subsidy on bills
        ),
        Indicator(
            id="indicator:house_prices",
            name="House prices",
            value=100,
            unit="index",
            low=30,
            high=300,
            interventions=["regulate", "tax"],  # lending rules, stamp duty
        ),
        Indicator(
            id="indicator:deficit",
            name="Budget deficit",
            value=4.5,
            unit="% GDP",
            low=-5,
            high=25,
            persistence=0.5,
        ),
        # Sterling against a basket of trading partners' currencies; markets set it, so no
        # policy aims at it directly (CA-3).
        Indicator(
            id="indicator:exchange_rate",
            name="Sterling exchange rate",
            value=100,
            unit="index",
            low=40,
            high=200,
        ),
    ]
    edges = [
        # Foreign relations and trade
        # The EU buys over 40% of UK exports; the UK is a small share of the EU's (CA-3).
        _edge("country:uk", "country:eu", E.TRADES_WITH, 0.10),
        _edge("country:eu", "country:uk", E.TRADES_WITH, 0.45),
        _edge("country:uk", "country:china", E.TRADES_WITH, 0.10),
        _edge("country:china", "country:uk", E.TRADES_WITH, 0.03),
        _edge("country:uk", "country:eu", E.ALLIED_WITH, 0.5),
        _edge("country:china", "sector:manufacturing", E.SUPPLIES, 0.2, unc=0.05),
        # Half of UK goods exports go to the EU, so its demand reaches factories directly (CA-3).
        _edge("country:eu", "sector:manufacturing", E.SUPPLIES, 0.3, lag=1),
        # Input-output links
        _edge("sector:energy", "sector:manufacturing", E.SUPPLIES, 0.15),
        _edge("sector:finance", "sector:housing", E.SUPPLIES, 0.25),
        # Sectors add up to the economy: a step (5%) off a sector's output takes roughly its
        # share of GDP off growth. Finance also carries credit to everyone else (CA-3).
        _edge("sector:finance", "country:uk", E.SUPPLIES, 0.4),
        _edge("sector:manufacturing", "country:uk", E.SUPPLIES, 0.5),
        _edge("sector:public", "country:uk", E.SUPPLIES, 0.5),
        # Sectors and institutions drive indicators
        # More energy output means lower prices
        _edge("sector:energy", "indicator:energy_prices", E.DRIVES, -0.6, unc=0.2),
        _edge("indicator:energy_prices", "indicator:inflation", E.DRIVES, 0.3, lag=1),
        # Dearer energy raises industry's costs and cuts its output (CA-3).
        _edge("indicator:energy_prices", "sector:manufacturing", E.DRIVES, -0.2, lag=1),
        # The Bank of England sets Bank Rate (Indicator.controlled_by); how it reacts to
        # inflation is applied straight to Bank Rate. Its support for the government does
        # not move rates.
        _edge("indicator:inflation", "indicator:interest_rate", E.DRIVES, 0.5, lag=1),
        # It also cuts when unemployment rises, as in 2008 and 2020 (CA-3).
        _edge("indicator:unemployment", "indicator:interest_rate", E.DRIVES, -0.5, lag=1),
        _edge("indicator:interest_rate", "indicator:inflation", E.DRIVES, -0.4, lag=6),
        _edge("indicator:interest_rate", "indicator:house_prices", E.DRIVES, -0.5, lag=3),
        _edge("indicator:interest_rate", "indicator:unemployment", E.DRIVES, 0.2, lag=6),
        _edge("sector:manufacturing", "indicator:unemployment", E.DRIVES, -0.2, lag=2),
        # Okun's law: slower growth costs jobs a turn later (CA-3).
        _edge("country:uk", "indicator:unemployment", E.DRIVES, -0.4, lag=1),
        _edge("sector:housing", "indicator:house_prices", E.DRIVES, -0.3, lag=12),
        # Credit: when banks lend more, buyers can borrow more and prices rise (CA-3).
        _edge("sector:finance", "indicator:house_prices", E.DRIVES, 0.4, lag=1),
        # Borrowing: markets demand higher rates and it feeds prices. What spending and tax
        # decisions cost the budget is added to the deficit directly (FISCAL_STEPS).
        _edge("indicator:deficit", "indicator:interest_rate", E.DRIVES, 0.3, lag=1),
        _edge("indicator:deficit", "indicator:inflation", E.DRIVES, 0.1, lag=2),
        # The pound: growth and higher rates attract money, heavy borrowing scares it off, and
        # a weaker pound makes imports dearer a couple of turns later (CA-3).
        _edge("country:uk", "indicator:exchange_rate", E.DRIVES, 0.3),
        _edge("indicator:interest_rate", "indicator:exchange_rate", E.DRIVES, 0.2),
        _edge("indicator:deficit", "indicator:exchange_rate", E.DRIVES, -0.4),
        _edge("indicator:exchange_rate", "indicator:inflation", E.DRIVES, -0.3, lag=2),
        # Employment
        _edge("sector:public", "group:public_workers", E.EMPLOYS, 0.9, unc=0.0),
        _edge("sector:finance", "group:business", E.EMPLOYS, 0.2, unc=0.0),
        _edge("sector:manufacturing", "group:business", E.EMPLOYS, 0.2, unc=0.0),
        _edge("sector:housing", "group:young_renters", E.EMPLOYS, 0.1, unc=0.0),
        # What each group cares about (negative weight: a rise lowers approval)
        _edge("group:pensioners", "indicator:inflation", E.CARES_ABOUT, -0.4, unc=0.05),
        _edge("group:pensioners", "indicator:energy_prices", E.CARES_ABOUT, -0.3, unc=0.05),
        _edge("group:pensioners", "indicator:house_prices", E.CARES_ABOUT, 0.1, unc=0.05),
        _edge("group:young_renters", "indicator:house_prices", E.CARES_ABOUT, -0.4, unc=0.05),
        _edge("group:young_renters", "indicator:unemployment", E.CARES_ABOUT, -0.3, unc=0.05),
        _edge("group:public_workers", "indicator:inflation", E.CARES_ABOUT, -0.3, unc=0.05),
        _edge("group:pensioners", "indicator:deficit", E.CARES_ABOUT, -0.15, unc=0.05),
        _edge("group:public_workers", "indicator:deficit", E.CARES_ABOUT, 0.1, unc=0.05),
        _edge("group:business", "indicator:interest_rate", E.CARES_ABOUT, -0.3, unc=0.05),
        _edge("group:business", "indicator:deficit", E.CARES_ABOUT, -0.3, unc=0.05),
        # Institutions
        _edge("institution:legislature", "group:public_workers", E.INFLUENCES, 0.1),
    ]
    return WorldState(
        player_country="country:uk",
        countries={c.id: c for c in countries},
        sectors={s.id: s for s in sectors},
        groups={g.id: g for g in groups},
        institutions={i.id: i for i in institutions},
        indicators={i.id: i for i in indicators},
        edges=edges,
        characters=default_cast(),
    )
