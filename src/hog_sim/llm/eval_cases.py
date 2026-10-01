"""Fixed player responses and what a sensible interpretation must contain.

Used two ways: ``python -m hog_sim.llm.evaluate`` runs them against the live model (and
records a cassette), and the test suite replays that cassette so CI checks real model
output without an API key. Cases are written against the toy world.
"""

from __future__ import annotations

from pydantic import Field

from hog_sim.core.models import Model, PolicyAction


class Expect(Model):
    """One action the interpretation must contain. Any listed kind/target is acceptable."""

    kinds: set[str]
    targets: set[str]
    sign: int = Field(0, description="+1 or -1 to require the magnitude's sign, 0 for any")

    def matches(self, action: PolicyAction) -> bool:
        if action.kind not in self.kinds or action.target not in self.targets:
            return False
        return self.sign == 0 or (action.magnitude > 0) == (self.sign > 0)


class Case(Model):
    text: str
    expect: list[Expect] = Field(default_factory=list)
    expects_question: bool = False
    forbid_kinds: set[str] = Field(default_factory=set)


def score(case: Case, actions: list[PolicyAction], question: str | None) -> list[str]:
    """Problems with an interpretation of ``case``; empty means it passes."""
    problems = []
    if case.expects_question and not question:
        problems.append("expected a clarifying question")
    for exp in case.expect:
        if not any(exp.matches(a) for a in actions):
            problems.append(f"missing {sorted(exp.kinds)} on {sorted(exp.targets)}")
    bad = [a.kind for a in actions if a.kind in case.forbid_kinds]
    if bad:
        problems.append(f"unexpected kinds {bad}")
    return problems


def _e(kinds: str, targets: str, sign: int = 0) -> Expect:
    return Expect(kinds=set(kinds.split()), targets=set(targets.split()), sign=sign)


ENERGY = "sector:energy indicator:energy_prices"
HOUSING = "sector:housing indicator:house_prices group:young_renters"
PUBLIC = "sector:public group:public_workers"

CASES: list[Case] = [
    Case(
        text="Put a windfall tax on energy company profits.",
        expect=[_e("tax", "sector:energy", +1)],
    ),
    Case(
        text="Cap household energy bills this winter and pay the suppliers the difference.",
        expect=[_e("spend regulate", ENERGY)],
    ),
    Case(
        text="Give nurses and teachers a 5% pay rise, funded by borrowing.",
        expect=[_e("spend", PUBLIC, +1)],
    ),
    Case(
        text="Freeze public sector pay for two years to get the deficit down.",
        expect=[_e("spend", PUBLIC + " indicator:deficit", -1)],
        forbid_kinds={"military"},
    ),
    Case(
        text="Build 300,000 council homes a year.",
        expect=[_e("spend legislate", HOUSING, +1)],
    ),
    Case(
        text="Scrap planning restrictions so developers can build faster.",
        expect=[_e("deregulate legislate", HOUSING)],
    ),
    Case(
        text="Introduce rent controls in big cities.",
        expect=[_e("regulate legislate", HOUSING)],
    ),
    Case(
        text="Cut corporation tax to attract investment, and tell business we are open for "
        "business.",
        expect=[
            _e("tax", "group:business sector:finance sector:manufacturing", -1),
            _e("communicate", "group:business"),
        ],
    ),
    Case(
        text="Raise the state pension by 8% in line with inflation.",
        expect=[_e("spend", "group:pensioners", +1)],
    ),
    Case(
        text="Tighten capital requirements on the banks after the last crash.",
        expect=[_e("regulate", "sector:finance")],
    ),
    Case(
        text="Open talks with Brussels on a closer trade deal.",
        expect=[_e("diplomatic", "country:eu", +1)],
    ),
    Case(
        text="Impose tariffs on Chinese steel and expel two of their diplomats.",
        expect=[_e("diplomatic tax", "country:china sector:manufacturing")],
    ),
    Case(
        text="Send a naval task force to the South China Sea.",
        expect=[_e("military", "country:china", +1)],
    ),
    Case(
        text="Do nothing. Let the market sort it out.",
        expect=[_e("do_nothing", "country:uk")],
    ),
    Case(
        text="Give a speech reassuring pensioners their savings are safe.",
        expect=[_e("communicate", "group:pensioners", +1)],
    ),
    Case(
        text="Sort it out.",
        expects_question=True,
    ),
    Case(
        text="Replace the Governor of the Bank of England with someone who will cut rates.",
        expect=[_e("appoint", "institution:central_bank")],
    ),
    Case(
        text="Subsidise British factories to switch to cheaper green energy.",
        expect=[_e("spend", "sector:manufacturing sector:energy", +1)],
    ),
    Case(
        text="Pass a law banning strikes in essential public services.",
        expect=[_e("legislate regulate", PUBLIC + " institution:legislature")],
    ),
    Case(
        text="Raise income tax by 2p and spend it all on the health service.",
        expect=[
            _e(
                "tax",
                "country:uk group:business group:public_workers group:pensioners "
                "group:young_renters sector:public",
                +1,
            ),
            _e("spend", "sector:public group:public_workers", +1),
        ],
    ),
]
