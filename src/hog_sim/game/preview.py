"""Decision preview (review of 13 Oct 2026, D2): what a proposal does by the election.

Before the player confirms, ``preview`` plays the rest of the parliament twice with the same
engine and approval rules ``resolve`` uses: once with the proposal and once doing nothing this
turn. Both runs face this turn's crisis (its scenario shocks) and then quiet turns, with no
outcome effects and no new decisions, so the only difference between them is the proposal.
For each measure it also runs the package without that measure, so a measure's cost is what
the package would save without it, including how it interacts with the rest.

It reports the deficit and debt path, each measure's debt and political capital cost, and
the projected vote and seats at the election. These are what the decisions alone do in a calm
world, not a forecast of what will happen: the real turns bring new crises and outcomes.
"""

from __future__ import annotations

from collections.abc import Sequence

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Delivery, Model, Outcome, Pledge, PolicyAction, Scenario
from hog_sim.core.state import WorldState
from hog_sim.game.loop import resolve
from hog_sim.policy.feasibility import Role
from hog_sim.policy.limits import capital_cost
from hog_sim.population.popularity import DEFICIT_ID, run_election, vote_intention

QUIET_SCENARIO = Scenario(title="(a quiet month)", briefing="", affected_nodes=[], urgency=0.0)
QUIET_OUTCOME = Outcome(narrative="", probability=1.0)


class Path(Model):
    """One run to the election: deficit and debt (% of GDP) after each turn."""

    deficit: list[float]
    debt: list[float]
    vote: float
    seats: int
    total_seats: int


class MeasureCost(Model):
    action: PolicyAction
    debt: float  # debt at the election with the package minus without this measure, % of GDP
    capital: float  # political capital it uses this turn


class Preview(Model):
    turns: list[int]  # the turn each path entry follows
    package: Path
    nothing: Path
    measures: list[MeasureCost]


def _run(
    start: WorldState,
    state: WorldState,
    scenario: Scenario,
    actions: Sequence[PolicyAction],
    config: GameConfig,
    pledges: Sequence[Pledge],
    delivery: Delivery | None,
    sacked: Sequence[str],
) -> Path:
    deficit, debt = [], []
    now = resolve(
        start, state, scenario, list(actions), QUIET_OUTCOME, config, pledges, delivery, sacked
    )
    while True:
        deficit.append(now.indicators[DEFICIT_ID].value if DEFICIT_ID in now.indicators else 0.0)
        debt.append(now.countries[now.player_country].debt_pct)
        if now.turn >= config.election_turn:
            break
        now = resolve(start, now, QUIET_SCENARIO, [], QUIET_OUTCOME, config)
    result = run_election(now)
    return Path(
        deficit=deficit,
        debt=debt,
        vote=vote_intention(now),
        seats=result.seats,
        total_seats=result.total_seats,
    )


def preview(
    start: WorldState,
    state: WorldState,
    scenario: Scenario,
    actions: Sequence[PolicyAction],
    config: GameConfig,
    pledges: Sequence[Pledge] = (),
    delivery: Delivery | None = None,
    sacked: Sequence[str] = (),
    role: Role = "prime_minister",
) -> Preview:
    """The proposal against doing nothing, played to the election in a calm world."""
    if state.turn >= config.election_turn:
        raise ValueError("the election has already been held")
    package = _run(start, state, scenario, actions, config, pledges, delivery, sacked)
    nothing = _run(start, state, scenario, [], config, (), None, ())
    measures = []
    for i, action in enumerate(actions):
        if action.kind == "do_nothing":
            continue
        rest = [a for j, a in enumerate(actions) if j != i]
        without = _run(start, state, scenario, rest, config, pledges, delivery, sacked)
        measures.append(
            MeasureCost(
                action=action,
                debt=package.debt[-1] - without.debt[-1],
                capital=capital_cost([action], state, role),
            )
        )
    return Preview(
        turns=list(range(state.turn + 1, config.election_turn + 1)),
        package=package,
        nothing=nothing,
        measures=measures,
    )


def _points(preview: Preview) -> list[int]:
    """Indexes to show: next turn, every sixth turn after it, and the election."""
    last = len(preview.turns) - 1
    return sorted({0, *range(5, last, 6), last})


def preview_lines(preview: Preview, describe=lambda a: f"{a.kind} {a.target}") -> list[str]:
    """The preview as lines for the review screen."""
    p, n = preview.package, preview.nothing
    lines = ["If nothing else happens (no new crises, no further decisions):", ""]
    lines.append(
        f"  {'after turn':<22}" + "".join(f"{preview.turns[i]:>8}" for i in _points(preview))
    )
    for label, values in (
        ("deficit, this plan", p.deficit),
        ("deficit, doing nothing", n.deficit),
        ("debt, this plan", p.debt),
        ("debt, doing nothing", n.debt),
    ):
        lines.append(f"  {label:<22}" + "".join(f"{values[i]:>8.1f}" for i in _points(preview)))
    lines.append("  (% of GDP)")
    lines.append("")
    lines += [
        f"  Election, this plan:      {p.vote * 100:.1f}% of the vote, {p.seats} seats",
        f"  Election, doing nothing:  {n.vote * 100:.1f}% of the vote, {n.seats} seats"
        f"  ({(p.vote - n.vote) * 100:+.1f} pts for this plan)",
    ]
    if preview.measures:
        lines += ["", "  Each measure's cost:"]
        for m in preview.measures:
            lines.append(f"  - {describe(m.action)}")
            lines.append(
                f"    {m.debt:+.2f}% of GDP on the debt by the election, "
                f"{m.capital:.2f} political capital"
            )
    return lines
