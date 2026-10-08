"""Historical backtests: does the engine move the right way in episodes we know (Project 11, CA-1)?

Each episode puts a real event into the starting world's terms: the shocks that hit and the
policy the government chose. The engine resolves it turn by turn with nothing else happening,
and each check asks whether one number went the way it did in reality. A check compares one of:

* the episode against its start;
* the episode against the same shocks with no policy, which isolates what the policy did;
* the shocks alone against the start, which isolates what the event did before the
  government stepped in.

Checks pass on direction, and a move smaller than ``MIN_STEPS`` counts as no move. A check can
also carry ``real``, the rough range the number actually moved within the episode's year, and
the report shows how big the engine's move is against it (CA-4). Sizes are a report only for
now; they become a test once the ranges are agreed.

Some checks fail because the world has no link for the effect yet. Those carry a ``gap`` note
saying what is missing, and the test expects them to keep failing until someone closes the gap,
so fixing one means updating its check here.

``uv run python -m hog_sim.game.backtests`` prints the report; ``tests/test_backtests.py``
asserts it.
"""

from __future__ import annotations

from typing import Literal

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Model, NodeKind, Outcome, PolicyAction, Scenario, Shock
from hog_sim.core.state import WorldState
from hog_sim.game.loop import resolve
from hog_sim.population.popularity import vote_intention
from hog_sim.world.propagation import METRICS, scale
from hog_sim.world.seed.toy import toy_world

VOTE = "vote"  # pseudo-node: the government's vote intention
NOTHING = Outcome(narrative="Nothing else happens", probability=1)
MIN_STEPS = 0.05  # a smaller peak move counts as no move at all


def act(kind: str, target: str, magnitude: float, turns: int = 1) -> PolicyAction:
    return PolicyAction(kind=kind, target=target, magnitude=magnitude, duration_turns=turns)


def shock(node: str, steps: float) -> Shock:
    return Shock(node=node, delta=steps)


class Check(Model):
    node: str
    direction: Literal[1, -1]
    against: Literal["start", "no_policy", "shocks_only"] = "start"
    gap: str | None = None  # why the engine gets this wrong today; None means it should pass
    # How far the number really moved, low to high, in its own units: percentage points for
    # indicators and UK growth, index points for price indices, per cent of output for sectors
    # and a share (0.1 = 10 points) for vote intention.
    real: tuple[float, float] | None = None

    def describe(self) -> str:
        way = "rises" if self.direction > 0 else "falls"
        base = {
            "start": "from the start",
            "no_policy": "against doing nothing",
            "shocks_only": "from the shock alone",
        }[self.against]
        return f"{self.node} {way} {base}"


class Episode(Model):
    name: str
    happened: str  # what happened, in plain words, as the reason for the checks
    shocks: list[Shock] = []
    actions: list[PolicyAction] = []
    checks: list[Check]
    turns: int = 12


class CheckResult(Model):
    episode: str
    check: Check
    peak: float  # largest change over the episode, in standard steps
    change: float  # the same change in the node's own units (see ``Check.real``)
    passed: bool

    @property
    def size(self) -> float | None:
        """The engine's move as a share of the middle of the real range."""
        if self.check.real is None:
            return None
        return self.change / (sum(self.check.real) / 2)


EPISODES = [
    Episode(
        name="2008 banking crisis and bailout",
        happened=(
            "Bank losses froze lending. The government recapitalised the banks, the deficit "
            "jumped, the Bank of England cut rates to 0.5%, unemployment rose from about 5% to "
            "8% and house prices fell."
        ),
        shocks=[shock("sector:finance", -3)],
        actions=[act("spend", "sector:finance", 0.8, 4)],
        checks=[
            Check(node="sector:finance", direction=-1),
            Check(node="sector:finance", direction=1, against="no_policy"),
            Check(node="indicator:deficit", direction=1, real=(4, 8)),
            Check(node="group:business", direction=-1),
            Check(node="indicator:unemployment", direction=1, against="shocks_only", real=(2, 3.5)),
            Check(
                node="indicator:interest_rate",
                direction=-1,
                against="shocks_only",
                real=(-4.5, -3.5),
            ),
            Check(
                node="indicator:house_prices", direction=-1, against="shocks_only", real=(-20, -12)
            ),
            Check(
                node="indicator:exchange_rate",
                direction=-1,
                against="shocks_only",
                real=(-30, -20),
            ),
        ],
    ),
    Episode(
        name="2010 austerity",
        happened=(
            "Spending cuts aimed at the deficit. Borrowing fell, public sector pay was frozen "
            "and around 500,000 public sector jobs went; public sector workers turned against "
            "the government."
        ),
        actions=[act("spend", "sector:public", -0.6, 12)],
        checks=[
            Check(node="indicator:deficit", direction=-1, real=(-3, -1)),
            Check(node="sector:public", direction=-1, real=(-4, -1)),
            Check(node="group:public_workers", direction=-1),
            Check(node="indicator:unemployment", direction=1, real=(0, 0.5)),
        ],
    ),
    Episode(
        name="2022 energy price shock",
        happened=(
            "Gas prices soared after the invasion of Ukraine. Inflation reached 11%, the Bank of "
            "England raised rates, house prices dipped and pensioners felt the squeeze."
        ),
        shocks=[shock("indicator:energy_prices", 3)],
        checks=[
            Check(node="indicator:energy_prices", direction=1, real=(50, 180)),
            Check(node="indicator:inflation", direction=1, real=(4, 7)),
            Check(node="indicator:interest_rate", direction=1, real=(2, 3)),
            Check(node="indicator:house_prices", direction=-1, real=(-5, -2)),
            Check(node="group:pensioners", direction=-1),
            Check(node="sector:manufacturing", direction=-1, real=(-5, -1)),
        ],
    ),
    Episode(
        name="2022 Energy Price Guarantee",
        happened=(
            "The government capped household bills and paid suppliers the difference. Bills "
            "and inflation peaked lower than they would have, at a cost to borrowing."
        ),
        shocks=[shock("indicator:energy_prices", 3)],
        actions=[act("spend", "indicator:energy_prices", -0.8, 6)],
        checks=[
            Check(
                node="indicator:energy_prices", direction=-1, against="no_policy", real=(-40, -25)
            ),
            Check(node="indicator:inflation", direction=-1, against="no_policy", real=(-4, -2)),
            Check(node="indicator:deficit", direction=1, against="no_policy", real=(1, 3)),
        ],
    ),
    Episode(
        name="2022 mini-budget",
        happened=(
            "Unfunded tax cuts spooked the markets. Gilt yields and mortgage rates jumped, "
            "house prices fell, and the government's poll rating collapsed."
        ),
        actions=[act("tax", "country:uk", -0.8, 6)],
        checks=[
            Check(node="indicator:deficit", direction=1, real=(1, 2)),
            Check(node="indicator:interest_rate", direction=1, real=(0.25, 1)),
            Check(node="indicator:house_prices", direction=-1, real=(-4, -1)),
            Check(node="group:business", direction=-1),
            Check(node=VOTE, direction=-1, real=(-0.12, -0.08)),
            Check(
                node="indicator:exchange_rate",
                direction=-1,
                gap="The tax cut's lift to growth and rates outweighs the borrowing; markets "
                "never lose faith in a government's plans.",
                real=(-8, -2),
            ),
        ],
    ),
    Episode(
        name="2020 lockdown and furlough",
        happened=(
            "Lockdown shut much of the economy and output fell by a fifth. Furlough paid wages "
            "so unemployment rose far less than feared, while borrowing hit a peacetime record."
        ),
        shocks=[shock("country:uk", -5), shock("sector:manufacturing", -3)],
        actions=[act("spend", "sector:manufacturing", 0.8, 6)],
        checks=[
            Check(node="indicator:unemployment", direction=1, real=(0.5, 1.5)),
            Check(node="indicator:unemployment", direction=-1, against="no_policy", real=(-6, -2)),
            Check(node="indicator:deficit", direction=1, real=(8, 13)),
            Check(
                node="indicator:interest_rate",
                direction=-1,
                against="shocks_only",
                real=(-0.7, -0.5),
            ),
        ],
    ),
    Episode(
        name="2016 Brexit referendum",
        happened=(
            "The vote to leave hit relations with the EU and the pound fell by about a sixth "
            "overnight. Imported inflation rose, business investment stalled and growth slowed."
        ),
        shocks=[shock("indicator:exchange_rate", -1.5)],
        actions=[act("diplomatic", "country:eu", -0.8, 6)],
        checks=[
            Check(node="country:uk", direction=-1, real=(-1.5, -0.5)),
            Check(node="indicator:exchange_rate", direction=-1, real=(-20, -10)),
            Check(
                node="indicator:inflation",
                direction=1,
                real=(1.5, 2.5),
            ),
            Check(node="sector:manufacturing", direction=-1),
        ],
    ),
]


def value(state: WorldState, node: str) -> float:
    if node == VOTE:
        return vote_intention(state)
    return getattr(state.node(node), METRICS[state.node(node).kind])


def step(state: WorldState, node: str) -> float:
    return 0.05 if node == VOTE else scale(state, node)


def play(episode: Episode, with_policy: bool = True) -> list[WorldState]:
    """The state after each turn of the episode; shocks and actions land on the first turn."""
    start = state = toy_world()
    states = []
    for turn in range(episode.turns):
        scenario = Scenario(
            title=episode.name,
            briefing=episode.happened,
            affected_nodes=[],
            urgency=0,
            shocks=episode.shocks if turn == 0 else [],
        )
        actions = episode.actions if with_policy and turn == 0 else []
        state = resolve(start, state, scenario, actions, NOTHING, GameConfig())
        states.append(state)
    return states


def peak_change(node: str, states: list[WorldState], base: list[WorldState]) -> float:
    """The largest change from ``base`` in any turn, in standard steps."""
    size = step(states[0], node)
    pairs = zip(states, base, strict=True)
    return max(((value(s, node) - value(b, node)) / size for s, b in pairs), key=abs)


def real_change(node: str, states: list[WorldState], base: list[WorldState]) -> float:
    """The largest change from ``base`` in the units of ``Check.real``."""
    change = max(
        (value(s, node) - value(b, node) for s, b in zip(states, base, strict=True)), key=abs
    )
    if node != VOTE and states[0].node(node).kind == NodeKind.SECTOR:
        return 100 * change / value(toy_world(), node)
    return change


def evaluate(episode: Episode) -> list[CheckResult]:
    states = play(episode)
    without = play(episode, with_policy=False)
    start = [toy_world()] * len(states)
    runs = {
        "start": (states, start),
        "no_policy": (states, without),
        "shocks_only": (without, start),
    }
    results = []
    for check in episode.checks:
        peak = peak_change(check.node, *runs[check.against])
        change = real_change(check.node, *runs[check.against])
        passed = peak * check.direction >= MIN_STEPS
        results.append(
            CheckResult(episode=episode.name, check=check, peak=peak, change=change, passed=passed)
        )
    return results


def run(episodes: list[Episode] = EPISODES) -> list[CheckResult]:
    return [r for e in episodes for r in evaluate(e)]


def report(results: list[CheckResult]) -> str:
    lines = []
    episode = None
    for r in results:
        if r.episode != episode:
            episode = r.episode
            lines.append(f"\n{episode}")
        mark = "ok " if r.passed else ("gap" if r.check.gap else "BAD")
        line = f"  {mark}  {r.check.describe():58s} {r.peak:+.2f} steps"
        if not r.passed and r.check.gap:
            line += f"\n         {r.check.gap}"
        lines.append(line)
    passed = sum(r.passed for r in results)
    gaps = sum(not r.passed and r.check.gap is not None for r in results)
    lines.append(f"\n{passed} of {len(results)} checks pass; {gaps} known gaps.")
    return "\n".join(lines).lstrip()


def sizes(results: list[CheckResult]) -> str:
    """How big each move is against the real one, for the checks that record a real range."""
    sized = [r for r in results if r.size is not None]
    lines = [f"{'Check':<62}{'Engine':>9}{'Real':>20}{'Share':>8}"]
    episode = None
    for r in sized:
        if r.episode != episode:
            episode = r.episode
            lines.append(f"\n{episode}")
        low, high = r.check.real
        lines.append(
            f"  {r.check.describe():60s}{r.change:>+9.2f}{f'{low:+.2f} to {high:+.2f}':>20}"
            f"{r.size:>8.0%}"
        )
    inside = sum(min(r.check.real) <= r.change <= max(r.check.real) for r in sized)
    shares = sorted(r.size for r in sized)
    middle = shares[len(shares) // 2] if shares else 0.0
    lines.append(
        f"\n{inside} of {len(sized)} sized checks fall inside the real range; "
        f"the middle engine move is {middle:.0%} of the real one."
    )
    return "\n".join(lines).lstrip()


def problems(results: list[CheckResult]) -> list[str]:
    """Checks that disagree with their record: should pass but fail, or a gap that now passes."""
    out = []
    for r in results:
        if r.check.gap is None and not r.passed:
            out.append(f"{r.episode}: {r.check.describe()} fails ({r.peak:+.2f} steps)")
        if r.check.gap is not None and r.passed:
            out.append(f"{r.episode}: {r.check.describe()} now passes; remove its gap note")
    return out


def main() -> None:
    results = run()
    print(report(results))
    print()
    print(sizes(results))
    for line in problems(results):
        print(f"PROBLEM: {line}")


if __name__ == "__main__":
    main()
