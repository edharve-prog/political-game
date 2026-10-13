"""Rule-based stand-ins for the LLM parts, so the game is playable and testable offline.

Project 5 replaces ``CannedScenarios`` and ``KeywordInterpreter``; Project 6 replaces
``EngineForecaster``. All three satisfy the protocols in ``game/interfaces.py``.
"""

from __future__ import annotations

import re

from hog_sim.core.config import make_rng
from hog_sim.core.models import (
    ApprovalEvent,
    Delivery,
    Outcome,
    Pledge,
    PolicyAction,
    Scenario,
    Shock,
)
from hog_sim.core.state import WorldState
from hog_sim.game.interfaces import NeedsClarification
from hog_sim.game.records import TurnRecord
from hog_sim.policy.pledges import broken_by
from hog_sim.population.popularity import policy_events, target_approval
from hog_sim.world.calendar import calendar_event, fill, scheduled_scenario
from hog_sim.world.cast import active_cast
from hog_sim.world.graph import build_graph, exposed_groups
from hog_sim.world.propagation import DeltaDistribution, apply_deltas

_SCENARIOS = [
    Scenario(
        title="Gas prices spike",
        briefing="A cold snap and supply outages in Europe push wholesale gas prices up sharply.",
        affected_nodes=["indicator:energy_prices", "sector:energy"],
        urgency=0.8,
        suggested_options=["Cap household bills", "Tax energy windfall profits", "Wait it out"],
        shocks=[Shock(node="indicator:energy_prices", delta=1.5)],
    ),
    Scenario(
        title="Inflation surprise",
        briefing=(
            "Monthly figures show prices rising faster than forecast; markets expect rates to rise."
        ),
        affected_nodes=["indicator:inflation", "indicator:interest_rate"],
        urgency=0.6,
        suggested_options=["Cut public spending", "Back the central bank", "Do nothing"],
        shocks=[Shock(node="indicator:inflation", delta=0.5)],
    ),
    Scenario(
        title="Housing crunch",
        briefing="Rents and house prices hit records and young voters are angry.",
        affected_nodes=["indicator:house_prices", "sector:housing", "group:young_renters"],
        urgency=0.5,
        suggested_options=["Spend on housebuilding", "Deregulate planning", "Do nothing"],
        shocks=[Shock(node="indicator:house_prices", delta=0.8)],
    ),
    Scenario(
        title="Trade dispute with China",
        briefing="China slaps tariffs on British exports after a diplomatic row.",
        affected_nodes=["country:china", "sector:manufacturing"],
        urgency=0.6,
        suggested_options=["Negotiate", "Retaliate", "Support manufacturers"],
        shocks=[Shock(node="sector:manufacturing", delta=-0.8)],
    ),
    Scenario(
        title="Public sector pay row",
        briefing="Unions threaten strikes across schools and hospitals over pay.",
        affected_nodes=["sector:public", "group:public_workers"],
        urgency=0.7,
        suggested_options=["Fund a pay rise", "Hold firm", "Negotiate"],
    ),
]


class CannedScenarios:
    """Picks a scenario from a fixed list, seeded and without immediate repeats. On calendar
    turns (story SD-6) it is the calendar event instead."""

    def __init__(self, seed: int = 0, calendar: bool = True) -> None:
        self.seed = seed
        self.calendar = calendar

    def next_scenario(self, state: WorldState, history: list[TurnRecord]) -> Scenario:
        event = calendar_event(state.turn) if self.calendar else None
        if event is not None:
            return scheduled_scenario(event, state)
        last = history[-1].scenario.title if history else None
        options = [s for s in _SCENARIOS if s.title != last]
        return make_rng(self.seed, state.turn, "scenario").choice(options).model_copy(deep=True)


# Keyword -> node id. Checked in order; first match wins. ``COUNTRY`` stands for the player's
# own country. Parliament or the Commons on their own are only a fallback target: "announce in
# Parliament a pay rise" is about the pay rise (E17); whip, backbench or vote language aims
# at the legislature.
COUNTRY = "@country"
_TARGETS = [
    (r"\bwhip|backbench|\bmps\b|\brebel|\bvotes?\b", "institution:legislature"),
    (r"furlough|job retention|wage subsid", "indicator:unemployment"),
    (r"\bbills?\b|price cap|\bcap\b", "indicator:energy_prices"),
    (r"energy|gas|electric", "sector:energy"),
    (r"bank|financ|city", "sector:finance"),
    (r"manufactur|industr|factor|export", "sector:manufacturing"),
    (r"household|families", COUNTRY),
    (r"hous|home|rent|planning|build", "sector:housing"),
    (r"public|nhs|hospital|school|pay|union|wages?|nurse|teacher", "sector:public"),
    (r"china", "country:china"),
    (r"\beu\b|europe", "country:eu"),
    (r"business|\bfirms?\b|employer|compan|enterprise", "group:business"),
    (r"growth|econom|the country", COUNTRY),
    (r"parliament|\bcommons\b", "institution:legislature"),
]
_SIZE = [(r"huge|major|big|massive", 0.9), (r"small|modest|slight", 0.25)]


class KeywordInterpreter:
    """Turns free text into PolicyActions with keyword rules: one per line of a package built
    from "Option:" and "Also:" lines (backlog story RB-1), otherwise one for the whole text."""

    def __init__(self) -> None:
        self._delivery: Delivery | None = None
        self._pledges: list[Pledge] = []
        self._sacked: list[str] = []

    def last_delivery(self) -> Delivery | None:
        return self._delivery

    def last_pledges(self) -> list[Pledge]:
        return list(self._pledges)

    def last_sacked(self) -> list[str]:
        return list(self._sacked)

    def delivery_for(self, text: str, state: WorldState) -> Delivery:
        return keyword_delivery(text, state)

    def pledges_for(self, text: str) -> list[Pledge]:
        return keyword_pledges(text)[0]

    def sacked_for(self, text: str, state: WorldState) -> list[str]:
        return keyword_sackings(text, state)[0]

    def interpret(self, text: str, state: WorldState, scenario: Scenario) -> list[PolicyAction]:
        self._delivery = keyword_delivery(text, state)
        self._pledges, text = keyword_pledges(text)
        self._sacked, text = keyword_sackings(text, state)
        unsupported(text)
        parts = re.findall(r"^\s*(?:Option|Also):\s*(.+)$", text, re.M)
        if len(parts) < 2:
            return self._one(text, state, scenario)
        actions = [a for part in parts for a in self._one(part, state, scenario)]
        return [a for a in actions if a.kind != "do_nothing"] or actions[:1]

    def _one(self, text: str, state: WorldState, scenario: Scenario) -> list[PolicyAction]:
        package = calendar_package(text, state, scenario)
        if package is not None:
            return package
        t = _VENUE.sub(" ", re.sub(r"^\s*(?:option|also):\s*", "", text.lower())).strip()
        if not t.strip(" .;,!?") or _DO_NOTHING.search(t) or _STEADY.search(t):
            return [PolicyAction(kind="do_nothing", target=state.player_country, magnitude=0)]
        # "Raise taxes to fund schools": the verb and target come from the instruction, not
        # from its purpose ("cut" in "spend to cut waiting lists" is not a spending cut).
        main, purpose = _split_purpose(t)
        if purpose and _FUNDS.match(purpose):
            first = self._read(main, text, state, scenario, fallback=state.player_country)
            if (
                first.kind == "tax"
                and first.magnitude > 0
                or (first.kind == "spend" and first.magnitude < 0 and first.instrument is None)
            ):
                spend = self._read(
                    _FUNDS.sub("", purpose),
                    text,
                    state,
                    scenario,
                    fallback=state.player_country,
                    kind="spend",
                )
                return [first, spend.model_copy(update={"magnitude": abs(first.magnitude)})]
        return [self._read(main, text, state, scenario, purpose=purpose)]

    def _read(
        self,
        t: str,
        text: str,
        state: WorldState,
        scenario: Scenario,
        purpose: str = "",
        fallback: str | None = None,
        kind: str | None = None,
    ) -> PolicyAction:
        target = _target(t, state) or _target(purpose, state)
        if target is None:
            target = fallback or next(
                (n for n in scenario.affected_nodes if not n.startswith("indicator:")),
                state.player_country,
            )
        size = next((m for pat, m in _SIZE if re.search(pat, t)), 0.5)
        if kind is not None:
            return PolicyAction(kind=kind, target=target, magnitude=size, rationale=f"from: {text}")
        if not _has_verb(t) and purpose:
            t = f"{t} {purpose}"

        if target.startswith("indicator:"):
            # Aimed at a price itself: the player wants it down unless they say otherwise.
            if re.search(r"\bcap|freez|limit|regulat", t):
                kind = "regulate"
                size = size if re.search(r"\braise|\bincrease|\bhigher", t) else -size
            else:
                # Spending on it (a furlough scheme, help with bills). "Raise" or "cut" is the
                # spending's own way, not the indicator's: cutting it lets the indicator rise.
                kind = "spend"
                if re.search(_CUT_SPENDING, t):
                    return PolicyAction(
                        kind=kind,
                        target=target,
                        magnitude=size,
                        instrument="cut",
                        rationale=f"from: {text}",
                    )
                size = -size
        elif target.startswith("country:") and target != state.player_country:
            kind = "diplomatic"
            size = -size if re.search(r"retaliat|sanction|tariff", t) else size
        elif re.search(r"\bcut\w*\b.*\b(?:tax|levy|duty)|\b(?:tax|levy|duty)\w*\b.*\bcut", t):
            kind, size = "tax", -size
        elif re.search(r"tax|levy|windfall", t):
            kind = "tax"
        elif re.search(r"deregulat|scrap rules|relax", t):
            kind = "deregulate"
        elif re.search(r"regulat|cap\b|ban|limit|tighten|\brules?\b", t):
            kind = "regulate"
        elif re.search(r"\bcut|austerity|squeeze", t):
            kind, size = "spend", -size
        elif re.search(r"spend|invest|fund|subsid|support|build|pay rise|\bhelp", t):
            kind = "spend"
        else:
            kind = "communicate"
        return PolicyAction(kind=kind, target=target, magnitude=size, rationale=f"from: {text}")


# A clause that asks for nothing: "do nothing", "wait it out", "hold firm". Only at the start
# of the clause (except holding firm), so "cut waiting lists" or "nothing less than a pay
# rise" still act (E15).
_DO_NOTHING = re.compile(
    r"^(?:do nothing|nothing\W*$|wait\b|sit tight|stay the course)|\b(?:hold|stand) firm\b"
)
_STEADY = re.compile(r"\bno (?:big |major )?changes?\b|\bsteady (?:budget|as she goes)")

# Where it is announced, which is delivery rather than a target (E17).
_VENUE = re.compile(
    r"\b(?:in|to|before|at|into)\s+(?:the\s+)?(parliament|house of commons|commons)\b"
)

# "X to Y" where Y is what X is for: "raise taxes to fund schools", "spend big to cut
# waiting lists". Not "reach out to the centre" or "talk to unions".
_PURPOSE = re.compile(
    r"\s+(?:in order |so as )?to\s+(?=(?:fund|pay|finance|boost|cut|bring|help|support|"
    r"reduce|lower|raise|protect|tackle|ease|cool|curb|keep|get|make|stop|end|save|create|"
    r"grow|lift|build|clear|balance|restore|calm)\b)"
)
_FUNDS = re.compile(r"^(?:fund|pay for|finance|help fund|help pay for)\s+")
_VERBS = re.compile(
    r"\b(?:cut|tax|levy|spend|invest|fund|subsid|support|build|regulat|deregulat|ban|cap|"
    r"limit|freez|raise|increase|relax|scrap|austerity|squeeze|retaliat|sanction|negotiat)"
)


def _has_verb(t: str) -> bool:
    return bool(_VERBS.search(t))


def _split_purpose(t: str) -> tuple[str, str]:
    parts = _PURPOSE.split(t, maxsplit=1)
    return (parts[0], parts[1]) if len(parts) == 2 and parts[0].strip() else (t, "")


def _target(t: str, state: WorldState) -> str | None:
    if not t:
        return None
    node = next((node for pat, node in _TARGETS if re.search(pat, t)), None)
    return state.player_country if node == COUNTRY else node


def calendar_package(text: str, state: WorldState, scenario: Scenario) -> list[PolicyAction] | None:
    """The ready-made actions for one of a calendar event's own options (E16), or None.

    The options' wording ("Raise taxes to fund public services") is for the player; reading
    it with keywords would aim at whatever the scenario lists first."""
    event = calendar_event(state.turn)
    if event is None or scenario.source != "scheduled" or scenario.title != event.title:
        return None
    filled = fill(event, state)
    wanted = _plain(text)
    for option, package in zip(filled.options, filled.packages, strict=True):
        if _plain(option) == wanted and package:
            return [a.model_copy(update={"rationale": f"from: {option}"}) for a in package]
    return None


def _plain(text: str) -> str:
    text = re.sub(r"^\s*(?:option|also):\s*", "", text.strip(), flags=re.I)
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


# Who "consulting X" means, for the keyword interpreter's delivery (story RB-4).
_CONSULT_WHO = [
    (r"union|workers|staff|nurses|teachers", "group:public_workers"),
    (r"business|industry|employers|firms|cbi", "group:business"),
    (r"pensioner", "group:pensioners"),
    (r"renter|tenant", "group:young_renters"),
    (r"\bmps\b|backbench|parliament|party", "institution:legislature"),
    (r"bank of england|the bank|governor", "institution:central_bank"),
]


def keyword_delivery(text: str, state: WorldState) -> Delivery:
    t = text.lower()
    consulted = []
    if re.search(r"consult|talks? with|negotiat|meet|sit down with|work with", t):
        known = set(state.node_ids())
        consulted = [n for pat, n in _CONSULT_WHO if re.search(pat, t) and n in known]
    phased = re.search(r"phase|gradual|staged|stagger|over (?:the next )?\w+ (?:years|months)", t)
    venue = _VENUE.search(t)
    return Delivery(
        consulted=consulted,
        speed="phased" if phased else "immediate",
        venue={"parliament": "Parliament"}.get(venue.group(1), "the Commons") if venue else "",
    )


_NEGATION = r"\b(?:no|not|never|won't|will not|nor)\b"

# Cutting a programme rather than spending on it: "cut spending on furlough", "end the
# furlough scheme".
_CUT_SPENDING = (
    r"\b(?:cut|cuts|cutting|reduc\w*|scrap\w*|end|ends|ending|withdraw\w*|slash\w*|"
    r"wind(?:s|ing)? down|phase out|axe|abolish\w*|stop)\b"
)

# Where one sentence holds two separate instructions: "raise taxes but do not cut pensions".
_CLAUSE_BREAK = (
    r"(?<=[.;!?])\s+|\n|,?\s+\b(?:but|while|whereas)\b\s+"
    r"|,?\s+\band\b\s+(?=(?:do not|don't|dont|never|will not|won't|we will not|not)\b)"
)

# A negation that governs what follows it: "do not increase", "we won't sack", "never cut".
# Bare "no" counts only before a verb ("no raising of taxes" is a pledge, handled above), so
# "ban no-fault evictions" or "fund it and no more" are not read as refusals.
_REFUSAL = re.compile(
    r"\b(?:do not|does not|don't|dont|doesn't|did not|didn't|will not|won't|shall not|"
    r"shan't|should not|shouldn't|must not|mustn't|cannot|can't|never|refuse to|"
    r"not to|rather than|instead of)\b"
)
_CONDITION = re.compile(r"^\s*(?:(?:option|also):\s*)?(?:if|unless|only if|in case|should)\b")


def clauses(text: str) -> list[str]:
    """``text`` split into the separate instructions it gives."""
    return [c for c in re.split(_CLAUSE_BREAK, text) if c and c.strip(" .;,!?")]


def unsupported(text: str) -> None:
    """Raise ``NeedsClarification`` for what keyword matching can't read safely.

    A refusal ("do not increase public spending") or a condition ("if inflation rises, cut
    spending") would otherwise be read as the plain instruction inside it, which can be the
    opposite of what the player meant. Pledges and sackings are taken out before this, so
    "we will not raise taxes" and "do not sack the Chancellor" still work."""
    for clause in clauses(text):
        words = re.sub(r"^\s*(?:Option|Also):\s*", "", clause).strip(" .;,!?")
        t = words.lower()
        if _REFUSAL.search(t):
            raise NeedsClarification(
                f'Offline play can\'t read "{words}": it says what not to do. Say what you '
                "will do instead (for example hold it where it is, or do nothing), or make it a "
                'promise ("we will not raise taxes", "no cuts to schools").'
            )
        if _CONDITION.search(t):
            raise NeedsClarification(
                f'Offline play can\'t act on a condition ("{words}"). Say what to do this '
                "turn; you can respond again next turn if things change."
            )


def keyword_pledges(text: str) -> tuple[list[Pledge], str]:
    """Promises in ``text`` ("we will not raise taxes", "no cuts to schools"), and the text
    with those sentences taken out so they don't also read as actions (story SD-5)."""
    pledges, kept = [], []
    for sentence in clauses(text):
        t = sentence.lower()
        words = re.sub(r"^\s*(?:Option|Also):\s*", "", sentence).strip(" .;!")
        pledge = None
        if re.search(_NEGATION, t):
            if re.search(r"\btax", t) and re.search(r"rais|increas|\bnew\b|\brise|put up", t):
                pledge = Pledge(text=words, kind="tax", direction="up")
            elif re.search(r"\bcuts?\b", t):
                target = next((node for pat, node in _TARGETS if re.search(pat, t)), None)
                pledge = Pledge(text=words, kind="spend", direction="down", target=target)
        if pledge:
            pledges.append(pledge)
        else:
            kept.append(sentence)
    return pledges, "\n".join(kept) if pledges else text


def keyword_sackings(text: str, state: WorldState) -> tuple[list[str], str]:
    """Ministers ``text`` sacks ("sack the Chancellor", "replace Owen Pryce"), and the text
    with each sacking taken out so the rest of the sentence still reads (story SD-4).

    The verb has to govern the minister (E18): "tighten fire safety rules in housing" or
    "replace housing benefit" sack nobody."""
    sacked, kept = [], []
    changed = False
    for sentence in clauses(text):
        t = sentence.lower()
        hits = []
        refused = False
        if re.search(_SACK, t):
            # "Do not sack the Chancellor", "I won't fire anyone": a refusal sacks nobody.
            refused = bool(re.search(_REFUSAL.pattern + r"(?:\W+\w+){0,2}?\W+" + _SACK, t))
            for character in active_cast(state):
                if not character.minister:
                    continue
                found = re.search(_sacking(character), t)
                if found:
                    hits.append(character.id)
                    t = t[: found.start()] + " " + t[found.end() :]
                    sentence = sentence[: found.start()] + " " + sentence[found.end() :]
        if refused:
            changed = True
            continue
        if hits:
            sacked += hits
            changed = True
            rest = re.sub(r"^[\s,;]*(?:and|then)?\s*", "", sentence, flags=re.I).strip(" .;,!?")
            if rest:
                kept.append(rest)
        else:
            kept.append(sentence)
    return sacked, "\n".join(kept) if changed else text


_SACK = r"\b(?:sack|fire|dismiss|sacking|replace)\b"


def _sacking(character) -> str:
    """A sacking verb with the minister as its object: their name or surname, their role, or
    the role's first long word ("the Chancellor"), with "secretary" or "minister" after it
    when the role has one ("the Housing Secretary")."""
    role = character.role.lower()
    words = [w for w in re.findall(r"[a-z]+", role) if len(w) > 4]
    names = [re.escape(character.name.lower()), re.escape(character.name.split()[-1].lower())]
    names.append(re.escape(role))
    if words:
        title = re.search(r"\b(secretary|minister)\b", role)
        names.append(re.escape(words[0]) + (r"\s+(?:secretary|minister)" if title else ""))
    who = "|".join(names)
    return (
        r"\b(?:sack|fire|dismiss|replace)\s+(?:the\s+|my\s+|our\s+)?(?:" + who + r")\b"
        r"(?:\s+with\s+[a-z ]+?(?=$|[.,;]|\s+and\b))?"
    )


class EngineForecaster:
    """Three candidates around the engine's expectation: as expected, backlash, welcomed.

    The engine decides who feels what: groups whose target approval the forecast raises are
    the ones who welcome the response, and groups it lowers are the ones who push back. The
    two turns are equally likely, so acting is not penalised on average (EB-7) and a response
    that helps people gets amplified rather than punished.
    """

    TURN = 2  # judge the response by the engine's forecast this many turns out
    THRESHOLD = 0.002  # smaller moves in target approval count as unaffected
    EFFECT = 0.03  # size of the reaction for groups the forecast moves
    FALLBACK_EFFECT = 0.02  # size when the forecast moves nobody, felt by the exposed groups

    def forecast(
        self,
        state: WorldState,
        scenario: Scenario,
        actions: list[PolicyAction],
        engine: DeltaDistribution,
        delivery: Delivery | None = None,
        limits: list[str] | None = None,
        sacked: list[str] | None = None,
    ) -> list[Outcome]:
        graph = build_graph(state)
        exposed: set[str] = set()
        for action in actions:
            if action.kind != "do_nothing":
                exposed |= exposed_groups(graph, action.target)
                if action.target in state.groups:
                    exposed.add(action.target)
        exposed |= {n for n in scenario.affected_nodes if n.startswith("group:")}
        expected = {
            n: round(v, 3) for n, v in engine.at(0).items() if n.startswith("indicator:") and v
        }

        # Judge the response by what it changes, not by effects already in flight.
        projected = apply_deltas(state, engine.response_at(min(self.TURN, engine.horizon - 1)))
        projected.events = [*state.events, *policy_events(actions, state)]
        before, after = target_approval(state, state), target_approval(projected, state)
        change = {g: after[g] - before[g] for g in state.groups}
        winners = sorted(g for g, c in change.items() if c > self.THRESHOLD)
        losers = sorted(g for g, c in change.items() if c < -self.THRESHOLD)

        def event(name: str, groups: list[str], size: float) -> list[ApprovalEvent]:
            if groups:
                return [ApprovalEvent(name=name, group_effects={g: size for g in groups})]
            if exposed:
                fallback = self.FALLBACK_EFFECT if size > 0 else -self.FALLBACK_EFFECT
                return [
                    ApprovalEvent(name=name, group_effects={g: fallback for g in sorted(exposed)})
                ]
            return []

        # Consulting first takes some of the risk of a backlash away (story RB-4), without
        # making a warm welcome any likelier.
        backlash = 0.12 if delivery is not None and delivery.consulted else 0.2
        broken = "".join(
            f' Critics say it breaks the pledge "{p.text}".' for p in broken_by(state, actions)
        )
        broken += "".join(
            f" {state.characters[c].name} is sacked as {state.characters[c].role}."
            for c in sacked or []
            if c in state.characters
        )
        outcomes = [
            Outcome(
                narrative=f"{scenario.title}: the response lands broadly as expected.",
                indicator_deltas=expected,
                probability=0.8 - backlash,
            ),
            Outcome(
                narrative=f"{scenario.title}: the response sparks a backlash from those affected.",
                indicator_deltas=expected,
                approval_events=event("Backlash", losers, -self.EFFECT),
                probability=backlash,
            ),
            Outcome(
                narrative=f"{scenario.title}: the response is welcomed as decisive.",
                indicator_deltas=expected,
                approval_events=event("Seen as decisive", winners, self.EFFECT),
                probability=0.2,
            ),
        ]
        for outcome in outcomes:
            outcome.narrative += broken
        return outcomes
