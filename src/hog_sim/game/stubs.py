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
    Reading,
    Scenario,
    Shock,
)
from hog_sim.core.state import WorldState
from hog_sim.game.interfaces import NeedsClarification
from hog_sim.game.records import TurnRecord
from hog_sim.policy.pledges import broken_by
from hog_sim.population.popularity import policy_events, target_approval
from hog_sim.world.calendar import (
    ECONOMY,
    HOUSEHOLDS,
    calendar_event,
    fill,
    scheduled_scenario,
    spread,
)
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


# Keyword -> node id. Checked in order; first match wins. ``HOUSEHOLDS`` and ``ECONOMY`` stand
# for the whole population and the whole economy: a speech goes to the nation, anything else
# is shared out over the groups or sectors (``spread``), since tax and spending reach the
# country through them. Parliament or the Commons on their own are only a fallback target:
# "announce in Parliament a pay rise" is about the pay rise (E17); whip, backbench or vote
# language aims at the legislature.
_TARGETS = [
    (r"\bwhip|backbench|\bmps\b|\brebel|\bvotes?\b", "institution:legislature"),
    (r"furlough|job retention|wage subsid", "indicator:unemployment"),
    # Taxes paid by people, not firms or sectors (FR-11).
    (r"income tax|national insurance|council tax|\bpaye\b", HOUSEHOLDS),
    (r"\bbills?\b|price cap|\bcap\b", "indicator:energy_prices"),
    (r"energy|gas|electric|emission|carbon|pollut", "sector:energy"),
    (r"bank|financ|city", "sector:finance"),
    (r"manufactur|industr|factor|export", "sector:manufacturing"),
    (r"household|families|poverty", HOUSEHOLDS),
    (r"hous|home|rent|planning|build", "sector:housing"),
    (
        r"public|nhs|hospital|school|pay|union|wages?|nurse|teacher|waiting (?:lists?|times?)|"
        r"backlog|crime|police",
        "sector:public",
    ),
    (r"china", "country:china"),
    (r"\beu\b|europe", "country:eu"),
    (r"business|\bfirms?\b|employer|compan|enterprise|red tape|bureaucra", "group:business"),
    (r"growth|econom|the country", ECONOMY),
    (r"parliament|\bcommons\b", "institution:legislature"),
]
_SIZE = [(r"huge|major|big|massive", 0.9), (r"small|modest|slight", 0.25)]


class KeywordInterpreter:
    """Turns free text into PolicyActions with keyword rules: one per line of a package built
    from "Option:" and "Also:" lines (backlog story RB-1), and one per measure within a line
    ("raise taxes on banks and cut taxes for families" is two, FR-7)."""

    def __init__(self) -> None:
        self._delivery: Delivery | None = None
        self._pledges: list[Pledge] = []
        self._sacked: list[str] = []
        self._readings: list[Reading] = []

    def last_delivery(self) -> Delivery | None:
        return self._delivery

    def last_readings(self) -> list[Reading]:
        return list(self._readings)

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
            parts = [text]
        read = [
            (piece, a)
            for part in parts
            for piece in (
                [part] if calendar_package(part, state, scenario) is not None else measures(part)
            )
            for a in self._one(piece, state, scenario)
        ]
        # "Hold firm on pay but fund nurses": holding firm on one thing doesn't cancel the rest.
        read = [(p, a) for p, a in read if a.kind != "do_nothing"] or read[:1]
        readings = [keyword_reading(part, action) for part, action in read]
        # A side remark next to real measures ("dismiss concerns and fund schools") is not a
        # speech to a guessed target.
        kept = [
            i
            for i, r in enumerate(readings)
            if not (
                r.kind == "communicate" and r.fallback and not re.search(_SPEECH, r.span.lower())
            )
        ]
        if kept and len(kept) < len(read):
            read, readings = [read[i] for i in kept], [readings[i] for i in kept]
        self._readings = readings
        return [action for _, action in read]

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
            broad = HOUSEHOLDS if re.search(r"\btax", main) else None  # a broad tax rise
            first = self._read(main, text, state, scenario, fallback=broad)
            if (
                first.kind == "tax"
                and first.magnitude > 0
                or (first.kind == "spend" and first.magnitude < 0 and first.instrument is None)
            ):
                spend = self._read(_FUNDS.sub("", purpose), text, state, scenario, kind="spend")
                spend = spend.model_copy(update={"magnitude": abs(first.magnitude)})
                return [*spread(first, state), *spread(spend, state)]
        return spread(self._read(main, text, state, scenario, purpose=purpose), state)

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
        target = _target(t) or _target(purpose)
        if target in _TAXED_SECTOR and re.search(r"\bvat\b|\btax|\bduty|\blevy", t):
            # "Cut VAT on energy bills" is a tax on the sector its customers pay, not spending
            # on the bills (FR-11).
            target = _TAXED_SECTOR[target]
        if target is None:
            # A spend for a purpose with no target of its own stays at home.
            skip = ("indicator:", "country:") if kind == "spend" else ("indicator:",)
            target = fallback or next(
                (n for n in scenario.affected_nodes if not n.startswith(skip)),
                ECONOMY if kind == "spend" else state.player_country,
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
                if re.search(r"guarantee|treasury|taxpayer|pay (?:the )?(?:suppliers|firms)", t):
                    # The state pays suppliers the difference (M4), as in 2022.
                    return PolicyAction(
                        kind=kind,
                        target=target,
                        magnitude=size,
                        borne_by="treasury",
                        rationale=f"from: {text}",
                    )
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
        elif _REDUCE_BAD.search(t) and not re.search(_MONEY, t):
            # Cutting a bad thing means acting to reduce it, not cutting spending (FR-8):
            # "cut waiting lists" funds the NHS, "cut red tape" loosens rules.
            if re.search(r"red tape|bureaucra", t):
                kind = "deregulate"
            elif re.search(r"emission|carbon|pollut", t):
                kind = "regulate"
            else:
                kind = "spend"
        elif re.search(rf"{_LOWER}\w*\b.*\b{_TAX_WORD}|\b{_TAX_WORD}\w*\b.*{_LOWER}", t):
            kind, size = "tax", -size
        elif re.search(r"tax|levy|windfall|\bduty\b|\bvat\b|national insurance", t):
            kind = "tax"
        elif re.search(r"deregulat|scrap rules|relax", t):
            kind = "deregulate"
        elif re.search(
            r"regulat|cap\b|ban|limit|tighten|(?:new|tough\w*|strict\w*|safety) rules?", t
        ):
            kind = "regulate"
        elif re.search(r"\bcut|austerity|squeeze", t):
            kind, size = "spend", -size
        elif re.search(r"spend|invest|fund|subsid|support|build|pay rise|\bhelp", t):
            kind = "spend"
        else:
            kind = "communicate"
        borne_by = None
        if kind == "tax" and target.startswith("sector:"):
            # Who the tax falls on (M4): a windfall tax on profits, VAT or a duty on customers.
            if re.search(r"windfall|profit|excess", t):
                borne_by = "profits"
            elif re.search(r"\bvat\b|duty|sales tax|on (?:bills|fuel|petrol)", t):
                borne_by = "consumers"
        return PolicyAction(
            kind=kind, target=target, magnitude=size, borne_by=borne_by, rationale=f"from: {text}"
        )


# Words for a tax, and for lowering one: "cut VAT", "lower income tax", "taxes ... slashed".
_TAX_WORD = r"(?:tax|levy|duty|vat|national insurance)"
_LOWER = r"\b(?:cut|reduc|lower|slash|scrap|abolish|axe)"
# Indicators a tax can't fall on, and the sector it falls on instead.
_TAXED_SECTOR = {"indicator:energy_prices": "sector:energy"}

# Bad things the player wants less of (FR-8). "Cut waiting lists" means act to bring them
# down, unless the words are about money ("cut funding for crime prevention").
_BAD = (
    r"waiting (?:lists?|times?)|backlogs?|crime|poverty|homeless\w*|emissions?|pollution|"
    r"carbon|red tape|bureaucracy"
)
_REDUCE_BAD = re.compile(
    rf"\b(?:cut\w*|reduc\w*|slash\w*|tackl\w*|lower\w*|halv\w*|clear\w*|bring\w* down|end|"
    rf"ending|fight\w*|beat)\b.*\b(?:{_BAD})\b"
    rf"|\b(?:{_BAD})\b.*\b(?:come|comes|go|goes|fall|falls|brought)\s+down\b"
)
_MONEY = r"\bspend|budget|\bfund|\btax|levy|\bduty\b|\bvat\b|subsid|\bpay\b"

# A verb that starts a measure of its own (FR-7). Nouns that double as verbs ("pay", "cap",
# "cuts", "spending", "funding", "building") don't, so "cut taxes and spending" or "tax banks
# and building societies" stay one measure.
_MEASURE_VERB = (
    r"(?:cut(?:ting)?|rais(?:e|es|ing)|increas(?:e|es|ing)|spend|invest(?:s|ing)?|fund|"
    r"subsidi[sz](?:e|es|ing)|support(?:s|ing)?|build|regulat(?:e|es|ing)|"
    r"deregulat(?:e|es|ing)|freez(?:e|es|ing)|relax(?:es|ing)?|scrap(?:s|ping)?|"
    r"help(?:s|ing)?|boost(?:s|ing)?|reduc(?:e|es|ing)|slash(?:es|ing)?|tighten(?:s|ing)?|"
    r"borrow(?:s|ing)?|lower(?:s|ing)?|hik(?:e|es|ing)|introduc(?:e|es|ing)|"
    r"abolish(?:es|ing)?|ax(?:e|es|ing)|hir(?:e|es|ing)|doubl(?:e|es|ing)|expand(?:s|ing)?|"
    r"tackl(?:e|es|ing)|ban|sanction|negotiat(?:e|es|ing)|tax(?:es|ing)?\s+(?:the|on)\b)\b"
)
_MEASURE_LEAD = (
    r"(?:then\s+|also\s+)?(?:we(?:'ll| will)\s+|i(?:'ll| will)\s+)?(?=" + _MEASURE_VERB + ")"
)
_MEASURE_BREAK = re.compile(
    r"\s*[,;]\s*(?:(?:and|but|then|also|plus)\s+)?"
    + _MEASURE_LEAD
    + r"|\s+(?:and|but|while|whereas|then|plus|also)\s+"
    + _MEASURE_LEAD,
    re.I,
)


def measures(text: str) -> list[str]:
    """``text`` split into its separate measures, each with its own verb (FR-7): "raise taxes
    on banks and cut taxes for families" is two measures, "help households and families" one.
    """
    return [m for m in _MEASURE_BREAK.split(text) if m.strip(" .;,!?")] or [text]


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


def _target(t: str) -> str | None:
    if not t:
        return None
    return next((node for pat, node in _TARGETS if re.search(pat, t)), None)


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


# Words that say a speech was meant; without one, "communicate" is only the default.
_SPEECH = (
    r"speech|announce|campaign|statement|reassur|explain|tell|communicat|talk|negotiat|rally|"
    r"reach out|whip|address|appeal|promise|condemn|warn|praise|back\b"
)


_UP = re.compile(r"\b(?:rais|increas|boost|hike|lift|higher|expand|doubl|more\b)")
_DOWN = re.compile(r"\b(?:cut|reduc|lower|slash|scrap|abolish|freez|axe|less\b)")


def _mixed(t: str) -> bool:
    """Both a rise and a cut in one measure, other than "X to Y", where Y is X's purpose."""
    main, _ = _split_purpose(t)
    return bool(_UP.search(main) and _DOWN.search(main))


def keyword_reading(part: str, action: PolicyAction) -> Reading:
    """How sure keyword matching is of ``action``, read from ``part`` (review D6). A target
    no keyword named was guessed from the scenario; a speech no word asked for is only the
    fallback kind."""
    span = re.sub(r"^\s*(?:Option|Also):\s*", "", part).strip()
    t = span.lower()
    named = any(re.search(pat, t) for pat, _ in _TARGETS)
    fallback = action.kind != "do_nothing" and not named
    if fallback:
        confidence = 0.4
    elif action.kind == "communicate" and not re.search(_SPEECH, t):
        confidence = 0.5
    elif action.kind != "do_nothing" and _mixed(t):
        # Asks for something up and something down in one measure that couldn't be split
        # ("raise taxes on banks, families get tax cuts"): one action may not be what was
        # meant, so the player is asked to check it (FR-7).
        confidence = 0.5
    else:
        confidence = 0.8
    return Reading(
        kind=action.kind,
        target=action.target,
        span=span,
        confidence=confidence,
        fallback=fallback,
    )


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
            # "No new tax cuts" rules out tax cuts, not rises (FR-12): the cut belongs to the tax.
            tax_cut = re.search(
                r"\btax\w*\s+(?:cuts?|reductions?|breaks?)\b|\b(?:cut|lower|reduc)\w*\s+"
                r"(?:\w+\s+){0,2}tax",
                t,
            )
            tax_rise = re.search(
                r"\btax\w*\s+(?:rises?|increases?|hikes?)\b|\b(?:rais|increas|hik|put up)\w*\s+"
                r"(?:\w+\s+){0,2}tax",
                t,
            )
            if tax_cut and not tax_rise:
                pledge = Pledge(text=words, kind="tax", direction="down")
            elif re.search(r"\btax", t) and re.search(r"rais|increas|\bnew\b|\brise|put up", t):
                pledge = Pledge(text=words, kind="tax", direction="up")
            elif re.search(r"\btax", t) and re.search(r"\bcut|\blower|reduc", t):
                # "We will not cut taxes" rules out a tax cut, not a spending cut (E10).
                pledge = Pledge(text=words, kind="tax", direction="down")
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
        # Not "replace the Chancellor's fiscal rule": a possessive names something of theirs.
        r"\b(?:sack|fire|dismiss|replace)\s+(?:the\s+|my\s+|our\s+)?(?:" + who + r")\b"
        r"(?!['’]s\b)"
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
