# Backlog: Richer Scenarios and Responses

Raised by Ed on 2026-10-01 after the first play-through: "the scenarios are basic and
repeating. It needs 100 times more complexity and easier to combine responses, come up with new
responses which feed into the LLM response."

Two causes sit behind that:

- **Offline mode has five fixed scenarios** (`game/stubs.py`). That is where the repetition
  comes from. Claude mode (`--llm`) writes a new scenario every turn but still sees one issue at
  a time, has no memory of storylines, and works from a toy world of 3 countries, 5 sectors and
  4 groups. PR #7 makes the mode visible at startup.
- **Responses are a single free-text box**. Suggested options are shown but can't be picked,
  combined, tuned or saved, and the player never sees how their words were turned into actions.

"More complexity" here means more going on and more that connects, not longer text. Each
briefing should stay readable in under a minute.

The epics map to new Projects 13 and 14 in the implementation plan, with two stories feeding
Projects 2 and 10. Story ids are stable: refer to them in PRs.

Priority: **P1** = needed before the 30-turn LLM review is worth doing, **P2** = next,
**P3** = later.

---

## Epic A: Living storylines (Project 13)

Turn one-shot scenarios into ongoing stories that escalate, branch and come back.

### SD-1 Storyline threads (P1, done in PR #21 for Claude mode, merged)
As a player, I want issues to continue across turns, so my earlier decisions come back to me.
- A `Storyline` record (id, title, stage, open/closed, linked nodes, turn opened, last turn
  touched) is saved in `WorldState` and survives save, resume and replay.
- Each generated scenario either continues an open storyline or opens a new one, and says which.
- An open storyline that is not addressed escalates at least one stage within 4 turns, or its
  urgency rises.
- A storyline closes only through an outcome that resolves it, never silently.
- Test: on a 20-turn FakeClient game, at least one storyline spans 3 or more turns and the
  scenario prompt for turn N includes its stage history.
- Note (PR #21): ignored storylines escalate after 3 idle turns. Offline library scenarios are
  still one-offs; giving them follow-up stages is a later content task.

### SD-2 Several issues per turn (P1, done in PR #22 for Claude mode, merged)
As a player, I want an in-tray of 2 to 4 issues each turn, so I have to prioritise.
- Each turn shows one lead issue plus 1 to 3 secondary items (one-paragraph briefings).
- The player can respond to any subset. An item left alone has consequences: it carries
  over, escalates or resolves on its own, and the outcome narrative says which.
- Engine resolution stays one deterministic step per turn, with all responses' actions
  combined.
- Replay reproduces the same state from the log.
- Note (PR #22): an item counts as acted on when an action targets one of its nodes. Minor
  items (urgency under 0.3) left alone fade; others carry over and escalate as storylines.

### SD-3 Scenario variety and anti-repetition (P1, done: offline in PR #11, Claude mode in PR #17)
As a player, I want each turn to feel different from the last.
- The scenario prompt receives the last 8 scenario titles and categories and must avoid them
  unless it is continuing an existing storyline (SD-1).
- The issue taxonomy has at least 12 categories: economy, energy, housing, health, education,
  crime and justice, immigration, defence and security, foreign affairs, environment and
  disasters, party politics and scandal, media and technology.
- Over a 30-turn Claude game, no category supplies more than 25% of lead issues and no two
  lead titles are near-duplicates (embedding or token-overlap check in an eval script).
- Offline mode gets a library of at least 60 handwritten scenarios with seeded,
  state-weighted selection and no repeat within 15 turns, so offline play stops feeling
  canned.

### SD-9 Storylines come to a head (P1, from the 30-turn review; done in PR #37, merged)
As a player, I want long-running issues to reach a climax and end, so the premiership
moves on.
- A storyline at stage 6, or 10 turns old, is in its final stage: the writer is told so, and
  its candidates must include one that resolves it.
- At most one in-tray item continues an open storyline; a new lead storyline opens at least
  every third turn.
- In a 30-turn Claude game, at least 10 distinct lead storylines, and none on the agenda in
  more than 40% of turns (the review had 6, and one on the agenda every turn).

### SD-4 Recurring characters (P2) - done (PR #44)
As a player, I want to deal with people, not abstractions.
- The game has a cast of fictional characters: cabinet ministers, the opposition leader,
  union general secretaries, the Bank governor, a tabloid editor, and backbench faction
  leaders. Each has a name, role, agenda, loyalty and a short memory of their dealings with
  the player.
- Scenarios may quote and involve them by id. Foreign leaders stay role-titled (existing
  rule), and no real living person is ever named.
- How the player treats a character (sacking, backing or ignoring them) changes that
  character's loyalty, and loyalty shows up in later scenarios (resignations, leaks,
  support).
- Done: `world/cast.py` holds eight invented people with agendas, loyalty and memory.
  Loyalty follows backing, crossing, consulting and ignoring them. Ministers can be sacked or
  resign and are replaced. Every prompt lists the cast, and scenarios name the people
  involved.

### SD-5 Consequence callbacks (P2) - done (PR #43)
As a player, I want past promises and choices to be remembered.
- Commitments the player makes ("we will not raise taxes") are extracted into a `Pledge`
  list.
- Scenarios and outcomes can reference a broken or kept pledge. A broken pledge triggers an
  ApprovalEvent against the groups that cared about it.
- The dashboard lists active pledges.
- Done: the interpreter (Claude and offline) extracts pledges; `policy/pledges.py` breaks them
  in `resolve()` and adds the approval hit; every prompt lists them; the player is warned
  before confirming a response that breaks one. Not done: the review's observation that most
  storylines end off-screen. That needs its own story if it still matters after the next
  review.

### SD-6 Scheduled and calendar events (P2) - done (PR #45)
As a player, I want a political calendar: budgets, party conference, PMQs, summits and
by-elections.
- A calendar of recurring events runs by turn, for example a budget every 12 turns and
  conference every 12 turns, offset by 6.
- On those turns the lead issue is the calendar event and its options are set by the event
  type (a budget asks for tax and spending choices).
- Done: `world/calendar.py` holds the Budget, a summit, party conference and a by-election,
  each every 12 turns. PMQs are left out because a turn is a month.

### SD-7 World events and outside shocks (P2, joins Project 7)
As a player, I want the wider world to intrude: wars, pandemics, commodity spikes and
elections abroad.
- About 1 turn in 5 has an outside shock that the engine applies before the player acts. It
  comes from news (Project 7) when news is on, or from a seeded event table when it is off.
- Foreign countries react to UK actions through the graph edges, and the outcome narrative
  names that reaction.
- Crises cost something if ignored (back-end review C4, enhancement 8). Every scenario,
  including in Claude mode, carries a shock, and that shock grows each turn it is left
  unaddressed. This also applies to SD-1 storylines.

### SD-8 Bigger world (P2, joins Project 2)
As a player, I want more groups, sectors and countries, so policies hit people unevenly.
- At least 12 population groups, split by age, income, region and occupation.
- At least 12 sectors, at least 8 countries or blocs, and new institutions: courts, media,
  unions and the governing party.
- All existing engine and popularity tests still pass, and the stability test runs 120 turns
  without blowing up.

---

## Epic B: Response builder (Project 14)

Make responding expressive: pick, combine, tune and invent policies, and see how they were
read.

### RB-1 Pick and combine options (P1, done in PR #19)
As a player, I want to choose several suggested options and add my own words, so I can build
a package.
- Options are numbered. Input like `1 3 + also freeze rail fares` selects options 1 and 3 and
  adds the free text.
- The interpreter receives the selected options and the free text as one package and returns
  one combined action list.
- Contradictory picks (for example "cut spending" and "spend more on housing") get a
  clarifying question instead of silently cancelling each other out.

### RB-2 Review before committing (P1, done in PR #19)
As a player, I want to see how my response was understood before the turn resolves.
- After interpretation, the game shows the action list: kind, target, size, duration, and any
  blocked by feasibility with the reason.
- The player can confirm, edit an action's size or duration (`2 size 0.3`), drop one, or
  rephrase.
- Only confirmed actions reach the engine. The log stores the confirmed list, so replay is
  unchanged.

### RB-3 Ask advisers for new options (P1, done in PR #23, merged)
As a player, I want to ask for more or different options ("what would the left of the party
want?", "something cheaper").
- The command `advise <question>` returns 2 to 4 new options from Claude, grounded in the
  state summary and the current issue, each with a one-line expected trade-off.
- New options join the numbered list and can be combined as in RB-1.
- Asking advisers does not advance the turn. Each call shows in the per-turn usage line.

### RB-4 Response feeds the outcome (P1, done in PR #24, merged)
As a player, I want how I acted to matter, not just what I did: framing, timing, consulting
people, and who announces it.
- The interpreter extracts *delivery* fields alongside actions: framing or message,
  announcement venue, stakeholders consulted, and speed (immediate or phased).
- These fields go into the outcome prompt (Project 6) and into base-rate and judge scoring.
  For example, consulting unions lowers the chance of a strike outcome.
- Outcome narratives quote or refer to the player's own framing at least once.
- Test: the same actions with "consulted unions" against "imposed overnight" give different
  probabilities for strike-tagged candidates, on a FakeClient fixture.

### RB-8 Blocked measures reach the outcome (P1, from the 30-turn review; done in PR #36, merged)
As a player, I want the outcome to reflect that the Commons blocked my measure.
- The limit notes go to the outcome prompt as "Blocked: ...", alongside the delivery.
- No chosen narrative claims a blocked measure took effect (the review's T18 did).

### RB-9 Fewer validation retries (P2, from the 30-turn review) - done (PR #42)
- A bare id (`business`) maps to its unique prefixed id (`group:business`) before checking.
- A clarifying question returned alongside actions is dropped, not retried.
- `do_nothing` mixed with real actions is dropped from LLM interpretations, as offline.
- The review had 11 retries in 131 calls; target under 3 per 30 turns.
- Done: `structured_call` takes a `repair` step that runs before the checks. The interpreter
  and the outcome writer use it for the three slips above. A closed storyline continued by
  the scenario writer still costs a retry, since there is no single right fix.

### RB-5 Policy library (P2) - done (PR #47)
As a player, I want to save a package I liked and reuse or adapt it later.
- `save <name>` stores the confirmed action list, and `use <name>` loads it into the
  builder for editing.
- The library is saved per game and exportable.
- Done: `game/packages.py` keeps packages in the save file. `save <name>` works at the
  review step, `use <name>` and `packages` at the response prompt, and
  `hog-sim packages list|export|import` from the command line.

### RB-6 New policy levers (P2)
As a player, I want to try things the game doesn't list yet, such as a citizens' assembly,
nationalisation or a referendum.
- Unknown policy ideas map to the nearest supported action kinds, and the game says what was
  approximated.
- A new action kind (`institutional_reform`, `public_ownership`, `referendum`) is added to
  `PolicyAction` with engine sign rules and feasibility checks. These are needed for the
  revolution goals in Project 9.

### RB-7 Forecast preview (P3, joins Project 10)
As a player, I want to see the advisers' forecast before committing, at a cost.
- Optionally, in an easier difficulty, the top 2 candidate outcomes and their probabilities
  are shown before confirmation.
- Previewing costs something in-game (time, political capital), so it isn't free information.

---

## Epic C: Trust and transparency (Project 10)

### TT-1 Mode and usage visibility (P1, done in PR #7)
- Startup banner says OFFLINE PRACTICE or CLAUDE with the model.
- `--check-llm` confirms the connection, and each Claude turn prints calls, tokens and cost.

### TT-2 Why did that happen? (P2) - done (PR #46)
As a player, I want to see the causal chain behind an outcome.
- `why` prints the top 3 engine paths from the player's actions to the biggest indicator
  and approval moves, for example "energy tax → prices +3% → low-income approval −4".
- `alternatives` lists the candidate outcomes that were not chosen, with their scores.
- Done: `game/explain.py`. `why` re-runs each action and shock through the engine on its own
  to find what drove the biggest indicator moves, shows the graph path it took, and names
  the events and indicators behind the biggest approval moves. Both commands work at the
  response prompt.

---

## Order

1. P1 stories first, in this order: SD-3, RB-2, RB-1, SD-1, SD-2, RB-3, RB-4.
   - SD-3's offline library fixes the repetition Ed saw today, even without Claude.
   - RB-2 and RB-1 make responses expressive on top of the current interpreter.
2. Then run the 30-turn Claude review (Project 6's done-when) and re-prioritise the P2
   stories from what it shows. Done 2026-10-03: see
   [../reviews/2026-10-03-claude-30-turn.md](../reviews/2026-10-03-claude-30-turn.md).
3. From the review: P1 SD-9 and RB-8 (with EB-13 and EB-14 in engine-balance.md), all done
   in PRs #36 to #39 (merged 2026-10-04); the re-run on 2026-10-04 met every target. RB-9
   done (PR #42), SD-5 done (PR #43), SD-4 done (PR #44), SD-6 done (PR #45), TT-2 done (PR #46), RB-5 done
   (PR #47). That finishes this order.
