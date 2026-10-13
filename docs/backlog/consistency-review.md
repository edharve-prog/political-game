# Backlog: Turn Consistency (Project 19)

Source: Ed's code review of commit 7be7c65, 2026-10-12. It found 15 prioritised problems.
The main one is consistency: for the same decision, the forecast, the simulation that
actually runs and the generated narrative can describe different consequences. Ed asked
for a plan and several threads to work through the fixes. The enhancements come later, in
Ed's order.

Each fix ships with a test that fails on main and passes once fixed, the same rule as
Project 17.

## Phase 1: fixes

### Turn engine consistency (owner: "Turn engine consistency" thread)

- **CR-1 Caps cause false reversals** (finding 1). Later changes are worked out before
  clipping, so the bounded state and the queued changes disagree. Reproduced: a positive
  shock moved an indicator from 9 to 10, then to 5, then to 2.5. Fix: work out later
  changes from the bounded state. **Done, PR #85.**
- **CR-2 Forecasts miss consequences already on the way** (finding 2). Forecasting ignores
  queued effects and several background mechanisms that resolution uses. Example: a queued
  inflation rise of 2 points was forecast as zero. Fix: forecasts and actual turns use one
  shared transition engine. **Done, PR #85, propagation step only; the full turn is SR-5.**
- **CR-8 Repeated shocks get past the limits** (finding 8). Limits apply to each entry on
  its own, so duplicate shocks add up with no overall cap. Fix: deduplicate events and cap
  the total for each node. **Done, PR #76.**
- **CR-9 Policy effects are cut short** (from the report). **Done, PR #85.**
- **CR-10 Future consequences are frozen** (from the report). **Done, PR #85.**
- **CR-11 Graph changes can apply partly** (from the report). **Done, PR #80.**

### Save and resume safety (owner: "Save and resume safety" thread)

- **CR-3 Finished turns can go unsaved** (finding 3). The next scenario is generated before
  the finished turn is saved, so a provider failure or an interruption loses progress. Fix:
  save the resolved turn before asking for the next scenario. **Done, PR #78.**
- **CR-12 Stale cached outcomes** (from the report). **Done, PR #84.**
- **CR-13 The scenario changes on resume** (from the report; also review finding M6,
  placed in Project 8). **Done, PR #78.**
- **CR-14 Turn-length settings are unused** (from the report). **Done, PR #79 (fixed at one month).**
- **CR-15 Election edge cases** (from the report). **Done, PR #82.**

### Interpretation and fiscal rules (owner: "Interpretation and fiscal rules" thread)

- **CR-4 Negated instructions become the opposite action** (finding 4). Stored
  interpretations are reused on word overlap, so "Increase public spending" and "Do not
  increase public spending" meet the reuse threshold. Fix: reuse only when action,
  direction, amounts and duration agree, and handle negation explicitly. **Done, PR #77; more phrasings in SR-3 and SR-4.**
- **CR-5 Offline mode drops pledges and sackings** (finding 5). The wrapper does not pass
  on metadata found by the interpreter beneath it. Fix: every path returns the whole
  interpretation object. **Done, PR #77.**
- **CR-6 Funded policies can be blocked** (finding 6). When the starting deficit is over the
  limit, spending is rejected on its own before offsetting taxes are counted. Fix: judge
  fiscal feasibility on the whole package. **Done, PR #81.**
- **CR-7 Tax cuts can count as revenue rises** (finding 7). Taxes aimed at an indicator use
  the absolute magnitude, which mixes up the intended outcome with which way the tax moves.
  Fix: keep the tax change, its fiscal cost and its expected effect on the indicator apart. **Done, PR #83.**

## Phase 2: enhancements (Ed's order)

Held with Project 20's enhancements (SE-1 to SE-12 in
[semantics-review.md](semantics-review.md)), which sharpen most of these.

1. **CE-1 Active-policy register.** Budgets, progress, expiry dates, amendments and
   cancellation for every policy in force.
2. **CE-2 Decision previews that compare.** Set doing nothing against the alternatives,
   with immediate and delayed costs, benefits and uncertainty. Builds on TT-2
   (`game/explain.py`) and needs CR-2's shared engine.
3. **CE-3 Explanations tied to the real calculation.** Show which policies, inherited
   commitments and events caused each change.
4. **CE-4 Delivery limits.** Staffing, procurement and department capacity decide how fast
   a policy works.
5. **CE-5 Parliamentary bargaining.** Factions, seats, concessions and amendments replace
   the single support threshold (it extends EB-14's whips and forcing).
6. **CE-6 More believable elections.** Keep government approval apart from party
   preference, turnout and constituency results. It takes in Project 4's enhancements 10
   and 11.
7. **CE-7 Storylines with consequences.** Unresolved crises get explicit deadlines and
   mechanical results. It joins SD-1 and SD-7 (C4).
8. **CE-8 Advisers with different aims.** Economic, political and delivery advice show
   competing assumptions. It shares its adviser design with CA-8f (unlabelled lens
   advisers).

## Order

The three fix threads run in parallel. CR-2 (one transition engine) is the biggest; CR-9
and CR-10 build on it, and CE-2 and CE-3 need it. Phase 2 starts once Phase 1 is merged,
unless Ed reorders.
