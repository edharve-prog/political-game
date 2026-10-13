# Backlog: Policy Semantics and Counterfactuals (Project 20)

Source: Ed's engine, reasoning and gameplay review of 2026-10-12, at snapshot 361a0ff (main
after PR #85). The full text, with reproductions and a script, is in
[review/2026-10-12-engine-review.md](../review/2026-10-12-engine-review.md).

The review found 11 defects and 6 design weaknesses. Its point: a passing suite does not yet
show that policy meaning, counterfactuals and explanations agree. Several items pick up where
Project 19 stopped. PR #85 shared the propagation step, but not the full turn. PR #77 handled
some negation, but not these phrasings.

Each defect ships with a test that fails on main and passes once fixed. Ed's exact
reproductions (section E of the review) become regressions.

## Defects

### Owner: "Subsidy and fiscal scale fixes" thread

- **SR-1 Cutting a wage subsidy still protects jobs** (finding 1, P1). A spend on
  unemployment gives job cover from the absolute magnitude whatever its instrument, while the
  fiscal side books a cut as a saving. A six-turn cut saves 1.25 points of deficit *and* gives
  80% cover. Fix: track real subsidy programmes; a rise adds cover, a cut removes existing
  cover and saves only what it removes, and a cut with nothing to cut is rejected.
- **SR-6 Sector spending is costed and felt on different scales** (finding 6, P2). Cost uses
  5% of current output; the physical effect uses 5% of baseline output. A second full spend on
  public services costs £55bn but adds £50bn. Fix: one agreed scale for both.
- **SR-10 Zero debt never grows** (finding 10, P3). Zero doubles as "debt not tracked". Fix: an
  explicit tracking flag, so tracked zero still accumulates deficits.
- **SR-11 Undoing a relationship change doesn't restore trade** (finding 11, P3). +0.1 then
  −0.1 leaves EU→UK trade at 0.448875, not 0.45. Fix: derive exposure from a stored base weight
  and the current relationship (unless hysteresis is meant, in which case model it openly).
- **SR-12 Legislating gives free output** (finding 12, high gameplay priority). `legislate`
  gets the same output push as `spend` but no fiscal cost. Fix: legislation authorises
  specific mechanisms with delays and capacity limits; add a legislation bot to the balance
  test.

### Owner: "Offline interpretation safety" thread

- **SR-3 Offline play acts on negated instructions** (finding 3, P1). "Do not increase public
  spending" becomes a spending rise, "Do not sack the Chancellor" sacks the Chancellor, and
  "Cut spending on furlough" becomes a rise. Fix: parse clauses with action, target, polarity,
  amount and duration; ask for clarification when unclear.
- **SR-4 Stored interpretations swap beneficiaries** (finding 4, P1). "Raise taxes on business
  and cut taxes on pensioners" matches its mirror image at similarity 1.0. Fix: reuse only on
  exact normalised text until clause-by-clause action–target–direction tuples can be compared.
- **SR-9 Outcome reuse ignores duration and instrument direction** (finding 9, P2). A one-turn
  and a 24-turn pension rise share a signature, as do energy spending "raise" and "cut". Fix:
  add duration, instrument direction and policy identity to the signature, and check each
  reused candidate on its own. Follows EB-13 (PR #72) and CR-12 (PR #84).

### Owner: "Forecast and explanation consistency" thread

- **SR-2 The no-action baseline already contains the policy** (finding 2, P1). Forecasts use
  the policy's job cover for both the policy path and the baseline, so a subsidy's reported
  contribution is 0.000 where separate runs give −0.487. The two paths also use different
  draws. Fix: separate cover for each path, paired draws, and clearly named comparisons.
- **SR-5 Forecast and resolution still differ** (finding 5, P2). PR #85 shared the propagation
  step, but resolution also settles institutions, updates loyalty, applies graph changes and
  moves approval. A whip forecast at +0.10 resolves at +0.08. Fix: one pure full-turn
  transition with a projection mode, and equivalence tests over several turns.
- **SR-7 Old proposals skip the current limits** (finding 7, P2). A proposal carries no turn
  or state version, and `commit()` doesn't recheck it. Fix: bind proposals to a turn and state
  version, add an idempotency key, and revalidate at commit.
- **SR-8 Explanations invent a market penalty** (finding 8, P2). "Why" simulates each action
  alone, so a funded tax cut is blamed for a confidence shock (−7.94) that the real package
  never had (sterling rose 3.22). Fix: an effect ledger recorded during resolution, and
  remove-one-action attribution with interactions reported openly.

## Design items (held until the defects land)

- **SR-13 Candidates aren't checked against their own effects** (finding 13). Every candidate
  is scored against the same pre-outcome forecast. Fix: a conditional forecast per candidate,
  and give the judge the full structured candidate.
- **SR-14 Generated crises carry no mechanical shock** (finding 14). Fix: generated scenarios
  include validated event specifications that the engine turns into bounded shocks, the same
  pipeline online and offline.
- **SR-15 Probabilities are uncalibrated relative scores** (finding 15). Fix: label them
  "relative likelihood" until calibrated, deduplicate overlapping candidates, and evaluate on
  held-out cases. Ties to Project 11.
- **SR-16 Hard-coded rules favour some strategies** (finding 16). The tax-cut-only market
  penalty, consultation that helps whoever is consulted, "phased" delivery with no schedule,
  and loyalty that ignores size. Fix: credibility from the whole financing plan, consultation
  matched to stakeholders, real phasing, and loyalty scaled by size. Overlaps CA-8's contested
  links.
- **SR-17 Elections and balance evidence are too narrow** (finding 17). Two-party vote from
  approval, fixed turnout, and a small fixed bot set. Fix: regional and multi-party voting
  later; adaptive and adversarial balance bots, more seeds and per-strategy results now.

## Enhancements (held until the defects land)

Ed's table, in his order. Several sharpen Project 19's CE items, noted in brackets.

| When | Enhancement | Done when |
|---|---|---|
| First | **SE-1** Policy registry with a lifecycle [CE-1] | A spending cut can remove an existing programme; its savings and lost benefits reconcile. |
| First | **SE-2** One authoritative transition pipeline [SR-5] | With randomness off, projected and actual states agree over several turns. |
| First | **SE-3** Structured policy confirmation | Negated or ambiguous text cannot silently become its opposite. |
| First | **SE-4** Actual-effect ledger [CE-3, SR-8] | Every reported change reconciles to the ledger; explanations never invent shocks. |
| Next | **SE-5** Counterfactual option comparison [CE-2] | Players see why an option is attractive and which assumptions could reverse the ranking. |
| Next | **SE-6** Implementation capacity and delays [CE-4] | Announcing a hospital or housebuilding programme does not create full output at once. |
| Next | **SE-7** Legislative bargaining [CE-5] | A weak government can negotiate passage rather than rely on one support threshold. |
| Next | **SE-8** Contextual advisers [CE-8, CA-8f] | Advice is feasible, differs for stated reasons, and has measurable forecast consequences. |
| Next | **SE-9** Consequential storylines [CE-7] | Ignoring a serious issue has a visible, traceable mechanical consequence. |
| Later | **SE-10** Regional and multi-party elections [CE-6] | Similar national approval can produce different plausible seat counts. |
| Later | **SE-11** Versioned replay and save migration | A save reproduces under its rules version or says clearly why it cannot. |
| Later | **SE-12** Reliability and cost controls | A failed provider call keeps the pending decision; recovery never silently changes the rules. |

## Order (Ed's section D)

1. Close the meaning and money loopholes: SR-1, SR-3, SR-4, SR-12.
2. Make comparisons trustworthy: SR-2, SR-6, SR-5.
3. Make the engine boundary authoritative: SR-7, SR-10, SR-9, then SE-11.
4. Make explanations auditable: SR-8, then SE-4.
5. Improve the political model: SE-6, SE-7, SR-16, SE-10.
6. Widen the balance evidence: SR-17, with legislation, long commitments, token consultation,
   tiny concessions, reversals and mixed instruments; a fast PR subset plus a larger scheduled
   batch.

The three defect threads run in parallel; the order above guides each thread's own sequence.
Property tests to add over time are listed in section D of the review.
