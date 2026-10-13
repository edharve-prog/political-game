# Political Game: engine, reasoning and gameplay review

Ed's review, 12 October 2026, of snapshot 361a0ff182cbe082806fcfc74015049b3f9bd81c (main after
PR #85, "shared-turn-engine"). Findings refer to that snapshot, not later changes. Copied as
Ed posted it; the plan's backlog is [backlog/semantics-review.md](../backlog/semantics-review.md).

## Assessment

The code has a useful foundation: typed state, deterministic random streams, a shared
propagation step, bounded outcome effects, persisted turn records, and substantial automated
coverage. However, a passing suite does not yet establish that the policy semantics,
counterfactuals and explanations agree. Those are the most valuable areas to fix before adding
more scenarios.

I found 11 concrete correctness or consistency defects, plus 6 modelling/design weaknesses. The
highest-impact defects concern subsidy cuts, negated instructions, stored interpretation reuse,
and a contaminated no-action baseline. Several others matter most to future integrations or
long-running/custom games; their reach is explicitly qualified below.

Validation performed locally on this snapshot:
- `uv run pytest -q`: 462 passed, 5 skipped, 97.92 seconds.
- `uv run ruff check .`: passed.
- `uv run ruff format --check src tests`: passed; 116 files already formatted.
- Additional small Python probes reproduced the numerical and parsing results reported below.
- No live model calls were made. No repository files were changed, and nothing was pushed.

Confidence: 0.96 overall for the reported code behaviour. Confidence is lower for gameplay
impact and economic realism, which require broader simulation and empirical calibration.
Per-finding confidence below describes confidence in the finding, not its estimated likelihood
during play.

Priority: P1 = fix before relying on outcomes; P2 = important consistency/robustness fix; P3 =
lower-impact or specialised case.

## A. Concrete defects

### 1. Cutting a wage subsidy saves money while still protecting jobs
P1 · Reproduced · Confidence 1.00

Source: `world/propagation.py`, `job_protection()` and fiscal functions.

Cause: Any spending action on unemployment activates job protection using the absolute
magnitude. It ignores whether instrument is raise or cut. Meanwhile, the fiscal calculation
correctly interprets `instrument="cut"` as a saving. Existing cover is combined using max, so
cuts cannot withdraw it either.

Reproduction: A six-turn unemployment spending action with magnitude −1 and instrument cut
passes policy constraints, lowers the deficit by 1.25 percentage points, and supplies 80% job
protection throughout all six turns.

Impact: A player can obtain fiscal savings and stronger employment protection simultaneously,
without an existing programme to cut. This undermines the core fiscal trade-off.

Fix: Track actual subsidy programmes and their funded coverage. A rise creates/increases cover;
a cut reduces existing cover and saves only the spending actually removed. Reject a cut when
there is no programme to reduce. Test both signs of magnitude separately from both instrument
directions.

### 2. The no-action forecast already includes the new subsidy
P1 · Reproduced · Confidence 1.00

Sources: `game/loop.py`, `forecast()`; `world/propagation.py`, `propagate()`.

Cause: `forecast()` supplies coverage from `job_protection(state, actions)`. `propagate()` uses
that same coverage for the policy path and its no-action baseline. Thus the counterfactual
includes a benefit of the very policy it is supposed to exclude. In addition, the policy path
uses the mean of uncertain draws while the baseline uses only central weights; that can
introduce unrelated differences in a nonlinear bounded model.

Reproduction: After a manufacturing slump, with edge uncertainty disabled, a subsidy's reported
three-turn unemployment contribution is 0.000 percentage points. Comparing a separate subsidy
run with a separate no-subsidy run gives −0.487 percentage points. The saving of jobs disappears
from the reported contribution because it is present in both sides of the internal comparison.

Impact: The offline forecaster can fail to credit a useful policy to its beneficiaries. Adviser
comparisons or future policy optimisation built on `response_at()` would inherit the error.

Fix: Use separate policy and baseline coverage. Run paired simulations with identical random
edge draws and identical starting state, changing only the intervention being assessed. Decide
explicitly whether the comparison is "policy versus no policy under the same crisis" or "policy
plus crisis versus neither"; offer distinct names for these measures.

### 3. Offline interpretation executes explicitly negated instructions
P1 · Reproduced · Confidence 1.00

Source: `game/stubs.py`, `KeywordInterpreter._one()` and `keyword_sackings()`.

Cause: The keyword parser recognises certain promises but lacks general negation handling. The
sacking parser checks for dismissal keywords without checking their polarity.

Reproductions:
- "Do not increase public spending" → positive public-sector spending, magnitude 0.5.
- "Do not sack the Chancellor" → the Chancellor is added to the sacked list.
- "Cut spending on furlough" → an unemployment spending action with no cut instrument, which
  defaults to an increase in spending.

Impact: Offline play can perform the opposite of the user's decision. The confirmation screen
helps players catch mistakes, but does not make the interpretation correct.

Fix: Parse clauses with explicit action, target, polarity, quantity and duration. Reject
ambiguous constructions and request clarification. For offline mode, a structured policy
builder is safer than silently assigning meaning to unsupported prose. Add tests for negation,
mixed clauses, conditional statements and explicit instrument cuts.

### 4. Stored interpretations confuse policies with reversed beneficiaries
P1 · Reproduced · Confidence 1.00

Source: `knowledge/offline.py`, `intent_key()` and `same_request()`.

Cause: Matching checks the sequence of direction words, numbers and negations, then compares
sets of words. It does not bind each direction to its target.

Reproduction: These opposite redistributive packages receive a similarity score of 1.0:
- "raise taxes on business and cut taxes on pensioners"
- "raise taxes on pensioners and cut taxes on business"

Impact: When the relevant stored interpretation exists, reuse can apply the earlier policy to
the wrong group. The negation safeguards do not prevent this role-swapping failure.

Fix: Initially restrict reuse to exact normalised text. Restore fuzzy reuse only after comparing
structured action–target–direction tuples clause by clause. Reordered policies should match only
when their semantic tuples agree.

### 5. Forecasts and resolution differ even without a random outcome
P2 · Reproduced · Confidence 1.00

Sources: `game/loop.py`, `forecast()` and `resolve()`; `world/propagation.py`, `_path()`.

Cause: The forecast shares the graph-propagation step but not the full turn transition.
Resolution also updates cast loyalty, applies action-driven graph changes, settles institutions
and advances approval. The forecast projects debt but omits these other mechanisms.

Reproduction: With uncertainty set to zero, a full-strength Commons-whipping action forecasts a
support increase of 0.10. Resolution produces 0.08 because institutional settling immediately
takes back part of it.

Impact: Forecasted institution values and longer-term consequences need not match actual
no-outcome play. Diplomatic changes to trade links also occur in resolution but are not carried
forward by the forecast's fixed graph. A shared low-level step is not full forecast/resolve
equivalence.

Fix: Extract a pure full-turn transition that can run in projection mode. Use it for both actual
resolution and forecast trajectories, with explicit treatment of future events. Add
deterministic one- and multi-turn equivalence tests for whipping, diplomacy, resignations, debt
and approval. If some fields remain unprojected, label that limitation in the forecast rather
than presenting them as equivalent.

### 6. Sector spending costs and physical effects use different scales
P2 · Reproduced · Confidence 1.00

Source: `world/propagation.py`, `fiscal_steps()` and `_Engine.scale()`.

Cause: Fiscal cost uses 5% of current sector output per standard step. The engine reconstructs
the sector's output without its carried deviation, and uses 5% of that baseline for the
physical effect.

Reproduction: Public output starts at £500bn. A full spending action lifts it to £550bn. A
second identical action is costed at £55bn, but contributes £50bn of new output after
separating the fading first impulse. This probe exercises the low-level resolver directly;
normal game repetition limits scale the action down but do not remove the mismatch in cost per
unit of effect.

Impact: The documented assumption that the fiscal cost equals added output stops being true as
soon as a sector already has a deviation. Booms make the same physical effect more expensive,
slumps make it cheaper, without an explicit productivity mechanism.

Fix: Store explicit policy amounts and convert both budget and output through one agreed scale.
Alternatively retain a stable baseline scale for both. Test repeated actions in expanded and
contracted sectors.

### 7. Old proposals can bypass the current turn's limits
P2 · Reproduced through the Python API · Confidence 1.00

Source: `game/loop.py`, `Proposal` and `Game.commit()`.

Cause: A proposal has no originating turn, scenario or state version. `commit()` trusts its
actions without verifying freshness or rerunning constraints.

Reproduction: Create a spending proposal, commit it, lower Commons support to 0.2, then commit
the same proposal again. Its full spending action still executes, although a newly checked
proposal would be blocked and the repeated action should also encounter diminishing returns.

Impact: Stale proposals, double submission, or future web/API clients can bypass intended
limits. The normal CLI creates fresh proposals, so this is primarily an engine-boundary defect,
not a claim that ordinary CLI play currently exposes a reuse button.

Fix: Bind proposals to a turn and state hash/version; reject stale submissions. Give commits an
idempotency key. Revalidate at the authoritative commit boundary, and require renewed
confirmation if the effective actions change.

### 8. The explanation engine invents a market penalty for a financed package
P2 · Reproduced · Confidence 1.00

Source: `game/explain.py`, `_causes()` and `_contribution()`.

Cause: Each action is converted into shocks in isolation. However, market confidence is
calculated at package level: offsetting taxes/spending cuts can prevent a tax-cut confidence
shock. Explaining the tax cut alone reintroduces that shock. Explanations also simulate from an
empty carry state without bounds, instead of explaining the actual transition with its history
and interactions.

Reproduction: A national tax cut paired with a public-sector tax rise leaves unfunded cuts below
the market-penalty threshold. Sterling rises about 3.22 index points in the probe, yet the
explanation calls the national tax cut the main cause with −7.94 points, a confidence shock that
the actual package did not generate.

Impact: The player learns the wrong causal rule from the "why" output, even when actual
resolution is internally correct.

Fix: Record an effect ledger during actual resolution, including package-level effects, carried
effects, bounds and background mechanisms. For counterfactual attribution, remove one action
from the complete package and recompute; report interaction/residual contributions explicitly
rather than pretending all effects are additive.

### 9. Outcome reuse ignores policy duration and instrument direction
P2 · Signature collision reproduced · Confidence 0.99

Sources: `knowledge/entries.py`, `action_signature()`; `knowledge/offline.py`,
`StoredForecaster.forecast()`.

Cause: Signatures include kind, target, magnitude sign and size bucket, but omit duration and
the instrument's own direction.

Reproductions: A one-turn and 24-turn pension increase have identical signatures. Energy-price
spending with `instrument="raise"` and with `instrument="cut"` also has the same signature when
target and magnitude agree.

Impact: Stored narratives, approval reactions and graph changes can be reused for materially
different policies. Rescoring numerical claims reduces the risk but does not eliminate it: the
short scoring horizon cannot distinguish many long-term commitments, and at least one fitting
candidate permits reuse of the whole set.

Fix: Include duration, instrument direction, policy identity and relevant timing in the semantic
signature. Validate retained candidates individually. Keep narrative context such as delivery
and sackings separate, and regenerate or discard stale contextual claims.

### 10. A country with zero debt cannot accumulate new debt
P3 · Reproduced · Confidence 1.00

Source: `world/debt.py`, `step_debt()` and `DebtOutlook`.

Cause: Zero debt doubles as the flag meaning "debt is not tracked". `step_debt()` returns
immediately when debt is zero. If tracked debt is paid down to zero, future deficits no longer
rebuild it.

Reproduction: Set debt to 0 and the annual deficit to 6% of GDP: after one monthly debt step,
debt remains 0, rather than accruing approximately 0.5 percentage points before any denominator
effects.

Impact: Incorrect custom starting worlds and a broken boundary condition for very long games.
The standard 24-turn UK start at 100% debt makes this less immediately important.

Fix: Add an explicit `debt_tracking_enabled` flag or a nullable debt field. Tracked zero is a
valid numeric state and must still accumulate deficits. Test positive debt → zero → positive
debt.

### 11. Reversing a relationship change does not restore trade exposure
P3 · Reproduced; design intent should be confirmed · Confidence 0.99

Source: `world/changes.py`, `_scale_trade()`.

Cause: Each relationship change multiplies the current trade weight by a factor based on the
increment. Equal positive and negative increments have a product below one.

Reproduction: EU→UK trade weight starts at 0.45. Relationship +0.1 followed by −0.1 returns the
relationship to its original value but leaves the trade weight at 0.448875. Repeated cycles
progressively reduce exposure.

Impact: Identical relationship levels can imply different trade weights due solely to
reversible oscillation. That contradicts a simple "relationship scales trade" interpretation
unless hysteresis is deliberately intended.

Fix: Calculate exposure from a stored base trade weight and current relationship, or update by
a ratio of the old/new relationship factors. If damage and recovery are intended to be
asymmetric, represent that as a separate trade-capacity state with documented recovery rules.

## B. Reasoning and modelling weaknesses

These are not all implementation bugs: some are intentional simplifications. They matter
because they shape which strategies the game rewards.

### 12. The action label can grant free output
High gameplay priority · Reproduced mechanism · Confidence 0.99

`_ACTION_SIGN` and `actions_to_shocks()` give legislate the same direct output push as spend,
but only tax/spend carry fiscal costs. At magnitude 1 on public services, both raise output from
£500bn to £550bn; spending raises the deficit from 4.5% to 6.352%, while legislation leaves it
at 4.5% on the first turn. Both require legislative authority and political capital. The
numerical mapping is reproduced; I have not established a full-game winning exploit.

Solution: Separate the policy instrument from its desired result. Legislation should authorise
specific mechanisms, such as changing planning permissions or funded staffing, with
implementation delays and capacity constraints. Do not let the verb itself create generic
productive output. Add a legislation bot to balance testing; the existing fixed-lever bot
catalogue does not include one.

### 13. Candidate plausibility is not checked against the candidate's actual effects
High reasoning priority · Code inspection · Confidence 0.97

`LLMForecaster.forecast()` scores every candidate against the same pre-outcome engine forecast.
Candidate-specific shocks, event tags and graph changes are applied only after selection. Its
judge text lists narrative, claimed shifts and tags, but does not explicitly enumerate all
structured shocks, approval effects and graph changes.

Consequence: Numerical prose can fit the baseline forecast while attached effects lead
elsewhere. Conversely, a plausible shock-driven scenario can be penalised for deviating from a
forecast that omitted its shock. Current bounds limit effect sizes but do not prove
narrative/effect consistency.

Solution: Build a conditional forecast for each candidate using its bounded effects. Separate
the likelihood of the event from the consistency of its consequences. Give the judge the
complete structured candidate. After resolution, generate the final narrative from the actual
effect ledger, clearly distinguishing forecasts from realised values.

### 14. LLM-generated crises need not impose a corresponding economic shock
High gameplay priority · Code inspection · Confidence 0.99

`ScenarioDraft` and `generate_scenario()` have no generated shock field. The resulting scenario
inherits an empty shock list. The handwritten offline library can supply explicit shocks.

Consequence: A generated briefing can describe a new supply disruption without a guaranteed
disruption in the engine. Later outcome effects may supply one, so it is not correct to say
that generated crises always have no effect; the problem is that the briefing itself has no
enforced mechanical contract. This also makes online/offline difficulty structurally different.

Solution: Distinguish descriptions of existing state from new events. Generate validated event
specifications with known severity, affected mechanisms and timing, then let the engine map
those to bounded exogenous shocks. Use the same event pipeline in both modes.

### 15. Displayed outcome probabilities are uncalibrated relative scores
Medium reasoning priority · Code inspection · Confidence 0.98

`forecasting/scoring.py` explicitly calls the base rates placeholders. A conjunction's
least-likely event is used as its base-rate score: this is an upper bound, not an estimate of
the joint probability. A same-family LLM judge is a separate call, not independent empirical
evidence. Normalising four candidates also makes their probabilities depend on which other
candidates were proposed; unique titles do not ensure distinct mechanisms.

Solution: Label outputs "relative scenario likelihood" until calibrated. Deduplicate overlapping
candidates, ensure meaningful outcome coverage, and evaluate calibration on held-out cases. Use
conditional event probabilities or an explicit event model. Do not describe the 10th–90th
percentile edge-uncertainty band as covering every source of political uncertainty.

### 16. Several hard-coded rules favour particular strategies without sufficient context
Medium modelling priority · Code inspection · Confidence 0.98

Examples include `unfunded_tax_cut()`, which applies a special market penalty to tax cuts but
not comparable spending, and `delivery_factor()`, where any consultation list reduces several
kinds of resistance regardless of whom the policy affects. "Phased" delivery reduces certain
risks without changing the timing of the physical policy effect. Cast loyalty rewards the sign
of an action rather than its size, so a tiny token concession can receive the same backing
credit as a large one.

Solution: Base fiscal credibility on the whole medium-term financing plan, existing debt,
inflation, capacity and credibility. Match consultation to affected stakeholders, with
time/capital costs and diminishing benefits. Implement actual phased schedules. Scale loyalty
changes by materiality and relevance. Make uncertain behavioural assumptions configurable and
inspectable instead of treating them as settled economic facts.

### 17. Election and balance results are too narrow to establish strategic realism
Medium modelling priority · Code inspection · Confidence 0.99

`vote_intention()` and `seat_share()` convert approval directly into two-party vote intention
and then seats. This cannot represent tactical voting, regional parties or most forms of hung
parliament. Fixed turnout and four stylised groups also compress voter behaviour heavily.

The balance harness tests a small fixed set of strategies and seeds. Requiring inaction never to
win is a game-design target, not evidence of realism; averaging all lever bots can conceal a
dominated strategy or an exploit. Historical directional checks are useful but do not
constitute out-of-sample causal validation.

Solution: Add regional vote shares, competing parties, turnout reactions and constituency
uncertainty if realistic UK politics is the goal. Expand testing to adaptive and adversarial
strategies, multiple starts, longer commitments and held-out seeds. Publish both individual
strategy results and confidence intervals. Keep realism checks separate from difficulty targets.

## C. Recommended enhancements

| Priority | Enhancement | What it changes | Acceptance criterion |
|---|---|---|---|
| First | Policy registry with a lifecycle | Give programmes stable IDs, funding, start/end dates, coverage, delivery stages and cancellation rules. Avoid anonymous overlapping pushes that cannot be amended. | A spending cut can remove an existing programme; its savings and lost benefits reconcile. |
| First | One authoritative transition pipeline | Use the same complete state transition for resolution, counterfactuals and projections, with explicit uncertainty inputs. | With randomness disabled, projected and actual states agree over several turns. |
| First | Structured policy confirmation | Show instrument, beneficiary, magnitude, monetary amount, duration, funding and expected consequences. Highlight uncertain interpretations. | Negated or ambiguous text cannot silently become its opposite. |
| First | Actual-effect ledger | Record new actions, prior commitments, external events, market effects, clipping and interactions. Drive both the dashboard and narrative from it. | Every reported change reconciles to the ledger; explanations never invent shocks. |
| Next | Counterfactual option comparison | Compare options under the same crisis and random draws, including doing nothing, by fiscal cost, delivery time, group effects and downside risk. | Players can see why an option is attractive and which assumptions could reverse the ranking. |
| Next | Implementation capacity and delays | Model procurement, staff availability, planning, administrative capacity and a ramp-up schedule. | Announcing a hospital or housebuilding programme does not create full output immediately. |
| Next | Legislative bargaining | Separate MPs, confidence, bill support and government popularity; model coalition deals, amendments and vote-specific support. | A weak government can negotiate passage rather than relying only on a support threshold. |
| Next | Contextual advisers | Give advisers distinct objectives and factual constraints, and make them cite engine paths and uncertainties. Compare recommendations against executable policies. | Advice is feasible, differs for stated reasons, and has measurable forecast consequences. |
| Next | Consequential storylines | Attach unresolved issues to explicit hazards or resource pressures. Resolve them using relevant actions and outcomes, not target overlap alone. | Ignoring a serious issue can have a visible, traceable mechanical consequence. |
| Later | Regional and multiparty elections | Add regional voting blocs, turnout, opposition strategies and constituency-level uncertainty. | Similar national approval can produce different plausible seat distributions. |
| Later | Versioned replay and save migration | Persist engine/rules, schema and calibration versions, plus state hashes. Support explicit migrations or historical replay. | A save either reproduces under its rules version or clearly explains why it cannot. |
| Later | Operational reliability and cost controls | Add per-turn model-call budgets, bounded recovery from invalid output, checkpointed decisions and an explicit user choice when changing modes. | A failed provider call preserves the pending decision; recovery does not silently change simulation semantics. |

## D. Targeted testing and implementation order

The existing CI already includes lint, formatting, Python 3.11–3.14 tests, and a built-wheel
install/save/resume smoke check. Preserve these; the missing value is mainly semantic coverage.

1. Close the meaning and money loopholes. Fix subsidy direction, offline negation and role-safe
   interpretation reuse. Add exact regressions for the phrases and subsidy cut above. Replace
   generic "legislate = output" with specific mechanisms.
2. Make comparisons trustworthy. Split action/baseline coverage, pair random draws, reconcile
   sector scales, and expand the shared transition beyond graph propagation. Test projection
   against actual no-outcome turns.
3. Make the engine boundary authoritative. Add proposal freshness/idempotency, explicit
   debt-tracking state, semantic reuse signatures and versioned saves.
4. Make explanations auditable. Create the effect ledger and test package interactions,
   previous shocks, bounds and background debt effects. Do not derive causal explanations by
   independently simulating each action.
5. Improve the political model. Add implementation schedules, legislative bargaining,
   contextual fiscal credibility and regionally differentiated elections, according to the
   intended realism level.
6. Expand balance evidence. Add legislation, very long commitments, token consultation, tiny
   concessions, policy reversals, and mixed-instrument strategies. Use a fast deterministic PR
   subset and a larger scheduled batch with retained results. Add structural candidate checks
   and offline recorded-model evaluations so model access is not needed for every build.

Useful properties to test across many generated cases:
- A subsidy cut never creates new coverage, and savings cannot exceed the programme removed.
- A counterfactual baseline contains none of the new intervention's benefits or costs.
- Reordering equivalent package clauses preserves meaning; swapping their beneficiaries does not.
- A committed action is valid against the state it actually executes in.
- Equivalent policy amounts produce consistent fiscal and physical effects.
- Positive deficits can rebuild tracked debt from zero.
- Explanations reconcile to actual state changes, including package-level interactions.
- Saved and uninterrupted games agree under the same rules version.

## E. Compact reproduction script

Run from the checked-out repository with `uv run python`. It prints the reviewed behaviour; it
demonstrates the defects at the snapshot rather than being a regression suite. No model
credentials or live calls are required.

```python
from hog_sim.core.config import GameConfig
from hog_sim.core.models import Outcome, PolicyAction, Scenario, Shock
from hog_sim.game.loop import forecast, resolve
from hog_sim.game.stubs import KeywordInterpreter
from hog_sim.knowledge.offline import same_request
from hog_sim.world.propagation import fiscal_cost, job_protection
from hog_sim.world.seed.toy import toy_world

s = toy_world()
for edge in s.edges:
    edge.uncertainty = 0
cfg = GameConfig(k_draws=1, horizon=6)
quiet = Scenario(title="Quiet", briefing="", affected_nodes=[], urgency=0)
outcome = Outcome(narrative="Quiet", probability=1)

cut = PolicyAction(kind="spend", target="indicator:unemployment",
                   magnitude=-1, instrument="cut", duration_turns=6)
print("Cut fiscal cost:", fiscal_cost(cut, s))        # -1.25
print("Cut job cover:", job_protection(s, [cut]))    # 0.8 for six turns

slump = quiet.model_copy(update={"shocks": [
    Shock(node="sector:manufacturing", delta=-2)
]})
s1 = resolve(s, s, slump, [], outcome, cfg)
subsidy = cut.model_copy(update={"instrument": "raise"})
with_policy = forecast(s, s1, quiet, [subsidy], cfg)
without_policy = forecast(s, s1, quiet, [], cfg)
node = "indicator:unemployment"
print("Reported contribution:", with_policy.response_at(2)[node])  # 0.0
print("Separate-run contribution:",
      with_policy.at(2)[node] - without_policy.at(2)[node])  # about -0.487

k = KeywordInterpreter()
print(k.interpret("Do not increase public spending", s, quiet))
k.interpret("Do not sack the Chancellor", s, quiet)
print("Sackings:", k.last_sacked())  # ['person:chancellor']
print("Opposite packages similarity:", same_request(
    "raise taxes on business and cut taxes on pensioners",
    "raise taxes on pensioners and cut taxes on business",
))  # 1.0
```
