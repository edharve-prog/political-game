# Project 17 — Engine correctness

**Status:** In review (2026-10-03). PRs #26 (EC-1 to EC-3), #28 (EC-4, EC-5), #29 (EC-6 to
EC-10) and #30 (EC-11). They are independent and merge cleanly in any order.
**Source:** Ed's 11-finding review, posted 2026-10-02 in the "Engine correctness" thread.
**Checked against:** main @ 5056df2 (after PR #23). No open PRs touched these files.

All 11 findings hold against the current code. Two overlap the Project 16 backlog and are
done here instead: finding 5 is **EB-12** (propagation limit), and finding 8 sits next to
**EB-13**'s consistency nit (L1 is a separate point and stays in EB-13).

## Stories and design

### EC-A Player action effects (PR #26: findings 1, 2, 4)
- **EC-1 Military and diplomacy stop moving target GDP the wrong way** (finding 1,
  `world/propagation.py` `_ACTION_SIGN`). Diplomatic and military actions no longer use the
  generic "push the target's primary metric" path. Instead:
  - Their standing effects stay in `world/changes.action_changes` (relationship, stability).
  - Diplomacy moves trade only with a trading partner (a `TRADES_WITH` edge either way).
    That is a smaller growth shock in the direction of the magnitude: a deal helps and
    sanctions hurt.
  - Military escalation is an explicit disruption shock that lowers the target's growth.
    Through trade links, that also costs the player. De-escalation (magnitude < 0) improves
    the relationship and moves nothing else.
- **EC-2 Player action changes are never truncated** (finding 2, `changes.py:160`).
  `apply_graph_changes(..., trusted=True)` skips the `MAX_CHANGES` cap for the
  deterministic `action_changes()` output. Every change is still checked for ids and fields.
  The cap and the size limit still apply to LLM outcome changes.
- **EC-3 Action–target compatibility** (finding 4). A table in `policy/feasibility.py`
  lists the action kinds each node kind accepts. For example, diplomacy and military need
  a foreign country, and appointments need an institution. Indicators accept only the
  interventions they declare in the new `Indicator.interventions` field. In the toy world:
  energy prices accept `regulate` (a price cap) and `spend` (a subsidy), house prices accept
  `regulate` and `tax` (lending rules, stamp duty), and the rest accept nothing. An
  incompatible action is blocked with a plain-English reason, so it shows up in the turn
  notes. The interpreter prompt lists the allowed direct interventions.

### EC-B One owner for limits (PR #28: findings 3, 7)
- **EC-4 Pure interpreters** (finding 3). `LLMInterpreter` and `StoredInterpreter` return
  what was interpreted. `Game._limit` → `policy.limits.constrain` is the only place that
  applies feasibility, diminishing returns and capital. `requested_actions` now means the
  same thing in every mode, and blocked LLM actions stay in the record and the knowledge
  store. A stored interpretation that is no longer feasible is reported as blocked. It no
  longer falls back to keyword matching. `ClarifyingInterpreter.dropped()` was unused, so it
  goes.
- **EC-5 Deficit ceiling applies to the package** (finding 7). After capital scaling,
  `constrain` projects the deficit as the current value, plus effects already landing this
  turn, plus every fiscal move in the package. If budget-loosening measures would take the
  projection past `DEFICIT_LIMIT` (10% of GDP), they are scaled down to fit, with a note.
  Tightening measures in the same package count towards the headroom.

### EC-C Engine, approval, scoring and selection (PR #29: findings 5, 6, 8, 9, 10)
- **EC-6 Convergence is checked** (finding 5 / EB-12). `simulate` raises
  `PropagationError` if the lag-0 fixed point hasn't settled after `MAX_ITERATIONS`.
  `lag0_gain(state)` bounds the loop gain. It takes the spectral radius of the damped lag-0
  matrix, with each weight at |weight| + 3σ uncertainty, and finds an upper bound by
  Collatz–Wielandt. LLM edge changes that would push it to `MAX_LAG0_GAIN` (0.9)
  or above are rejected. Tests cover a stable cycle, a rejected change and an unstable
  graph.
- **EC-7 Policy approval lasts exactly its duration** (finding 6). Policy events use
  `hold_turns = duration_turns - 1`. `hold_turns` is documented as the number of full-strength
  updates after the first one. A test checks the exact weight sequence.
- **EC-8 Omitted claims are not free** (finding 8). Each indicator the engine moves
  materially, but the candidate leaves out, counts as a claim of zero change. A candidate
  that makes no claims scores 1.0 only when the engine expects no material moves.
- **EC-9 Event tags** (finding 9). `"none"` must be the only tag if it is used
  (`check_candidates`). `base_rate` uses the rarest tag (the Fréchet upper bound on all of
  them happening), so adding tags can never make an outcome look more common.
- **EC-10 Scenario conditions are hard** (finding 10). The order of choice is: fresh and
  applicable, then any applicable except last turn's, then any applicable. A scenario whose
  conditions fail is never shown. If none applies, it raises a clear error. The built-in
  library has unconditional scenarios, so this can't happen in normal play.

### EC-D Validation at the boundary (PR #30: finding 11)
- **EC-11** `GameConfig`: `election_turn`, `horizon`, `k_draws` and `turn_length_months`
  must be at least 1, and `capital_per_turn` must be at least 0.
- Every model refuses NaN and infinity (`allow_inf_nan=False`).
- Nodes: `gdp_bn`, `output_bn` and `employment_k` must be at least 0.
- `Indicator`: `low <= high`, and the value must lie within the bounds.
- `WorldState`: `controlled_by` must name an institution, and groups must have a total
  population share above 0.

## Done when
Each story has a test that fails on main and passes after the change. The full suite, ruff
and the balance test all pass.

## Follow-ups noticed (not in scope)
- A tax aimed at an indicator always raises revenue (`fiscal_size` uses |magnitude|). So
  cutting stamp duty to push house prices up is booked as a tax rise. Fixing it needs the
  interpreter to say whether a tax goes up or down separately from the direction of the
  indicator.
