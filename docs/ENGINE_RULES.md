# Engine rules, in plain English

This guide explains how the game engine decides what happens. It describes the code on `main`
as it stands, not the plans. Every rule names the file it lives in (paths are under
`src/hog_sim/`), so when a rule changes, the matching section here should change with it.

The one-line summary: **the LLM proposes, the engine decides.** Claude writes the scenarios,
reads the player's response and suggests possible outcomes, but every number that changes in
the world is worked out by fixed Python rules, and anything Claude suggests is checked and
capped before it is used.

---

## 1. The world

*Source: `core/models.py`, `core/state.py`, `world/seed/toy.py`*

The world is a graph of **nodes** joined by **edges**. All of it lives in one `WorldState`,
which is the single source of truth; the NetworkX graph (`world/graph.py`) is a read-only view
rebuilt from it.

### Nodes

Each node has an id like `sector:energy` and one **main number** that the engine moves:

| Kind | Example | Main number | Other fields |
|---|---|---|---|
| Country | `country:uk` | growth (% a year) | GDP, relationship with the player (-1 to 1), stability (0 to 1) |
| Sector | `sector:housing` | output (£bn) | employment, sentiment (-1 to 1) |
| Population group | `group:pensioners` | approval of the government (0 to 1) | population share, turnout, lean |
| Institution | `institution:legislature` | support for the government (0 to 1) | power, independence |
| Indicator | `indicator:inflation` | its value (%, % of GDP or an index) | hard floor and ceiling, persistence, who controls it, which actions may target it |

A group's **lean** is the approval it drifts back to when nothing is happening.

### Edges

An edge says "a change here moves that over there". Each has a **weight** (how much), a **lag**
(how many turns before it lands) and an **uncertainty** (how much the weight can vary in the
forecast's random draws).

| Edge | Meaning | Used by |
|---|---|---|
| `DRIVES` | a node pushes an indicator (energy output lowers energy prices) | propagation |
| `SUPPLIES` | one sector or country feeds another's output | propagation |
| `TRADES_WITH` | one economy's growth moves another's | propagation |
| `INFLUENCES` | an institution moves another node, or a group's mood | propagation, approval |
| `EMPLOYS` | a sector's output affects a group's approval | approval |
| `CARES_ABOUT` | a group's approval depends on an indicator | approval |
| `ALLIED_WITH`, `RIVAL_OF` | context for Claude's prompts only | nothing in the engine yet |

A negative weight means the effect runs the other way: pensioners `CARES_ABOUT` inflation with
weight -0.4, so higher inflation lowers their approval.

### The starting world

The only world today is the hand-built UK in `world/seed/toy.py`: the UK, the EU and China; five
sectors (energy, finance, manufacturing, housing, public); seven groups that together make up
the whole electorate (pensioners 19%, young renters 15%, public sector workers 17%, business
owners 8%, private sector workers 22%, mortgaged homeowners 13%, students 6%); the House of
Commons and the Bank of England;
and seven indicators (inflation, unemployment, Bank Rate, energy prices, house prices, deficit,
and the pound's exchange rate).
Its numbers are illustrative round figures, not sourced data.

---

## 2. A turn, start to finish

*Source: `game/loop.py`*

A game is 24 turns of one month each, with a general election after the last one
(`core/config.py`). Each turn has two halves.

**Decide** (may call Claude, and everything it produces is logged):

1. A **scenario** arrives: a lead issue plus one to three smaller in-tray items (section 9).
2. The player types a response. The **interpreter** turns it into a list of policy actions,
   plus how it was delivered (who was consulted, phased or immediate). If the response is too
   vague, it asks a clarifying question instead and nothing happens. Offline, the keyword
   interpreter (`game/stubs.py`) also asks instead of guessing when a response says what *not*
   to do ("do not increase public spending") or sets a condition ("if inflation rises, cut
   spending"), since matching keywords would read the plain instruction inside it. Promises
   ("we will not raise taxes", "no cuts to schools") are still pledges, a refused sacking
   ("do not sack the Chancellor") sacks nobody, and cutting a programme ("cut spending on
   furlough") is a spending cut. A sentence with "but" or "while" is read as two instructions.
3. The **limits** cut the actions down to what is actually possible this turn (section 4). The
   player sees the result, with a note for every cut, and can edit and re-check it before
   confirming.
4. The engine runs its **forecast**: 100 random draws of the next 24 turns, starting from
   everything already in flight and adding this turn's actions, the scenario's shocks and the
   background pushes resolve will apply (section 5).
5. The **forecaster** proposes several possible outcomes, each is scored, and one is picked
   (section 8).

**Resolve** (pure Python, no Claude, so a saved game replays exactly from its log):

1. All the turn's shocks are gathered: the actions, the scenario's own shocks, any extra shocks
   in the chosen outcome, the shocks implied by the outcome's event tags (section 7), each
   sector's business mood, any crisis in an unstable foreign country (section 7), and any rise
   in the interest bill on the debt (section 3).
2. The engine moves on one turn, with the edge weights at their central values. It carries its
   own state from turn to turn (every node's fading push, its recent deviations and the shocks
   still running), so a policy's lagged effects keep arriving, a policy keeps working for as
   long as it runs however long that is, and later events (a graph change, job protection)
   act on effects already in flight.
3. Every node stays within its bounds as it moves (section 5).
4. Approval events from the actions and the outcome are added (section 6).
5. Business moods settle a fifth of the way back to neutral. Foreign-policy actions change the
   target country's relationship and stability, and any lasting graph changes in the outcome are
   checked and applied (section 7).
6. The debt grows by the month's deficit and its average rate reprices toward Bank Rate
   (section 3). Every group's approval moves one step (section 6).
7. Storylines advance (section 9), and the turn counter goes up.

Before anything changes, each indicator's value and each group's approval are added to their
history, so the next scenario can see what is moving.

### Why did that happen?

*Source: `game/explain.py`*

Typing `why` after a turn explains its three biggest indicator moves. The turn is resolved again
exactly as it happened, then without one cause at a time (each action, the scenario's shocks,
the chosen outcome's effects) and once without any of them. A cause's share is what the move
would have lost without it, and the run without any of them is the share of earlier turns and
background pressures. Package-level effects therefore land on the measures that made or
prevented them: a tax rise that pays for a tax cut is credited with keeping the markets calm,
not the cut blamed for a penalty that never came. Whatever is left after the shares (causes
that only act together, limits they share) is reported as "how this turn's causes combined"
rather than spread among them.

For the three biggest approval moves it shows where each group's approval is heading and why,
using the same terms the approval rules add up (section on approval below): the group's lean,
then the largest pushes away from it, in points. A push is an indicator the group cares about,
the sector it works in, an institution's standing, the penalty for a deficit above 5% of GDP,
or an approval event (or all events together, when they hit the cap).

---

## 3. Actions: what the player can do

*Source: `core/models.py` (`PolicyAction`), `world/propagation.py`, `policy/feasibility.py`*

Every response becomes one or more **actions**. An action has:

- a **kind**: tax, spend, regulate, deregulate, diplomatic, military, communicate, legislate,
  appoint, or do nothing;
- a **target**: one node;
- a **magnitude** from -1 to 1 (1 is as hard as possible; negative means the opposite, such as a
  tax cut);
- a **duration** in turns (default 1).

### Which actions fit which targets

| Target | Allowed kinds |
|---|---|
| The player's own country | communicate only (a speech to the nation) |
| A foreign country | diplomatic, military, communicate |
| A sector | tax, spend, regulate, deregulate, legislate, communicate |
| A population group | tax, spend, regulate, deregulate, legislate, communicate |
| An institution | spend, regulate, deregulate, legislate, communicate, appoint |
| An indicator | only the interventions it lists (energy prices: regulate or spend, meaning a price cap or a bill subsidy; house prices: regulate or tax, meaning lending rules or stamp duty; unemployment: spend, meaning a wage subsidy such as furlough) |

"Do nothing" fits anything. Any other indicator can only be moved indirectly, by acting on
something that drives it, so the world graph carries the trade-offs.

The player's own country's growth comes from what its sectors do, so tax, spending and rules
for "the country" have to be aimed at the sectors or groups they reach, where they carry their
costs. No action pushes the player's own growth rate directly, so a speech to the nation moves
no output. (Until 13 October 2026 national tax, spending and deregulation pushed UK growth 2
steps a unit; the extra growth cut the deficit and swelled GDP, so spending on the country paid
for itself and left debt lower than doing nothing.)

### How hard each kind pushes

One unit of magnitude is a push of **2 standard steps** on the target's main number, multiplied
by the kind's factor:

| Kind | Factor | So magnitude 1 is |
|---|---|---|
| spend | +1 | +2 steps |
| appoint, deregulate | +0.5 | +1 step |
| communicate | +0.2 | +0.4 steps (on a sector or country, once: see below) |
| regulate | -0.5 | -1 step |
| tax | -1 | -2 steps |
| diplomatic, military, do nothing | 0 | no direct push (see foreign policy below) |
| legislate | 0 (+0.5 on a group) | no direct push: a law creates no output by itself, and a law that funds, taxes or regulates is that action, with its cost; aimed at a group it wins approval of +1 step |

A **standard step** is a fixed size per kind of node (`scale()` in `world/propagation.py`):

| Node | One step is |
|---|---|
| Sector | 5% of the output it would have without the boom or slump it carries |
| Country | 1 point of growth |
| Group | 0.05 approval (5 points) |
| Institution | 0.1 support |
| Indicator in % or % of GDP | 1 point |
| Indicator as an index | 10 points |

So spending at full magnitude on the public sector (output £500bn) pushes its output up £50bn.
A second full spend a turn later still adds £50bn and still costs £50bn, though output is
then £550bn: a sector's step, for its output and for the budget alike, is measured against where
it would be without the boom or slump it already carries.

Special cases:

- **Aimed at an indicator**, the sign of the magnitude is the direction the player wants it to
  go, whatever the kind. "Cap energy bills" is regulate on energy prices with a negative
  magnitude, and it pushes prices down.
- **Regulating an indicator** also costs the sectors that drive it a quarter of the push in
  output (a price cap squeezes energy firms).
- **Aimed at a population group** (a pension rise, a tax on landlords), the action is not a
  push on the graph. It becomes an approval boost or hit for that group of 0.05 per step: a
  full-magnitude pension rise is +0.10 to pensioners' target approval, held for as long as it
  runs (section 6). Money handed to a group (spending on it, or a tax cut for it) also costs
  every other group a quarter of what that group wins, for as long as it runs: the others see
  someone else's pension rise or tax break. A tax rise on a group draws no such reaction.
- **Tax or spending on a group also moves demand.** The money put into (or taken out of) the
  group's pockets is spent on the sectors it buys from, held for as long as the money is paid
  (at least 6 turns). Pensioners and public sector workers spend 80% of it, young renters 90%,
  across the private sectors in proportion to their output (young renters weighted to housing);
  business owners spend 40% of it, as investment in manufacturing (60%) and finance (40%). A
  full-size pension rise (£34bn a year) therefore adds about £27bn of private-sector output, and
  taxing business owners takes investment away. Part of every transfer is saved, so it lifts
  output by less than spending the same money on a sector does.
- **A speech to a sector or a foreign country** lifts it once, by 0.4 steps per unit, and the lift fades
  like any one-off: it does not hold output up for as long as the speech "runs".

### Every lever has a loser or a risk

*Source: `world/levers.py`*

Levers that cost no money have a downside of their own, so a majority cannot be bought with
laws, speeches and red tape:

- **A rule aimed at a group** (legislate, regulate or deregulate: renters' rights, a law for
  pensioners) has a losing side, who lose as much as the group gains: a law for pensioners costs
  young renters, one for renters or public workers costs business owners, one for business
  costs public workers and young renters half each. Unless the same package puts money behind
  the group (spending on it, or a tax cut for it), the law's appeal is felt in full for 3 turns
  at most, then fades.
- **Deregulating a sector** costs workers their protections: young renters and public workers
  lose 0.3 steps of approval per unit each (public workers 0.6 when the public sector itself is
  deregulated), for as long as it runs. Energy deregulation also pushes bills up 0.2 steps per
  unit (riskier supply), finance deregulation pushes house prices up 0.5 steps per unit and
  doubles the odds of a market sell-off or capital flight at full size, and housing
  deregulation pushes house prices up 0.4 steps per unit.
- **A speech** runs for 3 turns at most, and to a sector or a foreign country it is a one-off lift.

### Policies that keep running

An action holds its push at full strength for its whole duration, then lets it fade (section
5). A 12-turn programme keeps working for 12 turns, and costs money for 12 turns.

### Money: the deficit

Spending and tax also move `indicator:deficit`, as a flow, for as long as the policy runs, and
for at least 6 turns even for a one-off. Aimed at a sector, the money spent (or taxed) is the
output the action adds (or removes), as a share of GDP: a full-size rise in public spending
adds £50bn of output, so the deficit stays **1.85 points of GDP higher**, while the same rise
for the smaller housing sector costs 0.56 points. Aimed at anything else (a group, an
indicator), each unit of spending or tax cut keeps the deficit **1.25
points higher**. A tax rise of the same size takes the same off. Spending or
taxing aimed at an indicator says separately whether the tax or spending itself goes up or down
(`instrument`), since the size's sign there is the way the player wants the indicator to move:
a subsidy to lower bills costs money, a stamp duty rise to cool house prices raises it, and a
stamp duty cut to lift them costs it (and counts as a tax cut for markets and pledges). Unset, it
reads as a rise. Actions aimed at the deficit itself have no separate cost.

**Markets and unfunded tax cuts.** A tax cut that the same turn's other decisions (tax rises,
spending cuts) do not pay for spooks the markets if it adds at least 0.75 points a turn to the
deficit (a tax cut of magnitude 0.6 or more on a group). Cuts and the savings that pay
for them are compared over their lives (cost a turn times the turns it is paid, at least 6), so a
one-turn tax rise pays for only a quarter of a cut that runs for two years. For each point of
unfunded cut, Bank Rate jumps 0.4
points and the pound falls 8 index points, both fading as usual, and every group takes a 0.12
approval hit that fades with the usual half-life. This is the 2022 mini-budget: £45bn of tax cuts
with no plan to pay for them sent gilt yields and mortgage rates up, the pound to a record low
and the government's poll rating down by over 10 points. Spending is not judged this way:
markets lent freely for the 2008 bailout and the 2020 lockdown.

**Job protection.** Spending aimed at unemployment is a wage subsidy, like the 2020 furlough
scheme: the state pays wages so firms keep their staff. It does not push unemployment down.
Instead, while it runs, it holds back a share of any rise in unemployment: 80% for a full-size
scheme (magnitude 1), 40% at half size. It costs the deficit like any other spending.
It covers any rise in unemployment while it runs, including job losses from a slump already under
way when it starts, and when it ends the jobs it was holding up are lost
after all if the economy has not recovered. In good times it only costs money. In 2020 output
fell by a fifth but unemployment rose only from 4% to about 5%.

Schemes running at the same time add up, to at most all of a rise, and each pays only for the
cover it adds, turn by turn. Topping up a running full-size scheme with a half-size one takes cover
from 80% to 100% and costs a quarter of a full scheme a turn while both run; once the first scheme
ends, the top-up's own 40% costs its full half.

Cutting a wage subsidy (spending on unemployment with the instrument "cut") winds down a scheme
that is running: it takes off that much cover for as long as the cut lasts, and saves only what
that cover cost, turn by turn, for the turns it would have run (so a cut across a scheme that
steps down later saves less in its later turns). With no scheme running there is nothing to cut,
so the cut is blocked; a cut bigger or longer than the scheme is trimmed to it.

**Debt and its interest bill.** The UK starts with public debt of **100% of GDP** at an average
rate of 4%, an interest bill of 4% of GDP a year that is already part of the starting deficit.
Each turn (a month) the debt grows by a twelfth of the deficit and shrinks by a twelfth of
nominal growth (growth plus inflation) times the debt; at the start the two cancel. The average
rate drifts toward Bank Rate as old debt is refinanced, half the gap in **24 turns**. Whatever
the bill has risen since the start is added to the deficit, so borrowing compounds and higher
rates cost money: Bank Rate about 2 points higher for two years adds nearly 1 point of GDP to the
deficit. A lower bill saves money the same way. The briefing shows the debt, rate and bill. A
world can leave debt untracked (a country with no debt figure); once tracked, debt paid down to
0 is still tracked and the next deficit builds it up again.

### Foreign policy

*Source: `world/propagation.py` (`foreign_shocks`), `world/changes.py` (`action_changes`)*

- **Diplomatic** action towards a country improves (or, with negative magnitude, worsens) the
  relationship by 0.1 per unit. If the two countries trade, it also moves the partner's growth by
  0.5 steps per unit, held for the action's duration (a trade deal or sanctions).
- **Military** escalation (positive magnitude) worsens the relationship by 0.15 per unit, cuts
  the country's stability by 0.05 per unit and knocks 1 step per unit off its growth, which
  reaches the UK through trade. De-escalation (negative magnitude) only mends the relationship.
- A relationship change also scales the trade between the two countries (section 7), so
  diplomacy deepens trade for good and a row thins it.
- **Communicate** aimed at a foreign country follows the general rule above and nudges that
  country's growth by 0.4 steps per unit. This is a side effect of the general rule rather
  than a deliberate design choice.

---

## 4. Limits: what the player can actually do in one turn

*Source: `policy/limits.py`, `policy/feasibility.py`*

The interpreter reports what the player asked for. The game then applies four limits, in this
order, in every mode (with Claude or offline). Each one that bites adds a plain-English note.

1. **Feasibility.** An action is blocked if:
   - its target doesn't exist, or the kind doesn't fit the target (table above);
   - it is tax, spend or legislation and the legislature's support is below **0.4**, too far
     short of a majority to force through;
   - it loosens the budget (more spending, or a tax cut) while the deficit is already above
     **10% of GDP**, and nothing in the package (a tax rise or a spending cut) pays for it for
     as long as it runs;
   - it tries to direct an independent institution (independence 0.7 or more, like the Bank of
     England) or an indicator that only such an institution sets (Bank Rate). Communicating with
     them is always allowed.

   With support from 0.4 up to 0.5 there is no majority, but the measure can be **forced
   through**: it costs **twice** its political capital, meets full resistance (1.0), and each
   measure forced through costs the legislature **0.05 support** that turn. A deficit above 5%
   gives a warning but does not block. The checks also compute a
   **resistance** score from 0 to 1: how thin the majority is ((1 - support) × the legislature's
   power) or how far the deficit is past 5%, whichever is larger. With Claude, it makes a
   backbench rebellion likelier (section 8).

   Institutions settle back toward their starting support, closing a fifth of the gap each
   turn, so a rebellion costs the majority for a few turns rather than the whole game. The
   Commons settles toward its starting support **plus half the change in national approval**
   since the start: when the polls fall 10 points, backbenchers drift 0.05 less supportive.
   Settling moves only the part of support the engine does not carry (a direct edit, a
   minister leaving, the polls' drift). What the engine carries (a whip, a rebellion's shock)
   fades at the engine's own rate, 72% kept a turn (section 5), and nothing else pulls it back,
   so a whip never ends below never whipping and a rebellion never recovers past the quiet
   path.

   **Whipping** rebuilds a majority: a speech (communicate) aimed at the legislature pushes its
   support up by 1 step (0.1) per unit of magnitude. It is priced as a full measure, not a
   speech, so a whip at full strength takes 1.0 of the 1.5 capital, and it does not wear out
   with repetition. Offline, any response mentioning whips, backbenchers, MPs, the Commons or
   Parliament is read as a whip.
2. **Diminishing returns.** Repeating the same kind of action on the same target, in the same
   direction, within 4 turns halves its magnitude for each earlier use (one repeat: 50%, two:
   25%). A policy meant to keep running should say so with its duration instead. Whipping the
   legislature is exempt. Going the other way on a measure that is still running (cutting a tax
   you raised for 6 turns, two turns in) is a **U-turn**, not a repeat: it keeps its full size
   but costs an extra 1.0 capital per unit for the credibility it spends.
3. **Political capital.** A turn's package costs the sum of its magnitudes, with speeches
   (communicate) at a quarter, whips at full price, measures forced through at double and do
   nothing free, times each measure's **lifetime**: every turn a measure runs after its first
   adds a tenth of its first turn's cost (a 12-turn measure costs 2.1 times a one-off, a 24-turn
   one 3.3 times), charged on the turn it is decided. Speeches run for 3 turns at most. The
   budget is **1.5 a turn**. Above it, every measure is scaled down by the same share to fit.
   The review screen shows each measure's capital and deficit cost over its life.
4. **Deficit ceiling.** The deficit the package would leave (today's value, plus what is already
   landing this turn, plus every tax and spending move in it) may not pass 10% of GDP, on any
   turn the package's costs run. Only the borrowing measures are scaled down to fit, and tax
   rises or spending cuts in the same package make room for them while they last: a tax rise
   that ends after six turns does not pay for a tax cut in its seventh. Once the deficit is past 10%, a package may still spend or cut taxes as
   long as its own tax rises and spending cuts pay for it: the borrowing is scaled to what they
   cover, so the package adds nothing to the deficit. Only borrowing that nothing in the package
   pays for is blocked outright.

The turn log keeps both the requested and the applied actions.

A checked package belongs to the turn and the world it was checked against. Confirming it
checks that it is still the same turn and that nothing in the world has changed since, and
runs the limits again to make sure the actions are exactly what they allow. Otherwise it is
refused and has to be checked afresh, so a package cannot be confirmed twice, after the
Commons has turned, or with its actions edited after the check.

---

## 5. How a change spreads: propagation

*Source: `world/propagation.py`*

The engine is a linear system on the graph, worked in standard steps so edge weights compare
across node kinds.

### Shocks

Everything that moves the world arrives as a **shock**: a push on one node, in steps, starting
some turns from now and lasting some turns. Actions, scenarios, outcomes and event tags all
become shocks.

- An ordinary shock adds its push each turn it runs. A one-off scenario shock adds it once.
- A **held** shock (every policy action) keeps the push at its size for its duration, which is
  how a running policy works.

### Fading

A node's own push fades once nothing sustains it. Each turn it keeps a share of it:

| Node | Kept each turn | Roughly halves in |
|---|---|---|
| Country | 80% | 3 turns |
| Sector | 90% | 6.5 turns |
| Institution | 72% | 2 turns |
| Indicator | 85% (unless it sets its own) | 4 turns |
| Deficit (its own setting) | 50% | 1 turn |

So shocks pass and the world drifts back, rather than every push being a permanent level shift.

### Spreading along edges

Each turn, a node's deviation from where it would otherwise be is its own (fading) push plus
what its drivers pass on: **weight × 0.9 × the driver's deviation**, taken `lag` turns ago. The
0.9 damping means a chain of links always loses strength. Same-turn (lag 0) links are solved
together until they settle. Only `DRIVES`, `SUPPLIES`, `TRADES_WITH` and `INFLUENCES` edges
spread changes, and never into or out of population groups (approval has its own rules,
section 6). Groups are not engine nodes at all, so a shock aimed at a group does nothing. An
outcome that tries one is sent back: a group's reaction goes in its group effects, under the
event cap.

Example from the starting world: higher energy prices raise inflation (weight 0.6, one turn
later) and cut manufacturing output (0.1, one turn later); higher inflation raises Bank Rate (0.5, one turn later); higher Bank Rate lowers house
prices (0.25, three turns later) and inflation (0.4, six turns later), and raises unemployment
(0.2, six turns later). Higher unemployment lowers Bank Rate (1.0, one turn later): the Bank cuts
in a slump. More output from finance raises house prices a turn later (0.4): credit. A higher
deficit raises Bank Rate (0.15, one turn) and inflation (0.1, two
turns). Finance, manufacturing and the public sector feed UK growth in the same turn (0.4, 0.5 and
0.5), and slower growth raises unemployment a turn later (0.4) and widens the deficit (0.5): tax
receipts fall and benefit bills rise. EU growth reaches UK growth
(0.45) and, a turn later, UK manufacturing (0.3), since the EU buys half of UK goods exports.
The pound rises with UK growth (0.3) and Bank Rate (0.2) and falls with the deficit (0.4), all in
the same turn; a weaker pound raises inflation two turns later (0.3). No policy can aim at the
pound directly.

### Stability guard

If the same-turn links formed a loop that amplified itself, the turn could never settle. The
engine refuses to run such a graph, and it rejects any proposed graph change that would push the
loop gain to 0.9 or above (section 7).

### Bounds

Indicators are held within their hard floor and ceiling, approval and support within 0 to 1,
and sector output at zero or above. A bound stops a node as it moves, within the turn, so what
it passes on is the bounded change, and the push that would have taken it past the bound is
dropped. It then fades back from the bound, never from beyond it: unemployment pushed from 24%
toward 30% sits at its 25% ceiling and eases back toward 24%, rather than reading 25%, then
24.1%, then on down past where it started. A policy that keeps pushing holds the node at the
bound for as long as it runs. The balance test treats bounds as a backstop: indicators should
sit at one in under 10% of turns.

### The forecast

Before outcomes are chosen, the engine runs the next 24 turns 100 times, each time drawing every
edge weight at random around its value (using its uncertainty). Each run plays the whole turn
as resolve does, from the same state: what is already in flight, plus this turn's actions, the
scenario's shocks, bills forced through, business mood, any crisis abroad and the interest bill.
After each turn's step it applies the same rules as resolve: ministers sacked or resigning cost
the Commons support, institutions settle back toward where they started, business mood settles
and the debt reprices, and a foreign-policy action's change to trade carries the effects from
the next turn on. So with the central weights and no outcome, the forecast is exactly what
resolve does for every indicator, sector, country and institution. It reports the average change
and the 10th and 90th percentiles for every node, within bounds. The draws are seeded from the
game seed and turn, so they reproduce exactly. Resolving the turn then uses one run with the
central weights, unless the game has a hidden world.

### A hidden world per game

New games from the command line have a **hidden true world** (D1, `GameConfig.hidden_world`;
`--central-world` turns it off). When the game starts, every edge weight is drawn once from its
uncertainty (a normal draw, kept within 3 standard deviations, the range the stability guard
already allows), seeded from the game seed, and every turn is resolved on those weights. The
forecast still draws around the central weights, so it shows the government's best guess and its
spread, and the world plays out as one draw from it. The forecast says so in its list of what it
does not project. Each edge keeps its draw for the whole game; when a lasting change moves an
edge's weight (a trade deal), the draw moves with it. After the election the game shows the five
links where this world differed most from the guess. With it off (the default in code, so the
backtests, the stability check and older saves), the world behaves exactly as the central
forecast says.

The forecast lists what it does not project: group approval (shown as no change), crises abroad
in later turns (drawn turn by turn), the Commons' drift with the polls after this turn (approval
is held where it is), and the outcome, which is not chosen yet.

The forecast also keeps a **baseline**: the same crisis with none of this turn's decisions (no
actions, no sackings, only the job protection already running). Each draw runs it on that
draw's own edge weights, so the difference is what the decisions alone do, and nothing else
differs between the two. A new wage subsidy is therefore credited with the jobs it saves rather
than counted on both sides. The offline forecaster uses that difference to decide who welcomes
the response and who pushes back; the scenario's own harm is not blamed on the response.

### The decision preview

*Source: `game/preview.py`*

The review screen, before you confirm, shows what the package does by the election. It plays the
rest of the parliament with resolve itself, the central weights and the full approval rules, once
with the package and once doing nothing this turn. Both face this turn's scenario and then quiet
months: no outcome effects, no new crises and no further decisions. It shows the deficit and debt
after the next turn, every six turns and at the election, and the vote and seats at the election
for each. For every measure it also plays the package without that measure: the measure's cost is
how much higher the debt is at the election with it, beside the political capital it uses. This is
what the decisions alone do in a calm world, not a forecast of what will happen.

---

## 6. Popularity, approval and the election

*Source: `population/popularity.py`*

### Each group's target

Every turn, each group has a **target approval**:

- start from its **lean**;
- for each indicator it cares about: + 0.05 × weight × the indicator's change from what voters
  are used to (below), in steps;
- for each sector that employs it: + 0.05 × weight × that sector's output change from what
  voters are used to, in steps;
- for each institution that influences it: + 0.05 × weight × how far the institution's support
  is above or below 0.5, in steps (measured from 0.5, not from the start, so an institution
  that already backs the government lifts the group from turn one);
- plus all active **approval events** for the group (below), gains and losses capped
  separately, each just under 0.15;
- minus the **debt penalty**: if the deficit is above 5% of GDP, every group loses 0.03 for each
  point above 5. Borrowing is tolerated up to a point, then costs credibility with everyone.

The target is kept within 0 to 1.

### What voters are used to

Voters adapt. Each indicator and sector a group feels starts with a **baseline** at its value
when the game began. After each turn the baseline closes part of the gap to the current value,
so voters take half of a lasting change as normal after **12 turns**. A price rise hurts most
when it is new; a year later it is partly forgotten, and a lasting improvement likewise stops
earning credit. Approval each turn is judged against the baselines voters held coming into it.

### Moving towards it

Approval closes **half the gap** to its target each turn. This gives voters memory: a shock is
felt over several turns, and approval drifts back to lean once it passes.

### Approval events

An approval event is a one-off boost or hit to some groups. It is felt at full strength for its
first turn (plus any extra "hold" turns), then halves every 3 turns, and is dropped once it is
negligible. They come from:

- actions aimed at a group, held at full strength for the policy's duration;
- the chosen outcome's group reactions (Claude may claim at most ±0.1 per group);
- broken pledges (see below).

A group's gains from events and its losses are capped separately. Each counts in full up to 0.10
and then tapers toward 0.15 without reaching it, so no pile of good news can carry a group on
its own, and good news never hides bad: with +0.25 of good news active, a −0.05 broken pledge
still takes 0.05 off the target. (When the cap applied to the total, it took nothing off.)

### Pledges

When the player promises something for the future ("we will not raise taxes", "no cuts to
the NHS"), the game records it as a pledge. A pledge rules out one kind of action in one
direction, either on one target or on anything of that kind. "No new taxes" rules out any
tax rise; "no cuts to the NHS" rules out spending cuts to the public sector. The direction is
always the tax's or the spending's own, not the way its target moves: "no stamp duty rise"
is broken by a stamp duty rise, though that rise aims to push house prices down, and not by
a cut.

From the next turn on, an applied action that a kept pledge rules out breaks it. A blocked
action doesn't count. The pledge is marked broken, and an approval event hits the groups who
cared: 0.05 each, or 0.02 for every group when the pledge named none. The event holds for one
extra turn and then halves every 4 turns. A pledge breaks only once, and the player is warned
before confirming a response that would break one. Every prompt lists the pledges, kept or
broken, so scenarios and outcomes can refer back to them.

### People

The game has a small cast of invented people. There are two ministers (the Chancellor and the
Housing Secretary), a union leader, the Bank governor, a tabloid editor, the opposition
leader, a backbench group chair and a business lobbyist. Each cares about a few nodes and
wants them helped or squeezed. The union leader wants the public sector helped; the
Chancellor wants it squeezed and the deficit down.

Each turn, loyalty follows what the turn's actions do to the nodes each person cares about,
not which nodes the actions name. The engine projects the actions alone over 4 turns (central
edge weights, from a quiet world) and takes each node's average move in standard steps; a
group's move is the approval the actions win or lose it directly. Each cared-for node's move
counts the way the person wants it, and the sum is their score. A score of 0.5 steps or more
raises loyalty by 0.08 (or lowers it by 0.08 when negative), a smaller one moves it in
proportion, and one under 0.05 steps is not noticed. So the Chancellor rewards a tax rise or
spending cut that brings the deficit down, and the Governor a price cap that brings inflation
down, though no action may aim at either directly. Consulting one of their nodes adds 0.04.
Someone the scenario involved, whom the leader neither helped nor crossed nor consulted,
loses 0.03.

The player can sack a minister. A minister whose loyalty falls below 0.25 resigns. Either way
the Commons loses 0.05 support and a new minister with loyalty 0.6 takes over. Other people
stay, and their loyalty shows up only in the story: prompts list everyone with their loyalty
and recent dealings, so a disloyal union leader may call a ballot and a loyal editor may
defend the leader.

### The election

*Runs after turn 24 (`GameConfig.election_turn`).*

- **National approval** is each group's approval weighted by population share.
- **Vote share** weights each group by population share × turnout, so high-turnout groups like
  pensioners count for more. Turnout rises with feeling, either way: a group's turnout is
  multiplied by 1 + 2 × |approval − lean| (at most 95%), so an angry group and an enthusiastic
  one both turn out more than a content one.
- **Seats** follow the two-party cube law: seat share = v³ / (v³ + (1 - v)³), out of 650 seats.
  A small lead in votes becomes a large lead in seats.
- The player **wins** with a majority: more than 325 seats.

---

## 7. What outcomes can change

*Source: `world/events.py`, `world/changes.py`, `world/standing.py`, `forecasting/candidates.py`*

Beyond its narrative, a chosen outcome can carry four kinds of effect, all capped.

### Group reactions

Approval events, at most ±0.1 per group (section 6).

### Extra shocks

New knock-on shocks, each at most ±1 step on a known node, for one turn.

### Event tags

Each outcome is tagged with the kinds of event in it, and some tags move the economy by fixed
amounts (in steps):

| Tag | Effect |
|---|---|
| strike | public sector output -0.5 |
| protest | legislature support -0.2 |
| backbench rebellion | legislature support -0.5 |
| market sell-off | Bank Rate +0.3, finance output -0.5 |
| market rally | Bank Rate -0.2, finance output +0.3 |
| capital flight | finance output -0.8, Bank Rate +0.2 |
| business investment | manufacturing output +0.5 |
| foreign retaliation | manufacturing output -0.4 |

Media backlash or praise, foreign praise and legal challenges have no economic effect; they
work through the outcome's group reactions. A tag listed twice counts once.

However an outcome's shocks are written, it moves any one node by at most **1 step** a turn:
its own knock-on shocks and its tags' shocks on the same node are added up, and if the total
is past 1 step every one of them is scaled down by the same share (`world/events.py`,
`MAX_OUTCOME_STEPS`). Five copies of a one-step shock are one step, not five. Note that legislature support matters: below 0.5,
tax, spending and legislation are blocked (section 4).

### Lasting changes to the world graph

An outcome may say the world itself has shifted: a trade row leaves relations worse, a bailout
ties two sectors together. Up to **3** such changes per outcome are allowed. They are checked
in order, each against the world as the earlier ones left it (after the player's own diplomatic
and military moves that turn), so a trade row that shrinks a trade edge also shrinks how far
that edge may then move:

- **Node changes**: only a country's relationship or stability, an institution's support or
  independence, or a sector's sentiment, by at most 0.2, kept within bounds.
- **Edge weight changes**: an existing edge may move by at most 0.1, or a quarter of its weight
  if that is larger, and stays within -1 to 1. Edges that define a population group (what it
  cares about, where it works, and institutions' influence on it) never change, so a run of
  outcomes cannot rewrite who a group is.
- **New edges**: weight non-zero and at most ±0.3, lag at most 6, never into a population group,
  and never `CARES_ABOUT` or `EMPLOYS` (who people are does not change in a month).
- No change may make the same-turn feedback loops unstable (loop gain 0.9 or more).

An outcome's changes are all or nothing: if any one fails, none of them is applied, so the world
never ends up half way to what the outcome described. The player's own diplomatic and military actions make their
relationship and stability changes through the same mechanism (section 3).

### What those node fields do

- **Relationship** scales trade. Both trade edges between a country and the UK weigh
  `1 + 0.5 × relationship` times what they would at a neutral relationship (from 0 to +0.1 is
  5% more trade), unless that would make same-turn feedback unstable. Trade follows the
  relationship's level, not its path: a row and its repair leave trade where it started. An
  outcome that reweights a trade edge sets its new level at today's relationship.
- **Stability** sets the odds of a crisis abroad. Below 0.5, each turn has a chance of a 1-step
  (1 point) hit to the country's growth, rising in a straight line to 30% a turn at stability 0.
  The draw comes from the game seed and the turn, so a replay sees the same crises.
- **Sentiment** is a sector's business mood. Each turn it pushes the sector's output by 0.2
  steps per unit, then settles a fifth of the way back to neutral, so a knock to confidence
  costs output for a while and then passes.

The starting world (sentiment 0, stability 0.5) has none of these effects until something
changes it.

---

## 8. Choosing the outcome

*Source: `forecasting/candidates.py`, `forecasting/scoring.py`, `forecasting/selection.py`*

### With Claude

1. Claude is shown the world, the scenario, the actions, the engine's forecast and any
   precedents from the knowledge store, and proposes **4 candidate outcomes**, one per
   **forecast slot**. Three slots are equally likely bands of the engine's forecast, ordered by
   luck for the government (`p17`, `p50`, `p83`: the 17th, 50th and 83rd percentile of luck),
   and each comes with the indicator values that define it. "Luck" reads each indicator on the
   side the voters who care about it prefer: higher inflation is unlucky, because the groups
   that care about it weigh it negatively. The fourth slot is a wildcard for what the engine
   cannot see. Every slot must be filled exactly once, so the candidates cover the forecast and
   differ by construction.
2. A separate judge call estimates each candidate's probability.
3. Each candidate gets three scores between 0 and 1:
   - **Consistency**: how well the indicator changes it claims (three turns out) fit the engine's
     forecast. Each claim is scored by how many spreads it sits from its slot's value (the
     engine's average for the `p50` band and the wildcard), and an indicator the engine expects to
     move but the candidate leaves out counts as a claim of no change. Since the bands are equally
     likely, an unlucky or lucky candidate that stays on its band scores as well as the expected
     one; the judge and base rates decide between them.
   - **Judge**: the judge's probability (or Claude's own estimate if the judge is off).
   - **Base rate**: how common its tagged events are in practice. The rates start from
     hand-set guesses and are fitted to 23 UK policy episodes and what followed them
     (`forecasting/base_rates.py`); `hog-sim base-rates` prints both, with how often each event
     came up in your saved games. Several tags all have to happen: the rarest sets the ceiling
     and each further tag lowers it a little (by its own rate to the power 0.3), because events
     cluster but don't always come together. A quiet outcome is rarer when the scenario is
     urgent (×(1 − 0.5 × urgency)) or the actions meet resistance (×(1 − 0.5 × resistance)).
4. The scores are blended (consistency 40%, judge 40%, base rate 20%, on a log scale) and turned
   into probabilities that add up to 1.

How the response was delivered shifts the base rates:

| Delivery | Effect |
|---|---|
| Stakeholders consulted | strikes, protests and rebellions 0.6×, legal challenges 0.7× |
| Imposed without consultation | strikes and protests 1.3×, rebellions 1.2×, media backlash 1.1× |
| Phased in | market sell-offs and capital flight 0.75× |

Consulting only counts for an event when the people consulted could start it: a group or
sector for a strike, a group for a protest, the Commons for a rebellion, a group or institution
for a legal challenge. Consulting the central bank does nothing for strike risk, and a strike is
still "imposed" on the workers. For other events, consulting anyone counts.

Outcome probabilities are shown as **relative likelihood** until the base rates are checked
against held-out episodes.

Resistance (section 4) multiplies the base rate of a backbench rebellion by 1 + 2 × resistance,
so a spending bill with the Commons at 0.6 support (resistance 0.32) is 1.64 times as likely to
spark one.

The engine's numbers are always the ones applied. A candidate's own claimed indicator changes
are only kept for display and logs.

### Offline

The offline forecaster (`game/stubs.py`, `EngineForecaster`) offers three outcomes around the
engine's forecast, judging the response by what it changes (the forecast less its baseline), not
by effects already in flight: as expected (60%), a backlash from the groups the forecast hurts (20%, or 12%
if stakeholders were consulted), and a welcome from the groups it helps (20%). Each reaction is
±0.03 to those groups.

When a knowledge store from earlier Claude games is available, offline play reuses the stored
outcomes for the same scenario and the same actions instead. Actions match on kind, target,
direction, whether the tax or spending itself went up or down, how long it runs (one turn, up to
6, up to 12, or longer) and rough size: small (magnitude up to 0.3), medium (up to 0.7) or large.
So a small spend never reuses outcomes written for a huge one, a one-off payment never reuses
outcomes written for a policy that runs all parliament, and cutting a bill subsidy never reuses
outcomes written for raising it. Outcomes from a turn that sacked someone are reused only on a
turn that sacks the same people, since their stories may mention it. Likewise they are reused
only on a turn delivered the same way (who was consulted, imposed or phased), breaking the same
pledges, with the same measures blocked or forced through; outcomes stored before this check
are reused only on a plain turn (no consultation, imposed at once, nothing broken or blocked).
Each stored outcome keeps
the numbers it claimed, so its story still matches them, and is re-scored against today's
forecast; only those that fit it (consistency at least 0.2) are offered.

The stored reading of a response is reused only for the same words, or for a close rewording
that agrees on every negation, direction (raise or cut), number and duration, clause by clause.
So "do not increase public spending" never reuses the reading of "increase public spending",
and "raise taxes on business and cut taxes on pensioners" never reuses the reading of the same
package with the groups swapped, though it does match the same two clauses in the other order.
Anything else goes to keyword matching instead. Delivery, pledges and sackings are always read from what the player
wrote that turn, whichever reading is used.

### Picking one

The way an outcome is picked is a difficulty profile (`--luck` when starting a game):

- `sample` (the default) draws an outcome at random by probability, so the game can't be solved.
- `fair` also draws by probability, but spreads the luck out. Outcomes are ranked best first for
  the government (by the turnout-weighted approval their events bring), and every block of four
  turns takes exactly one draw from each quarter of the luck range, in a seeded order. Bad
  outcomes therefore can't cluster: four aligned turns hold at most one draw from the unluckiest
  quarter.
- `argmax` always takes the most likely.
- `hard` takes the worse, for the government, of the two likeliest outcomes.

Every draw is seeded, so replays match.

---

## 9. Scenarios and storylines

*Source: `world/storylines.py`, `content/library.py`, `llm/scenario_gen.py`*

### Where scenarios come from

- **With Claude**, a scenario is written from the current world, its recent history and the open
  storylines. It has a lead issue, 2 to 4 suggested options, 2 to 5 stakeholder positions and one
  to three smaller in-tray items. Generated scenarios carry no shocks of their own today.
- **Offline**, scenarios come from the library (67 handwritten ones across 12 categories). An
  entry is shown only if its conditions on the world hold (for example "inflation above 5%").
  The pick is seeded and weighted, and avoids anything shown in the last 15 turns where it can.
  Some library entries carry shocks, such as a 1.5-step rise in energy prices for a gas spike.

### Storylines

Issues can run across turns:

- the lead scenario opens a storyline or moves an existing one up a stage;
- it closes only when the chosen outcome says it is resolved;
- an in-tray item is judged by what the turn's actions do to its nodes, not by whether they
  name one. The engine projects the actions alone over 4 turns (central edge weights) and
  signs each node's average move by the way voters want it to go (from the groups that care
  about it or work in it; output, growth and support are wanted up when no group feels them).
  When that helps by 0.05 steps or more, the item eases: it loses 0.25 pressure without moving
  up a stage, its idle clock restarts, and once its pressure is below 0.3 it is dealt with and
  closes. When it hurts (a cut to the sector it is about), it moves up a stage and gains 0.15
  pressure. One left alone carries over, unless it was minor (urgency below 0.3), in which
  case it fades away. So acting on an item never keeps it open longer than ignoring it;
- any open storyline nobody has touched for **3 turns** escalates a stage and gains 0.15
  pressure, so ignoring an issue never makes it quietly go away.

Storylines feed the next scenario prompt; they do not move numbers directly.

---

### The political calendar

Some turns are fixed by the calendar, and on them the lead issue is the event:

- the Budget on turns 3 and 15, with tax and spending options;
- an international summit on turns 6 and 18;
- party conference on turns 9 and 21;
- a by-election on turn 12 (and 24 in longer games), fought in a seat full of whichever group is least happy
  with the government.

Each event comes round every 12 turns. With Claude, Claude writes the event's briefing and
the rest of the in-tray, but the event sets the category and the options. Offline, the
event's own briefing is used. A calendar lead opens no storyline, and the rule that a new
storyline must lead every third turn skips calendar turns. Prompts and the dashboard show
what is coming in the next six turns.

## 10. Key numbers in one place

| Rule | Value | Where |
|---|---|---|
| Turns before the election | 24 | `core/config.py` |
| Forecast horizon, random draws | 24 turns, 100 draws | `core/config.py` |
| Political capital a turn | 1.5 | `core/config.py` |
| Push per unit of magnitude | 2 steps | `world/propagation.py` (`ACTION_STEPS`) |
| Deficit cost per unit of spending | on a sector, the output it pushes as a share of GDP (public 1.85, finance 0.85, manufacturing 0.81, housing 0.56, energy 0.30 points); otherwise 1.25 points; for at least 6 turns | `world/propagation.py` (`fiscal_steps`) |
| Damping per link | 0.9 | `world/propagation.py` |
| Max same-turn loop gain | 0.9 | `world/propagation.py` |
| Repeat window, repeat penalty | 4 turns, half per use in the same direction | `policy/limits.py` |
| Upkeep, longest speech, U-turn | +0.1 of the first turn's capital per extra turn; 3 turns; +1.0 capital per unit | `policy/limits.py` (`UPKEEP`, `MAX_TURNS`, `UTURN_CAPITAL`) |
| Law for a group: counterparty, hold without money | loses as much as the group gains; 3 turns | `world/levers.py` |
| Deregulation: workers' loss | 0.3 steps per unit for young renters and public workers (0.6 public workers for the public sector) | `world/levers.py` |
| Deficit warning, hard limit | 5%, 10% of GDP | `policy/feasibility.py` |
| Starting debt, rate repricing | 100% of GDP at 4%; half of a Bank Rate change in 24 turns | `world/seed/toy.py`, `world/debt.py` |
| Legislature majority, forcing floor | support 0.5; forced through from 0.4 at 2x capital and -0.05 support | `policy/feasibility.py` (`FORCE_*`) |
| Whip push | 0.1 support per unit, capital 1.0 per unit, no repeat penalty | `world/propagation.py` (`WHIP_STEPS`), `policy/limits.py` |
| Institutions settle back, Commons follows polls | 20% of the gap a turn (the part the engine does not carry); 0.5 support per point of approval | `game/loop.py` |
| Rebellion odds per unit of resistance | +200% | `forecasting/scoring.py` (`RESISTANCE_FACTOR`) |
| Independence threshold | 0.7 | `policy/feasibility.py` |
| Approval per weighted step | 0.05 | `population/popularity.py` (`K`) |
| Share of approval gap closed a turn | 50% | `population/popularity.py` |
| Voters get used to a lasting change | half-life 12 turns | `population/popularity.py` (`HABIT_HALF_LIFE`) |
| Approval event cap per group | gains and losses each tapering from 0.10 toward 0.15 | `population/popularity.py` |
| Debt penalty | 0.03 per point above 5% | `population/popularity.py` |
| Job protection (spending on unemployment) | holds back 80% of a rise in unemployment per unit, while it runs | `world/propagation.py` (`JOB_PROTECTION`) |
| Unfunded tax cut that spooks markets | 0.75 points a turn; then per point +0.4 Bank Rate, -8 pound, -0.12 approval | `world/propagation.py`, `population/popularity.py` |
| Seats, majority | 650, more than 325 | `population/popularity.py` |
| Outcome group reaction cap | ±0.1 | `forecasting/candidates.py` |
| Outcome extra shock cap | ±1 step each, and ±1 step a node in total with event tags; none on groups | `forecasting/candidates.py`, `world/events.py` |
| Graph changes per outcome | 3 | `world/changes.py` |
| Trade per unit of relationship | 50% of the trade weight | `world/changes.py` (`TRADE_PER_RELATIONSHIP`) |
| Crisis abroad | below stability 0.5, up to 30% a turn, -1 step growth | `world/standing.py` |
| Sentiment push, settling | 0.2 steps per unit a turn, 20% back a turn | `world/standing.py` |
| Storyline escalation | after 3 idle turns, +0.15 pressure | `world/storylines.py` |
| Broken pledge hit | 0.05 per caring group, or 0.02 for all | `policy/pledges.py` |
| Loyalty: backed, crossed, consulted, ignored | +0.08, -0.08, +0.04, -0.03 | `world/cast.py` |
| Minister resigns below, Commons cost | loyalty 0.25, 0.05 support | `world/cast.py` |
| Calendar: Budget, summit, conference, by-election | every 12 turns, from turns 3, 6, 9, 12 | `world/calendar.py` |

---

## 11. Worked example

*Spend on the public sector at full magnitude, once, from the starting world, with nothing else
happening.*

This is what the engine actually produces (no outcome effects, no scenario shocks):

| After turn | Public sector output | Deficit | Bank Rate | Unemployment | Public workers | Business owners |
|---|---|---|---|---|---|---|
| start | £500bn | 4.50% | 4.00% | 4.50% | 0.450 | 0.500 |
| 1 | £550bn | 6.35% | 4.00% | 4.50% | 0.482 | 0.466 |
| 2 | £545bn | 5.93% | 4.25% | 4.16% | 0.499 | 0.457 |
| 3 | £540bn | 5.96% | 4.50% | 4.19% | 0.500 | 0.449 |
| 6 | £530bn | 6.07% | 4.53% | 4.27% | 0.487 | 0.439 |
| 7 | £527bn | 5.17% | 4.51% | 4.29% | 0.492 | 0.458 |
| 8 | £524bn | 4.73% | 4.38% | 4.36% | 0.495 | 0.475 |

Reading it:

- Output jumps by 2 steps (2 × 5% of £500bn = £50bn), then keeps 90% of the push each turn.
- The deficit rises by the £50bn spent as a share of £2,700bn GDP, 1.85 points, and the
  spending holds it there for the minimum 6 turns, then it halves its excess each turn. The
  faster growth claws some back a turn later (0.5 × 0.9 × 0.9 ≈ 0.4 points), which is why it
  sits nearer 6%.
- A turn later, Bank Rate rises by 1.85 × 0.15 × 0.9 ≈ 0.25 points through the deficit link.
- The extra output lifts UK growth by 0.5 × 0.9 × 2 = 0.9 points, and a turn later the faster
  growth takes 0.4 × 0.9 × 0.9 ≈ 0.32 points off unemployment. Lower unemployment then nudges
  Bank Rate up by about 0.3 a turn after that, which is why it climbs to about 4.5%.
- Public sector workers are employed by the sector (weight 0.9), so their target rises by
  0.05 × 0.9 × 2 = 0.09, and their approval closes half the gap each turn. The debt penalty
  (0.03 for each point above 5%, so about 0.04 at 6.35%, felt by everyone) takes some of that
  back.
- Business owners care about the deficit and Bank Rate, so they lose approval until the
  spending's cost ends after turn 6 and the deficit starts to fall.

One quirk shows here: even with no policy at all, public workers would drift slightly above their
lean of 0.45 (to about 0.455), because the Commons' support of 0.6 is counted from 0.5, not from
where the game began.

---

## 12. Checking the balance

*Source: `game/balance.py`, `tests/test_balance.py`*

Scripted bots play full offline games pulling one lever every turn (deregulate energy, raise
pensions, spend on everything and so on), plus a "do nothing" bot. The test fails if:

- any single-lever bot wins more than 34% of its games;
- doing nothing ever wins;
- indicators sit at a hard bound in more than 10% of turns;
- acting does worse on average than doing nothing by more than half a point of vote share.

A schedule bot plays the strategy the review of 13 October found dominant: on turns 0 to 3,
a 24-turn law and a 24-turn speech for one group each, then nothing. It is held to the same
win limit as the lever bots.

The **exploit search** plays seeded random and greedy packages (each kind on each target it
fits, at sizes ±0.5 and ±1, for 1, 6 or 24 turns, up to 3 measures) once on turn 0 of a quiet
game with no outcomes, and fails on any package that costs nothing in deficit yet beats doing
nothing at the election by more than half a point of vote share.

Run `uv run python -m hog_sim.game.balance` to see the table, and add `--search 400` to run the
exploit search (it exits with an error when anything breaks a limit). A nightly workflow runs a
larger search.

### Checking against history

*Source: `game/backtests.py`, `tests/test_backtests.py`*

Seven real episodes (the 2008 bailout, 2010 austerity, the 2022 energy shock and price
guarantee, the 2022 mini-budget, the 2020 lockdown and the 2016 Brexit vote) are replayed on the
starting world as shocks plus the government's response. Each check asks whether one number
moved the way it did in reality, against the start, against the same shocks with no policy, or
for the shocks alone. A move under 0.05 steps counts as no move.

Checks the engine gets wrong today are marked as known gaps with the missing link, and the test
expects them to keep failing until the link is added. None are left today. Run `uv run python -m hog_sim.game.backtests` to see the report.

Most checks also record the rough range the number really moved within the episode's year, in
its own units (points for rates, index points for prices, per cent of output for sectors). The
same command prints how big the engine's move is against that range. This is a report, not a
test: 18 of 28 sized moves fall inside their range, and most of the rest are too small.

### Long games

*Source: `game/stability.py`, `tests/test_stability.py`*

Players from the balance test, plus one who picks a random suggested option, play 120 offline
turns. The test fails if any number turns non-finite, an indicator ends up more than 8 steps
from its start, an indicator's average distance from its start grows by more than half again
from the first half to the second (plus 0.5 steps) and ends up averaging 2 steps or more, indicators sit at a hard bound in more than 10%
of turns, or the world goes still: over the last 48 turns vote intention must move at least a
point and at least half the indicators must still move. Run
`uv run python -m hog_sim.game.stability` to see the table.
