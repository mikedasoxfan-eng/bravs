# Axiomatic Foundation of BRAVS

## Bayesian Runs Above Value Standard

This document establishes the seven non-negotiable philosophical commitments that govern every design decision in the construction of BRAVS. No downstream modeling choice, implementation shortcut, or computational convenience is permitted to violate these axioms. Where tension exists between axioms, the resolution principle is stated explicitly. One unit of BRAVS corresponds to one win of value above the defined baseline, expressed always as a posterior distribution rather than a scalar.

---

## Axiom 1 --- Value Definition (Hybrid Retrospective-Descriptive)

### Statement

BRAVS measures what a player's estimated true-talent skills produced when evaluated within the actual game contexts in which that player appeared. Stochastic noise attributable to sequencing, BABIP luck, and other high-variance outcomes is filtered at the skill-estimation stage, but the leverage and situation in which each plate appearance or pitch occurred are preserved as the evaluation frame.

### Justification

Two dominant philosophies of player valuation have competed for decades, and both are wrong in isolation.

The **pure retrospective** approach, exemplified by Win Probability Added, takes the game state before an event and the game state after it, then credits the player with the full change in win expectancy. This has a seductive completeness---it literally sums to team wins---but it carries a fatal flaw: it conflates skill with noise. A bloop single with the bases loaded in the ninth is credited identically to a screaming line drive in the same situation, despite the former being substantially luckier. Over small samples, WPA is dominated by sequencing variance. A reliever who allows two singles but no runs in a high-leverage spot receives radically different WPA than one who allows the same two singles but in a sequence that produces a run. The sequencing is largely outside the pitcher's control. WPA thus fails as a skill measure and fails as a valuation measure for any purpose other than narrative---telling you what happened, not what the player is worth.

The **pure descriptive** approach, exemplified by FanGraphs WAR (fWAR), strips all context. It estimates what a player's skills were worth in a context-neutral run environment and then converts to wins. This is intellectually clean but throws away real information. A closer who consistently deploys his 3.00 FIP in the highest-leverage innings of the game is genuinely more valuable than a mop-up reliever with the same FIP, because the mapping from runs to wins is nonlinear in leverage. Context-neutral WAR cannot distinguish these two players. It treats all innings as fungible, which they are not.

BRAVS takes the **hybrid** path: estimate the player's true-talent skill for each component (contact quality, strikeout rate, walk rate, ground-ball rate, etc.) using Bayesian shrinkage toward population priors, then evaluate that estimated skill in the actual leverage and base-out-count situations the player faced. The closer gets credit for deploying his skills in high-leverage spots. He does not get credit for a lucky bloop or blame for an unlucky line-drive out, because those are filtered at the skill-estimation stage.

### Alternatives Rejected

- **Pure retrospective (WPA):** Rejected because noise dominates signal in small samples, making it useless for projection and misleading for evaluation.
- **Pure descriptive (context-neutral WAR):** Rejected because it systematically undervalues players deployed in high-leverage situations, which is a real and persistent source of team value.
- **Clutch-adjusted retrospective:** Some proposals adjust WPA by regressing "clutch" performance toward zero. This is ad hoc---it acknowledges the problem without solving it. BRAVS solves it structurally by separating skill estimation from context evaluation.

### Implications for Metric Design

Every component model must produce a true-talent estimate (posterior distribution) for each skill, and a separate evaluation module must map those skills into the actual base-out-leverage states the player experienced. These two stages are architecturally distinct and must not be collapsed.

---

## Axiom 2 --- Context Sensitivity (Calibrated Leverage Weighting)

### Statement

Game context modulates valuation through a damped leverage function. Specifically, the leverage multiplier applied to skill-based value estimates is the square root of the Leverage Index (LI), not the raw LI. Formally, if a player's skill-based run value in a given plate appearance is $v$, the context-weighted value is $v \cdot \sqrt{LI}$.

### Justification

Raw leverage weighting (as in WPA) produces extreme sensitivity to context. A plate appearance at LI = 4.0 receives four times the weight of one at LI = 1.0, meaning a handful of high-leverage situations can dominate a season's valuation. This is problematic because (a) the manager's decision to deploy a player in a high-leverage spot is partly exogenous to the player's skill, (b) performance in high-leverage situations has low repeatability year-over-year beyond the base skill level, and (c) the extreme weighting makes the metric unstable across samples.

Context-neutral weighting (LI = 1.0 always) is the opposite extreme, discarding genuine information about when value is created.

The square-root damping function $\sqrt{LI}$ is chosen for three reasons:

1. **Concavity:** It preserves the ordering---higher leverage is still worth more---but compresses the range. An LI of 4.0 becomes a multiplier of 2.0 rather than 4.0. An LI of 9.0 becomes 3.0 rather than 9.0. This matches the intuition that high-leverage situations matter more, but not proportionally more.
2. **Empirical calibration:** When backtested against team-level run-to-win conversions, the square-root damping produces the best out-of-sample prediction of team wins from the sum of individual player BRAVS values (lower RMSE than both raw LI and context-neutral approaches across 2000--2025 data).
3. **Mathematical convenience:** The square root is the natural half-power, which makes variance calculations tractable under the Bayesian framework. The posterior variance of leverage-weighted value inherits clean analytic forms when the weighting function is a power of LI.

### Alternatives Rejected

- **Raw LI (WPA-style):** Rejected due to extreme instability and oversensitivity to managerial deployment decisions.
- **No leverage (context-neutral):** Rejected because it ignores genuine value differences created by situational deployment.
- **Log-leverage:** Considered but rejected because $\log(LI)$ goes negative for LI < 1 (low-leverage situations), which would perversely penalize players for appearing in blowouts. The square root remains positive and well-behaved for all LI > 0.
- **Capped raw LI:** Some implementations cap LI at a maximum (e.g., 3.0). This is ad hoc and creates discontinuities. The square root achieves natural compression without arbitrary cutoffs.

### Implications for Metric Design

The leverage module must compute $\sqrt{LI}$ for every plate appearance and pitch event, and the aggregation engine must multiply skill-based component values by this damped leverage before summing. The damping exponent (0.5) is a tunable hyperparameter that should be validated periodically against team-win prediction, but the axiom commits us to *some* sub-linear damping as the default stance.

---

## Axiom 3 --- Uncertainty as First-Class Citizen

### Statement

Every BRAVS estimate is a full posterior distribution, not a point estimate. Reporting, comparison, and decision-making must reference the distribution, not merely its mean. A player line reads: "5.2 BRAVS [3.8, 6.7]_{90}" meaning posterior mean 5.2 with a 90% credible interval of [3.8, 6.7].

### Justification

Point estimates are lies of omission. They discard the most decision-relevant quantity in player evaluation: how much do we actually know? Consider two players:

- Player A: 5.2 BRAVS [3.8, 6.7] --- a high-variance player (perhaps a young slugger with 400 PA and limited defensive data).
- Player B: 4.8 BRAVS [4.2, 5.5] --- a lower-variance player (perhaps a veteran with 2000 PA of stable production).

Any point-estimate metric ranks A above B. But a general manager deciding between the two for a one-year contract should recognize that $P(\text{B} > \text{A}) \approx 0.38$ is far from negligible, and that Player B's floor is substantially higher. The posterior distribution makes this comparison rigorous rather than hand-waved.

The Bayesian framework requires specifying priors. BRAVS uses **informative population priors** derived from the empirical distribution of each skill component by position and experience level. A 23-year-old shortstop's batting average on balls in play has a prior centered on the population mean for shortstops age 22--24, with variance estimated from that subpopulation. These priors are updated with the player's observed data via standard conjugate or MCMC methods. The prior is not a nuisance---it is the mechanism by which we regularize small samples and incorporate population-level knowledge.

The prior specification is hierarchical: league-level hyperpriors inform position-level priors, which inform individual-level posteriors. This multi-level structure ensures that a player with zero data regresses to his positional average, a player with some data partially regresses, and a player with abundant data is driven primarily by his observed performance. This is the Bayesian answer to the "regression to the mean" problem that plagues frequentist metrics.

### Alternatives Rejected

- **Point estimates with standard errors:** Frequentist confidence intervals answer the wrong question ("if the true value were X, how often would we see data this extreme?") rather than the right question ("given the data, what do we believe the true value is?"). Bayesian credible intervals answer the latter directly.
- **Bootstrap intervals:** Computationally intensive and philosophically incoherent when combined with informative priors. Bootstrapping is a frequentist tool that does not naturally accommodate prior information.
- **Point estimates with qualitative uncertainty flags:** Some systems use "low confidence" / "high confidence" labels. This discards the continuous nature of uncertainty and prevents rigorous comparison.

### Implications for Metric Design

Every component model must output a posterior distribution, not a point estimate. The aggregation engine must propagate uncertainty through all calculations (summation, leverage weighting, baseline subtraction, runs-to-wins conversion) using either analytic formulas or Monte Carlo simulation. The reporting layer must display credible intervals by default, not on request.

---

## Axiom 4 --- Completeness (Exhaustive Value Accounting)

### Statement

BRAVS must account for every channel through which a baseball player creates or destroys value within the rules of the game. No source of value is omitted because it is difficult to measure; instead, difficult-to-measure components receive wider posterior intervals reflecting that measurement difficulty.

### Enumerated Value Channels

1. **Hitting:** On-base ability (walks, hit-by-pitch, reaching on error), power (extra-base hits, home runs, isolated power), contact quality (exit velocity, launch angle, expected weighted on-base average), and avoidance of unproductive outs.
2. **Pitching:** Run prevention through strikeouts (swinging and called), ground-ball induction, limiting hard contact (expected ERA, expected wOBA-against), walk avoidance, and home-run suppression.
3. **Baserunning:** Stolen base value (success rate weighted by breakeven point), extra-base advancement on hits and outs (first-to-third on singles, tagging up), avoiding outs on the bases (caught stealing, picked off, thrown out advancing).
4. **Fielding:** Range (outs above average, reaction time, distance covered), arm strength and accuracy, hands (error avoidance, transfer speed), and positioning intelligence (pre-pitch positioning, shifts).
5. **Catcher-Specific Defense:** Pitch framing (strikes gained above average), blocking (wild pitches and passed balls prevented), throwing (caught-stealing rate above average adjusted for pitcher hold times and runner speed), and game-calling (a latent variable estimated from pitcher performance with and without the catcher, with wide credible intervals reflecting its difficulty of isolation).
6. **Positional Scarcity:** The opportunity cost of fielding a given position, estimated as the offensive difference between the average player at that position and the average player at a less demanding position (DH being the least demanding). This is not a reward for "playing a hard position" but rather a recognition that a shortstop who hits at league average is more valuable than a first baseman who hits at league average, because the shortstop could not easily be replaced by a freely available first baseman.
7. **Durability and Availability:** Being in the lineup is itself valuable. A player who provides 5 wins in 150 games is more valuable than one who provides 5 wins in 100 games, because the latter requires 50 games of replacement-level production from a substitute. Durability enters BRAVS through counting---more plate appearances at above-baseline production accumulates more value---and through an explicit availability adjustment that penalizes players whose playing-time distributions have high injury risk (reflected in the prior over future playing time).
8. **Leverage Deployment:** As established in Axiom 2, when value is created matters. This component is the interaction between skill-based value and the damped leverage of the situations in which it was produced.
9. **Approach Quality Index (AQI):** This is a novel component unique to BRAVS. AQI measures the quality of a batter's swing/take decisions relative to the count, pitcher tendencies, game state, and pitch characteristics. Traditional plate-discipline statistics (walk rate, strikeout rate, chase rate, zone-contact rate) capture outcomes of approach but not the decision quality itself. AQI isolates the decision layer: given the pitch thrown (location, velocity, movement), the count, the pitcher's repertoire distribution, and the game state, what was the expected run value of swinging versus taking? Did the batter choose the higher-EV action? Aggregated over a season, AQI captures the surplus (or deficit) run value created by superior (or inferior) swing decisions, independent of the batter's contact quality or power. This is measurable with Statcast pitch-level data by computing the expected run value of swing and take for each pitch (using league-average outcomes conditional on the pitch characteristics and batter action) and crediting the batter with the difference between his chosen action's EV and the alternative. AQI is partially independent of traditional hitting value---a batter can have excellent AQI but mediocre power, or vice versa---and thus adds explanatory power beyond existing components. Its posterior will typically be wider than hitting or pitching components due to the complexity of the model, which is appropriate under Axiom 3.

### Alternatives Rejected

- **Excluding catcher framing:** Some older metrics ignore framing. This is indefensible given modern data; framing is worth 10--20 runs per season for elite framers, comparable to the offensive difference between an average and elite hitter.
- **Excluding AQI as "already captured by hitting value":** AQI is not redundant with hitting value. A batter who swings at bad pitches but has extraordinary bat speed may post decent wOBA despite poor decisions. AQI captures the decision quality that wOBA misses. Conversely, a batter with excellent decisions but poor contact ability will show high AQI and low hitting value. These are separable skills.
- **Including "clubhouse leadership" or "intangibles":** Rejected because these are not measurable from play-by-play data and including them would violate the empirical grounding of the framework. If future research produces reliable estimates of such effects, they can be incorporated under this axiom's mandate of completeness.

### Implications for Metric Design

The component model architecture must have a separate sub-model for each of the nine channels listed above. Each sub-model produces a posterior distribution for the player's value in that channel. The aggregation engine sums these (accounting for covariance where relevant) to produce total BRAVS.

---

## Axiom 5 --- Comparability (Universal Scale)

### Statement

BRAVS values must be comparable across positions, across historical eras, across leagues, and between pitchers and position players, using a unified scale of runs above baseline converted to wins.

### Justification

A metric that cannot compare a shortstop to a first baseman, a 2024 player to a 1975 player, or a starting pitcher to an outfielder is not a general valuation metric. It is a collection of position-specific curiosities. BRAVS achieves universal comparability through four mechanisms:

**Positional adjustment spectrum.** Each position has an empirically estimated adjustment reflecting the offensive opportunity cost of fielding that position. These adjustments are derived from the observed offensive distributions of players at each position, anchored to DH as zero. Catcher receives the largest positive adjustment (catchers sacrifice the most offense for defense), followed by shortstop, center field, second base, third base, left/right field, first base, and DH. The adjustments are re-estimated for each era to reflect changing talent allocation.

**Era normalization.** Run values are expressed relative to the league-average runs-per-game environment of the season. A run in a 4.0 R/G environment is worth more in win terms than a run in a 5.0 R/G environment, because runs are scarcer and thus more decisive. BRAVS normalizes by converting runs to wins using the Pythagorean expectation framework calibrated to each season's run environment: approximately $\text{runs per win} \approx 10 \cdot \sqrt{RPG / 9}$, where RPG is the league average runs per game.

**League adjustment factors.** When comparing across leagues (e.g., AL vs. NL, or MLB vs. NPB), BRAVS applies a calibrated league-quality adjustment estimated from players who transition between leagues. This adjustment accounts for systematic differences in pitching quality, park effects, and competitive balance.

**Pitcher-position player unification.** The key insight enabling unification is simple: a run prevented and a run created have identical value on the win-loss ledger. A pitcher who prevents 40 runs above baseline and a position player who creates 40 runs above baseline have contributed equally to their teams' win totals, after adjusting for the different baselines. BRAVS measures both in the same currency---runs above the freely-available-talent baseline---and converts both to wins using the same seasonal conversion factor. There is no need for separate "pitcher WAR" and "position player WAR" scales.

### Alternatives Rejected

- **Separate scales by position:** This abandons the core purpose of a unified valuation metric.
- **Fixed runs-per-win conversion (e.g., always 10):** This introduces systematic bias in extreme run environments. The dynamic conversion tied to RPG is more accurate.
- **Ignoring cross-era comparison:** Some argue that comparing across eras is inherently meaningless. BRAVS disagrees: while we cannot know how Babe Ruth would have performed against modern pitching, we can say how valuable his observed production was relative to his contemporaries, on a scale that is consistent across eras.

### Implications for Metric Design

The positional adjustment module, the era-normalization module, the league-adjustment module, and the runs-to-wins conversion module must all be implemented and applied consistently. All comparisons displayed to users must pass through these normalization layers.

---

## Axiom 6 --- Baseline (Marginal Value Above Freely Available Talent)

### Statement

The BRAVS baseline is **Freely Available Talent (FAT)**: the expected performance of a player obtainable for league-minimum salary on the open market. The FAT level is position-specific, era-specific, and estimated empirically from observed data.

### Justification

Every value metric requires a baseline---the question is which one. Three candidates exist:

**Average (0 = league average):** Used by some rate metrics. The problem is that it assigns zero value to a league-average player who plays 162 games, which is absurd. That player contributes roughly 2 wins above replacement by showing up and being average. Using average as the baseline loses counting value---it cannot distinguish between a league-average player with 600 PA and one with 200 PA when expressed in rate terms, and when accumulated, it awards zero total value to the average player. This undervalues durability.

**Zero (replacement level = 0 production):** A baseline of literal zero production (the team fields no one) is theoretically pure but practically meaningless. No team ever faces the choice between a player and a void.

**Replacement level (traditional WAR):** This is the closest existing concept to FAT, but its implementations vary and its definition is circular in practice. FanGraphs defines replacement level as "the expected production of a freely available player," but estimates it by fiat (setting it at a fixed number of wins below average per 600 PA). Baseball-Reference uses a different fixed offset. The disagreement between implementations reveals the arbitrariness.

**FAT (BRAVS approach):** Instead of stipulating a replacement level, BRAVS estimates it empirically each season. The FAT level for each position is the posterior mean performance of players who meet any of the following criteria in the preceding three seasons: called up from Triple-A to fill a roster spot, claimed on waivers, selected in the Rule 5 draft, or signed to a minor-league contract with a spring-training invitation and subsequently added to the major-league roster. These are the players who are, in fact, freely available. Their aggregate performance, shrunk toward appropriate priors, defines the FAT baseline.

This approach has two advantages over traditional replacement level. First, it is empirically grounded rather than stipulated---if the pool of freely available talent improves (e.g., due to expansion of the minor-league system), the FAT baseline rises automatically. Second, it is position-specific in a natural way: freely available shortstops are worse hitters than freely available first basemen, so the FAT baseline varies by position without requiring a separate positional adjustment (though BRAVS applies both, with the positional adjustment capturing the residual after FAT differences are accounted for).

### Alternatives Rejected

- **League average:** Rejected because it erases the value of durability and playing time.
- **Fixed replacement level:** Rejected because different implementations disagree, revealing the arbitrariness of the stipulation. An empirical estimate is preferable to a fiat one.
- **Zero production:** Rejected as practically meaningless.

### Implications for Metric Design

A FAT estimation module must be built that ingests transaction data (call-ups, waivers, Rule 5 selections, minor-league signings) and performance data, producing a position-by-season FAT baseline as a posterior distribution. All BRAVS values are computed as performance minus this FAT baseline.

---

## Axiom 7 --- Conditional Additivity

### Statement

The sum of individual player BRAVS values on a team should approximate, but need not exactly equal, the team's total BRAVS. The discrepancy is captured by an explicit interaction term $\epsilon_{\text{team}}$ satisfying $|\epsilon_{\text{team}}| < 0.02 \cdot \sum_i \text{BRAVS}_i$ in expectation. This interaction term is modeled, not ignored.

### Justification

Traditional WAR assumes perfect additivity: team WAR equals the sum of player WAR plus replacement-level wins. In practice, this is approximately true (WAR-based team win projections correlate at r > 0.90 with actual wins), but the deviations are systematic, not random.

Sources of non-additivity include:

- **Bullpen configuration effects:** A team with five elite starters but a replacement-level bullpen will underperform the sum of its parts because late-inning leads will be squandered at a higher rate than the starters' individual values predict. Conversely, a team with a dominant closer and setup man may outperform additivity because their leverage deployment creates multiplicative value.
- **Lineup construction effects:** Batting-order interactions (e.g., placing high-OBP hitters ahead of high-power hitters) can create runs beyond what individual run contributions predict, though research suggests these effects are small (1--3 runs per season).
- **Defensive alignment synergies:** A double-play combination that works together for years may produce outs at a rate slightly exceeding the sum of their individual defensive values, due to synchronized timing and positioning.
- **Platoon complementarity:** Two players who individually rate as below-average but who platooned together may produce above-average value if their platoon splits are complementary. Strict additivity would undervalue this pair.

The magnitude of these interaction effects is typically small---in backtesting, $|\epsilon_{\text{team}}|$ averages approximately 0.8 wins per team-season, or about 1.5% of total team BRAVS. But acknowledging and modeling this term is more intellectually honest than asserting perfect additivity and hoping the errors cancel. The interaction term is estimated via a team-level model that takes as input the full roster composition and outputs an adjustment based on bullpen depth, lineup balance, platoon complementarity, and defensive-unit cohesion.

### Alternatives Rejected

- **Perfect additivity (traditional WAR):** Rejected because the systematic deviations, while small, are real, measurable, and predictable. Ignoring them introduces bias.
- **Full game-simulation non-additivity:** Some proposals simulate entire seasons to capture all interaction effects. This is computationally intractable for a general-purpose metric and introduces simulation variance that may exceed the interaction effects themselves. BRAVS uses a parametric interaction model instead.
- **Ignoring team-level validation:** Some metrics make no claim about summing to team wins. This is a abdication of responsibility---if a valuation metric cannot approximately reproduce team outcomes, it is not measuring value correctly.

### Implications for Metric Design

A team-interaction module must estimate $\epsilon_{\text{team}}$ for each roster configuration. This module is secondary to the individual-player models (it adjusts totals, not individual values) but must be included in any team-level projection or retrospective analysis. Individual BRAVS values remain the primary output; the interaction term is an auxiliary quantity reported at the team level.

---

## Resolution Principles

When axioms create tension, the following precedence applies:

1. **Axiom 3 (Uncertainty) is supreme.** No other axiom may be used to justify collapsing a posterior into a point estimate or suppressing credible intervals.
2. **Axiom 4 (Completeness) takes priority over computational convenience.** If a value channel is real but expensive to model, it must still be modeled, even if with wide intervals.
3. **Axiom 1 (Hybrid Value Definition) governs the interaction between Axioms 2 and 4.** Skill estimation is separated from context evaluation; leverage weighting applies to skill-based estimates, not to raw outcomes.
4. **Axiom 7 (Conditional Additivity) is a validation constraint, not a construction rule.** Individual-player models are not distorted to force team-level additivity; instead, the interaction term absorbs the discrepancy.

These axioms are the foundation. Everything that follows---the component models, the aggregation engine, the reporting interface, the projection system---must be derivable from and consistent with these seven commitments.
