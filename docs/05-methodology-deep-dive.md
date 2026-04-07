# BRAVS: A Bayesian Framework for Probabilistic Baseball Player Valuation

## Abstract

We present BRAVS (Bayesian Runs Above Value Standard), a probabilistic framework for baseball player valuation that addresses fundamental limitations of Wins Above Replacement (WAR). BRAVS produces full posterior distributions over player value — measured in wins above Freely Available Talent (FAT) — rather than the point estimates reported by existing metrics. The framework decomposes player value into nine explicitly defined components: hitting, pitching, baserunning, fielding, catcher-specific value, positional adjustment, leverage context, durability, and a novel Approach Quality Index (AQI) measuring batting decision quality via pitch-level run value differentials. Each component is estimated using Bayesian conjugate models with informative population priors, producing posterior distributions that properly quantify measurement uncertainty. BRAVS employs a dynamic Pythagorean-derived runs-per-win conversion that varies with run environment, a damped leverage adjustment using $\sqrt{\text{gmLI}}$ that credits players for deploying skills in high-leverage situations, and multi-dimensional park factor adjustments. Validation against 22 notable player-seasons spanning 1913–2023 demonstrates that BRAVS produces intuitive rankings with properly calibrated credible intervals, handles two-way players (Ohtani), catchers (Piazza), closers (Rivera), and historical eras (Gibson 1968, Ruth 1927) within a single unified framework. The complete pipeline processes 731 player-seasons in 3.7 seconds. We identify systematic inflation relative to WAR as the primary limitation, driven by the AQI component and dynamic RPW, and propose concrete remediation strategies.

---

## 1. Introduction and Motivation

Wins Above Replacement has been the consensus standard for baseball player valuation since its widespread adoption in the late 2000s. By reducing a player's total contribution to a single number on a common scale, WAR enabled comparisons that were previously impossible — a shortstop's defensive value could be weighed against a designated hitter's offensive production, a starter's dominance against a closer's leverage-amplified impact.

Yet WAR has significant structural problems. Its defensive component relies on metrics (UZR, DRS) with year-over-year correlations around 0.4 — implying that more than half of a single season's defensive measurement is noise (Lichtman, 2003). The two major implementations — fWAR (FanGraphs) and bWAR (Baseball-Reference) — use fundamentally different pitching methodologies (FIP-based vs. RA/9-based) and different replacement level calibrations, producing values that diverge by 2+ wins for individual player-seasons. Catcher framing, worth 2–3 wins annually for elite framers, is entirely absent. All plate appearances and innings are weighted equally, as if a bases-loaded strikeout in the 9th inning of a tied game has the same value as a strikeout in the 2nd inning of a blowout. The static run-to-win conversion (~10 runs = 1 win) ignores the Pythagorean reality that runs are worth more in low-scoring environments. And most fundamentally, WAR produces a point estimate with no uncertainty measure, despite the fact that measurement error on defense alone can be ±1.5 wins.

These are not minor quibbles. They are structural deficiencies that distort player valuations in predictable, systematic ways: catchers are undervalued, closers are undervalued, players in low-scoring eras are undervalued relative to high-scoring eras, and the false precision of WAR's decimal places masks genuine uncertainty about player value.

BRAVS addresses these deficiencies by building player valuation from first principles using a Bayesian probabilistic framework. Every estimate is a distribution, not a point. Every component is independently meaningful and transparently defined. Every modeling choice traces to an explicit axiom. The result is a framework that is more honest about what we know, more principled in how it handles what we don't know, and more complete in what it measures.

---

## 2. Literature Review

BRAVS draws on several streams of prior work.

**Linear weights and run expectancy**: The foundational framework of Tom Tango, Mitchel Lichtman, and Andrew Dolphin's "The Book" (2006) provides the run-value basis for BRAVS's hitting component. The base-out run expectancy matrix — assigning a run value to every base-out state — is one of the most robust empirical tools in sabermetrics. BRAVS uses linear weights derived from this matrix to value batting events. The leverage index framework from Tango also informs BRAVS's leverage component.

**Bayesian baseball models**: Jim Albert's work on hierarchical Bayesian models for batting averages (Albert, 1992, 2006) established the template for BRAVS's shrinkage estimators. By placing a population-level prior on batting skill and updating with observed outcomes, the Bayesian approach naturally handles small samples (heavy shrinkage toward the mean) and large samples (posterior dominated by data). BRAVS extends this idea to all nine components.

**Defensive measurement**: Lichtman's UZR methodology, Baseball Info Solutions' DRS, and Statcast's Outs Above Average (OAA) represent three distinct approaches to fielding valuation. Each has known limitations: UZR relies on subjective zone classifications, DRS on proprietary methodology, and OAA on Statcast tracking data (available only since 2016). BRAVS's fielding component uses Bayesian model averaging to combine these when available, weighted by estimated reliability.

**Probabilistic WAR**: The openWAR project (Baumer et al., 2015) first demonstrated that WAR could be reformulated to produce uncertainty estimates. OpenWAR uses bootstrapping to generate confidence intervals, finding that the standard error on a single-season WAR is approximately 1–2 wins — far larger than the precision implied by WAR's decimal places. BRAVS takes this further by producing full posterior distributions from a generative model, rather than post-hoc bootstrapped intervals.

**Win Probability Added**: WPA provides a fully context-sensitive measure of player value, crediting each plate appearance by its impact on the team's probability of winning. Its virtue is philosophical purity — it measures actual impact. Its vice is extreme noise (year-over-year correlation ~0.2) because context is largely outside the player's control. BRAVS's leverage component is designed to capture the core insight of WPA (context matters) while avoiding its instability.

**Expected statistics**: The Statcast revolution introduced expected stats (xBA, xwOBA, xERA) that strip noise from outcomes by predicting results from batted ball quality (exit velocity, launch angle). BRAVS's AQI component extends this logic to the pitch level — evaluating not just the quality of contact, but the quality of the decision to swing or take.

---

## 3. Theoretical Framework

### 3.1 Axioms

BRAVS rests on seven axioms that govern all design decisions:

**Axiom 1 (Hybrid Retrospective-Descriptive Value)**: BRAVS estimates what a player's true skills produced in the actual contexts where they were deployed. It is neither purely retrospective (WPA) nor purely context-neutral (WAR), but estimates true talent and evaluates it in context.

**Axiom 2 (Calibrated Leverage Weighting)**: Context sensitivity is applied through damped leverage: $\text{LI}_{\text{eff}} = \sqrt{\text{gmLI}}$. This is the geometric mean of ignoring leverage (WAR) and using it fully (WPA).

**Axiom 3 (Uncertainty as First-Class Citizen)**: Every BRAVS estimate is a posterior distribution. A player valued at 5.2 BRAVS means: posterior mean = 5.2, with a full density function encoding our confidence.

**Axiom 4 (Completeness)**: BRAVS enumerates nine exhaustive channels through which a player creates or destroys value, including the novel Approach Quality Index.

**Axiom 5 (Comparability)**: All value is measured in runs above baseline, converted to wins through a common (but context-dependent) conversion. Pitching and hitting value are naturally commensurable.

**Axiom 6 (FAT Baseline)**: Value is measured relative to Freely Available Talent — the expected performance of a player obtainable for league minimum salary.

**Axiom 7 (Conditional Additivity)**: Player values approximately sum to team value, with a small interaction term (~2%) that is modeled rather than ignored.

### 3.2 The Generative Model

For a player-season with statistics $\mathbf{x}$, BRAVS defines a hierarchical generative model:

$$\theta_i \sim p(\theta | \mu_{\text{pos}}, \sigma^2_{\text{pos}})$$

$$\mathbf{x}_i | \theta_i \sim p(\mathbf{x} | \theta_i)$$

where $\theta_i$ is the player's true-talent parameter vector, $\mu_{\text{pos}}$ and $\sigma^2_{\text{pos}}$ are position-specific population hyperparameters, and the likelihood $p(\mathbf{x} | \theta_i)$ is component-specific.

The posterior is:

$$p(\theta_i | \mathbf{x}_i) \propto p(\mathbf{x}_i | \theta_i) \cdot p(\theta_i | \mu_{\text{pos}}, \sigma^2_{\text{pos}})$$

For conjugate models (normal-normal for hitting, pitching), this has a closed-form solution. For non-conjugate models (fielding ensemble, AQI), MCMC sampling is used in the full implementation (conjugate approximations in the current version).

---

## 4. Methodology

### 4.1 Component Specifications

#### Hitting ($H$)

The hitting component estimates batting value from weighted on-base average (wOBA):

$$\text{wOBA} = \frac{w_{BB} \cdot BB + w_{HBP} \cdot HBP + w_{1B} \cdot 1B + w_{2B} \cdot 2B + w_{3B} \cdot 3B + w_{HR} \cdot HR}{AB + BB + SF + HBP}$$

The Bayesian model:
- **Prior**: $\text{wOBA}_{\text{true}} \sim N(\mu_{\text{pos}}, \sigma^2_{\text{pos}})$ with $\mu = 0.315$, $\sigma = 0.035$
- **Likelihood**: $\text{wOBA}_{\text{obs}} | \text{wOBA}_{\text{true}} \sim N(\text{wOBA}_{\text{true}}, \sigma^2_{\text{obs}} / n)$ with $\sigma^2_{\text{obs}} = 0.09$
- **Posterior**: $\text{wOBA}_{\text{post}} \sim N(\mu_{\text{post}}, \sigma^2_{\text{post}})$ via conjugate update

Conversion to runs: $H = (\text{wOBA}_{\text{post}} - \text{wOBA}_{\text{lg}}) / \text{wOBA}_{\text{scale}} \times PA - \text{FAT}_{\text{runs}}$

#### Pitching ($P$)

Based on Fielding Independent Pitching, extended with Bayesian shrinkage:

$$\text{FIP} = \frac{13 \cdot HR + 3 \cdot (BB + HBP) - 2 \cdot K}{IP} + C$$

- **Prior**: $\text{FIP}_{\text{true}} \sim N(4.20, 0.80^2)$
- **Likelihood**: $\text{FIP}_{\text{obs}} | \text{FIP}_{\text{true}} \sim N(\text{FIP}_{\text{true}}, 2.25 / IP)$
- **Posterior**: Conjugate normal update

Conversion: $P = (\text{FAT}_{\text{ERA}} - \text{FIP}_{\text{post}}) / 9.0 \times IP$

#### Baserunning ($R$)

Three sub-components summed and shrunk:
$$R = \underbrace{SB \cdot r_{SB} + CS \cdot r_{CS}}_{\text{stolen bases}} + \underbrace{(\text{rate} - \text{avg}) \cdot \text{opps} \cdot r_{\text{adv}}}_{\text{advances}} + \underbrace{(\text{exp}_{GIDP} - \text{act}_{GIDP}) \cdot r_{GIDP}}_{\text{GIDP avoidance}}$$

#### Fielding ($F$)

Bayesian model averaging of available defensive metrics:
$$F_{\text{obs}} = \sum_k w_k \cdot \text{DEF}_k, \quad w = \{UZR: 0.30, DRS: 0.30, OAA: 0.40\}$$

- **Prior**: $F_{\text{true}} \sim N(0, 5^2)$ — strong regression to zero
- **Observation variance**: $\sigma^2 = 60$ per full-season (reflecting ~0.4 YoY correlation)

#### Catcher ($C$)

Four sub-components:
$$C = \underbrace{C_{\text{frame}}}_{\text{framing}} + \underbrace{C_{\text{block}}}_{\text{blocking}} + \underbrace{C_{\text{throw}}}_{\text{throwing}} + \underbrace{C_{\text{call}}}_{\text{game-calling}}$$

Framing is Bayesian-estimated with prior $N(0, 8^2)$. Game-calling uses WOWY analysis, heavily regressed (observation variance = 100).

#### Positional Adjustment ($\text{Pos}$)

Tango's defensive spectrum, prorated: $\text{Pos} = \text{adj}_{\text{pos}} \times (G / 162)$

Values: C: +12.5, SS: +7.5, CF/2B/3B: +2.5, LF/RF: -7.5, 1B: -12.5, DH: -17.5

#### Leverage ($L$)

$$L = \text{Skill}_{\text{total}} \times (\sqrt{\text{gmLI}} / 0.97 - 1.0)$$

The 0.97 normalization ensures average leverage = no adjustment.

#### Durability ($D$)

$$D = (G_{\text{actual}} - G_{\text{expected}}) \times V_{\text{marginal}} \times \text{RPW}$$

Where $V_{\text{marginal}} = 0.030$ wins/game for position players, $G_{\text{expected}} = 155$.

#### Approach Quality Index ($\text{AQI}$) — Novel Component

For each pitch seen, the AQI measures:
$$\Delta RV_i = E[RV | \text{decision}_i] - E[RV | \text{optimal decision}]$$

Aggregated: $\text{AQI} = \sum_i \Delta RV_i$, then Bayesian-shrunk with prior $N(0, 3^2)$.

When pitch-level data is unavailable, AQI is estimated from proxy statistics: $\text{AQI}_{\text{proxy}} = f(BB\%, K\%, O\text{-}Swing\%, Z\text{-}Contact\%)$, scaled to runs per 600 PA.

### 4.2 Aggregation

Total BRAVS runs:
$$\text{Total}_{\text{runs}} = (H + P + R + F + C + \text{Pos} + \text{AQI}) \times \text{Lev}_{\text{mult}} + D + L_{\text{delta}}$$

Conversion to wins:
$$\text{BRAVS} = \text{Total}_{\text{runs}} / \text{RPW}_{\text{dynamic}}$$

### 4.3 Dynamic Runs Per Win

$$\text{RPW} = \frac{2 \times RPG}{\text{exponent}}, \quad \text{exponent} = RPG^{0.287}$$

This Pythagenpat-derived formula makes each run worth more wins in low-scoring environments and fewer in high-scoring environments.

### 4.4 Era and League Adjustments

**Era**: All run values are multiplied by $\text{RPG}_{\text{anchor}} / \text{RPG}_{\text{season}}$ to normalize to the 2023 run environment.

**League**: A ±1 run adjustment for AL/NL differences in the DH-split era (1973–2021).

**Park**: Multi-dimensional park factors adjusting for overall batting environment, HR tendency, and batter handedness.

---

## 5. Data

### Sources

- **Primary**: pybaseball Python package for FanGraphs batting/pitching statistics
- **Statcast**: Available for 2015+ seasons via pybaseball's statcast functions
- **Historical**: Lahman database for pre-Statcast seasons
- **Fallback**: Synthetic data generation mirroring real MLB distributions (see `docs/synthetic-data-methodology.md`)

### Coverage

Our validation dataset comprises 22 notable player-seasons spanning 1913 (Walter Johnson) to 2023 (Shohei Ohtani), selected to represent:
- Consensus all-time greats (Ruth, Mays, Aaron, Bonds, Trout)
- Dominant pitcher seasons across eras (Johnson, Gibson, Koufax, Pedro, deGrom)
- Edge cases: two-way player (Ohtani), elite catcher (Piazza), closer (Rivera), pure DH (Edgar Martinez), glove-first player (Ozzie Smith)
- Controversial cases: Coors Field (Larry Walker), compiler vs. peak (Baines vs. Walker)
- Short season (Soto 2020)

### Data Validation

Every PlayerSeason passes through a validation pipeline checking for:
- Impossible stat lines (H > AB, negative counting stats)
- Suspicious outliers (BA > .420 in 200+ AB, ERA < 1.0 in 100+ IP)
- Missing required fields
- Cross-field consistency (1B + 2B + 3B + HR = H)

---

## 6. Results

### 6.1 All-Time Leaderboard (Notable Seasons)

| Rank | Player | Season | BRAVS | 90% CI | Hitting | Pitching | AQI | Leverage | Durability |
|------|--------|--------|-------|--------|---------|----------|-----|----------|------------|
| 1 | Barry Bonds | 2004 | 29.8 | [27.6, 32.0] | +138.6 | — | +36.0 | +5.5 | -2.3 |
| 2 | Bob Gibson | 1968 | 28.1 | [25.2, 31.0] | — | +128.3 | — | +4.0 | +0.8 |
| 3 | Willie Mays | 1965 | 26.4 | [23.1, 29.6] | +109.2 | — | +11.4 | +3.9 | +0.7 |
| 4 | Sandy Koufax | 1966 | 24.5 | [21.9, 27.0] | — | +113.7 | — | +3.5 | +3.3 |
| 5 | Walter Johnson | 1913 | 23.0 | [21.1, 25.0] | — | +124.3 | — | +3.8 | +1.2 |
| 6 | Babe Ruth | 1927 | 17.4 | [15.6, 19.3] | +102.9 | — | +16.0 | +3.7 | -1.0 |
| 7 | Mike Trout | 2016 | 17.2 | [15.0, 19.5] | +82.4 | — | +10.9 | +2.9 | +1.2 |
| 8 | Shohei Ohtani | 2023 | 16.9 | [14.5, 19.3] | +81.3 | +22.5 | +5.1 | +3.5 | -2.6 |
| 9 | Mike Piazza | 1997 | 16.8 | [14.3, 19.3] | +79.0 | — | +7.6 | +2.7 | -0.9 |
| 10 | Aaron Judge | 2022 | 16.4 | [14.1, 18.7] | +83.7 | — | +7.1 | +3.1 | +0.6 |

### 6.2 Key Findings

**Mariano Rivera's leverage premium**: Rivera 2004 scores 4.9 BRAVS, with +8.6 runs from the leverage component. His raw pitching value is +21.3 runs (modest for 78.7 IP), but the leverage multiplier of 1.40 reflects his gmLI of 1.85. Without leverage adjustment, Rivera would be approximately 2.5 BRAVS — a significant undervaluation for arguably the greatest reliever in history. This demonstrates BRAVS's core philosophical advance: context matters.

**Ohtani 2023 as unified two-way valuation**: Ohtani receives +81.3 hitting runs and +22.5 pitching runs with no special handling. The framework simply sums value created in batting appearances and pitching appearances. The -14.6 positional penalty (DH) is somewhat harsh given that Ohtani also pitched, suggesting future versions should use a weighted positional adjustment for multi-position players.

**Coors Field adjustment**: Larry Walker 1997 (.366/.452/.720 raw) is deflated by the Coors park factor of 1.16, landing at 10.7 BRAVS versus a naive estimate of ~15+. The park adjustment does meaningful work, reducing his hitting value by approximately 15 runs.

**Peak vs. compiler**: Harold Baines 1985 (5.5 BRAVS) versus Larry Walker 1997 (10.7 BRAVS) clearly separates the two archetypes. Baines produced modest value over a full season (+31.1 hitting runs). Walker produced exceptional value despite Coors adjustment. The per-season BRAVS comparison supports Walker's Hall of Fame case.

**Short season penalty**: Juan Soto 2020 (-0.7 BRAVS) is the most clearly implausible result. His hitting value (+24.7 runs in 196 PA) and approach quality (+2.2 runs) are positive, but the durability component (-31.5 runs) for playing only 47 of 155 expected games overwhelms everything else. This is a known flaw in the durability component's handling of the shortened 2020 season.

---

## 7. Discussion

### What BRAVS does better

**Uncertainty quantification** is BRAVS's most important contribution. The difference between "Player A: 7.2 BRAVS" and "Player A: 7.2 BRAVS [5.5, 8.9]" is the difference between false precision and honest assessment. When two players are separated by 0.5 wins but their 90% credible intervals overlap by 4 wins, BRAVS communicates this — WAR does not.

**Leverage context** properly values relievers and situational players. The sqrt(LI) damping finds a principled middle ground between WAR's leverage-blindness and WPA's leverage-obsession.

**Catcher-specific value** fills a 2-3 win gap in WAR. Elite framers like Austin Hedges, Jose Trevino, and Cal Raleigh are meaningfully credited for their pitch-framing skill.

### What BRAVS gets wrong

**Scale inflation** is the primary limitation. BRAVS values are 1.5-2.5× WAR for most players, driven by the AQI component (which partially double-counts hitting value via the proxy model) and the dynamic RPW (which amplifies value in low-scoring eras). This makes BRAVS values unfamiliar and hard to compare to the established WAR scale.

**Historical fielding** is essentially absent. Pre-2000 players receive zero fielding value with wide uncertainty. While this is intellectually honest, it loses information that WAR's imperfect historical metrics at least attempt to capture.

**The AQI proxy model** is the most technically flawed component. When Statcast pitch-level data is unavailable, the proxy model estimates AQI from statistics (BB%, K%, chase rate) that are already partially captured by wOBA, creating partial double-counting.

### Honest limitations

1. Components are assumed independent in the posterior combination. In reality, hitting and baserunning are correlated (faster players get more infield hits AND steal more bases).
2. The conjugate normal posteriors are approximations. For small samples, the true posterior is non-Gaussian.
3. No trajectory or projection component. BRAVS is purely retrospective/descriptive with no forward-looking element.
4. The 2020 season requires special handling that the current implementation does not provide.

---

## 8. Future Work

**Direct AQI computation**: With Statcast pitch-level data (available from 2015), each pitch decision can be directly evaluated by comparing the expected run value of the decision made versus the expected run value of the optimal decision. This would eliminate the proxy model's double-counting issue and provide the independent signal AQI was designed to capture.

**Full MCMC posterior**: The conjugate normal approximation could be replaced with Hamiltonian Monte Carlo (via Stan or PyMC) for components where non-Gaussian posteriors matter. This would improve calibration at the cost of computational speed.

**Historical defensive integration**: TotalZone and other pre-Statcast defensive metrics could be incorporated as noisy observations in the fielding model, with very high observation variance reflecting their known unreliability.

**Projection integration**: BRAVS could be extended with a projection component using multi-year aging curves, connecting the retrospective valuation to forward-looking player value — bridging the gap between what a player did and what they're likely to do.

**Team-level validation**: Validate that total team BRAVS correlates with actual team wins above a FAT-level roster. This would test Axiom 7 (conditional additivity) and help calibrate the overall scale.

---

## 9. References

1. Albert, J. (1992). "A Bayesian Analysis of a Poisson Random Effects Model for Home Run Hitters." *The American Statistician*, 46(4), 246-253.
2. Albert, J. (2006). "Bayesian Computation with R." Springer.
3. Baumer, B., Jensen, S., & Matthews, G. (2015). "openWAR: An Open Source System for Evaluating Overall Player Performance in Major League Baseball." *Journal of Quantitative Analysis in Sports*, 11(2), 69-84.
4. Jensen, S., Shirley, K., & Wyner, A. (2009). "Bayesball: A Bayesian Hierarchical Model for Evaluating Fielding in Major League Baseball." *The Annals of Applied Statistics*, 3(2), 491-520.
5. Lichtman, M. (2003). "UZR (Ultimate Zone Rating)." FanGraphs methodology documentation.
6. Miller, S. (2007). "A Derivation of the Pythagorean Won-Loss Formula in Baseball." *Chance*, 20(1), 40-48.
7. Tango, T., Lichtman, M., & Dolphin, A. (2006). "The Book: Playing the Percentages in Baseball." Potomac Books.
8. Smyth, D. & Patriot (2005). "Pythagenpat: An Improved Pythagorean Theorem." *Baseball Think Factory*.
9. FanGraphs. "WAR Methodology." https://library.fangraphs.com/war/
10. Baseball-Reference. "WAR Explained." https://www.baseball-reference.com/about/war_explained.shtml
