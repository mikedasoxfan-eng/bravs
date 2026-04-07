# Design Decisions Log

Every significant design choice in the BRAVS framework, the alternatives considered, and the rationale for the decision made.

---

## Decision 1: Freely Available Talent (FAT) Baseline

**Choice**: Use Freely Available Talent (FAT) as the value baseline — defined as the expected performance of a player available for league minimum salary on the open market.

**Alternatives Considered**:
- *Replacement level (WAR-style)*: The traditional approach. FanGraphs and Baseball-Reference use slightly different calibrations (~57/43 vs ~55/45 pitcher/position split), creating a ~0.5 win systematic difference between fWAR and bWAR for every player. The definition is circular: replacement level is defined by how much WAR the league should produce, which depends on replacement level.
- *Average (0 WAR = average)*: Simple and interpretable, but loses the counting-stat property. A player who plays 162 games at average quality would be worth 0, indistinguishable from a player who played 0 games.
- *Zero baseline*: Meaningless — a team of zeroes would win 0 games, but that's not a useful reference point.

**Rationale**: FAT is empirically grounded. It can be estimated from actual data: the performance of minor league call-ups, waiver claims, Rule 5 picks, and minimum-salary free agents. It is position-specific (FAT catchers are worse than FAT outfielders) and era-specific (FAT quality has risen as analytics improve talent evaluation). Unlike replacement level, FAT does not require a top-down calibration of total league wins.

**Consequences**: BRAVS values are slightly higher than WAR because FAT is a somewhat lower baseline than traditional replacement level. This is a calibration difference, not a conceptual error. Future work could add a WAR-scale normalization for direct comparability.

---

## Decision 2: Normal-Normal Conjugate Priors (Not MCMC)

**Choice**: Use conjugate normal-normal Bayesian updates for all core components, producing Gaussian posteriors.

**Alternatives Considered**:
- *Full MCMC (Stan/PyMC)*: Would give exact posteriors for non-Gaussian likelihoods (e.g., batting outcomes are multinomial, not normal). More principled for small samples where normality breaks down.
- *Variational inference*: Faster than MCMC but still slower than conjugate updates. Approximate posteriors.

**Rationale**: Computational speed. BRAVS computes a single player-season in ~5ms using conjugate updates. Full MCMC would require ~500ms-5s per player-season, making full-season computation prohibitively slow (hours instead of seconds). For the sample sizes typical in baseball (200+ PA, 100+ IP), the central limit theorem makes the normal approximation excellent. The conjugate approach also makes the math transparent — every posterior has a closed-form expression.

**Consequences**: Posteriors are Gaussian approximations. For small samples (<50 PA), the true posterior for batting rates is Beta-distributed, which is skewed. Our Gaussian approximation misses this skewness. The impact is small because Bayesian shrinkage toward the prior dominates at small sample sizes anyway. Future versions could use a Beta-binomial model for the hitting component to capture this.

---

## Decision 3: Leverage Damping via sqrt(LI)

**Choice**: Use $\text{LI}_{\text{eff}} = \sqrt{\text{gmLI}}$ as the leverage multiplier on skill-based value.

**Alternatives Considered**:
- *No leverage (WAR approach)*: All situations weighted equally. Simple, stable, but philosophically indefensible — a 9th-inning bases-loaded at-bat is not worth the same as a 2nd-inning blowout at-bat.
- *Full leverage (WPA approach)*: Weight by raw leverage index. Context-maximally-sensitive, but extremely noisy (WPA has ~0.2 year-over-year correlation vs ~0.5 for WAR).
- *Log(LI)*: More conservative damping than sqrt. Would reduce leverage effects by ~30% relative to sqrt.
- *Capped LI*: Cap leverage at some maximum (e.g., 3.0). Ad hoc and creates a discontinuity.
- *Tunable exponent*: Treat the damping exponent as a hyperparameter. More flexible but requires a principled way to set it.

**Rationale**: sqrt(LI) is the geometric mean of "ignore leverage" (exponent=0, WAR) and "full leverage" (exponent=1, WPA). This is a principled middle ground. Empirically, it produces sensible results: Rivera 2004 gets +8.6 leverage runs (a meaningful ~40% boost over his raw pitching value), while position players with average leverage (~1.0) get essentially no adjustment. The sqrt function is concave, so it compresses extreme leverage differences, which is what we want — we want leverage to matter but not dominate.

**Consequences**: The leverage component adds ~2-4 runs for average position players (from the normalization) and ~5-10 runs for high-leverage relievers. This is the intended effect. The main risk is that gmLI (average game leverage index) is itself a noisy statistic for small samples, but it is treated as observed (not estimated) in the current implementation.

---

## Decision 4: Fielding Ensemble (UZR + DRS + OAA)

**Choice**: Bayesian model averaging of available defensive metrics with prior weights UZR: 0.30, DRS: 0.30, OAA: 0.40.

**Alternatives Considered**:
- *OAA only*: Best single metric (Statcast-based, more objective), but only available 2016+.
- *UZR only*: Most established, but relies on subjective BIS zone data and has known biases.
- *DRS only*: Similar methodology to UZR but from a different data provider.
- *Simple average*: Unweighted average of available metrics. Ignores reliability differences.
- *No fielding*: Just use the positional adjustment. Loses information.

**Rationale**: Defensive metrics are noisy (year-over-year correlation ~0.4) and frequently disagree with each other by 5+ runs. Using a weighted ensemble reduces the impact of any single system's biases. OAA gets the highest weight because it uses objective Statcast tracking data rather than subjective zone classifications. The Bayesian framework heavily regresses the ensemble toward zero (prior σ = 5 runs), which is appropriate given the large measurement error.

**Consequences**: For modern players with all three metrics, the ensemble is more stable than any individual metric. For historical players with no defensive data, the prior dominates (0 ± 8.2 runs at 90% CI). This is honest but loses the information that, e.g., Ozzie Smith was an elite defender. Future versions should incorporate historical defensive metrics (TotalZone) as a noisy observation.

---

## Decision 5: Approach Quality Index (AQI) as Novel Component

**Choice**: Add a per-pitch decision quality metric (AQI) as a new component that no existing public WAR implementation captures.

**Alternatives Considered**:
- *No novel component*: Simpler, fewer assumptions, directly comparable to WAR. But the requirement specified genuine novelty.
- *Defensive positioning intelligence*: Measuring how well players position themselves using Statcast tracking data. Data-intensive and hard to attribute to individuals.
- *Pitch sequencing value*: Measuring how well pitchers sequence their pitches. Interesting but even noisier than AQI.
- *Count leverage*: Weighting plate appearances by count leverage (full count PA worth more than 0-0 PA). Captures a real effect but is minor.

**Rationale**: AQI fills a genuine gap. Existing metrics measure what happened (hit, walk, strikeout) but not the quality of the decisions that led to the outcome. A batter who swings at a pitch he can't hit hard is making a bad decision even if he gets lucky and singles. A batter who takes a meatball down the middle is leaving value on the table even though "taking a strike" doesn't register as negative in any counting stat. With Statcast data, this can be precisely measured as the run-value differential between the actual decision and the optimal decision.

**Consequences**: The proxy model (for when pitch-level data is unavailable) uses BB%, K%, chase rate, and zone contact rate. These partially overlap with the hitting component (wOBA already rewards walks and penalizes strikeouts). This creates a partial double-counting problem: Bonds 2004 gets +36 AQI runs partly because his astronomical walk rate is already captured in his +138 hitting runs. This is the single biggest calibration issue in BRAVS. Fix: the proxy model should be regressed on wOBA residuals, not raw stats.

---

## Decision 6: Durability as Explicit Component

**Choice**: A separate durability component that credits/penalizes players for games played relative to positional expectations.

**Alternatives Considered**:
- *Implicit via counting stats (WAR approach)*: A player who plays 100 games just gets fewer plate appearances, so their counting-stat total is lower. No explicit availability credit.
- *Rate-stat only*: Report BRAVS as a per-game or per-PA rate. Loses the counting-stat property.
- *Availability probability*: Model the probability that a player is available for each game. More principled but requires injury history data.

**Rationale**: Making durability explicit forces transparency about how much of a player's value comes from being available. A player who produces 4.0 BRAVS in 162 games is more valuable than one who produces 4.0 BRAVS in 100 games — the first player's team didn't need to replace him for 62 games with a FAT-level fill-in.

**Consequences**: The durability component is punitive for shortened seasons. Juan Soto's 2020 (47 games in 60-game season) gets -31.5 durability runs despite elite hitting, because the formula assumes 155 expected games regardless of actual season length. This is a known flaw — the expected games should be prorated for the 2020 season (expected ~55, not 155).

---

## Decision 7: Dynamic Pythagorean RPW

**Choice**: Use RPW = 2 × RPG / exponent, with Pythagenpat exponent = RPG^0.287.

**Alternatives Considered**:
- *Static 10 R/W*: Simple, traditional. Ignores run environment effects.
- *Static 9.5 R/W*: Slightly better calibration for the modern era.
- *Team-specific RPW*: More granular but introduces another layer of estimation noise.

**Rationale**: The Pythagorean win formula is one of the most robust empirical relationships in baseball analytics. Its derivative gives us a principled, context-dependent way to convert runs to wins. In a low-run environment (1968: 3.42 R/G), RPW is ~4.7 — meaning each run is worth about twice as many wins as in a high-run environment (2000: 5.14 R/G, RPW ~6.6). This is correct: when runs are scarce, each marginal run has more impact on the outcome.

**Consequences**: Dead-ball and pitcher's-era seasons (1960s, 1910s) produce dramatically higher BRAVS values than the modern era. Gibson 1968 at 28.1 BRAVS would be ~14 BRAVS with static RPW=10. The dynamic RPW is mathematically correct but creates communication challenges — it's hard to compare a 28 BRAVS season from 1968 to a 17 BRAVS season from 2016 because the scales are different. Future versions could present both raw and era-standardized BRAVS.

---

## Decision 8: Positional Adjustment Scale (Tango)

**Choice**: Use Tom Tango's established positional adjustment spectrum (C: +12.5, SS: +7.5, CF: +2.5, 2B/3B: +2.5, LF/RF: -7.5, 1B: -12.5, DH: -17.5 per 162 games).

**Alternatives Considered**:
- *Derive our own from data*: Fit positional adjustments from offensive production by position. Would need 5+ years of data and might not improve on Tango's peer-reviewed values.
- *No positional adjustment*: Compare players only within position. Loses cross-positional comparability.
- *Smaller adjustments*: Scale down Tango's values by 50% to reduce the penalty on DHs/1B.

**Rationale**: Tango's scale represents decades of research and peer review. It captures a fundamental feature of baseball: teams accept worse hitting from shortstops because the defensive skills are scarce. Custom derivation would require substantial analysis to match this quality. The scale may slightly overweight catcher value (some argue C should be +10, not +12.5), but the difference is small.

**Consequences**: DH players face a -17.5 run penalty per 162 games. This means an elite DH like Edgar Martinez 1995 must overcome a ~15.7 run deficit just to break even on positional value. Combined with the DH's zero fielding contribution, this creates a steep hill for pure DHs. This is by design — it reflects the real opportunity cost of devoting a roster spot to a player who only hits.

---

## Decision 9: Era Adjustment as Multiplicative Factor

**Choice**: Multiply all run values by (anchor_RPG / season_RPG) to normalize across eras.

**Alternatives Considered**:
- *No era adjustment*: Compare raw values. Simple but misleading for cross-era analysis.
- *Additive adjustment*: Add a fixed run offset per season. Doesn't scale with player value.
- *Z-score normalization*: Convert to standard deviations above mean for each season. Loses the run/win interpretability.

**Rationale**: Multiplicative adjustment correctly scales all components proportionally. A player who is 50 runs above average in a 3.5 R/G environment is more impressive than one who is 50 runs above average in a 5.0 R/G environment, and the multiplicative factor captures this. The anchor season (2023, 4.62 R/G) provides a stable reference point.

**Consequences**: Combined with dynamic RPW, the era adjustment compounds for extreme eras. 1968 gets a 1.35× run multiplier AND a lower RPW, roughly doubling the BRAVS value relative to modern seasons. This is the primary driver of the historical inflation issue.

---

## Decision 10: Posterior Mean with 90% Credible Interval

**Choice**: Report the posterior mean as the point estimate, with 50% and 90% credible intervals.

**Alternatives Considered**:
- *Posterior median*: More robust to skewness. But our posteriors are approximately Gaussian (due to conjugate updates), so mean ≈ median.
- *MAP (maximum a posteriori)*: The mode of the posterior. For Gaussian posteriors, MAP = mean = median. No advantage.
- *95% credible interval*: More conservative. But 90% is the conventional standard in baseball analytics and provides better discrimination between players.

**Rationale**: The mean minimizes expected squared error, making it the optimal point estimator under quadratic loss. The 90% CI is the de facto standard in Bayesian sports analytics (matching the convention in openWAR and similar projects). We also report the 50% CI for a tighter "likely range."

**Consequences**: The intervals are symmetric (Gaussian posteriors). For highly skewed cases (e.g., a pitcher with very few innings), the true posterior might be asymmetric, and our symmetric intervals could be miscalibrated. This is a known limitation of the conjugate approach.
