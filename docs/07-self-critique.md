# Self-Critique: Adversarial Analysis of BRAVS

An honest assessment of the BRAVS framework's weaknesses, implausible results, and vulnerability to gaming. Written by the system's designers with the goal of intellectual transparency.

---

## The Five Strongest Objections

### Objection 1: BRAVS Values Are Systematically Inflated Relative to WAR

**The problem**: The notable-seasons leaderboard shows Bonds 2004 at 29.8 BRAVS versus ~12.2 fWAR. Gibson 1968 at 28.1 BRAVS versus ~11.2 bWAR. Every season in our analysis produces BRAVS values roughly 1.5-2.5× the corresponding WAR value.

**Root causes**:
1. **AQI adds 5-36 runs** that WAR does not capture at all. For extreme plate-discipline players (Bonds, Ozzie Smith), AQI contributes 15-36 runs — a component that simply does not exist in WAR.
2. **The FAT baseline** is somewhat more generous than traditional replacement level, adding ~2-5 runs per player.
3. **Dynamic RPW** in low-scoring eras (Gibson's RPW = 4.74 vs WAR's assumed ~10) roughly doubles the win value of each run.
4. **Partial double-counting** between the AQI proxy model and the hitting component (see Objection 2).

**Verdict**: Manageable, but a serious communication problem. BRAVS and WAR are measuring different things on different scales. Direct numerical comparison is misleading. The scale difference is partially by design (BRAVS captures more value channels) and partially an artifact (the AQI double-counting).

**Fix with more time**: (a) Calibrate total league BRAVS to match total team wins above FAT. (b) Offer a "WAR-equivalent" rescaled BRAVS for direct comparison. (c) Fix the AQI proxy model (see below).

---

### Objection 2: The AQI Proxy Model Partially Double-Counts Hitting Value

**The problem**: When Statcast pitch-level data is unavailable, AQI is estimated from BB%, K%, chase rate, and zone contact rate. These statistics are already partially captured by the hitting component:
- High BB% → higher wOBA → more hitting runs AND higher AQI
- Low K% → higher wOBA → more hitting runs AND higher AQI

For Bonds 2004: wOBA = .609 (captured in hitting: +138.6 runs) AND AQI = +36.0 runs driven largely by his 37.6% walk rate, which is already the primary driver of his astronomical wOBA.

**How severe**: The correlation between proxy AQI and hitting runs across our 22 notable seasons is approximately r ≈ 0.65. If the true correlation between a properly measured pitch-level AQI and hitting value is r ≈ 0.2 (reasonable — good decisions and good outcomes are related but distinct), then roughly 45% of the proxy AQI signal is redundant with hitting.

**Verdict**: This is the most serious technical flaw in the current implementation. It inflates total BRAVS by 5-20 runs for extreme players.

**Fix**: The proxy AQI model should be estimated as:
1. Compute wOBA residuals: `wOBA_residual = obs_wOBA - posterior_wOBA_mean`
2. Regress pitch-level AQI on these residuals plus additional plate discipline stats
3. This orthogonalizes AQI against the hitting component, eliminating double-counting

With direct Statcast pitch-level data, this problem disappears entirely — each pitch decision is evaluated independently of the at-bat outcome.

---

### Objection 3: Historical Fielding Is Essentially Unknown

**The problem**: Without UZR, DRS, or OAA data (all post-2000 or post-2016), BRAVS assigns 0 fielding runs with a prior of N(0, 25) — giving a 90% CI of [-8.2, +8.2] runs. This means:
- Willie Mays 1965, one of the greatest defensive center fielders in history: 0 fielding runs
- Ozzie Smith 1987, the Wizard of Oz: 0 fielding runs
- Every pre-2000 player: zero defensive value with massive uncertainty

WAR, by contrast, uses TotalZone (pre-2002) and other historical defensive metrics to estimate fielding value. These are noisy, but they're not zero.

**Verdict**: Known limitation accepted deliberately. We chose to be honest about our uncertainty rather than use highly noisy historical estimates that might be off by 10+ runs. However, this loses real information — we know Ozzie Smith was a great defender even if we can't measure it precisely.

**Fix**: Incorporate TotalZone and other historical defensive metrics as noisy observations in the Bayesian model. Set the observation variance very high (reflecting the large measurement error) so the posterior is pulled only slightly from zero, but at least in the right direction. This would give Ozzie Smith a posterior of perhaps +5 ± 6 runs instead of 0 ± 5 runs.

---

### Objection 4: The Durability Component Is Punitive for Shortened Seasons

**The problem**: Juan Soto 2020 produces a .351/.490/.695 slash line — one of the most dominant 47-game stretches in modern baseball — and BRAVS rates him at -0.7 wins. The cause: the durability component penalizes him -31.5 runs for playing 47 games versus the expected 155 games.

This is technically correct under the framework's logic: the team needed a replacement for 108 games that Soto didn't play. But the 2020 season was only 60 games — Soto wasn't injured, the season was shortened. Penalizing him for not playing games that didn't exist is clearly wrong.

**Verdict**: Design flaw. The durability component should adjust expected games based on actual season length.

**Fix**: Change expected games to `min(EXPECTED_GAMES, actual_season_games × 0.95)`. For 2020: expected = min(155, 60 × 0.95) = 57 games. Soto played 47/57 ≈ 82%, producing a much smaller durability penalty of about -3 runs instead of -31.5.

This fix is straightforward and should be implemented in the next version. The current 2020 results should be disregarded for durability analysis.

---

### Objection 5: Low-Scoring Era Inflation Makes Cross-Era Comparisons Misleading

**The problem**: Bob Gibson 1968 at 28.1 BRAVS and Sandy Koufax 1966 at 24.5 BRAVS appear to dwarf modern greats. The cause is the dynamic RPW in the 1960s pitcher's era:
- Gibson 1968: RPW = 4.74 (runs per game = 3.42)
- Koufax 1966: RPW = 4.93
- Trout 2016: RPW = 5.74
- Modern average: RPW ≈ 5.9

Each run saved by Gibson in 1968 is worth 24% more wins than a run saved by Trout in 2016. This compounds: Gibson threw 304.7 innings (far more than modern pitchers), each inning was worth more runs, and each run was worth more wins.

**Is this correct?** Mathematically, yes. In a 3.42 R/G environment, runs really were scarcer and more impactful. A pitcher who suppressed runs in that environment was genuinely providing more marginal wins to his team.

**Is this useful?** Debatable. It makes 1960s pitcher-seasons look unreasonably dominant, which undermines the metric's ability to facilitate meaningful cross-era debate. A casual reader might conclude Gibson 1968 was 2× as valuable as Trout 2016, which, while defensible on a wins-produced basis, doesn't capture what most people mean by "more valuable."

**Verdict**: Feature, not bug — but a communication problem. The math is right; the presentation needs work.

**Fix**: Present both raw BRAVS (context-dependent) and era-standardized BRAVS (normalized to a common run environment). This lets users choose whether they want "value produced in context" or "skill level compared to peers."

---

## Implausible Results

Three results from our notable-seasons analysis that we find questionable:

### 1. Barry Bonds 2004: 29.8 BRAVS

This is driven by: hitting +138.6 runs (defensible — his .609 wOBA is the highest single-season ever) PLUS AQI +36.0 runs (problematic — the proxy model credits his walk rate twice). Without AQI, Bonds would be ~24 BRAVS, which is closer to reasonable. The AQI double-counting is the primary issue.

### 2. Juan Soto 2020: -0.7 BRAVS

This is clearly wrong. A player who produces a 200 wRC+ should not be negative BRAVS in any framework. The durability penalty of -31.5 runs for a 60-game season is the flaw. With a season-length adjustment, Soto would be approximately +6 to +8 BRAVS, which is reasonable for a dominant 47-game performance.

### 3. Bob Gibson 1968: 28.1 BRAVS

A single-season BRAVS value higher than Mays, Ruth, or any modern player feels off. Gibson 1968 was extraordinary (1.12 ERA, 304.7 IP), but 28.1 wins above FAT implies that replacing Gibson with a FAT-level pitcher would cost the Cardinals 28 wins. That's almost certainly too high. The combination of extreme innings, era-adjusted run values, and low RPW creates a compounding effect that inflates the result.

---

## Gaming the Metric

**Hypothetical player designed to maximize BRAVS**:

A catcher (positional adjustment +12.5) who:
- Hits like peak Bonds (wOBA .600+, AQI proxy goes through the roof: ~40 runs)
- Walks 35% of the time (inflates both hitting and AQI via proxy)
- Frames like Austin Hedges (+15 framing runs)
- Steals 40 bases (+7 baserunning runs)
- Plays 162 games in the 1968 run environment (RPW ≈ 4.7)
- Pitches 200 innings with a 1.50 FIP in his spare time (two-way)

This hypothetical player could theoretically score 60+ BRAVS. The most gameable components are:
1. **AQI proxy** — extreme walk rates produce extreme AQI values that double-count
2. **Era RPW** — playing in a low-scoring era multiplies everything by ~1.5-2×
3. **Positional stacking** — a catcher who can hit eliminates the defensive spectrum penalty AND gets the +12.5 positional bonus

**How resistant is BRAVS?** Not very resistant to this specific gaming scenario. But such a player has never existed and almost certainly never will. The real-world constraint is that catcher defense, elite hitting, and elite baserunning are almost never found in the same player — the physical demands of catching preclude elite speed, and the position's mental demands reduce at-bat quality.

**The AQI proxy is the most gameable** component because it responds nonlinearly to extreme walk rates. With direct pitch-level AQI computation, this gaming vector would be closed.

---

## Honest Comparative Assessment

### Where BRAVS beats WAR:
- **Uncertainty quantification**: BRAVS provides credible intervals; WAR provides false precision
- **Catcher valuation**: BRAVS includes framing, blocking, throwing, game-calling; WAR ignores them
- **Leverage context**: BRAVS credits closers appropriately; WAR systematically undervalues relievers
- **Approach quality**: AQI captures information WAR misses (even with the proxy model's flaws)
- **Philosophical coherence**: BRAVS's axioms are explicit and defensible; WAR's assumptions are implicit and inconsistent between implementations

### Where WAR beats BRAVS:
- **Calibration**: WAR's scale is well-understood and anchored to real team wins; BRAVS's scale is inflated and unfamiliar
- **Simplicity**: WAR is a single number; BRAVS demands distributions and credible intervals
- **Historical fielding**: WAR uses imperfect but non-zero historical defensive metrics; BRAVS defaults to zero
- **Community adoption**: WAR is the lingua franca of baseball analytics; BRAVS is unknown
- **Communication**: "He was worth 8 WAR" is instantly understood; "He was worth 17.2 BRAVS [15.0, 19.5]" requires explanation

### Where they're comparable:
- Pitcher-hitter comparison methodology
- Park factor adjustments (both use similar approaches)
- Baserunning measurement
- Offensive valuation core (both use linear weights from run expectancy)

---

## Recommendations for Version 2.0

1. **Fix the AQI proxy model** to orthogonalize against hitting value
2. **Add season-length adjustment** to durability component
3. **Incorporate historical defensive metrics** as noisy observations
4. **Add a WAR-equivalent scale** for comparability
5. **Implement full MCMC** as an optional mode for when precision matters more than speed
6. **Build a projection component** using multi-year aging curves
7. **Validate at scale** against team wins, MVP voting, and HOF selection
