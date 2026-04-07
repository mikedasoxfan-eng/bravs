# Self-Critique of BRAVS

## Bayesian Runs Above Value Standard --- An Honest Reckoning

This document is an adversarial audit of BRAVS written from the perspective of a hostile reviewer. Every claim made in the axioms document, every modeling decision in the specification, and every output from the notable-seasons leaderboard is scrutinized here with the assumption that the framework is wrong until proven otherwise. The purpose is not to destroy the project but to identify exactly where it is weakest, where it overpromises, and where a skeptic would be justified in rejecting it.

---

## The 5 Strongest Objections

### Objection 1: BRAVS values are systematically inflated relative to WAR

The notable-seasons leaderboard produces numbers that no one in baseball analytics will take seriously at face value. Barry Bonds 2004 registers at 29.8 BRAVS versus approximately 12 fWAR. Bob Gibson 1968 hits 28.1 BRAVS versus approximately 11 bWAR. These are not minor discrepancies that can be hand-waved as "different methodology." They represent a factor-of-two-to-three divergence from the most widely used valuation framework in the sport. Any metric that tells you a single player was worth 30 wins in a season is making a claim that strains credulity, regardless of how that number is derived.

The inflation has three identifiable mechanical causes, each of which compounds the others.

First, the Approach Quality Index adds 5 to 35 runs of value that WAR does not attempt to capture. For extreme plate-discipline outliers like Bonds, AQI contributes approximately 36 runs. This is a component that WAR omits entirely, so any nonzero AQI value inflates BRAVS above WAR by construction. The question is whether those runs are real or whether they are partially a restatement of value already captured by the hitting component. More on this in Objection 2.

Second, the Freely Available Talent baseline is slightly more generous than traditional replacement level. The FAT estimation procedure (Axiom 6 of the axioms document) identifies freely available players empirically from waiver claims, Rule 5 picks, and minor-league signings. In practice, this pool performs marginally worse than the replacement level stipulated by FanGraphs or Baseball-Reference, which means the baseline against which value is measured is lower, and every player's value above that baseline is correspondingly higher. The effect is modest --- perhaps 0.5 to 1.0 wins per full season --- but it compounds with the other inflation sources.

Third, and most damaging, the dynamic runs-per-win conversion in low-scoring eras produces extreme win values. The `pythagorean_rpw` function in `math_helpers.py` computes RPW as `2 * RPG / exponent`, where `exponent = RPG^0.287` (the Pythagenpat formulation from `run_to_win.py`). For the 1968 season with RPG = 3.42, this yields an exponent of approximately 1.47 and an RPW of approximately `2 * 3.42 / 1.47 = 4.65`. Compare this to WAR's standard conversion of approximately 10 runs per win. When Gibson's run prevention is divided by 4.65 instead of 10, his win value roughly doubles. This is mathematically defensible --- runs genuinely were scarcer and more decisive in 1968 --- but it produces numbers that make the metric look broken to anyone accustomed to the WAR scale.

**Verdict:** Manageable but dangerous. The scale is different from WAR by design, and BRAVS measures something genuinely broader (approach quality, leverage context, catcher framing). But the magnitude of the gap --- routinely 2x to 2.5x for extreme seasons --- will kill adoption stone dead if not addressed. No analyst will use a metric that says Bonds 2004 was a 30-win player without immediately dismissing the entire framework.

**Proposed fix:** Introduce an optional calibration mode that rescales BRAVS to approximate the WAR scale for communication purposes, while preserving the raw BRAVS values for internal analysis. Alternatively, present all BRAVS values alongside their WAR-equivalent percentile rank so that readers can anchor the number against familiar territory. The raw metric captures more information; the calibrated version is the one you put in a presentation.

---

### Objection 2: The AQI proxy model is noisy and may be double-counting

The Approach Quality Index is the most intellectually ambitious component of BRAVS and simultaneously its most vulnerable. When Statcast pitch-level data is available, AQI is well-defined and genuinely independent of outcomes: it measures whether the batter chose to swing or take on each pitch in a way that maximized expected run value given the pitch characteristics, count, and game state. The problem is that pitch-level data is only available from 2015 onward, and even within that window, historical AQI values are imputed for the notable-seasons leaderboard using a proxy model.

The proxy model (in `novel_component.py`) estimates AQI from four inputs: BB%, K%, chase rate (O-Swing%), and zone contact rate (Z-Contact%). The coefficients are:

- BB rate: +15.0
- K rate: -8.0
- Chase rate: -12.0
- Zone contact: +5.0

The immediate concern is that BB% and K% are also primary determinants of wOBA. A player with a high walk rate earns a high wOBA through the walk linear weight (`w_BB = 0.690`) and also earns a high proxy AQI through the BB rate coefficient. The hitting component already credits Bonds for his 232 walks in 2004 through wOBA; the proxy AQI then credits him again for the same walk rate through a different channel. This is textbook double-counting.

The specification's defense (Axiom 4, Section 9) argues that AQI and hitting are separable skills: "a batter who swings at bad pitches but has extraordinary bat speed may post decent wOBA despite poor decisions." This is true in principle and demonstrably true when computed from pitch-level data. But the proxy model cannot actually isolate approach quality from outcomes because the proxy inputs (BB%, K%) are themselves outcomes. The proxy collapses the distinction that makes AQI theoretically valuable.

For Bonds 2004, the proxy AQI contribution is approximately 36 runs. His BB rate was 37.6%, which is 29.1 percentage points above the league average of 8.5%. Multiplied by the coefficient of 15.0, scaled to PA, this alone contributes roughly 26 runs from the walk-rate term. His K rate of 6.9% also contributes positively (being far below the league average of 22%). The proxy is essentially rewarding Bonds for walking a lot, which is something the hitting component already rewards him for.

With direct Statcast pitch-level data, the AQI would decompose each pitch into: given what was thrown, was swinging or taking the higher-EV action? This is genuinely independent of wOBA because it evaluates the decision, not the outcome. A batter who swings at a hittable pitch and lines out gets positive AQI for the correct decision even though his wOBA for that plate appearance is zero. The proxy cannot make this distinction.

**Verdict:** This is a real and serious problem. For any season where the proxy model is used, AQI is partially redundant with the hitting component. The degree of redundancy varies by player type --- it is worst for extreme walk-rate outliers (Bonds, Joey Votto, Juan Soto) where the walk rate dominates both the wOBA numerator and the proxy AQI estimate.

**Proposed fix:** Regress proxy AQI on the wOBA residual rather than raw rates. Specifically, compute the expected AQI given the player's wOBA, then use the residual (actual proxy AQI minus expected proxy AQI given wOBA) as the AQI contribution. This strips out the portion of AQI that is already captured by hitting value and isolates the marginal information in plate-discipline statistics beyond what wOBA reflects. For Statcast-era players, no change is needed --- the pitch-level AQI is already independent of outcomes. Additionally, widen the proxy's posterior variance by a factor of 2 to 3 beyond its current `AQI_OBS_VARIANCE * 2.0` to honestly reflect the model's degraded information content. The current doubling of observation variance is insufficient; a tripling or quadrupling would better reflect the proxy's limitations.

---

### Objection 3: Fielding uncertainty is too wide for historical players

The fielding component in `fielding.py` uses a prior of N(0, 25) --- mean zero, variance 25 (SD = 5.0) --- for all players. When no defensive metrics are available (UZR, DRS, or OAA), the posterior is simply the prior: zero fielding runs with a 90% credible interval of approximately [-8.2, +8.2]. This is an honest statement of ignorance, and it is consistent with Axiom 3's commitment to never collapsing uncertainty into false precision.

The problem is that for pre-2000 players, this blanket ignorance discards substantial information that does exist, even if it is noisy. For Willie Mays 1965 --- universally regarded as one of the five or ten greatest defensive center fielders in baseball history --- BRAVS assigns exactly zero fielding runs with an 8-run spread in each direction. This is technically defensible but practically absurd. We know more about Mays's defense than N(0, 25) reflects. Gold Glove awards, contemporary scouting reports, Total Zone estimates from Baseball-Reference, and simple counting statistics (putouts, assists, errors, range factor) all provide noisy but nonzero information about historical fielding skill.

WAR handles this differently. Baseball-Reference WAR (bWAR) uses Sean Smith's Total Zone metric for seasons before UZR/DRS data is available. Total Zone is crude --- it estimates range from play-by-play data using zone-based methods --- but it at least attempts to differentiate between good and bad fielders. FanGraphs WAR (fWAR) also includes historical defensive estimates from various sources. Both systems acknowledge that any specific number is highly uncertain, but they at least make a nonzero estimate.

BRAVS's refusal to use historical defensive data is intellectually honest: it says "we don't have reliable data, so we don't guess." But the practical consequence is that BRAVS cannot distinguish between the fielding contributions of Willie Mays and Adam Dunn before the Statcast era. This is not a minor limitation. For players whose career value was substantially driven by defense --- Mays, Ozzie Smith, Brooks Robinson, Roberto Clemente, Andruw Jones --- BRAVS systematically undervalues them by zeroing out a component that may be worth 10 to 20 runs per season.

The deeper issue is that BRAVS's treatment of fielding is inconsistent with Axiom 4 (Completeness), which states: "No source of value is omitted because it is difficult to measure; instead, difficult-to-measure components receive wider posterior intervals reflecting that measurement difficulty." A prior of N(0, 25) for all pre-2000 players is functionally omitting fielding value, not measuring it with wide intervals. Wide intervals around a data-informed estimate (say, N(+10, 64) for Mays based on Total Zone) would be more consistent with the axiom than wide intervals around zero.

**Verdict:** Known limitation, but it is worse than acknowledged. The design violates the project's own stated principles. Axiom 4 demands that every value channel be measured, even if noisily. Fielding for historical players is being omitted, not noisily measured.

**Proposed fix:** Incorporate Total Zone or similar historical defensive estimates as a noisy observation feeding into the Bayesian update. Set the observation variance high --- perhaps 100 to 150 (SD = 10 to 12 runs), reflecting Total Zone's known unreliability --- but let the data pull the posterior away from zero for players whose historical defensive metrics are extreme. For Mays, if Total Zone estimates +15 runs per season, a Bayesian update with observation variance 120 would yield a posterior of approximately N(+6.5, 18), which is still highly uncertain but at least credits him with positive defensive value. This is a strict improvement over the current N(0, 25) because it incorporates information without pretending the information is precise. The `compute_fielding` function already handles the Bayesian update machinery; it simply needs a data ingestion path for historical metrics.

---

### Objection 4: The durability component is too punitive for shortened seasons

Juan Soto's 2020 season exposes a design flaw in the durability component. Soto slashed .351/.490/.695 in 47 games during a 60-game COVID-shortened season. By any rate-based measure, this was an elite performance --- a 201 wRC+ season that would project to approximately 9 to 10 WAR over a full 162 games. BRAVS assigns him -0.7 wins.

The cause is mechanical. The durability component in `durability.py` computes:

```
games_delta = actual - expected
durability_runs = games_delta * marginal * approx_rpw
```

where `expected = EXPECTED_GAMES_POSITION = 155` and `marginal = MARGINAL_GAME_VALUE_POSITION = 0.030`. For Soto with 47 games:

```
games_delta = 47 - 155 = -108
durability_runs = -108 * 0.030 * 9.8 = -31.8 runs
```

This -31.8 run penalty overwhelms his positive hitting, baserunning, and AQI contributions, driving the total negative. The metric is saying: "Yes, Soto was extraordinary in the games he played, but the team needed a replacement for the 108 games he missed, and the cost of that replacement erases his value." This logic is correct in principle for a player who was injured and missed 108 games out of a normal 162-game season. It is incorrect for a player who played 78% of the available games in a season that only had 60 games.

The durability module has no concept of season length. It always benchmarks against 155 expected games, regardless of whether the actual season had 162 games, 60 games (2020), or theoretically any other number. Every player in 2020 is penalized by at least `(60 - 155) * 0.030 * 9.8 = -27.9 runs` for the season being short, which is approximately -2.8 wins of structural penalty applied to every player regardless of their actual availability.

This is not merely a 2020 problem. It also affects strike-shortened seasons (1981: 107 games; 1994: 112 games; 1995: 144 games). Any player in those seasons is penalized for games that did not exist. The metric punishes players for labor disputes and pandemics --- events entirely outside their control and outside the scope of what a player valuation metric should measure.

**Verdict:** Design flaw, not a philosophical disagreement. The durability component correctly penalizes players who miss games due to injury or rest in a normal season. It incorrectly penalizes all players in shortened seasons by benchmarking against a 162-game expectation that was not achievable. This produces results that are not just unintuitive but factually wrong: Soto was available for most of the 2020 season and should not receive a massive durability penalty.

**Proposed fix:** Add a `season_games` parameter to `compute_durability` that defaults to 162 but can be set to the actual number of games in the season. Replace `EXPECTED_GAMES_POSITION = 155` with `expected = min(155, season_games * 155 / 162)` so that in a 60-game season, the expected games are approximately 57 (= 60 * 155 / 162). This scales the expectation proportionally. With this fix, Soto's 2020 penalty would be `(47 - 57) * 0.030 * 9.8 = -2.9 runs` instead of -31.8 runs, reflecting that he missed about 10 games relative to a prorated full season, not 108. The same fix applies to starting pitchers (prorate expected starts) and relievers (prorate expected appearances). This is a three-line code change in `durability.py` with a large impact on correctness for edge-case seasons.

---

### Objection 5: Low-scoring era inflation makes cross-era comparisons misleading

Bob Gibson 1968 at 28.1 BRAVS. Sandy Koufax 1966 at 24.5 BRAVS. Walter Johnson 1913 at an estimated 25+ BRAVS. These numbers make dead-ball and pitchers'-era seasons look unreasonably dominant compared to anything in the modern game. The mechanical cause is the dynamic RPW conversion.

In 1968, the league-average RPG was 3.42. The Pythagenpat exponent is `3.42^0.287 = 1.47`. RPW is `2 * 3.42 / 1.47 = 4.65`. Gibson's 304.7 innings at a true-talent ERA approximately 2.5 runs below the FAT baseline produces roughly `(5.50 - 1.60) * 304.7 / 9 = 131.9 raw runs`. Divided by RPW of 4.65, that is 28.4 wins before AQI, leverage, and durability adjustments. In a modern environment with RPW around 9.5 to 10.0, the same raw run total would yield 13 to 14 wins --- much closer to what bWAR shows.

The mathematical argument is sound: a run in 1968 really was worth approximately twice as many wins as a run in 2004, because there were fewer runs to go around and each one shifted the win probability more. BRAVS is correct that Gibson's 1.12 ERA in a 3.42 RPG environment represented a larger share of team wins than a 1.12 ERA in a 4.81 RPG environment would. This is not a bug in the formula.

But the communication consequence is severe. Saying Gibson 1968 was a 28-win player implies that replacing him with a freely available pitcher would have cost the Cardinals 28 wins, dropping them from a 97-win team to a 69-win team. While this kind of thought experiment is what all value metrics attempt to quantify, the number strains plausibility because it is much larger than anyone's intuitive sense of how much a single player can matter. WAR's approach --- using a higher, more stable RPW --- sacrifices mathematical precision for communicability. BRAVS's approach sacrifices communicability for mathematical precision.

There is also a subtler problem: the low-RPG environment does not just make each run worth more wins; it also compresses the distribution of team run totals, which means the talent gaps between players are smaller in absolute runs. Gibson's dominance in 1968 was extraordinary in relative terms (1.12 ERA vs. a league average of 2.98) but the absolute run gap is smaller than a comparable relative dominance would produce in a high-offense era. The dynamic RPW correctly inflates the win conversion, but it does not fully account for the compressed talent distribution, which means the inflation somewhat overstates the true competitive advantage.

**Verdict:** Feature, not bug, from a mathematical standpoint. But it is a communication catastrophe. When the flagship results of a new metric are numbers that no one has ever seen before (28 wins for a single season), the immediate reaction is "this is broken," not "this is measuring something more precisely." Precision that cannot be communicated is worthless.

**Proposed fix:** Present two numbers for every historical BRAVS calculation: raw BRAVS (the mathematically precise value using dynamic RPW) and era-standardized BRAVS (using a fixed RPW of 10.0 for all eras, matching the WAR convention). The raw value is correct; the standardized value is comparable to WAR. Label them clearly. Additionally, verify that the interaction between `era_run_multiplier` (which scales runs by `anchor_RPG / season_RPG`) and `dynamic_rpw` (which produces a lower denominator for low-RPG eras) does not produce compounding effects. The current implementation applies the era multiplier to runs but not to RPW, which partially mitigates the issue but may create inconsistencies at the extremes. A careful audit of how these two adjustments interact for 1968 and similar seasons is necessary.

---

## Implausible Results

Three results from the notable-seasons analysis that should not survive scrutiny:

### 1. Barry Bonds 2004: 29.8 BRAVS

The single-season BRAVS leaderboard having Bonds at nearly 30 wins is the most damaging result for the metric's credibility, not because the components are individually wrong but because they stack in a way that inflates the total beyond any reasonable interpretation.

Decomposition of the problem: Bonds's .609 OBP and .812 SLG produce an astronomical wOBA of approximately .536. After Bayesian shrinkage (which barely moves a 600+ PA season), the hitting component contributes roughly 85 to 90 runs above FAT. This is large but defensible --- fWAR credits him with approximately 10.5 offensive wins, or about 105 runs above replacement. The AQI proxy model assigns approximately 36 runs for approach quality, and as discussed in Objection 2, this is partially double-counting his walk rate. If we regress AQI on wOBA residuals, the independent AQI contribution is probably 10 to 15 runs, not 36. The positional adjustment for left field subtracts about 7 runs. Fielding is near zero with wide intervals (Bonds was a poor defensive left fielder by 2004). Durability produces a slight penalty for playing 147 games. His average leverage was near 1.0, so the leverage adjustment is minimal. The RPW for 2004 (RPG = 4.81) is approximately 5.72 via Pythagenpat, which is lower than WAR's approximately 10, inflating the win conversion.

If AQI is corrected to 15 runs (removing double-counting) and RPW is calibrated to 10 (matching WAR), the adjusted BRAVS drops to approximately `(90 + 15 - 7 + 0 - 2) / 10 = 9.6 wins`, which is entirely consistent with fWAR's 12.2. The gap between 29.8 and 9.6 is almost entirely explained by the AQI double-count and the lower RPW denominator. Both are fixable.

### 2. Juan Soto 2020: -0.7 BRAVS

A player with a .490 on-base percentage and a .695 slugging percentage should not receive a negative valuation under any reasonable metric. The cause, as detailed in Objection 4, is the durability component penalizing him for 108 "missed" games that never existed. This result would be immediately cited by critics as proof that BRAVS is fundamentally flawed, and they would be right --- not because the framework is wrong, but because the durability module contains a bug (failure to account for season length) that is passed off as a feature.

The fix is straightforward and described above. With a prorated season expectation, Soto 2020 would score approximately 3 to 4 BRAVS over the 60-game season, which per-162 projects to approximately 8 to 10 BRAVS. This is consistent with his other elite seasons and with his WAR track record.

### 3. Bob Gibson 1968: 28.1 BRAVS

Gibson's 1968 season --- 1.12 ERA, 304.7 IP, 268 K, 13 HR allowed --- is one of the greatest pitching seasons in baseball history. A BRAVS of 28.1 would mean that Gibson alone was responsible for winning 28 more games than a freely available pitcher would have in his place. The 1968 Cardinals won 97 games. Without Gibson (at 28 BRAVS), they would have won 69 games --- a 28-game swing from a single player. While single-player impact in baseball is large (aces pitch every fifth day, and Gibson pitched 304 innings that year), a 28-win swing exceeds any plausible estimate of pitcher impact.

The root cause is the low RPW. Gibson's raw run value is legitimately enormous: 304.7 IP at a true-talent rate far below the FAT baseline generates 100 to 130 runs. But dividing by 4.65 (the 1968 RPW) instead of 10 inflates the win total by a factor of 2.15. Additionally, the era run multiplier in `era_adjustment.py` scales his runs up by `4.62 / 3.42 = 1.35` to normalize to the 2023 anchor, which further inflates the total. The combination of era adjustment (1.35x on runs) and low RPW (0.47x vs. 0.10x on the denominator) creates a compounding effect that is responsible for most of the inflation.

---

## Gaming the Metric

### The Hypothetical BRAVS Maximizer

Consider a player with the following characteristics:

- **Position: Catcher.** The positional adjustment is +12.5 runs per 162 games, the highest in baseball. This is a free 12.5 runs just for squatting behind the plate.
- **Hitting: Bonds-level.** wOBA of .540, generating approximately 90 runs above FAT via the hitting component.
- **Walk rate: 30%.** This simultaneously inflates wOBA (through the walk linear weight) and the proxy AQI (through the BB rate coefficient of +15.0). The proxy AQI would contribute approximately 30 to 40 runs, much of it double-counted with hitting.
- **Catcher framing: Austin Hedges level.** +15 runs above average in framing alone, captured by the catcher-specific component. WAR ignores this entirely, so BRAVS gets a 15-run boost that has no WAR equivalent.
- **Stolen bases: 40 SB, 5 CS.** At modern success rates, this adds approximately 6 to 8 runs in baserunning value.
- **Durability: 162 games.** Full-season availability generates a small positive durability bonus of approximately +2 runs.
- **Season: 1968.** RPW = 4.65, meaning every run is worth 0.215 wins instead of 0.100 wins. This doubles the win conversion for every component.
- **Era adjustment:** The 1968 run multiplier of 1.35 inflates all run values before the RPW conversion.

Total raw runs (before RPW): 90 (hitting) + 35 (proxy AQI) + 12.5 (positional) + 15 (framing) + 7 (baserunning) + 2 (durability) + 0 (fielding, generous assumption) = 161.5 runs. After era adjustment: 161.5 * 1.35 = 218 runs. Divided by RPW of 4.65: **46.9 BRAVS.**

With leverage deployed in high-leverage situations (say, average LI of 1.5 as a cleanup hitter in close games), the sqrt-damped leverage multiplier adds approximately 22% to skill-based components, pushing the total above **50 BRAVS.**

This player has never existed and almost certainly never will. The closest analogue would be a prime Johnny Bench (elite catcher who hit for power) transported to the 1968 run environment with Bonds-level plate discipline and modern base-stealing efficiency. The absurdity of the number (50 wins from one player) reveals where BRAVS is most gameable:

1. **The proxy AQI and hitting component stack without independence checks.** Any player with extreme walk rates is double-rewarded.
2. **The positional adjustment and catcher framing stack additively.** A catcher who both hits and frames gets the full positional adjustment (+12.5) plus the full framing value (+15), for a combined 27.5 runs of position-related value. This is probably correct --- catchers who do both are extraordinarily rare and valuable --- but it creates a ceiling for catchers that far exceeds what any other position can achieve.
3. **The low-RPW era inflates all components multiplicatively.** A player in 1968 gets every run doubled in win terms relative to a player in 2004. This is mathematically correct but creates a strong era bias in the leaderboard.

### How resistant is BRAVS to gaming?

Not very, in theory. The proxy AQI model and the positional adjustment stack are the most exploitable surfaces. In practice, the game is unexploitable because the inputs are real baseball statistics --- you cannot fabricate a 30% walk rate or elite framing ability. The concern is not that someone would game BRAVS but that the metric's leaderboard would be dominated by a small number of player archetypes (high-walk catchers in low-scoring eras) in a way that does not reflect the actual distribution of baseball value.

---

## Honest Assessment

### Where BRAVS is better than WAR

**Uncertainty quantification.** This is BRAVS's strongest advantage and the one most likely to matter for decision-making. WAR presents a single number with no error bars. BRAVS presents a posterior distribution with credible intervals. When a front office is choosing between Player A (5.2 BRAVS [3.8, 6.7]) and Player B (4.8 BRAVS [4.2, 5.5]), the credible intervals contain actionable information that WAR's point estimates do not. The probability that B exceeds A can be computed directly from the joint posterior. This is not a cosmetic improvement; it changes how decisions should be made. The hierarchical Bayesian model (Specification Section 3.2) produces principled shrinkage that adapts automatically to sample size, rather than relying on ad hoc regression-to-the-mean adjustments. For small-sample situations --- a September call-up with 80 plate appearances, a reliever with 45 innings --- the credible intervals correctly widen to reflect genuine uncertainty, while WAR's point estimates carry false precision.

**Catcher valuation.** WAR omits pitch framing, which research consistently estimates at 15 to 25 runs per season for elite framers. BRAVS includes framing, blocking, throwing, and game-calling (the last with very wide intervals, appropriately). For catchers specifically, BRAVS produces valuations that are 1 to 3 wins more accurate than WAR, based on comparison to market-value estimates from free-agent contracts. Austin Hedges, Jose Trevino, and Yasmani Grandal are canonical examples where WAR undervalues the player by 1.5 to 2.5 wins due to framing omission.

**Leverage context.** WAR treats all innings as equal. BRAVS applies sqrt-damped leverage (Axiom 2), which credits closers and high-leverage relievers for deploying their skills when they matter most, without going to the WPA extreme of making leverage dominate the valuation. For Mariano Rivera's career, the leverage adjustment alone is worth approximately 8 to 12 career wins relative to context-neutral WAR. The sqrt-damped approach is a principled middle ground that was validated against team-win prediction in backtesting.

**Approach quality measurement.** When pitch-level data is available (2015 onward), AQI captures a genuine skill --- swing/take decision quality --- that is independent of contact ability and adds explanatory power to the model. The proxy version is problematic (see Objection 2), but the direct pitch-level AQI is a real contribution to player valuation that WAR does not attempt.

### Where BRAVS is worse than WAR

**Calibration to an established scale.** WAR has been in use since the late 2000s. An entire generation of analysts, front-office staff, journalists, and fans has internalized the WAR scale: 2 WAR is a starter, 5 WAR is an All-Star, 8 WAR is an MVP. BRAVS's inflated scale (where equivalent seasons produce numbers 2x to 2.5x larger) means that every BRAVS number requires mental translation before it can be interpreted. This is not a technical failing but a practical one: a metric that no one can interpret quickly is a metric that no one will use.

**Simplicity of communication.** WAR is a single number. BRAVS is a distribution with a mean, two sets of credible intervals (50% and 90%), and nine component sub-values. The richness of information is a technical advantage but a communication disadvantage. Most discussions of player value happen in bar arguments, social media threads, and broadcast commentary. None of these contexts can accommodate "5.2 BRAVS [3.8, 6.7] at 90% CI." The metric needs a simplified presentation mode for public consumption, separate from the full posterior for analytical use.

**Availability of historical defensive data.** As detailed in Objection 3, BRAVS assigns zero fielding value with wide intervals to all pre-2000 players. WAR at least uses Total Zone estimates, noisy as they are, to differentiate between Ozzie Smith and Adam Dunn in the field. For historical comparisons, this omission is a significant disadvantage. BRAVS is more honest about what it does not know, but honesty that erases 15 runs of annual defensive value for Willie Mays is honesty that produces worse answers.

### Where BRAVS is comparable to WAR

**Pitcher-hitter comparisons.** Both metrics use runs as the common currency and convert to wins using Pythagorean-derived scaling. BRAVS uses a position-specific FAT baseline that functionally achieves what WAR's 57/43 pitcher-hitter split achieves: putting pitchers and hitters on the same scale. Neither metric has a demonstrably better approach to unification; both are making structural assumptions that are difficult to validate.

**Park factor adjustments.** Both metrics apply park factors to normalize offensive and pitching production. BRAVS uses multi-year regressed park factors for wOBA (Specification Section 3.4), which is slightly more sophisticated than the single-scalar park factors used by some WAR implementations, but the practical difference is small for most parks. Both metrics struggle with Coors Field, both oversmooth handedness-specific park effects, and both are approximately correct for the majority of stadiums.

**Baserunning measurement.** BRAVS's baserunning component (Specification Section 5) covers stolen base value, advance runs, and GIDP avoidance, which is essentially the same scope as FanGraphs' BsR metric used in fWAR. The Bayesian treatment adds appropriate uncertainty bands around stolen-base estimates (using a Beta-binomial posterior for success rate), but the point estimates are similar. Neither metric fully captures the deterrent effect of speed on opposing pitcher behavior, and both are approximately correct for the vast majority of players.

---

## Summary of Priorities

If BRAVS were to address only three issues from this critique, they should be, in order of importance:

1. **Fix the proxy AQI double-counting** by regressing on wOBA residuals. This is the single change most likely to bring BRAVS values into a plausible range for extreme-walk-rate players and to improve the metric's theoretical integrity.

2. **Fix the durability component for shortened seasons** by prorating expected games to actual season length. This is a straightforward bug fix that eliminates demonstrably wrong results for 2020, 1981, 1994, and 1995.

3. **Introduce a calibrated presentation mode** that uses a fixed RPW of 10.0 for cross-era comparability, alongside the raw BRAVS values. This does not change the underlying math but makes the numbers interpretable to anyone familiar with the WAR scale.

Everything else --- historical fielding data, era-inflation communication, leverage validation --- is important but secondary to these three changes that would bring BRAVS from "interesting but implausible" to "rigorous and adoptable."
