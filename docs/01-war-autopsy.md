# WAR Autopsy: A Technical Examination of Wins Above Replacement

## Preface: What WAR Gets Right

Before dissecting its failures, WAR deserves credit. It was the first widely adopted framework to collapse the full breadth of player contribution -- hitting, fielding, baserunning, positional difficulty -- into a single number denominated in wins. For front offices drowning in disconnected counting stats, WAR provided a common currency. It correctly identifies that a shortstop hitting .270 is more valuable than a first baseman hitting .270. It correctly penalizes players who accumulate stats in high-run environments. It correctly frames value against a replacement-level baseline rather than an average one, which better reflects actual roster decisions. For rough comparisons -- separating a 7 WAR season from a 2 WAR season -- it works. The problems begin when we treat it as precise, when we compare players separated by a win or less, when we use it across eras, and when we forget the structural assumptions buried inside it.

This document is a section-by-section audit of those assumptions, aimed at identifying what a successor metric must fix.

---

## 1. Defensive Metrics: The Shakiest Foundation

WAR's most fragile component is its defensive valuation. FanGraphs WAR (fWAR) uses Ultimate Zone Rating (UZR); Baseball-Reference WAR (bWAR) uses Defensive Runs Saved (DRS); Statcast has introduced Outs Above Average (OAA). None of them agree with each other reliably, and none of them agree with themselves year over year.

**Sample size instability.** Defensive metrics require roughly three full seasons of data to stabilize, per research by Mitchell Lichtman (the creator of UZR) and others. The year-over-year correlation for UZR hovers around 0.4 -- compare that to wOBA's year-over-year correlation north of 0.7. This means that in any single season, defensive metrics carry enormous noise. A player rated at +15 runs in one year might genuinely be anywhere from +5 to +25. That is a two-win spread from defense alone.

**Metric disagreement.** Andrelton Simmons is the canonical example. In his peak years with the Angels (2017-2019), Simmons was widely regarded as the best defensive shortstop in baseball. Yet in 2019, DRS credited him with +4 runs while UZR gave him +1.4. In 2017, DRS had him at +18 while UZR put him at +7.8. These are not minor discrepancies. They propagate directly into WAR, meaning fWAR and bWAR can disagree on Simmons by more than a full win in a single season based purely on which defensive system they trust. OAA, Statcast's ball-tracking metric, introduced yet another perspective, sometimes splitting the difference and sometimes diverging further.

**Park factor contamination.** Defensive metrics are supposed to be park-adjusted, but the adjustments are crude. A ground ball to shortstop in Tropicana Field (artificial turf, fast surface, predictable hops) is not the same play as a ground ball to shortstop at Wrigley Field (natural grass, variable conditions, sun angles in day games). UZR and DRS use zone-based systems that attempt to control for park, but they cannot fully capture how a specific park's surface, foul territory, wall geometry, and lighting conditions alter the difficulty of individual plays. OAA improves on this by using Statcast's ball-tracking data to estimate the probability of an out on each specific batted ball, but even OAA depends on correctly modeling the expected catch probability, which itself requires assumptions about how parks affect fielder positioning and route efficiency.

**Positional bias.** WAR applies a positional adjustment to account for the difficulty spectrum from catcher to designated hitter. These adjustments are based on historical data about the offensive levels at each position, but they are static within a given era. The adjustment for shortstop assumes a fixed relationship between defensive difficulty and offensive opportunity cost. If the talent pool at shortstop shifts -- as it has dramatically in recent years with the influx of athletic, offensive-minded shortstops like Trea Turner, Corey Seager, and Wander Franco -- the positional adjustment lags behind reality.

The net effect: for any player whose WAR depends heavily on defensive value, the number should be treated with a confidence interval of roughly plus or minus 1.5 wins. For a player like Simmons, whose career narrative hinges on elite defense, the difference between defensive systems can alter his legacy by 5 to 10 career WAR.

---

## 2. Replacement Level: The Arbitrary Anchor

WAR measures value above "replacement level," defined as the performance of a freely available minor-league or AAAA-type player. This sounds intuitive, but the implementation is circular and fragile.

FanGraphs calibrates replacement level so that the total WAR across MLB sums to approximately 1000 per season across 30 teams (roughly 33.3 WAR per team), split approximately 57% for position players and 43% for pitchers. Baseball-Reference uses a different calibration scheme that arrives at a similar but not identical total. The difference between them averages about 0.5 WAR per player across the board -- small in absolute terms but decisive when comparing players or when a Hall of Fame argument rests on a career total of 60 vs 64 WAR.

The fundamental problem is that replacement level is not an observable quantity. You cannot point to a game and say "that was replacement-level play." It is inferred from the aggregate performance of players at the bottom of the talent pool, and it shifts as the talent pool shifts. When MLB expanded rosters to 26 in 2020, the replacement-level baseline theoretically changed because the 780th-best player in baseball is different from the 750th-best. When minor-league development improves, replacement level rises. When it stagnates, replacement level falls.

If the replacement-level calibration is wrong by half a win -- and there is no empirical way to prove it is not -- then a player credited with 4.5 WAR might truly be worth 4.0 or 5.0 wins above a correctly calibrated replacement. This error compounds over careers: a 0.5 win-per-season error over 15 years is 7.5 career WAR, enough to change Hall of Fame candidacies.

The 57/43 split between position players and pitchers is itself a policy choice, not an empirical fact. It implies that roughly 57% of the wins in baseball are generated by position-player performance and 43% by pitching. But the marginal win-value of pitching versus hitting depends on the run environment, roster construction norms, and bullpen usage patterns, all of which have changed dramatically since the split was calibrated.

---

## 3. Run-to-Win Conversion: The Linear Lie

WAR converts runs to wins using a roughly linear conversion factor: approximately 10 runs equals 1 win. This is derived from the Pythagorean expectation framework, which models a team's expected win percentage as a function of runs scored and runs allowed. The standard formulation uses an exponent of approximately 1.83, and in a league-average run environment (around 4.5 runs per game), the marginal value of a run works out to about one-tenth of a win.

But the conversion is not linear. It is concave. In a low-run environment (3 runs per game, roughly the dead-ball era or the peak of the pitching-dominant 2014 season), each marginal run is worth more wins. In a high-run environment (5.5 runs per game, the steroid era), each marginal run is worth fewer wins. The difference can be 15 to 20 percent: a run might be worth 0.12 wins in a depressed environment and 0.09 wins in an inflated one.

This matters most for cross-era comparison. When we say that Barry Bonds' 2001 season (11.9 bWAR) was comparable to Babe Ruth's 1923 season (14.1 bWAR), we are implicitly trusting that the run-to-win conversion correctly accounts for the difference in run environments. But the conversion factor is typically calculated at the league-average level for each season, not at the team level, meaning a player on a high-scoring team (where marginal runs are worth less) gets the same conversion as a player on a low-scoring team (where marginal runs are worth more). This is a structural error that biases WAR in favor of players on low-scoring teams.

Additionally, the linearity assumption breaks down at the extremes of the win curve. A team that scores 900 runs and allows 700 is not simply "200 runs better" than a .500 team; the relationship between run differential and wins is S-shaped, and the marginal value of a run depends on where the team sits on that curve.

---

## 4. Baserunning: The Afterthought

WAR includes a baserunning component, typically derived from FanGraphs' BsR (Baserunning Runs) or a similar metric on Baseball-Reference. This captures stolen bases, caught stealings, and some advancement on hits and outs. What it does not capture is the full richness of baserunning skill.

Billy Hamilton is the obvious test case. From 2014 to 2018, Hamilton was widely considered the best baserunner in baseball by visual observation. His ability to go first-to-third on singles, to tag up from second on shallow fly balls, to advance on wild pitches and passed balls, and to disrupt pitcher timing was evident to anyone watching. Yet his BsR numbers, while positive, were not dramatically higher than several less heralded baserunners. Part of this is because BsR is built on event-level data (did the runner advance or not?) without fully weighting the difficulty or value of the specific advancement. A first-to-third advancement on a single to right field is treated similarly regardless of the outfielder's arm strength, the runner's read off the bat, or the game situation.

The deeper problem is methodological: baserunning in WAR is bolted on as a small additive component, typically worth -2 to +5 runs for the most extreme players. But baserunning interacts with offensive and strategic decisions in ways that a simple additive model cannot capture. Hamilton's presence on base forced pitchers to alter their pitch selection and timing, which benefited the batter at the plate. This cascading effect is invisible to WAR.

The omission of advance-on-wild-pitch events, subtle positioning reads (how deep a runner plays off the bag at second, how quickly they react to a ball in the dirt), and the deterrent effect of speed on defensive positioning all contribute to a systematic undervaluation of elite baserunners and a failure to penalize poor ones. For most players, the baserunning component is small enough that errors are drowned out. For specialists like Hamilton, Dee Strange-Gordon, or Trea Turner, the error can be meaningful.

---

## 5. Catcher Framing and Game-Calling: The Missing Wins

Perhaps the single largest known omission from traditional WAR is catcher pitch framing. Research from StatCorner, Baseball Prospectus, and Statcast's own strike zone models has consistently shown that elite pitch framers are worth 15 to 25 runs per season relative to average catchers, purely through their ability to receive borderline pitches in a way that increases called-strike probability. That is 1.5 to 2.5 wins per season from a skill that WAR simply ignores.

Austin Hedges is a prime example. Throughout his career with the Padres, Indians, and Pirates, Hedges was one of the worst offensive catchers in baseball, regularly posting sub-.600 OPS seasons. His fWAR hovered around 0 to 1 per season. But framing metrics consistently rated him as one of the five best framers in baseball, worth approximately 15 to 20 runs above average per season behind the plate. Including framing, Hedges was a solidly above-average player. Without it, WAR painted him as a replacement-level bat with a glove, which was at best half the picture.

Jose Trevino's breakout with the Yankees in 2022 is another case study. His 2.9 fWAR that season was built almost entirely on a .248/.283/.388 batting line and solid defensive ratings. But framing metrics credited him with approximately 18 runs above average in framing alone -- value that was invisible in his WAR total. Yasmani Grandal's career arc tells the same story: his market value and analytical reputation far exceeded his WAR totals because teams understood that framing was worth real wins.

Game-calling is even harder to quantify but almost certainly matters. The idea that a catcher's pitch-sequencing decisions, game-planning with pitchers, and in-game adjustments affect run prevention is supported by both pitcher testimonials and emerging research into catcher ERA differentials. But the effect is entangled with pitcher quality, sample size is a nightmare, and isolating the catcher's contribution from the pitcher's own adjustment ability remains an open problem. WAR ignores it entirely, which is defensible given the measurement difficulty but means that the metric systematically undervalues the best game-callers.

---

## 6. Leverage Blindness: Not All Innings Are Equal

WAR treats every plate appearance and every inning pitched as equally important. A strikeout in the bottom of the ninth with the bases loaded and the tying run on third is worth exactly the same as a strikeout in the third inning of a 12-1 blowout. This is by design -- WAR measures what happened, not when it happened -- but it creates systematic distortions, particularly for relief pitchers.

Mariano Rivera is the defining example. His career 56.3 bWAR is outstanding by any standard, but it almost certainly understates his true contribution to the Yankees. Rivera pitched almost exclusively in maximum-leverage situations: ninth innings with small leads, tie games in the playoffs, high-stakes moments where a single run changes the game outcome. WPA (Win Probability Added) captures this contextual value, and Rivera's career WPA of 56.6 is the highest of any reliever in history by a wide margin.

The disconnect between WAR and WPA for relievers is structural. A starting pitcher who accumulates 200 innings of average-leverage performance will generate more WAR than a reliever who accumulates 70 innings of extreme-leverage performance, even if the reliever's innings were more valuable in terms of actual wins produced. This is why WAR systematically undervalues closers and elite setup men relative to their actual impact on team wins.

For hitters, the leverage issue is less pronounced because most everyday players face a reasonably representative distribution of leverage situations over a full season. But there are edge cases: pinch hitters who bat almost exclusively in high-leverage situations, platoon players who face a skewed sample of pitchers, and situational batters whose managers deploy them strategically. For all of these, WAR's context-neutrality is a feature (it measures talent independent of situation) and a bug (it does not measure how that talent was deployed).

---

## 7. Pitching/Hitting Unification: The Fragile Bridge

WAR claims to put pitchers and hitters on the same scale, but the methodologies are fundamentally different. For hitters, both fWAR and bWAR use variants of linear weights (wRAA) to convert offensive production into runs above average, then add defensive and baserunning components. For pitchers, fWAR uses FIP (Fielding Independent Pitching), which values only strikeouts, walks, hit-by-pitches, and home runs. bWAR uses RA/9 (runs allowed per nine innings), which includes all run prevention, including the contributions of the defense behind the pitcher.

These are not minor methodological footnotes. FIP-based fWAR and RA/9-based bWAR can disagree on a pitcher's value by 2 or more wins in a single season. A pitcher who induces weak contact and benefits from a good defense will look better in bWAR than fWAR. A pitcher who strikes out 12 per nine but allows hard contact when the ball is put in play will look better in fWAR than bWAR.

The replacement-level calibration is supposed to make pitching WAR and hitting WAR comparable, so that Mike Trout's 8.3 fWAR in 2019 and Jacob deGrom's 7.0 fWAR in 2018 can be placed on the same scale. But this comparability rests on the 57/43 split discussed in Section 2 and on the assumption that the run-to-win conversion is equally valid for pitching runs and hitting runs. Neither assumption is well-grounded. The bridge between pitching WAR and hitting WAR is a policy decision dressed up as an empirical fact.

This matters for MVP voting, Hall of Fame arguments, and contract valuation. When we say Trout's 2019 was "worth more" than deGrom's 2018, we are trusting the entire chain: linear weights for Trout, FIP (or RA/9) for deGrom, the same replacement-level baseline for both, the same run-to-win conversion for both, and the same positional framework for both. Each link in that chain carries its own uncertainty, and the compounded uncertainty is larger than most WAR users realize.

---

## 8. Aging, Injury, and Durability: The Retrospective Trap

WAR is a purely retrospective metric. It tells you what a player did, not what he is likely to do. This is philosophically defensible but practically limiting, especially regarding durability.

Consider two hypothetical players in 2023: Player A puts up 5.0 WAR in 162 games, Player B puts up 5.0 WAR in 80 games before a torn ACL ends his season. Their WAR totals are identical. But Player B's rate performance was dramatically better (approximately 10 WAR per 162 games), while Player A's durability was dramatically better. In terms of realized value to their team, they were equal. In terms of future value, contract value, and trade value, they are very different players.

WAR has no mechanism to credit durability or penalize fragility. A player who is available for 150+ games every year provides a compounding benefit that WAR captures only indirectly (through accumulated plate appearances) and never explicitly values. The insurance value of reliability -- knowing that you will not need to replace production mid-season with a replacement-level callup -- is real and significant, but invisible in WAR.

Similarly, WAR has no trajectory awareness. A 28-year-old posting 5 WAR is on a different career arc than a 35-year-old posting 5 WAR. The younger player's WAR is more likely to persist or increase; the older player's is more likely to decline. For team-building purposes, these are not equivalent seasons, but WAR treats them identically.

---

## 9. Platoon Splits, Lineup Protection, and Sequencing

WAR does not account for the context in which a player's performance occurs, beyond basic park and league adjustments. Several important contextual factors are either ignored or handwaved.

**Platoon splits.** An everyday player who hits .350 wOBA against right-handed pitchers and .280 wOBA against lefties has a very different profile than one who hits .315 against both. Their seasonal WAR might be identical, but the platoon-variant player is more susceptible to managerial matchup exploitation and less valuable in a postseason series where opposing managers can stack their bullpen. WAR treats both players the same. Platoon splits can exceed 50 points of wOBA for extreme cases, and they interact with lineup construction in ways that a single-number metric cannot represent.

**Lineup protection.** Whether the presence of a strong hitter in an adjacent lineup spot affects a player's production is one of the most debated topics in sabermetrics. The evidence is mixed but not zero: research has shown small but measurable effects on walk rates and pitch quality when protection changes. WAR implicitly assumes that each player's production is independent of his lineup context, which is approximately but not exactly true.

**Sequencing and BABIP.** WAR, as built on wOBA for hitters, does not distinguish between a player whose .330 BABIP is skill-based (high line-drive rate, elite exit velocity) and one whose .330 BABIP is luck-driven (normal contact quality, fortunate placement). Over a full season, BABIP can vary by 30 or more points from a player's true-talent level, which translates to roughly 1 to 1.5 WAR of noise. Similarly, HR/FB rate fluctuations (a player running a 15% HR/FB when his true talent is 12%) inject additional variance that WAR does not filter.

---

## 10. Park Factors: The Illusion of Precision

WAR applies park factors to normalize offensive production across ballparks. The concept is sound; the execution is crude. Park factors are typically calculated as simple multiplicative scalars applied uniformly to all offensive events. Coors Field might carry a park factor of 114 (meaning 14% more runs are scored there than at a neutral park), and every Rockies hitter's offensive production is adjusted downward accordingly.

But parks do not affect all hitters or all types of offense equally. Yankee Stadium's short right-field porch (314 feet down the line) creates a paradise for left-handed pull hitters while being relatively neutral for right-handed hitters. A single park factor applied to both Aaron Judge (right-handed) and Anthony Rizzo (left-handed) in the same stadium is applying the wrong adjustment to at least one of them. The short porch inflates Rizzo's home-run totals far more than Judge's, but the park factor does not distinguish between them.

Coors Field is the most extreme and well-studied case. The altitude affects fly-ball carry, the thin air reduces pitch movement (making hitters look better), the large outfield dimensions increase BABIP on non-home-run fly balls, and the humidor (introduced in 2002, expanded to all baseballs in 2022) has altered the park's effect over time. A simple multiplicative factor cannot capture the interaction between altitude and batted-ball profile. A fly-ball hitter at Coors gets a different (and larger) boost than a ground-ball hitter, but both receive the same park adjustment in WAR.

Day-night splits, weather effects, altitude, humidity, foul territory size, wall height, and wall distance create a complex web of factors that a single number per park cannot represent. Research by Dr. Alan Nathan and others on batted-ball physics has shown that temperature and humidity alone can alter home-run probability by 10 to 15 percent between a cold April night and a humid August afternoon in the same park.

The error introduced by crude park factors is typically small for players in neutral parks but can be substantial for players who spend 81 games per year in extreme environments. Larry Walker's career bWAR of 72.7 reflects Coors adjustments, but whether those adjustments are correct -- whether Walker really was a 72.7 WAR player rather than a 68 or 76 WAR player -- depends entirely on the accuracy of park factors that are known to be oversimplified.

---

## Conclusion: What Comes Next

WAR is not useless. It is an imperfect compression of a high-dimensional problem into a single scalar, and it does that compression better than any predecessor metric. But treating it as ground truth, as precision measurement rather than rough estimate, leads to systematic errors in player evaluation, compensation, and historical comparison.

The confidence interval on a single-season WAR is at minimum plus or minus 1 win, and likely closer to plus or minus 1.5 wins when defensive uncertainty, park factor error, baserunning omissions, and framing exclusions are compounded. For career WAR, the accumulated uncertainty can reach 10 or more wins -- enough to move players across tiers in Hall of Fame arguments.

A successor metric must address the structural issues identified here: defense must carry explicit uncertainty bands; replacement level must be empirically grounded rather than calibrated by fiat; the run-to-win conversion must account for nonlinearity; catcher framing must be included; leverage must be optionally weighted; and park factors must be decomposed by batted-ball type, handedness, and environmental conditions. Whether these improvements can be collapsed back into a single number, or whether the single-number paradigm itself is the problem, is the central design question for what comes next.
