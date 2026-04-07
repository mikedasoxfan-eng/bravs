# BRAVS — Bayesian Runs Above Value Standard

**A probabilistic baseball player valuation framework that produces posterior distributions over player value, measured in wins above Freely Available Talent.**

## What is BRAVS?

BRAVS is a next-generation baseball player valuation metric designed to address the fundamental limitations of Wins Above Replacement (WAR). Where WAR produces a single number — "this player was worth 8.2 wins" — BRAVS produces a full probability distribution: "this player was worth 8.2 wins on average, with a 90% chance their true value falls between 6.5 and 9.9 wins."

This matters because baseball measurement is noisy. Defensive metrics disagree with each other by multiple wins. A player's "true talent" batting average can only be estimated from noisy at-bat outcomes. The run value of a stolen base depends on the base-out state. BRAVS embraces this uncertainty rather than hiding it behind a false precision that WAR's decimal places imply.

## What does BRAVS do that WAR can't?

1. **Uncertainty quantification**: Every BRAVS estimate comes with credible intervals. When two players are separated by 0.5 wins but their intervals overlap by 3 wins, BRAVS tells you that — WAR doesn't.

2. **Leverage-aware valuation**: WAR treats all innings and plate appearances as equally important. BRAVS uses a damped leverage adjustment (sqrt of leverage index) that credits closers and late-inning relievers for deploying their skills when they matter most, without the extreme noise of Win Probability Added.

3. **Catcher-specific value**: WAR completely ignores pitch framing, which can be worth 2-3 wins per season for elite framers. BRAVS includes framing, blocking, throwing, and game-calling as explicit components.

4. **Approach Quality Index (AQI)**: A novel component measuring how well a batter manages the strike zone relative to count and situation. This captures swing decision quality beyond traditional plate discipline stats.

5. **Dynamic run-to-win conversion**: Instead of the static "10 runs = 1 win" assumption, BRAVS uses a Pythagorean-derived conversion that varies with run environment — correctly valuing runs more in low-scoring environments.

## Quick Start

```bash
# Install
pip install -e .

# Compute BRAVS for notable historical player-seasons
python -m baseball_metric --notable-seasons

# Compute BRAVS for a full season (uses pybaseball or synthetic data)
python -m baseball_metric --season 2023

# Filter to a specific player
python -m baseball_metric --season 2023 --player "Mike Trout"

# Verbose output with component-level detail
python -m baseball_metric --notable-seasons -v
```

## Headline Findings

From our analysis of 22 notable player-seasons spanning 1913-2023:

| Rank | Player | Season | BRAVS | 90% CI |
|------|--------|--------|-------|--------|
| 1 | Barry Bonds | 2004 | 29.8 | [27.6, 32.0] |
| 2 | Bob Gibson | 1968 | 28.1 | [25.2, 31.0] |
| 3 | Willie Mays | 1965 | 26.4 | [23.1, 29.6] |
| 4 | Sandy Koufax | 1966 | 24.5 | [21.9, 27.0] |
| 5 | Walter Johnson | 1913 | 23.0 | [21.1, 25.0] |

**Key findings:**

- **Mariano Rivera's 2004 gets a 40% leverage boost** (+8.6 runs from leverage context), properly crediting his elite skills deployed in maximum-leverage situations. Under WAR, Rivera's value is systematically understated.

- **Ohtani 2023 is naturally a two-way player** at 16.9 BRAVS (hitting: +81.3 runs, pitching: +22.5 runs). No special handling needed — the unified framework sums hitting and pitching value directly.

- **Larry Walker's Coors-inflated 1997** is properly deflated from a Coors park factor of 1.16, landing at 10.7 BRAVS — still excellent, but the park adjustment does meaningful work.

- **Harold Baines (5.5) vs. Larry Walker (10.7)** clearly separates the compiler from the peak player, supporting the case that Walker is the more deserving Hall of Famer.

## Project Structure

```
baseball_metric/          # Core Python package
├── core/                 # Central model, types, posterior computation
├── components/           # 9 value components (hitting, pitching, fielding, etc.)
├── adjustments/          # Park factors, era normalization, dynamic RPW
├── data/                 # Data ingestion, synthetic data, validation
├── analysis/             # Sensitivity, stability, backtesting, bias detection
└── visualization/        # Player cards, leaderboards, posterior plots

docs/                     # Full documentation
├── 01-war-autopsy.md     # Comprehensive teardown of WAR's limitations
├── 02-literature-review.md # Survey of existing metrics and academic work
├── 03-axioms.md          # Philosophical foundation (7 axioms)
├── 04-metric-specification.md # Complete mathematical specification
├── 05-methodology-deep-dive.md # Publishable-quality technical paper
├── 06-design-decisions.md # Every significant design choice documented
└── 07-self-critique.md   # Honest adversarial analysis

tests/                    # 116 unit tests covering all components
logs/                     # Test run and computation logs
output/                   # Generated visualizations
```

## Documentation

- [WAR Autopsy](docs/01-war-autopsy.md) — What's wrong with the current gold standard
- [Literature Review](docs/02-literature-review.md) — The landscape of baseball analytics
- [Axioms](docs/03-axioms.md) — The philosophical foundation
- [Metric Specification](docs/04-metric-specification.md) — Complete mathematical definition
- [Methodology Deep-Dive](docs/05-methodology-deep-dive.md) — Publishable technical paper
- [Design Decisions](docs/06-design-decisions.md) — Why we made each choice
- [Self-Critique](docs/07-self-critique.md) — Honest assessment of limitations

## Testing

```bash
# Run full test suite
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=baseball_metric --cov-report=term-missing
```

## Known Limitations

1. **Scale inflation**: BRAVS values are systematically higher than WAR due to the AQI component and dynamic RPW. This is partially by design (BRAVS measures more) and partially a calibration issue.

2. **Historical fielding**: Pre-2000 players get zero fielding value with wide uncertainty. We chose intellectual honesty over noisy estimates.

3. **AQI proxy model**: Without Statcast pitch-level data, AQI is estimated from traditional plate discipline stats, which partially overlaps with the hitting component.

4. **2020 season**: The durability component heavily penalizes the 60-game season. A season-length adjustment should be applied.

See [Self-Critique](docs/07-self-critique.md) for a full adversarial analysis.

## License

MIT
