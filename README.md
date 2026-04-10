# BRAVS — Bayesian Runs Above Value Standard

A probabilistic baseball player valuation framework. Computes posterior distributions over player value for every MLB player from 1920 to present.

**299,120 MLB + 223,855 MiLB player-seasons. 7 PyTorch models (3.5M params). 38 API endpoints. 9-tab web app. GPU-accelerated.**

## What It Does

BRAVS decomposes a player's total value into 9 measurable components — hitting, pitching, baserunning, fielding, catcher-specific value, positional adjustment, leverage context, durability, and approach quality — then combines them using Bayesian inference to produce a full probability distribution, not just a point estimate.

Every valuation comes with a credible interval. "Player X was worth 8.2 wins" becomes "Player X was worth 8.2 wins, with 90% probability between 6.5 and 9.9."

## How It Compares to WAR

BRAVS was validated against 32 career fWAR benchmarks:

| Metric | Value |
|--------|-------|
| Pearson correlation with fWAR | 0.890 |
| RMSE | 10.8 wins |
| Mean ratio | 1.01 (perfectly calibrated on average) |
| HOF classification accuracy | 96.0% at 90% confidence |

Near-exact career matches: Trout (+0.5), Henderson (+0.4), Mays (+2.8), Mantle (-1.2), Williams (+1.4), A-Rod (+1.6), Morgan (+2.4), Seaver (+3.7).

## Quick Start

```bash
# Install dependencies
pip install numpy scipy pandas matplotlib torch pybaseball

# Download Lahman data (one-time setup)
# Place baseballdatabank CSV files in data/lahman2025/

# Compute BRAVS for all of baseball history on GPU
python scripts/compute_everything.py

# Run the web app
python web/app.py
# Open http://localhost:5000

# Generate a pitcher performance card
python scripts/pitcher_card.py --pitcher-id 543037 --date 2024-06-19
```

## All-Time Leaderboard

Computed from 128,598 batting rows (1871-2025) via Lahman database:

| Rank | Player | Career WAR-eq |
|------|--------|--------------|
| 1 | Willie Mays | 142.5 |
| 2 | Barry Bonds | 141.2 |
| 3 | Babe Ruth | 134.7 |
| 4 | Ted Williams | 132.8 |
| 5 | Alex Rodriguez | 127.9 |
| 6 | Stan Musial | 124.8 |
| 7 | Greg Maddux | 123.5 |
| 8 | Hank Aaron | 121.0 |
| 9 | Roger Clemens | 120.8 |
| 10 | Mickey Mantle | 117.5 |

## Architecture

```
baseball_metric/            # Core Python package (37 modules)
├── core/
│   ├── model.py            # Central BRAVS computation (Python)
│   ├── gpu_engine_v3.py    # CUDA-accelerated batch engine (PyTorch)
│   ├── posterior.py         # Multivariate posterior combination
│   ├── types.py             # PlayerSeason, BRAVSResult types
│   └── mcmc.py              # Optional Metropolis-Hastings sampler
├── components/              # 9 value components
│   ├── hitting.py           # Bayesian wOBA
│   ├── pitching.py          # Bayesian FIP+ with walk penalty
│   ├── baserunning.py       # SB/CS + GIDP avoidance
│   ├── fielding.py          # Range factor + Gold Glove/All-Star bonus
│   ├── catcher.py           # Framing + blocking + throwing + game-calling
│   ├── positional.py        # Multi-position weighted adjustment
│   ├── leverage.py          # Damped sqrt(gmLI) context weighting
│   ├── durability.py        # Season-length-prorated availability
│   └── novel_component.py   # Approach Quality Index (orthogonalized)
├── adjustments/             # Park factors, era normalization, dynamic RPW
├── data/
│   └── lahman.py            # Lahman database integration (1871-2025)
├── analysis/                # Sensitivity, stability, projections
└── visualization/           # Player cards, leaderboards, plots

├── lineup_optimizer/        # GPU-accelerated lineup optimization
│   ├── optimizer.py         # 30K-candidate GPU search (0.05s/team)
│   ├── model.py             # LineupValueNetwork + SlotInteractionModel
│   ├── platoon.py           # Bayesian hierarchical platoon splits
│   ├── fatigue.py           # Continuous fatigue model (age/position/workload)
│   ├── series_optimizer.py  # 3-game series joint optimization
│   ├── season_optimizer.py  # 162-game playing time allocation
│   ├── trade_impact.py      # Trade simulation with marginal value
│   ├── backtest.py          # Historical backtesting engine
│   └── data_builder.py      # Training data from Lahman + BRAVS

bravs_engine/               # Rust engine with PyO3 bindings
web/                        # Flask web app (7 tabs)
scripts/                    # 22 analysis scripts
data/                       # Lahman CSVs (128K batting, 57K pitching, 174K fielding)
```

## Web App

9 tabs: Player, Award Races, Team, Live MVP, Dynasties, Dream Team, Lineup Optimizer, MiLB, Video.

- Search any player in MLB history, instant results from pre-computed CSV data
- Player headshots and team logos from MLB CDN
- Career sparkline chart, future projections, player similarity finder
- "What If" position swap mode
- Side-by-side comparison (up to 6 players)
- Award race viewer for every MVP and Cy Young since 1956

## Lineup Optimizer

GPU-accelerated lineup optimization system validated across 120 team-seasons (2022-2025):

| Metric | Value |
|--------|-------|
| Lineup value vs actual runs | r = 0.905 |
| Expected wins vs actual wins | r = 0.765 |
| Candidates evaluated per team | 30,000 |
| Time per team (RTX 5060 Ti) | 0.05s |
| Full backtest (120 teams) | 6.2s |

Features:
- **Single-game**: Optimal 9-man lineup + batting order from 30K GPU-evaluated candidates
- **Series-level**: Joint 3-game optimization with fatigue, platoon, and rest management
- **Season-long**: 162-game playing time allocation with projected wins
- **Trade simulator**: Marginal lineup value of acquisitions/trades
- **Bayesian platoon splits**: Hierarchical shrinkage for L/R matchup effects
- **Fatigue model**: Age/position/workload-dependent performance degradation

Correctly identifies 2023 Braves as the best lineup (411.7), 2024 White Sox as the worst (38.0), and all recent WS champions rank in the top tier.

See `docs/lineup-optimizer-methodology.md` for full technical details and `docs/lineup-optimizer-findings.md` for analysis.

## Minor League Data (MiLB)

283,141 minor league player-seasons (2005-2026) across all affiliated levels:

| Level | Batting Seasons | Avg wOBA |
|-------|----------------|----------|
| AAA | 12,803 | .334 |
| AA | 9,093 | .314 |
| A+ | 9,101 | .313 |
| A | 9,235 | .307 |
| A- | 4,121 | .302 |
| Rk | 16,769 | .310 |

Level translation rates calibrated to "The Book" (Tango/Lichtman/Dolphin):
AAA 75%, AA 60%, A+ 50%, A 42%, Rk 25%.

Includes prospect projection model (GBM) predicting MLB career WAR from MiLB stats.

Source: [armstjc/milb-data-repository](https://github.com/armstjc/milb-data-repository)

## Trained Models

| Model | Type | Performance | Purpose |
|-------|------|-------------|---------|
| BRAVS v3.8 GPU Engine | PyTorch CUDA | r=0.873 vs fWAR | Core player valuation |
| Win Prediction | Neural Net | r=0.903, 6.6 RMSE | Team wins from roster |
| HOF Classifier | Neural Net | 97.4% accuracy | Hall of Fame probability |
| Prospect Projector (GBM) | GradientBoosting | R²=0.20 | MiLB→MLB career WAR |
| Prospect Projector (NN) | PyTorch | r=0.405 | MiLB→MLB with uncertainty |
| Breakout Predictor | GradientBoosting | AUC=0.774 | Predict 3+ WAR jumps |
| Player Embeddings | Autoencoder | 16-dim latent | Historical player similarity |
| Lineup Value | Neural Net + Transformer | r=0.905 vs runs | Optimal batting order |
| Pitch Arsenal Grades | Percentile model | 6,573 pitch-types | Per-pitch effectiveness |
| Fatigue Model | Continuous/differentiable | Calibrated to known rates | Workload management |
| Platoon Model | Bayesian hierarchical | Shrunk to population priors | L/R split estimation |

## GPU Engine

The v3.7 GPU engine processes all 75,265 qualified player-seasons in 1.38 seconds on an RTX 5060 Ti:

- All Bayesian updates, wOBA computation, FIP, posterior sampling vectorized as CUDA tensor operations
- 2,000 posterior samples per player
- Talent dilution adjustment for pre-expansion eras
- Era-adjusted pitcher calibration (pre-1985 vs post-1985)
- Gold Glove and All-Star fielding bonuses from awards data
- Progressive walk penalty for extreme BB/9 pitchers

## Data

All historical analysis runs from local CSV files with zero API calls:

| Dataset | Rows | Range | Source |
|---------|------|-------|--------|
| Batting | 128,598 | 1871-2025 | CRAN Lahman v14 |
| Pitching | 57,630 | 1871-2025 | CRAN Lahman v14 |
| Fielding | 174,332 | 1871-2025 | CRAN Lahman v14 |
| Appearances | 128,512 | 1871-2025 | Games at each position |
| People | 24,270 | All time | Biographical data |
| HOF voting | 6,426 | 1936-2026 | Full ballot history |
| Awards | 12,667 | 1877-2025 | MVP, Cy Young, Gold Glove, All-Star |
| Postseason | 18,687 | 1884-2025 | Playoff stats |

The MLB Stats API is used only for 2026 (current in-progress season).

## Analyses

22 analysis scripts covering:

- Every MVP and Cy Young race 1956-2025 (265 races, 42.6% voter agreement)
- Active player HOF check (12 locks: Kershaw, Trout, Verlander, Scherzer, Freeman, Goldschmidt, McCutchen, Betts, Arenado, Altuve, Sale, Harper)
- All-time career leaderboard (14,092 careers)
- Best single season ever (Ruth 1921 at 21.0 WAR-eq)
- Best 5-year dynasty windows (Ruth 1920-1924 at 72.8 WAR-eq)
- Steroid era comparison
- Talent trends by decade (baseball getting more evenly distributed)
- Best teams ever (1921 Cleveland at 76.7 team WAR-eq)
- Pitcher performance cards from Statcast data (2016+)
- Player similarity finder
- Trade analyzer
- "What if they stayed healthy" projections
- Lineup optimizer backtesting (120 team-seasons, 2022-2025)

## Metric Evolution

The metric went through 10 major calibration rounds:

| Version | r with fWAR | RMSE | Key Change |
|---------|-------------|------|------------|
| v1.0 | 0.532 | 27.3 | Initial implementation |
| v3.0 | 0.784 | 15.3 | Walk penalty, cube-root era dampening |
| v3.3 | 0.861 | 15.7 | Talent dilution for pre-expansion eras |
| v3.6 | 0.882 | 11.0 | Era-adjusted pitcher calibration |
| v3.7 | **0.890** | **10.8** | Tighter fielding, Gold Glove/All-Star bonuses |

Remaining structural gaps (not fixable from box-score data):
- Bonds (-14.2): missing actual UZR/DRS defensive metrics for 8 Gold Glove seasons
- Ruth (-22.4): pre-1920 pitching career not in dataset
- Maddux (+23.4): FIP overvalues command (needs RA/9 or SIERA)

## Testing

```bash
python -m pytest tests/ -v    # 116 tests, all passing
```

## License

MIT
