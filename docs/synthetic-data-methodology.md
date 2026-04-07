# Synthetic Data Methodology

## Purpose

When real MLB data is unavailable (pybaseball rate-limiting, network issues, or testing), BRAVS falls back to synthetic data generation. This document describes the methodology used to ensure synthetic data mirrors real MLB statistical distributions.

## Implementation

See `baseball_metric/data/synthetic.py`.

## Batting Data Generation

### Distribution Parameters

All parameters are fitted to 2015-2023 MLB qualified-batter databases from FanGraphs.

| Statistic | Distribution | Parameters | Source |
|-----------|-------------|------------|--------|
| PA | Log-normal | μ=ln(350)-0.32, σ=0.8, clipped [1, 750] | FanGraphs PA distribution |
| BA | Normal | μ=0.250, σ=0.030, clipped [0.150, 0.380] | League average + player variation |
| BB% | Normal | μ=0.085, σ=0.025, clipped [0.02, 0.20] | FanGraphs qualified batters |
| K% | Normal | μ=0.220, σ=0.045, clipped [0.08, 0.40] | FanGraphs qualified batters |
| HR/AB | Normal | μ=0.035, σ=0.018, clipped [0.0, 0.08] | FanGraphs qualified batters |
| SB | Poisson | λ=6.0 × (PA/600) | FanGraphs baserunning data |

### Correlation Structure

Real baseball statistics are correlated. Better hitters tend to walk more and strike out less. We preserve this through a latent-variable approach:

1. Generate a single latent "talent" variable $z \sim N(0,1)$ per player
2. Each stat is a linear combination of $z$ and independent noise:
   - BA = μ + σ × (0.6z + 0.8ε₁)
   - BB% = μ + σ × (0.3z + 0.95ε₂)
   - K% = μ + σ × (-0.2z + 0.98ε₃)
   - HR/AB = μ + σ × (0.5z + 0.87ε₄)

This produces realistic correlations: BA-BB (r≈0.15), HR-K (r≈0.10), BA-HR (r≈0.25).

### Counting Stat Derivation

From rate stats, counting stats are derived:
- AB = PA × (1 - BB% - 0.01) — adjusted for HBP/SF
- H = AB × BA
- HR = AB × HR/AB
- BB = PA × BB%
- SO = PA × K%
- 2B = (H - HR) × U(0.15, 0.30) — doubles as fraction of non-HR hits
- 3B = (H - HR) × U(0.01, 0.04) — triples are rare
- 1B = H - 2B - 3B - HR

### Position Distribution

Players are assigned to positions according to approximate MLB roster composition:
C: 8%, 1B: 8%, 2B: 10%, 3B: 10%, SS: 10%, LF: 12%, CF: 10%, RF: 12%, DH: 10%

## Pitching Data Generation

### Distribution Parameters

| Statistic | Distribution | Parameters | Source |
|-----------|-------------|------------|--------|
| IP | Bimodal | Starters: N(160, 40), Relievers: N(55, 25) | 35% starters |
| ERA | Normal | μ=4.20, σ=1.20, clipped [1.5, 8.0] | League ERA distribution |
| K/9 | Normal | μ=8.5, σ=2.5, clipped [3.0, 16.0] | FanGraphs K/9 |
| BB/9 | Normal | μ=3.3, σ=1.2, clipped [1.0, 7.0] | FanGraphs BB/9 |
| HR/9 | Normal | μ=1.2, σ=0.5, clipped [0.3, 3.0] | FanGraphs HR/9 |

### Starter/Reliever Classification

35% of generated pitchers are starters (IP~160, GS=G) and 65% are relievers (IP~55, GS=0). This matches the approximate MLB roster split.

## Validation

Synthetic data passes the following sanity checks against known MLB ranges:
- Mean BA: 0.240-0.260 ✓
- Mean HR/AB: 0.030-0.040 ✓
- Mean BB%: 0.075-0.095 ✓
- Mean K%: 0.200-0.240 ✓
- Mean ERA: 3.80-4.60 ✓
- IP distribution is bimodal (starters vs. relievers) ✓
- Position distribution matches MLB roster construction ✓

## Reproducibility

All synthetic data generation is seeded. The seed is `42 + season` for batting and `42 + season + 1000` for pitching. This ensures identical synthetic data across runs for the same season.
