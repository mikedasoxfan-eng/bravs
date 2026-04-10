# BRAVS Lineup Optimizer — Methodology

## Overview

The BRAVS Lineup Optimizer is a GPU-accelerated system that finds optimal batting orders, position assignments, and playing-time allocations by evaluating tens of thousands of candidate lineups in parallel on an NVIDIA RTX 5060 Ti.

It operates at three time horizons:
1. **Single-game**: Optimal 9-man lineup + batting order for today's game
2. **Series-level**: Joint optimization across a 3-4 game series (fatigue + platoon + rest)
3. **Season-long**: 162-game playing time allocation across the full roster

## Architecture

### Phase 1: Training Data Construction (`data_builder.py`)

Training data is built from 780 team-season records (2000-2025), combining:
- Pre-computed BRAVS decompositions from `bravs_all_seasons.csv`
- Actual team outcomes from Lahman's Teams table

**17 features per team-season:**
- Lineup hitting quality (sum, mean, std, best, worst of top 9)
- Baserunning aggregate
- Fielding aggregate
- Positional value
- Total BRAVS (sum + mean)
- Roster depth (count of WAR-eq > 0 players)
- Power concentration (top-3 HR share)
- Speed (total SB)
- Rotation quality (top-5 pitcher BRAVS)
- Position diversity

**Target**: Team runs scored per game (R/G).

### Phase 2: Models (`model.py`)

**LineupValueNetwork** — 3-layer neural network with Gaussian NLL loss:
- Input: 17 team-composition features
- Hidden: 128 → 128 → 64 (ReLU + dropout 0.1)
- Output: mean R/G prediction + log-variance (uncertainty)
- Trained with cosine annealing LR schedule, 500 epochs
- Validation split: 15%

**SlotInteractionModel** — Transformer for batting-order effects:
- Input: 9 player feature vectors (one per slot)
- Slot positional embeddings (learned, not sinusoidal)
- 2-head, 2-layer transformer encoder
- Output: scalar run adjustment from ordering
- Captures lineup protection, table-setter effects, OBP-before-power patterns

### Phase 3: Lineup Optimizer (`optimizer.py`)

**GPU-parallelized search** evaluating N candidate batting orders simultaneously:

1. **Starter selection**: Rank all roster players by total BRAVS value, ensure at least one catcher, select top 9
2. **Position assignment**: Greedy assignment filling premium positions (C, SS, CF) first, matching players to their primary position
3. **Feature construction**: 8-dimensional feature vector per player (hitting, baserunning, fielding, positional, AQI, HR, SB, PA)
4. **Candidate generation**: N random permutations of the 9 batting slots
5. **GPU evaluation**: All N orders evaluated in parallel using `torch.gather`:
   - Base value: sum of hitting runs
   - Leadoff bonus: 5% of slot-0 OBP proxy
   - 3-hole bonus: 3% of slot-2 value
   - Cleanup bonus: 2% of slot-3 power
   - Adjacency bonus: consecutive positive-value hitters (+0.01 per pair)
6. **Top-K extraction**: `torch.topk` returns best lineups

**Performance**: 30,000 candidates evaluated in ~0.05s on RTX 5060 Ti.

### Phase 3b: Platoon Splits (`platoon.py`)

Bayesian hierarchical model for L/R platoon effects:

**Population priors** (from "The Book" by Tango/Lichtman/Dolphin):
- LHB vs LHP: -3.0 runs/600PA penalty (same-side)
- RHB vs RHP: -1.6 runs/600PA penalty
- Switch hitters: -0.4 runs/600PA (minimal)

**Hierarchical shrinkage**: Each player's observed split is shrunk toward the population mean for their handedness group, weighted by sample size. Low-PA players get population-average splits; high-PA players retain more of their individual signal.

**Position modifiers**: Catchers +10%, DHs +12%, middle infielders -5% to -7% (more two-way capable).

### Phase 3c: Fatigue Model (`fatigue.py`)

Continuous, differentiable fatigue model:

**Inputs**: games_last_7, games_last_14, games_last_30, age, position
**Output**: multiplicative performance factor in [0.85, 1.05]

**Key dynamics**:
- Short-term fatigue (last 7 days) weighted 2x vs medium-term (8-14 days)
- Age modifier: `1 + 0.0008 * max(age - 28, 0)^1.5`
- Position rates: catchers 1.35x base fatigue, DHs 0.75x
- Recovery: +0.02 per day off (up to 3 days)
- Rhythm loss: after 4+ days off, -0.005 * excess_rest^1.5

**Calibration targets**:
- Playing 14 straight: ~0.97
- After 1 day off: ~1.02
- After 7+ days off: ~0.98 (rhythm loss)

Both float and PyTorch tensor paths supported for gradient-based optimization.

### Phase 4: Series Optimizer (`series_optimizer.py`)

Joint optimization across a 3-game series using greedy + local search:

1. **Greedy pass**: Optimize each game independently with accumulated fatigue state
2. **Local search**: For top-K players, try swapping rest days between games. Accept swaps that improve total series expected value
3. **Convergence**: Stop when no improving swap found or iteration limit reached

**Exhaustive mode** available for short series: enumerate all valid rest patterns (at most 1 rest day per player, at most 2 players resting per game), prune to ~100 viable patterns, evaluate each.

### Phase 4b: Season Optimizer (`season_optimizer.py`)

162-game playing time allocation:

**Player tiers**:
- Stars (total value > 40): 140-150 games (more rest for age > 30)
- Regulars (10-40): 138 games
- Bench (< 10): 80 games (top 5 only)

**Win projection**: Team wins ≈ 47 (FAT baseline) + total projected WAR-eq

**Roster flexibility**: +0.5 wins per multi-position player

### Phase 5: Trade Impact (`trade_impact.py`)

Marginal lineup value of roster changes:

1. Optimize lineup with current roster → before_value
2. Remove outgoing players, add incoming players
3. Optimize lineup with new roster → after_value
4. Marginal impact = after_value - before_value
5. Positional surplus changes identify which positions are strengthened/weakened

### Phase 6: Historical Backtesting (`backtest.py`)

For each team-season:
1. Build roster from pre-computed BRAVS
2. Optimize lineup (30K candidates)
3. Run season optimizer
4. Compute positional surplus
5. Compare to actual team performance

## Validation Results

### Backtesting (120 team-seasons, 2022-2025)

| Metric | Value |
|--------|-------|
| Lineup Value vs Actual Runs | r = 0.905 |
| Expected Wins vs Actual Wins | r = 0.765 |
| Wins RMSE | 19.2 |
| Total backtest time | 6.2s (GPU) |
| Avg time per team | 0.05s |

### Known Case Validation

| Team | Year | Actual W | BRAVS Lineup Value | Note |
|------|------|----------|-------------------|------|
| ATL | 2023 | 104 | 411.7 (1st) | Best lineup in dataset |
| LAN | 2023 | 100 | 329.2 (2nd) | |
| TEX | 2023 | 90 | 292.5 (4th) | WS champions |
| LAN | 2024 | 98 | 291.9 (5th) | WS champions |
| CHA | 2024 | 41 | 38.0 (last) | Worst team in modern history |

### Why wins RMSE is 19.2

The model uses only position-player WAR to predict wins. It does not include:
- Pitching quality (accounts for ~50% of win variance)
- Bullpen leverage
- In-season injuries/roster moves
- Schedule strength

The 0.905 correlation with runs scored (which is batting-only) validates the model is capturing lineup quality correctly. The wins gap is expected without pitching.

## Design Decisions

1. **GPU over CPU**: Even for simple sum-based evaluation, GPU parallelism dominates when N > 5000 candidates
2. **Heuristic ordering bonus over learned model**: The transformer ordering model requires Retrosheet lineup-card training data we don't have. The heuristic bonuses (leadoff OBP, cleanup power, adjacency) are well-established in sabermetric literature
3. **Greedy + local search over DP**: For 3-game series, the greedy pass is near-optimal. Local search closes the gap without the exponential state space of full DP
4. **Bayesian platoon over raw splits**: Small-sample platoon data is extremely noisy. Hierarchical shrinkage produces better-calibrated estimates
5. **Continuous fatigue model**: Differentiable in all inputs, enabling future integration with gradient-based season planning
