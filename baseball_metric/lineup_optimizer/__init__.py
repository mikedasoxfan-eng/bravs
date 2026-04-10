"""BRAVS Lineup Optimizer — GPU-accelerated optimal lineup construction.

Modules:
    data_builder   — Training data construction from Lahman + BRAVS
    model          — LineupValueNetwork + SlotInteractionModel (PyTorch)
    optimizer      — GPU-parallelized lineup search (50K candidates in <1s)
    season_optimizer — 162-game playing time allocation
    platoon        — Bayesian platoon split estimation with hierarchical shrinkage
    fatigue        — Continuous fatigue model (age/position/workload-dependent)
    series_optimizer — 3-game series joint optimization (greedy + local search)
    trade_impact   — Trade simulation with marginal lineup value
    backtest       — Historical backtesting across team-seasons
"""

from baseball_metric.lineup_optimizer.optimizer import (
    optimize_lineup,
    select_starters,
    assign_positions,
    LineupConfig,
)
from baseball_metric.lineup_optimizer.season_optimizer import (
    optimize_season,
    compute_positional_surplus,
)
from baseball_metric.lineup_optimizer.trade_impact import (
    simulate_trade,
    find_biggest_upgrade_positions,
    compute_player_marginal_value,
)
from baseball_metric.lineup_optimizer.backtest import (
    backtest_team_season,
    backtest_all_teams,
    backtest_multi_year,
)
from baseball_metric.lineup_optimizer.fatigue import FatigueModel
from baseball_metric.lineup_optimizer.series_optimizer import (
    optimize_series,
    SeriesResult,
    OpposingPitcher,
)

__all__ = [
    "optimize_lineup",
    "select_starters",
    "assign_positions",
    "LineupConfig",
    "optimize_season",
    "compute_positional_surplus",
    "simulate_trade",
    "find_biggest_upgrade_positions",
    "compute_player_marginal_value",
    "backtest_team_season",
    "backtest_all_teams",
    "backtest_multi_year",
    "FatigueModel",
    "optimize_series",
    "SeriesResult",
    "OpposingPitcher",
]
