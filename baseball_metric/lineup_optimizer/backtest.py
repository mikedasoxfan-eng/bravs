"""Phase 5: Historical Backtesting — how many runs do teams leave on the table?

For each team-season 2020-2025, computes the optimal lineup and compares
to the team's actual performance. Quantifies the cost of suboptimal decisions.
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np

from baseball_metric.lineup_optimizer.optimizer import optimize_lineup, select_starters
from baseball_metric.lineup_optimizer.season_optimizer import optimize_season, compute_positional_surplus

log = logging.getLogger(__name__)


def backtest_team_season(
    team: str,
    year: int,
    seasons_csv: str = "data/bravs_all_seasons.csv",
) -> dict:
    """Backtest lineup optimization for a single team-season.

    Computes what the optimal lineup would have been and estimates
    how many runs the team left on the table.
    """
    seasons = pd.read_csv(seasons_csv)
    team_data = seasons[(seasons.yearID == year) & (seasons.team == team) & (seasons.PA >= 50)]

    if len(team_data) < 9:
        return {"error": f"Not enough players for {team} {year}"}

    # Build roster
    roster = []
    for _, r in team_data.iterrows():
        roster.append({
            "name": r.get("name", "?"),
            "playerID": r.playerID,
            "position": r.position,
            "hitting_runs": float(r.hitting_runs),
            "baserunning_runs": float(r.baserunning_runs),
            "fielding_runs": float(r.fielding_runs),
            "positional_runs": float(r.positional_runs),
            "aqi_runs": float(r.get("aqi_runs", 0)),
            "HR": int(r.HR),
            "SB": int(r.SB),
            "PA": int(r.PA),
            "G": int(r.G),
            "bravs_war_eq": float(r.bravs_war_eq),
        })

    # Optimize
    optimal = optimize_lineup(roster, n_candidates=30000, top_n=1)
    if not optimal:
        return {"error": "Optimization failed"}

    # Season optimization
    season_plan = optimize_season(roster)

    # Actual team WAR-eq
    actual_war = sum(r["bravs_war_eq"] for r in roster)

    # Optimal lineup WAR-eq (top 9 starters)
    starters = select_starters(roster, 9)
    optimal_war = sum(s.get("bravs_war_eq", 0) for s in starters)

    # Surplus analysis
    surplus = compute_positional_surplus(roster)
    weakest = min(surplus.items(), key=lambda x: x[1]) if surplus else ("?", 0)

    return {
        "team": team,
        "year": year,
        "roster_size": len(roster),
        "actual_team_war": round(actual_war, 1),
        "optimal_lineup_war": round(optimal_war, 1),
        "optimal_lineup_value": round(optimal[0].expected_runs, 1),
        "expected_wins": season_plan["expected_wins"],
        "weakest_position": weakest[0],
        "weakest_surplus": weakest[1],
        "optimal_lineup": optimal[0].explanation,
    }


def backtest_all_teams(year: int) -> pd.DataFrame:
    """Backtest all teams for a given season."""
    seasons = pd.read_csv("data/bravs_all_seasons.csv")
    teams = seasons[seasons.yearID == year].team.unique()

    results = []
    for team in sorted(teams):
        if not team or pd.isna(team):
            continue
        try:
            result = backtest_team_season(team, year)
            if "error" not in result:
                results.append(result)
                log.info("  %s %d: WAR=%.1f, expected wins=%d, weakest=%s (%.1f)",
                         team, year, result["actual_team_war"],
                         result["expected_wins"], result["weakest_position"],
                         result["weakest_surplus"])
        except Exception as e:
            log.warning("  %s %d failed: %s", team, year, e)

    return pd.DataFrame(results)


def backtest_multi_year(years: list[int] = None) -> pd.DataFrame:
    """Backtest across multiple seasons."""
    if years is None:
        years = [2022, 2023, 2024, 2025]

    all_results = []
    for year in years:
        log.info("Backtesting %d...", year)
        yr_results = backtest_all_teams(year)
        all_results.append(yr_results)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
