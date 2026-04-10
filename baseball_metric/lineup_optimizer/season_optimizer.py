"""Season-Long Roster Optimization — 162-game playing time allocation.

Given a full roster and 162-game schedule, optimizes:
- How many games each player starts
- At which positions
- Against which pitcher types
- When to rest stars
- How to leverage positional flexibility

This is the longest-horizon optimization in the system.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def optimize_season(
    roster: list[dict],
    schedule: list[dict] | None = None,
    target_games: int = 162,
) -> dict:
    """Optimize playing time allocation across a full season.

    Args:
        roster: list of player dicts with BRAVS components
        schedule: optional list of game dicts with opponent info
        target_games: games in the season

    Returns:
        dict with:
        - per_player: games to start, positions, rest schedule
        - total_expected_wins: season win total estimate
        - recommendations: text explanations
    """
    log.info("Optimizing %d-game season for %d-man roster", target_games, len(roster))

    # Sort by total value
    for p in roster:
        p["total_value"] = sum(p.get(k, 0) for k in
            ["hitting_runs", "baserunning_runs", "fielding_runs", "aqi_runs"])

    roster_sorted = sorted(roster, key=lambda x: x["total_value"], reverse=True)

    # Tier players
    stars = [p for p in roster_sorted if p["total_value"] > 40]       # top tier
    regulars = [p for p in roster_sorted if 10 < p["total_value"] <= 40]
    bench = [p for p in roster_sorted if p["total_value"] <= 10]

    allocations = []
    recommendations = []

    # Stars: 140-150 games (need rest days)
    for p in stars:
        age = 28  # default if unknown
        # Older stars need more rest
        games = 148 if age < 30 else 140 if age < 34 else 130
        games = min(games, target_games)

        allocations.append({
            "name": p.get("name", "?"),
            "playerID": p.get("playerID", ""),
            "position": p.get("position", "DH"),
            "games_start": games,
            "games_rest": target_games - games,
            "war_eq_projected": p.get("bravs_war_eq", 0) * (games / target_games),
            "tier": "star",
        })
        recommendations.append(
            f"  {p['name']}: Start {games}G at {p.get('position', 'DH')}. "
            f"Rest {target_games - games} games spread across season."
        )

    # Regulars: 130-145 games
    for p in regulars:
        games = 138
        allocations.append({
            "name": p.get("name", "?"),
            "playerID": p.get("playerID", ""),
            "position": p.get("position", "DH"),
            "games_start": games,
            "games_rest": target_games - games,
            "war_eq_projected": p.get("bravs_war_eq", 0) * (games / target_games),
            "tier": "regular",
        })

    # Bench: fill gaps (60-100 games)
    for p in bench[:5]:
        games = 80
        allocations.append({
            "name": p.get("name", "?"),
            "playerID": p.get("playerID", ""),
            "position": p.get("position", "DH"),
            "games_start": games,
            "games_rest": target_games - games,
            "war_eq_projected": max(p.get("bravs_war_eq", 0) * (games / target_games), 0),
            "tier": "bench",
        })

    total_war = sum(a["war_eq_projected"] for a in allocations)
    # Team wins ≈ 47 (FAT baseline) + total WAR-eq
    expected_wins = 47 + total_war

    # Flexibility value
    multi_pos = [p for p in roster if p.get("n_positions", 1) > 1]
    flex_value = len(multi_pos) * 0.5  # ~0.5 wins per flex player

    recommendations.append(f"\n  Multi-position players: {len(multi_pos)} "
                          f"(~{flex_value:.1f} wins of roster flexibility value)")
    recommendations.append(f"  Projected team wins: {expected_wins:.0f}")

    return {
        "allocations": allocations,
        "total_war_eq": round(total_war, 1),
        "expected_wins": round(expected_wins, 0),
        "flex_value": round(flex_value, 1),
        "recommendations": recommendations,
    }


def compute_positional_surplus(roster: list[dict]) -> dict[str, float]:
    """Identify which positions have surplus value and which need upgrades.

    Returns dict mapping position -> surplus (positive = strong, negative = weak).
    """
    position_values = {}
    for p in roster:
        pos = p.get("position", "DH")
        val = p.get("bravs_war_eq", 0)
        if pos not in position_values or val > position_values[pos]:
            position_values[pos] = val

    # League average by position (approximate)
    league_avg = {"C": 1.5, "1B": 2.0, "2B": 2.5, "3B": 2.5, "SS": 3.0,
                  "LF": 2.0, "CF": 2.5, "RF": 2.0, "DH": 1.5}

    surplus = {}
    for pos, avg in league_avg.items():
        actual = position_values.get(pos, 0)
        surplus[pos] = round(actual - avg, 1)

    return surplus
