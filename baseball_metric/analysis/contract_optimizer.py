"""Free Agent Contract Optimizer — recommends optimal signings for each team.

Given a team's current roster, budget, and needs (from positional surplus),
identifies the best free agent targets and projects contract values.
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

DOLLAR_PER_WAR = 9_500_000  # 2026 market rate


def get_team_needs(team: str, year: int = 2025) -> dict:
    """Identify team's positional needs from BRAVS surplus analysis."""
    seasons = pd.read_csv("data/bravs_all_seasons.csv")
    td = seasons[(seasons.yearID == year) & (seasons.team == team) & (seasons.PA >= 100)]

    league_avg = {"C": 1.5, "1B": 2.0, "2B": 2.5, "3B": 2.5, "SS": 3.0,
                  "LF": 2.0, "CF": 2.5, "RF": 2.0, "DH": 1.5}

    needs = {}
    for pos, avg in league_avg.items():
        pos_players = td[td.position == pos]
        best = pos_players.bravs_war_eq.max() if len(pos_players) > 0 else 0
        surplus = best - avg
        if surplus < 0:
            needs[pos] = round(surplus, 1)

    return dict(sorted(needs.items(), key=lambda x: x[1]))


def find_free_agent_targets(needs: dict, budget_millions: float = 50.0) -> list[dict]:
    """Find the best free agent targets for a team's needs."""
    projections = pd.read_csv("data/projections_2026.csv")

    targets = []
    for pos, deficit in needs.items():
        pos_players = projections[
            (projections.position == pos) &
            (projections.projected_war >= 1.0)
        ].sort_values("projected_war", ascending=False)

        for _, p in pos_players.head(5).iterrows():
            # Estimate contract
            age = int(p.get("age_2026", 28))
            war = p.projected_war
            years = max(1, min(7, 35 - age))  # contract length estimate
            aav = war * DOLLAR_PER_WAR * 0.6  # FA premium discount
            total = aav * years

            if aav / 1e6 <= budget_millions * 0.4:  # no single signing > 40% of budget
                targets.append({
                    "name": p["name"],
                    "position": pos,
                    "team": p.get("team", "?"),
                    "age": age,
                    "projected_war": round(war, 1),
                    "deficit_filled": round(-deficit, 1),
                    "est_aav": round(aav / 1e6, 1),
                    "est_years": years,
                    "est_total": round(total / 1e6, 1),
                })

    targets.sort(key=lambda x: x["projected_war"], reverse=True)
    return targets


def optimize_fa_spending(team: str, budget_millions: float = 100.0) -> dict:
    """Recommend optimal FA signings for a team."""
    needs = get_team_needs(team)

    if not needs:
        return {"team": team, "budget": budget_millions, "needs": {},
                "recommendations": [], "message": "No significant needs found"}

    targets = find_free_agent_targets(needs, budget_millions)

    # Greedy: fill biggest needs first
    recommendations = []
    spent = 0
    filled_positions = set()

    for target in targets:
        if target["position"] in filled_positions:
            continue
        if spent + target["est_aav"] > budget_millions:
            continue

        recommendations.append(target)
        spent += target["est_aav"]
        filled_positions.add(target["position"])

    total_war_added = sum(r["projected_war"] for r in recommendations)

    return {
        "team": team,
        "budget": budget_millions,
        "spent": round(spent, 1),
        "remaining": round(budget_millions - spent, 1),
        "needs": needs,
        "recommendations": recommendations,
        "total_war_added": round(total_war_added, 1),
    }


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  FREE AGENT CONTRACT OPTIMIZER")
    print("=" * 60)

    for team, budget in [("NYA", 80), ("BAL", 60), ("SEA", 50), ("CHA", 40), ("LAN", 100)]:
        result = optimize_fa_spending(team, budget)
        print(f"\n  {team} (${budget}M budget):")
        print(f"  Needs: {result['needs']}")
        for r in result["recommendations"]:
            print(f"    Sign {r['name']:<22} {r['position']:<4} "
                  f"proj {r['projected_war']:+.1f} WAR | "
                  f"${r['est_aav']:.0f}M/yr x {r['est_years']}yr")
        print(f"  Total: ${result['spent']:.0f}M spent, {result['total_war_added']:+.1f} WAR added")

    print("\n" + "=" * 60)
