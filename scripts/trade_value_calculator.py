"""Trade Value Calculator — who's worth the most right now?

Combines BRAVS projections + contract data to compute surplus value
for every active player. Answers: "If you could trade for anyone,
who gives you the most wins per dollar?"
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from baseball_metric.analysis.projections_v2 import (
    build_aging_curves, project_player, compute_trade_value,
)
from baseball_metric.data import lahman


def main():
    seasons = pd.read_csv("data/bravs_all_seasons.csv")
    careers = pd.read_csv("data/bravs_careers.csv")
    people = lahman._people()

    print("Building empirical aging curves from 14,092 careers...")
    curves = build_aging_curves(seasons)
    for group, curve in curves.items():
        peak_retention = curve.get(0, 1.0)
        decline_5yr = curve.get(5, 0.5)
        print(f"  {group}: peak={peak_retention:.2f}, 5yr post-peak={decline_5yr:.2f}")

    print("\n" + "=" * 80)
    print("  TRADE VALUE CALCULATOR — Top Active Players")
    print("=" * 80)

    # Active players: those who played in 2024 or 2025
    active_ids = set(seasons[seasons.yearID >= 2024].playerID.unique())
    active = careers[careers.playerID.isin(active_ids) & (careers.career_war_eq > 10)]

    # Get birth years for age calculation
    birth_years = {}
    for _, p in people.iterrows():
        if pd.notna(p.birthYear):
            birth_years[p.playerID] = int(p.birthYear)

    results = []
    for _, player in active.iterrows():
        pid = player.playerID
        birth = birth_years.get(pid)
        if not birth:
            continue
        age = 2026 - birth

        # Estimate salary (crude: based on career WAR-eq and service time)
        # Real data would come from Spotrac/Cots
        career_war = player.career_war_eq
        service = player.seasons
        if service <= 3:
            salary = 1.0  # pre-arb
        elif service <= 6:
            salary = career_war / service * 1.5  # arb years
        else:
            salary = max(career_war / service * 2.0, 5.0)  # free agent
        salary = round(min(salary, 40.0), 1)  # cap at $40M

        tv = compute_trade_value(pid, seasons, age, salary, curves)
        if tv["remaining_war_eq"] > 0:
            tv["salary_est"] = salary
            results.append(tv)

    results.sort(key=lambda x: x["total_surplus_M"], reverse=True)

    print(f"\n{'Rank':<5}{'Player':<24}{'Age':>4}{'Rem WAR':>8}{'Salary':>8}{'Surplus':>9}")
    print("-" * 60)
    for i, r in enumerate(results[:30], 1):
        print(f"{i:<5}{r['player_name']:<24}{r['current_age']:>4}"
              f"{r['remaining_war_eq']:>8.1f}{r['remaining_salary_M']:>7.1f}M"
              f"{r['total_surplus_M']:>8.1f}M")

    # Categories
    print(f"\n{'=' * 80}")
    print("  BEST VALUE BY CATEGORY")
    print(f"{'=' * 80}")

    # Best young players (age <= 27)
    young = [r for r in results if r["current_age"] <= 27]
    print(f"\n  BEST YOUNG PLAYERS (age <= 27):")
    for r in young[:5]:
        print(f"    {r['player_name']:<24} age {r['current_age']}  surplus: ${r['total_surplus_M']:.0f}M")

    # Best veterans (age >= 33)
    vets = [r for r in results if r["current_age"] >= 33]
    print(f"\n  BEST VETERAN VALUE (age >= 33):")
    for r in vets[:5]:
        print(f"    {r['player_name']:<24} age {r['current_age']}  surplus: ${r['total_surplus_M']:.0f}M")

    # Worst contracts (most negative surplus)
    worst = sorted(results, key=lambda x: x["total_surplus_M"])
    print(f"\n  WORST CONTRACTS (most negative surplus):")
    for r in worst[:5]:
        print(f"    {r['player_name']:<24} age {r['current_age']}  surplus: ${r['total_surplus_M']:.0f}M")

    print(f"\n  Computed trade values for {len(results)} active players")


if __name__ == "__main__":
    main()
