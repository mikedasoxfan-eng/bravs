"""Generate a full BRAVS player report — MLB + MiLB career, comps, projections.

Usage:
    python scripts/player_report.py "Mike Trout"
    python scripts/player_report.py "Cam Schlittler"
    python scripts/player_report.py "Jordan Lawlar"
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np


def find_player(name: str, mlb: pd.DataFrame, milb: pd.DataFrame, careers: pd.DataFrame):
    """Find player across all datasets."""
    parts = name.strip().split()
    if len(parts) < 2:
        return None, None, None

    last = parts[-1]
    first = parts[0]

    # Search MLB seasons
    mlb_match = mlb[mlb.name.str.contains(last, na=False) & mlb.name.str.contains(first[:3], na=False)]

    # Search MiLB seasons
    milb_match = milb[milb.name.str.contains(last, na=False) & milb.name.str.contains(first[:3], na=False)]

    # Search careers
    career_match = careers[careers.name.str.contains(last, na=False) & careers.name.str.contains(first[:3], na=False)]
    if len(career_match) > 1:
        career_match = career_match.sort_values("career_war_eq", ascending=False)

    return mlb_match, milb_match, career_match


def print_report(name: str):
    print(f"\n{'=' * 70}")
    print(f"  BRAVS PLAYER REPORT: {name}")
    print(f"{'=' * 70}")

    mlb = pd.read_csv("data/bravs_all_seasons.csv")
    milb = pd.read_csv("data/bravs_milb_seasons.csv")
    careers = pd.read_csv("data/bravs_careers.csv")

    mlb_data, milb_data, career_data = find_player(name, mlb, milb, careers)

    # ─── Career Summary ───
    if career_data is not None and len(career_data) > 0:
        c = career_data.iloc[0]
        print(f"\n  MLB CAREER SUMMARY")
        print(f"  {'-' * 50}")
        print(f"  Name:           {c['name']}")
        print(f"  Seasons:        {int(c.seasons)} ({int(c.first_year)}-{int(c.last_year)})")
        print(f"  Games:          {int(c.total_G)}")
        print(f"  PA / IP:        {int(c.total_PA)} / {c.total_IP:.0f}")
        print(f"  Career WAR-eq:  {c.career_war_eq:.1f}")
        print(f"  Peak season:    {c.peak_bravs:.1f}")
        print(f"  Peak 5-year:    {c.peak5_bravs:.1f}")
        print(f"  HOF worthy:     {'Yes' if c.get('hof') else 'No'}")
    else:
        print(f"\n  (No MLB career data found)")

    # ─── MLB Season-by-Season ───
    if mlb_data is not None and len(mlb_data) > 0:
        print(f"\n  MLB SEASONS")
        print(f"  {'-' * 50}")
        mlb_sorted = mlb_data.sort_values("yearID")
        print(f"  {'Year':>4} {'Team':<5} {'Pos':<4} {'G':>4} {'PA':>5} {'WAR':>6} {'Hit':>6} {'Fld':>5} {'Pos':>5}")
        for _, r in mlb_sorted.iterrows():
            print(f"  {int(r.yearID):>4} {str(r.team):<5} {str(r.position):<4} "
                  f"{int(r.G):>4} {int(r.PA):>5} {r.bravs_war_eq:>+6.1f} "
                  f"{r.hitting_runs:>+6.1f} {r.fielding_runs:>+5.1f} {r.positional_runs:>+5.1f}")
        total_war = mlb_sorted.bravs_war_eq.sum()
        print(f"  {'':>4} {'TOTAL':<5} {'':4} {int(mlb_sorted.G.sum()):>4} "
              f"{int(mlb_sorted.PA.sum()):>5} {total_war:>+6.1f}")
    else:
        print(f"\n  (No MLB seasons found)")

    # ─── MiLB Career ───
    if milb_data is not None and len(milb_data) > 0:
        milb_bat = milb_data[milb_data.PA > 0] if "PA" in milb_data.columns else milb_data
        if len(milb_bat) > 0:
            print(f"\n  MiLB SEASONS")
            print(f"  {'-' * 50}")
            milb_sorted = milb_bat.sort_values("yearID")
            print(f"  {'Year':>4} {'Level':<5} {'Team':<6} {'PA':>4} {'wOBA':>6} {'WAR':>6} {'Hit':>6}")
            for _, r in milb_sorted.iterrows():
                woba = r.get("wOBA", 0) or 0
                hit = r.get("hitting_runs", 0) or 0
                print(f"  {int(r.yearID):>4} {str(r.level):<5} {str(r.team):<6} "
                      f"{int(r.PA):>4} {woba:>6.3f} {r.bravs_war_eq:>+6.2f} {hit:>+6.1f}")
            total_milb = milb_sorted.bravs_war_eq.sum()
            print(f"  {'':>4} {'TOTAL':<5} {'':6} {int(milb_sorted.PA.sum()):>4} "
                  f"{'':>6} {total_milb:>+6.2f}")

            # Path through minors
            print(f"\n  DEVELOPMENT PATH")
            print(f"  {'-' * 50}")
            levels_visited = milb_sorted.groupby("level").agg(
                years=("yearID", lambda x: f"{int(x.min())}-{int(x.max())}"),
                total_pa=("PA", "sum"),
                avg_woba=("wOBA", "mean"),
                best_war=("bravs_war_eq", "max"),
            ).reset_index()
            level_order = {"RK": 0, "A-": 1, "A": 2, "A+": 3, "AA": 4, "AAA": 5, "WIN": 6}
            levels_visited["order"] = levels_visited["level"].map(level_order).fillna(-1)
            levels_visited = levels_visited.sort_values("order")
            for _, lv in levels_visited.iterrows():
                print(f"  {lv.level:<5} {lv.years:<12} {int(lv.total_pa):>5} PA  "
                      f"wOBA={lv.avg_woba:.3f}  best WAR={lv.best_war:+.2f}")
    else:
        print(f"\n  (No MiLB data found)")

    # ─── Projection (if active) ───
    if milb_data is not None and len(milb_data) > 0:
        try:
            prospects = pd.read_csv("data/prospect_rankings.csv")
            # Find by name
            parts = name.strip().split()
            last = parts[-1]
            match = prospects[prospects.name.str.contains(last, na=False)]
            if len(match) > 0:
                p = match.sort_values("projected_mlb_war", ascending=False).iloc[0]
                print(f"\n  MLB PROJECTION (from MiLB stats)")
                print(f"  {'-' * 50}")
                print(f"  Projected MLB career WAR: {p.projected_mlb_war:+.1f}")
                print(f"  Based on: {int(p.milb_total_pa)} MiLB PA, "
                      f"{p.milb_avg_woba:.3f} avg wOBA, "
                      f"highest level: {p.milb_highest_level}")
        except Exception:
            pass

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/player_report.py \"Player Name\"")
        sys.exit(1)

    player_name = " ".join(sys.argv[1:])
    print_report(player_name)
