"""Team Chemistry / Roster Continuity Analysis.

Does it matter how much a team's roster turns over year-to-year?
We measure "roster continuity" as the fraction of the current team's
WAR that comes from players who were on the team last year.

Then we check: do high-continuity teams outperform their expected wins
more than high-turnover teams?
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

log = logging.getLogger(__name__)


def compute_roster_continuity(seasons: pd.DataFrame) -> pd.DataFrame:
    """For each team-season, compute what fraction of roster (by WAR) returned
    from the previous season.
    """
    # Build (team, year) -> {playerID: bravs_war_eq}
    team_rosters = {}
    for (team, year), grp in seasons[seasons.PA >= 50].groupby(["team", "yearID"]):
        team_rosters[(team, year)] = dict(zip(grp.playerID, grp.bravs_war_eq))

    continuity_rows = []
    for (team, year), current in team_rosters.items():
        prev = team_rosters.get((team, year - 1))
        if not prev:
            continue

        current_war = sum(current.values())
        if current_war == 0:
            continue

        # WAR from returning players
        returning_war = sum(
            current[pid] for pid in current if pid in prev
        )
        # WAR from new arrivals
        new_war = current_war - returning_war

        # Count of returning players
        returning_count = sum(1 for pid in current if pid in prev)
        new_count = len(current) - returning_count

        # Returning players' previous year WAR
        prev_returning_war = sum(prev[pid] for pid in current if pid in prev)

        continuity_rows.append({
            "team": team,
            "year": year,
            "n_players": len(current),
            "returning_count": returning_count,
            "new_count": new_count,
            "current_war": round(current_war, 1),
            "returning_war": round(returning_war, 1),
            "new_war": round(new_war, 1),
            "prev_returning_war": round(prev_returning_war, 1),
            "continuity_pct": returning_count / max(len(current), 1),
            "war_continuity_pct": returning_war / max(current_war, 0.1) if current_war > 0 else 0,
            # Improvement of returners
            "returner_delta": round(returning_war - prev_returning_war, 1),
        })

    return pd.DataFrame(continuity_rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 72)
    print("  TEAM CHEMISTRY / ROSTER CONTINUITY ANALYSIS")
    print("=" * 72)

    seasons = pd.read_csv("data/bravs_all_seasons.csv")
    teams_csv = pd.read_csv("data/lahman2025/Teams.csv")

    print("\n--- Computing roster continuity ---")
    continuity = compute_roster_continuity(seasons)
    print(f"  Team-seasons with continuity data: {len(continuity)}")

    # Merge with team outcomes for modern era
    modern = continuity[continuity.year >= 2000].copy()
    tc = teams_csv[teams_csv.yearID >= 2000][["yearID", "teamID", "W", "L", "R", "RA"]]
    modern = modern.merge(
        tc.rename(columns={"yearID": "year", "teamID": "team"}),
        on=["team", "year"], how="left"
    )

    # Pythagorean expected wins
    modern["pyth_winpct"] = modern.R ** 1.83 / (
        modern.R ** 1.83 + modern.RA ** 1.83).clip(lower=0.01)
    modern["pyth_exp_W"] = modern.pyth_winpct * (modern.W + modern.L)
    modern["pyth_resid"] = modern.W - modern.pyth_exp_W

    # Previous year's record
    modern = modern.sort_values(["team", "year"])
    modern["prev_W"] = modern.groupby("team").W.shift(1)
    modern["W_change"] = modern.W - modern.prev_W

    valid = modern.dropna(subset=["W", "pyth_resid", "prev_W"]).copy()
    print(f"  Modern team-seasons (2000+): {len(valid)}")

    # ─── Correlation between continuity and success ───
    print("\n--- Continuity vs Performance ---")
    corr_war = valid.war_continuity_pct.corr(valid.current_war)
    corr_pyth = valid.war_continuity_pct.corr(valid.pyth_resid)
    corr_wchg = valid.war_continuity_pct.corr(valid.W_change)

    print(f"  War continuity% vs current team WAR:     r = {corr_war:+.3f}")
    print(f"  War continuity% vs Pythagorean residual: r = {corr_pyth:+.3f}")
    print(f"  War continuity% vs year-over-year W chg: r = {corr_wchg:+.3f}")

    # Bin continuity
    valid["continuity_bin"] = pd.cut(
        valid.war_continuity_pct,
        bins=[0, 0.3, 0.5, 0.7, 0.85, 1.0],
        labels=["<30% (rebuild)", "30-50%", "50-70%", "70-85%", ">85% (stable)"],
    )

    print("\n--- Performance by Continuity Bin ---")
    binned = valid.groupby("continuity_bin", observed=True).agg(
        n=("team", "count"),
        avg_war=("current_war", "mean"),
        avg_W=("W", "mean"),
        avg_pyth_resid=("pyth_resid", "mean"),
        avg_W_change=("W_change", "mean"),
    ).round(2)
    print(binned)

    # ─── Biggest continuity: stable dynasties ───
    print("\n--- Most Continuous Rosters (WAR continuity > 80%) ---")
    stable = valid[valid.war_continuity_pct >= 0.80].sort_values("current_war", ascending=False)
    print(f"  Found {len(stable)} team-seasons")
    for _, r in stable.head(15).iterrows():
        print(f"  {r.team} {int(r.year)}: {r.war_continuity_pct:.0%} continuity, "
              f"{int(r.W)}W, WAR={r.current_war:+.1f}")

    # ─── Biggest rebuilds ───
    print("\n--- Biggest Rebuilds (WAR continuity < 20%) ---")
    rebuilds = valid[valid.war_continuity_pct < 0.20].sort_values("W", ascending=False)
    print(f"  Found {len(rebuilds)} team-seasons")
    for _, r in rebuilds.head(10).iterrows():
        print(f"  {r.team} {int(r.year)}: {r.war_continuity_pct:.0%} continuity, "
              f"{int(r.W)}W (prev {int(r.prev_W)}W), WAR={r.current_war:+.1f}")

    # ─── Does continuity help returners? ───
    # Do returning players improve/decline?
    print("\n--- Do Returning Players Improve? ---")
    valid["avg_returner_delta"] = valid.returner_delta / valid.returning_count.clip(lower=1)
    print(f"  Average returning-player WAR change (current - prev): "
          f"{valid.returner_delta.mean():+.2f} team-level")
    print(f"  (Positive = returning players did BETTER this year)")

    # High vs low continuity
    high = valid[valid.war_continuity_pct >= 0.70]
    low = valid[valid.war_continuity_pct <= 0.40]
    print(f"\n  High-continuity (>70%): avg returner delta = {high.returner_delta.mean():+.2f}")
    print(f"  Low-continuity  (<40%): avg returner delta = {low.returner_delta.mean():+.2f}")

    # ─── Which teams build dynasties? ───
    # Compute 5-year rolling continuity per franchise
    print("\n--- Most Continuous Franchises (2015-2024) ---")
    recent = valid[valid.year >= 2015].groupby("team").agg(
        avg_continuity=("war_continuity_pct", "mean"),
        total_war=("current_war", "sum"),
        total_wins=("W", "sum"),
    ).round(3).sort_values("avg_continuity", ascending=False)
    print(recent.head(15))

    print("\n--- Least Continuous Franchises (most turnover) ---")
    print(recent.tail(15))

    # Save
    valid.to_csv("data/team_continuity.csv", index=False)
    print(f"\n  Saved data/team_continuity.csv ({len(valid)} team-seasons)")
    print("=" * 72)


if __name__ == "__main__":
    main()
