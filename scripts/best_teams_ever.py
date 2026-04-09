"""Greatest teams ever by total roster BRAVS WAR-eq.

Loads data/bravs_all_seasons.csv and for each team-season:
  - Sums the bravs_war_eq of all players on the team
  - Shows the top 25 team-seasons ever
  - For the top 5, shows the full roster breakdown
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import pandas as pd
import numpy as np


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bravs_all_seasons.csv")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "best_teams_ever.log")


def analyze(df):
    """Run the best-teams analysis, returning the report as a string."""
    buf = io.StringIO()

    def p(text=""):
        buf.write(text + "\n")

    # ------------------------------------------------------------------
    # Aggregate by team-season
    # ------------------------------------------------------------------
    team_seasons = (
        df.groupby(["team", "yearID"])
        .agg(
            total_war_eq=("bravs_war_eq", "sum"),
            total_bravs=("bravs", "sum"),
            n_players=("playerID", "nunique"),
            top_player_war=("bravs_war_eq", "max"),
        )
        .reset_index()
        .sort_values("total_war_eq", ascending=False)
    )

    # Find the name of the top player for each team-season
    idx_max = df.groupby(["team", "yearID"])["bravs_war_eq"].idxmax()
    top_players = df.loc[idx_max, ["team", "yearID", "name", "bravs_war_eq"]].rename(
        columns={"name": "best_player", "bravs_war_eq": "best_war_eq"}
    )
    team_seasons = team_seasons.merge(top_players[["team", "yearID", "best_player"]], on=["team", "yearID"], how="left")

    # ------------------------------------------------------------------
    # Top 25 team-seasons
    # ------------------------------------------------------------------
    top25 = team_seasons.head(25)

    p("=" * 90)
    p("  GREATEST TEAMS EVER BY TOTAL ROSTER BRAVS WAR-eq")
    p("=" * 90)
    p()
    p(f"  {'Rank':<6}{'Team':<7}{'Year':>5}{'Total WAR-eq':>14}{'Players':>9}{'Best Player':<24}{'His WAR-eq':>11}")
    p("  " + "-" * 76)

    for i, (_, r) in enumerate(top25.iterrows(), 1):
        p(f"  {i:<6}"
          f"{r['team']:<7}"
          f"{r['yearID']:>5.0f}"
          f"{r['total_war_eq']:>14.1f}"
          f"{r['n_players']:>9}"
          f"  {r['best_player']:<22}"
          f"{r['top_player_war']:>11.1f}")

    # ------------------------------------------------------------------
    # Full roster breakdown for top 5
    # ------------------------------------------------------------------
    top5 = top25.head(5)

    p()
    p("=" * 90)
    p("  FULL ROSTER BREAKDOWN: TOP 5 TEAM-SEASONS")
    p("=" * 90)

    for rank, (_, ts) in enumerate(top5.iterrows(), 1):
        team = ts["team"]
        year = int(ts["yearID"])
        roster = df[(df["team"] == team) & (df["yearID"] == year)].copy()
        roster = roster.sort_values("bravs_war_eq", ascending=False)

        total_war = roster["bravs_war_eq"].sum()
        total_bravs = roster["bravs"].sum()
        n = len(roster)

        p()
        p(f"  #{rank}  {team} {year}   |   Total WAR-eq: {total_war:.1f}   |   Roster: {n} players")
        p(f"  {'':>4}{'Player':<26}{'Pos':<5}{'WAR-eq':>8}{'BRAVS':>8}{'Hit':>7}{'Pit':>7}{'Fld':>7}{'BR':>6}")
        p("  " + "-" * 78)

        # Position players first, then pitchers, each sorted by WAR-eq desc
        hitters = roster[roster["position"] != "P"]
        pitchers = roster[roster["position"] == "P"]

        if not hitters.empty:
            p(f"  {'':>4}--- POSITION PLAYERS ---")
            for _, r in hitters.iterrows():
                p(f"  {'':>4}"
                  f"{r['name']:<26}"
                  f"{r['position']:<5}"
                  f"{r['bravs_war_eq']:>8.1f}"
                  f"{r['bravs']:>8.1f}"
                  f"{r['hitting_runs']:>7.1f}"
                  f"{r['pitching_runs']:>7.1f}"
                  f"{r['fielding_runs']:>7.1f}"
                  f"{r['baserunning_runs']:>6.1f}")

        if not pitchers.empty:
            p(f"  {'':>4}--- PITCHERS ---")
            for _, r in pitchers.iterrows():
                p(f"  {'':>4}"
                  f"{r['name']:<26}"
                  f"{'P':<5}"
                  f"{r['bravs_war_eq']:>8.1f}"
                  f"{r['bravs']:>8.1f}"
                  f"{r['hitting_runs']:>7.1f}"
                  f"{r['pitching_runs']:>7.1f}"
                  f"{r['fielding_runs']:>7.1f}"
                  f"{r['baserunning_runs']:>6.1f}")

        # Summary line for this team
        hit_total = hitters["bravs_war_eq"].sum() if not hitters.empty else 0
        pit_total = pitchers["bravs_war_eq"].sum() if not pitchers.empty else 0
        p(f"  {'':>4}{'':>26}{'':>5}{'--------':>8}")
        p(f"  {'':>4}{'TOTAL':<26}{'':>5}{total_war:>8.1f}")
        p(f"  {'':>4}  Position players: {hit_total:.1f} WAR-eq  |  Pitchers: {pit_total:.1f} WAR-eq")

    # ------------------------------------------------------------------
    # Additional stats
    # ------------------------------------------------------------------
    p()
    p("=" * 90)
    p("  ADDITIONAL TEAM-SEASON STATISTICS")
    p("=" * 90)

    # Most balanced team (smallest gap between hitting and pitching WAR-eq)
    p()
    p("  MOST BALANCED TEAMS (closest hitting/pitching WAR-eq split):")
    p()

    team_splits = []
    for (team, year), grp in df.groupby(["team", "yearID"]):
        hit_war = grp[grp["position"] != "P"]["bravs_war_eq"].sum()
        pit_war = grp[grp["position"] == "P"]["bravs_war_eq"].sum()
        total = hit_war + pit_war
        if total >= 30:  # Only consider reasonably good teams
            team_splits.append({
                "team": team, "year": year,
                "hit_war": hit_war, "pit_war": pit_war,
                "total": total,
                "gap": abs(hit_war - pit_war),
            })

    if team_splits:
        splits_df = pd.DataFrame(team_splits).sort_values("gap")
        p(f"  {'Rank':<6}{'Team':<7}{'Year':>5}{'Hit WAR':>9}{'Pit WAR':>9}{'Total':>8}{'Gap':>7}")
        p("  " + "-" * 51)
        for i, (_, r) in enumerate(splits_df.head(10).iterrows(), 1):
            p(f"  {i:<6}{r['team']:<7}{r['year']:>5.0f}{r['hit_war']:>9.1f}{r['pit_war']:>9.1f}"
              f"{r['total']:>8.1f}{r['gap']:>7.1f}")

    # Teams with the single most dominant player (highest % of team WAR)
    p()
    p("  MOST DOMINANT INDIVIDUAL CONTRIBUTIONS (player % of team WAR):")
    p()

    team_seasons_dom = team_seasons[team_seasons["total_war_eq"] >= 20].copy()
    team_seasons_dom["pct"] = team_seasons_dom["top_player_war"] / team_seasons_dom["total_war_eq"] * 100
    top_dom = team_seasons_dom.nlargest(10, "pct")

    p(f"  {'Rank':<6}{'Team':<7}{'Year':>5}{'Player':<24}{'Player WAR':>11}{'Team WAR':>10}{'Pct':>7}")
    p("  " + "-" * 70)
    for i, (_, r) in enumerate(top_dom.iterrows(), 1):
        p(f"  {i:<6}{r['team']:<7}{r['yearID']:>5.0f}  {r['best_player']:<22}"
          f"{r['top_player_war']:>11.1f}{r['total_war_eq']:>10.1f}{r['pct']:>6.1f}%")

    # Decade breakdown of best teams
    p()
    p("  BEST TEAM PER DECADE:")
    p()
    team_seasons_copy = team_seasons.copy()
    team_seasons_copy["decade"] = team_seasons_copy["yearID"].apply(lambda y: f"{(int(y)//10)*10}s")
    p(f"  {'Decade':<9}{'Team':<7}{'Year':>5}{'WAR-eq':>9}{'Best Player':<24}")
    p("  " + "-" * 54)
    for dec in sorted(team_seasons_copy["decade"].unique()):
        dec_best = team_seasons_copy[team_seasons_copy["decade"] == dec].nlargest(1, "total_war_eq")
        if not dec_best.empty:
            r = dec_best.iloc[0]
            p(f"  {dec:<9}{r['team']:<7}{r['yearID']:>5.0f}{r['total_war_eq']:>9.1f}  {r['best_player']:<22}")

    # Average team WAR by decade
    p()
    p("  AVERAGE TEAM WAR-eq BY DECADE:")
    p()
    decade_avg = (
        team_seasons_copy.groupby("decade")["total_war_eq"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .sort_values("decade")
    )
    p(f"  {'Decade':<9}{'Mean':>8}{'Median':>9}{'Std Dev':>9}{'Teams':>7}")
    p("  " + "-" * 42)
    for _, r in decade_avg.iterrows():
        p(f"  {r['decade']:<9}{r['mean']:>8.1f}{r['median']:>9.1f}{r['std']:>9.1f}{r['count']:>7.0f}")

    p()
    return buf.getvalue()


def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df):,} player-seasons, {df['yearID'].min()}-{df['yearID'].max()}")

    report = analyze(df)
    print(report)

    # Save to logs
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[Saved to {LOG_PATH}]")


if __name__ == "__main__":
    main()
