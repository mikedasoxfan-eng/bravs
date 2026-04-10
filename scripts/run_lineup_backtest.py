"""Run comprehensive lineup optimizer backtesting and validation."""

import sys, os, logging, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

import pandas as pd
import numpy as np

from baseball_metric.lineup_optimizer.optimizer import optimize_lineup, select_starters
from baseball_metric.lineup_optimizer.season_optimizer import optimize_season, compute_positional_surplus
from baseball_metric.lineup_optimizer.fatigue import FatigueModel
from baseball_metric.lineup_optimizer.backtest import backtest_team_season


def build_roster(team_data):
    """Convert DataFrame rows to roster dicts."""
    roster = []
    for _, r in team_data.iterrows():
        roster.append({
            "name": r.get("name", "?"),
            "playerID": r.playerID,
            "position": r.position,
            "hitting_runs": float(r.hitting_runs),
            "baserunning_runs": float(r.baserunning_runs),
            "fielding_runs": float(r.fielding_runs),
            "positional_runs": float(r.get("positional_runs", 0)),
            "aqi_runs": float(r.get("aqi_runs", 0)),
            "HR": int(r.HR),
            "SB": int(r.SB),
            "PA": int(r.PA),
            "G": int(r.G),
            "bravs_war_eq": float(r.bravs_war_eq),
        })
    return roster


def main():
    print("=" * 70)
    print("  BRAVS LINEUP OPTIMIZER — Full Backtesting & Validation")
    print("=" * 70)

    seasons = pd.read_csv("data/bravs_all_seasons.csv")
    teams_csv = pd.read_csv("data/lahman2025/Teams.csv")

    fatigue_model = FatigueModel()

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Backtest 2022-2025, all teams
    # ═══════════════════════════════════════════════════════════════
    print("\n--- Phase 1: Historical Backtesting (2022-2025) ---")
    all_results = []

    for year in [2022, 2023, 2024, 2025]:
        teams = seasons[seasons.yearID == year].team.dropna().unique()
        print(f"\n  {year}: {len(teams)} teams")

        for team in sorted(teams):
            team_data = seasons[(seasons.yearID == year) & (seasons.team == team) & (seasons.PA >= 50)]
            if len(team_data) < 9:
                continue

            roster = build_roster(team_data)

            t0 = time.perf_counter()
            try:
                # Single-game optimizer
                results = optimize_lineup(roster, n_candidates=20000, top_n=3)
                if not results:
                    continue

                # Season optimizer
                season_plan = optimize_season(roster)

                # Positional surplus
                surplus = compute_positional_surplus(roster)
                weakest = min(surplus.items(), key=lambda x: x[1]) if surplus else ("?", 0)
                strongest = max(surplus.items(), key=lambda x: x[1]) if surplus else ("?", 0)

                actual_war = sum(r["bravs_war_eq"] for r in roster)
                starters = select_starters(roster, 9)
                starter_war = sum(s.get("bravs_war_eq", 0) for s in starters)

                # Get actual team record
                team_record = teams_csv[(teams_csv.yearID == year) & (teams_csv.teamID == team)]
                actual_wins = int(team_record.iloc[0].W) if len(team_record) > 0 else 0
                actual_runs = int(team_record.iloc[0].R) if len(team_record) > 0 else 0

                elapsed = time.perf_counter() - t0

                row = {
                    "team": team,
                    "year": year,
                    "roster_size": len(roster),
                    "actual_team_war": round(actual_war, 1),
                    "starter_war": round(starter_war, 1),
                    "optimal_lineup_value": round(results[0].expected_runs, 1),
                    "lineup_spread": round(results[0].expected_runs - results[-1].expected_runs, 2),
                    "expected_wins": season_plan["expected_wins"],
                    "actual_wins": actual_wins,
                    "actual_runs": actual_runs,
                    "weakest_pos": weakest[0],
                    "weakest_surplus": weakest[1],
                    "strongest_pos": strongest[0],
                    "strongest_surplus": strongest[1],
                    "elapsed_s": round(elapsed, 2),
                }
                all_results.append(row)

                print(f"    {team}: WAR={actual_war:.1f}, OptVal={results[0].expected_runs:.1f}, "
                      f"ExpW={season_plan['expected_wins']:.0f} vs ActW={actual_wins}, "
                      f"weak={weakest[0]} ({weakest[1]:+.1f}), {elapsed:.2f}s")

            except Exception as e:
                print(f"    {team}: FAILED - {e}")

    df = pd.DataFrame(all_results)

    if len(df) > 0:
        # Save results
        os.makedirs("data/lineup_optimizer", exist_ok=True)
        df.to_csv("data/lineup_optimizer/backtest_results.csv", index=False)

        # ═══════════════════════════════════════════════════════════════
        # Phase 2: Accuracy Metrics
        # ═══════════════════════════════════════════════════════════════
        print("\n--- Phase 2: Accuracy Metrics ---")

        # Expected wins vs actual wins correlation
        valid = df[(df.expected_wins > 0) & (df.actual_wins > 0)]
        if len(valid) > 10:
            corr = valid["expected_wins"].corr(valid["actual_wins"])
            rmse = np.sqrt(((valid["expected_wins"] - valid["actual_wins"]) ** 2).mean())
            mae = (valid["expected_wins"] - valid["actual_wins"]).abs().mean()
            print(f"  Expected vs Actual Wins:")
            print(f"    Correlation:  r = {corr:.3f}")
            print(f"    RMSE:         {rmse:.1f} wins")
            print(f"    MAE:          {mae:.1f} wins")
            print(f"    Samples:      {len(valid)}")

        # Lineup value vs actual runs correlation
        if len(valid) > 10:
            corr_runs = valid["optimal_lineup_value"].corr(valid["actual_runs"])
            print(f"\n  Lineup Value vs Actual Runs Scored:")
            print(f"    Correlation:  r = {corr_runs:.3f}")

        # ═══════════════════════════════════════════════════════════════
        # Phase 3: Known Case Validation
        # ═══════════════════════════════════════════════════════════════
        print("\n--- Phase 3: Known Case Validation ---")

        # 2023 Rangers (WS champions) should have strong lineup
        tex_23 = df[(df.team == "TEX") & (df.year == 2023)]
        if len(tex_23) > 0:
            r = tex_23.iloc[0]
            print(f"  2023 TEX (WS champs): WAR={r.actual_team_war}, ExpW={r.expected_wins}, "
                  f"weak={r.weakest_pos} ({r.weakest_surplus:+.1f})")

        # 2022 Astros (WS champions)
        hou_22 = df[(df.team == "HOU") & (df.year == 2022)]
        if len(hou_22) > 0:
            r = hou_22.iloc[0]
            print(f"  2022 HOU (WS champs): WAR={r.actual_team_war}, ExpW={r.expected_wins}, "
                  f"weak={r.weakest_pos} ({r.weakest_surplus:+.1f})")

        # 2024 Dodgers (WS champions)
        lad_24 = df[(df.team == "LAN") & (df.year == 2024)]
        if len(lad_24) > 0:
            r = lad_24.iloc[0]
            print(f"  2024 LAN (WS champs): WAR={r.actual_team_war}, ExpW={r.expected_wins}, "
                  f"weak={r.weakest_pos} ({r.weakest_surplus:+.1f})")

        # ═══════════════════════════════════════════════════════════════
        # Phase 4: Fatigue Model Validation
        # ═══════════════════════════════════════════════════════════════
        print("\n--- Phase 4: Fatigue Model Validation ---")

        # Check fatigue model produces sensible outputs
        fm = FatigueModel()
        test_cases = [
            ("Fresh (1 off-day)", 5, 10, 22, 28, "SS"),
            ("Tired (7 straight)", 7, 13, 27, 28, "SS"),
            ("Exhausted catcher", 7, 14, 30, 34, "C"),
            ("Rested DH", 3, 8, 18, 28, "DH"),
            ("Young speedster", 6, 12, 25, 23, "CF"),
            ("Aging slugger", 6, 13, 27, 36, "1B"),
        ]
        for label, g7, g14, g30, age, pos in test_cases:
            factor = fm.compute_fatigue_factor(g7, g14, g30, age, pos)
            print(f"  {label:24s}: factor={factor:.4f}")

        # ═══════════════════════════════════════════════════════════════
        # Phase 5: Summary Statistics
        # ═══════════════════════════════════════════════════════════════
        print("\n--- Phase 5: Summary Statistics ---")

        for year in sorted(df.year.unique()):
            yr = df[df.year == year]
            print(f"\n  {year} ({len(yr)} teams):")
            print(f"    Avg team WAR:     {yr.actual_team_war.mean():.1f}")
            print(f"    Avg expected wins: {yr.expected_wins.mean():.0f}")
            print(f"    Avg actual wins:   {yr.actual_wins.mean():.0f}")
            print(f"    Avg lineup value:  {yr.optimal_lineup_value.mean():.1f}")
            print(f"    Avg lineup spread: {yr.lineup_spread.mean():.2f}")

            # Most common weakest position
            weak_counts = yr.weakest_pos.value_counts()
            print(f"    Most common weakness: {weak_counts.index[0]} ({weak_counts.iloc[0]}/{len(yr)} teams)")

        # Overall
        print(f"\n  OVERALL ({len(df)} team-seasons):")
        print(f"    Avg optimizer time:  {df.elapsed_s.mean():.2f}s per team")
        print(f"    Total compute time:  {df.elapsed_s.sum():.1f}s")
        print(f"    Wins prediction:     RMSE={rmse:.1f}, r={corr:.3f}")

        # ═══════════════════════════════════════════════════════════════
        # Phase 6: Best & Worst Teams
        # ═══════════════════════════════════════════════════════════════
        print("\n--- Phase 6: Best & Worst Lineup Values ---")

        # Best lineups
        top = df.nlargest(10, "optimal_lineup_value")
        print("\n  Top 10 Lineup Values (2022-2025):")
        for _, r in top.iterrows():
            print(f"    {r.team} {r.year}: value={r.optimal_lineup_value:.1f}, "
                  f"WAR={r.actual_team_war:.1f}, W={r.actual_wins}")

        # Worst
        bottom = df.nsmallest(10, "optimal_lineup_value")
        print("\n  Bottom 10 Lineup Values (2022-2025):")
        for _, r in bottom.iterrows():
            print(f"    {r.team} {r.year}: value={r.optimal_lineup_value:.1f}, "
                  f"WAR={r.actual_team_war:.1f}, W={r.actual_wins}")

        # Biggest expected vs actual wins gaps
        valid["win_gap"] = valid["expected_wins"] - valid["actual_wins"]
        print("\n  Biggest Overperformers (won more than expected):")
        for _, r in valid.nsmallest(5, "win_gap").iterrows():
            print(f"    {r.team} {r.year}: ExpW={r.expected_wins:.0f}, ActW={r.actual_wins}, "
                  f"gap={r.win_gap:+.0f}")

        print("\n  Biggest Underperformers (won fewer than expected):")
        for _, r in valid.nlargest(5, "win_gap").iterrows():
            print(f"    {r.team} {r.year}: ExpW={r.expected_wins:.0f}, ActW={r.actual_wins}, "
                  f"gap={r.win_gap:+.0f}")

    print("\n" + "=" * 70)
    print("  Backtesting complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
