"""Postseason Manager Performance Model.

Analyzes 767 playoff games (2005-2025) to identify managers who
outperform or underperform their roster quality in October.

Unlike regular season, where luck dominates short series, postseason
success IS the manager's job — bullpen sequencing, lineup matchups,
defensive alignment, and pinch-hit timing matter more in 7-game series
than in 162-game seasons.

Key metrics:
- Series winning percentage vs roster talent
- One-run/extra-inning record in postseason
- Survival rate: how many series won vs expected
- Pythagorean residual in postseason
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

log = logging.getLogger(__name__)


def build_postseason_manager_stats():
    """Aggregate postseason games per manager."""
    games = pd.read_csv("data/retrosheet/postseason_parsed.csv")

    # Build two views (home + away), combine
    home = games[["date", "year", "series_type", "home_team", "home_mgr_id",
                  "home_mgr_name", "home_runs", "vis_runs", "home_won", "run_diff",
                  "extra_innings", "one_run_game", "blowout",
                  "home_pitchers_used", "vis_pitchers_used"]].rename(columns={
        "home_team": "team", "home_mgr_id": "mgr_id",
        "home_mgr_name": "mgr_name", "home_runs": "runs_for",
        "vis_runs": "runs_against",
        "home_pitchers_used": "pitchers_used",
        "vis_pitchers_used": "opp_pitchers_used",
    })
    home["won"] = home.home_won
    home["is_home"] = 1

    away = games[["date", "year", "series_type", "vis_team", "vis_mgr_id",
                  "vis_mgr_name", "vis_runs", "home_runs", "home_won", "run_diff",
                  "extra_innings", "one_run_game", "blowout",
                  "vis_pitchers_used", "home_pitchers_used"]].rename(columns={
        "vis_team": "team", "vis_mgr_id": "mgr_id",
        "vis_mgr_name": "mgr_name", "vis_runs": "runs_for",
        "home_runs": "runs_against",
        "vis_pitchers_used": "pitchers_used",
        "home_pitchers_used": "opp_pitchers_used",
    })
    away["won"] = 1 - away.home_won
    away["is_home"] = 0

    for df in [home, away]:
        df.drop(columns="home_won", inplace=True, errors="ignore")

    mgr_games = pd.concat([home, away], ignore_index=True)
    mgr_games = mgr_games[mgr_games.mgr_id.str.len() > 0].copy()

    return mgr_games


def compute_manager_postseason_value(mgr_games: pd.DataFrame,
                                      regular_season: pd.DataFrame) -> pd.DataFrame:
    """For each manager, compute postseason performance vs roster expectation."""
    # Aggregate per manager across all postseason games
    agg = mgr_games.groupby("mgr_id").agg(
        mgr_name=("mgr_name", "first"),
        games=("date", "count"),
        wins=("won", "sum"),
        runs_for=("runs_for", "sum"),
        runs_against=("runs_against", "sum"),
        one_run_games=("one_run_game", "sum"),
        one_run_wins=("one_run_game", lambda x:
            (mgr_games.loc[x.index].one_run_game * mgr_games.loc[x.index].won).sum()),
        extra_inn_games=("extra_innings", "sum"),
        extra_inn_wins=("extra_innings", lambda x:
            (mgr_games.loc[x.index].extra_innings * mgr_games.loc[x.index].won).sum()),
        blowout_games=("blowout", "sum"),
        blowout_wins=("blowout", lambda x:
            (mgr_games.loc[x.index].blowout * mgr_games.loc[x.index].won).sum()),
        total_pitchers=("pitchers_used", "sum"),
        years_of_postseason=("year", "nunique"),
    ).reset_index()

    agg["winpct"] = agg.wins / agg.games.clip(lower=1)
    agg["one_run_winpct"] = agg.one_run_wins / agg.one_run_games.clip(lower=1)
    agg["extra_inn_winpct"] = agg.extra_inn_wins / agg.extra_inn_games.clip(lower=1)
    agg["blowout_winpct"] = agg.blowout_wins / agg.blowout_games.clip(lower=1)
    agg["bullpen_aggr"] = agg.total_pitchers / agg.games.clip(lower=1)

    # Pythagorean expected postseason winpct
    agg["pyth_winpct"] = agg.runs_for ** 1.83 / (
        agg.runs_for ** 1.83 + agg.runs_against ** 1.83).clip(lower=0.01)
    agg["pyth_expected_W"] = agg.pyth_winpct * agg.games
    agg["pyth_residual"] = agg.wins - agg.pyth_expected_W

    # Close game residual (1-run + extra inn) — the pure manager signal
    agg["close_games"] = agg.one_run_games + agg.extra_inn_games
    agg["close_wins"] = agg.one_run_wins + agg.extra_inn_wins
    agg["close_winpct"] = agg.close_wins / agg.close_games.clip(lower=1)
    agg["close_residual"] = agg.close_wins - 0.5 * agg.close_games

    # Blowout residual — tells us if the team got crushed by better teams
    # (that's roster quality) vs lost close games (that's manager)
    agg["blowout_residual"] = agg.blowout_wins - 0.5 * agg.blowout_games

    return agg


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 72)
    print("  POSTSEASON MANAGER PERFORMANCE MODEL")
    print("  Does the model recognize bad October managers?")
    print("=" * 72)

    mgr_games = build_postseason_manager_stats()
    print(f"\n  Manager-postseason-game rows: {len(mgr_games)}")
    print(f"  Unique managers: {mgr_games.mgr_id.nunique()}")

    agg = compute_manager_postseason_value(mgr_games, None)
    agg = agg[agg.games >= 10].copy()  # at least 10 playoff games

    print(f"\n  Managers with 10+ postseason games: {len(agg)}")

    # Rank by pythagorean residual (wins above R/RA expectation)
    print("\n" + "=" * 72)
    print("  BEST POSTSEASON MANAGERS (by Pythagorean residual)")
    print("=" * 72)
    best = agg.sort_values("pyth_residual", ascending=False)
    print(f'{"#":>3} {"Name":<24} {"Yrs":>4} {"G":>3} {"W-L":<8} {"Win%":>6} '
          f'{"1-Run":>8} {"Pyth+":>7}')
    print("-" * 72)
    for i, (_, r) in enumerate(best.head(20).iterrows()):
        wl = f"{int(r.wins)}-{int(r.games - r.wins)}"
        one_run = f"{int(r.one_run_wins)}/{int(r.one_run_games)}" if r.one_run_games > 0 else "0/0"
        print(f'{i+1:>3} {r["mgr_name"]:<24} {int(r.years_of_postseason):>4} '
              f'{int(r.games):>3} {wl:<8} {r.winpct:>5.1%} '
              f'{one_run:>8} {r.pyth_residual:>+7.1f}')

    print(f"\n--- WORST POSTSEASON MANAGERS ---")
    worst = agg.sort_values("pyth_residual")
    for i, (_, r) in enumerate(worst.head(15).iterrows()):
        wl = f"{int(r.wins)}-{int(r.games - r.wins)}"
        one_run_pct = r.one_run_winpct
        print(f'{i+1:>3} {r["mgr_name"]:<24} {int(r.games):>3}G {wl:<8} '
              f'{r.winpct:>5.1%}  1-run:{one_run_pct:>5.1%}  '
              f'pyth:{r.pyth_residual:>+6.1f}')

    # THE BIG QUESTION: BOONE
    print(f"\n{'=' * 72}")
    print("  🎯 AARON BOONE POSTSEASON BREAKDOWN")
    print(f"{'=' * 72}")
    boone_games = mgr_games[mgr_games.mgr_name.str.contains("Aaron Boone", na=False)]
    if len(boone_games) > 0:
        print(f"\n  Total postseason games: {len(boone_games)}")
        by_year = boone_games.groupby("year").agg(
            games=("date", "count"),
            wins=("won", "sum"),
            rf=("runs_for", "sum"),
            ra=("runs_against", "sum"),
            series=("series_type", lambda x: ",".join(sorted(set(x)))),
            one_run_g=("one_run_game", "sum"),
            one_run_w=("one_run_game", lambda x:
                (boone_games.loc[x.index].one_run_game * boone_games.loc[x.index].won).sum()),
            pit_used=("pitchers_used", "mean"),
        ).reset_index()

        print(f'\n{"Year":>4} {"Series":<12} {"W-L":<7} {"RF":>4} {"RA":>4} '
              f'{"1-Run":>8} {"Pen/G":>6}')
        print("-" * 55)
        total_w = 0
        total_g = 0
        total_rf = 0
        total_ra = 0
        total_1w = 0
        total_1g = 0
        for _, r in by_year.iterrows():
            wl = f"{int(r.wins)}-{int(r.games - r.wins)}"
            one_run = f"{int(r.one_run_w)}/{int(r.one_run_g)}" if r.one_run_g > 0 else "0/0"
            print(f'{int(r.year):>4} {r.series:<12} {wl:<7} {int(r.rf):>4} '
                  f'{int(r.ra):>4} {one_run:>8} {r.pit_used:>6.2f}')
            total_w += r.wins
            total_g += r.games
            total_rf += r.rf
            total_ra += r.ra
            total_1w += r.one_run_w
            total_1g += r.one_run_g
        print("-" * 55)
        print(f'{"TOTAL":>4} {"":<12} {int(total_w)}-{int(total_g-total_w):<5} '
              f'{int(total_rf):>4} {int(total_ra):>4} '
              f'{int(total_1w)}/{int(total_1g)}')

        # Expected winpct
        pyth = total_rf**1.83 / (total_rf**1.83 + total_ra**1.83) if total_rf+total_ra > 0 else 0.5
        exp_w = pyth * total_g

        print(f"\n  OVERALL: {int(total_w)}-{int(total_g - total_w)} ({total_w/total_g:.1%})")
        print(f"  Pythagorean expected: {exp_w:.1f} wins")
        print(f"  Residual: {total_w - exp_w:+.1f} wins vs expected")
        if total_1g > 0:
            one_run_pct = total_1w / total_1g
            print(f"  One-run record: {int(total_1w)}-{int(total_1g - total_1w)} ({one_run_pct:.1%})")

        # Rank him
        boone_row = agg[agg.mgr_name.str.contains("Aaron Boone", na=False)]
        if len(boone_row) > 0:
            boone_rank = (agg.pyth_residual >= boone_row.pyth_residual.iloc[0]).sum()
            print(f"\n  RANK: {boone_rank} of {len(agg)} managers by postseason Pyth residual")
            print(f"  Percentile: {(1 - boone_rank/len(agg))*100:.0f}th")

    # Compare to other managers of elite teams
    print(f"\n\n--- MANAGERS OF TEAMS WITH 3+ PLAYOFF RUNS (adjusting for talent) ---")
    recent = agg[agg.years_of_postseason >= 3].sort_values("pyth_residual", ascending=False)
    print(f'{"Name":<24} {"Yrs":>4} {"G":>3} {"W-L":<8} {"Win%":>6} '
          f'{"1-Run%":>8} {"Pyth+":>7}')
    for _, r in recent.head(20).iterrows():
        wl = f"{int(r.wins)}-{int(r.games - r.wins)}"
        print(f'{r["mgr_name"]:<24} {int(r.years_of_postseason):>4} '
              f'{int(r.games):>3} {wl:<8} {r.winpct:>5.1%} '
              f'{r.one_run_winpct:>7.1%} {r.pyth_residual:>+7.1f}')

    # Save
    agg.to_csv("data/manager_postseason.csv", index=False)
    print(f"\n  Saved data/manager_postseason.csv")
    print("=" * 72)


if __name__ == "__main__":
    main()
