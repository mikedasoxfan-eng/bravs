"""Parse Retrosheet game log files into a rich manager decisions CSV.

Retrosheet game logs have 161 fields per game including:
- Pitchers used (bullpen aggressiveness)
- Extra innings (close-game record)
- Run differential (one-run games)
- Managers for both teams
- Starting lineups

Output: data/retrosheet/game_logs_parsed.csv with one row per (manager, game)
"""

import glob
import os
import csv
import pandas as pd

# Field indices (0-based, from glfields.txt)
F_DATE = 0
F_GAMENUM = 1
F_DOW = 2
F_VIS_TEAM = 3
F_VIS_LG = 4
F_VIS_GAMENUM = 5
F_HOME_TEAM = 6
F_HOME_LG = 7
F_HOME_GAMENUM = 8
F_VIS_RUNS = 9
F_HOME_RUNS = 10
F_LENGTH_OUTS = 11
F_DAYNIGHT = 12
F_PARK_ID = 16
F_ATTEND = 17
F_TIME_MIN = 18

# Visiting offensive stats start at field 21 (22 1-indexed)
F_VIS_AB = 21
F_VIS_SB = 32  # 33 1-indexed (21 + 11)
F_VIS_CS = 33

# Visiting pitching at field 38
F_VIS_PITCHERS_USED = 38
F_VIS_EARNED_RUNS = 39

# Home offensive stats start at field 49 (50 1-indexed)
F_HOME_AB = 49
F_HOME_SB = 60
F_HOME_CS = 61

# Home pitching at 66
F_HOME_PITCHERS_USED = 66
F_HOME_EARNED_RUNS = 67

# Managers
F_VIS_MGR_ID = 89
F_VIS_MGR_NAME = 90
F_HOME_MGR_ID = 91
F_HOME_MGR_NAME = 92

# Starting pitchers
F_VIS_SP_ID = 101
F_VIS_SP_NAME = 102
F_HOME_SP_ID = 103
F_HOME_SP_NAME = 104


def parse_game_log_file(path: str) -> list[dict]:
    """Parse a single game log file and return list of games."""
    games = []
    with open(path, "r", encoding="latin-1") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 110:
                continue

            def gi(i: int, default="") -> str:
                if i < len(row):
                    return row[i].strip('"').strip()
                return default

            def gfloat(i: int, default=0.0) -> float:
                try:
                    v = gi(i)
                    return float(v) if v else default
                except (ValueError, TypeError):
                    return default

            try:
                date_str = gi(F_DATE)
                if not date_str or len(date_str) != 8:
                    continue

                year = int(date_str[:4])
                vis_runs = int(gfloat(F_VIS_RUNS))
                home_runs = int(gfloat(F_HOME_RUNS))
                length_outs = int(gfloat(F_LENGTH_OUTS, 54))

                run_diff = abs(home_runs - vis_runs)
                extra_innings = length_outs > 54
                one_run_game = run_diff == 1
                blowout = run_diff >= 6

                games.append({
                    "date": date_str,
                    "year": year,
                    "vis_team": gi(F_VIS_TEAM),
                    "home_team": gi(F_HOME_TEAM),
                    "vis_runs": vis_runs,
                    "home_runs": home_runs,
                    "run_diff": run_diff,
                    "length_outs": length_outs,
                    "extra_innings": int(extra_innings),
                    "one_run_game": int(one_run_game),
                    "blowout": int(blowout),
                    "vis_pitchers_used": int(gfloat(F_VIS_PITCHERS_USED, 1)),
                    "home_pitchers_used": int(gfloat(F_HOME_PITCHERS_USED, 1)),
                    "vis_sb": int(gfloat(F_VIS_SB)),
                    "vis_cs": int(gfloat(F_VIS_CS)),
                    "home_sb": int(gfloat(F_HOME_SB)),
                    "home_cs": int(gfloat(F_HOME_CS)),
                    "vis_mgr_id": gi(F_VIS_MGR_ID),
                    "vis_mgr_name": gi(F_VIS_MGR_NAME),
                    "home_mgr_id": gi(F_HOME_MGR_ID),
                    "home_mgr_name": gi(F_HOME_MGR_NAME),
                    "vis_sp_id": gi(F_VIS_SP_ID),
                    "home_sp_id": gi(F_HOME_SP_ID),
                    "home_won": int(home_runs > vis_runs),
                })
            except (ValueError, IndexError):
                continue

    return games


def main():
    files = sorted(glob.glob("data/retrosheet/gamelogs/gl*.txt"))
    print(f"Parsing {len(files)} game log files...")

    all_games = []
    for f in files:
        games = parse_game_log_file(f)
        all_games.extend(games)
        print(f"  {os.path.basename(f)}: {len(games)} games")

    df = pd.DataFrame(all_games)
    print(f"\nTotal games parsed: {len(df)}")
    print(f"Year range: {df.year.min()}-{df.year.max()}")
    print(f"Unique managers: {len(set(df.vis_mgr_id) | set(df.home_mgr_id))}")

    df.to_csv("data/retrosheet/game_logs_parsed.csv", index=False)
    print(f"Saved data/retrosheet/game_logs_parsed.csv")

    # Quick sanity check: extra inning rate and one-run rate per year
    print("\nSample yearly stats:")
    yearly = df.groupby("year").agg(
        games=("date", "count"),
        extra_inn_rate=("extra_innings", "mean"),
        one_run_rate=("one_run_game", "mean"),
        avg_vis_pitchers=("vis_pitchers_used", "mean"),
        avg_home_pitchers=("home_pitchers_used", "mean"),
    )
    for yr, row in yearly.iterrows():
        print(f"  {int(yr)}: {int(row.games)} games, "
              f"{row.extra_inn_rate:.1%} extra innings, "
              f"{row.one_run_rate:.1%} one-run, "
              f"{row.avg_home_pitchers:.1f} pit/game (home)")


if __name__ == "__main__":
    main()
