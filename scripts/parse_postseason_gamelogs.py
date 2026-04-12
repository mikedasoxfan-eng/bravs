"""Parse Retrosheet postseason game logs (WC, DS, LCS, WS).

Uses the same field definitions as regular season game logs.
Output: data/retrosheet/postseason_parsed.csv
"""

import glob
import os
import csv
import pandas as pd

F_DATE = 0
F_VIS_TEAM = 3
F_HOME_TEAM = 6
F_VIS_RUNS = 9
F_HOME_RUNS = 10
F_LENGTH_OUTS = 11
F_VIS_PITCHERS_USED = 38
F_HOME_PITCHERS_USED = 66
F_VIS_SB = 32
F_VIS_CS = 33
F_HOME_SB = 60
F_HOME_CS = 61
F_VIS_MGR_ID = 89
F_VIS_MGR_NAME = 90
F_HOME_MGR_ID = 91
F_HOME_MGR_NAME = 92
F_VIS_SP_ID = 101
F_HOME_SP_ID = 103


def parse_file(path: str, series_type: str) -> list[dict]:
    games = []
    with open(path, "r", encoding="latin-1") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 110:
                continue

            def gi(i, default=""):
                return row[i].strip('"').strip() if i < len(row) else default

            def gfloat(i, default=0.0):
                try:
                    v = gi(i)
                    return float(v) if v else default
                except (ValueError, TypeError):
                    return default

            try:
                date = gi(F_DATE)
                if not date or len(date) != 8:
                    continue
                year = int(date[:4])
                if year < 2005:  # modern managers only
                    continue

                vis_r = int(gfloat(F_VIS_RUNS))
                home_r = int(gfloat(F_HOME_RUNS))
                length = int(gfloat(F_LENGTH_OUTS, 54))
                run_diff = abs(home_r - vis_r)

                games.append({
                    "date": date,
                    "year": year,
                    "series_type": series_type,
                    "vis_team": gi(F_VIS_TEAM),
                    "home_team": gi(F_HOME_TEAM),
                    "vis_runs": vis_r,
                    "home_runs": home_r,
                    "run_diff": run_diff,
                    "length_outs": length,
                    "extra_innings": int(length > 54),
                    "one_run_game": int(run_diff == 1),
                    "blowout": int(run_diff >= 5),
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
                    "home_won": int(home_r > vis_r),
                })
            except (ValueError, IndexError):
                continue

    return games


def main():
    files = {
        "WC": "data/retrosheet/postseason/wc.txt",
        "DS": "data/retrosheet/postseason/ds.txt",
        "LCS": "data/retrosheet/postseason/lcs.txt",
        "WS": "data/retrosheet/postseason/ws.txt",
    }

    all_games = []
    for series, path in files.items():
        if not os.path.exists(path):
            continue
        games = parse_file(path, series)
        all_games.extend(games)
        print(f"  {series}: {len(games)} games (2005+)")

    df = pd.DataFrame(all_games)
    print(f"\nTotal postseason games (2005-2025): {len(df)}")
    print(f"Year range: {df.year.min()}-{df.year.max()}")

    df.to_csv("data/retrosheet/postseason_parsed.csv", index=False)
    print(f"Saved data/retrosheet/postseason_parsed.csv")

    # Quick stats
    print(f"\nBreakdown by series:")
    for series in ["WC", "DS", "LCS", "WS"]:
        n = (df.series_type == series).sum()
        print(f"  {series}: {n}")

    print(f"\nExtra innings rate: {df.extra_innings.mean():.1%}")
    print(f"One-run rate: {df.one_run_game.mean():.1%}")
    print(f"Blowout rate (5+ runs): {df.blowout.mean():.1%}")


if __name__ == "__main__":
    main()
