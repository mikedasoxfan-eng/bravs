"""Download MiLB 2026 season-to-date stats from MLB Stats API.

Writes per-level CSVs matching the schema of existing data/milb/batting/
and data/milb/pitching/ files so they plug into compute_milb.py unchanged.

Levels (sportId): 11=AAA, 12=AA, 13=A+, 14=A, 16=Rk (no A- sport id; historical
a- files came from MiLB's short-season A level which was reorganized).
"""
from __future__ import annotations

import os
import sys
import time
import requests
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_BAT = os.path.join(ROOT, "data", "milb", "batting")
OUT_PIT = os.path.join(ROOT, "data", "milb", "pitching")
os.makedirs(OUT_BAT, exist_ok=True)
os.makedirs(OUT_PIT, exist_ok=True)

YEAR = 2026
API = "https://statsapi.mlb.com/api/v1"
UA = {"User-Agent": "Mozilla/5.0 BRAVS/1.0"}

LEVELS = [
    (11, "aaa"),
    (12, "aa"),
    (13, "a+"),
    (14, "a"),
    (16, "rk"),
]

# Column maps: MLB Stats API -> legacy file schema used by compute_milb.py
BAT_COLS = {
    "gamesPlayed": "G",
    "plateAppearances": "batting_PA",
    "atBats": "batting_AB",
    "hits": "batting_H",
    "doubles": "batting_2B",
    "triples": "batting_3B",
    "homeRuns": "batting_HR",
    "rbi": "batting_RBI",
    "stolenBases": "batting_SB",
    "caughtStealing": "batting_CS",
    "baseOnBalls": "batting_BB",
    "intentionalWalks": "batting_IBB",
    "strikeOuts": "batting_SO",
    "totalBases": "batting_TB",
    "sacBunts": "batting_SH",
    "sacFlies": "batting_SF",
    "hitByPitch": "batting_HBP",
    "groundIntoDoublePlay": "batting_GiDP",
    "leftOnBase": "batting_LOB",
    "avg": "batting_AVG",
    "obp": "batting_OBP",
    "slg": "batting_SLG",
    "ops": "batting_OPS",
    "babip": "batting_BABiP",
}

PIT_COLS = {
    "gamesPlayed": "G",
    "gamesStarted": "GS",
    "inningsPitched": "pitching_IP",
    "hits": "pitching_H",
    "runs": "pitching_R",
    "earnedRuns": "pitching_ER",
    "homeRuns": "pitching_HR",
    "baseOnBalls": "pitching_BB",
    "intentionalWalks": "pitching_IBB",
    "strikeOuts": "pitching_SO",
    "hitByPitch": "pitching_HBP",
    "wins": "pitching_W",
    "losses": "pitching_L",
    "saves": "pitching_SV",
    "holds": "pitching_HLD",
    "blownSaves": "pitching_BS",
    "battersFaced": "pitching_BF",
    "era": "pitching_ERA",
    "whip": "pitching_WHIP",
    "avg": "pitching_AVG",
}


def fetch_level(sport_id: int, group: str, limit: int = 2000) -> list[dict]:
    """Fetch all player-season rows for a sport/group from MLB Stats API."""
    rows = []
    offset = 0
    while True:
        url = (
            f"{API}/stats?stats=season&group={group}"
            f"&season={YEAR}&sportId={sport_id}"
            f"&limit={limit}&offset={offset}&playerPool=ALL"
        )
        try:
            r = requests.get(url, headers=UA, timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    error sport={sport_id} offset={offset}: {e}")
            break
        splits = []
        for s in data.get("stats", []):
            splits.extend(s.get("splits", []))
        if not splits:
            break
        rows.extend(splits)
        if len(splits) < limit:
            break
        offset += limit
        time.sleep(0.2)
    return rows


def rows_to_df(rows: list[dict], col_map: dict, group: str) -> pd.DataFrame:
    out = []
    for s in rows:
        stat = s.get("stat", {})
        player = s.get("player", {})
        team = s.get("team", {})
        league = s.get("league", {})
        sport = s.get("sport", {})
        rec = {
            "season": YEAR,
            "team_id": team.get("id"),
            "team_abv": team.get("abbreviation", ""),
            "team_name": team.get("name", ""),
            "team_league_id": league.get("id"),
            "team_league": league.get("name", ""),
            "team_level_id": sport.get("id"),
            "team_level_abv": sport.get("abbreviation", ""),
            "player_id": player.get("id"),
            "player_full_name": player.get("fullName", ""),
            "player_use_name": player.get("fullName", ""),
            "player_position": s.get("position", {}).get("abbreviation", ""),
        }
        for api_k, out_k in col_map.items():
            v = stat.get(api_k)
            if isinstance(v, str) and v.replace(".", "", 1).replace("-", "", 1).isdigit():
                try:
                    v = float(v) if "." in v else int(v)
                except ValueError:
                    pass
            rec[out_k] = v
        out.append(rec)
    return pd.DataFrame(out)


def main() -> int:
    print(f"Downloading MiLB {YEAR} season stats from MLB Stats API")
    totals = {"batting": 0, "pitching": 0}

    for sport_id, level_abv in LEVELS:
        print(f"\n[{level_abv.upper()}] sportId={sport_id}")

        # Batting
        t0 = time.time()
        rows = fetch_level(sport_id, "hitting")
        print(f"  hitting: {len(rows):>5} rows ({time.time()-t0:.1f}s)")
        if rows:
            df = rows_to_df(rows, BAT_COLS, "hitting")
            path = os.path.join(OUT_BAT, f"{YEAR}_{level_abv}_season_batting_stats.csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            totals["batting"] += len(df)
            print(f"  -> {path}")

        # Pitching
        t0 = time.time()
        rows = fetch_level(sport_id, "pitching")
        print(f"  pitching: {len(rows):>5} rows ({time.time()-t0:.1f}s)")
        if rows:
            df = rows_to_df(rows, PIT_COLS, "pitching")
            path = os.path.join(OUT_PIT, f"{YEAR}_{level_abv}_season_pitching_stats.csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            totals["pitching"] += len(df)
            print(f"  -> {path}")
        time.sleep(0.3)

    print(f"\nTotal: {totals['batting']:,} batting rows, {totals['pitching']:,} pitching rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
