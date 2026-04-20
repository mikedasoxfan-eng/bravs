"""Download Baseball Savant season leaderboards for player movement stats.

Tables fetched via the Savant /leaderboard CSV endpoints:
  - sprint_speed       (2015-2025)   running/baserunning athleticism
  - arm_strength       (2020-2025)   OF throwing velocity (tracking era)
  - bat_tracking       (2023-2025)   swing path / bat speed / squared up rate
  - pitch_tempo        (2015-2025)   delivery tempo metrics
  - exit_velocity      (2015-2025)   hitters' EV / HH%
  - pitch_movement     (2015-2025)   pitcher spin/movement by pitch type

All saved under data/statcast/ as {table}_{year}.csv plus a concatenated
{table}_all.csv. Failures are soft: if a year/endpoint returns HTML or 404,
we log and move on.
"""
from __future__ import annotations

import io
import os
import sys
import time
from typing import Callable

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "statcast")
os.makedirs(OUT_DIR, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 BRAVS/1.0"}


def _get_csv(url: str) -> pd.DataFrame | None:
    """GET a Savant CSV endpoint. Returns None if not CSV or HTTP error."""
    try:
        r = requests.get(url, headers=UA, timeout=90)
        r.raise_for_status()
    except Exception as e:
        print(f"    HTTP error: {e}")
        return None
    text = r.content.decode("utf-8-sig", errors="replace")
    if text.lstrip().startswith("<"):  # HTML error page
        print(f"    got HTML instead of CSV")
        return None
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as e:
        print(f"    parse error: {e}")
        return None
    if df.empty:
        print(f"    empty CSV")
        return None
    return df


def sprint_speed(year: int) -> pd.DataFrame | None:
    # Savant sprint speed leaderboard — position=, min_opp=10 filters.
    url = (
        "https://baseballsavant.mlb.com/leaderboard/sprint_speed"
        f"?year={year}&position=&team=&min=10&csv=true"
    )
    return _get_csv(url)


def arm_strength(year: int) -> pd.DataFrame | None:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/arm-strength"
        f"?type=Fielder&year={year}&team=&min=100&csv=true"
    )
    return _get_csv(url)


def bat_tracking(year: int) -> pd.DataFrame | None:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/bat-tracking"
        f"?type=batter&year={year}&csv=true"
    )
    return _get_csv(url)


def pitch_tempo(year: int) -> pd.DataFrame | None:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/pitch-tempo"
        f"?year={year}&team=&min=&csv=true"
    )
    return _get_csv(url)


def exit_velocity(year: int) -> pd.DataFrame | None:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/statcast"
        f"?type=batter&year={year}&position=&team=&min=q&csv=true"
    )
    return _get_csv(url)


def pitch_movement(year: int) -> pd.DataFrame | None:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/pitch-movement"
        f"?year={year}&team=&min=200&pitch_type=ALL&hand=&csv=true"
    )
    return _get_csv(url)


def poptime(year: int) -> pd.DataFrame | None:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/poptime"
        f"?year={year}&team=&min2=5&min3=0&csv=true"
    )
    return _get_csv(url)


def catcher_blocking(year: int) -> pd.DataFrame | None:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/catcher-blocking"
        f"?year={year}&team=&min=100&csv=true"
    )
    return _get_csv(url)


def catcher_throwing(year: int) -> pd.DataFrame | None:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/catcher-throwing"
        f"?year={year}&team=&min=5&csv=true"
    )
    return _get_csv(url)


def expected_pitching(year: int) -> pd.DataFrame | None:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
        f"?type=pitcher&year={year}&team=&min=50&csv=true"
    )
    return _get_csv(url)


def percentile_rankings_batter(year: int) -> pd.DataFrame | None:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/percentile-rankings"
        f"?type=batter&year={year}&team=&csv=true"
    )
    return _get_csv(url)


def percentile_rankings_pitcher(year: int) -> pd.DataFrame | None:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/percentile-rankings"
        f"?type=pitcher&year={year}&team=&csv=true"
    )
    return _get_csv(url)


TABLES: list[tuple[str, Callable[[int], "pd.DataFrame | None"], list[int]]] = [
    ("sprint_speed",   sprint_speed,   list(range(2015, 2026))),
    ("arm_strength",   arm_strength,   list(range(2020, 2026))),
    ("bat_tracking",   bat_tracking,   list(range(2023, 2026))),
    ("pitch_tempo",    pitch_tempo,    list(range(2015, 2026))),
    ("exit_velocity",  exit_velocity,  list(range(2015, 2026))),
    ("pitch_movement", pitch_movement, list(range(2015, 2026))),
    ("poptime",                 poptime,                 list(range(2015, 2026))),
    ("catcher_blocking",        catcher_blocking,        list(range(2018, 2026))),
    ("catcher_throwing",        catcher_throwing,        list(range(2016, 2026))),
    ("expected_pitching",       expected_pitching,       list(range(2015, 2026))),
    ("percentile_rankings_batter",  percentile_rankings_batter,  list(range(2015, 2026))),
    ("percentile_rankings_pitcher", percentile_rankings_pitcher, list(range(2015, 2026))),
]


def main() -> int:
    summary = {}
    for name, fn, years in TABLES:
        print(f"\n{'=' * 60}")
        print(f"  {name}  ({years[0]}-{years[-1]})")
        print("=" * 60)
        frames = []
        for y in years:
            out = os.path.join(OUT_DIR, f"{name}_{y}.csv")
            if os.path.exists(out) and os.path.getsize(out) > 200:
                try:
                    df = pd.read_csv(out)
                    frames.append(df.assign(year=y) if "year" not in df.columns else df)
                    print(f"  [{y}] cached ({len(df):>5} rows)")
                    continue
                except Exception:
                    pass
            print(f"  [{y}] downloading ...", end=" ", flush=True)
            t0 = time.time()
            df = fn(y)
            if df is None:
                print("SKIPPED")
                continue
            if "year" not in df.columns:
                df.insert(0, "year", y)
            df.to_csv(out, index=False, encoding="utf-8-sig")
            frames.append(df)
            print(f"OK  {len(df):>5} rows  ({time.time() - t0:.1f}s)")
            time.sleep(0.3)  # polite rate limit

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined_path = os.path.join(OUT_DIR, f"{name}_all.csv")
            combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
            print(f"  -> {name}_all.csv ({len(combined):,} rows)")
            summary[name] = len(combined)
        else:
            summary[name] = 0

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<20} {v:>8,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
