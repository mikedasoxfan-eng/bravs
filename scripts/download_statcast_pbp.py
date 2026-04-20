"""Download full Statcast pitch-by-pitch data from Baseball Savant.

Writes one parquet per season to data/statcast/pbp/statcast_{year}.parquet.
Each row is one pitch with ~90 columns including:
  - pitch_type, release_speed, release_spin_rate, release_pos_x/y/z
  - plate_x, plate_z, zone, description, events, type
  - hit_location, bb_type, launch_speed, launch_angle
  - estimated_ba_using_speedangle, estimated_woba_using_speedangle
  - woba_value, woba_denom, babip_value, iso_value
  - effective_speed, pitch_number, home_score, away_score
  - at_bat_number, inning, inning_topbot, fielder_2/3/.../9

Uses pybaseball.statcast() which chunks requests to stay under Savant's
25K row per-request limit. Each year takes ~5-20 minutes depending on
rate limiting.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "statcast", "pbp")
os.makedirs(OUT_DIR, exist_ok=True)

# Season date ranges. Statcast launched 2015. Use opening day ~ Nov 1 to catch
# regular season + postseason.
SEASON_RANGES = {
    2015: ("2015-04-05", "2015-11-05"),
    2016: ("2016-04-03", "2016-11-05"),
    2017: ("2017-04-02", "2017-11-05"),
    2018: ("2018-03-29", "2018-11-01"),
    2019: ("2019-03-20", "2019-10-31"),
    2020: ("2020-07-23", "2020-10-28"),  # COVID shortened
    2021: ("2021-04-01", "2021-11-03"),
    2022: ("2022-04-07", "2022-11-06"),
    2023: ("2023-03-30", "2023-11-02"),
    2024: ("2024-03-28", "2024-11-01"),
    2025: ("2025-03-27", "2025-11-02"),
}


def download_year(year: int, start: str, end: str) -> bool:
    out_path = os.path.join(OUT_DIR, f"statcast_{year}.parquet")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10_000_000:
        try:
            existing = pd.read_parquet(out_path, columns=["game_pk"])
            n = len(existing)
            print(f"  [{year}] cached: {n:,} pitches, {os.path.getsize(out_path)/1e6:.0f} MB")
            return True
        except Exception:
            pass

    from pybaseball import statcast
    print(f"  [{year}] downloading {start} -> {end} ...", flush=True)
    t0 = time.time()
    try:
        df = statcast(start_dt=start, end_dt=end, verbose=False)
    except Exception as e:
        print(f"    FAILED: {e}")
        return False

    if df is None or len(df) == 0:
        print(f"    empty")
        return False

    elapsed = time.time() - t0
    # Optimize dtypes a bit before writing
    for col in df.columns:
        if df[col].dtype == "object":
            # Leave strings as-is; parquet handles them
            pass
    df.to_parquet(out_path, index=False, compression="zstd")
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"    OK: {len(df):,} pitches, {size_mb:.0f} MB parquet, {elapsed/60:.1f} min")
    return True


def main() -> int:
    years = list(range(2015, 2026))
    print(f"Downloading Statcast pitch-by-pitch for {years[0]}-{years[-1]}")
    print(f"Output: {OUT_DIR}")
    print()

    ok = 0
    for year in years:
        start, end = SEASON_RANGES[year]
        if download_year(year, start, end):
            ok += 1

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_mb = 0
    total_rows = 0
    for year in years:
        p = os.path.join(OUT_DIR, f"statcast_{year}.parquet")
        if os.path.exists(p):
            size_mb = os.path.getsize(p) / 1e6
            total_mb += size_mb
            try:
                n = len(pd.read_parquet(p, columns=["game_pk"]))
                total_rows += n
                print(f"  {year}: {n:>8,} pitches  {size_mb:>6.0f} MB")
            except Exception:
                print(f"  {year}: {size_mb:>6.0f} MB (read error)")
    print(f"\n  TOTAL: {total_rows:,} pitches in {total_mb:.0f} MB ({total_mb/1024:.1f} GB)")
    return 0 if ok == len(years) else 1


if __name__ == "__main__":
    sys.exit(main())
