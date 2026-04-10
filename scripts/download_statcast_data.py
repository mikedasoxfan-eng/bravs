"""
Download Statcast expected stats, catcher framing data, and consolidate salary data.

Tasks:
  1. statcast_batter_expected_stats (2015-2025) -> data/statcast/expected_batting_YYYY.csv
  2. statcast_catcher_framing (2015-2025)       -> data/statcast/catcher_framing_YYYY.csv
  3. Salary consolidation check                  -> data/salaries_all.csv
"""

import io
import os
import sys
import time

import pandas as pd
import requests

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT, "data")
STATCAST = os.path.join(DATA, "statcast")

os.makedirs(STATCAST, exist_ok=True)
print(f"Project root : {PROJECT}")
print(f"Statcast dir : {STATCAST}")
print()

# ── helpers ──────────────────────────────────────────────────────────────────
YEARS = list(range(2015, 2026))

summary = {
    "expected_batting": {},
    "catcher_framing": {},
    "salaries": None,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Task 1 – Statcast expected batting stats
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TASK 1: Statcast Expected Batting Stats (2015-2025)")
print("=" * 70)

from pybaseball import statcast_batter_expected_stats

frames_batting = []

for year in YEARS:
    path = os.path.join(STATCAST, f"expected_batting_{year}.csv")
    print(f"  [{year}] downloading ...", end=" ", flush=True)
    try:
        t0 = time.time()
        df = statcast_batter_expected_stats(year, minPA=50)
        elapsed = time.time() - t0
        df.to_csv(path, index=False)
        frames_batting.append(df)
        summary["expected_batting"][year] = len(df)
        print(f"OK  {len(df):>5} rows  ({elapsed:.1f}s)")
    except Exception as e:
        print(f"FAILED  ({e})")
        summary["expected_batting"][year] = "FAILED"

if frames_batting:
    all_batting = pd.concat(frames_batting, ignore_index=True)
    all_path = os.path.join(STATCAST, "expected_batting_all.csv")
    all_batting.to_csv(all_path, index=False)
    print(f"\n  Combined file : {all_path}")
    print(f"  Total rows    : {len(all_batting):,}")
    print(f"  Columns       : {list(all_batting.columns)}")
else:
    print("\n  No batting data was downloaded.")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# Task 2 – Catcher Framing
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TASK 2: Statcast Catcher Framing (2015-2025)")
print("=" * 70)

from pybaseball import statcast_catcher_framing


def fetch_catcher_framing_fallback(year, min_called_p=100):
    """Fallback downloader using the newer Baseball Savant endpoint
    (/leaderboard/catcher-framing) which works for 2018+ seasons where
    the old /catcher_framing endpoint returns HTML instead of CSV."""
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/catcher-framing"
        f"?year={year}&team=&min={min_called_p}&csv=true"
    )
    res = requests.get(url, timeout=60)
    res.raise_for_status()
    text = res.content.decode("utf-8-sig")
    df = pd.read_csv(io.StringIO(text))
    # Add year column if not present
    if "year" not in [c.lower() for c in df.columns]:
        df.insert(0, "year", year)
    return df


frames_framing = []

for year in YEARS:
    path = os.path.join(STATCAST, f"catcher_framing_{year}.csv")
    print(f"  [{year}] downloading ...", end=" ", flush=True)
    try:
        t0 = time.time()
        df = statcast_catcher_framing(year, min_called_p=100)
        elapsed = time.time() - t0
        df.to_csv(path, index=False)
        frames_framing.append(df)
        summary["catcher_framing"][year] = len(df)
        print(f"OK  {len(df):>5} rows  ({elapsed:.1f}s)  [pybaseball]")
    except Exception:
        # Fallback to direct download with the newer URL (retries on 502)
        succeeded = False
        for attempt in range(3):
            try:
                if attempt > 0:
                    time.sleep(3)
                df = fetch_catcher_framing_fallback(year, min_called_p=100)
                elapsed = time.time() - t0
                df.to_csv(path, index=False)
                frames_framing.append(df)
                summary["catcher_framing"][year] = len(df)
                print(f"OK  {len(df):>5} rows  ({elapsed:.1f}s)  [fallback]")
                succeeded = True
                break
            except Exception as e2:
                if attempt < 2:
                    print(f"retry...", end=" ", flush=True)
                else:
                    print(f"FAILED  ({e2})")
                    summary["catcher_framing"][year] = "FAILED"

if frames_framing:
    all_framing = pd.concat(frames_framing, ignore_index=True)
    all_path = os.path.join(STATCAST, "catcher_framing_all.csv")
    all_framing.to_csv(all_path, index=False)
    print(f"\n  Combined file : {all_path}")
    print(f"  Total rows    : {len(all_framing):,}")
    print(f"  Columns       : {list(all_framing.columns)}")
else:
    print("\n  No framing data was downloaded.")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# Task 3 – Salary data
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TASK 3: Salary Data Consolidation")
print("=" * 70)

lahman_salary = os.path.join(DATA, "lahman2025", "Salaries.csv")
baseball_main_dir = os.path.join(DATA, "baseball-main", "data", "salaries")
salaries_all_path = os.path.join(DATA, "salaries_all.csv")

# Check Lahman Salaries.csv
if os.path.isfile(lahman_salary):
    lahman_df = pd.read_csv(lahman_salary)
    year_range = f"{lahman_df['yearID'].min()}-{lahman_df['yearID'].max()}"
    print(f"  Lahman Salaries.csv exists: {len(lahman_df):,} rows, years {year_range}")
    print(f"  Columns: {list(lahman_df.columns)}")
else:
    lahman_df = None
    print("  Lahman Salaries.csv NOT found.")

# Check baseball-main salary CSVs
bm_frames = []
if os.path.isdir(baseball_main_dir):
    csv_files = sorted([f for f in os.listdir(baseball_main_dir) if f.endswith(".csv") and f != "summary.csv"])
    print(f"\n  baseball-main/data/salaries/ has {len(csv_files)} CSV files")
    for fname in csv_files:
        fpath = os.path.join(baseball_main_dir, fname)
        try:
            df = pd.read_csv(fpath)
            bm_frames.append(df)
        except Exception as e:
            print(f"    WARNING: could not read {fname}: {e}")
    if bm_frames:
        bm_all = pd.concat(bm_frames, ignore_index=True)
        year_range = f"{bm_all['Year'].min()}-{bm_all['Year'].max()}"
        print(f"  Combined baseball-main salaries: {len(bm_all):,} rows, years {year_range}")
        print(f"  Columns: {list(bm_all.columns)}")

# Consolidate
# Lahman covers up through ~2016. baseball-main covers 2000-2025 with Player names.
# We'll save both sources into salaries_all.csv, preferring baseball-main for its
# broader and more up-to-date coverage.
if bm_frames:
    bm_all.to_csv(salaries_all_path, index=False)
    print(f"\n  Saved consolidated salaries -> {salaries_all_path}")
    print(f"  Total rows: {len(bm_all):,}")
    summary["salaries"] = f"{len(bm_all):,} rows from baseball-main (2000-2025)"
elif lahman_df is not None:
    lahman_df.to_csv(salaries_all_path, index=False)
    print(f"\n  Saved Lahman salaries -> {salaries_all_path}")
    print(f"  Total rows: {len(lahman_df):,}")
    summary["salaries"] = f"{len(lahman_df):,} rows from Lahman"
else:
    print("\n  No salary data found anywhere.")
    summary["salaries"] = "NONE"

print()

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print("\n  Expected Batting Stats:")
total_bat = 0
for y in YEARS:
    v = summary["expected_batting"].get(y, "N/A")
    status = f"{v} rows" if isinstance(v, int) else v
    if isinstance(v, int):
        total_bat += v
    print(f"    {y}: {status}")
print(f"    TOTAL: {total_bat:,} rows")

print("\n  Catcher Framing:")
total_frame = 0
for y in YEARS:
    v = summary["catcher_framing"].get(y, "N/A")
    status = f"{v} rows" if isinstance(v, int) else v
    if isinstance(v, int):
        total_frame += v
    print(f"    {y}: {status}")
print(f"    TOTAL: {total_frame:,} rows")

print(f"\n  Salaries: {summary['salaries']}")

print("\nDone.")
