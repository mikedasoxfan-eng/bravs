"""Download Retrosheet annual event file zips (2015-2024) and extract them.

Retrosheet publishes per-season event archives at:
    https://www.retrosheet.org/events/{year}eve.zip

Each zip contains .EVA/.EVN play-by-play files, .ROS rosters, and a TEAM file.
pybaseball's retrosheet module is broken, so we use urllib directly.
"""
from __future__ import annotations

import io
import os
import sys
import time
import urllib.request
import urllib.error
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(ROOT, "data", "retrosheet", "events")

YEARS = list(range(2000, 2026))  # 2000..2025 inclusive (2025 may not be published yet)


def download_year(year: int) -> bool:
    out_dir = os.path.join(OUT_ROOT, str(year))
    os.makedirs(out_dir, exist_ok=True)

    # Skip if already extracted (at least one .EVA/.EVN file present)
    existing = [f for f in os.listdir(out_dir) if f.upper().endswith((".EVA", ".EVN"))]
    if existing:
        print(f"  {year}: already have {len(existing)} event files, skipping download")
        return True

    url = f"https://www.retrosheet.org/events/{year}eve.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        print(f"  {year}: HTTP error {e.code} - {e.reason}")
        return False
    except Exception as e:
        print(f"  {year}: error {type(e).__name__}: {e}")
        return False

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(out_dir)
    except zipfile.BadZipFile:
        print(f"  {year}: bad zip file")
        return False

    files = os.listdir(out_dir)
    ev_count = sum(1 for f in files if f.upper().endswith((".EVA", ".EVN")))
    ros_count = sum(1 for f in files if f.upper().endswith(".ROS"))
    size_mb = len(data) / 1024 / 1024
    print(
        f"  {year}: downloaded {size_mb:.1f} MB in {time.time() - t0:.1f}s "
        f"-> {ev_count} event files, {ros_count} rosters"
    )
    return True


def main() -> int:
    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f"Downloading Retrosheet event files for {YEARS[0]}-{YEARS[-1]}")
    print(f"Destination: {OUT_ROOT}")
    ok = 0
    for y in YEARS:
        if download_year(y):
            ok += 1
    print(f"Done. {ok}/{len(YEARS)} years available.")
    return 0 if ok == len(YEARS) else 1


if __name__ == "__main__":
    sys.exit(main())
