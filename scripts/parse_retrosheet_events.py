"""Parse Retrosheet .EVA/.EVN event files into a single play-level dataset.

Input : data/retrosheet/events/{year}/*.EVA / *.EVN
Output: data/retrosheet/events_parsed.parquet (or .csv fallback)

Retrosheet event-file format reference: https://www.retrosheet.org/eventfile.htm

We write a pure-Python parser (no Chadwick, no retrosheet-cli). Per-play rows
include game metadata, batter, pitcher, count, pitch sequence, raw event, and
a simplified event class.
"""
from __future__ import annotations

import os
import random
import re
import sys
import time
from collections import Counter
from typing import Iterator

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS_ROOT = os.path.join(ROOT, "data", "retrosheet", "events")
OUT_PARQUET = os.path.join(ROOT, "data", "retrosheet", "events_parsed.parquet")
OUT_CSV = os.path.join(ROOT, "data", "retrosheet", "events_parsed.csv")

YEARS = list(range(2015, 2025))

# ---------------------------------------------------------------------------
# Event-field classification
# ---------------------------------------------------------------------------

# The event field may contain modifiers separated by '/' and advances after '.'
# e.g. "S9/L9", "HR/78/F", "K+SB2", "W.1-3", "63/G6", "E4/G", "FC6/G6.1X2(64)"
# We only care about the core play token (before '/' or '.').
_ADV_SPLIT = re.compile(r"[./]")


def classify_event(ev: str) -> str:
    """Return a simplified event class for a Retrosheet play event string.

    Classes: HIT / OUT / WALK / K / HR / HBP / ERROR / FC / OTHER / NOPLAY
    (HIT covers singles, doubles, triples; HR is separate.)
    """
    if not ev:
        return "OTHER"
    ev = ev.strip()
    if ev == "NP":
        return "NOPLAY"

    # Strip trailing '!' / '?' / '#' annotation markers sprinkled in by scorers.
    core = _ADV_SPLIT.split(ev, 1)[0]
    core = core.strip().rstrip("!#?")
    if not core:
        return "OTHER"

    # Order matters: check multi-char tokens first.
    if core.startswith("HR") or core == "H" or core.startswith("H "):
        # "H" on its own = inside-the-park HR or HR w/ no trajectory marker.
        # "HR/..." classified above at core level since we split on '/'.
        return "HR"
    if core.startswith("HP"):
        return "HBP"
    if core.startswith("K"):
        # "K", "K23", "K+SB2", "K+WP" etc.
        return "K"
    if core.startswith("W") and not core.startswith("WP"):
        # W = unintentional walk, IW / I = intentional walk
        return "WALK"
    if core.startswith("IW") or core == "I":
        return "WALK"
    if core.startswith("S") and not core.startswith("SB"):
        # Single: "S", "S7", "S8/L"; exclude "SB" (stolen base, not a PA).
        return "HIT"
    if core.startswith("D") and not core.startswith("DI"):
        # Double: "D", "D7", "DGR" (ground-rule double). Exclude "DI" (def indiff).
        return "HIT"
    if core.startswith("T") and not core.startswith("TH"):
        # Triple: "T", "T9". Exclude "TH" throwing.
        return "HIT"
    if core.startswith("E"):
        return "ERROR"
    if core.startswith("FC"):
        return "FC"
    if core.startswith("FLE"):
        return "ERROR"  # Error on foul fly.

    # Non-PA events that can appear on their own play line.
    if core.startswith(("SB", "CS", "PO", "WP", "PB", "BK", "DI", "OA", "FLE", "NP")):
        return "NOPLAY"

    # Pure digit sequence = out recorded by those fielders, e.g. "63", "8", "1(B)3".
    stripped = re.sub(r"[()!#?]", "", core)
    if stripped and stripped[0].isdigit():
        return "OUT"

    return "OTHER"


def is_pa_class(cls: str) -> bool:
    """Whether a simplified class counts as a plate appearance result."""
    return cls in {"HIT", "OUT", "WALK", "K", "HR", "HBP", "ERROR", "FC"}


# ---------------------------------------------------------------------------
# Pitch-sequence cleaning
# ---------------------------------------------------------------------------

# Retrosheet pitch codes include many non-pitch annotations we want stripped
# when counting pitches: '.', '*', '+', '1', '2', '3' (pickoffs / catcher throws),
# '>' (strike called), etc. See
# https://www.retrosheet.org/eventfile.htm#pitches
_NON_PITCH_CHARS = set(".*+123>")
# Characters that are actual pitches thrown to the batter:
# B,C,F,H,I,K,L,M,O,P,Q,R,S,T,U,V,X,Y (Retrosheet pitch codes)
_PITCH_CHARS = set("BCFHIKLMOPQRSTUVXY")


def count_pitches(seq: str) -> int:
    if not seq:
        return 0
    return sum(1 for ch in seq if ch in _PITCH_CHARS)


# ---------------------------------------------------------------------------
# File parser
# ---------------------------------------------------------------------------


def parse_event_file(path: str) -> Iterator[dict]:
    """Yield dict rows for each play line in a single .EVA/.EVN file."""
    with open(path, "r", encoding="latin-1", errors="replace") as fh:
        lines = fh.readlines()

    # Per-game state, reset on each 'id,' line.
    game_id: str | None = None
    date = ""
    home = ""
    away = ""
    gametype = "regular"
    cur_pitcher = {0: "", 1: ""}  # 0 = away pitcher, 1 = home pitcher
    lineup_pos_player: dict[tuple[int, int], str] = {}  # (team, slot)->pid
    outs_in_half = 0
    cur_inning = 0
    cur_half = -1  # 0 or 1
    skip_game = False

    rows: list[dict] = []

    def flush_if_game_ok() -> None:
        if not skip_game:
            for r in rows:
                yield_rows.append(r)

    yield_rows: list[dict] = []

    def reset_game() -> None:
        nonlocal game_id, date, home, away, gametype, outs_in_half, cur_inning, cur_half, skip_game
        game_id = None
        date = ""
        home = ""
        away = ""
        gametype = "regular"
        cur_pitcher[0] = ""
        cur_pitcher[1] = ""
        lineup_pos_player.clear()
        outs_in_half = 0
        cur_inning = 0
        cur_half = -1
        skip_game = False
        rows.clear()

    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            continue
        try:
            parts = line.split(",")
            tag = parts[0]

            if tag == "id":
                # Emit previous game's rows (if any) before starting new one.
                if game_id is not None:
                    for r in rows:
                        yield_rows.append(r)
                reset_game()
                game_id = parts[1] if len(parts) > 1 else None

            elif tag == "info" and len(parts) >= 3:
                key = parts[1]
                val = ",".join(parts[2:])
                if key == "visteam":
                    away = val
                elif key == "hometeam":
                    home = val
                elif key == "date":
                    date = val.replace("/", "-")
                elif key == "gametype":
                    gametype = val.strip().lower()
                    if gametype and gametype != "regular":
                        skip_game = True

            elif tag == "start" or tag == "sub":
                # start,pid,"Name",home(0/1),batting_order,position
                if len(parts) < 6:
                    continue
                pid = parts[1]
                # parts[2] is the quoted name which may contain commas; rebuild.
                # Re-split by locating last 3 fields which are numeric.
                try:
                    pos = int(parts[-1])
                    order = int(parts[-2])
                    team_side = int(parts[-3])
                except ValueError:
                    continue
                # Track current pitcher (position 1) per side.
                if pos == 1:
                    cur_pitcher[team_side] = pid
                lineup_pos_player[(team_side, order)] = pid

            elif tag == "play":
                # play,inning,home/away,batter_id,count,pitches,event
                if len(parts) < 7 or skip_game:
                    continue
                try:
                    inning = int(parts[1])
                except ValueError:
                    continue
                half_flag = parts[2]  # 0 = top (away bats), 1 = bottom (home bats)
                batter_id = parts[3]
                count_str = parts[4]
                pitches = parts[5]
                # Event may itself contain commas (rare but possible); rejoin.
                event_raw = ",".join(parts[6:]).strip()

                # Reset outs counter when half inning flips.
                if inning != cur_inning or half_flag != (str(cur_half) if cur_half >= 0 else ""):
                    cur_inning = inning
                    try:
                        cur_half = int(half_flag)
                    except ValueError:
                        cur_half = -1
                    outs_in_half = 0

                try:
                    is_home_bat = int(half_flag)
                except ValueError:
                    is_home_bat = 0
                # Pitcher = the opposing team's current pitcher.
                pitcher_id = cur_pitcher[1 - is_home_bat] if is_home_bat in (0, 1) else ""

                cls = classify_event(event_raw)
                if cls == "NOPLAY":
                    # NP / stolen base / pickoff lines -- not a plate appearance.
                    # We still want to skip them from the PA dataset but some
                    # downstream uses want them, so keep classification but
                    # don't emit to keep dataset PA-focused.
                    continue

                outs_before = outs_in_half
                # Update outs: OUT, K, FC (treat as out), some errors can be outs too
                # but we only bump for clean outs / strikeouts.
                if cls in ("OUT", "K"):
                    outs_in_half += 1
                    if outs_in_half >= 3:
                        outs_in_half = 0

                n_pitches = count_pitches(pitches)
                half_label = "T" if is_home_bat == 0 else "B"

                rows.append({
                    "game_id": game_id,
                    "date": date,
                    "home_team": home,
                    "away_team": away,
                    "inning": inning,
                    "half": half_label,
                    "outs_before": outs_before,
                    "batter_id": batter_id,
                    "pitcher_id": pitcher_id,
                    "count": count_str,
                    "pitches": pitches,
                    "num_pitches": n_pitches,
                    "event_raw": event_raw,
                    "event_simple": cls,
                    "is_hit": int(cls in ("HIT", "HR")),
                    "is_out": int(cls == "OUT"),
                    "is_walk": int(cls == "WALK"),
                    "is_k": int(cls == "K"),
                    "is_hr": int(cls == "HR"),
                    "is_hbp": int(cls == "HBP"),
                })

            # 'data' and 'com' lines are ignored.
        except Exception:
            # Skip any malformed lines without aborting the file.
            continue

    # Flush last game.
    if game_id is not None and not skip_game:
        for r in rows:
            yield_rows.append(r)

    for r in yield_rows:
        yield r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    all_rows: list[dict] = []
    file_count = 0
    t0 = time.time()

    for year in YEARS:
        year_dir = os.path.join(EVENTS_ROOT, str(year))
        if not os.path.isdir(year_dir):
            print(f"[warn] missing {year_dir}")
            continue
        year_rows_before = len(all_rows)
        year_files = [
            f for f in sorted(os.listdir(year_dir))
            if f.upper().endswith((".EVA", ".EVN"))
        ]
        for fname in year_files:
            fpath = os.path.join(year_dir, fname)
            for row in parse_event_file(fpath):
                all_rows.append(row)
            file_count += 1
        print(
            f"  {year}: {len(year_files)} files, "
            f"+{len(all_rows) - year_rows_before:,} plays "
            f"(cum {len(all_rows):,}, {time.time() - t0:.1f}s)"
        )

    print(f"\nParsed {file_count} event files -> {len(all_rows):,} rows in {time.time() - t0:.1f}s")

    if not all_rows:
        print("No rows parsed. Aborting.")
        return 1

    df = pd.DataFrame(all_rows)
    # Compute summary before writing (cheap either way).
    games = df["game_id"].nunique()
    plays = len(df)
    by_cls = df["event_simple"].value_counts().to_dict()

    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    out_path = OUT_PARQUET
    try:
        df.to_parquet(OUT_PARQUET, index=False)
        print(f"Wrote parquet: {OUT_PARQUET} ({os.path.getsize(OUT_PARQUET) / 1024 / 1024:.1f} MB)")
    except Exception as e:
        print(f"Parquet write failed ({e}); falling back to CSV.")
        df.to_csv(OUT_CSV, index=False)
        out_path = OUT_CSV
        print(f"Wrote csv: {OUT_CSV} ({os.path.getsize(OUT_CSV) / 1024 / 1024:.1f} MB)")

    # ------------------------------------------------------------------
    # Validation summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Games parsed:  {games:,}")
    print(f"Plays parsed:  {plays:,}")
    print(f"Years:         {sorted(df['date'].str.slice(0, 4).unique().tolist())}")
    print("\nBreakdown by event_simple:")
    for k, v in sorted(by_cls.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<8} {v:>10,}  ({v / plays * 100:5.2f}%)")

    print("\n3 random sample plays:")
    sample = df.sample(n=3, random_state=random.randint(0, 10**6))
    for _, r in sample.iterrows():
        print(
            f"  [{r['date']} {r['game_id']}] inn {r['inning']}{r['half']} "
            f"{r['batter_id']} vs {r['pitcher_id']} "
            f"count={r['count']} pit={r['pitches']!r} "
            f"ev={r['event_raw']!r} -> {r['event_simple']}"
        )

    print(f"\nOutput: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
