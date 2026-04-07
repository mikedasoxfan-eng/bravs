"""Validate BRAVS against known WAR values for benchmark player-seasons."""

from __future__ import annotations

import math
import os
import sys
import time
from datetime import datetime

# Allow running as a standalone script from the scripts/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from baseball_metric.core.model import compute_bravs  # noqa: E402
from baseball_metric.core.types import PlayerSeason  # noqa: E402
from baseball_metric.run import NOTABLE_SEASONS  # noqa: E402

# ---------------------------------------------------------------------------
# Helper: look up a PlayerSeason from the run.py NOTABLE_SEASONS by player_id
# ---------------------------------------------------------------------------

_NOTABLE_MAP: dict[str, PlayerSeason] = {ps.player_id: ps for ps in NOTABLE_SEASONS}


def _get_notable(player_id: str) -> PlayerSeason | None:
    return _NOTABLE_MAP.get(player_id)


# ---------------------------------------------------------------------------
# Benchmark definitions: (player_id_or_key, known_fwar, PlayerSeason_or_None)
#
# If the third element is None we pull from NOTABLE_SEASONS.
# Otherwise we supply a freshly-constructed PlayerSeason.
# ---------------------------------------------------------------------------

BENCHMARKS: list[tuple[str, float, PlayerSeason | None]] = [
    # --- From NOTABLE_SEASONS ---
    ("trout01",       10.5, None),   # Trout 2016
    ("bonds01",       10.6, None),   # Bonds 2004
    ("degrom01",       9.6, None),   # deGrom 2018
    ("judge01",       11.4, None),   # Judge 2022
    ("ohtani01",      10.0, None),   # Ohtani 2023
    ("pedro01",       11.7, None),   # Pedro 2000
    ("clemens01",     10.4, None),   # Clemens 1997
    ("verlander01",    8.2, None),   # Verlander 2011
    ("ruth01",        12.9, None),   # Ruth 1927
    ("mays01",        10.0, None),   # Mays 1965
    ("aaron01",        5.1, None),   # Aaron 1971
    ("gibson01",      11.2, None),   # Gibson 1968
    ("koufax01",       9.8, None),   # Koufax 1966
    ("rivera01",       4.3, None),   # Rivera 2004
    ("smith_oz01",     5.3, None),   # Ozzie Smith 1987
    ("walker01",       9.3, None),   # Walker 1997
    ("piazza01",       6.3, None),   # Piazza 1997
    ("martinez_e01",   7.0, None),   # Edgar Martinez 1995
    ("baines01",       2.8, None),   # Baines 1985
    ("betts01",       10.9, None),   # Betts 2018
    ("soto01",         4.9, None),   # Soto 2020

    # --- Not in NOTABLE_SEASONS — constructed manually ---
    ("pedro99", 9.8, PlayerSeason(
        player_id="pedro99", player_name="Pedro Martinez", season=1999, team="BOS",
        position="P", ip=213.3, er=49, hits_allowed=160, hr_allowed=9,
        bb_allowed=37, hbp_allowed=9, k_pitching=313, games_pitched=31,
        games_started=29, park_factor=1.04, league_rpg=5.18, league="AL",
    )),
    ("maddux95", 9.0, PlayerSeason(
        player_id="maddux95", player_name="Greg Maddux", season=1995, team="ATL",
        position="P", ip=209.7, er=38, hits_allowed=147, hr_allowed=8,
        bb_allowed=23, hbp_allowed=4, k_pitching=181, games_pitched=28,
        games_started=28, park_factor=1.00, league_rpg=4.63,
    )),
    ("kershaw14", 7.2, PlayerSeason(
        player_id="kershaw14", player_name="Clayton Kershaw", season=2014, team="LAD",
        position="P", ip=198.3, er=39, hits_allowed=139, hr_allowed=9,
        bb_allowed=31, hbp_allowed=2, k_pitching=239, games_pitched=27,
        games_started=27, park_factor=0.95, league_rpg=3.74,
    )),
    ("rjohnson01", 10.2, PlayerSeason(
        player_id="rjohnson01", player_name="Randy Johnson", season=2001, team="ARI",
        position="P", ip=249.7, er=62, hits_allowed=181, hr_allowed=19,
        bb_allowed=71, hbp_allowed=18, k_pitching=372, games_pitched=35,
        games_started=35, park_factor=1.06, league_rpg=4.69,
    )),
    ("cole19", 7.4, PlayerSeason(
        player_id="cole19", player_name="Gerrit Cole", season=2019, team="HOU",
        position="P", ip=212.3, er=52, hits_allowed=142, hr_allowed=29,
        bb_allowed=48, hbp_allowed=7, k_pitching=326, games_pitched=33,
        games_started=33, park_factor=1.02, league_rpg=4.83, league="AL",
    )),
]


# ---------------------------------------------------------------------------
# Resolve each benchmark to (PlayerSeason, fWAR)
# ---------------------------------------------------------------------------

def _resolve_benchmarks() -> list[tuple[PlayerSeason, float]]:
    """Return a list of (PlayerSeason, known_fwar) for all valid benchmarks."""
    resolved: list[tuple[PlayerSeason, float]] = []
    for key, fwar, ps_override in BENCHMARKS:
        if ps_override is not None:
            resolved.append((ps_override, fwar))
        else:
            ps = _get_notable(key)
            if ps is None:
                print(f"WARNING: Could not find {key} in NOTABLE_SEASONS, skipping")
                continue
            resolved.append((ps, fwar))
    return resolved


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / (n - 1))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / (n - 1))
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    return cov / (sx * sy)


def _mae(xs: list[float], ys: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)


def _rmse(xs: list[float], ys: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(xs, ys)) / len(xs))


# ---------------------------------------------------------------------------
# Position and era classification
# ---------------------------------------------------------------------------

_POSITION_GROUPS = {
    "P": "Pitcher",
    "C": "Catcher",
    "DH": "DH",
}


def _position_group(pos: str) -> str:
    return _POSITION_GROUPS.get(pos, "Position Player")


def _era_bucket(season: int) -> str:
    if season < 1920:
        return "Dead Ball (<1920)"
    if season < 1947:
        return "Pre-Integration (1920-1946)"
    if season < 1969:
        return "Expansion (1947-1968)"
    if season < 1993:
        return "Modern (1969-1992)"
    if season < 2006:
        return "Steroid (1993-2005)"
    if season < 2015:
        return "Post-Steroid (2006-2014)"
    return "Statcast (2015+)"


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def run_validation() -> str:
    """Run the full validation suite and return the report as a string."""
    benchmarks = _resolve_benchmarks()
    lines: list[str] = []

    header = (
        f"BRAVS Validation Report\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'=' * 90}\n"
    )
    lines.append(header)

    bravs_vals: list[float] = []
    fwar_vals: list[float] = []
    diffs: list[tuple[str, int, float, float, float]] = []  # name, season, bravs_cal, fwar, diff

    # Per-group accumulators for bias analysis
    pos_diffs: dict[str, list[float]] = {}
    era_diffs: dict[str, list[float]] = {}

    lines.append(f"{'Player':<25} {'Year':>5} {'Pos':<4} {'BRAVS':>6} {'ErStd':>6} "
                 f"{'WAReq':>6} {'fWAR':>6} {'Diff':>7}")
    lines.append("-" * 90)

    total_start = time.perf_counter()

    for ps, fwar in benchmarks:
        result = compute_bravs(ps)
        cal = result.bravs_calibrated
        era_std = result.bravs_era_standardized
        diff = cal - fwar

        bravs_vals.append(cal)
        fwar_vals.append(fwar)
        diffs.append((ps.player_name, ps.season, cal, fwar, diff))

        pg = _position_group(ps.position)
        pos_diffs.setdefault(pg, []).append(diff)

        eb = _era_bucket(ps.season)
        era_diffs.setdefault(eb, []).append(diff)

        lines.append(
            f"{ps.player_name:<25} {ps.season:>5} {ps.position:<4} "
            f"{result.bravs:>6.1f} {era_std:>6.1f} {cal:>6.1f} {fwar:>6.1f} "
            f"{diff:>+7.2f}"
        )

    elapsed = time.perf_counter() - total_start

    # --- Summary statistics ---
    lines.append("")
    lines.append("=" * 90)
    lines.append("SUMMARY STATISTICS")
    lines.append("=" * 90)
    r = _pearson(bravs_vals, fwar_vals)
    mae = _mae(bravs_vals, fwar_vals)
    rmse = _rmse(bravs_vals, fwar_vals)
    mean_diff = sum(d for _, _, _, _, d in diffs) / len(diffs)

    lines.append(f"  N benchmarks:          {len(diffs)}")
    lines.append(f"  Pearson correlation:   {r:.4f}")
    lines.append(f"  Mean Absolute Error:   {mae:.2f} wins")
    lines.append(f"  RMSE:                  {rmse:.2f} wins")
    lines.append(f"  Mean Bias (BRAVS-fWAR):{mean_diff:+.2f} wins")
    lines.append(f"  Computation time:      {elapsed:.2f}s")

    # --- Biggest divergences ---
    sorted_diffs = sorted(diffs, key=lambda t: t[4], reverse=True)

    lines.append("")
    lines.append("-" * 90)
    lines.append("BIGGEST POSITIVE DIVERGENCES (BRAVS overvalues vs fWAR)")
    lines.append("-" * 90)
    for name, season, cal, fwar, diff in sorted_diffs[:5]:
        lines.append(f"  {name:<25} {season:>5}  WAReq={cal:>6.1f}  fWAR={fwar:>6.1f}  diff={diff:>+.2f}")

    lines.append("")
    lines.append("-" * 90)
    lines.append("BIGGEST NEGATIVE DIVERGENCES (BRAVS undervalues vs fWAR)")
    lines.append("-" * 90)
    for name, season, cal, fwar, diff in sorted_diffs[-5:]:
        lines.append(f"  {name:<25} {season:>5}  WAReq={cal:>6.1f}  fWAR={fwar:>6.1f}  diff={diff:>+.2f}")

    # --- Systematic biases by position ---
    lines.append("")
    lines.append("-" * 90)
    lines.append("SYSTEMATIC BIASES BY POSITION GROUP")
    lines.append("-" * 90)
    for group in sorted(pos_diffs.keys()):
        ds = pos_diffs[group]
        avg = sum(ds) / len(ds)
        lines.append(f"  {group:<20}  n={len(ds):>2}  mean bias={avg:>+.2f} wins")

    # --- Systematic biases by era ---
    lines.append("")
    lines.append("-" * 90)
    lines.append("SYSTEMATIC BIASES BY ERA")
    lines.append("-" * 90)
    for era in sorted(era_diffs.keys()):
        ds = era_diffs[era]
        avg = sum(ds) / len(ds)
        lines.append(f"  {era:<30}  n={len(ds):>2}  mean bias={avg:>+.2f} wins")

    lines.append("")
    lines.append("=" * 90)
    lines.append("END OF REPORT")
    lines.append("=" * 90)

    return "\n".join(lines)


def main() -> None:
    report = run_validation()

    # Print to stdout
    print(report)

    # Save to logs/
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "validation_report.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport saved to {os.path.abspath(log_path)}")


if __name__ == "__main__":
    main()
