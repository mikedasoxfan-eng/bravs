"""Best single season by position — BRAVS leaderboard at every position.

For each defensive position (C, 1B, 2B, 3B, SS, LF, CF, RF, DH, SP, RP),
build the top 3 historical seasons with approximate realistic stats,
compute BRAVS, and rank within position.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime
from baseball_metric.core.model import compute_bravs
from baseball_metric.core.types import PlayerSeason, BRAVSResult

# ---------------------------------------------------------------------------
# Top 3 seasons per position (approximate stats)
# ---------------------------------------------------------------------------

POSITIONS: dict[str, list[PlayerSeason]] = {}

# --- Catcher ---
POSITIONS["C"] = [
    PlayerSeason(
        player_id="piazzmi01", player_name="Mike Piazza", season=1997, team="LAD",
        position="C", pa=633, ab=556, hits=201, doubles=32, triples=1, hr=40,
        bb=69, ibb=7, hbp=2, k=77, sf=6, games=152, sb=5, cs=1, gidp=14,
        framing_runs=5.0, blocking_runs=0.5, throwing_runs=-2.0,
        inn_fielded=1200.0,
        park_factor=0.97, league_rpg=4.77,
    ),
    PlayerSeason(
        player_id="benchjo01", player_name="Johnny Bench", season=1972, team="CIN",
        position="C", pa=604, ab=538, hits=145, doubles=22, triples=2, hr=40,
        bb=100, ibb=14, hbp=1, k=84, sf=4, games=147, sb=6, cs=4, gidp=11,
        framing_runs=8.0, blocking_runs=3.0, throwing_runs=5.0,
        inn_fielded=1250.0,
        park_factor=1.02, league_rpg=3.69,
    ),
    PlayerSeason(
        player_id="poseybu01", player_name="Buster Posey", season=2012, team="SF",
        position="C", pa=610, ab=530, hits=178, doubles=39, triples=1, hr=24,
        bb=69, ibb=13, hbp=1, k=96, sf=7, games=148, sb=1, cs=1, gidp=16,
        framing_runs=18.0, blocking_runs=2.0, throwing_runs=1.0,
        inn_fielded=1150.0,
        park_factor=0.93, league_rpg=4.32,
    ),
]

# --- First Base ---
POSITIONS["1B"] = [
    PlayerSeason(
        player_id="gehrilo01", player_name="Lou Gehrig", season=1927, team="NYY",
        position="1B", pa=717, ab=584, hits=218, doubles=52, triples=18, hr=47,
        bb=109, ibb=0, hbp=3, k=84, sh=21, games=155, sb=10, cs=8, gidp=12,
        park_factor=1.05, league_rpg=4.73, league="AL",
    ),
    PlayerSeason(
        player_id="pujolal01", player_name="Albert Pujols", season=2006, team="STL",
        position="1B", pa=634, ab=535, hits=177, doubles=33, triples=1, hr=49,
        bb=92, ibb=28, hbp=4, k=50, sf=3, games=143, sb=7, cs=2, gidp=20,
        uzr=4.0, drs=6, inn_fielded=1200.0,
        park_factor=0.99, league_rpg=4.76,
    ),
    PlayerSeason(
        player_id="bagweje01", player_name="Jeff Bagwell", season=1994, team="HOU",
        position="1B", pa=536, ab=400, hits=147, doubles=32, triples=2, hr=39,
        bb=65, ibb=13, hbp=9, k=65, sf=3, games=110, sb=15, cs=4, gidp=10,
        park_factor=1.00, league_rpg=4.62,
    ),
]

# --- Second Base ---
POSITIONS["2B"] = [
    PlayerSeason(
        player_id="hornsro01", player_name="Rogers Hornsby", season=1922, team="STL",
        position="2B", pa=673, ab=623, hits=250, doubles=46, triples=14, hr=42,
        bb=65, ibb=0, hbp=1, k=50, sh=12, games=154, sb=17, cs=12, gidp=10,
        park_factor=1.01, league_rpg=4.82,
    ),
    PlayerSeason(
        player_id="altuvjo01", player_name="Jose Altuve", season=2017, team="HOU",
        position="2B", pa=662, ab=590, hits=204, doubles=39, triples=4, hr=24,
        bb=58, ibb=4, hbp=6, k=84, sf=4, games=153, sb=32, cs=6, gidp=13,
        uzr=2.0, drs=3, oaa=4, inn_fielded=1300.0,
        park_factor=1.02, league_rpg=4.65, league="AL",
    ),
    PlayerSeason(
        player_id="kentje01", player_name="Jeff Kent", season=2000, team="SF",
        position="2B", pa=651, ab=587, hits=196, doubles=41, triples=7, hr=33,
        bb=90, ibb=11, hbp=5, k=107, sf=2, games=159, sb=12, cs=6, gidp=16,
        uzr=-2.0, drs=-3, inn_fielded=1360.0,
        park_factor=0.93, league_rpg=5.14,
    ),
]

# --- Third Base ---
POSITIONS["3B"] = [
    PlayerSeason(
        player_id="schmimi01", player_name="Mike Schmidt", season=1980, team="PHI",
        position="3B", pa=624, ab=548, hits=157, doubles=25, triples=8, hr=48,
        bb=89, ibb=12, hbp=2, k=119, sf=6, games=150, sb=12, cs=4, gidp=7,
        uzr=10.0, drs=12, inn_fielded=1280.0,
        park_factor=1.03, league_rpg=4.20,
    ),
    PlayerSeason(
        player_id="brettge01", player_name="George Brett", season=1980, team="KC",
        position="3B", pa=515, ab=449, hits=175, doubles=33, triples=9, hr=24,
        bb=58, ibb=12, hbp=1, k=22, sf=5, games=117, sb=15, cs=5, gidp=6,
        uzr=3.0, drs=4, inn_fielded=980.0,
        park_factor=1.01, league_rpg=4.30, league="AL",
    ),
    PlayerSeason(
        player_id="beltrad01", player_name="Adrian Beltre", season=2004, team="LAD",
        position="3B", pa=657, ab=598, hits=200, doubles=32, triples=0, hr=48,
        bb=53, ibb=4, hbp=8, k=87, sf=3, games=156, sb=7, cs=3, gidp=13,
        uzr=12.0, drs=14, inn_fielded=1350.0,
        park_factor=0.97, league_rpg=4.81,
    ),
]

# --- Shortstop ---
POSITIONS["SS"] = [
    PlayerSeason(
        player_id="wagneho01", player_name="Honus Wagner", season=1908, team="PIT",
        position="SS", pa=615, ab=568, hits=201, doubles=39, triples=19, hr=10,
        bb=54, ibb=0, hbp=1, k=0, sh=17, games=151, sb=53, cs=20, gidp=6,
        park_factor=0.98, league_rpg=3.37,
    ),
    PlayerSeason(
        player_id="rodrial01", player_name="Alex Rodriguez", season=1996, team="SEA",
        position="SS", pa=641, ab=601, hits=215, doubles=54, triples=1, hr=36,
        bb=59, ibb=6, hbp=9, k=104, sf=4, games=146, sb=15, cs=4, gidp=18,
        uzr=4.0, drs=5, inn_fielded=1280.0,
        park_factor=0.97, league_rpg=5.39, league="AL",
    ),
    PlayerSeason(
        player_id="ripkeca01", player_name="Cal Ripken", season=1991, team="BAL",
        position="SS", pa=680, ab=650, hits=210, doubles=46, triples=5, hr=34,
        bb=53, ibb=6, hbp=5, k=46, sf=1, games=162, sb=6, cs=1, gidp=14,
        uzr=8.0, drs=10, inn_fielded=1440.0,
        park_factor=1.01, league_rpg=4.32, league="AL",
    ),
]

# --- Left Field ---
POSITIONS["LF"] = [
    PlayerSeason(
        player_id="bondsba01", player_name="Barry Bonds", season=2001, team="SF",
        position="LF", pa=664, ab=476, hits=156, doubles=32, triples=2, hr=73,
        bb=177, ibb=35, hbp=9, k=93, sf=2, games=153, sb=13, cs=3, gidp=5,
        uzr=-2.0, drs=-1, inn_fielded=1200.0,
        park_factor=0.93, league_rpg=4.78,
    ),
    PlayerSeason(
        player_id="willite01", player_name="Ted Williams", season=1941, team="BOS",
        position="LF", pa=606, ab=456, hits=185, doubles=33, triples=3, hr=37,
        bb=147, ibb=0, hbp=3, k=27, games=143, sb=2, cs=4, gidp=12,
        park_factor=1.04, league_rpg=4.99, league="AL",
    ),
    PlayerSeason(
        player_id="henderi01", player_name="Rickey Henderson", season=1990, team="OAK",
        position="LF", pa=621, ab=489, hits=159, doubles=33, triples=3, hr=28,
        bb=97, ibb=13, hbp=2, k=60, sf=4, games=136, sb=65, cs=10, gidp=4,
        uzr=1.0, drs=2, inn_fielded=1100.0,
        park_factor=0.98, league_rpg=4.30, league="AL",
    ),
]

# --- Center Field ---
POSITIONS["CF"] = [
    PlayerSeason(
        player_id="troutmi01", player_name="Mike Trout", season=2016, team="LAA",
        position="CF", pa=681, ab=549, hits=173, doubles=32, triples=5, hr=29,
        bb=116, ibb=9, hbp=7, k=137, sf=7, sh=2, games=159, sb=30, cs=7, gidp=9,
        uzr=1.0, drs=2, oaa=5, inn_fielded=1350.0,
        park_factor=0.98, league_rpg=4.48, league="AL",
    ),
    PlayerSeason(
        player_id="mayswi01", player_name="Willie Mays", season=1965, team="SF",
        position="CF", pa=659, ab=558, hits=177, doubles=21, triples=3, hr=52,
        bb=76, ibb=15, hbp=5, k=71, sf=5, games=157, sb=9, cs=4, gidp=9,
        uzr=5.0, drs=7, inn_fielded=1350.0,
        park_factor=0.93, league_rpg=4.03,
    ),
    PlayerSeason(
        player_id="mantlmi01", player_name="Mickey Mantle", season=1956, team="NYY",
        position="CF", pa=650, ab=533, hits=188, doubles=22, triples=5, hr=52,
        bb=112, ibb=14, hbp=1, k=99, sf=4, games=150, sb=10, cs=1, gidp=8,
        uzr=3.0, drs=4, inn_fielded=1300.0,
        park_factor=1.05, league_rpg=4.61, league="AL",
    ),
]

# --- Right Field ---
POSITIONS["RF"] = [
    PlayerSeason(
        player_id="ruthba01", player_name="Babe Ruth", season=1927, team="NYY",
        position="RF", pa=691, ab=540, hits=192, doubles=29, triples=8, hr=60,
        bb=137, ibb=0, hbp=0, k=89, sh=14, games=151, sb=7, cs=6, gidp=10,
        park_factor=1.05, league_rpg=4.73, league="AL",
    ),
    PlayerSeason(
        player_id="aaronha01", player_name="Hank Aaron", season=1971, team="ATL",
        position="RF", pa=623, ab=495, hits=162, doubles=22, triples=3, hr=47,
        bb=71, ibb=20, hbp=2, k=58, sf=5, games=139, sb=1, cs=1, gidp=13,
        park_factor=1.01, league_rpg=3.91,
    ),
    PlayerSeason(
        player_id="bettsmo01", player_name="Mookie Betts", season=2018, team="BOS",
        position="RF", pa=614, ab=520, hits=180, doubles=47, triples=5, hr=32,
        bb=81, ibb=5, hbp=5, k=82, sf=3, games=136, sb=30, cs=6, gidp=12,
        uzr=10.5, drs=14, oaa=11, inn_fielded=1150.0,
        park_factor=1.04, league_rpg=4.45, league="AL",
    ),
]

# --- Designated Hitter ---
POSITIONS["DH"] = [
    PlayerSeason(
        player_id="martied01", player_name="Edgar Martinez", season=1995, team="SEA",
        position="DH", pa=639, ab=511, hits=182, doubles=52, triples=0, hr=29,
        bb=116, ibb=11, hbp=6, k=87, sf=6, games=145, sb=4, cs=2, gidp=14,
        park_factor=0.97, league_rpg=5.06, league="AL",
    ),
    PlayerSeason(
        player_id="ortizda01", player_name="David Ortiz", season=2016, team="BOS",
        position="DH", pa=614, ab=537, hits=169, doubles=48, triples=1, hr=38,
        bb=80, ibb=14, hbp=4, k=86, sf=4, games=151, sb=2, cs=0, gidp=19,
        park_factor=1.04, league_rpg=4.40, league="AL",
    ),
    PlayerSeason(
        player_id="cruzne01", player_name="Nelson Cruz", season=2017, team="SEA",
        position="DH", pa=643, ab=556, hits=160, doubles=28, triples=1, hr=39,
        bb=70, ibb=7, hbp=6, k=140, sf=5, games=155, sb=1, cs=1, gidp=18,
        park_factor=0.97, league_rpg=4.65, league="AL",
    ),
]

# --- Starting Pitcher ---
POSITIONS["SP"] = [
    PlayerSeason(
        player_id="martipe02", player_name="Pedro Martinez", season=2000, team="BOS",
        position="P", ip=217.0, er=42, hits_allowed=128, hr_allowed=17,
        bb_allowed=32, hbp_allowed=6, k_pitching=284, games_pitched=29,
        games_started=29, park_factor=1.04, league_rpg=5.14, league="AL",
    ),
    PlayerSeason(
        player_id="gibsobo01", player_name="Bob Gibson", season=1968, team="STL",
        position="P", ip=304.7, er=38, hits_allowed=198, hr_allowed=11,
        bb_allowed=62, hbp_allowed=4, k_pitching=268, games_pitched=34,
        games_started=34, park_factor=0.98, league_rpg=3.42,
    ),
    PlayerSeason(
        player_id="maddugr01", player_name="Greg Maddux", season=1995, team="ATL",
        position="P", ip=209.7, er=38, hits_allowed=147, hr_allowed=8,
        bb_allowed=23, hbp_allowed=4, k_pitching=181, games_pitched=28,
        games_started=28, park_factor=1.00, league_rpg=4.63,
    ),
]

# --- Relief Pitcher ---
POSITIONS["RP"] = [
    PlayerSeason(
        player_id="riverma01", player_name="Mariano Rivera", season=2004, team="NYY",
        position="P", ip=78.7, er=15, hits_allowed=65, hr_allowed=3,
        bb_allowed=20, hbp_allowed=2, k_pitching=66, games_pitched=74,
        games_started=0, saves=53, holds=0,
        avg_leverage_index=2.10,
        park_factor=1.05, league_rpg=4.83, league="AL",
    ),
    PlayerSeason(
        player_id="gagneer01", player_name="Eric Gagne", season=2003, team="LAD",
        position="P", ip=82.3, er=12, hits_allowed=37, hr_allowed=3,
        bb_allowed=20, hbp_allowed=2, k_pitching=137, games_pitched=77,
        games_started=0, saves=55, holds=0,
        avg_leverage_index=2.00,
        park_factor=0.97, league_rpg=4.61,
    ),
    PlayerSeason(
        player_id="eckerde01", player_name="Dennis Eckersley", season=1990, team="OAK",
        position="P", ip=73.3, er=5, hits_allowed=41, hr_allowed=2,
        bb_allowed=4, hbp_allowed=1, k_pitching=73, games_pitched=63,
        games_started=0, saves=48, holds=0,
        avg_leverage_index=1.95,
        park_factor=0.98, league_rpg=4.30, league="AL",
    ),
]


def main() -> None:
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "logs"), exist_ok=True)
    log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "best_by_position.log")

    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    out("=" * 110)
    out("  BEST SINGLE SEASON BY POSITION — BRAVS RANKINGS")
    out(f"  Generated {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    out("=" * 110)
    out()

    # Compute and store all results
    all_results: dict[str, list[tuple[PlayerSeason, BRAVSResult]]] = {}

    for pos_label, seasons in POSITIONS.items():
        all_results[pos_label] = []
        for ps in seasons:
            result = compute_bravs(ps, fast=True)
            all_results[pos_label].append((ps, result))
        # Sort by BRAVS within position
        all_results[pos_label].sort(key=lambda x: x[1].bravs, reverse=True)

    # --- Position-by-position breakdown ---
    position_order = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "SP", "RP"]

    for pos_label in position_order:
        entries = all_results[pos_label]
        out("-" * 110)
        pos_name = {
            "C": "CATCHER", "1B": "FIRST BASE", "2B": "SECOND BASE",
            "3B": "THIRD BASE", "SS": "SHORTSTOP", "LF": "LEFT FIELD",
            "CF": "CENTER FIELD", "RF": "RIGHT FIELD", "DH": "DESIGNATED HITTER",
            "SP": "STARTING PITCHER", "RP": "RELIEF PITCHER",
        }[pos_label]
        out(f"  {pos_name} ({pos_label})")
        out("-" * 110)
        out(f"  {'Rank':>4}  {'Player':<26}{'Year':>5}{'BRAVS':>8}{'ErStd':>8}{'WAReq':>8}"
            f"  {'90% CI':>16}  {'Key Component Runs':>30}")
        out("  " + "-" * 106)

        for i, (ps, r) in enumerate(entries, 1):
            ci = r.bravs_ci_90

            # Identify the top contributing component (excluding positional/durability)
            skip = {"positional", "durability", "leverage"}
            key_comps = [(n, c.runs_mean) for n, c in r.components.items() if n not in skip]
            key_comps.sort(key=lambda x: abs(x[1]), reverse=True)
            comp_str = "  ".join(f"{n}:{v:+.0f}" for n, v in key_comps[:3])

            out(f"  {i:>4}. {ps.player_name:<26}{ps.season:>5}{r.bravs:>8.1f}{r.bravs_era_standardized:>8.1f}"
                f"{r.bravs_calibrated:>8.1f}  [{ci[0]:>5.1f}, {ci[1]:>5.1f}]  {comp_str}")

        out()

        # Detailed breakdown for #1 at each position
        top_ps, top_r = entries[0]
        out(f"  BEST EVER at {pos_label}: {top_ps.player_name} ({top_ps.season})")
        out(f"  BRAVS = {top_r.bravs:.1f} | Era-Std = {top_r.bravs_era_standardized:.1f}"
            f" | WAR-eq = {top_r.bravs_calibrated:.1f}")
        out(f"  Component breakdown:")
        for comp_name, comp in sorted(top_r.components.items()):
            out(f"    {comp_name:20s}: {comp.runs_mean:+7.1f} runs  "
                f"[{comp.ci_90[0]:+.1f}, {comp.ci_90[1]:+.1f}]")
        out()

    # --- Cross-position comparison ---
    out()
    out("=" * 110)
    out("  CROSS-POSITION: BEST SINGLE SEASON EVER AT EACH POSITION")
    out("=" * 110)
    out()

    cross_position: list[tuple[str, str, int, float, float, float]] = []
    for pos_label in position_order:
        top_ps, top_r = all_results[pos_label][0]
        cross_position.append((
            pos_label,
            top_ps.player_name,
            top_ps.season,
            top_r.bravs,
            top_r.bravs_era_standardized,
            top_r.bravs_calibrated,
        ))

    cross_position.sort(key=lambda x: x[4], reverse=True)

    out(f"  {'Rank':>4}  {'Pos':>4}  {'Player':<26}{'Year':>5}{'BRAVS':>8}{'ErStd':>8}{'WAReq':>8}")
    out("  " + "-" * 75)
    for i, (pos, name, year, bravs, era_std, wareq) in enumerate(cross_position, 1):
        out(f"  {i:>4}. {pos:>4}  {name:<26}{year:>5}{bravs:>8.1f}{era_std:>8.1f}{wareq:>8.1f}")

    out()

    # --- Positional value discussion ---
    out("=" * 110)
    out("  POSITIONAL VALUE PREMIUM ANALYSIS")
    out("=" * 110)
    out()
    out("  The positional adjustment built into BRAVS (per 162 games):")
    from baseball_metric.utils.constants import POS_ADJ
    for pos in ["C", "SS", "CF", "2B", "3B", "LF", "RF", "1B", "DH"]:
        adj = POS_ADJ.get(pos, 0.0)
        out(f"    {pos:>4}: {adj:+.1f} runs")
    out()
    out("  This means a catcher with identical offensive stats to a DH gets")
    out("  +30 runs of positional credit per full season — roughly 3 extra wins.")
    out("  Johnny Bench and Buster Posey benefit enormously from this premium.")
    out()
    out("  Relievers produce low raw BRAVS due to limited innings, but their")
    out("  high-leverage contexts generate leverage multiplier boosts that")
    out("  partially compensate.")
    out()

    # Write log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[Saved to {log_path}]")


if __name__ == "__main__":
    main()
