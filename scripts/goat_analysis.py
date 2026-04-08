"""All-time career BRAVS leaderboard — greatest players ever.

Build representative peak seasons for ~15 all-time greats, compute BRAVS
for each season, sum for career estimates, and rank.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime
from baseball_metric.core.model import compute_bravs
from baseball_metric.core.types import PlayerSeason

# ---------------------------------------------------------------------------
# Player peak seasons — 3-5 representative seasons per player with
# approximate but realistic stat lines.  Goal is directional, not
# decimal-point precision.
# ---------------------------------------------------------------------------

PLAYERS: dict[str, list[PlayerSeason]] = {}

# --- Babe Ruth (RF, 1920-1933 peak) ---
PLAYERS["Babe Ruth"] = [
    PlayerSeason(
        player_id="ruthba01", player_name="Babe Ruth", season=1921, team="NYY",
        position="RF", pa=693, ab=540, hits=204, doubles=44, triples=16, hr=59,
        bb=145, ibb=0, hbp=4, k=81, sf=0, sh=4, games=152,
        sb=17, cs=13, gidp=5,
        park_factor=1.05, league_rpg=4.81, league="AL",
    ),
    PlayerSeason(
        player_id="ruthba01", player_name="Babe Ruth", season=1923, team="NYY",
        position="RF", pa=699, ab=522, hits=205, doubles=45, triples=13, hr=41,
        bb=170, ibb=0, hbp=4, k=93, sf=0, sh=3, games=152,
        sb=17, cs=21, gidp=6,
        park_factor=1.05, league_rpg=4.79, league="AL",
    ),
    PlayerSeason(
        player_id="ruthba01", player_name="Babe Ruth", season=1927, team="NYY",
        position="RF", pa=691, ab=540, hits=192, doubles=29, triples=8, hr=60,
        bb=137, ibb=0, hbp=0, k=89, sf=0, sh=14, games=151,
        sb=7, cs=6, gidp=10,
        park_factor=1.05, league_rpg=4.73, league="AL",
    ),
    PlayerSeason(
        player_id="ruthba01", player_name="Babe Ruth", season=1930, team="NYY",
        position="RF", pa=666, ab=518, hits=186, doubles=28, triples=9, hr=49,
        bb=136, ibb=0, hbp=3, k=61, sf=0, sh=9, games=145,
        sb=10, cs=10, gidp=8,
        park_factor=1.05, league_rpg=5.41, league="AL",
    ),
    PlayerSeason(
        player_id="ruthba01", player_name="Babe Ruth", season=1926, team="NYY",
        position="RF", pa=672, ab=495, hits=184, doubles=30, triples=5, hr=47,
        bb=144, ibb=0, hbp=3, k=76, sf=0, sh=10, games=152,
        sb=11, cs=9, gidp=9,
        park_factor=1.05, league_rpg=4.73, league="AL",
    ),
]

# --- Willie Mays (CF, 1954-1966 peak) ---
PLAYERS["Willie Mays"] = [
    PlayerSeason(
        player_id="mayswi01", player_name="Willie Mays", season=1954, team="NYG",
        position="CF", pa=619, ab=565, hits=195, doubles=33, triples=13, hr=41,
        bb=66, ibb=6, hbp=1, k=57, sf=7, games=151,
        sb=8, cs=4, gidp=7,
        uzr=8.0, drs=10, inn_fielded=1300.0,
        park_factor=0.96, league_rpg=4.56,
    ),
    PlayerSeason(
        player_id="mayswi01", player_name="Willie Mays", season=1962, team="SF",
        position="CF", pa=685, ab=621, hits=189, doubles=36, triples=5, hr=49,
        bb=78, ibb=18, hbp=2, k=85, sf=3, games=162,
        sb=18, cs=7, gidp=10,
        uzr=6.0, drs=8, inn_fielded=1400.0,
        park_factor=0.93, league_rpg=4.46,
    ),
    PlayerSeason(
        player_id="mayswi01", player_name="Willie Mays", season=1965, team="SF",
        position="CF", pa=659, ab=558, hits=177, doubles=21, triples=3, hr=52,
        bb=76, ibb=15, hbp=5, k=71, sf=5, games=157,
        sb=9, cs=4, gidp=9,
        uzr=5.0, drs=7, inn_fielded=1350.0,
        park_factor=0.93, league_rpg=4.03,
    ),
]

# --- Hank Aaron (RF, 1957-1973) ---
PLAYERS["Hank Aaron"] = [
    PlayerSeason(
        player_id="aaronha01", player_name="Hank Aaron", season=1959, team="MIL",
        position="RF", pa=669, ab=629, hits=223, doubles=46, triples=7, hr=39,
        bb=51, ibb=4, hbp=4, k=54, sf=9, games=154,
        sb=8, cs=2, gidp=19,
        park_factor=1.00, league_rpg=4.68,
    ),
    PlayerSeason(
        player_id="aaronha01", player_name="Hank Aaron", season=1963, team="MIL",
        position="RF", pa=689, ab=631, hits=201, doubles=29, triples=4, hr=44,
        bb=78, ibb=11, hbp=0, k=94, sf=5, games=161,
        sb=31, cs=5, gidp=14,
        park_factor=1.00, league_rpg=3.95,
    ),
    PlayerSeason(
        player_id="aaronha01", player_name="Hank Aaron", season=1971, team="ATL",
        position="RF", pa=623, ab=495, hits=162, doubles=22, triples=3, hr=47,
        bb=71, ibb=20, hbp=2, k=58, sf=5, games=139,
        sb=1, cs=1, gidp=13,
        park_factor=1.01, league_rpg=3.91,
    ),
]

# --- Barry Bonds (LF, 1990-2004) ---
PLAYERS["Barry Bonds"] = [
    PlayerSeason(
        player_id="bondsba01", player_name="Barry Bonds", season=1993, team="SF",
        position="LF", pa=674, ab=539, hits=181, doubles=38, triples=4, hr=46,
        bb=126, ibb=43, hbp=2, k=79, sf=4, sh=3, games=159,
        sb=29, cs=12, gidp=8,
        uzr=0.0, drs=1, inn_fielded=1300.0,
        park_factor=0.93, league_rpg=4.60,
    ),
    PlayerSeason(
        player_id="bondsba01", player_name="Barry Bonds", season=2001, team="SF",
        position="LF", pa=664, ab=476, hits=156, doubles=32, triples=2, hr=73,
        bb=177, ibb=35, hbp=9, k=93, sf=2, games=153,
        sb=13, cs=3, gidp=5,
        uzr=-2.0, drs=-1, inn_fielded=1200.0,
        park_factor=0.93, league_rpg=4.78,
    ),
    PlayerSeason(
        player_id="bondsba01", player_name="Barry Bonds", season=2002, team="SF",
        position="LF", pa=612, ab=403, hits=149, doubles=31, triples=2, hr=46,
        bb=198, ibb=68, hbp=9, k=47, sf=2, games=143,
        sb=9, cs=2, gidp=5,
        uzr=-3.0, drs=-2, inn_fielded=1100.0,
        park_factor=0.93, league_rpg=4.62,
    ),
    PlayerSeason(
        player_id="bondsba01", player_name="Barry Bonds", season=2004, team="SF",
        position="LF", pa=617, ab=373, hits=135, doubles=27, triples=3, hr=45,
        bb=232, ibb=120, hbp=9, k=41, sf=3, games=147,
        sb=6, cs=1, gidp=3,
        uzr=-4.0, drs=-3, inn_fielded=1000.0,
        park_factor=0.93, league_rpg=4.81,
    ),
]

# --- Mike Trout (CF, 2012-2019) ---
PLAYERS["Mike Trout"] = [
    PlayerSeason(
        player_id="troutmi01", player_name="Mike Trout", season=2012, team="LAA",
        position="CF", pa=639, ab=559, hits=182, doubles=27, triples=8, hr=30,
        bb=67, ibb=8, hbp=4, k=139, sf=7, games=139,
        sb=49, cs=5, gidp=7,
        uzr=4.0, drs=4, inn_fielded=1150.0,
        park_factor=0.98, league_rpg=4.32, league="AL",
    ),
    PlayerSeason(
        player_id="troutmi01", player_name="Mike Trout", season=2016, team="LAA",
        position="CF", pa=681, ab=549, hits=173, doubles=32, triples=5, hr=29,
        bb=116, ibb=9, hbp=7, k=137, sf=7, sh=2, games=159,
        sb=30, cs=7, gidp=9,
        uzr=1.0, drs=2, oaa=5, inn_fielded=1350.0,
        park_factor=0.98, league_rpg=4.48, league="AL",
    ),
    PlayerSeason(
        player_id="troutmi01", player_name="Mike Trout", season=2018, team="LAA",
        position="CF", pa=608, ab=471, hits=147, doubles=24, triples=4, hr=39,
        bb=122, ibb=12, hbp=6, k=124, sf=6, sh=3, games=140,
        sb=24, cs=2, gidp=7,
        uzr=2.5, drs=3, oaa=4, inn_fielded=1200.0,
        park_factor=0.98, league_rpg=4.45, league="AL",
    ),
]

# --- Ted Williams (LF, 1939-1960) ---
PLAYERS["Ted Williams"] = [
    PlayerSeason(
        player_id="willite01", player_name="Ted Williams", season=1941, team="BOS",
        position="LF", pa=606, ab=456, hits=185, doubles=33, triples=3, hr=37,
        bb=147, ibb=0, hbp=3, k=27, sf=0, games=143,
        sb=2, cs=4, gidp=12,
        park_factor=1.04, league_rpg=4.99, league="AL",
    ),
    PlayerSeason(
        player_id="willite01", player_name="Ted Williams", season=1946, team="BOS",
        position="LF", pa=672, ab=514, hits=176, doubles=37, triples=8, hr=38,
        bb=156, ibb=0, hbp=2, k=44, sf=0, games=150,
        sb=0, cs=0, gidp=14,
        park_factor=1.04, league_rpg=4.22, league="AL",
    ),
    PlayerSeason(
        player_id="willite01", player_name="Ted Williams", season=1957, team="BOS",
        position="LF", pa=547, ab=420, hits=163, doubles=28, triples=1, hr=38,
        bb=119, ibb=33, hbp=5, k=43, sf=3, games=132,
        sb=0, cs=1, gidp=9,
        park_factor=1.04, league_rpg=4.35, league="AL",
    ),
]

# --- Lou Gehrig (1B, 1926-1938) ---
PLAYERS["Lou Gehrig"] = [
    PlayerSeason(
        player_id="gehrilo01", player_name="Lou Gehrig", season=1927, team="NYY",
        position="1B", pa=717, ab=584, hits=218, doubles=52, triples=18, hr=47,
        bb=109, ibb=0, hbp=3, k=84, sf=0, sh=21, games=155,
        sb=10, cs=8, gidp=12,
        park_factor=1.05, league_rpg=4.73, league="AL",
    ),
    PlayerSeason(
        player_id="gehrilo01", player_name="Lou Gehrig", season=1930, team="NYY",
        position="1B", pa=696, ab=581, hits=220, doubles=42, triples=17, hr=41,
        bb=101, ibb=0, hbp=3, k=63, sf=0, sh=11, games=154,
        sb=12, cs=12, gidp=14,
        park_factor=1.05, league_rpg=5.41, league="AL",
    ),
    PlayerSeason(
        player_id="gehrilo01", player_name="Lou Gehrig", season=1934, team="NYY",
        position="1B", pa=676, ab=579, hits=210, doubles=40, triples=6, hr=49,
        bb=109, ibb=0, hbp=0, k=31, sf=0, sh=2, games=154,
        sb=9, cs=7, gidp=15,
        park_factor=1.05, league_rpg=5.11, league="AL",
    ),
]

# --- Stan Musial (LF/1B, 1943-1963) ---
PLAYERS["Stan Musial"] = [
    PlayerSeason(
        player_id="musiast01", player_name="Stan Musial", season=1948, team="STL",
        position="LF", pa=664, ab=611, hits=230, doubles=46, triples=18, hr=39,
        bb=79, ibb=0, hbp=0, k=34, sf=0, sh=5, games=155,
        sb=7, cs=3, gidp=16,
        park_factor=1.01, league_rpg=4.49,
    ),
    PlayerSeason(
        player_id="musiast01", player_name="Stan Musial", season=1951, team="STL",
        position="LF", pa=680, ab=578, hits=205, doubles=30, triples=12, hr=32,
        bb=98, ibb=0, hbp=1, k=40, sf=3, games=152,
        sb=4, cs=5, gidp=13,
        park_factor=1.01, league_rpg=4.63,
    ),
    PlayerSeason(
        player_id="musiast01", player_name="Stan Musial", season=1946, team="STL",
        position="1B", pa=679, ab=624, hits=228, doubles=50, triples=20, hr=16,
        bb=73, ibb=0, hbp=0, k=31, sf=0, sh=4, games=156,
        sb=7, cs=2, gidp=14,
        park_factor=1.01, league_rpg=3.97,
    ),
]

# --- Rogers Hornsby (2B, 1920-1929) ---
PLAYERS["Rogers Hornsby"] = [
    PlayerSeason(
        player_id="hornsro01", player_name="Rogers Hornsby", season=1922, team="STL",
        position="2B", pa=673, ab=623, hits=250, doubles=46, triples=14, hr=42,
        bb=65, ibb=0, hbp=1, k=50, sf=0, sh=12, games=154,
        sb=17, cs=12, gidp=10,
        park_factor=1.01, league_rpg=4.82,
    ),
    PlayerSeason(
        player_id="hornsro01", player_name="Rogers Hornsby", season=1924, team="STL",
        position="2B", pa=696, ab=536, hits=227, doubles=43, triples=14, hr=25,
        bb=89, ibb=0, hbp=2, k=32, sf=0, sh=10, games=143,
        sb=5, cs=8, gidp=8,
        park_factor=1.01, league_rpg=4.82,
    ),
    PlayerSeason(
        player_id="hornsro01", player_name="Rogers Hornsby", season=1925, team="STL",
        position="2B", pa=637, ab=504, hits=203, doubles=41, triples=10, hr=39,
        bb=83, ibb=0, hbp=2, k=39, sf=0, sh=7, games=138,
        sb=5, cs=4, gidp=9,
        park_factor=1.01, league_rpg=5.06,
    ),
]

# --- Honus Wagner (SS, 1900-1913) ---
PLAYERS["Honus Wagner"] = [
    PlayerSeason(
        player_id="wagneho01", player_name="Honus Wagner", season=1908, team="PIT",
        position="SS", pa=615, ab=568, hits=201, doubles=39, triples=19, hr=10,
        bb=54, ibb=0, hbp=1, k=0, sf=0, sh=17, games=151,
        sb=53, cs=20, gidp=6,
        park_factor=0.98, league_rpg=3.37,
    ),
    PlayerSeason(
        player_id="wagneho01", player_name="Honus Wagner", season=1905, team="PIT",
        position="SS", pa=620, ab=548, hits=199, doubles=32, triples=14, hr=6,
        bb=54, ibb=0, hbp=3, k=0, sf=0, sh=15, games=147,
        sb=57, cs=22, gidp=5,
        park_factor=0.98, league_rpg=3.73,
    ),
    PlayerSeason(
        player_id="wagneho01", player_name="Honus Wagner", season=1911, team="PIT",
        position="SS", pa=601, ab=473, hits=158, doubles=35, triples=16, hr=9,
        bb=67, ibb=0, hbp=2, k=0, sf=0, sh=12, games=130,
        sb=35, cs=16, gidp=6,
        park_factor=0.98, league_rpg=4.38,
    ),
]

# --- Mickey Mantle (CF, 1952-1964) ---
PLAYERS["Mickey Mantle"] = [
    PlayerSeason(
        player_id="mantlmi01", player_name="Mickey Mantle", season=1956, team="NYY",
        position="CF", pa=650, ab=533, hits=188, doubles=22, triples=5, hr=52,
        bb=112, ibb=14, hbp=1, k=99, sf=4, games=150,
        sb=10, cs=1, gidp=8,
        uzr=3.0, drs=4, inn_fielded=1300.0,
        park_factor=1.05, league_rpg=4.61, league="AL",
    ),
    PlayerSeason(
        player_id="mantlmi01", player_name="Mickey Mantle", season=1957, team="NYY",
        position="CF", pa=654, ab=474, hits=173, doubles=28, triples=6, hr=34,
        bb=146, ibb=18, hbp=0, k=75, sf=4, games=144,
        sb=16, cs=3, gidp=6,
        uzr=2.0, drs=3, inn_fielded=1250.0,
        park_factor=1.05, league_rpg=4.26, league="AL",
    ),
    PlayerSeason(
        player_id="mantlmi01", player_name="Mickey Mantle", season=1961, team="NYY",
        position="CF", pa=646, ab=514, hits=163, doubles=16, triples=6, hr=54,
        bb=126, ibb=9, hbp=2, k=112, sf=4, games=153,
        sb=12, cs=1, gidp=7,
        uzr=1.0, drs=2, inn_fielded=1300.0,
        park_factor=1.05, league_rpg=4.53, league="AL",
    ),
]

# --- Pedro Martinez (P, 1997-2003) ---
PLAYERS["Pedro Martinez"] = [
    PlayerSeason(
        player_id="martipe02", player_name="Pedro Martinez", season=1997, team="MON",
        position="P", ip=241.3, er=65, hits_allowed=158, hr_allowed=16,
        bb_allowed=67, hbp_allowed=9, k_pitching=305, games_pitched=31,
        games_started=31, park_factor=0.98, league_rpg=4.60,
    ),
    PlayerSeason(
        player_id="martipe02", player_name="Pedro Martinez", season=1999, team="BOS",
        position="P", ip=213.3, er=49, hits_allowed=160, hr_allowed=9,
        bb_allowed=37, hbp_allowed=9, k_pitching=313, games_pitched=31,
        games_started=31, park_factor=1.04, league_rpg=5.08, league="AL",
    ),
    PlayerSeason(
        player_id="martipe02", player_name="Pedro Martinez", season=2000, team="BOS",
        position="P", ip=217.0, er=42, hits_allowed=128, hr_allowed=17,
        bb_allowed=32, hbp_allowed=6, k_pitching=284, games_pitched=29,
        games_started=29, park_factor=1.04, league_rpg=5.14, league="AL",
    ),
]

# --- Walter Johnson (P, 1910-1919) ---
PLAYERS["Walter Johnson"] = [
    PlayerSeason(
        player_id="johnswa01", player_name="Walter Johnson", season=1912, team="WSH",
        position="P", ip=369.0, er=69, hits_allowed=259, hr_allowed=4,
        bb_allowed=76, hbp_allowed=11, k_pitching=303, games_pitched=50,
        games_started=37, park_factor=0.97, league_rpg=4.10, league="AL",
    ),
    PlayerSeason(
        player_id="johnswa01", player_name="Walter Johnson", season=1913, team="WSH",
        position="P", ip=346.0, er=56, hits_allowed=232, hr_allowed=2,
        bb_allowed=38, hbp_allowed=9, k_pitching=243, games_pitched=48,
        games_started=36, park_factor=0.97, league_rpg=3.93, league="AL",
    ),
    PlayerSeason(
        player_id="johnswa01", player_name="Walter Johnson", season=1919, team="WSH",
        position="P", ip=290.3, er=51, hits_allowed=235, hr_allowed=2,
        bb_allowed=51, hbp_allowed=5, k_pitching=147, games_pitched=39,
        games_started=29, park_factor=0.97, league_rpg=3.61, league="AL",
    ),
]

# --- Greg Maddux (P, 1992-1998) ---
PLAYERS["Greg Maddux"] = [
    PlayerSeason(
        player_id="maddugr01", player_name="Greg Maddux", season=1994, team="ATL",
        position="P", ip=202.0, er=35, hits_allowed=150, hr_allowed=4,
        bb_allowed=31, hbp_allowed=2, k_pitching=156, games_pitched=25,
        games_started=25, park_factor=1.00, league_rpg=4.63,
    ),
    PlayerSeason(
        player_id="maddugr01", player_name="Greg Maddux", season=1995, team="ATL",
        position="P", ip=209.7, er=38, hits_allowed=147, hr_allowed=8,
        bb_allowed=23, hbp_allowed=4, k_pitching=181, games_pitched=28,
        games_started=28, park_factor=1.00, league_rpg=4.63,
    ),
    PlayerSeason(
        player_id="maddugr01", player_name="Greg Maddux", season=1997, team="ATL",
        position="P", ip=232.7, er=52, hits_allowed=200, hr_allowed=9,
        bb_allowed=20, hbp_allowed=4, k_pitching=177, games_pitched=33,
        games_started=33, park_factor=1.00, league_rpg=4.77,
    ),
]

# --- Randy Johnson (P, 1993-2002) ---
PLAYERS["Randy Johnson"] = [
    PlayerSeason(
        player_id="johnsra05", player_name="Randy Johnson", season=1995, team="SEA",
        position="P", ip=214.3, er=59, hits_allowed=159, hr_allowed=12,
        bb_allowed=65, hbp_allowed=6, k_pitching=294, games_pitched=30,
        games_started=30, park_factor=0.97, league_rpg=5.06, league="AL",
    ),
    PlayerSeason(
        player_id="johnsra05", player_name="Randy Johnson", season=1999, team="ARI",
        position="P", ip=271.7, er=66, hits_allowed=207, hr_allowed=20,
        bb_allowed=70, hbp_allowed=11, k_pitching=364, games_pitched=35,
        games_started=35, park_factor=1.04, league_rpg=5.04,
    ),
    PlayerSeason(
        player_id="johnsra05", player_name="Randy Johnson", season=2001, team="ARI",
        position="P", ip=249.7, er=64, hits_allowed=181, hr_allowed=19,
        bb_allowed=71, hbp_allowed=18, k_pitching=372, games_pitched=35,
        games_started=35, park_factor=1.04, league_rpg=4.78,
    ),
]


def main() -> None:
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "logs"), exist_ok=True)
    log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "goat_analysis.log")

    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    out("=" * 100)
    out("  ALL-TIME CAREER BRAVS LEADERBOARD")
    out(f"  Generated {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    out("=" * 100)
    out()

    # Compute BRAVS for every player-season
    career_data: dict[str, list[tuple[int, float, float, float]]] = {}

    for name, seasons in PLAYERS.items():
        career_data[name] = []
        for ps in seasons:
            result = compute_bravs(ps, fast=True)
            career_data[name].append((
                ps.season,
                result.bravs,
                result.bravs_era_standardized,
                result.bravs_calibrated,
            ))

    # --- Per-player season breakdown ---
    out("-" * 100)
    out(f"  {'Player':<22}{'Year':>6}{'BRAVS':>8}{'ErStd':>8}{'WAReq':>8}")
    out("-" * 100)

    for name, season_list in sorted(PLAYERS.items()):
        for yr, bravs, era_std, war_eq in career_data[name]:
            out(f"  {name:<22}{yr:>6}{bravs:>8.1f}{era_std:>8.1f}{war_eq:>8.1f}")
        out()

    # --- Career totals (sum of representative peak seasons) ---
    out()
    out("=" * 100)
    out("  CAREER BRAVS ESTIMATES (sum of peak seasons modeled)")
    out("  Note: This sums only 3-5 representative peak seasons per player.")
    out("  Actual career totals would be higher with full season coverage.")
    out("=" * 100)
    out()

    career_totals: list[tuple[str, float, float, float, int]] = []
    for name, season_list in career_data.items():
        total_bravs = sum(b for _, b, _, _ in season_list)
        total_era_std = sum(e for _, _, e, _ in season_list)
        total_war_eq = sum(w for _, _, _, w in season_list)
        n_seasons = len(season_list)
        career_totals.append((name, total_bravs, total_era_std, total_war_eq, n_seasons))

    # Sort by era-standardized BRAVS (fairest cross-era comparison)
    career_totals.sort(key=lambda x: x[2], reverse=True)

    out(f"{'Rank':>4}  {'Player':<22}{'Seasons':>8}{'CarBRAVS':>10}{'CarErStd':>10}{'CarWAReq':>10}{'Avg/Szn':>10}")
    out("-" * 100)

    for i, (name, tot_b, tot_e, tot_w, n) in enumerate(career_totals, 1):
        avg_per = tot_e / n
        out(f"{i:>4}. {name:<22}{n:>8}{tot_b:>10.1f}{tot_e:>10.1f}{tot_w:>10.1f}{avg_per:>10.1f}")

    out()
    out("=" * 100)
    out("  TIER ANALYSIS")
    out("=" * 100)
    out()

    # Group into tiers by avg per season (era-standardized)
    tier1, tier2, tier3 = [], [], []
    for name, tot_b, tot_e, tot_w, n in career_totals:
        avg = tot_e / n
        if avg >= 8.0:
            tier1.append((name, avg))
        elif avg >= 5.0:
            tier2.append((name, avg))
        else:
            tier3.append((name, avg))

    out("  TIER 1 — Transcendent (8+ era-std BRAVS per peak season):")
    for name, avg in tier1:
        out(f"    {name:<22} ({avg:.1f} avg)")
    out()

    out("  TIER 2 — Inner Circle (5-8 era-std BRAVS per peak season):")
    for name, avg in tier2:
        out(f"    {name:<22} ({avg:.1f} avg)")
    out()

    out("  TIER 3 — Elite (below 5 era-std BRAVS per peak season):")
    for name, avg in tier3:
        out(f"    {name:<22} ({avg:.1f} avg)")
    out()

    # --- Position player vs pitcher analysis ---
    out("=" * 100)
    out("  POSITION PLAYER vs PITCHER PEAK VALUE")
    out("=" * 100)
    out()

    pitchers = {"Pedro Martinez", "Walter Johnson", "Greg Maddux", "Randy Johnson"}
    hitter_avgs = [(n, t/s) for n, _, t, _, s in career_totals if n not in pitchers]
    pitcher_avgs = [(n, t/s) for n, _, t, _, s in career_totals if n in pitchers]

    out("  Best peak season avg (era-standardized) among hitters:")
    for name, avg in sorted(hitter_avgs, key=lambda x: x[1], reverse=True)[:5]:
        out(f"    {name:<22} {avg:.1f} BRAVS/season")

    out()
    out("  Best peak season avg (era-standardized) among pitchers:")
    for name, avg in sorted(pitcher_avgs, key=lambda x: x[1], reverse=True):
        out(f"    {name:<22} {avg:.1f} BRAVS/season")

    out()
    out("  Note: Pitchers produce fewer BRAVS per season because they only contribute")
    out("  via the pitching component (no hitting, baserunning, or approach quality).")
    out("  This is inherent in the framework and reflects real-world value distribution.")
    out()

    # Write log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[Saved to {log_path}]")


if __name__ == "__main__":
    main()
