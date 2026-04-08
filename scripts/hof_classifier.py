"""Hall of Fame classifier — can BRAVS separate HOFers from non-HOFers?

Creates two groups (HOF vs non-HOF), computes career BRAVS estimates
for each, and evaluates classification performance with AUC.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime
import numpy as np
from baseball_metric.core.model import compute_bravs
from baseball_metric.core.types import PlayerSeason

# ---------------------------------------------------------------------------
# Group 1: Clear Hall of Famers (15 players, 2-3 peak seasons each)
# ---------------------------------------------------------------------------
HOF: dict[str, list[PlayerSeason]] = {}

HOF["Babe Ruth"] = [
    PlayerSeason(player_id="ruthba01", player_name="Babe Ruth", season=1921, team="NYY",
        position="RF", pa=693, ab=540, hits=204, doubles=44, triples=16, hr=59,
        bb=145, ibb=0, hbp=4, k=81, games=152, sb=17, cs=13, gidp=5,
        park_factor=1.05, league_rpg=4.81, league="AL"),
    PlayerSeason(player_id="ruthba01", player_name="Babe Ruth", season=1927, team="NYY",
        position="RF", pa=691, ab=540, hits=192, doubles=29, triples=8, hr=60,
        bb=137, ibb=0, hbp=0, k=89, sh=14, games=151, sb=7, cs=6, gidp=10,
        park_factor=1.05, league_rpg=4.73, league="AL"),
]

HOF["Willie Mays"] = [
    PlayerSeason(player_id="mayswi01", player_name="Willie Mays", season=1965, team="SF",
        position="CF", pa=659, ab=558, hits=177, doubles=21, triples=3, hr=52,
        bb=76, ibb=15, hbp=5, k=71, sf=5, games=157, sb=9, cs=4, gidp=9,
        uzr=5.0, drs=7, inn_fielded=1350.0, park_factor=0.93, league_rpg=4.03),
    PlayerSeason(player_id="mayswi01", player_name="Willie Mays", season=1962, team="SF",
        position="CF", pa=685, ab=621, hits=189, doubles=36, triples=5, hr=49,
        bb=78, ibb=18, hbp=2, k=85, sf=3, games=162, sb=18, cs=7, gidp=10,
        uzr=6.0, drs=8, inn_fielded=1400.0, park_factor=0.93, league_rpg=4.46),
]

HOF["Hank Aaron"] = [
    PlayerSeason(player_id="aaronha01", player_name="Hank Aaron", season=1963, team="MIL",
        position="RF", pa=689, ab=631, hits=201, doubles=29, triples=4, hr=44,
        bb=78, ibb=11, k=94, sf=5, games=161, sb=31, cs=5, gidp=14,
        park_factor=1.00, league_rpg=3.95),
    PlayerSeason(player_id="aaronha01", player_name="Hank Aaron", season=1971, team="ATL",
        position="RF", pa=623, ab=495, hits=162, doubles=22, triples=3, hr=47,
        bb=71, ibb=20, hbp=2, k=58, sf=5, games=139, sb=1, cs=1, gidp=13,
        park_factor=1.01, league_rpg=3.91),
]

HOF["Barry Bonds*"] = [
    PlayerSeason(player_id="bondsba01", player_name="Barry Bonds", season=2001, team="SF",
        position="LF", pa=664, ab=476, hits=156, doubles=32, triples=2, hr=73,
        bb=177, ibb=35, hbp=9, k=93, sf=2, games=153, sb=13, cs=3, gidp=5,
        uzr=-2.0, drs=-1, inn_fielded=1200.0, park_factor=0.93, league_rpg=4.78),
    PlayerSeason(player_id="bondsba01", player_name="Barry Bonds", season=1993, team="SF",
        position="LF", pa=674, ab=539, hits=181, doubles=38, triples=4, hr=46,
        bb=126, ibb=43, hbp=2, k=79, sf=4, sh=3, games=159, sb=29, cs=12, gidp=8,
        uzr=0.0, drs=1, inn_fielded=1300.0, park_factor=0.93, league_rpg=4.60),
]

HOF["Mike Trout*"] = [
    PlayerSeason(player_id="troutmi01", player_name="Mike Trout", season=2016, team="LAA",
        position="CF", pa=681, ab=549, hits=173, doubles=32, triples=5, hr=29,
        bb=116, ibb=9, hbp=7, k=137, sf=7, sh=2, games=159, sb=30, cs=7, gidp=9,
        uzr=1.0, drs=2, oaa=5, inn_fielded=1350.0,
        park_factor=0.98, league_rpg=4.48, league="AL"),
    PlayerSeason(player_id="troutmi01", player_name="Mike Trout", season=2018, team="LAA",
        position="CF", pa=608, ab=471, hits=147, doubles=24, triples=4, hr=39,
        bb=122, ibb=12, hbp=6, k=124, sf=6, sh=3, games=140, sb=24, cs=2, gidp=7,
        uzr=2.5, drs=3, oaa=4, inn_fielded=1200.0,
        park_factor=0.98, league_rpg=4.45, league="AL"),
]

HOF["Ted Williams"] = [
    PlayerSeason(player_id="willite01", player_name="Ted Williams", season=1941, team="BOS",
        position="LF", pa=606, ab=456, hits=185, doubles=33, triples=3, hr=37,
        bb=147, ibb=0, hbp=3, k=27, games=143, sb=2, cs=4, gidp=12,
        park_factor=1.04, league_rpg=4.99, league="AL"),
    PlayerSeason(player_id="willite01", player_name="Ted Williams", season=1946, team="BOS",
        position="LF", pa=672, ab=514, hits=176, doubles=37, triples=8, hr=38,
        bb=156, ibb=0, hbp=2, k=44, games=150, sb=0, cs=0, gidp=14,
        park_factor=1.04, league_rpg=4.22, league="AL"),
]

HOF["Ken Griffey Jr"] = [
    PlayerSeason(player_id="griffke02", player_name="Ken Griffey Jr", season=1997, team="SEA",
        position="CF", pa=674, ab=608, hits=185, doubles=34, triples=3, hr=56,
        bb=76, ibb=11, hbp=3, k=121, sf=5, games=157, sb=15, cs=4, gidp=13,
        uzr=4.0, drs=5, inn_fielded=1350.0,
        park_factor=0.97, league_rpg=4.77, league="AL"),
    PlayerSeason(player_id="griffke02", player_name="Ken Griffey Jr", season=1996, team="SEA",
        position="CF", pa=618, ab=545, hits=165, doubles=26, triples=2, hr=49,
        bb=78, ibb=7, hbp=2, k=104, sf=2, games=140, sb=16, cs=5, gidp=12,
        uzr=6.0, drs=7, inn_fielded=1200.0,
        park_factor=0.97, league_rpg=5.39, league="AL"),
]

HOF["Cal Ripken"] = [
    PlayerSeason(player_id="ripkeca01", player_name="Cal Ripken", season=1991, team="BAL",
        position="SS", pa=680, ab=650, hits=210, doubles=46, triples=5, hr=34,
        bb=53, ibb=6, hbp=5, k=46, sf=1, games=162, sb=6, cs=1, gidp=14,
        uzr=8.0, drs=10, inn_fielded=1440.0,
        park_factor=1.01, league_rpg=4.32, league="AL"),
    PlayerSeason(player_id="ripkeca01", player_name="Cal Ripken", season=1984, team="BAL",
        position="SS", pa=685, ab=641, hits=195, doubles=37, triples=7, hr=27,
        bb=71, ibb=6, hbp=2, k=89, sf=6, games=162, sb=2, cs=1, gidp=16,
        uzr=5.0, drs=7, inn_fielded=1440.0,
        park_factor=1.01, league_rpg=4.47, league="AL"),
]

HOF["Derek Jeter"] = [
    PlayerSeason(player_id="jeterde01", player_name="Derek Jeter", season=1999, team="NYY",
        position="SS", pa=739, ab=627, hits=219, doubles=37, triples=9, hr=24,
        bb=91, ibb=8, hbp=14, k=116, sf=7, games=158, sb=19, cs=8, gidp=15,
        uzr=-5.0, drs=-8, inn_fielded=1380.0,
        park_factor=1.05, league_rpg=5.08, league="AL"),
    PlayerSeason(player_id="jeterde01", player_name="Derek Jeter", season=2006, team="NYY",
        position="SS", pa=715, ab=623, hits=214, doubles=39, triples=3, hr=14,
        bb=69, ibb=6, hbp=12, k=102, sf=5, games=154, sb=34, cs=5, gidp=15,
        uzr=-7.0, drs=-12, inn_fielded=1350.0,
        park_factor=1.05, league_rpg=4.86, league="AL"),
]

HOF["Roger Clemens*"] = [
    PlayerSeason(player_id="clemero02", player_name="Roger Clemens", season=1997, team="TOR",
        position="P", ip=264.0, er=65, hits_allowed=204, hr_allowed=19,
        bb_allowed=68, hbp_allowed=8, k_pitching=292, games_pitched=34,
        games_started=34, park_factor=1.03, league_rpg=4.77, league="AL"),
    PlayerSeason(player_id="clemero02", player_name="Roger Clemens", season=2005, team="HOU",
        position="P", ip=211.3, er=44, hits_allowed=151, hr_allowed=11,
        bb_allowed=62, hbp_allowed=3, k_pitching=185, games_pitched=32,
        games_started=32, park_factor=1.02, league_rpg=4.45),
]

HOF["Pedro Martinez"] = [
    PlayerSeason(player_id="martipe02", player_name="Pedro Martinez", season=1999, team="BOS",
        position="P", ip=213.3, er=49, hits_allowed=160, hr_allowed=9,
        bb_allowed=37, hbp_allowed=9, k_pitching=313, games_pitched=31,
        games_started=31, park_factor=1.04, league_rpg=5.08, league="AL"),
    PlayerSeason(player_id="martipe02", player_name="Pedro Martinez", season=2000, team="BOS",
        position="P", ip=217.0, er=42, hits_allowed=128, hr_allowed=17,
        bb_allowed=32, hbp_allowed=6, k_pitching=284, games_pitched=29,
        games_started=29, park_factor=1.04, league_rpg=5.14, league="AL"),
]

HOF["Randy Johnson"] = [
    PlayerSeason(player_id="johnsra05", player_name="Randy Johnson", season=2001, team="ARI",
        position="P", ip=249.7, er=64, hits_allowed=181, hr_allowed=19,
        bb_allowed=71, hbp_allowed=18, k_pitching=372, games_pitched=35,
        games_started=35, park_factor=1.04, league_rpg=4.78),
    PlayerSeason(player_id="johnsra05", player_name="Randy Johnson", season=1999, team="ARI",
        position="P", ip=271.7, er=66, hits_allowed=207, hr_allowed=20,
        bb_allowed=70, hbp_allowed=11, k_pitching=364, games_pitched=35,
        games_started=35, park_factor=1.04, league_rpg=5.04),
]

HOF["Greg Maddux"] = [
    PlayerSeason(player_id="maddugr01", player_name="Greg Maddux", season=1995, team="ATL",
        position="P", ip=209.7, er=38, hits_allowed=147, hr_allowed=8,
        bb_allowed=23, hbp_allowed=4, k_pitching=181, games_pitched=28,
        games_started=28, park_factor=1.00, league_rpg=4.63),
    PlayerSeason(player_id="maddugr01", player_name="Greg Maddux", season=1994, team="ATL",
        position="P", ip=202.0, er=35, hits_allowed=150, hr_allowed=4,
        bb_allowed=31, hbp_allowed=2, k_pitching=156, games_pitched=25,
        games_started=25, park_factor=1.00, league_rpg=4.63),
]

HOF["Mariano Rivera"] = [
    PlayerSeason(player_id="riverma01", player_name="Mariano Rivera", season=2004, team="NYY",
        position="P", ip=78.7, er=15, hits_allowed=65, hr_allowed=3,
        bb_allowed=20, hbp_allowed=2, k_pitching=66, games_pitched=74,
        games_started=0, saves=53, holds=0,
        avg_leverage_index=2.10,
        park_factor=1.05, league_rpg=4.83, league="AL"),
    PlayerSeason(player_id="riverma01", player_name="Mariano Rivera", season=2008, team="NYY",
        position="P", ip=70.7, er=8, hits_allowed=41, hr_allowed=3,
        bb_allowed=6, hbp_allowed=1, k_pitching=77, games_pitched=64,
        games_started=0, saves=39, holds=0,
        avg_leverage_index=2.05,
        park_factor=1.05, league_rpg=4.65, league="AL"),
]

HOF["Mike Piazza"] = [
    PlayerSeason(player_id="piazzmi01", player_name="Mike Piazza", season=1997, team="LAD",
        position="C", pa=633, ab=556, hits=201, doubles=32, triples=1, hr=40,
        bb=69, ibb=7, hbp=2, k=77, sf=6, games=152, sb=5, cs=1, gidp=14,
        framing_runs=5.0, blocking_runs=0.5, throwing_runs=-2.0,
        inn_fielded=1200.0,
        park_factor=0.97, league_rpg=4.77),
    PlayerSeason(player_id="piazzmi01", player_name="Mike Piazza", season=2000, team="NYM",
        position="C", pa=538, ab=482, hits=156, doubles=26, triples=0, hr=38,
        bb=58, ibb=14, hbp=4, k=69, sf=3, games=136, sb=4, cs=2, gidp=13,
        framing_runs=3.0, blocking_runs=0.0, throwing_runs=-3.0,
        inn_fielded=1050.0,
        park_factor=0.95, league_rpg=5.14),
]


# ---------------------------------------------------------------------------
# Group 2: Non-Hall of Famers / borderline / controversial (15 players)
# ---------------------------------------------------------------------------
NON_HOF: dict[str, list[PlayerSeason]] = {}

NON_HOF["Harold Baines"] = [
    PlayerSeason(player_id="baineh01", player_name="Harold Baines", season=1984, team="CHW",
        position="DH", pa=616, ab=569, hits=173, doubles=28, triples=10, hr=29,
        bb=54, ibb=6, hbp=2, k=63, sf=4, games=147, sb=1, cs=2, gidp=14,
        park_factor=0.97, league_rpg=4.29, league="AL"),
    PlayerSeason(player_id="baineh01", player_name="Harold Baines", season=1987, team="CHW",
        position="DH", pa=633, ab=505, hits=148, doubles=26, triples=4, hr=20,
        bb=73, ibb=9, hbp=3, k=57, sf=3, games=132, sb=0, cs=1, gidp=18,
        park_factor=0.97, league_rpg=4.71, league="AL"),
]

NON_HOF["Fred McGriff"] = [
    PlayerSeason(player_id="mcgrifr01", player_name="Fred McGriff", season=1992, team="SD",
        position="1B", pa=603, ab=531, hits=152, doubles=30, triples=4, hr=35,
        bb=96, ibb=14, hbp=1, k=108, sf=6, games=153, sb=8, cs=3, gidp=12,
        park_factor=0.95, league_rpg=3.88),
    PlayerSeason(player_id="mcgrifr01", player_name="Fred McGriff", season=1994, team="ATL",
        position="1B", pa=472, ab=424, hits=135, doubles=25, triples=1, hr=34,
        bb=50, ibb=6, hbp=2, k=76, sf=5, games=113, sb=7, cs=3, gidp=10,
        park_factor=1.00, league_rpg=4.63),
]

NON_HOF["Bobby Abreu"] = [
    PlayerSeason(player_id="abreubo01", player_name="Bobby Abreu", season=2004, team="PHI",
        position="RF", pa=700, ab=574, hits=173, doubles=47, triples=1, hr=30,
        bb=127, ibb=12, hbp=8, k=117, sf=5, games=159, sb=40, cs=5, gidp=12,
        uzr=3.0, drs=2, inn_fielded=1350.0,
        park_factor=1.03, league_rpg=4.81),
    PlayerSeason(player_id="abreubo01", player_name="Bobby Abreu", season=2005, team="PHI",
        position="RF", pa=719, ab=588, hits=168, doubles=37, triples=1, hr=24,
        bb=117, ibb=12, hbp=7, k=128, sf=7, games=162, sb=31, cs=10, gidp=12,
        uzr=1.0, drs=0, inn_fielded=1400.0,
        park_factor=1.03, league_rpg=4.52),
]

NON_HOF["Andruw Jones"] = [
    PlayerSeason(player_id="jonesan01", player_name="Andruw Jones", season=2005, team="ATL",
        position="CF", pa=631, ab=586, hits=154, doubles=24, triples=5, hr=51,
        bb=64, ibb=16, hbp=8, k=115, sf=2, games=160, sb=5, cs=3, gidp=16,
        uzr=18.0, drs=21, inn_fielded=1380.0,
        park_factor=1.00, league_rpg=4.52),
    PlayerSeason(player_id="jonesan01", player_name="Andruw Jones", season=1998, team="ATL",
        position="CF", pa=635, ab=582, hits=158, doubles=29, triples=8, hr=31,
        bb=40, ibb=5, hbp=6, k=129, sf=3, games=159, sb=27, cs=6, gidp=11,
        uzr=14.0, drs=17, inn_fielded=1370.0,
        park_factor=1.00, league_rpg=4.59),
]

NON_HOF["Kenny Lofton"] = [
    PlayerSeason(player_id="loftoken01", player_name="Kenny Lofton", season=1994, team="CLE",
        position="CF", pa=536, ab=459, hits=160, doubles=32, triples=9, hr=12,
        bb=52, ibb=3, hbp=5, k=56, sf=3, games=112, sb=60, cs=12, gidp=6,
        uzr=8.0, drs=10, inn_fielded=950.0,
        park_factor=1.01, league_rpg=5.23, league="AL"),
    PlayerSeason(player_id="loftoken01", player_name="Kenny Lofton", season=1996, team="CLE",
        position="CF", pa=728, ab=662, hits=210, doubles=35, triples=4, hr=14,
        bb=61, ibb=2, hbp=4, k=72, sf=1, games=154, sb=75, cs=17, gidp=9,
        uzr=6.0, drs=8, inn_fielded=1350.0,
        park_factor=1.01, league_rpg=5.39, league="AL"),
]

NON_HOF["Jim Edmonds"] = [
    PlayerSeason(player_id="edmonji01", player_name="Jim Edmonds", season=2004, team="STL",
        position="CF", pa=598, ab=498, hits=150, doubles=38, triples=1, hr=42,
        bb=101, ibb=16, hbp=4, k=150, sf=6, games=153, sb=4, cs=2, gidp=9,
        uzr=10.0, drs=12, inn_fielded=1300.0,
        park_factor=0.99, league_rpg=4.81),
    PlayerSeason(player_id="edmonji01", player_name="Jim Edmonds", season=2000, team="STL",
        position="CF", pa=624, ab=525, hits=155, doubles=25, triples=0, hr=42,
        bb=103, ibb=19, hbp=7, k=167, sf=4, games=152, sb=10, cs=2, gidp=7,
        uzr=8.0, drs=9, inn_fielded=1320.0,
        park_factor=0.99, league_rpg=5.14),
]

NON_HOF["Scott Rolen"] = [
    PlayerSeason(player_id="rolens01", player_name="Scott Rolen", season=2004, team="STL",
        position="3B", pa=604, ab=500, hits=124, doubles=32, triples=1, hr=34,
        bb=74, ibb=5, hbp=13, k=96, sf=5, games=142, sb=3, cs=1, gidp=9,
        uzr=15.0, drs=18, inn_fielded=1230.0,
        park_factor=0.99, league_rpg=4.81),
    PlayerSeason(player_id="rolens01", player_name="Scott Rolen", season=2002, team="PHI",
        position="3B", pa=554, ab=474, hits=130, doubles=28, triples=1, hr=31,
        bb=71, ibb=5, hbp=10, k=112, sf=3, games=155, sb=4, cs=2, gidp=7,
        uzr=12.0, drs=14, inn_fielded=1340.0,
        park_factor=1.03, league_rpg=4.63),
]

NON_HOF["Larry Walker"] = [
    PlayerSeason(player_id="walkela01", player_name="Larry Walker", season=1997, team="COL",
        position="RF", pa=613, ab=568, hits=208, doubles=46, triples=4, hr=49,
        bb=78, ibb=13, hbp=5, k=90, sf=4, games=153, sb=33, cs=6, gidp=8,
        uzr=5.0, drs=6, inn_fielded=1300.0,
        park_factor=1.18, league_rpg=4.77),
    PlayerSeason(player_id="walkela01", player_name="Larry Walker", season=2001, team="COL",
        position="RF", pa=497, ab=437, hits=174, doubles=35, triples=3, hr=38,
        bb=49, ibb=7, hbp=14, k=73, sf=5, games=142, sb=14, cs=5, gidp=10,
        uzr=3.0, drs=4, inn_fielded=1100.0,
        park_factor=1.18, league_rpg=4.78),
]

NON_HOF["David Ortiz"] = [
    PlayerSeason(player_id="ortizda01", player_name="David Ortiz", season=2006, team="BOS",
        position="DH", pa=686, ab=558, hits=160, doubles=29, triples=2, hr=54,
        bb=119, ibb=16, hbp=1, k=117, sf=5, games=151, sb=1, cs=0, gidp=14,
        park_factor=1.04, league_rpg=4.86, league="AL"),
    PlayerSeason(player_id="ortizda01", player_name="David Ortiz", season=2016, team="BOS",
        position="DH", pa=614, ab=537, hits=169, doubles=48, triples=1, hr=38,
        bb=80, ibb=14, hbp=4, k=86, sf=4, games=151, sb=2, cs=0, gidp=19,
        park_factor=1.04, league_rpg=4.40, league="AL"),
]

NON_HOF["Mark McGwire"] = [
    PlayerSeason(player_id="mcgwima01", player_name="Mark McGwire", season=1998, team="STL",
        position="1B", pa=681, ab=509, hits=152, doubles=21, triples=0, hr=70,
        bb=162, ibb=28, hbp=6, k=155, sf=4, games=155, sb=1, cs=0, gidp=8,
        park_factor=0.99, league_rpg=4.59),
    PlayerSeason(player_id="mcgwima01", player_name="Mark McGwire", season=1996, team="OAK",
        position="1B", pa=548, ab=423, hits=132, doubles=21, triples=0, hr=52,
        bb=116, ibb=16, hbp=6, k=112, sf=3, games=130, sb=0, cs=1, gidp=7,
        park_factor=0.98, league_rpg=5.39, league="AL"),
]

NON_HOF["Sammy Sosa"] = [
    PlayerSeason(player_id="sosasa01", player_name="Sammy Sosa", season=2001, team="CHC",
        position="RF", pa=711, ab=577, hits=189, doubles=34, triples=5, hr=64,
        bb=116, ibb=22, hbp=8, k=153, sf=7, games=160, sb=0, cs=2, gidp=18,
        uzr=-5.0, drs=-6, inn_fielded=1350.0,
        park_factor=1.03, league_rpg=4.78),
    PlayerSeason(player_id="sosasa01", player_name="Sammy Sosa", season=1998, team="CHC",
        position="RF", pa=693, ab=643, hits=198, doubles=20, triples=0, hr=66,
        bb=73, ibb=14, hbp=1, k=171, sf=5, games=159, sb=18, cs=9, gidp=20,
        uzr=-3.0, drs=-4, inn_fielded=1370.0,
        park_factor=1.03, league_rpg=4.59),
]

NON_HOF["Rafael Palmeiro"] = [
    PlayerSeason(player_id="palmera01", player_name="Rafael Palmeiro", season=1999, team="TEX",
        position="1B", pa=700, ab=565, hits=183, doubles=30, triples=1, hr=47,
        bb=97, ibb=14, hbp=3, k=75, sf=4, games=158, sb=2, cs=2, gidp=21,
        park_factor=1.04, league_rpg=5.18, league="AL"),
    PlayerSeason(player_id="palmera01", player_name="Rafael Palmeiro", season=2001, team="TEX",
        position="1B", pa=661, ab=600, hits=164, doubles=33, triples=0, hr=47,
        bb=64, ibb=8, hbp=5, k=91, sf=3, games=160, sb=2, cs=1, gidp=15,
        park_factor=1.04, league_rpg=4.78, league="AL"),
]

NON_HOF["Johan Santana"] = [
    PlayerSeason(player_id="santajo02", player_name="Johan Santana", season=2004, team="MIN",
        position="P", ip=228.0, er=66, hits_allowed=156, hr_allowed=24,
        bb_allowed=54, hbp_allowed=5, k_pitching=265, games_pitched=34,
        games_started=34, park_factor=1.01, league_rpg=4.83, league="AL"),
    PlayerSeason(player_id="santajo02", player_name="Johan Santana", season=2006, team="MIN",
        position="P", ip=233.7, er=63, hits_allowed=186, hr_allowed=20,
        bb_allowed=47, hbp_allowed=8, k_pitching=245, games_pitched=34,
        games_started=34, park_factor=1.01, league_rpg=4.86, league="AL"),
]

NON_HOF["Roy Halladay"] = [
    PlayerSeason(player_id="hallaro01", player_name="Roy Halladay", season=2010, team="PHI",
        position="P", ip=250.7, er=62, hits_allowed=231, hr_allowed=16,
        bb_allowed=30, hbp_allowed=4, k_pitching=219, games_pitched=33,
        games_started=33, park_factor=1.03, league_rpg=4.45),
    PlayerSeason(player_id="hallaro01", player_name="Roy Halladay", season=2011, team="PHI",
        position="P", ip=233.7, er=53, hits_allowed=208, hr_allowed=14,
        bb_allowed=35, hbp_allowed=3, k_pitching=220, games_pitched=32,
        games_started=32, park_factor=1.03, league_rpg=4.14),
]

NON_HOF["Curt Schilling"] = [
    PlayerSeason(player_id="schilcu01", player_name="Curt Schilling", season=2001, team="ARI",
        position="P", ip=256.7, er=67, hits_allowed=237, hr_allowed=20,
        bb_allowed=39, hbp_allowed=7, k_pitching=293, games_pitched=35,
        games_started=35, park_factor=1.04, league_rpg=4.78),
    PlayerSeason(player_id="schilcu01", player_name="Curt Schilling", season=2004, team="BOS",
        position="P", ip=226.7, er=64, hits_allowed=206, hr_allowed=23,
        bb_allowed=35, hbp_allowed=3, k_pitching=203, games_pitched=32,
        games_started=32, park_factor=1.04, league_rpg=4.83, league="AL"),
]


def main() -> None:
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "logs"), exist_ok=True)
    log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "hof_classifier.log")

    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    out("=" * 100)
    out("  HALL OF FAME CLASSIFIER — BRAVS SEPARATION ANALYSIS")
    out(f"  Generated {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    out("=" * 100)
    out()

    # Compute career BRAVS for both groups
    def compute_group(group: dict[str, list[PlayerSeason]]) -> dict[str, tuple[float, float, float]]:
        results: dict[str, tuple[float, float, float]] = {}
        for name, seasons in group.items():
            total_bravs = 0.0
            total_era_std = 0.0
            total_war_eq = 0.0
            for ps in seasons:
                r = compute_bravs(ps, fast=True)
                total_bravs += r.bravs
                total_era_std += r.bravs_era_standardized
                total_war_eq += r.bravs_calibrated
            results[name] = (total_bravs, total_era_std, total_war_eq)
        return results

    out("Computing HOF group...")
    hof_results = compute_group(HOF)
    out("Computing non-HOF group...")
    non_hof_results = compute_group(NON_HOF)

    # --- Print results by group ---
    out()
    out("=" * 100)
    out("  HALL OF FAMERS — Career BRAVS (from peak seasons modeled)")
    out("=" * 100)
    out(f"  {'Player':<25}{'CarBRAVS':>10}{'CarErStd':>10}{'CarWAReq':>10}{'Seasons':>10}")
    out("-" * 75)
    for name in sorted(hof_results, key=lambda n: hof_results[n][1], reverse=True):
        b, e, w = hof_results[name]
        n_szn = len(HOF[name])
        out(f"  {name:<25}{b:>10.1f}{e:>10.1f}{w:>10.1f}{n_szn:>10}")

    out()
    out("=" * 100)
    out("  NON-HALL OF FAMERS (or borderline) — Career BRAVS")
    out("=" * 100)
    out(f"  {'Player':<25}{'CarBRAVS':>10}{'CarErStd':>10}{'CarWAReq':>10}{'Seasons':>10}")
    out("-" * 75)
    for name in sorted(non_hof_results, key=lambda n: non_hof_results[n][1], reverse=True):
        b, e, w = non_hof_results[name]
        n_szn = len(NON_HOF[name])
        out(f"  {name:<25}{b:>10.1f}{e:>10.1f}{w:>10.1f}{n_szn:>10}")

    # --- Distribution analysis ---
    hof_vals = np.array([v[1] for v in hof_results.values()])  # era-standardized
    non_hof_vals = np.array([v[1] for v in non_hof_results.values()])

    out()
    out("=" * 100)
    out("  DISTRIBUTION ANALYSIS (era-standardized career BRAVS)")
    out("=" * 100)
    out()
    out(f"  HOF group:     mean={np.mean(hof_vals):.1f}  median={np.median(hof_vals):.1f}"
        f"  min={np.min(hof_vals):.1f}  max={np.max(hof_vals):.1f}  sd={np.std(hof_vals):.1f}")
    out(f"  Non-HOF group: mean={np.mean(non_hof_vals):.1f}  median={np.median(non_hof_vals):.1f}"
        f"  min={np.min(non_hof_vals):.1f}  max={np.max(non_hof_vals):.1f}  sd={np.std(non_hof_vals):.1f}")
    out()

    # Text histogram
    all_vals = np.concatenate([hof_vals, non_hof_vals])
    bin_min = float(np.floor(np.min(all_vals) / 2) * 2)
    bin_max = float(np.ceil(np.max(all_vals) / 2) * 2) + 2
    bins = np.arange(bin_min, bin_max + 1, 2)

    out("  TEXT HISTOGRAM (era-standardized career BRAVS from peak seasons)")
    out(f"  {'Range':<16}{'HOF':>6}{'non-HOF':>8}  Distribution")
    out("  " + "-" * 70)

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        h_count = int(np.sum((hof_vals >= lo) & (hof_vals < hi)))
        n_count = int(np.sum((non_hof_vals >= lo) & (non_hof_vals < hi)))
        h_bar = "#" * h_count
        n_bar = "." * n_count
        out(f"  [{lo:>5.0f},{hi:>5.0f}) {h_count:>5} {n_count:>7}  {h_bar}{n_bar}")

    out()
    out("  Legend: # = HOF, . = non-HOF")

    # --- Optimal threshold ---
    out()
    out("=" * 100)
    out("  OPTIMAL THRESHOLD ANALYSIS")
    out("=" * 100)
    out()

    # Try thresholds and find best separation
    labels = np.array([1] * len(hof_vals) + [0] * len(non_hof_vals))
    scores = np.concatenate([hof_vals, non_hof_vals])

    best_threshold = 0.0
    best_accuracy = 0.0
    best_tp, best_fp, best_tn, best_fn = 0, 0, 0, 0

    thresholds = np.arange(float(np.min(scores)) - 1, float(np.max(scores)) + 1, 0.5)
    for t in thresholds:
        predicted = (scores >= t).astype(int)
        tp = int(np.sum((predicted == 1) & (labels == 1)))
        fp = int(np.sum((predicted == 1) & (labels == 0)))
        tn = int(np.sum((predicted == 0) & (labels == 0)))
        fn = int(np.sum((predicted == 0) & (labels == 1)))
        accuracy = (tp + tn) / len(labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(t)
            best_tp, best_fp, best_tn, best_fn = tp, fp, tn, fn

    out(f"  Optimal threshold: {best_threshold:.1f} era-standardized career BRAVS")
    out(f"  Accuracy at threshold: {best_accuracy:.1%}")
    out(f"  Confusion matrix:")
    out(f"    True Positives  (HOF correctly classified): {best_tp}")
    out(f"    False Positives (non-HOF classified as HOF): {best_fp}")
    out(f"    True Negatives  (non-HOF correctly classified): {best_tn}")
    out(f"    False Negatives (HOF classified as non-HOF): {best_fn}")
    out()

    # Misclassified players
    predicted = (scores >= best_threshold).astype(int)
    all_names = list(hof_results.keys()) + list(non_hof_results.keys())

    misclassified = []
    for i, name in enumerate(all_names):
        if predicted[i] != labels[i]:
            actual = "HOF" if labels[i] == 1 else "non-HOF"
            pred_label = "HOF" if predicted[i] == 1 else "non-HOF"
            misclassified.append((name, actual, pred_label, float(scores[i])))

    if misclassified:
        out("  Misclassified players:")
        for name, actual, pred, val in misclassified:
            out(f"    {name:<25} actual={actual:<8} predicted={pred:<8} BRAVS={val:.1f}")
    else:
        out("  Perfect separation!")

    # --- AUC computation ---
    out()
    out("=" * 100)
    out("  ROC / AUC ANALYSIS")
    out("=" * 100)
    out()

    try:
        from scipy.stats import mannwhitneyu
        # AUC = P(score_positive > score_negative)
        # This is equivalent to the Mann-Whitney U statistic normalized
        u_stat, p_value = mannwhitneyu(hof_vals, non_hof_vals, alternative="greater")
        auc = u_stat / (len(hof_vals) * len(non_hof_vals))
        out(f"  AUC (area under ROC curve): {auc:.4f}")
        out(f"  Mann-Whitney U p-value:     {p_value:.6f}")
        out()
        if auc >= 0.90:
            out("  Interpretation: EXCELLENT separation (AUC >= 0.90)")
        elif auc >= 0.80:
            out("  Interpretation: GOOD separation (AUC >= 0.80)")
        elif auc >= 0.70:
            out("  Interpretation: FAIR separation (AUC >= 0.70)")
        else:
            out("  Interpretation: POOR separation (AUC < 0.70)")
    except ImportError:
        # Manual AUC computation if scipy not available
        out("  (scipy not available, computing AUC manually)")
        concordant = 0
        discordant = 0
        ties = 0
        for h in hof_vals:
            for n in non_hof_vals:
                if h > n:
                    concordant += 1
                elif h < n:
                    discordant += 1
                else:
                    ties += 1
        auc = (concordant + 0.5 * ties) / (len(hof_vals) * len(non_hof_vals))
        out(f"  AUC (area under ROC curve): {auc:.4f}")

    out()

    # --- BRAVS vs calibrated (WAR-equivalent) comparison ---
    out("=" * 100)
    out("  RAW BRAVS vs WAR-CALIBRATED BRAVS SEPARATION")
    out("=" * 100)
    out()

    hof_raw = np.array([v[0] for v in hof_results.values()])
    non_hof_raw = np.array([v[0] for v in non_hof_results.values()])
    hof_wareq = np.array([v[2] for v in hof_results.values()])
    non_hof_wareq = np.array([v[2] for v in non_hof_results.values()])

    for label, hv, nv in [
        ("Raw BRAVS", hof_raw, non_hof_raw),
        ("Era-standardized", hof_vals, non_hof_vals),
        ("WAR-equivalent", hof_wareq, non_hof_wareq),
    ]:
        sep = float(np.mean(hv) - np.mean(nv))
        pooled_sd = float(np.sqrt((np.var(hv) + np.var(nv)) / 2))
        cohens_d = sep / pooled_sd if pooled_sd > 0 else 0.0
        out(f"  {label:20s}: HOF mean={np.mean(hv):>6.1f}  non-HOF mean={np.mean(nv):>6.1f}"
            f"  gap={sep:>5.1f}  Cohen's d={cohens_d:.2f}")

    out()
    out("  Cohen's d interpretation: <0.5 small, 0.5-0.8 medium, >0.8 large")
    out()
    out("  Note: This analysis uses only 2-3 peak seasons per player.")
    out("  A full career analysis with all seasons would show even stronger")
    out("  separation, since HOFers maintain production over longer careers.")
    out()

    # Write log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[Saved to {log_path}]")


if __name__ == "__main__":
    main()
