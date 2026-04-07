"""Find the greatest pitching season ever according to BRAVS."""

from baseball_metric.core.model import compute_bravs
from baseball_metric.core.types import PlayerSeason

seasons = [
    # Dead-ball era
    PlayerSeason(player_id="wj1913", player_name="Walter Johnson", season=1913, team="WSH",
        position="P", ip=346.0, er=56, hits_allowed=232, hr_allowed=2,
        bb_allowed=38, hbp_allowed=9, k_pitching=243, games_pitched=48,
        games_started=36, park_factor=0.97, league_rpg=3.93, league="AL"),
    PlayerSeason(player_id="wj1912", player_name="Walter Johnson", season=1912, team="WSH",
        position="P", ip=369.0, er=69, hits_allowed=259, hr_allowed=4,
        bb_allowed=76, hbp_allowed=11, k_pitching=303, games_pitched=50,
        games_started=37, park_factor=0.97, league_rpg=4.10, league="AL"),
    PlayerSeason(player_id="pa1915", player_name="Pete Alexander", season=1915, team="PHI",
        position="P", ip=376.3, er=56, hits_allowed=253, hr_allowed=3,
        bb_allowed=64, hbp_allowed=5, k_pitching=241, games_pitched=49,
        games_started=42, park_factor=1.02, league_rpg=3.65),
    PlayerSeason(player_id="cm1908", player_name="Christy Mathewson", season=1908, team="NYG",
        position="P", ip=390.7, er=42, hits_allowed=285, hr_allowed=1,
        bb_allowed=42, hbp_allowed=5, k_pitching=259, games_pitched=56,
        games_started=44, park_factor=0.95, league_rpg=3.37),

    # 1960s pitcher's era
    PlayerSeason(player_id="bg1968", player_name="Bob Gibson", season=1968, team="STL",
        position="P", ip=304.7, er=38, hits_allowed=198, hr_allowed=11,
        bb_allowed=62, hbp_allowed=4, k_pitching=268, games_pitched=34,
        games_started=34, park_factor=0.98, league_rpg=3.42),
    PlayerSeason(player_id="sk1966", player_name="Sandy Koufax", season=1966, team="LAD",
        position="P", ip=323.0, er=62, hits_allowed=241, hr_allowed=19,
        bb_allowed=77, hbp_allowed=2, k_pitching=317, games_pitched=41,
        games_started=41, park_factor=0.95, league_rpg=3.89),
    PlayerSeason(player_id="sk1965", player_name="Sandy Koufax", season=1965, team="LAD",
        position="P", ip=335.7, er=76, hits_allowed=216, hr_allowed=26,
        bb_allowed=71, hbp_allowed=3, k_pitching=382, games_pitched=43,
        games_started=41, park_factor=0.95, league_rpg=4.03),
    PlayerSeason(player_id="jm1966", player_name="Juan Marichal", season=1966, team="SF",
        position="P", ip=307.3, er=60, hits_allowed=228, hr_allowed=17,
        bb_allowed=36, hbp_allowed=5, k_pitching=222, games_pitched=37,
        games_started=36, park_factor=0.93, league_rpg=3.89),

    # Modern dominance
    PlayerSeason(player_id="pm1999", player_name="Pedro Martinez", season=1999, team="BOS",
        position="P", ip=213.3, er=49, hits_allowed=160, hr_allowed=9,
        bb_allowed=37, hbp_allowed=9, k_pitching=313, games_pitched=31,
        games_started=31, park_factor=1.04, league_rpg=5.08, league="AL"),
    PlayerSeason(player_id="pm2000", player_name="Pedro Martinez", season=2000, team="BOS",
        position="P", ip=217.0, er=42, hits_allowed=128, hr_allowed=17,
        bb_allowed=32, hbp_allowed=6, k_pitching=284, games_pitched=29,
        games_started=29, park_factor=1.04, league_rpg=5.14, league="AL"),
    PlayerSeason(player_id="gm1995", player_name="Greg Maddux", season=1995, team="ATL",
        position="P", ip=209.7, er=38, hits_allowed=147, hr_allowed=8,
        bb_allowed=23, hbp_allowed=4, k_pitching=181, games_pitched=28,
        games_started=28, park_factor=1.00, league_rpg=4.63),
    PlayerSeason(player_id="dg1985", player_name="Dwight Gooden", season=1985, team="NYM",
        position="P", ip=276.7, er=51, hits_allowed=198, hr_allowed=13,
        bb_allowed=69, hbp_allowed=3, k_pitching=268, games_pitched=35,
        games_started=35, park_factor=0.97, league_rpg=4.07),
    PlayerSeason(player_id="rc1997", player_name="Roger Clemens", season=1997, team="TOR",
        position="P", ip=264.0, er=65, hits_allowed=204, hr_allowed=19,
        bb_allowed=68, hbp_allowed=8, k_pitching=292, games_pitched=34,
        games_started=34, park_factor=1.03, league_rpg=4.77, league="AL"),

    # Statcast era
    PlayerSeason(player_id="jd2018", player_name="Jacob deGrom", season=2018, team="NYM",
        position="P", ip=217.0, er=48, hits_allowed=152, hr_allowed=10,
        bb_allowed=46, hbp_allowed=5, k_pitching=269, games_pitched=32,
        games_started=32, park_factor=0.95, league_rpg=4.45),
    PlayerSeason(player_id="jd2021", player_name="Jacob deGrom", season=2021, team="NYM",
        position="P", ip=92.0, er=13, hits_allowed=44, hr_allowed=5,
        bb_allowed=11, hbp_allowed=3, k_pitching=146, games_pitched=15,
        games_started=15, park_factor=0.95, league_rpg=4.26),
    PlayerSeason(player_id="gc2019", player_name="Gerrit Cole", season=2019, team="HOU",
        position="P", ip=212.3, er=52, hits_allowed=142, hr_allowed=29,
        bb_allowed=48, hbp_allowed=7, k_pitching=326, games_pitched=33,
        games_started=33, park_factor=0.98, league_rpg=4.83, league="AL"),
    PlayerSeason(player_id="jv2011", player_name="Justin Verlander", season=2011, team="DET",
        position="P", ip=251.0, er=57, hits_allowed=174, hr_allowed=24,
        bb_allowed=57, hbp_allowed=4, k_pitching=250, games_pitched=34,
        games_started=34, park_factor=0.97, league_rpg=4.28, league="AL"),
    PlayerSeason(player_id="ck2014", player_name="Clayton Kershaw", season=2014, team="LAD",
        position="P", ip=198.3, er=39, hits_allowed=139, hr_allowed=9,
        bb_allowed=31, hbp_allowed=2, k_pitching=239, games_pitched=27,
        games_started=27, park_factor=0.97, league_rpg=4.07),
    PlayerSeason(player_id="rj2001", player_name="Randy Johnson", season=2001, team="ARI",
        position="P", ip=249.7, er=64, hits_allowed=181, hr_allowed=19,
        bb_allowed=71, hbp_allowed=18, k_pitching=372, games_pitched=35,
        games_started=35, park_factor=1.04, league_rpg=4.78),
    PlayerSeason(player_id="nr1973", player_name="Nolan Ryan", season=1973, team="CAL",
        position="P", ip=326.0, er=82, hits_allowed=238, hr_allowed=18,
        bb_allowed=162, hbp_allowed=6, k_pitching=383, games_pitched=41,
        games_started=39, park_factor=0.96, league_rpg=4.28, league="AL"),
]

results = []
for player in seasons:
    r = compute_bravs(player)
    results.append(r)

# --- Raw BRAVS ranking ---
results.sort(key=lambda r: r.bravs, reverse=True)

print("=" * 100)
print("  GREATEST PITCHING SEASONS - RAW BRAVS (context-dependent, dynamic RPW)")
print("=" * 100)
print(f"{'Rank':>4}  {'Pitcher':<26} {'Year':>4}  {'ERA':>5} {'IP':>6} {'K':>4}"
      f"  {'BRAVS':>6} {'ErStd':>6} {'WAReq':>6}  {'90% CI':>16}")
print("-" * 100)
for i, r in enumerate(results, 1):
    p = r.player
    era = p.er / p.ip * 9
    ci = r.bravs_ci_90
    print(f"{i:>4}. {p.player_name:<26} {p.season:>4}  {era:>5.2f} {p.ip:>6.1f} {p.k_pitching:>4}"
          f"  {r.bravs:>6.1f} {r.bravs_era_standardized:>6.1f} {r.bravs_calibrated:>6.1f}"
          f"  [{ci[0]:>5.1f}, {ci[1]:>5.1f}]")

# --- Era-standardized ranking ---
results_es = sorted(results, key=lambda r: r.bravs_era_standardized, reverse=True)

print()
print("=" * 100)
print("  ERA-STANDARDIZED RANKING (fixed RPW=5.9 - removes low-scoring-era inflation)")
print("=" * 100)
print(f"{'Rank':>4}  {'Pitcher':<26} {'Year':>4}  {'ERA':>5} {'IP':>6} {'K':>4}"
      f"  {'ErStd':>6} {'BRAVS':>6} {'WAReq':>6}  {'RPW':>5}")
print("-" * 100)
for i, r in enumerate(results_es, 1):
    p = r.player
    era = p.er / p.ip * 9
    print(f"{i:>4}. {p.player_name:<26} {p.season:>4}  {era:>5.2f} {p.ip:>6.1f} {p.k_pitching:>4}"
          f"  {r.bravs_era_standardized:>6.1f} {r.bravs:>6.1f} {r.bravs_calibrated:>6.1f}"
          f"  {r.rpw:>5.2f}")

# --- Top 5 component breakdown ---
print()
print("=" * 100)
print("  TOP 5 (ERA-STANDARDIZED) - Component Breakdown")
print("=" * 100)
for i, r in enumerate(results_es[:5], 1):
    p = r.player
    era = p.er / p.ip * 9
    pit = r.components["pitching"]
    lev = r.components["leverage"]
    dur = r.components["durability"]
    print(f"{i}. {p.player_name} {p.season}  (ERA {era:.2f}, {p.ip:.0f} IP, {p.k_pitching} K)")
    print(f"   Pitching: {pit.runs_mean:+.1f} runs  "
          f"(FIP: {pit.metadata['obs_fip']}, post_FIP: {pit.metadata['post_fip_mean']})")
    print(f"   Leverage: {lev.runs_mean:+.1f}   Durability: {dur.runs_mean:+.1f}   RPW: {r.rpw:.2f}")
    print(f"   Raw BRAVS: {r.bravs:.1f}   Era-Std: {r.bravs_era_standardized:.1f}   "
          f"WAR-eq: {r.bravs_calibrated:.1f}")
    print()
