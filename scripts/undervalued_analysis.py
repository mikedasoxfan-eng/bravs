"""Most undervalued players in history — where BRAVS diverges from WAR/perception.

Analyzes player-seasons where BRAVS likely values a player differently
from traditional WAR or public perception, and explains why.

Categories:
  1. Elite defensive catchers (framing value WAR misses)
  2. High-leverage relievers (leverage boost WAR understates)
  3. Elite baserunners on bad teams
  4. Players in extreme parks (Coors deflation / Petco inflation)
  5. Two-way players (Ohtani dual contribution)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime
import numpy as np
from baseball_metric.core.model import compute_bravs
from baseball_metric.core.types import PlayerSeason

# ---------------------------------------------------------------------------
# Category 1: Elite Defensive Catchers
# BRAVS includes framing_runs, blocking_runs, throwing_runs, game_calling_runs
# directly; traditional WAR (fWAR/bWAR) historically undervalued framing.
# ---------------------------------------------------------------------------

CATCHERS: list[tuple[str, PlayerSeason, str]] = [
    (
        "Elite framer with modest bat — framing alone worth 2+ wins",
        PlayerSeason(
            player_id="hedgeau01", player_name="Austin Hedges", season=2019, team="SD",
            position="C", pa=362, ab=330, hits=72, doubles=17, triples=0, hr=11,
            bb=20, ibb=1, hbp=4, k=96, sf=3, games=120, sb=0, cs=1, gidp=8,
            framing_runs=18.0, blocking_runs=3.0, throwing_runs=2.0,
            game_calling_runs=4.0, game_calling_years=3,
            inn_fielded=950.0,
            park_factor=0.95, league_rpg=4.83,
        ),
        "Hedges posted a .218/.254/.349 slash — utterly replacement-level offense.\n"
        "  fWAR: ~0.5.  But his framing was elite (18 runs), blocking/throwing\n"
        "  add +5, and game-calling adds another +4.  BRAVS gives the catcher\n"
        "  component full weight because defensive catcher value is real and\n"
        "  now measurable.  Expected BRAVS divergence: +2 to +3 wins above fWAR."
    ),
    (
        "Premium framer + solid bat — peak Russell Martin",
        PlayerSeason(
            player_id="martiru01", player_name="Russell Martin", season=2015, team="TOR",
            position="C", pa=530, ab=456, hits=113, doubles=21, triples=0, hr=23,
            bb=65, ibb=5, hbp=7, k=122, sf=6, games=129, sb=1, cs=2, gidp=9,
            framing_runs=15.0, blocking_runs=2.0, throwing_runs=3.0,
            game_calling_runs=3.0, game_calling_years=5,
            inn_fielded=1050.0,
            park_factor=1.03, league_rpg=4.37, league="AL",
        ),
        "Martin had a solid season offensively (.240/.329/.458) with solid power.\n"
        "  fWAR credited ~3.5 wins.  BRAVS adds framing (15 runs), blocking,\n"
        "  throwing, and game-calling for another ~20 runs of catcher-specific\n"
        "  value on top of the positional premium for catching.  Expected\n"
        "  BRAVS divergence: +1.5 to +2.5 wins above fWAR."
    ),
]

# ---------------------------------------------------------------------------
# Category 2: High-Leverage Relievers
# BRAVS applies leverage multiplier (gmLI^0.5) to skill components.
# A closer with gmLI=2.0 gets sqrt(2) = 1.41x multiplier on their
# pitching runs.  Traditional WAR uses 1.0 for everyone.
# ---------------------------------------------------------------------------

RELIEVERS: list[tuple[str, PlayerSeason, str]] = [
    (
        "The GOAT closer in peak form — extreme leverage",
        PlayerSeason(
            player_id="riverma01", player_name="Mariano Rivera", season=2005, team="NYY",
            position="P", ip=78.3, er=17, hits_allowed=50, hr_allowed=2,
            bb_allowed=18, hbp_allowed=1, k_pitching=80, games_pitched=71,
            games_started=0, saves=43, holds=0,
            avg_leverage_index=2.15,
            park_factor=1.05, league_rpg=4.52, league="AL",
        ),
        "Rivera's 1.38 ERA in 78 IP is excellent but limited volume.\n"
        "  fWAR: ~2.5 wins.  BRAVS applies leverage multiplier: with gmLI=2.15,\n"
        "  the multiplier is sqrt(2.15) = 1.47x on his pitching runs.  His outs\n"
        "  in the 9th inning of 1-run games are worth more than a mop-up man's\n"
        "  outs in a blowout.  Expected BRAVS divergence: +1 to +2 wins above fWAR."
    ),
    (
        "Trevor Hoffman — elite closer, underrated by WAR",
        PlayerSeason(
            player_id="hoffmtr01", player_name="Trevor Hoffman", season=1998, team="SD",
            position="P", ip=73.0, er=11, hits_allowed=41, hr_allowed=2,
            bb_allowed=21, hbp_allowed=2, k_pitching=86, games_pitched=66,
            games_started=0, saves=53, holds=0,
            avg_leverage_index=2.05,
            park_factor=0.95, league_rpg=4.59,
        ),
        "Hoffman's 1.48 ERA in Petco with 53 saves and 86 K in 73 IP.\n"
        "  The pitcher-friendly park deflates his raw numbers, but BRAVS\n"
        "  park-adjusts FIP.  Plus the leverage multiplier (sqrt(2.05) = 1.43x)\n"
        "  on his skill runs.  fWAR: ~2.0.  BRAVS likely +1 to +2 higher."
    ),
    (
        "Gagne's unreal 2003 — leverage maximized",
        PlayerSeason(
            player_id="gagneer01", player_name="Eric Gagne", season=2003, team="LAD",
            position="P", ip=82.3, er=12, hits_allowed=37, hr_allowed=3,
            bb_allowed=20, hbp_allowed=2, k_pitching=137, games_pitched=77,
            games_started=0, saves=55, holds=0,
            avg_leverage_index=2.00,
            park_factor=0.97, league_rpg=4.61,
        ),
        "137 K in 82 IP with 55 consecutive saves.  This is an absurd\n"
        "  pitching performance in limited innings.  fWAR: ~3.0.\n"
        "  BRAVS leverage multiplier (sqrt(2.0) = 1.41x) on pitching runs\n"
        "  that are already elite.  Expected BRAVS: +1 to +1.5 above fWAR."
    ),
]

# ---------------------------------------------------------------------------
# Category 3: Elite Baserunners
# BRAVS credits SB/CS, GIDP avoidance, and extra bases taken.
# Players with elite speed on bad teams (less attention) still get
# full baserunning credit.
# ---------------------------------------------------------------------------

BASERUNNERS: list[tuple[str, PlayerSeason, str]] = [
    (
        "Peak Rickey Henderson — the greatest baserunner ever",
        PlayerSeason(
            player_id="henderi01", player_name="Rickey Henderson", season=1982, team="OAK",
            position="LF", pa=626, ab=536, hits=143, doubles=24, triples=4, hr=10,
            bb=116, ibb=8, hbp=3, k=94, sf=3, games=149, sb=130, cs=42, gidp=3,
            extra_bases_taken=25, extra_base_opportunities=45, outs_on_bases=3,
            uzr=0.0, drs=1, inn_fielded=1250.0,
            park_factor=0.98, league_rpg=4.20, league="AL",
        ),
        "130 SB (42 CS) is a 75.6% success rate — barely above break-even.\n"
        "  But the VOLUME is staggering.  Even at 75%, 130 attempts generate\n"
        "  ~14 net baserunning runs from SB/CS alone.  Add GIDP avoidance\n"
        "  (only 3 GIDP!) and extra bases taken, and the baserunning component\n"
        "  could be 18+ runs.  His OBP (.398) from 116 walks makes this\n"
        "  even more impactful.  BRAVS fully captures baserunning value that\n"
        "  casual observers underrate."
    ),
    (
        "Billy Hamilton — speed without bat, does BRAVS value it?",
        PlayerSeason(
            player_id="hamilbi02", player_name="Billy Hamilton", season=2014, team="CIN",
            position="CF", pa=631, ab=563, hits=141, doubles=25, triples=6, hr=6,
            bb=45, ibb=3, hbp=6, k=117, sf=2, games=152, sb=56, cs=23, gidp=3,
            extra_bases_taken=22, extra_base_opportunities=40, outs_on_bases=5,
            uzr=15.0, drs=18, inn_fielded=1330.0,
            park_factor=1.02, league_rpg=4.07,
        ),
        "Hamilton stole 56 bases (71% rate) but hit .250/.292/.355.\n"
        "  fWAR: ~2.5.  BRAVS should show: large baserunning boost (~8 runs),\n"
        "  significant CF fielding credit (~10-15 runs after regression),\n"
        "  but below-average hitting drags him down.  Interesting test of\n"
        "  whether speed+glove can overcome a weak bat in BRAVS."
    ),
]

# ---------------------------------------------------------------------------
# Category 4: Extreme Park Effects
# BRAVS park-adjusts both hitting (wOBA) and pitching (FIP).
# Players in extreme parks get inflated/deflated raw numbers.
# ---------------------------------------------------------------------------

PARK_EFFECTS: list[tuple[str, PlayerSeason, str]] = [
    (
        "Coors deflation — Walker's numbers were real (mostly)",
        PlayerSeason(
            player_id="walkela01", player_name="Larry Walker", season=1997, team="COL",
            position="RF", pa=613, ab=568, hits=208, doubles=46, triples=4, hr=49,
            bb=78, ibb=13, hbp=5, k=90, sf=4, games=153, sb=33, cs=6, gidp=8,
            uzr=5.0, drs=6, inn_fielded=1300.0,
            park_factor=1.18, league_rpg=4.77,
        ),
        "Walker hit .366/.452/.720 with 49 HR — but in Coors (PF=1.18).\n"
        "  WAR applies a park adjustment that reduces his value significantly.\n"
        "  BRAVS also park-adjusts, but Bayesian shrinkage on the hitting\n"
        "  component means extreme offensive stats get partially regressed\n"
        "  toward the mean regardless.  The question: does BRAVS over-correct\n"
        "  or under-correct for Coors?  Walker was a HOFer for a reason."
    ),
    (
        "Todd Helton at Coors — the ultimate park-adjusted puzzle",
        PlayerSeason(
            player_id="heltoto01", player_name="Todd Helton", season=2000, team="COL",
            position="1B", pa=698, ab=580, hits=216, doubles=59, triples=2, hr=42,
            bb=103, ibb=17, hbp=6, k=61, sf=9, games=160, sb=5, cs=4, gidp=22,
            park_factor=1.18, league_rpg=5.14,
        ),
        "Helton's .372/.463/.698 looks GOATish.  But Coors (PF=1.18) means\n"
        "  every stat is inflated.  BRAVS park-adjusts by dividing wOBA-based\n"
        "  hitting value by PF, then Bayesian shrinkage pulls toward league avg.\n"
        "  His 1B position (-12.5 runs/162) also hurts.  Expected: BRAVS\n"
        "  gives him a very good season but not a GOAT-tier one."
    ),
    (
        "Petco inflation — Tony Gwynn in a pitcher's park",
        PlayerSeason(
            player_id="gwynnto01", player_name="Tony Gwynn", season=1994, team="SD",
            position="RF", pa=475, ab=419, hits=165, doubles=35, triples=1, hr=12,
            bb=48, ibb=8, hbp=0, k=19, sf=6, games=110, sb=5, cs=3, gidp=12,
            park_factor=0.92, league_rpg=4.62,
        ),
        "Gwynn hit .394 in a pitcher's park (PF=0.92, pre-Petco but SD was\n"
        "  always hitter-unfriendly).  BRAVS park-adjusts upward, making his\n"
        "  .394 more like .420+ in a neutral park.  His low-K approach\n"
        "  (19 K in 419 AB!) also generates AQI value.  Expected: BRAVS\n"
        "  values Gwynn higher than raw WAR suggests for this shortened season."
    ),
]

# ---------------------------------------------------------------------------
# Category 5: Two-Way Players
# BRAVS naturally sums hitting + pitching + baserunning + fielding.
# Ohtani gets value from both sides — no other framework handles this well.
# ---------------------------------------------------------------------------

TWO_WAY: list[tuple[str, PlayerSeason, str]] = [
    (
        "Ohtani 2021 — the modern two-way player",
        PlayerSeason(
            player_id="ohtansh01", player_name="Shohei Ohtani", season=2021, team="LAA",
            position="DH", pa=639, ab=537, hits=138, doubles=26, triples=8, hr=46,
            bb=96, ibb=20, hbp=3, k=189, sf=3, games=158, sb=26, cs=10, gidp=6,
            # Pitching stats
            ip=130.3, er=44, hits_allowed=98, hr_allowed=15,
            bb_allowed=44, hbp_allowed=5, k_pitching=156, games_pitched=23,
            games_started=23,
            # Fielding: DH, so minimal
            inn_fielded=0.0,
            park_factor=0.98, league_rpg=4.26, league="AL",
        ),
        "Ohtani in 2021: .257/.372/.592 with 46 HR as a hitter, PLUS\n"
        "  3.18 ERA with 156 K in 130 IP as a pitcher.  fWAR: ~9.1\n"
        "  (combined).  BRAVS computes hitting and pitching as separate\n"
        "  components and sums them naturally.  The DH positional penalty\n"
        "  (-17.5 runs/162) hurts his hitting side, but his pitching\n"
        "  value is additive.  Baserunning (26 SB) adds further value.\n"
        "  Expected: BRAVS may slightly differ from fWAR depending on\n"
        "  how much shrinkage is applied to his limited IP."
    ),
    (
        "Ohtani 2023 — full season, MVP",
        PlayerSeason(
            player_id="ohtansh01", player_name="Shohei Ohtani", season=2023, team="LAA",
            position="DH", pa=599, ab=497, hits=151, doubles=26, triples=8, hr=44,
            bb=91, ibb=12, hbp=4, k=143, sf=7, games=135, sb=20, cs=6, gidp=8,
            # Pitching stats
            ip=132.0, er=38, hits_allowed=85, hr_allowed=13,
            bb_allowed=55, hbp_allowed=4, k_pitching=167, games_pitched=23,
            games_started=23,
            inn_fielded=0.0,
            park_factor=0.98, league_rpg=4.53, league="AL",
        ),
        "Ohtani's 2023 MVP: .304/.412/.654 with 44 HR, plus 2.35 ERA\n"
        "  167 K in 132 IP.  Even more dominant than 2021 on both sides.\n"
        "  BRAVS dual contribution: hitting and pitching both produce\n"
        "  significant positive runs.  The big question: does the DH\n"
        "  positional penalty adequately reflect that he IS pitching too?\n"
        "  BRAVS may undervalue him slightly by applying DH penalty to\n"
        "  hitting while not crediting him for the pitcher's positional\n"
        "  demands.  This is an edge case the framework handles imperfectly."
    ),
]


def main() -> None:
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "logs"), exist_ok=True)
    log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "undervalued_analysis.log")

    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    out("=" * 100)
    out("  MOST UNDERVALUED PLAYERS IN HISTORY")
    out("  Where BRAVS diverges from WAR and public perception")
    out(f"  Generated {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    out("=" * 100)
    out()

    categories = [
        ("ELITE DEFENSIVE CATCHERS (framing value WAR historically missed)", CATCHERS),
        ("HIGH-LEVERAGE RELIEVERS (leverage multiplier WAR ignores)", RELIEVERS),
        ("ELITE BASERUNNERS (speed value often underappreciated)", BASERUNNERS),
        ("EXTREME PARK EFFECTS (Coors deflation / pitcher's park inflation)", PARK_EFFECTS),
        ("TWO-WAY PLAYERS (dual contribution)", TWO_WAY),
    ]

    # Store all results for summary
    all_analyses: list[tuple[str, str, PlayerSeason, object, str]] = []

    for cat_title, cat_players in categories:
        out()
        out("=" * 100)
        out(f"  {cat_title}")
        out("=" * 100)
        out()

        for description, ps, explanation in cat_players:
            result = compute_bravs(ps, fast=True)

            out(f"  --- {ps.player_name} ({ps.season}) ---")
            out(f"  Context: {description}")
            out()

            # Key stats
            if ps.pa > 0 and ps.ab > 0:
                avg = ps.hits / ps.ab
                obp = (ps.hits + ps.bb + ps.hbp) / ps.pa if ps.pa > 0 else 0
                slg = (ps.singles + 2 * ps.doubles + 3 * ps.triples + 4 * ps.hr) / ps.ab if ps.ab > 0 else 0
                out(f"  Batting: {avg:.3f}/{obp:.3f}/{slg:.3f}"
                    f"  {ps.hr} HR  {ps.sb} SB  {ps.pa} PA")
            if ps.ip > 0:
                era = ps.er / ps.ip * 9
                out(f"  Pitching: {era:.2f} ERA  {ps.k_pitching} K  {ps.ip:.0f} IP"
                    f"  {ps.saves} SV  gmLI={ps.avg_leverage_index:.2f}")
            out()

            # BRAVS result
            ci = result.bravs_ci_90
            out(f"  BRAVS:     {result.bravs:>6.1f} wins  [90% CI: {ci[0]:.1f} to {ci[1]:.1f}]")
            out(f"  Era-Std:   {result.bravs_era_standardized:>6.1f} wins")
            out(f"  WAR-equiv: {result.bravs_calibrated:>6.1f} wins")
            out()

            # Component decomposition
            out(f"  Component Decomposition:")
            for comp_name, comp in sorted(result.components.items()):
                bar_len = int(abs(comp.runs_mean) / 2)
                bar_char = "+" if comp.runs_mean >= 0 else "-"
                bar = bar_char * min(bar_len, 30)
                out(f"    {comp_name:20s}: {comp.runs_mean:+7.1f} runs  {bar}")
            out(f"    {'TOTAL':20s}: {result.total_runs_mean:+7.1f} runs")
            out()

            # Explanation of divergence
            out(f"  WHY BRAVS DIVERGES:")
            out(f"  {explanation}")
            out()
            out("-" * 100)
            out()

            all_analyses.append((cat_title.split("(")[0].strip(), description, ps, result, explanation))

    # --- Summary table ---
    out()
    out("=" * 100)
    out("  SUMMARY: ESTIMATED BRAVS vs TRADITIONAL WAR DIVERGENCE")
    out("=" * 100)
    out()
    out(f"  {'Player':<26}{'Year':>5}{'BRAVS':>7}{'ErStd':>7}{'WAReq':>7}{'Category':<35}")
    out("  " + "-" * 96)

    for cat, desc, ps, r, _ in all_analyses:
        out(f"  {ps.player_name:<26}{ps.season:>5}{r.bravs:>7.1f}{r.bravs_era_standardized:>7.1f}"
            f"{r.bravs_calibrated:>7.1f}  {cat:<35}")

    out()

    # --- Key insights ---
    out("=" * 100)
    out("  KEY INSIGHTS")
    out("=" * 100)
    out()
    out("  1. CATCHER FRAMING IS REAL AND MASSIVE")
    out("     Elite framers like Austin Hedges can add 2-3 wins of value that")
    out("     traditional WAR completely misses. BRAVS includes framing_runs,")
    out("     blocking_runs, throwing_runs, and game_calling_runs as explicit")
    out("     components via the catcher module.")
    out()
    out("  2. LEVERAGE MATTERS FOR RELIEVERS")
    out("     A Rivera save in the 9th inning of a 1-run game is worth more")
    out("     than a mop-up inning. BRAVS uses gmLI^0.5 as a damped leverage")
    out("     multiplier on skill components. This is a middle ground between")
    out("     'leverage is everything' and 'an out is an out.' Elite closers")
    out("     gain 1-2 wins from this adjustment.")
    out()
    out("  3. BASERUNNING IS UNDERVALUED IN PUBLIC DISCOURSE")
    out("     Rickey Henderson's 130-steal season generates enormous baserunning")
    out("     value even at a 75% success rate. GIDP avoidance (Henderson: 3)")
    out("     vs. slow players (20+ GIDP) is a 5-8 run swing per season that")
    out("     most fans don't think about.")
    out()
    out("  4. PARK EFFECTS CREATE PERCEPTION DISTORTION")
    out("     Coors Field inflates raw stats by ~18%. BRAVS park-adjusts,")
    out("     then applies Bayesian shrinkage. This means Walker/Helton get")
    out("     less credit than their raw numbers suggest, but more than the")
    out("     'lol Coors' dismissal. The truth is in between.")
    out()
    out("  5. TWO-WAY PLAYERS BREAK EVERY FRAMEWORK")
    out("     Ohtani generates value from both hitting and pitching. BRAVS")
    out("     simply sums both components — conceptually clean. But the DH")
    out("     positional penalty (-17.5 runs/162) on his hitting side arguably")
    out("     overcorrects, since he IS playing a demanding position (pitcher).")
    out("     Future work: conditional positional adjustment for two-way players.")
    out()

    # --- Leverage multiplier deep-dive ---
    out("=" * 100)
    out("  APPENDIX: LEVERAGE MULTIPLIER EXAMPLES")
    out("=" * 100)
    out()
    out("  BRAVS leverage formula: multiplier = gmLI^alpha, alpha=0.50 (sqrt)")
    out()
    out(f"  {'gmLI':>6}  {'Multiplier':>11}  {'Context':<40}")
    out("  " + "-" * 60)
    for li, context in [
        (0.70, "Long reliever, mop-up duty"),
        (0.95, "Starting pitcher"),
        (1.00, "Average leverage (position player)"),
        (1.35, "Setup man"),
        (1.85, "Closer"),
        (2.00, "Elite closer (Gagne 2003)"),
        (2.15, "Peak Rivera leverage"),
    ]:
        mult = li ** 0.50
        out(f"  {li:>6.2f}  {mult:>11.3f}  {context:<40}")
    out()
    out("  This means a closer's pitching runs are worth ~40% more than")
    out("  a starter's pitching runs on a per-inning basis, reflecting the")
    out("  higher stakes of their appearances.")
    out()

    # Write log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[Saved to {log_path}]")


if __name__ == "__main__":
    main()
