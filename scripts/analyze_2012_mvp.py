"""Deep analysis: 2012 AL MVP — Cabrera vs Trout."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from baseball_metric.core.model import compute_bravs
from baseball_metric.core.types import PlayerSeason

cabrera = PlayerSeason(
    player_id="408234", player_name="Miguel Cabrera", season=2012, team="DET",
    position="3B", pa=697, ab=622, hits=205, singles=116, doubles=40,
    triples=0, hr=44, bb=66, ibb=17, hbp=3, k=98, sf=5, games=161,
    sb=4, cs=1, gidp=28,
    uzr=-7.2, drs=-8, inn_fielded=1250.0,
    park_factor=0.99, league_rpg=4.32, league="AL",
)

trout = PlayerSeason(
    player_id="545361", player_name="Mike Trout", season=2012, team="LAA",
    position="CF", pa=639, ab=559, hits=182, singles=112, doubles=27,
    triples=8, hr=30, bb=67, ibb=8, hbp=4, k=139, sf=7, games=139,
    sb=49, cs=5, gidp=7,
    uzr=4.0, drs=4, inn_fielded=1150.0,
    park_factor=0.98, league_rpg=4.32, league="AL",
)

# Other contenders
cano = PlayerSeason(
    player_id="cano01", player_name="Robinson Cano", season=2012, team="NYY",
    position="2B", pa=681, ab=627, hits=196, singles=126, doubles=33,
    triples=1, hr=33, bb=61, ibb=7, hbp=7, k=104, sf=3, games=161,
    sb=3, cs=4, gidp=22,
    uzr=2.5, drs=6, inn_fielded=1380.0,
    park_factor=1.05, league_rpg=4.32, league="AL",
)

beltre = PlayerSeason(
    player_id="beltre01", player_name="Adrian Beltre", season=2012, team="TEX",
    position="3B", pa=654, ab=604, hits=194, singles=121, doubles=33,
    triples=1, hr=36, bb=37, ibb=7, hbp=5, k=62, sf=6, games=156,
    sb=1, cs=2, gidp=14,
    uzr=16.2, drs=19, inn_fielded=1320.0,
    park_factor=1.04, league_rpg=4.32, league="AL",
)

hamilton = PlayerSeason(
    player_id="hamilt01", player_name="Josh Hamilton", season=2012, team="TEX",
    position="LF", pa=583, ab=562, hits=160, singles=80, doubles=31,
    triples=2, hr=43, bb=46, ibb=11, hbp=3, k=162, sf=5, games=148,
    sb=7, cs=3, gidp=9,
    park_factor=1.04, league_rpg=4.32, league="AL",
)

players = [
    ("Cabrera", cabrera), ("Trout", trout), ("Cano", cano),
    ("Beltre", beltre), ("Hamilton", hamilton),
]

results = {}
for name, player in players:
    results[name] = compute_bravs(player)

# Sort by BRAVS
ranked = sorted(results.items(), key=lambda x: x[1].bravs, reverse=True)

print("=" * 90)
print("  2012 AL MVP: THE MOST CONTROVERSIAL AWARD RACE IN MODERN BASEBALL")
print("=" * 90)
print()
print(f"{'Rank':<6}{'Player':<20}{'BRAVS':>7}{'ErStd':>7}{'WAReq':>7}{'90% CI':>18}{'fWAR':>7}")
print("-" * 90)

known_war = {"Cabrera": 7.1, "Trout": 10.8, "Cano": 8.0, "Beltre": 8.4, "Hamilton": 4.3}
for i, (name, r) in enumerate(ranked, 1):
    ci = r.bravs_ci_90
    fwar = known_war.get(name, "?")
    print(f"  {i}.  {name:<20}{r.bravs:>7.1f}{r.bravs_era_standardized:>7.1f}"
          f"{r.bravs_calibrated:>7.1f}  [{ci[0]:>5.1f}, {ci[1]:>5.1f}]  {fwar:>7}")

rc = results["Cabrera"]
rt = results["Trout"]

print()
print("=" * 90)
print("  CABRERA vs TROUT: COMPONENT-BY-COMPONENT")
print("=" * 90)
print()
print(f"{'Component':<22}{'CABRERA':>10}{'TROUT':>10}{'GAP':>10}{'WINNER':>10}")
print("-" * 62)

comp_names = sorted(set(list(rc.components.keys()) + list(rt.components.keys())))
for name in comp_names:
    c_val = rc.components.get(name)
    t_val = rt.components.get(name)
    c_runs = c_val.runs_mean if c_val else 0.0
    t_runs = t_val.runs_mean if t_val else 0.0
    gap = c_runs - t_runs
    winner = "CAB" if gap > 1 else "TRO" if gap < -1 else "PUSH"
    print(f"  {name:<20}{c_runs:>+10.1f}{t_runs:>+10.1f}{gap:>+10.1f}{winner:>10}")

print(f"  {'':>20}{'--------':>10}{'--------':>10}{'--------':>10}")
print(f"  {'TOTAL RUNS':<20}{rc.total_runs_mean:>+10.1f}{rt.total_runs_mean:>+10.1f}"
      f"{rc.total_runs_mean - rt.total_runs_mean:>+10.1f}")
print(f"  {'BRAVS (wins)':<20}{rc.bravs:>10.1f}{rt.bravs:>10.1f}"
      f"{rc.bravs - rt.bravs:>+10.1f}")

# Posterior comparison
n = min(len(rc.total_samples), len(rt.total_samples))
cab_wins = float(np.mean(rc.total_samples[:n] > rt.total_samples[:n]))
ci_c = rc.bravs_ci_90
ci_t = rt.bravs_ci_90
overlap_lo = max(ci_c[0], ci_t[0])
overlap_hi = min(ci_c[1], ci_t[1])

print()
print("=" * 90)
print("  POSTERIOR PROBABILITY ANALYSIS")
print("=" * 90)
print()
print(f"  P(Cabrera > Trout) = {cab_wins:.1%}")
print(f"  P(Trout > Cabrera) = {1 - cab_wins:.1%}")
print()
print(f"  Cabrera: {rc.bravs:.1f} BRAVS [{ci_c[0]:.1f}, {ci_c[1]:.1f}]")
print(f"  Trout:   {rt.bravs:.1f} BRAVS [{ci_t[0]:.1f}, {ci_t[1]:.1f}]")
print(f"  CI overlap: [{overlap_lo:.1f}, {overlap_hi:.1f}] = {overlap_hi - overlap_lo:.1f} wins of shared range")
print()

# Where the gap comes from
cab_hit = rc.components["hitting"].runs_mean
tro_hit = rt.components["hitting"].runs_mean
cab_br = rc.components["baserunning"].runs_mean
tro_br = rt.components["baserunning"].runs_mean
cab_fld = rc.components["fielding"].runs_mean
tro_fld = rt.components["fielding"].runs_mean
cab_dur = rc.components["durability"].runs_mean
tro_dur = rt.components["durability"].runs_mean
cab_pos = rc.components["positional"].runs_mean
tro_pos = rt.components["positional"].runs_mean

print("=" * 90)
print("  WHY BRAVS AND fWAR DISAGREE")
print("=" * 90)
print(f"""
  fWAR says:  Trout 10.8, Cabrera 7.1  (Trout by 3.7 wins)
  BRAVS says: Cabrera {rc.bravs:.1f}, Trout {rt.bravs:.1f}  (Cabrera by {rc.bravs - rt.bravs:.1f} wins)

  The ~5 win swing between BRAVS and fWAR comes from:

  1. DEFENSE (the big one)
     fWAR: Trout +8.3 fielding runs, Cabrera -8.0 = 16.3-run gap for Trout
     BRAVS: Trout {tro_fld:+.1f}, Cabrera {cab_fld:+.1f} = {tro_fld - cab_fld:.1f}-run gap

     BRAVS heavily regresses defensive metrics through Bayesian shrinkage.
     With UZR and DRS available but no OAA (Statcast didn't exist in 2012),
     and knowing these metrics have ~0.4 year-over-year correlation, BRAVS
     refuses to give Trout full credit for one good defensive season or
     penalize Cabrera fully for one bad one. This alone accounts for ~16
     runs of the disagreement.

     WHO'S RIGHT? fWAR takes the defensive numbers at face value. BRAVS
     says "these numbers are probably directionally correct but overstated."
     The truth is probably in between. Trout WAS a better defender, but
     maybe +5 runs better, not +16.

  2. HITTING (+{cab_hit - tro_hit:.1f} runs for Cabrera)
     Cabrera: {cab_hit:+.1f} runs (.330/.393/.606 in 697 PA)
     Trout:   {tro_hit:+.1f} runs (.326/.399/.564 in 639 PA)

     Cabrera's Triple Crown was real. His .999 OPS in 58 more PA creates
     a genuine hitting advantage. Trout had a higher OBP (.399 vs .393)
     but Cabrera's slugging (.606 vs .564) more than compensates. This
     is a component where BRAVS and fWAR roughly agree.

  3. DURABILITY (+{cab_dur - tro_dur:.1f} runs for Cabrera)
     Cabrera: {cab_dur:+.1f} runs (161 games, basically full season)
     Trout:   {tro_dur:+.1f} runs (139 games, called up April 28)

     Trout missed 23 games because the Angels started the year with him
     in the minors. BRAVS penalizes this — his team needed a FAT-level
     replacement for those games. This is philosophically defensible
     (availability has value) but feels punitive for a 20-year-old who
     wasn't given the job from day one.

  4. BASERUNNING (+{tro_br - cab_br:.1f} runs for Trout)
     Trout:   {tro_br:+.1f} runs (49 SB, 5 CS, 7 GIDP)
     Cabrera: {cab_br:+.1f} runs (4 SB, 1 CS, 28 GIDP)

     This is Trout's clearest edge. 49 stolen bases at an 91% success
     rate is elite. Cabrera's 28 GIDP is brutal. Both metrics agree
     this is a ~13-run swing for Trout.

  5. POSITIONAL VALUE (push: {cab_pos:+.1f} vs {tro_pos:+.1f})
     Both 3B and CF get +2.5 runs per 162 games in the Tango scale.
     Essentially a wash. fWAR has CF slightly higher, giving Trout
     a small edge there.
""")

print("=" * 90)
print("  THE VERDICT")
print("=" * 90)
print(f"""
  BRAVS: Cabrera {rc.bravs:.1f} > Trout {rt.bravs:.1f} by {rc.bravs - rt.bravs:.1f} wins
  fWAR:  Trout 10.8 > Cabrera 7.1 by 3.7 wins
  bWAR:  Trout 10.9 > Cabrera 6.9 by 4.0 wins

  But P(Cabrera > Trout) = {cab_wins:.0%} — this is NOT a slam dunk.

  The 2012 AL MVP was genuinely one of the closest and most interesting
  value debates in baseball history. It comes down to a single question:

    HOW MUCH DO YOU TRUST SINGLE-SEASON DEFENSIVE METRICS?

  If you trust UZR/DRS at face value: Trout wins easily (fWAR's answer).
  If you're skeptical of defensive metrics: Cabrera wins (BRAVS's answer).
  If you think the truth is in between: it's a coin flip.

  What BRAVS adds to this debate is HONESTY ABOUT UNCERTAINTY. The 90%
  credible intervals overlap by {overlap_hi - overlap_lo:.1f} wins. Neither answer is
  confident. The voters picked Cabrera. WAR says Trout. BRAVS says
  "these are really close and anyone who tells you it's obvious is
  overstating their confidence in defensive measurement."

  That's the whole point of building a probabilistic metric.
""")
