"""Rerun every analysis we've done using BRAVS v2.0 and the Lahman dataset.

Uses pre-computed data/bravs_all_seasons.csv and data/bravs_careers.csv
from the GPU v2.0 run. No API calls, no recomputation needed.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from scipy import stats as sp_stats


def load_data():
    seasons = pd.read_csv("data/bravs_all_seasons.csv")
    careers = pd.read_csv("data/bravs_careers.csv")
    return seasons, careers


def section(title):
    print(f"\n\n{'#' * 90}")
    print(f"  {title}")
    print(f"{'#' * 90}")


def award_race(seasons, title, year, league, top_n=10, is_pitching=False):
    """Run an MVP or Cy Young race from pre-computed data."""
    s = seasons[seasons.yearID == year].copy()
    if is_pitching:
        s = s[s.IP >= 100]
    else:
        s = s[s.PA >= 300]
    s = s.sort_values("bravs_war_eq", ascending=False)

    print(f"\n  {title}")
    print(f"  {'Rank':<5}{'Player':<24}{'Pos':<5}{'WAR-eq':>7}{'BRAVS':>7}{'HR':>5}{'SB':>5}")
    print(f"  {'-' * 60}")
    for i, (_, r) in enumerate(s.head(top_n).iterrows(), 1):
        print(f"  {i:<5}{r['name']:<24}{r.position:<5}{r.bravs_war_eq:>7.1f}{r.bravs:>7.1f}"
              f"{r.HR:>5.0f}{r.SB:>5.0f}")


def career_leaderboard(careers, title, top_n=25):
    top = careers.nlargest(top_n, "career_war_eq")
    print(f"\n  {title}")
    print(f"  {'Rank':<5}{'Player':<25}{'Years':>12}{'WAR-eq':>8}{'Peak':>7}{'HOF':>5}")
    print(f"  {'-' * 65}")
    for i, (_, r) in enumerate(top.iterrows(), 1):
        h = "Y" if r.hof else ""
        print(f"  {i:<5}{r['name']:<25}{r.first_year:.0f}-{r.last_year:.0f}"
              f"{r.career_war_eq:>8.1f}{r.peak_bravs:>7.1f}{h:>5}")


def hof_classifier(careers):
    from baseball_metric.data import lahman
    hof_ids = set(lahman.get_hof_inducted())
    careers_copy = careers.copy()
    careers_copy["is_hof"] = careers_copy.playerID.isin(hof_ids)

    hof = careers_copy[careers_copy.is_hof & (careers_copy.seasons >= 3)]
    non = careers_copy[~careers_copy.is_hof & (careers_copy.seasons >= 8) & (careers_copy.total_PA >= 2000)]

    if len(hof) == 0 or len(non) == 0:
        print("  Insufficient data for HOF classifier")
        return

    # Sample non-HOF
    np.random.seed(42)
    non_sample = non.sample(min(200, len(non)))

    hof_wars = hof.career_war_eq.values
    non_wars = non_sample.career_war_eq.values

    u_stat, p_val = sp_stats.mannwhitneyu(hof_wars, non_wars, alternative="greater")
    auc = u_stat / (len(hof_wars) * len(non_wars))
    d = (hof_wars.mean() - non_wars.mean()) / np.sqrt((hof_wars.var() + non_wars.var()) / 2)

    print(f"\n  HOF:     n={len(hof_wars):>4}  mean={hof_wars.mean():>6.1f}  median={np.median(hof_wars):>6.1f}")
    print(f"  Non-HOF: n={len(non_wars):>4}  mean={non_wars.mean():>6.1f}  median={np.median(non_wars):>6.1f}")
    print(f"  AUC: {auc:.4f}  Cohen's d: {d:.2f}  p: {p_val:.2e}")

    # Optimal threshold
    all_vals = list(zip(hof_wars, [True]*len(hof_wars))) + list(zip(non_wars, [False]*len(non_wars)))
    best_acc, best_t = 0, 0
    for t in np.arange(10, 80, 0.5):
        correct = sum(1 for v, h in all_vals if (v >= t) == h)
        acc = correct / len(all_vals)
        if acc > best_acc:
            best_acc, best_t = acc, t
    print(f"  Optimal threshold: {best_t:.1f} WAR-eq ({best_acc:.1%} accuracy)")


def main():
    seasons, careers = load_data()

    section("1. 2017 AL MVP RACE: JUDGE vs ALTUVE")
    award_race(seasons, "2017 AL MVP", 2017, "AL")
    s17 = seasons[(seasons.yearID == 2017) & (seasons.PA >= 300)]
    judge = s17[s17.name.str.contains("Aaron Judge", na=False)]
    altuve = s17[s17.name.str.contains("Jose Altuve", na=False)]
    if not judge.empty and not altuve.empty:
        j, a = judge.iloc[0], altuve.iloc[0]
        print(f"\n  Judge:  {j.bravs_war_eq:.1f} WAR-eq  (hitting: {j.hitting_runs:+.1f}, BR: {j.baserunning_runs:+.1f}, pos: {j.positional_runs:+.1f})")
        print(f"  Altuve: {a.bravs_war_eq:.1f} WAR-eq  (hitting: {a.hitting_runs:+.1f}, BR: {a.baserunning_runs:+.1f}, pos: {a.positional_runs:+.1f})")

    section("2. 2016 AL CY YOUNG: PORCELLO vs VERLANDER vs KLUBER vs SALE")
    award_race(seasons, "2016 AL Cy Young", 2016, "AL", is_pitching=True)

    section("3. 2012 AL MVP: CABRERA (TRIPLE CROWN) vs TROUT")
    award_race(seasons, "2012 AL MVP", 2012, "AL")
    s12 = seasons[(seasons.yearID == 2012) & (seasons.PA >= 300)]
    cab = s12[s12.name.str.contains("Miguel Cabrera", na=False)]
    tro = s12[s12.name.str.contains("Mike Trout", na=False)]
    if not cab.empty and not tro.empty:
        c, t = cab.iloc[0], tro.iloc[0]
        print(f"\n  Cabrera: {c.bravs_war_eq:.1f} WAR-eq  (hit: {c.hitting_runs:+.1f}, BR: {c.baserunning_runs:+.1f}, fld: {c.fielding_runs:+.1f}, pos: {c.positional_runs:+.1f})")
        print(f"  Trout:   {t.bravs_war_eq:.1f} WAR-eq  (hit: {t.hitting_runs:+.1f}, BR: {t.baserunning_runs:+.1f}, fld: {t.fielding_runs:+.1f}, pos: {t.positional_runs:+.1f})")

    section("4. 1941 AL MVP: TED WILLIAMS (.406) vs JOE DiMAGGIO (56-GAME STREAK)")
    award_race(seasons, "1941 AL MVP", 1941, "AL")

    section("5. 2001 NL MVP: BONDS (73 HR) vs SOSA vs PUJOLS vs HELTON")
    award_race(seasons, "2001 NL MVP", 2001, "NL")

    section("6. 2011 NL MVP: BRAUN vs KEMP vs VOTTO")
    award_race(seasons, "2011 NL MVP", 2011, "NL")

    section("7. BEST PITCHING SEASON EVER")
    pit = seasons[seasons.IP >= 150].nlargest(15, "bravs_war_eq")
    print(f"\n  {'Rank':<5}{'Player':<24}{'Year':>5}{'IP':>6}{'WAR-eq':>8}{'BRAVS':>8}")
    print(f"  {'-' * 58}")
    for i, (_, r) in enumerate(pit.iterrows(), 1):
        print(f"  {i:<5}{r['name']:<24}{r.yearID:>5.0f}{r.IP:>6.0f}{r.bravs_war_eq:>8.1f}{r.bravs:>8.1f}")

    section("8. ALL-TIME CAREER LEADERBOARD (TOP 25)")
    career_leaderboard(careers, "Top 25 Careers")

    section("9. BEST SINGLE SEASON EVER (TOP 25)")
    top_s = seasons.nlargest(25, "bravs_war_eq")
    print(f"\n  {'Rank':<5}{'Player':<24}{'Year':>5}{'Pos':<5}{'WAR-eq':>8}{'BRAVS':>8}")
    print(f"  {'-' * 57}")
    for i, (_, r) in enumerate(top_s.iterrows(), 1):
        print(f"  {i:<5}{r['name']:<24}{r.yearID:>5.0f} {r.position:<4}{r.bravs_war_eq:>8.1f}{r.bravs:>8.1f}")

    section("10. HALL OF FAME CLASSIFIER")
    hof_classifier(careers)

    section("11. ACTIVE PLAYER HOF CHECK")
    # Players active in 2024 or 2025
    active_ids = set(seasons[seasons.yearID >= 2024].playerID.unique())
    active = careers[careers.playerID.isin(active_ids)].nlargest(25, "career_war_eq")
    print(f"\n  {'Rank':<5}{'Player':<24}{'WAR-eq':>8}{'Peak':>7}{'Yrs':>5}{'Verdict':>12}")
    print(f"  {'-' * 63}")
    for i, (_, r) in enumerate(active.iterrows(), 1):
        war = r.career_war_eq
        verdict = "LOCK" if war >= 60 else "STRONG" if war >= 45 else "BORDER" if war >= 35 else "NEEDS MORE" if war >= 25 else "NOT YET"
        print(f"  {i:<5}{r['name']:<24}{war:>8.1f}{r.peak_bravs:>7.1f}{r.seasons:>5.0f}{verdict:>12}")

    section("12. STANTON HOF ANALYSIS")
    stanton = careers[careers.name.str.contains("Stanton", na=False)]
    if not stanton.empty:
        s = stanton.iloc[0]
        print(f"\n  Career: {s.career_war_eq:.1f} WAR-eq, {s.career_bravs:.1f} BRAVS")
        print(f"  Peak: {s.peak_bravs:.1f}, Seasons: {s.seasons:.0f}, HR: {s.total_HR:.0f}")
        verdict = "LOCK" if s.career_war_eq >= 60 else "STRONG" if s.career_war_eq >= 45 else "BORDER" if s.career_war_eq >= 35 else "NO"
        print(f"  Verdict: {verdict}")

    section("13. STEROID ERA: PED USERS vs CLEAN COMPARABLES")
    ped = ["bondsba01", "mcgwima01", "sosasa01", "rodrial01", "clemero02"]
    clean = ["griffke02", "thomafr04", "troutmi01", "pujolal01", "maddugr01", "martipe02"]
    print(f"\n  SUSPECTED PED USERS:")
    for pid in ped:
        r = careers[careers.playerID == pid]
        if not r.empty:
            r = r.iloc[0]
            print(f"    {r['name']:<24} {r.career_war_eq:>7.1f} WAR-eq  peak={r.peak_bravs:.1f}")
    print(f"\n  CLEAN COMPARABLES:")
    for pid in clean:
        r = careers[careers.playerID == pid]
        if not r.empty:
            r = r.iloc[0]
            print(f"    {r['name']:<24} {r.career_war_eq:>7.1f} WAR-eq  peak={r.peak_bravs:.1f}")

    section("14. DYNASTY RANKINGS: BEST 5-YEAR WINDOWS")
    # For top 10 career players, find their best 5-year window
    top_careers = careers.nlargest(15, "career_war_eq")
    dynasties = []
    for _, cr in top_careers.iterrows():
        pid = cr.playerID
        ps = seasons[seasons.playerID == pid].sort_values("yearID")
        if len(ps) < 5:
            continue
        best_total = -999
        best_start = 0
        yrs = ps.yearID.values
        bravs_vals = ps.bravs_war_eq.values
        for i in range(len(yrs)):
            window = [(y, b) for y, b in zip(yrs, bravs_vals) if yrs[i] <= y < yrs[i] + 5]
            total = sum(b for _, b in window)
            if total > best_total and len(window) >= 3:
                best_total = total
                best_start = int(yrs[i])
        dynasties.append({"name": cr["name"], "start": best_start, "end": best_start + 4,
                         "total": round(best_total, 1), "career": round(cr.career_war_eq, 1)})
    dynasties.sort(key=lambda x: x["total"], reverse=True)
    print(f"\n  {'Rank':<5}{'Player':<24}{'Window':>12}{'5yr WAR-eq':>12}{'Career':>8}")
    print(f"  {'-' * 63}")
    for i, d in enumerate(dynasties[:10], 1):
        print(f"  {i:<5}{d['name']:<24}{d['start']}-{d['end']}{d['total']:>12.1f}{d['career']:>8.1f}")

    section("15. DREAM TEAM: BEST SEASON AT EACH POSITION")
    positions = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "P"]
    total_war = 0
    for pos in positions:
        if pos == "P":
            best = seasons[(seasons.position == "P") & (seasons.IP >= 150)].nlargest(1, "bravs_war_eq")
        else:
            best = seasons[(seasons.position == pos) & (seasons.PA >= 200)].nlargest(1, "bravs_war_eq")
        if not best.empty:
            b = best.iloc[0]
            total_war += b.bravs_war_eq
            print(f"  {pos:<4} {b['name']:<24} {b.yearID:.0f}  {b.bravs_war_eq:>6.1f} WAR-eq")
    print(f"  {'':>4} {'TOTAL':<24}       {total_war:>6.1f}")

    section("16. BEST SEASON BY POSITION (TOP 3 EACH)")
    for pos in positions:
        if pos == "P":
            top3 = seasons[(seasons.position == "P") & (seasons.IP >= 150)].nlargest(3, "bravs_war_eq")
        else:
            top3 = seasons[(seasons.position == pos) & (seasons.PA >= 200)].nlargest(3, "bravs_war_eq")
        print(f"\n  {pos}:")
        for _, r in top3.iterrows():
            print(f"    {r['name']:<24} {r.yearID:.0f}  {r.bravs_war_eq:>6.1f} WAR-eq  {r.bravs:>6.1f} BRAVS")

    section("17. MOST UNDERVALUED: BIGGEST BASERUNNING CONTRIBUTORS")
    br_leaders = seasons[seasons.PA >= 300].nlargest(15, "baserunning_runs")
    print(f"\n  {'Player':<24}{'Year':>5}{'SB':>5}{'BR Runs':>9}{'WAR-eq':>8}")
    print(f"  {'-' * 53}")
    for _, r in br_leaders.iterrows():
        print(f"  {r['name']:<24}{r.yearID:>5.0f}{r.SB:>5.0f}{r.baserunning_runs:>9.1f}{r.bravs_war_eq:>8.1f}")

    section("18. BEST DEFENSIVE SEASONS (FIELDING RUNS)")
    fld_leaders = seasons[(seasons.PA >= 200) & (seasons.position != "P")].nlargest(15, "fielding_runs")
    print(f"\n  {'Player':<24}{'Year':>5}{'Pos':<5}{'Fld Runs':>9}{'WAR-eq':>8}")
    print(f"  {'-' * 53}")
    for _, r in fld_leaders.iterrows():
        print(f"  {r['name']:<24}{r.yearID:>5.0f}{r.position:<5}{r.fielding_runs:>9.1f}{r.bravs_war_eq:>8.1f}")

    section("19. LEVERAGE KINGS: BEST CLOSER SEASONS")
    closers = seasons[(seasons.IP >= 40) & (seasons.IP < 120)].copy()
    closers = closers[closers.leverage_runs > 0].nlargest(15, "leverage_runs")
    print(f"\n  {'Player':<24}{'Year':>5}{'IP':>6}{'Lev Runs':>9}{'WAR-eq':>8}")
    print(f"  {'-' * 55}")
    for _, r in closers.iterrows():
        print(f"  {r['name']:<24}{r.yearID:>5.0f}{r.IP:>6.0f}{r.leverage_runs:>9.1f}{r.bravs_war_eq:>8.1f}")

    section("20. 2025 SEASON LEADERBOARD")
    s25 = seasons[seasons.yearID == 2025]
    # Hitters
    top_hit = s25[s25.PA >= 300].nlargest(10, "bravs_war_eq")
    print(f"\n  HITTERS:")
    print(f"  {'Rank':<5}{'Player':<24}{'Pos':<5}{'WAR-eq':>7}{'HR':>5}{'SB':>5}")
    print(f"  {'-' * 53}")
    for i, (_, r) in enumerate(top_hit.iterrows(), 1):
        print(f"  {i:<5}{r['name']:<24}{r.position:<5}{r.bravs_war_eq:>7.1f}{r.HR:>5.0f}{r.SB:>5.0f}")
    # Pitchers
    top_pit = s25[s25.IP >= 100].nlargest(10, "bravs_war_eq")
    print(f"\n  PITCHERS:")
    for i, (_, r) in enumerate(top_pit.iterrows(), 1):
        print(f"  {i:<5}{r['name']:<24}{'P':<5}{r.bravs_war_eq:>7.1f}{'':>5}{r.IP:>5.0f} IP")

    print(f"\n\n{'#' * 90}")
    print(f"  ALL ANALYSES COMPLETE — BRAVS v2.0")
    print(f"  75,265 player-seasons, 14,092 careers, 1920-2025")
    print(f"{'#' * 90}")


if __name__ == "__main__":
    main()
