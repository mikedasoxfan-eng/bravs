"""HOF Classifier using FULL Lahman database — all 256 HOFers vs non-HOFers."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
from scipy import stats as sp_stats
from baseball_metric.data.lahman import (
    get_hof_inducted, get_player_season, get_all_seasons, _people, get_primary_position,
)
from baseball_metric.core.model import compute_bravs


def compute_career_bravs(player_id):
    """Compute career BRAVS from Lahman data."""
    years = get_all_seasons(player_id)
    if not years:
        return None

    total_bravs = 0.0
    total_war = 0.0
    seasons_played = 0
    peak = 0.0

    for yr in years:
        if yr < 1920 or yr > 2021:
            continue
        ps = get_player_season(player_id, yr)
        if not ps or (ps.pa < 30 and ps.ip < 5):
            continue
        try:
            r = compute_bravs(ps, fast=True)
            total_bravs += r.bravs
            total_war += r.bravs_calibrated
            peak = max(peak, r.bravs)
            seasons_played += 1
        except Exception:
            continue

    if seasons_played == 0:
        return None

    return {
        "player_id": player_id,
        "career_bravs": round(total_bravs, 1),
        "career_war": round(total_war, 1),
        "peak_bravs": round(peak, 1),
        "seasons": seasons_played,
    }


def main():
    t_start = time.perf_counter()

    print("=" * 85)
    print("  HALL OF FAME CLASSIFIER — FULL LAHMAN DATABASE")
    print("  Every inducted player vs qualified non-HOFers")
    print("=" * 85)

    # Get all HOF inductees
    hof_ids = set(get_hof_inducted())
    people = _people()

    # Filter to players with enough career data (debut before 2010)
    eligible = people[
        (people.debut.notna()) &
        (people.debut.str[:4].astype(float, errors="ignore") < 2010)
    ]

    # Build HOF group: all inducted players
    print(f"\n  Computing career BRAVS for {len(hof_ids)} HOF inductees...")
    hof_results = []
    for pid in hof_ids:
        r = compute_career_bravs(pid)
        if r and r["seasons"] >= 3:
            r["hof"] = True
            hof_results.append(r)

    print(f"  Computed: {len(hof_results)} HOFers with 3+ seasons")

    # Build non-HOF group: players with 10+ seasons and 2000+ PA who are NOT in HOF
    # (these are the "good but not great" baseline)
    print(f"  Computing non-HOF comparison group...")
    from baseball_metric.data.lahman import _batting
    bat = _batting()
    career_pa = bat.groupby("playerID").agg({"AB": "sum", "BB": "sum", "G": "sum"}).reset_index()
    career_pa["PA_est"] = career_pa.AB + career_pa.BB
    career_pa["seasons"] = bat.groupby("playerID").yearID.nunique().values

    # Non-HOF: 10+ seasons, 2000+ PA, not inducted
    non_hof_pool = career_pa[
        (career_pa.PA_est >= 2000) &
        (career_pa.seasons >= 8) &
        (~career_pa.playerID.isin(hof_ids))
    ].playerID.tolist()

    # Sample to keep runtime manageable
    np.random.seed(42)
    sample_size = min(len(non_hof_pool), 200)
    non_hof_sample = list(np.random.choice(non_hof_pool, sample_size, replace=False))

    non_hof_results = []
    for i, pid in enumerate(non_hof_sample):
        if i % 50 == 0:
            print(f"    Processing non-HOF {i}/{sample_size}...")
        r = compute_career_bravs(pid)
        if r and r["seasons"] >= 3:
            r["hof"] = False
            non_hof_results.append(r)

    print(f"  Computed: {len(non_hof_results)} non-HOFers")

    t_compute = time.perf_counter()
    print(f"\n  Total computation time: {t_compute - t_start:.0f}s")

    # Analysis
    hof_wars = [r["career_war"] for r in hof_results]
    non_wars = [r["career_war"] for r in non_hof_results]

    print(f"\n{'=' * 85}")
    print(f"  RESULTS")
    print(f"{'=' * 85}")

    print(f"\n  HOF group:     n={len(hof_wars):>4}  mean={np.mean(hof_wars):>6.1f}  median={np.median(hof_wars):>6.1f}  sd={np.std(hof_wars):>5.1f}")
    print(f"  Non-HOF group: n={len(non_wars):>4}  mean={np.mean(non_wars):>6.1f}  median={np.median(non_wars):>6.1f}  sd={np.std(non_wars):>5.1f}")

    # Mann-Whitney U test
    u_stat, p_value = sp_stats.mannwhitneyu(hof_wars, non_wars, alternative="greater")
    auc = u_stat / (len(hof_wars) * len(non_wars))

    print(f"\n  AUC (Mann-Whitney): {auc:.4f}")
    print(f"  p-value: {p_value:.2e}")

    # Cohen's d
    pooled_sd = np.sqrt((np.var(hof_wars) + np.var(non_wars)) / 2)
    cohens_d = (np.mean(hof_wars) - np.mean(non_wars)) / pooled_sd if pooled_sd > 0 else 0
    print(f"  Cohen's d: {cohens_d:.2f}")

    # Find optimal threshold
    all_data = hof_results + non_hof_results
    best_acc = 0
    best_thresh = 0
    for thresh in np.arange(5, 60, 0.5):
        correct = sum(1 for r in all_data if (r["career_war"] >= thresh) == r["hof"])
        acc = correct / len(all_data)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    print(f"\n  Optimal threshold: {best_thresh:.1f} career WAR-eq")
    print(f"  Accuracy at threshold: {best_acc:.1%}")

    # Confusion matrix
    tp = sum(1 for r in hof_results if r["career_war"] >= best_thresh)
    fn = sum(1 for r in hof_results if r["career_war"] < best_thresh)
    fp = sum(1 for r in non_hof_results if r["career_war"] >= best_thresh)
    tn = sum(1 for r in non_hof_results if r["career_war"] < best_thresh)

    print(f"\n  Confusion matrix:")
    print(f"    True Positives  (HOF correctly classified): {tp}")
    print(f"    False Negatives (HOF missed):               {fn}")
    print(f"    False Positives (non-HOF classified as HOF): {fp}")
    print(f"    True Negatives  (non-HOF correct):          {tn}")
    print(f"    Precision: {tp/(tp+fp):.1%}" if tp+fp > 0 else "")
    print(f"    Recall:    {tp/(tp+fn):.1%}" if tp+fn > 0 else "")

    # Top HOF players by career BRAVS
    hof_sorted = sorted(hof_results, key=lambda x: x["career_war"], reverse=True)
    print(f"\n  TOP 25 HALL OF FAMERS BY CAREER WAR-eq:")
    print(f"  {'Rank':<6}{'Player ID':<18}{'WAR-eq':>8}{'BRAVS':>8}{'Peak':>7}{'Yrs':>5}")
    print(f"  {'-' * 55}")
    for i, r in enumerate(hof_sorted[:25], 1):
        print(f"  {i:<6}{r['player_id']:<18}{r['career_war']:>8.1f}{r['career_bravs']:>8.1f}"
              f"{r['peak_bravs']:>7.1f}{r['seasons']:>5}")

    # Biggest HOF misses (HOFers with low BRAVS)
    hof_low = sorted(hof_results, key=lambda x: x["career_war"])
    print(f"\n  LOWEST-RATED HOFers (questionable inductees by BRAVS):")
    for r in hof_low[:10]:
        print(f"    {r['player_id']:<18} {r['career_war']:>6.1f} WAR-eq  ({r['seasons']} seasons)")

    # Highest non-HOFers (snubs)
    non_high = sorted(non_hof_results, key=lambda x: x["career_war"], reverse=True)
    print(f"\n  HIGHEST-RATED NON-HOFers (potential snubs):")
    for r in non_high[:10]:
        print(f"    {r['player_id']:<18} {r['career_war']:>6.1f} WAR-eq  ({r['seasons']} seasons)")

    total_time = time.perf_counter() - t_start
    print(f"\n  Total runtime: {total_time:.0f}s")


if __name__ == "__main__":
    main()
