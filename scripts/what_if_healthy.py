"""What If They Stayed Healthy? — injury-adjusted career projections.

Estimates what players' careers would have looked like without injuries,
using their peak performance + standard aging curves.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from baseball_metric.analysis.projections import _get_aging_factor, HITTING_AGING, PITCHING_AGING


def project_healthy_career(name, peak_bravs, peak_age, debut_age, actual_career_bravs,
                           actual_seasons, is_pitcher=False, retirement_age=40):
    """Project what a career would look like with no injuries."""
    curve = PITCHING_AGING if is_pitcher else HITTING_AGING
    peak_factor = _get_aging_factor(curve, peak_age)
    talent = peak_bravs / peak_factor if peak_factor > 0 else peak_bravs

    healthy_total = 0.0
    seasons = []
    for age in range(debut_age, retirement_age + 1):
        factor = _get_aging_factor(curve, age)
        season_bravs = talent * factor
        if season_bravs < 0.5:
            break
        healthy_total += season_bravs
        seasons.append((age, round(season_bravs, 1)))

    lost = healthy_total - actual_career_bravs
    return {
        "name": name,
        "actual_career": actual_career_bravs,
        "healthy_career": round(healthy_total, 1),
        "lost_to_injury": round(lost, 1),
        "actual_seasons": actual_seasons,
        "healthy_seasons": len(seasons),
        "peak_bravs": peak_bravs,
        "peak_age": peak_age,
        "seasons": seasons,
    }


PLAYERS = [
    # (name, peak_bravs, peak_age, debut_age, actual_career_bravs, actual_seasons, is_pitcher)
    ("Mike Trout", 15.4, 25, 21, 85.0, 14, False),
    ("Ken Griffey Jr.", 15.8, 26, 20, 77.0, 22, False),
    ("Sandy Koufax", 24.5, 30, 20, 49.0, 12, True),
    ("Mickey Mantle", 18.4, 25, 20, 112.0, 18, False),
    ("Josh Hamilton", 11.0, 27, 22, 28.0, 9, False),
    ("Jacob deGrom", 15.1, 30, 26, 38.0, 9, True),
    ("Jose Fernandez", 10.0, 23, 20, 17.0, 4, True),
    ("Mark Prior", 9.0, 22, 22, 12.0, 5, True),
    ("Bo Jackson", 8.0, 27, 24, 14.0, 5, False),
    ("Yoenis Cespedes", 7.0, 28, 26, 22.0, 8, False),
    ("David Wright", 10.5, 26, 22, 52.0, 14, False),
    ("Nomar Garciaparra", 11.0, 27, 23, 38.0, 12, False),
    ("Johan Santana", 13.5, 27, 22, 44.0, 12, True),
    ("Matt Harvey", 8.5, 24, 23, 12.0, 7, True),
    ("Prince Fielder", 10.0, 28, 22, 42.0, 12, False),
]


def main():
    print("=" * 90)
    print("  WHAT IF THEY STAYED HEALTHY?")
    print("  Injury-adjusted career projections using BRAVS aging curves")
    print("=" * 90)

    results = []
    for name, peak, peak_age, debut, actual, seasons, is_p in PLAYERS:
        r = project_healthy_career(name, peak, peak_age, debut, actual, seasons, is_p)
        results.append(r)

    results.sort(key=lambda x: x["lost_to_injury"], reverse=True)

    print(f"\n  {'Rank':<6}{'Player':<22}{'Actual':>8}{'Healthy':>9}{'Lost':>8}{'Seasons':>10}")
    print("  " + "-" * 65)

    for i, r in enumerate(results, 1):
        print(f"  {i:>3}.  {r['name']:<22}{r['actual_career']:>8.1f}"
              f"{r['healthy_career']:>9.1f}{r['lost_to_injury']:>+8.1f}"
              f"  {r['actual_seasons']:>2}/{r['healthy_seasons']}")

    # Top 5 detailed
    for r in results[:5]:
        print(f"\n  {'~' * 60}")
        print(f"  {r['name']}: {r['lost_to_injury']:+.1f} BRAVS lost to injuries")
        print(f"  {'~' * 60}")
        print(f"  Peak: {r['peak_bravs']:.1f} BRAVS at age {r['peak_age']}")
        print(f"  Actual career: {r['actual_career']:.1f} BRAVS in {r['actual_seasons']} seasons")
        print(f"  Healthy career: {r['healthy_career']:.1f} BRAVS in {r['healthy_seasons']} seasons")
        print()

        # Mini sparkline
        for age, bravs in r["seasons"]:
            bar = "#" * int(max(bravs, 0) / 1.2)
            actual_marker = " <-- peak" if age == r["peak_age"] else ""
            cut = ""
            if age > r["peak_age"] + (r["actual_seasons"] - (r["peak_age"] - r["seasons"][0][0])):
                cut = "  (lost)"
            print(f"    age {age}: {bravs:>5.1f}  {bar}{actual_marker}{cut}")

    print(f"""
  {'=' * 90}
  KEY FINDINGS
  {'=' * 90}

  The biggest "what ifs" in baseball history are players whose
  peak talent was undeniable but whose bodies couldn't sustain it.

  BRAVS aging curves project what each player WOULD have produced
  with normal aging (no injuries). The "lost" column is the gap
  between their projected healthy career and their actual output.

  Caveats:
  - Assumes the player would have aged normally (no guarantee)
  - Doesn't account for potential improvement (some young players
    never reached their ceiling due to early injury)
  - Actual career BRAVS estimates are approximate (not computed
    from full season-by-season data)
  - Some players (Koufax) retired by choice due to injury risk,
    not because they couldn't play at all
""")


if __name__ == "__main__":
    main()
