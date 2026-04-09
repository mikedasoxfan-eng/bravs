"""Decade-by-decade talent analysis — is baseball talent improving, concentrating, or spreading?

Loads data/bravs_all_seasons.csv and computes per-decade:
  - Average WAR-eq of the top 10 position players
  - Average WAR-eq of the top 5 pitchers
  - Standard deviation of WAR-eq (talent spread)
  - Number of 8+ WAR-eq superstar seasons
  - Number of negative WAR-eq seasons among qualifiers
  - Most/least competitive decades
  - Pitching vs hitting dominance shifts
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import pandas as pd
import numpy as np


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bravs_all_seasons.csv")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "talent_trends.log")


def decade_label(year):
    """Return a label like '1920s', '1930s', etc."""
    d = (year // 10) * 10
    return f"{d}s"


def analyze(df):
    """Run the full talent-trend analysis, returning the report as a string."""
    buf = io.StringIO()

    def p(text=""):
        buf.write(text + "\n")

    # ------------------------------------------------------------------
    # Assign decade
    # ------------------------------------------------------------------
    df = df.copy()
    df["decade"] = df["yearID"].apply(decade_label)

    # Position players = non-pitchers; pitchers = position == 'P'
    hitters = df[df["position"] != "P"]
    pitchers = df[df["position"] == "P"]

    # Qualifier thresholds
    qual_hit = hitters[hitters["PA"] >= 300]
    qual_pit = pitchers[pitchers["IP"] >= 100]

    decades = sorted(df["decade"].unique())

    # ------------------------------------------------------------------
    # Per-decade metrics
    # ------------------------------------------------------------------
    p("=" * 90)
    p("  TALENT TRENDS: DECADE-BY-DECADE ANALYSIS")
    p("=" * 90)

    results = []
    for dec in decades:
        dh = qual_hit[qual_hit["decade"] == dec]
        dp = qual_pit[qual_pit["decade"] == dec]

        top10_hit = dh.nlargest(10, "bravs_war_eq")["bravs_war_eq"]
        top5_pit = dp.nlargest(5, "bravs_war_eq")["bravs_war_eq"]

        avg_top10 = top10_hit.mean() if len(top10_hit) > 0 else 0
        avg_top5_pit = top5_pit.mean() if len(top5_pit) > 0 else 0

        # Talent spread among all qualifiers
        all_qual = pd.concat([dh["bravs_war_eq"], dp["bravs_war_eq"]])
        std_war = all_qual.std() if len(all_qual) > 1 else 0

        # Superstar seasons (8+ WAR-eq)
        superstar = (all_qual >= 8).sum()

        # Negative WAR-eq among qualifiers
        negative = (all_qual < 0).sum()

        # Gap between #1 and #10 hitter
        if len(top10_hit) >= 10:
            gap_1_10 = top10_hit.iloc[0] - top10_hit.iloc[9]
        elif len(top10_hit) >= 2:
            gap_1_10 = top10_hit.iloc[0] - top10_hit.iloc[-1]
        else:
            gap_1_10 = 0

        results.append({
            "decade": dec,
            "avg_top10_hit": avg_top10,
            "avg_top5_pit": avg_top5_pit,
            "std_war": std_war,
            "superstar": int(superstar),
            "negative": int(negative),
            "gap_1_10": gap_1_10,
            "n_hit_qual": len(dh),
            "n_pit_qual": len(dp),
        })

    # Print the table
    p()
    header = (f"  {'Decade':<9}{'Top10 Hit':>10}{'Top5 Pit':>10}{'Std Dev':>9}"
              f"{'8+ WAR':>8}{'Neg WAR':>9}{'#1-#10 Gap':>12}{'Qual H':>8}{'Qual P':>8}")
    p(header)
    p("  " + "-" * (len(header) - 2))

    for r in results:
        p(f"  {r['decade']:<9}"
          f"{r['avg_top10_hit']:>10.2f}"
          f"{r['avg_top5_pit']:>10.2f}"
          f"{r['std_war']:>9.2f}"
          f"{r['superstar']:>8}"
          f"{r['negative']:>9}"
          f"{r['gap_1_10']:>12.2f}"
          f"{r['n_hit_qual']:>8}"
          f"{r['n_pit_qual']:>8}")

    # ------------------------------------------------------------------
    # Summary findings
    # ------------------------------------------------------------------
    p()
    p("=" * 90)
    p("  SUMMARY FINDINGS")
    p("=" * 90)

    # Trend in top-10 hitter average
    first_3 = [r for r in results if r["decade"] in ("1920s", "1930s", "1940s")]
    last_3 = [r for r in results if r["decade"] in ("2000s", "2010s", "2020s")]

    if first_3 and last_3:
        early_avg = np.mean([r["avg_top10_hit"] for r in first_3])
        late_avg = np.mean([r["avg_top10_hit"] for r in last_3])
        p()
        p(f"  Top-10 hitter WAR-eq (avg of 1920s-1940s): {early_avg:.2f}")
        p(f"  Top-10 hitter WAR-eq (avg of 2000s-2020s): {late_avg:.2f}")
        if late_avg < early_avg:
            p(f"  -> Top-end hitting talent appears LOWER in recent decades ({early_avg - late_avg:+.2f})")
            p(f"     This likely reflects talent SPREADING (more parity), not declining ability.")
        else:
            p(f"  -> Top-end hitting talent is HIGHER in recent decades ({late_avg - early_avg:+.2f})")

    # Talent spread trend
    if first_3 and last_3:
        early_std = np.mean([r["std_war"] for r in first_3])
        late_std = np.mean([r["std_war"] for r in last_3])
        p()
        p(f"  WAR-eq std dev (1920s-1940s avg): {early_std:.2f}")
        p(f"  WAR-eq std dev (2000s-2020s avg): {late_std:.2f}")
        if late_std < early_std:
            p("  -> Talent is MORE EVENLY DISTRIBUTED in the modern era.")
            p("     The gap between stars and average players has narrowed.")
        else:
            p("  -> Talent spread has INCREASED — the gap between stars and average players has grown.")

    # Superstar count trend
    if first_3 and last_3:
        early_ss = sum(r["superstar"] for r in first_3)
        late_ss = sum(r["superstar"] for r in last_3)
        p()
        p(f"  8+ WAR-eq seasons (1920s-1940s total): {early_ss}")
        p(f"  8+ WAR-eq seasons (2000s-2020s total): {late_ss}")
        if late_ss > early_ss:
            p("  -> More superstar seasons in the modern era, despite talent leveling.")
        elif late_ss < early_ss:
            p("  -> Fewer superstar seasons today — consistent with talent compression.")
        else:
            p("  -> Superstar production is roughly constant across eras.")

    # ------------------------------------------------------------------
    # Most competitive / most top-heavy decades
    # ------------------------------------------------------------------
    p()
    p("-" * 90)
    p("  COMPETITIVENESS ANALYSIS")
    p("-" * 90)

    valid = [r for r in results if r["gap_1_10"] > 0]
    if valid:
        most_competitive = min(valid, key=lambda r: r["gap_1_10"])
        most_topheavy = max(valid, key=lambda r: r["gap_1_10"])
        p()
        p(f"  Most competitive decade (smallest #1-#10 gap):")
        p(f"    {most_competitive['decade']}  gap = {most_competitive['gap_1_10']:.2f} WAR-eq")
        p()
        p(f"  Most top-heavy decade (biggest #1-#10 gap):")
        p(f"    {most_topheavy['decade']}  gap = {most_topheavy['gap_1_10']:.2f} WAR-eq")
        p()
        p(f"  All decades ranked by competitiveness (ascending gap):")
        for r in sorted(valid, key=lambda r: r["gap_1_10"]):
            p(f"    {r['decade']}  {r['gap_1_10']:>6.2f}")

    # ------------------------------------------------------------------
    # Pitching vs hitting dominance
    # ------------------------------------------------------------------
    p()
    p("-" * 90)
    p("  PITCHING vs HITTING DOMINANCE SHIFT")
    p("-" * 90)
    p()
    p(f"  {'Decade':<9}{'Top10 Hit':>10}{'Top5 Pit':>10}{'Ratio P/H':>10}{'Dominance':>14}")
    p("  " + "-" * 53)

    for r in results:
        if r["avg_top10_hit"] > 0:
            ratio = r["avg_top5_pit"] / r["avg_top10_hit"]
        else:
            ratio = 0
        if ratio > 1.1:
            dom = "PITCHING"
        elif ratio < 0.9:
            dom = "HITTING"
        else:
            dom = "BALANCED"
        p(f"  {r['decade']:<9}{r['avg_top10_hit']:>10.2f}{r['avg_top5_pit']:>10.2f}{ratio:>10.2f}{dom:>14}")

    # Check for shift over time
    if first_3 and last_3:
        early_ratios = []
        for r in first_3:
            if r["avg_top10_hit"] > 0:
                early_ratios.append(r["avg_top5_pit"] / r["avg_top10_hit"])
        late_ratios = []
        for r in last_3:
            if r["avg_top10_hit"] > 0:
                late_ratios.append(r["avg_top5_pit"] / r["avg_top10_hit"])

        if early_ratios and late_ratios:
            early_r = np.mean(early_ratios)
            late_r = np.mean(late_ratios)
            p()
            p(f"  Avg pitching/hitting ratio (1920s-1940s): {early_r:.3f}")
            p(f"  Avg pitching/hitting ratio (2000s-2020s): {late_r:.3f}")
            if late_r > early_r + 0.05:
                p("  -> Pitching has become MORE dominant relative to hitting over time.")
            elif late_r < early_r - 0.05:
                p("  -> Hitting has become MORE dominant relative to pitching over time.")
            else:
                p("  -> The balance between pitching and hitting has remained relatively stable.")

    # ------------------------------------------------------------------
    # Negative WAR trend
    # ------------------------------------------------------------------
    p()
    p("-" * 90)
    p("  REPLACEMENT-LEVEL PLAY")
    p("-" * 90)
    p()
    p(f"  {'Decade':<9}{'Neg WAR':>9}{'Qualifiers':>12}{'Pct Neg':>9}")
    p("  " + "-" * 39)
    for r in results:
        total_q = r["n_hit_qual"] + r["n_pit_qual"]
        pct = (r["negative"] / total_q * 100) if total_q > 0 else 0
        p(f"  {r['decade']:<9}{r['negative']:>9}{total_q:>12}{pct:>8.1f}%")

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    p()
    p("=" * 90)
    p("  VERDICT")
    p("=" * 90)
    p()
    if first_3 and last_3:
        early_std_v = np.mean([r["std_war"] for r in first_3])
        late_std_v = np.mean([r["std_war"] for r in last_3])
        early_top = np.mean([r["avg_top10_hit"] for r in first_3])
        late_top = np.mean([r["avg_top10_hit"] for r in last_3])

        if late_std_v < early_std_v and late_top <= early_top:
            p("  Baseball talent is becoming MORE EVENLY DISTRIBUTED.")
            p("  The talent pool is deeper, making it harder for individuals to stand out.")
            p("  The top-end WAR-eq numbers have declined as the average quality of play")
            p("  has risen — the classic 'Stephen Jay Gould' theory of vanishing .400 hitters.")
        elif late_std_v >= early_std_v and late_top > early_top:
            p("  Baseball talent is getting BETTER AND MORE CONCENTRATED.")
            p("  Top players today are producing at historic levels, with larger gaps to average.")
        elif late_std_v >= early_std_v and late_top <= early_top:
            p("  Talent spread is WIDENING but the peaks are not higher.")
            p("  This suggests greater variance in player quality without improvement at the top.")
        else:
            p("  Talent is getting better overall while the distribution compresses.")
            p("  The floor has risen faster than the ceiling.")
    else:
        p("  Insufficient data to draw era-spanning conclusions.")

    p()
    return buf.getvalue()


def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df):,} player-seasons, {df['yearID'].min()}-{df['yearID'].max()}")

    report = analyze(df)
    print(report)

    # Save to logs
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[Saved to {LOG_PATH}]")


if __name__ == "__main__":
    main()
