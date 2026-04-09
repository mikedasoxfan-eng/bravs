"""Run every MVP and Cy Young race from 1956-2025 using pre-computed BRAVS data."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from collections import defaultdict

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
LOG_PATH = os.path.join(BASE_DIR, "logs", "all_award_races.log")


def load_bravs():
    """Load pre-computed BRAVS data for all seasons."""
    path = os.path.join(BASE_DIR, "data", "bravs_all_seasons.csv")
    df = pd.read_csv(path)
    # Keep only AL/NL
    df = df[df["lgID"].isin(["AL", "NL"])].copy()
    return df


def load_actual_winners():
    """Load actual MVP and Cy Young winners from Lahman AwardsPlayers.csv.

    Returns dict: (awardType, yearID, lgID) -> playerID
    awardType is 'MVP' or 'CY'.
    For pre-1967 Cy Young (lgID='ML'), we store under both AL and NL with
    a special 'ML' entry so we can match against whichever league the winner
    actually pitched in.
    """
    path = os.path.join(BASE_DIR, "data", "lahman2025", "AwardsPlayers.csv")
    awards_df = pd.read_csv(path)

    winners = {}

    # MVP winners
    mvp = awards_df[awards_df["awardID"] == "Most Valuable Player"]
    for _, row in mvp.iterrows():
        yr = int(row["yearID"])
        lg = row["lgID"]
        winners[("MVP", yr, lg)] = row["playerID"]

    # Cy Young winners
    cy = awards_df[awards_df["awardID"] == "Cy Young Award"]
    for _, row in cy.iterrows():
        yr = int(row["yearID"])
        lg = row["lgID"]
        if lg == "ML":
            # Pre-1967: single award across both leagues.
            # Store under ML so we can look it up regardless of league.
            winners[("CY", yr, "ML")] = row["playerID"]
        else:
            winners[("CY", yr, lg)] = row["playerID"]

    return winners


def get_actual_winner(winners, award_type, year, league):
    """Look up the actual winner, handling the pre-1967 Cy Young 'ML' case."""
    key = (award_type, year, league)
    if key in winners:
        return winners[key]
    # For Cy Young pre-1967, the award was given as 'ML' (both leagues)
    ml_key = (award_type, year, "ML")
    if ml_key in winners:
        return winners[ml_key]
    return None


def run_all_races(bravs_df, winners):
    """Run every award race and return structured results."""
    results = []
    years = range(1956, 2026)
    leagues = ["AL", "NL"]

    for year in years:
        for league in leagues:
            season = bravs_df[(bravs_df["yearID"] == year) & (bravs_df["lgID"] == league)]

            # --- MVP race: top 5 position players by bravs_war_eq, PA >= 400 ---
            mvp_pool = season[season["PA"] >= 400].nlargest(5, "bravs_war_eq")
            if len(mvp_pool) > 0:
                actual_pid = get_actual_winner(winners, "MVP", year, league)
                bravs_top = mvp_pool.iloc[0]
                bravs_pid = bravs_top["playerID"]
                bravs_val = bravs_top["bravs_war_eq"]

                # Find actual winner's BRAVS value
                actual_val = None
                actual_rank = None
                if actual_pid:
                    actual_row = season[season["playerID"] == actual_pid]
                    if len(actual_row) > 0:
                        actual_val = actual_row.iloc[0]["bravs_war_eq"]
                    # Find rank among all qualified
                    full_pool = season[season["PA"] >= 400].sort_values(
                        "bravs_war_eq", ascending=False
                    ).reset_index(drop=True)
                    match = full_pool[full_pool["playerID"] == actual_pid]
                    if len(match) > 0:
                        actual_rank = match.index[0] + 1

                agreed = (actual_pid is not None) and (bravs_pid == actual_pid)
                gap = (bravs_val - actual_val) if actual_val is not None else None

                results.append({
                    "award": "MVP",
                    "year": year,
                    "league": league,
                    "bravs_pick": bravs_pid,
                    "bravs_pick_name": bravs_top["name"],
                    "bravs_pick_val": bravs_val,
                    "actual_winner": actual_pid,
                    "actual_val": actual_val,
                    "actual_rank": actual_rank,
                    "agreed": agreed,
                    "gap": gap,
                    "top5": mvp_pool[["playerID", "name", "bravs_war_eq"]].values.tolist(),
                })

            # --- Cy Young race: top 3 pitchers by bravs_war_eq, IP >= 100 ---
            cy_pool = season[season["IP"] >= 100].nlargest(3, "bravs_war_eq")
            if len(cy_pool) > 0:
                actual_pid = get_actual_winner(winners, "CY", year, league)

                # Pre-1967 Cy Young was a single award for both leagues.
                # If the ML winner is NOT in this league's data, skip this league.
                if actual_pid is not None:
                    actual_in_league = season[season["playerID"] == actual_pid]
                    if len(actual_in_league) == 0:
                        # The award winner played in the other league -- skip
                        # this league for the pre-1967 single-award era.
                        key_specific = ("CY", year, league)
                        ml_key = ("CY", year, "ML")
                        if key_specific not in winners and ml_key in winners:
                            continue

                bravs_top = cy_pool.iloc[0]
                bravs_pid = bravs_top["playerID"]
                bravs_val = bravs_top["bravs_war_eq"]

                actual_val = None
                actual_rank = None
                if actual_pid:
                    actual_row = season[season["playerID"] == actual_pid]
                    if len(actual_row) > 0:
                        actual_val = actual_row.iloc[0]["bravs_war_eq"]
                    full_pool = season[season["IP"] >= 100].sort_values(
                        "bravs_war_eq", ascending=False
                    ).reset_index(drop=True)
                    match = full_pool[full_pool["playerID"] == actual_pid]
                    if len(match) > 0:
                        actual_rank = match.index[0] + 1

                agreed = (actual_pid is not None) and (bravs_pid == actual_pid)
                gap = (bravs_val - actual_val) if actual_val is not None else None

                results.append({
                    "award": "CY",
                    "year": year,
                    "league": league,
                    "bravs_pick": bravs_pid,
                    "bravs_pick_name": bravs_top["name"],
                    "bravs_pick_val": bravs_val,
                    "actual_winner": actual_pid,
                    "actual_val": actual_val,
                    "actual_rank": actual_rank,
                    "agreed": agreed,
                    "gap": gap,
                    "top5": cy_pool[["playerID", "name", "bravs_war_eq"]].values.tolist(),
                })

    return results


def format_report(results):
    """Format the full report as a string."""
    lines = []
    w = lines.append

    w("=" * 90)
    w("  BRAVS AWARD RACE ANALYSIS: EVERY MVP & CY YOUNG, 1956-2025")
    w("=" * 90)

    # --- Overall agreement ---
    mvp_results = [r for r in results if r["award"] == "MVP" and r["actual_winner"] is not None]
    cy_results = [r for r in results if r["award"] == "CY" and r["actual_winner"] is not None]

    mvp_agree = sum(1 for r in mvp_results if r["agreed"])
    cy_agree = sum(1 for r in cy_results if r["agreed"])
    total = len(mvp_results) + len(cy_results)
    total_agree = mvp_agree + cy_agree

    w("")
    w(f"  OVERALL AGREEMENT: {total_agree}/{total} ({100*total_agree/total:.1f}%)")
    w(f"    MVP:      {mvp_agree}/{len(mvp_results)} ({100*mvp_agree/len(mvp_results):.1f}%)")
    w(f"    Cy Young: {cy_agree}/{len(cy_results)} ({100*cy_agree/len(cy_results):.1f}%)")

    # --- Decade breakdown ---
    w("")
    w("-" * 90)
    w("  AGREEMENT BY DECADE")
    w("-" * 90)

    decade_data = defaultdict(lambda: {"agree": 0, "total": 0})
    for r in results:
        if r["actual_winner"] is None:
            continue
        decade = (r["year"] // 10) * 10
        decade_data[decade]["total"] += 1
        if r["agreed"]:
            decade_data[decade]["agree"] += 1

    best_decade = None
    worst_decade = None
    best_pct = -1
    worst_pct = 2

    for decade in sorted(decade_data.keys()):
        d = decade_data[decade]
        pct = d["agree"] / d["total"] if d["total"] > 0 else 0
        label = f"{decade}s"
        bar = "#" * int(pct * 40)
        w(f"    {label:<8} {d['agree']:>3}/{d['total']:<3}  ({100*pct:5.1f}%)  {bar}")
        if pct > best_pct:
            best_pct = pct
            best_decade = decade
        if pct < worst_pct:
            worst_pct = pct
            worst_decade = decade

    w("")
    w(f"  Best decade:  {best_decade}s ({100*best_pct:.1f}% agreement)")
    w(f"  Worst decade: {worst_decade}s ({100*worst_pct:.1f}% agreement)")

    # --- All disagreements ---
    disagreements = [r for r in results if r["actual_winner"] is not None and not r["agreed"]]
    disagreements.sort(key=lambda r: (r["gap"] if r["gap"] is not None else 0), reverse=True)

    w("")
    w("-" * 90)
    w("  ALL DISAGREEMENTS (BRAVS #1 != actual winner)")
    w("-" * 90)
    w(f"  {'Year':<6}{'Lg':<4}{'Award':<6}{'BRAVS Pick':<24}{'WAReq':>6}"
      f"  {'Actual Winner':<20}{'WAReq':>6}{'Gap':>7}{'Rank':>6}")
    w("  " + "-" * 86)

    for r in disagreements:
        bravs_name = r["bravs_pick_name"]
        actual_pid = r["actual_winner"]
        # Find actual winner name from top5 or from the data
        actual_name = actual_pid  # fallback
        for entry in r["top5"]:
            if entry[0] == actual_pid:
                actual_name = entry[1]
                break
        actual_val_str = f"{r['actual_val']:.1f}" if r["actual_val"] is not None else "N/A"
        gap_str = f"{r['gap']:+.1f}" if r["gap"] is not None else "N/A"
        rank_str = f"#{r['actual_rank']}" if r["actual_rank"] is not None else "N/A"

        w(f"  {r['year']:<6}{r['league']:<4}{r['award']:<6}{bravs_name:<24}"
          f"{r['bravs_pick_val']:>6.1f}  {actual_name:<20}{actual_val_str:>6}"
          f"{gap_str:>7}{rank_str:>6}")

    # --- Biggest snub ---
    snubs_with_gap = [r for r in disagreements if r["gap"] is not None]
    if snubs_with_gap:
        biggest = max(snubs_with_gap, key=lambda r: r["gap"])
        w("")
        w("=" * 90)
        w("  BIGGEST SNUB IN HISTORY")
        w("=" * 90)
        actual_name = biggest["actual_winner"]
        for entry in biggest["top5"]:
            if entry[0] == biggest["actual_winner"]:
                actual_name = entry[1]
                break
        w(f"  {biggest['year']} {biggest['league']} {biggest['award']}")
        w(f"    BRAVS says:  {biggest['bravs_pick_name']} "
          f"({biggest['bravs_pick_val']:.1f} WAR-eq)")
        actual_val_str = (f"{biggest['actual_val']:.1f}"
                          if biggest["actual_val"] is not None else "N/A")
        w(f"    Actual winner: {actual_name} ({actual_val_str} WAR-eq)")
        w(f"    Gap: {biggest['gap']:+.1f} WAR-eq")
        if biggest["actual_rank"] is not None:
            w(f"    Actual winner ranked #{biggest['actual_rank']} by BRAVS")

    # --- Full race listings ---
    w("")
    w("=" * 90)
    w("  FULL RACE LISTINGS")
    w("=" * 90)

    for r in sorted(results, key=lambda x: (x["year"], x["league"], x["award"])):
        award_label = "MVP" if r["award"] == "MVP" else "Cy Young"
        marker = "AGREE" if r["agreed"] else "DISAGREE" if r["actual_winner"] else "NO DATA"
        w("")
        w(f"  {r['year']} {r['league']} {award_label}  [{marker}]")
        n = len(r["top5"])
        label = "Top 5" if r["award"] == "MVP" else "Top 3"
        for i, entry in enumerate(r["top5"]):
            pid, name, val = entry
            tag = ""
            if r["actual_winner"] and pid == r["actual_winner"]:
                tag = " <-- ACTUAL WINNER"
            if i == 0:
                tag += " <-- BRAVS #1" if tag else " <-- BRAVS #1"
            w(f"    {i+1}. {name:<28} {val:>6.1f}{tag}")

        # If actual winner not in the top list, show where they are
        if r["actual_winner"] and r["actual_rank"] is not None:
            in_top = any(entry[0] == r["actual_winner"] for entry in r["top5"])
            if not in_top:
                actual_val_str = (f"{r['actual_val']:.1f}"
                                  if r["actual_val"] is not None else "N/A")
                w(f"    Actual winner: {r['actual_winner']} "
                  f"(#{r['actual_rank']}, {actual_val_str} WAR-eq)")

    return "\n".join(lines)


def main():
    print("Loading BRAVS data...")
    bravs_df = load_bravs()
    print(f"  {len(bravs_df):,} player-seasons loaded")

    print("Loading actual award winners...")
    winners = load_actual_winners()
    mvp_count = sum(1 for k in winners if k[0] == "MVP")
    cy_count = sum(1 for k in winners if k[0] == "CY")
    print(f"  {mvp_count} MVP winners, {cy_count} Cy Young winners")

    print("Running all award races 1956-2025...")
    results = run_all_races(bravs_df, winners)
    print(f"  {len(results)} total races evaluated")

    report = format_report(results)

    # Print to stdout
    print(report)

    # Save to log
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSaved to {LOG_PATH}")


if __name__ == "__main__":
    main()
