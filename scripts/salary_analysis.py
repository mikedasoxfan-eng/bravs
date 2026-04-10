"""$/WAR Salary Valuation Model.

Merges Lahman salary data (1985-2016) and baseball-main salary data (2017-2025)
with BRAVS season data to compute $/WAR curves, surplus value, and identify
the best and worst contracts in baseball history.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent / "data"
LAHMAN_SALARIES = BASE / "lahman2025" / "Salaries.csv"
BB_SALARIES_DIR = BASE / "baseball-main" / "data" / "salaries"
BRAVS_SEASONS = BASE / "bravs_all_seasons.csv"
BRAVS_CAREERS = BASE / "bravs_careers.csv"
OUTPUT = BASE / "salary_analysis.csv"


# ---------------------------------------------------------------------------
# 1. Load salary data
# ---------------------------------------------------------------------------
def load_lahman_salaries() -> pd.DataFrame:
    """Load Lahman Salaries.csv (1985-2016): playerID, yearID, teamID, lgID, salary."""
    df = pd.read_csv(LAHMAN_SALARIES)
    df = df.rename(columns=str.strip)
    # Ensure types
    df["yearID"] = df["yearID"].astype(int)
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    return df[["playerID", "yearID", "teamID", "lgID", "salary"]].dropna(subset=["salary"])


def _parse_bb_name_to_lahman_hint(name_str: str) -> str:
    """Parse 'Last, First' format from baseball-main into 'first last' for matching."""
    if "," in name_str:
        parts = name_str.split(",", 1)
        return (parts[1].strip() + " " + parts[0].strip()).lower()
    return name_str.strip().lower()


def load_baseball_main_salaries() -> pd.DataFrame:
    """Load per-year CSVs from baseball-main (2017-2025).
    Format: Year, Player, Pos, Salary
    These have 'Last, First' names — we'll need to match to Lahman playerIDs later.
    """
    frames = []
    if not BB_SALARIES_DIR.exists():
        return pd.DataFrame()

    for fname in sorted(BB_SALARIES_DIR.glob("*.csv")):
        if fname.stem == "summary" or not fname.stem.isdigit():
            continue
        year = int(fname.stem)
        if year <= 2016:
            # Lahman already covers these
            continue
        try:
            df = pd.read_csv(fname)
            df = df.rename(columns=str.strip)
            df["yearID"] = year
            df["salary"] = pd.to_numeric(df["Salary"], errors="coerce")
            df["player_name_raw"] = df["Player"].astype(str)
            frames.append(df[["yearID", "player_name_raw", "salary"]].dropna(subset=["salary"]))
        except Exception as e:
            print(f"  Warning: could not read {fname}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. Load BRAVS data
# ---------------------------------------------------------------------------
def load_bravs() -> pd.DataFrame:
    """Load BRAVS season data."""
    df = pd.read_csv(BRAVS_SEASONS)
    df["yearID"] = df["yearID"].astype(int)
    return df


# ---------------------------------------------------------------------------
# 3. Merge salary + BRAVS
# ---------------------------------------------------------------------------
def merge_lahman(salaries: pd.DataFrame, bravs: pd.DataFrame) -> pd.DataFrame:
    """Merge Lahman salaries with BRAVS on playerID + yearID."""
    merged = salaries.merge(
        bravs[["playerID", "yearID", "name", "team", "position", "bravs_war_eq", "bravs", "G", "PA", "IP"]],
        on=["playerID", "yearID"],
        how="inner",
    )
    return merged


def merge_baseball_main(bb_salaries: pd.DataFrame, bravs: pd.DataFrame) -> pd.DataFrame:
    """Merge baseball-main salaries (name-based) with BRAVS.

    baseball-main uses 'Last, First' format. BRAVS uses 'First Last'.
    We build a lookup from BRAVS name -> playerID, then fuzzy match.
    """
    if bb_salaries.empty:
        return pd.DataFrame()

    # Build a mapping: (yearID, normalized_name) -> BRAVS row
    bravs_by_year_name = {}
    for _, row in bravs.iterrows():
        yr = int(row["yearID"])
        name_lower = str(row["name"]).strip().lower()
        bravs_by_year_name[(yr, name_lower)] = row

    results = []
    for _, sal_row in bb_salaries.iterrows():
        yr = int(sal_row["yearID"])
        raw_name = sal_row["player_name_raw"]
        search_name = _parse_bb_name_to_lahman_hint(raw_name)

        # Try exact match first
        bravs_row = bravs_by_year_name.get((yr, search_name))

        if bravs_row is None:
            # Try partial: match last name + first initial
            parts = search_name.split()
            if len(parts) >= 2:
                first = parts[0]
                last = parts[-1]
                for (byr, bname), brow in bravs_by_year_name.items():
                    if byr == yr and last in bname and bname.startswith(first[:3]):
                        bravs_row = brow
                        break

        if bravs_row is not None:
            results.append({
                "playerID": bravs_row["playerID"],
                "yearID": yr,
                "teamID": str(bravs_row.get("team", "")),
                "lgID": str(bravs_row.get("lgID", "")),
                "salary": sal_row["salary"],
                "name": bravs_row["name"],
                "team": bravs_row.get("team", ""),
                "position": bravs_row.get("position", ""),
                "bravs_war_eq": float(bravs_row["bravs_war_eq"]),
                "bravs": float(bravs_row["bravs"]),
                "G": int(bravs_row.get("G", 0)),
                "PA": int(bravs_row.get("PA", 0)),
                "IP": float(bravs_row.get("IP", 0)),
            })

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 4. Compute $/WAR curves by year
# ---------------------------------------------------------------------------
def compute_dollar_per_war(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute the market $/WAR rate for each year.

    Method: For each year, take players with >= 2 WAR (established contributors)
    and compute median salary / median WAR. This avoids distortion from
    replacement-level players and mega-contracts on injured stars.

    Also compute the 'free agent' rate using higher-salary players (top quartile)
    which better reflects what teams actually pay on the open market.
    """
    records = []
    for year, group in merged.groupby("yearID"):
        # Filter to meaningful seasons (at least some games)
        qualified = group[group["bravs_war_eq"] >= 0.5].copy()
        if len(qualified) < 10:
            continue

        # Overall market rate: median of (salary / WAR) for 2+ WAR players
        contributors = qualified[qualified["bravs_war_eq"] >= 2.0]
        if len(contributors) < 5:
            contributors = qualified[qualified["bravs_war_eq"] >= 1.0]

        if len(contributors) < 3:
            continue

        dollar_per_war_values = contributors["salary"] / contributors["bravs_war_eq"]
        median_rate = dollar_per_war_values.median()
        mean_rate = dollar_per_war_values.mean()

        # Free agent rate: top salary quartile
        salary_75 = qualified["salary"].quantile(0.75)
        fa_tier = qualified[qualified["salary"] >= salary_75]
        if len(fa_tier) > 0 and fa_tier["bravs_war_eq"].sum() > 0:
            fa_rate = fa_tier["salary"].sum() / fa_tier["bravs_war_eq"].clip(lower=0.1).sum()
        else:
            fa_rate = median_rate

        total_salary = qualified["salary"].sum()
        total_war = qualified["bravs_war_eq"].sum()
        avg_rate = total_salary / total_war if total_war > 0 else 0

        records.append({
            "yearID": int(year),
            "n_players": len(qualified),
            "n_contributors": len(contributors),
            "median_dollar_per_war": round(median_rate),
            "mean_dollar_per_war": round(mean_rate),
            "aggregate_dollar_per_war": round(avg_rate),
            "fa_dollar_per_war": round(fa_rate),
            "total_salary": round(total_salary),
            "total_war": round(total_war, 1),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. Compute surplus value for each player-season
# ---------------------------------------------------------------------------
def compute_surplus_value(merged: pd.DataFrame, dollar_per_war: pd.DataFrame) -> pd.DataFrame:
    """Add surplus value columns to merged data.

    Surplus = (WAR * market_rate) - actual_salary
    Positive = team got a bargain, Negative = team overpaid.
    """
    rate_map = dict(zip(dollar_per_war["yearID"], dollar_per_war["median_dollar_per_war"]))

    merged = merged.copy()
    merged["market_rate"] = merged["yearID"].map(rate_map)
    merged["market_value"] = merged["bravs_war_eq"] * merged["market_rate"]
    merged["surplus_value"] = merged["market_value"] - merged["salary"]
    merged["surplus_pct"] = (merged["surplus_value"] / merged["salary"].clip(lower=1)) * 100

    # Inflation-adjusted surplus (normalize all to 2016 dollars using $/WAR growth)
    rate_2016 = rate_map.get(2016, rate_map.get(max(rate_map.keys())))
    merged["inflation_factor"] = merged["market_rate"].apply(
        lambda x: rate_2016 / x if x and x > 0 else 1.0
    )
    merged["surplus_adj"] = merged["surplus_value"] * merged["inflation_factor"]

    return merged


# ---------------------------------------------------------------------------
# 6. Identify best / worst contracts
# ---------------------------------------------------------------------------
def find_best_worst_contracts(merged: pd.DataFrame, n: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find the best and worst single-season contracts by surplus value."""
    # Filter to meaningful seasons
    qualified = merged[
        (merged["bravs_war_eq"].notna()) &
        (merged["salary"].notna()) &
        (merged["market_rate"].notna()) &
        (merged["G"] >= 20)
    ].copy()

    best = qualified.nlargest(n, "surplus_value")
    worst = qualified.nsmallest(n, "surplus_value")

    return best, worst


def find_best_worst_adjusted(merged: pd.DataFrame, n: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Best/worst contracts inflation-adjusted."""
    qualified = merged[
        (merged["surplus_adj"].notna()) &
        (merged["salary"].notna()) &
        (merged["G"] >= 20)
    ].copy()

    best = qualified.nlargest(n, "surplus_adj")
    worst = qualified.nsmallest(n, "surplus_adj")

    return best, worst


# ---------------------------------------------------------------------------
# 7. Print key findings
# ---------------------------------------------------------------------------
def print_findings(
    dollar_per_war: pd.DataFrame,
    best: pd.DataFrame,
    worst: pd.DataFrame,
    best_adj: pd.DataFrame,
    worst_adj: pd.DataFrame,
    merged: pd.DataFrame,
) -> None:
    """Print a summary of key findings."""
    print("\n" + "=" * 72)
    print("  $/WAR SALARY VALUATION MODEL — KEY FINDINGS")
    print("=" * 72)

    # $/WAR curve
    print("\n--- $/WAR by Year (median market rate) ---")
    print(f"  {'Year':<6} {'$/WAR':>12} {'FA $/WAR':>12} {'Players':>8}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*8}")
    for _, row in dollar_per_war.iterrows():
        yr = int(row["yearID"])
        rate = row["median_dollar_per_war"]
        fa_rate = row["fa_dollar_per_war"]
        n = int(row["n_players"])
        print(f"  {yr:<6} ${rate:>11,.0f} ${fa_rate:>11,.0f} {n:>8}")

    # Highlight key benchmarks
    print("\n--- Key Benchmarks ---")
    for benchmark_year in [1990, 2000, 2005, 2010, 2015, 2016]:
        row = dollar_per_war[dollar_per_war["yearID"] == benchmark_year]
        if not row.empty:
            rate = row.iloc[0]["median_dollar_per_war"]
            print(f"  {benchmark_year}: ${rate:,.0f} per WAR")

    # Best contracts (raw)
    print(f"\n--- Top 15 Best Contracts (raw surplus) ---")
    print(f"  {'Player':<22} {'Year':>5} {'WAR':>5} {'Salary':>12} {'Market Val':>12} {'Surplus':>12}")
    print(f"  {'-'*22} {'-'*5} {'-'*5} {'-'*12} {'-'*12} {'-'*12}")
    for _, row in best.head(15).iterrows():
        print(f"  {str(row['name'])[:22]:<22} {int(row['yearID']):>5} {row['bravs_war_eq']:>5.1f} "
              f"${row['salary']:>11,.0f} ${row['market_value']:>11,.0f} ${row['surplus_value']:>11,.0f}")

    # Worst contracts (raw)
    print(f"\n--- Top 15 Worst Contracts (raw surplus) ---")
    print(f"  {'Player':<22} {'Year':>5} {'WAR':>5} {'Salary':>12} {'Market Val':>12} {'Surplus':>12}")
    print(f"  {'-'*22} {'-'*5} {'-'*5} {'-'*12} {'-'*12} {'-'*12}")
    for _, row in worst.head(15).iterrows():
        print(f"  {str(row['name'])[:22]:<22} {int(row['yearID']):>5} {row['bravs_war_eq']:>5.1f} "
              f"${row['salary']:>11,.0f} ${row['market_value']:>11,.0f} ${row['surplus_value']:>11,.0f}")

    # Best contracts (inflation-adjusted)
    print(f"\n--- Top 15 Best Contracts (inflation-adjusted to 2016 dollars) ---")
    print(f"  {'Player':<22} {'Year':>5} {'WAR':>5} {'Adj Surplus':>14}")
    print(f"  {'-'*22} {'-'*5} {'-'*5} {'-'*14}")
    for _, row in best_adj.head(15).iterrows():
        print(f"  {str(row['name'])[:22]:<22} {int(row['yearID']):>5} {row['bravs_war_eq']:>5.1f} "
              f"${row['surplus_adj']:>13,.0f}")

    # Worst contracts (inflation-adjusted)
    print(f"\n--- Top 15 Worst Contracts (inflation-adjusted to 2016 dollars) ---")
    print(f"  {'Player':<22} {'Year':>5} {'WAR':>5} {'Adj Surplus':>14}")
    print(f"  {'-'*22} {'-'*5} {'-'*5} {'-'*14}")
    for _, row in worst_adj.head(15).iterrows():
        print(f"  {str(row['name'])[:22]:<22} {int(row['yearID']):>5} {row['bravs_war_eq']:>5.1f} "
              f"${row['surplus_adj']:>13,.0f}")

    # Summary stats
    print(f"\n--- Summary ---")
    print(f"  Total player-seasons with salary + BRAVS: {len(merged):,}")
    print(f"  Year range: {int(merged['yearID'].min())} - {int(merged['yearID'].max())}")
    total_surplus = merged["surplus_value"].sum()
    avg_surplus = merged["surplus_value"].mean()
    print(f"  Average surplus per player-season: ${avg_surplus:,.0f}")
    print(f"  Median surplus per player-season:  ${merged['surplus_value'].median():,.0f}")

    # Biggest bargain teams
    print(f"\n--- Teams with Most Surplus Value (all years combined) ---")
    team_surplus = merged.groupby("team")["surplus_value"].sum().sort_values(ascending=False)
    for team, surplus in team_surplus.head(10).items():
        print(f"  {team:<6} ${surplus:>14,.0f}")

    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading salary data...")
    lahman = load_lahman_salaries()
    print(f"  Lahman: {len(lahman):,} records ({int(lahman.yearID.min())}-{int(lahman.yearID.max())})")

    bb_main = load_baseball_main_salaries()
    if not bb_main.empty:
        print(f"  Baseball-main: {len(bb_main):,} records ({int(bb_main.yearID.min())}-{int(bb_main.yearID.max())})")
    else:
        print("  Baseball-main: no additional salary data found")

    print("\nLoading BRAVS season data...")
    bravs = load_bravs()
    print(f"  {len(bravs):,} player-seasons")

    print("\nMerging Lahman salaries with BRAVS...")
    merged_lahman = merge_lahman(lahman, bravs)
    print(f"  Matched: {len(merged_lahman):,} player-seasons")

    merged_bb = merge_baseball_main(bb_main, bravs)
    if not merged_bb.empty:
        print(f"\nMerging baseball-main salaries with BRAVS...")
        print(f"  Matched: {len(merged_bb):,} player-seasons")

    # Combine all merged data
    if not merged_bb.empty:
        # Ensure same columns
        common_cols = list(set(merged_lahman.columns) & set(merged_bb.columns))
        merged = pd.concat([merged_lahman[common_cols], merged_bb[common_cols]], ignore_index=True)
    else:
        merged = merged_lahman

    print(f"\nTotal merged records: {len(merged):,}")

    # Compute $/WAR curves
    print("\nComputing $/WAR curves by year...")
    dollar_per_war = compute_dollar_per_war(merged)
    print(f"  Computed for {len(dollar_per_war)} years")

    # Compute surplus value
    print("Computing surplus values...")
    merged = compute_surplus_value(merged, dollar_per_war)

    # Find best/worst contracts
    best, worst = find_best_worst_contracts(merged)
    best_adj, worst_adj = find_best_worst_adjusted(merged)

    # Save results
    print(f"\nSaving results to {OUTPUT}...")
    out_cols = [
        "playerID", "yearID", "name", "team", "position", "salary",
        "bravs_war_eq", "bravs", "market_rate", "market_value",
        "surplus_value", "surplus_pct", "surplus_adj",
    ]
    available_cols = [c for c in out_cols if c in merged.columns]
    merged[available_cols].sort_values(
        ["yearID", "surplus_value"], ascending=[True, False]
    ).to_csv(OUTPUT, index=False)
    print(f"  Saved {len(merged):,} records")

    # Print findings
    print_findings(dollar_per_war, best, worst, best_adj, worst_adj, merged)


if __name__ == "__main__":
    main()
