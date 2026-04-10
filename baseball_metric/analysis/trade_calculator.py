"""Trade Value Calculator — Surplus-value model for MLB trade evaluation.

Computes player trade value as projected future WAR * $/WAR - remaining salary
(surplus value over contract). For prospects, uses projected MLB WAR from the
prospect neural-net model.

Given a target player to acquire, finds what combination of prospects and/or
MLB players constitutes a fair-value trade package.

Usage:
    python -m baseball_metric.analysis.trade_calculator --player "Gunnar Henderson"
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────

# 2026 estimated market rate: ~$9.5M per WAR on free agent market
DOLLARS_PER_WAR = 9_500_000

# MLB minimum salary for 2026 estimate
MIN_SALARY = 770_000

# Discount rate for future WAR (time-value of money / risk)
DISCOUNT_RATE = 0.08

# Pitcher positions
PITCHER_POSITIONS = {"P", "SP", "RP"}

# Aging curves (simplified, from projections module)
HITTING_AGING = {
    20: 0.70, 21: 0.78, 22: 0.85, 23: 0.90, 24: 0.94,
    25: 0.97, 26: 0.99, 27: 1.00, 28: 0.99, 29: 0.97,
    30: 0.94, 31: 0.90, 32: 0.86, 33: 0.81, 34: 0.76,
    35: 0.70, 36: 0.63, 37: 0.56, 38: 0.48, 39: 0.40,
    40: 0.32,
}

PITCHING_AGING = {
    20: 0.65, 21: 0.75, 22: 0.83, 23: 0.89, 24: 0.93,
    25: 0.96, 26: 0.98, 27: 1.00, 28: 0.99, 29: 0.97,
    30: 0.95, 31: 0.92, 32: 0.88, 33: 0.84, 34: 0.79,
    35: 0.73, 36: 0.66, 37: 0.58, 38: 0.50, 39: 0.42,
    40: 0.34,
}

# Years of team control by service time (rough estimate)
# Pre-arb: 3 years, Arb: 3 years, then free agent
DEFAULT_CONTROL_YEARS = 6  # for young pre-arb players
ARB_SALARY_ESCALATION = [0.4, 0.6, 0.8]  # fraction of market rate in arb years


# ── Data classes ───────────────────────────────────────────────────────

@dataclass
class PlayerValue:
    """Computed trade value for a single player."""
    player_id: str
    name: str
    team: str
    position: str
    age: int
    projected_war_2026: float
    years_of_control: int
    total_projected_war: float
    total_salary: float
    gross_value: float
    surplus_value: float
    is_prospect: bool = False

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "team": self.team,
            "position": self.position,
            "age": self.age,
            "projected_war_2026": round(self.projected_war_2026, 1),
            "years_of_control": self.years_of_control,
            "total_projected_war": round(self.total_projected_war, 1),
            "total_salary": int(self.total_salary),
            "total_salary_fmt": f"${self.total_salary / 1_000_000:.1f}M",
            "gross_value": int(self.gross_value),
            "gross_value_fmt": f"${self.gross_value / 1_000_000:.1f}M",
            "surplus_value": int(self.surplus_value),
            "surplus_value_fmt": f"${self.surplus_value / 1_000_000:.1f}M",
            "is_prospect": self.is_prospect,
        }


@dataclass
class TradePackage:
    """A proposed trade package for a target player."""
    target: PlayerValue
    pieces: list[PlayerValue] = field(default_factory=list)

    @property
    def package_surplus(self) -> float:
        return sum(p.surplus_value for p in self.pieces)

    @property
    def value_match_pct(self) -> float:
        if self.target.surplus_value <= 0:
            return 100.0
        return (self.package_surplus / self.target.surplus_value) * 100

    def to_dict(self) -> dict:
        return {
            "target": self.target.to_dict(),
            "package": [p.to_dict() for p in self.pieces],
            "package_total_surplus": int(self.package_surplus),
            "package_total_surplus_fmt": f"${self.package_surplus / 1_000_000:.1f}M",
            "value_match_pct": round(self.value_match_pct, 1),
        }


# ── Data loading ───────────────────────────────────────────────────────

def _data_path(filename: str) -> str:
    """Resolve data file path relative to project root."""
    # Try relative to cwd first, then relative to this file
    if os.path.exists(filename):
        return filename
    base = os.path.join(os.path.dirname(__file__), "..", "..")
    return os.path.join(base, filename)


def load_projections() -> pd.DataFrame:
    """Load 2026 projections."""
    path = _data_path("data/projections_2026.csv")
    df = pd.read_csv(path)
    return df


def load_salaries() -> pd.DataFrame:
    """Load salary data — use most recent year available per player."""
    path = _data_path("data/salary_analysis.csv")
    df = pd.read_csv(path)
    # Keep most recent salary per player
    df = df.sort_values("yearID", ascending=False).drop_duplicates("playerID", keep="first")
    return df


def load_prospects() -> pd.DataFrame:
    """Load prospect rankings with projected MLB WAR."""
    path = _data_path("data/prospect_rankings.csv")
    df = pd.read_csv(path)
    return df


# ── Aging / projection helpers ─────────────────────────────────────────

def _aging_factor(age: int, is_pitcher: bool) -> float:
    """Get aging curve multiplier for a given age."""
    curve = PITCHING_AGING if is_pitcher else HITTING_AGING
    if age in curve:
        return curve[age]
    ages = sorted(curve.keys())
    if age < ages[0]:
        return curve[ages[0]]
    if age > ages[-1]:
        return max(curve[ages[-1]] - 0.05 * (age - ages[-1]), 0.0)
    return 0.5


def _estimate_control_years(age: int) -> int:
    """Estimate remaining years of team control based on age.

    Rough heuristic: younger players have more pre-arb + arb years.
    """
    if age <= 23:
        return 6
    elif age <= 24:
        return 5
    elif age <= 25:
        return 4
    elif age <= 26:
        return 3
    elif age <= 27:
        return 2
    elif age <= 28:
        return 1
    else:
        return 0  # likely free agent or near it


def _estimate_annual_salary(base_salary: float, year_offset: int, control_years: int) -> float:
    """Estimate salary for a future year based on control status.

    Pre-arb players earn near minimum. Arb players escalate.
    Free agents earn market rate (no surplus).
    """
    if year_offset >= control_years:
        # Free agent — no surplus value
        return float("inf")

    # If already making significant money, assume signed long-term deal
    if base_salary > 10_000_000:
        # Rough: salary stays similar with small escalation
        return base_salary * (1 + 0.03 * year_offset)

    # Pre-arb (first 3 control years)
    if year_offset < 3 and control_years >= 4:
        return max(MIN_SALARY, base_salary * 1.05)

    # Arbitration years
    arb_idx = min(year_offset - 3, 2) if year_offset >= 3 else 0
    arb_frac = ARB_SALARY_ESCALATION[arb_idx] if year_offset >= 3 else 0.05
    return max(MIN_SALARY, base_salary * (1 + arb_frac * (year_offset + 1)))


def project_future_war(
    war_2026: float, age_2026: int, is_pitcher: bool, years: int
) -> list[float]:
    """Project WAR for each of the next N years using aging curves."""
    current_factor = _aging_factor(age_2026, is_pitcher)
    if current_factor <= 0:
        return [0.0] * years
    peak_war = war_2026 / current_factor

    projected = []
    for yr in range(years):
        future_age = age_2026 + yr
        factor = _aging_factor(future_age, is_pitcher)
        war = peak_war * factor
        # Apply discount rate for future uncertainty
        discounted = war / ((1 + DISCOUNT_RATE) ** yr)
        projected.append(max(discounted, 0.0))
    return projected


# ── Core trade value computation ───────────────────────────────────────

def compute_player_value(
    player_id: str,
    projections: pd.DataFrame,
    salaries: pd.DataFrame,
    prospects: pd.DataFrame,
) -> Optional[PlayerValue]:
    """Compute the trade value (surplus value) for any player.

    Checks MLB projections first, then prospect rankings.
    """
    # Try MLB player first
    proj_match = projections[projections["playerID"] == player_id]
    if len(proj_match) > 0:
        row = proj_match.iloc[0]
        name = row["name"]
        team = row["team"]
        position = row["position"]
        age = int(row["age_2026"])
        war_2026 = float(row["projected_war"])
        is_pitcher = position in PITCHER_POSITIONS

        # Get salary
        sal_match = salaries[salaries["playerID"] == player_id]
        if len(sal_match) > 0:
            base_salary = float(sal_match.iloc[0]["salary"])
        else:
            base_salary = MIN_SALARY

        # Estimate control
        control_years = _estimate_control_years(age)

        # If high salary, assume longer term deal with fixed cost
        if base_salary > 15_000_000:
            control_years = max(control_years, 3)
        if base_salary > 25_000_000:
            control_years = max(control_years, 5)

        # Project future WAR
        future_wars = project_future_war(war_2026, age, is_pitcher, control_years)
        total_war = sum(future_wars)

        # Project future salary
        total_salary = 0.0
        for yr in range(control_years):
            annual_sal = _estimate_annual_salary(base_salary, yr, control_years)
            if annual_sal == float("inf"):
                break
            total_salary += annual_sal

        gross_value = total_war * DOLLARS_PER_WAR
        surplus = gross_value - total_salary

        return PlayerValue(
            player_id=player_id,
            name=name,
            team=team,
            position=position,
            age=age,
            projected_war_2026=war_2026,
            years_of_control=control_years,
            total_projected_war=total_war,
            total_salary=total_salary,
            gross_value=gross_value,
            surplus_value=surplus,
            is_prospect=False,
        )

    # Try prospect
    prosp_match = prospects[prospects["name"].str.lower() == player_id.lower()]
    if len(prosp_match) == 0:
        # Also try playerID column
        prosp_match = prospects[prospects["playerID"].astype(str) == str(player_id)]
    if len(prosp_match) > 0:
        row = prosp_match.iloc[0]
        name = str(row["name"])
        proj_mlb_war = float(row.get("projected_mlb_war", 0))

        # projected_mlb_war is total career WAR. Convert to annual rate
        # over a realistic 10-year career span, then only count the 6
        # controllable years. Cap annual rate at 5.0 WAR (even elite
        # prospects rarely sustain higher immediately).
        annual_rate = min(proj_mlb_war / 10.0, 5.0)

        # Prospects are young, assume 6 years of control at near-minimum salary
        control_years = 6
        age = 22  # rough estimate for a prospect
        future_wars = project_future_war(annual_rate, age, False, control_years)
        total_war = sum(future_wars)
        total_salary = MIN_SALARY * control_years
        gross_value = total_war * DOLLARS_PER_WAR
        surplus = gross_value - total_salary

        return PlayerValue(
            player_id=str(row.get("playerID", "")),
            name=name,
            team="MiLB",
            position="UTIL",
            age=age,
            projected_war_2026=0.0,
            years_of_control=control_years,
            total_projected_war=total_war,
            total_salary=total_salary,
            gross_value=gross_value,
            surplus_value=surplus,
            is_prospect=True,
        )

    return None


def compute_value_by_name(
    name: str,
    projections: pd.DataFrame,
    salaries: pd.DataFrame,
    prospects: pd.DataFrame,
) -> Optional[PlayerValue]:
    """Look up a player by name and compute trade value."""
    name_lower = name.strip().lower()

    # Search projections by name
    match = projections[projections["name"].str.lower() == name_lower]
    if len(match) > 0:
        pid = match.iloc[0]["playerID"]
        return compute_player_value(pid, projections, salaries, prospects)

    # Partial match: query contained in CSV name, or CSV name contained in query
    match = projections[
        projections["name"].str.lower().str.contains(name_lower, na=False)
        | projections["name"].str.lower().apply(lambda x: x in name_lower)
    ]
    if len(match) > 0:
        pid = match.iloc[0]["playerID"]
        return compute_player_value(pid, projections, salaries, prospects)

    # Try prospects by name (both directions)
    match = prospects[
        prospects["name"].str.lower().str.contains(name_lower, na=False)
        | prospects["name"].str.lower().apply(lambda x: x in name_lower)
    ]
    if len(match) > 0:
        return compute_player_value(match.iloc[0]["name"], projections, salaries, prospects)

    return None


# ── Trade package builder ──────────────────────────────────────────────

def _build_tradeable_pool(
    projections: pd.DataFrame,
    salaries: pd.DataFrame,
    prospects: pd.DataFrame,
    exclude_team: str = "",
) -> list[PlayerValue]:
    """Build a ranked pool of tradeable assets (prospects + controllable MLB).

    Returns players sorted by surplus value descending.
    """
    pool: list[PlayerValue] = []

    # Add controllable MLB players (age <= 28, not superstars)
    young = projections[
        (projections["age_2026"] <= 28)
        & (projections["projected_war"] > 0)
        & (projections["projected_war"] < 6)  # exclude franchise players
    ]
    if exclude_team:
        young = young[young["team"] != exclude_team]

    for _, row in young.iterrows():
        val = compute_player_value(row["playerID"], projections, salaries, prospects)
        if val and val.surplus_value > 0:
            pool.append(val)

    # Add top prospects
    top_prospects = prospects.nlargest(100, "projected_mlb_war")
    for _, row in top_prospects.iterrows():
        val = compute_player_value(str(row["name"]), projections, salaries, prospects)
        if val and val.surplus_value > 0:
            pool.append(val)

    # Deduplicate by player_id and sort
    seen = set()
    unique_pool = []
    for p in pool:
        key = p.player_id or p.name
        if key not in seen:
            seen.add(key)
            unique_pool.append(p)

    unique_pool.sort(key=lambda x: x.surplus_value, reverse=True)
    return unique_pool


def compute_trade_package(
    target_player: str,
    budget: float = float("inf"),
    projections: pd.DataFrame | None = None,
    salaries: pd.DataFrame | None = None,
    prospects: pd.DataFrame | None = None,
) -> Optional[TradePackage]:
    """Find a fair-value trade package for a target player.

    Args:
        target_player: Name of the player to acquire.
        budget: Maximum salary the acquiring team will take on.
        projections: Projections DataFrame (loaded if None).
        salaries: Salary DataFrame (loaded if None).
        prospects: Prospects DataFrame (loaded if None).

    Returns:
        TradePackage with pieces that approximate the target's surplus value,
        or None if the target player is not found.
    """
    if projections is None:
        projections = load_projections()
    if salaries is None:
        salaries = load_salaries()
    if prospects is None:
        prospects = load_prospects()

    target_val = compute_value_by_name(target_player, projections, salaries, prospects)
    if target_val is None:
        return None

    target_surplus = target_val.surplus_value
    if target_surplus <= 0:
        # Player is overpaid or a free agent — cheap to acquire
        return TradePackage(target=target_val, pieces=[])

    # Build pool of tradeable pieces (exclude target's team — you trade WITH them)
    pool = _build_tradeable_pool(
        projections, salaries, prospects, exclude_team=target_val.team
    )

    # Greedy knapsack: pick pieces that sum to ~100% of target surplus value
    # Aim for 90-110% of target value (realistic overpay)
    package_pieces: list[PlayerValue] = []
    remaining = target_surplus
    max_pieces = 5  # realistic trade limit

    for candidate in pool:
        if len(package_pieces) >= max_pieces:
            break
        if candidate.surplus_value <= 0:
            continue
        if budget < float("inf") and candidate.total_salary > budget:
            continue

        # Don't overshoot too much
        if remaining <= 0:
            break
        if candidate.surplus_value > remaining * 1.5 and len(package_pieces) > 0:
            continue

        package_pieces.append(candidate)
        remaining -= candidate.surplus_value

        # Close enough (within 10%)
        if remaining <= target_surplus * 0.10:
            break

    return TradePackage(target=target_val, pieces=package_pieces)


# ── CLI entry point ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trade Value Calculator")
    parser.add_argument("--player", type=str, help="Player name to evaluate")
    parser.add_argument("--package-for", type=str, help="Find trade package for this player")
    parser.add_argument("--budget", type=float, default=float("inf"),
                        help="Max salary budget for trade package (in millions)")
    args = parser.parse_args()

    print("Loading data...")
    projections = load_projections()
    salaries = load_salaries()
    prospects = load_prospects()

    if args.player:
        val = compute_value_by_name(args.player, projections, salaries, prospects)
        if val:
            print(f"\n{'='*60}")
            print(f"  Trade Value: {val.name}")
            print(f"{'='*60}")
            print(f"  Team: {val.team}  |  Position: {val.position}  |  Age: {val.age}")
            print(f"  2026 Projected WAR: {val.projected_war_2026:.1f}")
            print(f"  Years of Control: {val.years_of_control}")
            print(f"  Total Projected WAR: {val.total_projected_war:.1f}")
            print(f"  Gross Value: ${val.gross_value / 1e6:.1f}M")
            print(f"  Total Salary: ${val.total_salary / 1e6:.1f}M")
            print(f"  SURPLUS VALUE: ${val.surplus_value / 1e6:.1f}M")
            if val.is_prospect:
                print(f"  (Prospect)")
        else:
            print(f"Player '{args.player}' not found.")

    target_name = args.package_for or args.player
    if target_name:
        budget = args.budget * 1e6 if args.budget < float("inf") else float("inf")
        pkg = compute_trade_package(target_name, budget, projections, salaries, prospects)
        if pkg:
            print(f"\n{'='*60}")
            print(f"  Trade Package to Acquire: {pkg.target.name}")
            print(f"  Target Surplus Value: ${pkg.target.surplus_value / 1e6:.1f}M")
            print(f"{'='*60}")
            if not pkg.pieces:
                print("  Player has negative surplus — cheap to acquire (salary dump).")
            else:
                for i, p in enumerate(pkg.pieces, 1):
                    tag = " (prospect)" if p.is_prospect else ""
                    print(f"  {i}. {p.name} ({p.team}, {p.position}){tag}")
                    print(f"     Surplus: ${p.surplus_value / 1e6:.1f}M  |  "
                          f"Proj WAR: {p.total_projected_war:.1f}")
                print(f"\n  Package Total Surplus: ${pkg.package_surplus / 1e6:.1f}M")
                print(f"  Value Match: {pkg.value_match_pct:.0f}%")

    # Demo examples if no args provided
    if not args.player and not args.package_for:
        print("\n" + "=" * 60)
        print("  TRADE VALUE CALCULATOR — Demo Examples")
        print("=" * 60)

        examples = ["Gunnar Henderson", "Juan Soto", "Bobby Witt Jr."]
        for name in examples:
            val = compute_value_by_name(name, projections, salaries, prospects)
            if val:
                print(f"\n  {val.name} ({val.team}, {val.position}, age {val.age})")
                print(f"    2026 WAR: {val.projected_war_2026:.1f}  |  "
                      f"Control: {val.years_of_control}yr  |  "
                      f"Surplus: ${val.surplus_value / 1e6:.1f}M")

        # Trade package example
        print(f"\n{'='*60}")
        print("  What would it cost to trade for Gunnar Henderson?")
        print("=" * 60)
        pkg = compute_trade_package("Gunnar Henderson",
                                    projections=projections,
                                    salaries=salaries,
                                    prospects=prospects)
        if pkg and pkg.pieces:
            for i, p in enumerate(pkg.pieces, 1):
                tag = " (prospect)" if p.is_prospect else ""
                print(f"  {i}. {p.name} ({p.team}){tag} — "
                      f"${p.surplus_value / 1e6:.1f}M surplus")
            print(f"\n  Package: ${pkg.package_surplus / 1e6:.1f}M vs "
                  f"Target: ${pkg.target.surplus_value / 1e6:.1f}M "
                  f"({pkg.value_match_pct:.0f}% match)")


if __name__ == "__main__":
    main()
