"""Roster Optimizer — Build the best 26-man roster under a payroll budget.

Given BRAVS projections and salary data, uses greedy $/WAR optimization to
fill a 26-man roster:
  - 13 position players: C, 1B, 2B, 3B, SS, LF, CF, RF, DH + 4 bench
  - 13 pitchers: 5 SP + 8 RP

Usage:
    python -m baseball_metric.analysis.roster_optimizer --budget 200
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────

# Minimum salary (MLB minimum ~$740K in 2025, round up for 2026)
MIN_SALARY = 750_000

# Position slots that must be filled for a 26-man roster
REQUIRED_POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
NUM_BENCH = 4
NUM_SP = 5
NUM_RP = 8

# Which projection positions map to pitcher
PITCHER_POSITIONS = {"P", "SP", "RP"}


@dataclass
class RosterSlot:
    position: str
    player_id: str
    name: str
    team: str
    projected_war: float
    salary: float
    dollar_per_war: float


@dataclass
class OptimalRoster:
    budget: float
    position_players: list[RosterSlot] = field(default_factory=list)
    pitchers: list[RosterSlot] = field(default_factory=list)

    @property
    def total_war(self) -> float:
        return sum(s.projected_war for s in self.position_players + self.pitchers)

    @property
    def total_salary(self) -> float:
        return sum(s.salary for s in self.position_players + self.pitchers)

    @property
    def roster_size(self) -> int:
        return len(self.position_players) + len(self.pitchers)

    def to_dict(self) -> dict:
        """Serialize for JSON API response."""
        def slot_dict(s: RosterSlot) -> dict:
            return {
                "position": s.position,
                "player_id": s.player_id,
                "name": s.name,
                "team": s.team,
                "projected_war": round(s.projected_war, 1),
                "salary": int(s.salary),
                "salary_fmt": f"${s.salary / 1_000_000:.1f}M",
                "dollar_per_war": round(s.dollar_per_war, 1) if s.dollar_per_war < 1e9 else None,
            }

        return {
            "budget": int(self.budget),
            "budget_fmt": f"${self.budget / 1_000_000:.0f}M",
            "total_war": round(self.total_war, 1),
            "total_salary": int(self.total_salary),
            "total_salary_fmt": f"${self.total_salary / 1_000_000:.1f}M",
            "remaining_budget": int(self.budget - self.total_salary),
            "remaining_budget_fmt": f"${(self.budget - self.total_salary) / 1_000_000:.1f}M",
            "roster_size": self.roster_size,
            "position_players": [slot_dict(s) for s in self.position_players],
            "pitchers": [slot_dict(s) for s in self.pitchers],
        }


# ── Data Loading ───────────────────────────────────────────────────────

def _find_data_dir() -> str:
    """Locate the data/ directory relative to project root."""
    # Try common locations
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "data"),
        os.path.join(os.getcwd(), "data"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    raise FileNotFoundError("Cannot find data/ directory")


def load_projections(data_dir: str | None = None) -> pd.DataFrame:
    """Load 2026 BRAVS projections."""
    if data_dir is None:
        data_dir = _find_data_dir()
    path = os.path.join(data_dir, "projections_2026.csv")
    df = pd.read_csv(path)
    # Standardize columns
    df = df.rename(columns={"playerID": "player_id"})
    if "player_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "player_id"})
    return df


def load_salaries(data_dir: str | None = None) -> pd.DataFrame:
    """Load salary data, keeping the most recent salary per player."""
    if data_dir is None:
        data_dir = _find_data_dir()
    path = os.path.join(data_dir, "salary_analysis.csv")
    df = pd.read_csv(path)
    df = df.rename(columns={"playerID": "player_id"})
    if "player_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "player_id"})
    # Keep most recent year per player
    df = df.sort_values("yearID", ascending=False)
    df = df.drop_duplicates(subset=["player_id"], keep="first")
    return df


def merge_projection_salary(
    projections: pd.DataFrame, salaries: pd.DataFrame
) -> pd.DataFrame:
    """Merge projections with salary data. Players without salary get minimum."""
    merged = projections.merge(
        salaries[["player_id", "salary", "yearID"]],
        on="player_id",
        how="left",
    )
    # Fill missing salaries with league minimum
    merged["salary"] = merged["salary"].fillna(MIN_SALARY)
    # Ensure salary is at least minimum
    merged.loc[merged["salary"] < MIN_SALARY, "salary"] = MIN_SALARY
    # Compute $/WAR (avoid division by zero)
    merged["war_for_sort"] = merged["projected_war"].clip(lower=0.01)
    merged["dollar_per_war"] = merged["salary"] / merged["war_for_sort"]
    return merged


# ── Optimizer ──────────────────────────────────────────────────────────

def _is_pitcher(pos: str) -> bool:
    return pos.upper() in PITCHER_POSITIONS


def _position_eligible(player_pos: str, target_pos: str) -> bool:
    """Check if a player can fill a target position slot."""
    player_pos = player_pos.upper()
    target_pos = target_pos.upper()
    if player_pos == target_pos:
        return True
    # DH can be filled by anyone
    if target_pos == "DH":
        return player_pos not in PITCHER_POSITIONS
    # Bench can be filled by any position player
    if target_pos == "BENCH":
        return player_pos not in PITCHER_POSITIONS
    # Corner outfielders can swap
    if target_pos in ("LF", "RF") and player_pos in ("LF", "RF", "CF"):
        return True
    # CF can play corners but not vice versa without penalty
    return False


def optimize_roster(
    budget: float,
    data_dir: str | None = None,
    projections: pd.DataFrame | None = None,
    salaries: pd.DataFrame | None = None,
) -> OptimalRoster:
    """Find the optimal 26-man roster under a payroll budget.

    Uses greedy optimization: fill each required position with the best
    value (lowest $/WAR) player available that fits under budget.

    Args:
        budget: Total payroll budget in dollars (e.g., 200_000_000).
        data_dir: Path to data directory. Auto-detected if None.
        projections: Pre-loaded projections DataFrame.
        salaries: Pre-loaded salaries DataFrame.

    Returns:
        OptimalRoster with filled position and pitcher slots.
    """
    if projections is None:
        projections = load_projections(data_dir)
    if salaries is None:
        salaries = load_salaries(data_dir)

    pool = merge_projection_salary(projections, salaries)
    roster = OptimalRoster(budget=budget)
    used_ids: set[str] = set()
    remaining_budget = budget

    # Total slots to fill
    total_slots = len(REQUIRED_POSITIONS) + NUM_BENCH + NUM_SP + NUM_RP  # 26

    # ── Helper: pick best available for a position ──
    def pick_best(
        target_pos: str,
        is_pitcher_slot: bool = False,
        slot_budget: float | None = None,
    ) -> RosterSlot | None:
        nonlocal remaining_budget
        max_spend = min(remaining_budget, slot_budget) if slot_budget else remaining_budget

        if is_pitcher_slot:
            candidates = pool[
                pool["position"].apply(_is_pitcher)
                & ~pool["player_id"].isin(used_ids)
                & (pool["salary"] <= max_spend)
                & (pool["projected_war"] > 0)
            ].copy()
        else:
            candidates = pool[
                pool["position"].apply(lambda p: _position_eligible(p, target_pos))
                & ~pool["player_id"].isin(used_ids)
                & (pool["salary"] <= max_spend)
                & (pool["projected_war"] > 0)
            ].copy()

        if candidates.empty:
            # Fall back: allow negative-WAR players if no positive ones
            if is_pitcher_slot:
                candidates = pool[
                    pool["position"].apply(_is_pitcher)
                    & ~pool["player_id"].isin(used_ids)
                    & (pool["salary"] <= max_spend)
                ].copy()
            else:
                candidates = pool[
                    pool["position"].apply(lambda p: _position_eligible(p, target_pos))
                    & ~pool["player_id"].isin(used_ids)
                    & (pool["salary"] <= max_spend)
                ].copy()
            if candidates.empty:
                return None
            # Pick cheapest
            best = candidates.nsmallest(1, "salary").iloc[0]
        else:
            # Pick highest projected WAR within budget
            best = candidates.nlargest(1, "projected_war").iloc[0]

        slot = RosterSlot(
            position=target_pos if not is_pitcher_slot else best["position"],
            player_id=best["player_id"],
            name=best["name"],
            team=best["team"],
            projected_war=best["projected_war"],
            salary=best["salary"],
            dollar_per_war=best["dollar_per_war"],
        )
        used_ids.add(best["player_id"])
        remaining_budget -= best["salary"]
        return slot

    # ── Fill required position slots (scarcest positions first) ──
    # Catchers are scarce, fill them first; then middle infield, etc.
    # Starters get a larger share of per-slot budget; bench/RP get less.
    position_priority = ["C", "SS", "2B", "CF", "3B", "1B", "LF", "RF", "DH"]
    # Allocate budget: starters get 60%, pitchers 30%, bench 10%
    starter_budget_each = (budget * 0.60) / len(position_priority)
    bench_budget_each = (budget * 0.10) / NUM_BENCH
    sp_budget_each = (budget * 0.20) / NUM_SP
    rp_budget_each = (budget * 0.10) / NUM_RP

    for pos in position_priority:
        slot = pick_best(pos, slot_budget=starter_budget_each)
        if slot:
            slot.position = pos
            roster.position_players.append(slot)

    # ── Fill bench spots (best available position players) ──
    for _ in range(NUM_BENCH):
        slot = pick_best("BENCH", slot_budget=bench_budget_each)
        if slot:
            slot.position = "BENCH"
            roster.position_players.append(slot)

    # ── Fill starting pitchers ──
    for _ in range(NUM_SP):
        slot = pick_best("SP", is_pitcher_slot=True, slot_budget=sp_budget_each)
        if slot:
            slot.position = "SP"
            roster.pitchers.append(slot)

    # ── Fill relief pitchers ──
    for _ in range(NUM_RP):
        slot = pick_best("RP", is_pitcher_slot=True, slot_budget=rp_budget_each)
        if slot:
            slot.position = "RP"
            roster.pitchers.append(slot)

    return roster


# ── Pretty Print ───────────────────────────────────────────────────────

def print_roster(roster: OptimalRoster) -> None:
    """Print the optimized roster to stdout."""
    d = roster.to_dict()
    print(f"\n{'='*65}")
    print(f"  OPTIMAL 26-MAN ROSTER  |  Budget: {d['budget_fmt']}")
    print(f"{'='*65}")

    print(f"\n  POSITION PLAYERS ({len(roster.position_players)})")
    print(f"  {'Pos':<6} {'Player':<24} {'Team':<5} {'WAR':>5} {'Salary':>10} {'$/WAR':>10}")
    print(f"  {'-'*60}")
    for s in roster.position_players:
        sal_fmt = f"${s.salary/1e6:.1f}M"
        dw = f"${s.dollar_per_war/1e6:.1f}M" if s.dollar_per_war < 1e9 else "N/A"
        print(f"  {s.position:<6} {s.name:<24} {s.team:<5} {s.projected_war:>5.1f} {sal_fmt:>10} {dw:>10}")

    print(f"\n  PITCHERS ({len(roster.pitchers)})")
    print(f"  {'Pos':<6} {'Player':<24} {'Team':<5} {'WAR':>5} {'Salary':>10} {'$/WAR':>10}")
    print(f"  {'-'*60}")
    for s in roster.pitchers:
        sal_fmt = f"${s.salary/1e6:.1f}M"
        dw = f"${s.dollar_per_war/1e6:.1f}M" if s.dollar_per_war < 1e9 else "N/A"
        print(f"  {s.position:<6} {s.name:<24} {s.team:<5} {s.projected_war:>5.1f} {sal_fmt:>10} {dw:>10}")

    print(f"\n  {'-'*40}")
    print(f"  Total projected WAR:  {d['total_war']}")
    print(f"  Total payroll:        {d['total_salary_fmt']}")
    print(f"  Remaining budget:     {d['remaining_budget_fmt']}")
    print(f"  Roster spots filled:  {d['roster_size']} / 26")
    print()


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BRAVS Roster Optimizer")
    parser.add_argument(
        "--budget", type=float, nargs="+", default=[100, 200, 300],
        help="Payroll budget(s) in millions (default: 100 200 300)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to data/ directory",
    )
    args = parser.parse_args()

    # Load data once
    data_dir = args.data_dir
    projections = load_projections(data_dir)
    salaries = load_salaries(data_dir)

    print(f"\nLoaded {len(projections)} projections, {len(salaries)} salary records")

    for budget_m in args.budget:
        budget = budget_m * 1_000_000
        roster = optimize_roster(
            budget, projections=projections, salaries=salaries
        )
        print_roster(roster)


if __name__ == "__main__":
    main()
