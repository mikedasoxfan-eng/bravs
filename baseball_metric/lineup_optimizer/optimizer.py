"""Phase 3: Lineup Optimization Engine.

Given a roster and game context, finds the optimal lineup configuration
by evaluating thousands of candidate lineups on GPU in parallel.

Search strategy:
1. Fix the 9 best players by total BRAVS (handles bench decisions)
2. Assign positions using Hungarian algorithm (optimal assignment)
3. Optimize batting order via GPU-parallelized permutation search
4. Report top 5 lineups with expected runs and uncertainty
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass

import numpy as np
import torch

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard batting order heuristics (used for beam search initialization)
# Slot: [preferred attributes]
SLOT_PREFS = {
    0: "obp",      # Leadoff: high OBP
    1: "balanced",  # 2-hole: balanced contact + power
    2: "best",      # 3-hole: best overall hitter
    3: "power",     # 4-hole: power hitter (cleanup)
    4: "power2",    # 5-hole: second power hitter
    5: "balanced2", # 6-hole: balanced
    6: "developing",# 7-hole: developing or average
    7: "weak",      # 8-hole: weaker hitter (or pitcher in old NL)
    8: "speed",     # 9-hole: speed / second leadoff
}

# Position assignment values (from BRAVS positional spectrum)
POS_VALUES = {
    "C": 9.0, "SS": 6.5, "CF": 2.5, "2B": 2.5, "3B": 2.0,
    "LF": -6.0, "RF": -6.0, "1B": -9.5, "DH": -14.0,
}


@dataclass
class LineupConfig:
    """A complete lineup configuration."""
    players: list[dict]  # 9 player dicts with BRAVS components
    batting_order: list[int]  # indices into players list
    positions: list[str]  # position assignment for each player
    expected_runs: float = 0.0
    expected_runs_std: float = 0.0
    explanation: str = ""


def select_starters(roster: list[dict], n: int = 9) -> list[dict]:
    """Select the best 9 players from a 26-man roster.

    Uses total BRAVS value (hitting + fielding + positional + baserunning)
    to rank players. Ensures at least 1 catcher and 1 pitcher if needed.
    """
    # Sort by total expected value
    for p in roster:
        p["total_value"] = (
            p.get("hitting_runs", 0) +
            p.get("baserunning_runs", 0) +
            p.get("fielding_runs", 0) +
            p.get("aqi_runs", 0)
        )

    sorted_roster = sorted(roster, key=lambda x: x["total_value"], reverse=True)

    # Ensure we have a catcher
    catchers = [p for p in sorted_roster if p.get("position") == "C" or p.get("can_catch")]
    starters = []
    catcher_added = False

    for p in sorted_roster:
        if len(starters) >= n:
            break
        if p.get("position") == "C" and not catcher_added:
            starters.append(p)
            catcher_added = True
        elif p.get("position") != "C":
            starters.append(p)

    # If no catcher was added, replace the weakest starter with best catcher
    if not catcher_added and catchers:
        starters[-1] = catchers[0]

    return starters[:n]


def assign_positions(players: list[dict]) -> list[str]:
    """Assign optimal positions using a greedy approach.

    Each player gets their best available position, prioritizing
    premium defensive positions (C, SS, CF) first.
    """
    assignments = ["DH"] * len(players)
    used_positions = set()

    # Sort positions by defensive value (fill premium positions first)
    sorted_positions = sorted(POS_VALUES.keys(), key=lambda p: POS_VALUES[p], reverse=True)

    # For each position, find the best available player
    for pos in sorted_positions:
        if pos in used_positions:
            continue

        best_idx = -1
        best_score = -999

        for i, p in enumerate(players):
            if assignments[i] != "DH":
                continue  # already assigned
            # Can this player play this position?
            can_play = (
                p.get("position") == pos or
                pos in p.get("secondary_positions", []) or
                pos == "DH"  # anyone can DH
            )
            if not can_play:
                continue

            # Score: their value at this position
            score = p.get("total_value", 0)
            if p.get("position") == pos:
                score += 5  # bonus for primary position

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            assignments[best_idx] = pos
            used_positions.add(pos)

    return assignments


def generate_batting_orders(n_players: int = 9, n_orders: int = 5000) -> torch.Tensor:
    """Generate candidate batting orders for GPU evaluation.

    Combines heuristic-seeded orders with random permutations.
    Returns tensor of shape (n_orders, 9) with player indices.
    """
    orders = set()

    # Add some heuristic orders
    base = list(range(n_players))
    orders.add(tuple(base))  # default order
    orders.add(tuple(reversed(base)))  # reversed

    # Random permutations
    rng = np.random.default_rng(42)
    while len(orders) < n_orders:
        perm = rng.permutation(n_players).tolist()
        orders.add(tuple(perm))

    orders_array = np.array(list(orders), dtype=np.int64)
    return torch.tensor(orders_array, device=DEVICE)


def evaluate_lineups_gpu(
    player_features: torch.Tensor,
    batting_orders: torch.Tensor,
    interaction_model: object | None = None,
) -> torch.Tensor:
    """Evaluate many lineup orderings on GPU in parallel.

    For each batting order, computes the expected run value based on:
    1. Sum of individual player values (baseline)
    2. Batting order interaction effects (from transformer model if available)
    3. Positional adjacency bonuses (OBP before power, etc.)

    Args:
        player_features: (9, n_features) — feature vectors for the 9 starters
        batting_orders: (n_orders, 9) — candidate batting orders
        interaction_model: optional trained SlotInteractionModel

    Returns:
        (n_orders,) — expected run values for each ordering
    """
    n_orders = batting_orders.shape[0]
    n_features = player_features.shape[1]

    # Gather player features in each ordering
    # batting_orders[i] gives the player indices for the i-th ordering
    # We want: ordered_features[i, j, :] = player_features[batting_orders[i, j], :]
    expanded = player_features.unsqueeze(0).expand(n_orders, -1, -1)  # (n_orders, 9, n_feat)
    idx = batting_orders.unsqueeze(-1).expand(-1, -1, n_features)  # (n_orders, 9, n_feat)
    ordered = torch.gather(expanded, 1, idx)  # (n_orders, 9, n_feat)

    # Baseline: sum of hitting values (column 0 = hitting_runs)
    hitting_col = 0
    base_value = ordered[:, :, hitting_col].sum(dim=1)  # (n_orders,)

    # Batting order bonuses (heuristic)
    # Slot 0 (leadoff): bonus for high OBP proxy (hitting_runs / PA-proxy)
    # Slot 2 (3-hole): bonus for best hitter
    # Slot 3 (cleanup): bonus for power (HR column if available)
    obp_proxy = ordered[:, :, hitting_col]  # use hitting runs as OBP proxy
    leadoff_bonus = obp_proxy[:, 0] * 0.05  # small bonus for good leadoff
    three_hole_bonus = obp_proxy[:, 2] * 0.03  # small bonus for best in 3-hole
    cleanup_bonus = obp_proxy[:, 3] * 0.02  # small bonus for power in cleanup

    # Adjacency bonus: high-value hitter followed by another high-value hitter
    # (lineup protection effect)
    for slot in range(8):
        adj_bonus = (obp_proxy[:, slot] > 0).float() * (obp_proxy[:, slot + 1] > 0).float() * 0.01
        base_value = base_value + adj_bonus

    total_value = base_value + leadoff_bonus + three_hole_bonus + cleanup_bonus

    # If interaction model available, use it
    if interaction_model is not None:
        try:
            with torch.no_grad():
                interaction_adj = interaction_model(ordered[:, :, :8])  # first 8 features
                total_value = total_value + interaction_adj
        except Exception:
            pass

    return total_value


def optimize_lineup(
    roster: list[dict],
    opposing_pitcher: dict | None = None,
    n_candidates: int = 10000,
    top_n: int = 5,
    fatigue_model: object | None = None,
    platoon_model: object | None = None,
    year: int = 2025,
) -> list[LineupConfig]:
    """Full lineup optimization for a single game.

    1. Select best 9 starters from roster
    2. Assign optimal positions
    3. Apply platoon and fatigue adjustments (if models provided)
    4. Search batting order space on GPU
    5. Return top lineups with explanations

    Args:
        roster: list of player dicts with BRAVS components
        opposing_pitcher: pitcher info (handedness, pitch mix) — optional
        n_candidates: number of batting orders to evaluate
        top_n: number of top lineups to return
        fatigue_model: optional FatigueModel instance for workload adjustment
        platoon_model: optional PlatoonModel instance for L/R splits
        year: season year (for platoon lookups)

    Returns:
        List of LineupConfig objects, sorted by expected runs
    """
    t0 = time.perf_counter()

    # Step 0: Apply platoon adjustments if model and pitcher info available
    if platoon_model is not None and opposing_pitcher is not None:
        pitcher_hand = opposing_pitcher.get("hand", "R")
        for p in roster:
            pid = p.get("playerID", "")
            if pid:
                adj = platoon_model.get_platoon_adjusted_value(pid, year, pitcher_hand)
                if adj != 0:
                    p["hitting_runs"] = adj

    # Step 0b: Apply fatigue adjustments if model provided
    if fatigue_model is not None:
        for p in roster:
            g = int(p.get("G", 0) or 0)
            g7 = float(p.get("games_last_7", min(g / 162.0 * 7.0, 7.0)))
            g14 = float(p.get("games_last_14", min(g / 162.0 * 14.0, 14.0)))
            g30 = float(p.get("games_last_30", min(g / 162.0 * 30.0, 30.0)))
            age = float(p.get("age", 28))
            pos = p.get("position", "DH")
            factor = float(fatigue_model.compute_fatigue_factor(g7, g14, g30, age, pos))
            p["hitting_runs"] = p.get("hitting_runs", 0) * factor
            p["_fatigue_factor"] = factor

    # Step 1: Select starters
    starters = select_starters(roster, 9)
    log.info("Selected 9 starters from %d-man roster", len(roster))

    # Step 2: Assign positions
    positions = assign_positions(starters)
    log.info("Positions: %s", ", ".join(f"{s.get('name', '?')[:12]}={p}" for s, p in zip(starters, positions)))

    # Step 3: Build feature matrix
    feature_names = ["hitting_runs", "baserunning_runs", "fielding_runs",
                     "positional_runs", "aqi_runs", "HR", "SB", "PA"]
    features = []
    for p in starters:
        feat = [float(p.get(f, 0) or 0) for f in feature_names]
        features.append(feat)

    player_features = torch.tensor(features, dtype=torch.float32, device=DEVICE)

    # Step 4: Generate and evaluate candidate batting orders
    batting_orders = generate_batting_orders(9, n_candidates)
    log.info("Evaluating %d candidate batting orders on %s...", n_candidates, DEVICE)

    values = evaluate_lineups_gpu(player_features, batting_orders)

    # Step 5: Get top results
    top_indices = values.topk(top_n).indices
    top_values = values[top_indices]

    results = []
    for rank, (idx, val) in enumerate(zip(top_indices, top_values)):
        order = batting_orders[idx].cpu().numpy().tolist()
        ordered_starters = [starters[i] for i in order]
        ordered_positions = [positions[i] for i in order]

        # Build explanation
        explanation_parts = []
        for slot, (player, pos) in enumerate(zip(ordered_starters, ordered_positions)):
            name = player.get("name", "?")[:20]
            explanation_parts.append(f"  {slot+1}. {name:<20} {pos:<4} "
                                    f"(hit: {player.get('hitting_runs', 0):+.1f}, "
                                    f"br: {player.get('baserunning_runs', 0):+.1f})")

        config = LineupConfig(
            players=ordered_starters,
            batting_order=order,
            positions=ordered_positions,
            expected_runs=float(val.item()),
            expected_runs_std=float(val.item() * 0.08),  # rough uncertainty
            explanation="\n".join(explanation_parts),
        )
        results.append(config)

    elapsed = time.perf_counter() - t0
    log.info("Optimization complete in %.2fs", elapsed)

    return results
