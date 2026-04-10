"""Trade Impact Simulator — quantifies how acquisitions change optimal lineups.

"If we acquire Player X, how does our optimal lineup change?"

Computes:
- Marginal lineup value of adding/removing a player
- Positional surplus at each spot
- Roster fit effects (a 5-WAR SS is worth more to a team with 0-WAR SS)
"""

from __future__ import annotations

import logging
from copy import deepcopy

from baseball_metric.lineup_optimizer.optimizer import optimize_lineup, select_starters
from baseball_metric.lineup_optimizer.season_optimizer import compute_positional_surplus

log = logging.getLogger(__name__)


def simulate_trade(
    current_roster: list[dict],
    players_out: list[dict],
    players_in: list[dict],
) -> dict:
    """Simulate a trade's impact on lineup optimization.

    Args:
        current_roster: current team roster
        players_out: players being traded away
        players_in: players being acquired

    Returns:
        dict with before/after lineup values, marginal impact, explanation
    """
    # Before trade
    before_lineup = optimize_lineup(current_roster, n_candidates=20000, top_n=1)
    before_value = before_lineup[0].expected_runs if before_lineup else 0

    before_surplus = compute_positional_surplus(current_roster)

    # After trade: remove outgoing, add incoming
    out_ids = {p.get("playerID") for p in players_out}
    after_roster = [p for p in current_roster if p.get("playerID") not in out_ids]
    after_roster.extend(players_in)

    after_lineup = optimize_lineup(after_roster, n_candidates=20000, top_n=1)
    after_value = after_lineup[0].expected_runs if after_lineup else 0

    after_surplus = compute_positional_surplus(after_roster)

    # Marginal value
    marginal = after_value - before_value

    # Positional impact
    pos_changes = {}
    for pos in set(list(before_surplus.keys()) + list(after_surplus.keys())):
        before_s = before_surplus.get(pos, 0)
        after_s = after_surplus.get(pos, 0)
        if abs(after_s - before_s) > 0.5:
            pos_changes[pos] = round(after_s - before_s, 1)

    # Build explanation
    out_names = [p.get("name", "?") for p in players_out]
    in_names = [p.get("name", "?") for p in players_in]

    explanation = [
        f"Trade: {', '.join(out_names)} for {', '.join(in_names)}",
        f"Before: lineup value = {before_value:.1f}",
        f"After:  lineup value = {after_value:.1f}",
        f"Marginal impact: {marginal:+.1f} run-value units",
    ]
    if pos_changes:
        explanation.append("Positional surplus changes:")
        for pos, change in sorted(pos_changes.items(), key=lambda x: -abs(x[1])):
            explanation.append(f"  {pos}: {change:+.1f}")

    return {
        "before_value": round(before_value, 1),
        "after_value": round(after_value, 1),
        "marginal_impact": round(marginal, 1),
        "before_lineup": before_lineup[0] if before_lineup else None,
        "after_lineup": after_lineup[0] if after_lineup else None,
        "positional_changes": pos_changes,
        "explanation": "\n".join(explanation),
    }


def find_biggest_upgrade_positions(roster: list[dict]) -> list[tuple[str, float]]:
    """Find which positions would benefit most from an upgrade.

    Returns list of (position, marginal_win_value) sorted by impact.
    """
    surplus = compute_positional_surplus(roster)

    # Positions with negative surplus = biggest upgrade opportunities
    upgrades = [(pos, -val) for pos, val in surplus.items() if val < 0]
    upgrades.sort(key=lambda x: x[1], reverse=True)

    return upgrades


def compute_player_marginal_value(roster: list[dict], player: dict) -> float:
    """Compute the marginal value of adding a player to a roster.

    This captures roster fit: a player is worth more to a team that
    needs their position than to a team that already has a star there.
    """
    # Value without the player
    without = optimize_lineup(roster, n_candidates=10000, top_n=1)
    val_without = without[0].expected_runs if without else 0

    # Value with the player
    with_player = roster + [player]
    with_result = optimize_lineup(with_player, n_candidates=10000, top_n=1)
    val_with = with_result[0].expected_runs if with_result else 0

    return round(val_with - val_without, 1)
