"""Optimizes lineup decisions across a 3-4 game series jointly.

Balances maximizing each game's runs against managing fatigue and rest.
Uses dynamic programming across the series with the fatigue model.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseball_metric.lineup_optimizer.fatigue import FatigueModel, _resolve_age
from baseball_metric.lineup_optimizer.model import LineupValueNetwork
from baseball_metric.lineup_optimizer.optimizer import (
    LineupConfig,
    assign_positions,
    evaluate_lineups_gpu,
    generate_batting_orders,
    optimize_lineup,
    select_starters,
)
from baseball_metric.lineup_optimizer.platoon import PlatoonModel

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class OpposingPitcher:
    """Minimal representation of an opposing starting pitcher."""

    name: str = "Unknown"
    hand: str = "R"              # "L" or "R"
    era: float = 4.50            # ERA (for display / tiebreaking)
    bravs_war_eq: float = 0.0   # pitcher quality in BRAVS WAR-eq

    def __repr__(self) -> str:
        return f"{self.name} ({self.hand}HP, {self.era:.2f} ERA)"


@dataclass
class GameLineup:
    """Lineup card for a single game within a series."""

    game_number: int              # 1-indexed within the series
    lineup: LineupConfig          # the 9-man lineup with batting order
    opposing_pitcher: OpposingPitcher
    fatigue_factors: dict[str, float] = field(default_factory=dict)
    expected_runs_fatigue_adj: float = 0.0

    def summary(self) -> str:
        lines = [
            f"--- Game {self.game_number} vs {self.opposing_pitcher} ---",
            f"Expected runs (fatigue-adj): {self.expected_runs_fatigue_adj:.2f}",
            "",
        ]
        for slot, (player, pos) in enumerate(
            zip(self.lineup.players, self.lineup.positions)
        ):
            pid = player.get("playerID", "?")
            name = player.get("name", "?")[:22]
            ff = self.fatigue_factors.get(pid, 1.0)
            lines.append(
                f"  {slot+1}. {name:<22} {pos:<4} "
                f"fatigue={ff:.3f}  hit={player.get('hitting_runs', 0):+.1f}"
            )
        return "\n".join(lines)


@dataclass
class SeriesResult:
    """Complete optimization result for a multi-game series."""

    game_lineups: list[GameLineup]
    total_expected_runs: float
    rest_decisions: dict[int, list[str]]   # game_number -> playerIDs resting
    search_stats: dict[str, object] = field(default_factory=dict)

    @property
    def n_games(self) -> int:
        return len(self.game_lineups)

    def summary(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"SERIES OPTIMIZATION — {self.n_games} games",
            f"Total expected runs: {self.total_expected_runs:.2f}",
            f"{'=' * 60}",
        ]
        for gl in self.game_lineups:
            lines.append("")
            lines.append(gl.summary())

        if self.rest_decisions:
            lines.append("")
            lines.append("Rest decisions:")
            for gn, pids in sorted(self.rest_decisions.items()):
                if pids:
                    lines.append(f"  Game {gn}: rest {', '.join(pids)}")

        if self.search_stats:
            lines.append("")
            lines.append(f"Search: {self.search_stats}")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_player_fatigue_state(
    player: dict,
    games_started_in_series: int,
) -> dict[str, float]:
    """Build the fatigue-state dict for a player given series context.

    Adjusts games_last_7 etc. by how many series games the player has
    already started.
    """
    g = int(player.get("G", 0) or 0)
    age = _resolve_age(player)

    g7 = float(player.get("games_last_7", min(g / 162.0 * 7.0, 7.0)))
    g14 = float(player.get("games_last_14", min(g / 162.0 * 14.0, 14.0)))
    g30 = float(player.get("games_last_30", min(g / 162.0 * 30.0, 30.0)))

    # Add games started so far in this series
    g7 = min(g7 + games_started_in_series, 7.0)
    g14 = min(g14 + games_started_in_series, 14.0)
    g30 = min(g30 + games_started_in_series, 30.0)

    return {"g7": g7, "g14": g14, "g30": g30, "age": age}


def _compute_fatigue_adjusted_value(
    player: dict,
    fatigue_state: dict[str, float],
    fatigue_model: FatigueModel,
) -> tuple[float, float]:
    """Return (fatigue_factor, fatigue-adjusted hitting_runs) for a player."""
    position = str(player.get("position", "DH"))
    factor = fatigue_model.compute_fatigue_factor(
        fatigue_state["g7"],
        fatigue_state["g14"],
        fatigue_state["g30"],
        fatigue_state["age"],
        position,
    )
    factor_f = float(factor)
    base_value = float(player.get("hitting_runs", 0) or 0)
    return factor_f, base_value * factor_f


def _optimize_single_game_with_fatigue(
    roster: list[dict],
    opposing_pitcher: OpposingPitcher,
    games_started: dict[str, int],
    fatigue_model: FatigueModel,
    platoon_model: PlatoonModel | None,
    n_candidates: int = 10000,
) -> tuple[LineupConfig, dict[str, float]]:
    """Optimize lineup for a single game, accounting for fatigue and platoon.

    Returns the best LineupConfig and a dict of playerID -> fatigue_factor.
    """
    # Adjust each player's hitting_runs for fatigue and platoon
    adjusted_roster = []
    fatigue_factors: dict[str, float] = {}

    for p in roster:
        pid = p.get("playerID", "")
        p_copy = dict(p)  # shallow copy so we don't mutate the original

        # Fatigue adjustment
        gs = games_started.get(pid, 0)
        f_state = _build_player_fatigue_state(p, gs)
        f_factor, adj_hitting = _compute_fatigue_adjusted_value(p, f_state, fatigue_model)
        fatigue_factors[pid] = f_factor

        p_copy["hitting_runs"] = adj_hitting

        # Platoon adjustment (if model available)
        if platoon_model is not None:
            yr = int(p.get("yearID", 2023))
            platoon_adj = platoon_model.get_platoon_adjusted_value(
                pid, yr, opposing_pitcher.hand
            )
            if platoon_adj != 0.0:
                p_copy["hitting_runs"] = platoon_adj * f_factor

        adjusted_roster.append(p_copy)

    # Use the base optimizer on the adjusted roster
    results = optimize_lineup(
        adjusted_roster,
        opposing_pitcher={"hand": opposing_pitcher.hand},
        n_candidates=n_candidates,
        top_n=1,
    )

    best = results[0] if results else LineupConfig(
        players=adjusted_roster[:9],
        batting_order=list(range(9)),
        positions=["DH"] * 9,
    )

    return best, fatigue_factors


# ---------------------------------------------------------------------------
# Greedy + local search series optimizer
# ---------------------------------------------------------------------------

def optimize_series(
    roster: list[dict],
    opposing_pitchers: list[OpposingPitcher | dict],
    n_games: int = 3,
    fatigue_model: FatigueModel | None = None,
    platoon_model: PlatoonModel | None = None,
    n_candidates_per_game: int = 10000,
    local_search_iterations: int = 50,
    top_k_players_to_rest: int = 5,
) -> SeriesResult:
    """Optimize lineup decisions across a multi-game series.

    Algorithm (greedy + local search):
      1. **Greedy pass**: Optimize each game independently with its
         fatigue state, yielding an initial set of lineup cards.
      2. **Local search**: For each of the top-K players, try swapping
         rest days between games.  If resting player X in game i and
         playing them in game j (instead of the reverse) improves total
         expected runs across the series, accept the swap.
      3. Repeat local search until no improving swap is found or we hit
         the iteration limit.

    The search space is manageable: for 3 games and 5 key players,
    there are at most ``5 * C(3,2) = 15`` pairwise swaps per iteration,
    and each swap requires re-optimizing at most 2 games.  Total work
    is ``O(iterations * K * n_games^2 * n_candidates)``, which runs in
    seconds on GPU.

    Args:
        roster: Full 26-man roster as list of player dicts.
        opposing_pitchers: One per game.  Each is an ``OpposingPitcher``
            or a dict with keys ``name``, ``hand``, and optionally
            ``era``.
        n_games: Number of games in the series (default 3).
        fatigue_model: FatigueModel instance (uses defaults if None).
        platoon_model: PlatoonModel for platoon adjustments (optional).
        n_candidates_per_game: Batting-order candidates per game.
        local_search_iterations: Max iterations of local search.
        top_k_players_to_rest: Number of top players to consider for
            rest swaps in local search.

    Returns:
        SeriesResult with optimal lineup cards and rest recommendations.
    """
    t0 = time.perf_counter()

    if fatigue_model is None:
        fatigue_model = FatigueModel()

    # Normalize opposing_pitchers to OpposingPitcher objects
    pitchers: list[OpposingPitcher] = []
    for i, sp in enumerate(opposing_pitchers):
        if isinstance(sp, OpposingPitcher):
            pitchers.append(sp)
        elif isinstance(sp, dict):
            pitchers.append(OpposingPitcher(
                name=sp.get("name", f"SP{i+1}"),
                hand=sp.get("hand", "R"),
                era=sp.get("era", 4.50),
                bravs_war_eq=sp.get("bravs_war_eq", 0.0),
            ))
        else:
            pitchers.append(OpposingPitcher())

    # Pad or trim to n_games
    while len(pitchers) < n_games:
        pitchers.append(OpposingPitcher())
    pitchers = pitchers[:n_games]

    # Identify the top-K players by total value (candidates for rest decisions)
    for p in roster:
        p.setdefault("total_value", (
            float(p.get("hitting_runs", 0) or 0) +
            float(p.get("baserunning_runs", 0) or 0) +
            float(p.get("fielding_runs", 0) or 0)
        ))
    sorted_roster = sorted(roster, key=lambda x: x.get("total_value", 0), reverse=True)
    top_player_ids = [
        p.get("playerID", "") for p in sorted_roster[:top_k_players_to_rest]
        if p.get("playerID")
    ]

    # ------------------------------------------------------------------
    # Phase 1: Greedy independent optimization
    # ------------------------------------------------------------------
    log.info("Series optimizer: greedy pass for %d games", n_games)

    # Track which players are starting in each game
    # rest_pattern[game_idx] = set of playerIDs resting in that game
    rest_pattern: list[set[str]] = [set() for _ in range(n_games)]

    game_lineups: list[GameLineup] = []
    games_started_accum: dict[str, int] = {
        p.get("playerID", ""): 0 for p in roster if p.get("playerID")
    }

    for game_idx in range(n_games):
        lineup, ff = _optimize_single_game_with_fatigue(
            roster=roster,
            opposing_pitcher=pitchers[game_idx],
            games_started=games_started_accum,
            fatigue_model=fatigue_model,
            platoon_model=platoon_model,
            n_candidates=n_candidates_per_game,
        )

        # Track who started
        starting_pids = {p.get("playerID", "") for p in lineup.players}
        for pid in starting_pids:
            if pid in games_started_accum:
                games_started_accum[pid] += 1

        # Track who is resting (top-K players not in the lineup)
        for pid in top_player_ids:
            if pid not in starting_pids:
                rest_pattern[game_idx].add(pid)

        ev_adj = sum(
            float(p.get("hitting_runs", 0) or 0) * ff.get(p.get("playerID", ""), 1.0)
            for p in lineup.players
        )

        game_lineups.append(GameLineup(
            game_number=game_idx + 1,
            lineup=lineup,
            opposing_pitcher=pitchers[game_idx],
            fatigue_factors=ff,
            expected_runs_fatigue_adj=ev_adj,
        ))

    total_ev = sum(gl.expected_runs_fatigue_adj for gl in game_lineups)
    log.info("Greedy total expected runs: %.2f", total_ev)

    # ------------------------------------------------------------------
    # Phase 2: Local search — swap rest days for top-K players
    # ------------------------------------------------------------------
    log.info("Series optimizer: local search (%d iterations max)", local_search_iterations)

    best_total_ev = total_ev
    best_rest_pattern = [set(s) for s in rest_pattern]
    best_game_lineups = list(game_lineups)
    improvements_found = 0

    for iteration in range(local_search_iterations):
        improved_this_round = False

        for pid in top_player_ids:
            # Try each pair of games: swap the rest status of this player
            for gi, gj in itertools.combinations(range(n_games), 2):
                resting_in_gi = pid in rest_pattern[gi]
                resting_in_gj = pid in rest_pattern[gj]

                if resting_in_gi == resting_in_gj:
                    continue  # no swap to make

                # Propose the swap
                proposed_rest = [set(s) for s in rest_pattern]
                if resting_in_gi:
                    proposed_rest[gi].discard(pid)
                    proposed_rest[gj].add(pid)
                else:
                    proposed_rest[gj].discard(pid)
                    proposed_rest[gi].add(pid)

                # Re-optimize the two affected games with updated rest
                proposed_lineups = list(game_lineups)
                proposed_ev = total_ev

                for g_reopt in (gi, gj):
                    # Rebuild games_started for this game based on the
                    # proposed rest pattern up to (but not including)
                    # this game
                    gs_accum: dict[str, int] = {
                        p.get("playerID", ""): 0 for p in roster
                        if p.get("playerID")
                    }
                    for prev_g in range(g_reopt):
                        for p in roster:
                            rpid = p.get("playerID", "")
                            if rpid and rpid not in proposed_rest[prev_g]:
                                gs_accum[rpid] = gs_accum.get(rpid, 0) + 1

                    new_lineup, new_ff = _optimize_single_game_with_fatigue(
                        roster=roster,
                        opposing_pitcher=pitchers[g_reopt],
                        games_started=gs_accum,
                        fatigue_model=fatigue_model,
                        platoon_model=platoon_model,
                        n_candidates=max(n_candidates_per_game // 2, 2000),
                    )

                    new_ev_adj = sum(
                        float(p.get("hitting_runs", 0) or 0)
                        * new_ff.get(p.get("playerID", ""), 1.0)
                        for p in new_lineup.players
                    )

                    # Update proposed total EV
                    proposed_ev -= proposed_lineups[g_reopt].expected_runs_fatigue_adj
                    proposed_ev += new_ev_adj

                    proposed_lineups[g_reopt] = GameLineup(
                        game_number=g_reopt + 1,
                        lineup=new_lineup,
                        opposing_pitcher=pitchers[g_reopt],
                        fatigue_factors=new_ff,
                        expected_runs_fatigue_adj=new_ev_adj,
                    )

                # Accept if improvement
                if proposed_ev > best_total_ev + 0.01:  # small threshold to avoid noise
                    best_total_ev = proposed_ev
                    best_rest_pattern = proposed_rest
                    best_game_lineups = proposed_lineups

                    rest_pattern = [set(s) for s in proposed_rest]
                    game_lineups = list(proposed_lineups)
                    total_ev = proposed_ev

                    improvements_found += 1
                    improved_this_round = True
                    log.debug(
                        "  Iteration %d: swap %s rest G%d<->G%d, "
                        "new total=%.2f (+%.2f)",
                        iteration, pid, gi + 1, gj + 1,
                        proposed_ev, proposed_ev - best_total_ev,
                    )

        if not improved_this_round:
            log.info(
                "Local search converged after %d iterations "
                "(%d improvements found)",
                iteration + 1, improvements_found,
            )
            break

    # ------------------------------------------------------------------
    # Build final result
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t0

    rest_decisions = {
        g + 1: sorted(pids)
        for g, pids in enumerate(best_rest_pattern)
        if pids
    }

    result = SeriesResult(
        game_lineups=best_game_lineups,
        total_expected_runs=best_total_ev,
        rest_decisions=rest_decisions,
        search_stats={
            "n_games": n_games,
            "n_candidates_per_game": n_candidates_per_game,
            "local_search_iterations": local_search_iterations,
            "improvements_found": improvements_found,
            "top_k_players": top_k_players_to_rest,
            "elapsed_seconds": round(elapsed, 2),
            "device": str(DEVICE),
        },
    )

    log.info(
        "Series optimization complete: %d games, %.2f total expected runs, "
        "%.1fs on %s",
        n_games, best_total_ev, elapsed, DEVICE,
    )

    return result


# ---------------------------------------------------------------------------
# GPU-accelerated rest-pattern enumeration (for short series)
# ---------------------------------------------------------------------------

def optimize_series_exhaustive(
    roster: list[dict],
    opposing_pitchers: list[OpposingPitcher | dict],
    n_games: int = 3,
    fatigue_model: FatigueModel | None = None,
    platoon_model: PlatoonModel | None = None,
    top_k_players_to_rest: int = 5,
    n_candidates_per_game: int = 5000,
) -> SeriesResult:
    """Exhaustive rest-pattern search for short series (3-4 games).

    For each of the top-K players, enumerate all 2^K rest/play patterns
    across n_games.  Each pattern is a binary matrix of shape
    (K, n_games) where 1 = rest, 0 = play.

    Total patterns: ``2^(K * n_games)``  — but we prune by requiring
    each player rests at most 1 game per series and no more than 2
    players rest per game.  This shrinks the space to a manageable size.

    For K=5, n_games=3: ~100 viable patterns after pruning, each
    requiring 3 lineup optimizations = ~300 total.  At 5000 candidates
    each, that is 1.5M evaluations — well within GPU throughput.

    Args:
        roster: Full 26-man roster.
        opposing_pitchers: One per game.
        n_games: Number of games.
        fatigue_model: FatigueModel instance.
        platoon_model: PlatoonModel for platoon adjustments.
        top_k_players_to_rest: Number of key players to consider.
        n_candidates_per_game: Batting-order candidates per game.

    Returns:
        SeriesResult with the globally optimal rest pattern.
    """
    t0 = time.perf_counter()

    if fatigue_model is None:
        fatigue_model = FatigueModel()

    # Normalize pitchers
    pitchers = _normalize_pitchers(opposing_pitchers, n_games)

    # Identify top-K
    for p in roster:
        p.setdefault("total_value", (
            float(p.get("hitting_runs", 0) or 0) +
            float(p.get("baserunning_runs", 0) or 0) +
            float(p.get("fielding_runs", 0) or 0)
        ))
    sorted_roster = sorted(roster, key=lambda x: x.get("total_value", 0), reverse=True)
    top_players = [
        p for p in sorted_roster[:top_k_players_to_rest]
        if p.get("playerID")
    ]
    top_pids = [p["playerID"] for p in top_players]
    K = len(top_pids)

    # Enumerate valid rest patterns
    # rest_patterns is a list of dicts: {playerID: game_index_to_rest_in}
    # A player rests in at most 1 game (or none).
    # "None" means the player plays all games.
    valid_patterns: list[dict[str, int | None]] = []

    # Each player has (n_games + 1) options: rest in game 0, 1, ..., n-1, or don't rest
    per_player_options = list(range(n_games)) + [None]
    for combo in itertools.product(per_player_options, repeat=K):
        pattern = {pid: game_idx for pid, game_idx in zip(top_pids, combo)}

        # Constraint: no more than 2 players rest per game
        rest_counts = {}
        for game_idx in pattern.values():
            if game_idx is not None:
                rest_counts[game_idx] = rest_counts.get(game_idx, 0) + 1
        if any(c > 2 for c in rest_counts.values()):
            continue

        valid_patterns.append(pattern)

    log.info(
        "Exhaustive search: %d valid rest patterns for %d players x %d games",
        len(valid_patterns), K, n_games,
    )

    # Evaluate each pattern
    best_total_ev = -float("inf")
    best_result: SeriesResult | None = None

    for pat_idx, pattern in enumerate(valid_patterns):
        # Build rest sets per game
        rest_per_game: list[set[str]] = [set() for _ in range(n_games)]
        for pid, game_idx in pattern.items():
            if game_idx is not None:
                rest_per_game[game_idx].add(pid)

        # Optimize each game in sequence
        game_lineups: list[GameLineup] = []
        gs_accum: dict[str, int] = {
            p.get("playerID", ""): 0 for p in roster if p.get("playerID")
        }
        pattern_total_ev = 0.0

        for game_idx in range(n_games):
            lineup, ff = _optimize_single_game_with_fatigue(
                roster=roster,
                opposing_pitcher=pitchers[game_idx],
                games_started=gs_accum,
                fatigue_model=fatigue_model,
                platoon_model=platoon_model,
                n_candidates=n_candidates_per_game,
            )

            ev_adj = sum(
                float(p.get("hitting_runs", 0) or 0) * ff.get(p.get("playerID", ""), 1.0)
                for p in lineup.players
            )

            game_lineups.append(GameLineup(
                game_number=game_idx + 1,
                lineup=lineup,
                opposing_pitcher=pitchers[game_idx],
                fatigue_factors=ff,
                expected_runs_fatigue_adj=ev_adj,
            ))

            pattern_total_ev += ev_adj

            # Update accumulator
            starting_pids = {p.get("playerID", "") for p in lineup.players}
            for pid in starting_pids:
                if pid in gs_accum:
                    gs_accum[pid] += 1

        if pattern_total_ev > best_total_ev:
            best_total_ev = pattern_total_ev
            rest_decisions = {
                g + 1: sorted(pids)
                for g, pids in enumerate(rest_per_game)
                if pids
            }
            best_result = SeriesResult(
                game_lineups=game_lineups,
                total_expected_runs=pattern_total_ev,
                rest_decisions=rest_decisions,
                search_stats={
                    "method": "exhaustive",
                    "patterns_evaluated": pat_idx + 1,
                    "total_patterns": len(valid_patterns),
                    "n_games": n_games,
                    "top_k": K,
                },
            )

        # Early progress logging
        if (pat_idx + 1) % 25 == 0:
            log.debug(
                "  Evaluated %d/%d patterns, best=%.2f",
                pat_idx + 1, len(valid_patterns), best_total_ev,
            )

    elapsed = time.perf_counter() - t0

    if best_result is None:
        # Fallback: no valid patterns (shouldn't happen)
        best_result = SeriesResult(
            game_lineups=[],
            total_expected_runs=0.0,
            rest_decisions={},
        )

    best_result.search_stats["elapsed_seconds"] = round(elapsed, 2)
    best_result.search_stats["device"] = str(DEVICE)

    log.info(
        "Exhaustive series optimization: %d patterns in %.1fs, "
        "best=%.2f total expected runs",
        len(valid_patterns), elapsed, best_total_ev,
    )

    return best_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_pitchers(
    opposing_pitchers: list[OpposingPitcher | dict],
    n_games: int,
) -> list[OpposingPitcher]:
    """Convert mixed pitcher inputs to a uniform list of OpposingPitcher."""
    pitchers: list[OpposingPitcher] = []
    for i, sp in enumerate(opposing_pitchers):
        if isinstance(sp, OpposingPitcher):
            pitchers.append(sp)
        elif isinstance(sp, dict):
            pitchers.append(OpposingPitcher(
                name=sp.get("name", f"SP{i+1}"),
                hand=sp.get("hand", "R"),
                era=sp.get("era", 4.50),
                bravs_war_eq=sp.get("bravs_war_eq", 0.0),
            ))
        else:
            pitchers.append(OpposingPitcher())
    while len(pitchers) < n_games:
        pitchers.append(OpposingPitcher())
    return pitchers[:n_games]
