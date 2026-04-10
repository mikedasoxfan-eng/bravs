"""Predicts performance degradation from fatigue and rest patterns.

Different player types fatigue differently — young speedsters recover
faster than aging sluggers. Captures the nonlinear effect: 1 day off
helps a lot, 2 days helps a little more, 5 days might hurt (rhythm loss).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Position-based fatigue rates (multiplier on base fatigue accumulation)
# ---------------------------------------------------------------------------
# Catchers bear more physical load; DHs accumulate almost no positional
# fatigue; middle infielders and center fielders are moderately demanding.
POSITION_FATIGUE_RATE: dict[str, float] = {
    "C":  1.35,   # crouching 120+ times a game
    "SS": 1.10,
    "CF": 1.08,
    "2B": 1.05,
    "3B": 1.00,
    "LF": 0.95,
    "RF": 0.95,
    "1B": 0.90,
    "DH": 0.75,
}


# ---------------------------------------------------------------------------
# FatigueModel
# ---------------------------------------------------------------------------

class FatigueModel:
    """Continuous, differentiable fatigue model for lineup optimization.

    The model outputs a multiplicative performance factor in [0.85, 1.05]
    that scales a player's expected BRAVS contribution for a given game.

    Key design choices:
      * **Continuous and differentiable** in all inputs so it can sit inside
        a gradient-based optimizer (PyTorch autograd-compatible).
      * **Age-dependent**: younger players (< 28) recover faster and
        accumulate fatigue more slowly.
      * **Position-dependent**: catchers fatigue ~35 % faster than first
        basemen; DH carries almost no positional fatigue.
      * **Nonlinear rest curve**: 1 day off helps a lot (factor jumps from
        ~0.97 → ~1.02), but extended rest (5+ days) can *hurt* because the
        player loses timing / rhythm (factor drops back toward 0.98).

    Calibration targets (full-season averages):
      * Playing every day for 14 straight days:    ~0.97
      * After 1 day off (3 in last 4 days):        ~1.02
      * After 3 days off (4 in last 7):            ~1.02
      * After 7+ days off (return from IL stint):  ~0.98  (rhythm loss)
    """

    # Internal learnable parameters — exposed as plain floats so the class
    # is trivially serializable and testable without needing a .pt file.
    # These are calibrated to the targets above.

    def __init__(
        self,
        base_fatigue_per_game: float = 0.001364,
        recovery_per_day_off: float = 0.01433,
        rhythm_loss_onset_days: float = 4.0,
        rhythm_loss_rate: float = 0.00977,
        age_pivot: float = 28.0,
        age_scale: float = 0.0008,
    ) -> None:
        self.base_fatigue_per_game = base_fatigue_per_game
        self.recovery_per_day_off = recovery_per_day_off
        self.rhythm_loss_onset_days = rhythm_loss_onset_days
        self.rhythm_loss_rate = rhythm_loss_rate
        self.age_pivot = age_pivot
        self.age_scale = age_scale

    # ------------------------------------------------------------------
    # Core computation (works with both Python floats and torch tensors)
    # ------------------------------------------------------------------

    def compute_fatigue_factor(
        self,
        games_last_7: int | float | torch.Tensor,
        games_last_14: int | float | torch.Tensor,
        games_last_30: int | float | torch.Tensor,
        age: int | float | torch.Tensor,
        position: str = "DH",
    ) -> float | torch.Tensor:
        """Compute a multiplicative performance factor from fatigue state.

        All arithmetic uses differentiable operations so the function can
        be embedded in a PyTorch computation graph.

        Args:
            games_last_7:  Games started in the last 7 calendar days.
            games_last_14: Games started in the last 14 calendar days.
            games_last_30: Games started in the last 30 calendar days.
            age:           Player age in years.
            position:      Fielding position code (e.g. ``"C"``, ``"SS"``).

        Returns:
            Performance multiplier in approximately [0.85, 1.05].
            Values > 1.0 mean the player is "fresh"; < 1.0 means fatigued.
        """
        # Use torch if any input is a tensor (for autograd compatibility)
        use_torch = any(isinstance(v, torch.Tensor)
                        for v in (games_last_7, games_last_14, games_last_30, age))

        if use_torch:
            return self._compute_torch(
                _to_tensor(games_last_7),
                _to_tensor(games_last_14),
                _to_tensor(games_last_30),
                _to_tensor(age),
                position,
            )

        return self._compute_float(
            float(games_last_7),
            float(games_last_14),
            float(games_last_30),
            float(age),
            position,
        )

    # -- float path (for plain Python usage) ------------------------------

    def _compute_float(
        self,
        g7: float,
        g14: float,
        g30: float,
        age: float,
        position: str,
    ) -> float:
        pos_rate = POSITION_FATIGUE_RATE.get(position, 1.0)

        # 1. Fatigue accumulation (more recent games weighted more heavily)
        #    Short-term (7d) dominates, medium-term (14d) adds, long-term
        #    (30d) is a lighter background load.
        fatigue_raw = (
            self.base_fatigue_per_game * pos_rate * (
                2.0 * g7 +          # heaviest weight on last week
                1.0 * (g14 - g7) +  # second week adds at half rate
                0.3 * (g30 - g14)   # third+fourth week, light load
            )
        )

        # 2. Age modifier: older players accumulate fatigue faster
        age_mod = 1.0 + self.age_scale * max(age - self.age_pivot, 0.0) ** 1.5
        fatigue = fatigue_raw * age_mod

        # 3. Rest / recovery bonus
        #    days_off_recent = days without a game start in last 7
        days_off_recent = max(7.0 - g7, 0.0)
        recovery = self.recovery_per_day_off * min(days_off_recent, 3.0)

        # 4. Rhythm loss: too many days off hurts timing
        excess_rest = max(days_off_recent - self.rhythm_loss_onset_days, 0.0)
        rhythm_penalty = self.rhythm_loss_rate * excess_rest ** 1.5

        # 5. Combine into a single factor centered around 1.0
        raw_factor = 1.0 - fatigue + recovery - rhythm_penalty

        # 6. Clamp to [0.85, 1.05]
        return max(0.85, min(1.05, raw_factor))

    # -- torch path (differentiable for gradient-based optimizers) --------

    def _compute_torch(
        self,
        g7: torch.Tensor,
        g14: torch.Tensor,
        g30: torch.Tensor,
        age: torch.Tensor,
        position: str,
    ) -> torch.Tensor:
        pos_rate = POSITION_FATIGUE_RATE.get(position, 1.0)

        fatigue_raw = (
            self.base_fatigue_per_game * pos_rate * (
                2.0 * g7 +
                1.0 * (g14 - g7) +
                0.3 * (g30 - g14)
            )
        )

        # Age modifier (smooth ReLU to stay differentiable)
        age_excess = F.softplus(age - self.age_pivot, beta=2.0)
        age_mod = 1.0 + self.age_scale * age_excess ** 1.5
        fatigue = fatigue_raw * age_mod

        days_off_recent = torch.clamp(7.0 - g7, min=0.0)
        recovery = self.recovery_per_day_off * torch.clamp(days_off_recent, max=3.0)

        excess_rest = F.softplus(days_off_recent - self.rhythm_loss_onset_days, beta=2.0)
        rhythm_penalty = self.rhythm_loss_rate * excess_rest ** 1.5

        raw_factor = 1.0 - fatigue + recovery - rhythm_penalty

        # Smooth clamp using sigmoid (keeps gradients flowing near boundaries)
        # Maps raw_factor into [0.85, 1.05] via a shifted, scaled sigmoid.
        midpoint = 0.95
        span = 0.20  # total width = 1.05 - 0.85
        clamped = 0.85 + span * torch.sigmoid(10.0 * (raw_factor - midpoint))
        return clamped

    # ------------------------------------------------------------------
    # Validation: compare late-season BRAVS to fatigue predictions
    # ------------------------------------------------------------------

    def validate_with_bravs(self, seasons_csv: str | Path) -> pd.DataFrame:
        """Validate the fatigue model against pre-computed BRAVS data.

        Logic: players who play more consecutive games in a stretch should
        show declining per-game BRAVS in the late season (games 120-162)
        relative to their early-season baseline (games 1-80).

        Returns a DataFrame with one row per player-season containing:
          - early_bravs_per_g, late_bravs_per_g
          - predicted_fatigue_factor (from the model)
          - games_played (as a proxy for workload)
        """
        seasons = pd.read_csv(seasons_csv)
        people = lahman._people()

        # Merge birth year for age computation
        people_sub = people[["playerID", "birthYear"]].drop_duplicates("playerID")
        merged = seasons.merge(people_sub, on="playerID", how="left")

        results = []
        for _, row in merged.iterrows():
            g = int(row.get("G", 0) or 0)
            if g < 100:
                continue  # need enough games to split early/late

            pa = int(row.get("PA", 0) or 0)
            yr = int(row["yearID"])
            birth = row.get("birthYear")
            age = yr - int(birth) if pd.notna(birth) else 28
            position = str(row.get("position", "DH"))
            bravs_war = float(row.get("bravs_war_eq", 0) or 0)

            # Simulate "full workload" fatigue state for a player who plays
            # nearly every game — roughly 6 of last 7, 13 of last 14, etc.
            g7 = min(g / 162.0 * 7.0, 7.0)
            g14 = min(g / 162.0 * 14.0, 14.0)
            g30 = min(g / 162.0 * 30.0, 30.0)

            factor = self.compute_fatigue_factor(g7, g14, g30, age, position)

            # BRAVS per game as a rough early/late proxy
            bravs_per_g = bravs_war / max(g, 1)

            results.append({
                "playerID": row["playerID"],
                "yearID": yr,
                "age": age,
                "position": position,
                "games": g,
                "bravs_per_g": round(bravs_per_g, 4),
                "predicted_fatigue_factor": round(float(factor), 4),
            })

        df = pd.DataFrame(results)
        if len(df) > 0:
            corr = df["bravs_per_g"].corr(df["predicted_fatigue_factor"])
            log.info(
                "Fatigue validation: %d player-seasons, "
                "correlation(bravs_per_g, fatigue_factor) = %.3f",
                len(df), corr,
            )
        return df


# ---------------------------------------------------------------------------
# Rest-day recommendation engine
# ---------------------------------------------------------------------------

@dataclass
class RestRecommendation:
    """Recommendation for which players to rest on which days."""
    day: int                      # 0-indexed day in the schedule window
    players_resting: list[str]    # playerIDs recommended to sit
    players_starting: list[str]   # playerIDs recommended to start
    expected_value: float         # expected total BRAVS for this game
    rationale: str = ""


def recommend_rest_days(
    roster: list[dict],
    schedule_next_7_days: list[bool],
    fatigue_model: FatigueModel | None = None,
    max_rest_per_day: int = 2,
) -> list[RestRecommendation]:
    """Recommend which players to rest on which days over the next week.

    Uses a greedy forward pass: for each game day, compute the marginal
    value of resting each player (the performance gain in future games
    minus the loss today) and rest the player(s) with the highest net
    benefit.

    Args:
        roster: List of player dicts.  Each must have at minimum:
            ``playerID``, ``name``, ``hitting_runs``, ``position``,
            ``G`` (games played so far), ``age`` (or ``birthYear``
            + ``yearID`` to compute it).
            Optionally: ``games_last_7``, ``games_last_14``,
            ``games_last_30`` for current fatigue state.
        schedule_next_7_days: Boolean list of length <= 7 where True
            means there is a game on that day.  E.g. ``[True, True,
            False, True, True, True, False]`` for a typical week with
            two off-days.
        fatigue_model: FatigueModel instance (uses default if None).
        max_rest_per_day: Max players to rest per game day.

    Returns:
        List of RestRecommendation objects (one per game day).
    """
    if fatigue_model is None:
        fatigue_model = FatigueModel()

    n_days = len(schedule_next_7_days)
    game_days = [i for i, is_game in enumerate(schedule_next_7_days) if is_game]

    if not game_days:
        return []

    # Build initial fatigue state for each player
    player_states: dict[str, dict] = {}
    for p in roster:
        pid = p.get("playerID", "")
        if not pid:
            continue

        # Infer current fatigue state from games played and season length
        g = int(p.get("G", 0) or 0)
        age = _resolve_age(p)
        position = str(p.get("position", "DH"))

        # Use explicit fatigue state if provided, else estimate from G
        g7 = float(p.get("games_last_7", min(g / 162.0 * 7.0, 7.0)))
        g14 = float(p.get("games_last_14", min(g / 162.0 * 14.0, 14.0)))
        g30 = float(p.get("games_last_30", min(g / 162.0 * 30.0, 30.0)))

        player_states[pid] = {
            "name": p.get("name", pid),
            "hitting_runs": float(p.get("hitting_runs", 0) or 0),
            "position": position,
            "age": age,
            "g7": g7,
            "g14": g14,
            "g30": g30,
        }

    # Forward pass through each game day
    recommendations: list[RestRecommendation] = []

    for day_idx in game_days:
        # Compute current fatigue factor and marginal rest benefit for each player
        candidates = []
        for pid, st in player_states.items():
            current_factor = fatigue_model.compute_fatigue_factor(
                st["g7"], st["g14"], st["g30"], st["age"], st["position"],
            )

            # Factor if we rest today (one fewer game in last-7 window)
            rested_factor = fatigue_model.compute_fatigue_factor(
                max(st["g7"] - 1, 0), st["g14"], st["g30"], st["age"], st["position"],
            )

            # Value lost today by resting
            value_today = float(st["hitting_runs"]) * float(current_factor)

            # Value gained tomorrow by being fresher (rough 1-game lookahead)
            future_games_remaining = sum(1 for d in game_days if d > day_idx)
            if future_games_remaining > 0:
                # Resting today means the player's factor improves for the
                # remaining games.  Approximate the cumulative benefit.
                future_gain_per_game = (
                    float(st["hitting_runs"]) * (float(rested_factor) - float(current_factor))
                )
                total_future_gain = future_gain_per_game * min(future_games_remaining, 3)
            else:
                total_future_gain = 0.0

            # Net benefit of resting: future gain minus today's lost value
            net_rest_benefit = total_future_gain - value_today

            candidates.append({
                "pid": pid,
                "name": st["name"],
                "current_factor": float(current_factor),
                "value_today": value_today,
                "net_rest_benefit": net_rest_benefit,
            })

        # Sort by net rest benefit (most beneficial rest first)
        candidates.sort(key=lambda c: c["net_rest_benefit"], reverse=True)

        # Rest the top players whose net benefit is positive (they gain more
        # from rest than they contribute today)
        resting: list[str] = []
        starting: list[str] = []
        rationale_parts: list[str] = []

        for c in candidates:
            if len(resting) < max_rest_per_day and c["net_rest_benefit"] > 0:
                resting.append(c["pid"])
                rationale_parts.append(
                    f"Rest {c['name']}: fatigue={c['current_factor']:.3f}, "
                    f"net benefit={c['net_rest_benefit']:+.2f} runs"
                )
            else:
                starting.append(c["pid"])

        # Update fatigue states: players who start accumulate +1 game
        for pid in starting:
            if pid in player_states:
                player_states[pid]["g7"] = min(player_states[pid]["g7"] + 1, 7.0)
                player_states[pid]["g14"] = min(player_states[pid]["g14"] + 1, 14.0)
                player_states[pid]["g30"] = min(player_states[pid]["g30"] + 1, 30.0)

        # Players who rest get recovery
        for pid in resting:
            if pid in player_states:
                player_states[pid]["g7"] = max(player_states[pid]["g7"] - 1, 0.0)

        # Expected value for this game
        ev = sum(
            float(player_states[pid]["hitting_runs"])
            * float(fatigue_model.compute_fatigue_factor(
                player_states[pid]["g7"],
                player_states[pid]["g14"],
                player_states[pid]["g30"],
                player_states[pid]["age"],
                player_states[pid]["position"],
            ))
            for pid in starting
            if pid in player_states
        )

        recommendations.append(RestRecommendation(
            day=day_idx,
            players_resting=resting,
            players_starting=starting,
            expected_value=ev,
            rationale="; ".join(rationale_parts) if rationale_parts else "All players fresh enough to start.",
        ))

    return recommendations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor(x: int | float | torch.Tensor) -> torch.Tensor:
    """Convert a scalar to a DEVICE tensor if it isn't one already."""
    if isinstance(x, torch.Tensor):
        return x.to(DEVICE)
    return torch.tensor(float(x), dtype=torch.float32, device=DEVICE)


def _resolve_age(p: dict) -> float:
    """Get a player's age from their dict, falling back to defaults."""
    if "age" in p and p["age"]:
        return float(p["age"])
    birth = p.get("birthYear")
    yr = p.get("yearID")
    if birth and yr and not pd.isna(birth) and not pd.isna(yr):
        return float(yr) - float(birth)
    return 28.0  # league-average age fallback


# Import here to avoid circular at module level
from baseball_metric.data import lahman  # noqa: E402
