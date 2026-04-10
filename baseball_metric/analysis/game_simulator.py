"""Monte Carlo Game Simulator — simulate full baseball games using BRAVS.

Takes two team rosters with BRAVS components and simulates a game
inning-by-inning using probabilistic models for each plate appearance.

Outputs: win probability, expected runs, run distribution,
key matchup advantages.
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class SimPlayer:
    """A player in the simulation."""
    name: str = "Unknown"
    position: str = "DH"
    obp: float = 0.320        # on-base probability per PA
    slg: float = 0.400        # slugging (determines extra-base hit probability)
    hr_rate: float = 0.030    # HR probability per PA
    bb_rate: float = 0.080    # walk rate
    k_rate: float = 0.200     # strikeout rate
    speed: float = 0.5        # 0-1 scale, affects stolen bases and advancing on hits


@dataclass
class SimResult:
    """Result of a game simulation."""
    home_wins: int = 0
    away_wins: int = 0
    n_sims: int = 0
    home_win_pct: float = 0.0
    avg_home_runs: float = 0.0
    avg_away_runs: float = 0.0
    home_run_dist: list = field(default_factory=list)
    away_run_dist: list = field(default_factory=list)
    summary: str = ""


def build_sim_player_from_bravs(player_dict: dict) -> SimPlayer:
    """Convert a BRAVS player dict into a SimPlayer."""
    pa = max(float(player_dict.get("PA", 400) or 400), 1)
    hr = float(player_dict.get("HR", 0) or 0)
    sb = float(player_dict.get("SB", 0) or 0)
    hitting_runs = float(player_dict.get("hitting_runs", 0) or 0)

    # Estimate OBP from hitting_runs
    # hitting_runs = (wOBA - 0.315) / 1.25 * PA
    # wOBA ~ OBP * 1.15 roughly
    woba_est = 0.315 + hitting_runs * 1.25 / pa
    obp_est = max(0.200, min(0.500, woba_est / 1.15))

    hr_rate = hr / pa
    bb_rate = max(0.05, obp_est - 0.250)  # rough
    k_rate = max(0.10, 0.35 - obp_est)     # inverse of OBP roughly
    slg_est = 0.350 + hr_rate * 15 + hitting_runs / pa * 2

    speed = min(1.0, sb / pa * 10) if pa > 0 else 0.3

    return SimPlayer(
        name=player_dict.get("name", "Unknown"),
        position=player_dict.get("position", "DH"),
        obp=obp_est,
        slg=slg_est,
        hr_rate=hr_rate,
        bb_rate=bb_rate,
        k_rate=k_rate,
        speed=speed,
    )


def simulate_plate_appearance(batter: SimPlayer, rng: np.random.Generator) -> str:
    """Simulate a single plate appearance. Returns outcome string."""
    roll = rng.random()

    if roll < batter.hr_rate:
        return "HR"
    elif roll < batter.hr_rate + batter.bb_rate:
        return "BB"
    elif roll < batter.hr_rate + batter.bb_rate + batter.k_rate:
        return "K"
    elif roll < batter.obp:
        # Hit (not HR) — determine type
        hit_roll = rng.random()
        if hit_roll < 0.60:
            return "1B"
        elif hit_roll < 0.82:
            return "2B"
        elif hit_roll < 0.86:
            return "3B"
        else:
            return "1B"  # extra singles
    else:
        return "OUT"


def simulate_half_inning(lineup: list[SimPlayer], lineup_idx: int,
                         rng: np.random.Generator) -> tuple[int, int]:
    """Simulate a half-inning. Returns (runs_scored, new_lineup_idx)."""
    outs = 0
    runs = 0
    bases = [False, False, False]  # 1st, 2nd, 3rd
    idx = lineup_idx

    while outs < 3:
        batter = lineup[idx % len(lineup)]
        outcome = simulate_plate_appearance(batter, rng)
        idx += 1

        if outcome == "HR":
            runs += 1 + sum(bases)
            bases = [False, False, False]
        elif outcome == "3B":
            runs += sum(bases)
            bases = [False, False, True]
        elif outcome == "2B":
            runs += bases[1] + bases[2]
            bases = [False, True, False]
            if bases[0]:
                runs += 1  # runner from 1st scores on double sometimes
        elif outcome == "1B":
            if bases[2]:
                runs += 1
                bases[2] = False
            if bases[1]:
                # Runner on 2nd scores ~60% of the time on a single
                if rng.random() < 0.60:
                    runs += 1
                else:
                    bases[2] = True
                bases[1] = False
            if bases[0]:
                bases[1] = True
            bases[0] = True
        elif outcome == "BB":
            if bases[0] and bases[1] and bases[2]:
                runs += 1  # bases loaded walk
            elif bases[0] and bases[1]:
                bases[2] = True
            elif bases[0]:
                bases[1] = True
            bases[0] = True
        elif outcome == "K":
            outs += 1
        else:  # OUT
            outs += 1
            # Sacrifice fly possibility
            if outs < 3 and bases[2] and rng.random() < 0.30:
                runs += 1
                bases[2] = False

    return runs, idx


def simulate_game(home_lineup: list[SimPlayer], away_lineup: list[SimPlayer],
                  rng: np.random.Generator) -> tuple[int, int]:
    """Simulate a full 9-inning game. Returns (home_runs, away_runs)."""
    home_runs = 0
    away_runs = 0
    home_idx = 0
    away_idx = 0

    for inning in range(9):
        # Top of inning (away bats)
        r, away_idx = simulate_half_inning(away_lineup, away_idx, rng)
        away_runs += r

        # Bottom of inning (home bats)
        # If bottom 9 and home is ahead, game over
        if inning == 8 and home_runs > away_runs:
            break

        r, home_idx = simulate_half_inning(home_lineup, home_idx, rng)
        home_runs += r

    # Extra innings if tied
    while home_runs == away_runs:
        r, away_idx = simulate_half_inning(away_lineup, away_idx, rng)
        away_runs += r
        r, home_idx = simulate_half_inning(home_lineup, home_idx, rng)
        home_runs += r

    return home_runs, away_runs


def simulate_matchup(
    home_roster: list[dict],
    away_roster: list[dict],
    n_sims: int = 10000,
    seed: int = 42,
) -> SimResult:
    """Simulate a full matchup between two teams.

    Args:
        home_roster: list of player dicts with BRAVS components
        away_roster: list of player dicts with BRAVS components
        n_sims: number of simulations

    Returns:
        SimResult with win probabilities and run distributions
    """
    rng = np.random.default_rng(seed)

    # Build lineups (top 9 by hitting value)
    def build_lineup(roster):
        sorted_r = sorted(roster, key=lambda x: x.get("hitting_runs", 0), reverse=True)
        return [build_sim_player_from_bravs(p) for p in sorted_r[:9]]

    home_lineup = build_lineup(home_roster)
    away_lineup = build_lineup(away_roster)

    home_wins = 0
    away_wins = 0
    home_runs_list = []
    away_runs_list = []

    for _ in range(n_sims):
        hr, ar = simulate_game(home_lineup, away_lineup, rng)
        home_runs_list.append(hr)
        away_runs_list.append(ar)
        if hr > ar:
            home_wins += 1
        else:
            away_wins += 1

    home_wp = home_wins / n_sims
    avg_hr = np.mean(home_runs_list)
    avg_ar = np.mean(away_runs_list)

    # Build summary
    home_name = home_roster[0].get("team", "Home") if home_roster else "Home"
    away_name = away_roster[0].get("team", "Away") if away_roster else "Away"

    summary = (
        f"{away_name} @ {home_name}: {n_sims:,} simulations\n"
        f"Home win probability: {home_wp:.1%}\n"
        f"Avg score: {away_name} {avg_ar:.1f} - {home_name} {avg_hr:.1f}\n"
    )

    return SimResult(
        home_wins=home_wins,
        away_wins=away_wins,
        n_sims=n_sims,
        home_win_pct=home_wp,
        avg_home_runs=avg_hr,
        avg_away_runs=avg_ar,
        home_run_dist=home_runs_list,
        away_run_dist=away_runs_list,
        summary=summary,
    )


if __name__ == "__main__":
    import pandas as pd
    import sys, os, time
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    seasons = pd.read_csv("data/bravs_all_seasons.csv")

    # Simulate: 2024 Dodgers vs 2024 Yankees (WS matchup)
    print("=" * 60)
    print("  BRAVS GAME SIMULATOR")
    print("=" * 60)

    for matchup in [("LAN", "NYA", 2024), ("ATL", "LAN", 2023), ("HOU", "PHI", 2022)]:
        away_t, home_t, year = matchup
        away_data = seasons[(seasons.yearID == year) & (seasons.team == away_t) & (seasons.PA >= 100)]
        home_data = seasons[(seasons.yearID == year) & (seasons.team == home_t) & (seasons.PA >= 100)]

        away_roster = away_data.to_dict("records")
        home_roster = home_data.to_dict("records")

        if not away_roster or not home_roster:
            continue

        t0 = time.perf_counter()
        result = simulate_matchup(home_roster, away_roster, n_sims=10000)
        elapsed = time.perf_counter() - t0

        print(f"\n  {away_t} @ {home_t} ({year})")
        print(f"  {result.summary}")
        print(f"  Simulated in {elapsed:.2f}s")

    print("=" * 60)
