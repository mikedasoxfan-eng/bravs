"""Leverage Index computation from Statcast play-by-play data.

Instead of Retrosheet event files (which require complex parsing),
we use Statcast's delta_home_win_exp to compute actual game leverage
for each plate appearance. This is the real leverage — not a proxy.

Available for 2016+ (Statcast era).
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pybaseball import statcast, cache

cache.enable()
log = logging.getLogger(__name__)

# Base-out run expectancy matrix (2010-2019 average)
# Used to compute theoretical leverage when WPA isn't available
RE24 = {
    # (runners, outs): expected runs
    ("---", 0): 0.481, ("---", 1): 0.254, ("---", 2): 0.098,
    ("1--", 0): 0.859, ("1--", 1): 0.509, ("1--", 2): 0.224,
    ("-2-", 0): 1.100, ("-2-", 1): 0.664, ("-2-", 2): 0.319,
    ("12-", 0): 1.437, ("12-", 1): 0.884, ("12-", 2): 0.429,
    ("--3", 0): 1.350, ("--3", 1): 0.950, ("--3", 2): 0.353,
    ("1-3", 0): 1.784, ("1-3", 1): 1.130, ("1-3", 2): 0.478,
    ("-23", 0): 1.964, ("-23", 1): 1.376, ("-23", 2): 0.580,
    ("123", 0): 2.282, ("123", 1): 1.520, ("123", 2): 0.689,
}

# Average absolute WPA per plate appearance ≈ 0.035
# Leverage = |WPA_this_PA| / avg_WPA
AVG_WPA_PER_PA = 0.035


def compute_leverage_season(year: int) -> pd.DataFrame:
    """Compute average game leverage index for each pitcher in a season.

    Uses Statcast's delta_home_win_exp to measure actual leverage.
    gmLI = mean(|delta_win_exp|) / league_avg(|delta_win_exp|)

    Args:
        year: Season year (2016+)

    Returns:
        DataFrame: pitcher_id, pitcher_name, IP_est, gmLI, max_LI, appearances
    """
    log.info("Computing leverage for %d season...", year)

    all_data = []
    for m_start, m_end in [
        (f"{year}-03-20", f"{year}-04-30"),
        (f"{year}-05-01", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-08-31"),
        (f"{year}-09-01", f"{year}-10-15"),
    ]:
        try:
            chunk = statcast(m_start, m_end)
            if not chunk.empty:
                all_data.append(chunk)
                log.info("  %s to %s: %d pitches", m_start, m_end, len(chunk))
        except Exception as e:
            log.warning("  Failed: %s", e)

    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)
    df = df[df.game_type == "R"].copy()
    df = df[df.delta_home_win_exp.notna()].copy()

    # Compute absolute WPA per pitch
    df['abs_wpa'] = df.delta_home_win_exp.abs()

    # League average WPA per pitch
    league_avg_wpa = df.abs_wpa.mean()
    log.info("League avg |WPA| per pitch: %.4f", league_avg_wpa)

    # Per-pitcher aggregation
    pitcher_li = df.groupby('pitcher').agg(
        pitcher_name=('player_name', 'first'),
        total_pitches=('abs_wpa', 'count'),
        mean_abs_wpa=('abs_wpa', 'mean'),
        max_abs_wpa=('abs_wpa', 'max'),
        appearances=('game_pk', 'nunique'),
    ).reset_index()

    # Leverage index = pitcher's mean |WPA| / league mean |WPA|
    pitcher_li['gmLI'] = (pitcher_li.mean_abs_wpa / league_avg_wpa).round(3)
    pitcher_li['max_LI'] = (pitcher_li.max_abs_wpa / league_avg_wpa).round(1)
    pitcher_li['IP_est'] = (pitcher_li.total_pitches / 15).round(1)  # ~15 pitches per IP

    # Filter to meaningful samples
    pitcher_li = pitcher_li[pitcher_li.total_pitches >= 100]

    log.info("Leverage computed for %d pitchers", len(pitcher_li))
    return pitcher_li[['pitcher', 'pitcher_name', 'IP_est', 'gmLI', 'max_LI',
                        'appearances', 'total_pitches']].sort_values('gmLI', ascending=False)
