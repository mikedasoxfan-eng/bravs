"""Central probabilistic model for BRAVS computation.

Orchestrates the computation of all components, applies adjustments,
combines posteriors, and produces the final BRAVSResult.
"""

from __future__ import annotations

import logging

import numpy as np

from baseball_metric.adjustments.era_adjustment import era_run_multiplier
from baseball_metric.adjustments.league_adjustment import league_adjustment
from baseball_metric.adjustments.run_to_win import dynamic_rpw
from baseball_metric.components.baserunning import compute_baserunning
from baseball_metric.components.catcher import compute_catcher
from baseball_metric.components.durability import compute_durability
from baseball_metric.components.fielding import compute_fielding
from baseball_metric.components.hitting import compute_hitting
from baseball_metric.components.leverage import (
    compute_leverage_adjustment,
    compute_leverage_multiplier,
)
from baseball_metric.components.novel_component import compute_aqi
from baseball_metric.components.pitching import compute_pitching
from baseball_metric.components.positional import compute_positional
from baseball_metric.core.posterior import combine_component_posteriors
from baseball_metric.core.types import BRAVSResult, PlayerSeason

logger = logging.getLogger(__name__)


def compute_bravs(
    player: PlayerSeason,
    n_samples: int = 10000,
    seed: int = 42,
    apply_era_adjustment: bool = True,
    league_era: float | None = None,
    fast: bool = False,
) -> BRAVSResult:
    """Compute the full BRAVS valuation for a player-season.

    This is the main entry point for the BRAVS framework. It:
    1. Computes each component independently
    2. Applies era and league adjustments
    3. Computes the leverage multiplier
    4. Combines all components into a total posterior
    5. Converts from runs to wins

    Args:
        player: PlayerSeason with all available statistics.
        n_samples: Number of posterior samples for uncertainty estimation.
        seed: Random seed for reproducibility.
        apply_era_adjustment: Whether to apply era normalization.
        league_era: League ERA for pitching calibration. If None, estimated from league_rpg.
        fast: If True, use 2000 samples and skip correlation matrix for ~5x speedup.

    Returns:
        BRAVSResult with full component decomposition and posterior.
    """
    if fast:
        n_samples = 2000
    rng = np.random.default_rng(seed)

    # Determine league ERA from runs per game if not provided
    if league_era is None:
        league_era = player.league_rpg * 0.92  # approximate ERA from R/G (ER ≈ 92% of R)

    logger.info("Computing BRAVS for %s (%d)", player.player_name, player.season)

    # --- Step 1: Compute all components ---
    components: dict[str, object] = {}

    # Hitting (only for position players or two-way players)
    if player.pa >= 1:
        hitting = compute_hitting(player, rng=rng, n_samples=n_samples)
        components["hitting"] = hitting
        logger.debug("  Hitting: %.1f runs", hitting.runs_mean)

    # Pitching (only for pitchers or two-way players)
    if player.ip >= 1.0:
        pitching = compute_pitching(player, league_era=league_era, rng=rng, n_samples=n_samples)
        components["pitching"] = pitching
        logger.debug("  Pitching: %.1f runs", pitching.runs_mean)

    # Baserunning
    if player.pa >= 1:
        baserunning = compute_baserunning(player, rng=rng, n_samples=n_samples)
        components["baserunning"] = baserunning
        logger.debug("  Baserunning: %.1f runs", baserunning.runs_mean)

    # Fielding
    fielding = compute_fielding(player, rng=rng, n_samples=n_samples)
    components["fielding"] = fielding
    logger.debug("  Fielding: %.1f runs", fielding.runs_mean)

    # Catcher-specific
    if player.is_catcher:
        catcher = compute_catcher(player, rng=rng, n_samples=n_samples)
        components["catcher"] = catcher
        logger.debug("  Catcher: %.1f runs", catcher.runs_mean)

    # Positional adjustment
    positional = compute_positional(player, rng=rng, n_samples=n_samples)
    components["positional"] = positional
    logger.debug("  Positional: %.1f runs", positional.runs_mean)

    # Approach Quality Index (novel component)
    if player.pa >= 1:
        aqi = compute_aqi(player, rng=rng, n_samples=n_samples)
        components["approach_quality"] = aqi
        logger.debug("  AQI: %.1f runs", aqi.runs_mean)

    # Durability
    durability = compute_durability(player, rng=rng, n_samples=n_samples)
    components["durability"] = durability
    logger.debug("  Durability: %.1f runs", durability.runs_mean)

    # --- Step 2: Apply era and league adjustments ---
    if apply_era_adjustment:
        era_mult = era_run_multiplier(player.season)
        for name, comp in components.items():
            comp.runs_mean *= era_mult  # type: ignore[union-attr]
            comp.runs_var *= era_mult ** 2  # type: ignore[union-attr]
            if comp.samples is not None:  # type: ignore[union-attr]
                comp.samples *= era_mult  # type: ignore[union-attr]

    # League adjustment (small, added to hitting)
    lg_adj = league_adjustment(player.league, player.season)
    if "hitting" in components:
        components["hitting"].runs_mean += lg_adj  # type: ignore[union-attr]

    # --- Step 3: Leverage ---
    leverage_mult = compute_leverage_multiplier(player)

    # Compute skill-based total for leverage adjustment component
    skill_components = {"hitting", "pitching", "baserunning", "fielding",
                        "catcher", "approach_quality"}
    skill_runs = sum(
        c.runs_mean for name, c in components.items()  # type: ignore[union-attr]
        if name in skill_components
    )
    skill_var = sum(
        c.runs_var for name, c in components.items()  # type: ignore[union-attr]
        if name in skill_components
    )

    leverage_comp = compute_leverage_adjustment(
        player, skill_runs, skill_var, rng=rng, n_samples=n_samples
    )
    components["leverage"] = leverage_comp

    # --- Step 4: Dynamic RPW ---
    rpw = dynamic_rpw(player.season, player.park_factor)

    # --- Step 5: Combine posteriors ---
    total_mean, total_var, total_samples = combine_component_posteriors(
        components=components,  # type: ignore[arg-type]
        leverage_multiplier=leverage_mult,
        rpw=rpw,
        rng=rng,
        n_samples=n_samples,
    )

    logger.info(
        "  BRAVS: %.1f wins [%.1f, %.1f]",
        total_mean / rpw,
        float(np.percentile(total_samples / rpw, 5)),
        float(np.percentile(total_samples / rpw, 95)),
    )

    return BRAVSResult(
        player=player,
        components=components,  # type: ignore[arg-type]
        total_runs_mean=total_mean,
        total_runs_var=total_var,
        rpw=rpw,
        leverage_multiplier=leverage_mult,
        total_samples=total_samples,
    )
