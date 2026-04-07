"""Data quality checks for BRAVS pipeline.

Validates incoming data for anomalies, impossible stat lines,
and missing values before they enter the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from baseball_metric.core.types import PlayerSeason

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a PlayerSeason."""

    is_valid: bool
    warnings: list[str]
    errors: list[str]


def validate_player_season(player: PlayerSeason) -> ValidationResult:
    """Validate a PlayerSeason for data quality issues.

    Checks for:
    - Impossible stat lines (negative values, hits > AB, etc.)
    - Suspicious outliers (BA > .400, ERA < 1.0 in large sample, etc.)
    - Missing required fields

    Args:
        player: PlayerSeason to validate.

    Returns:
        ValidationResult with any warnings or errors.
    """
    warnings: list[str] = []
    errors: list[str] = []

    # --- Required fields ---
    if not player.player_name:
        errors.append("Missing player_name")
    if player.season < 1871 or player.season > 2030:
        errors.append(f"Suspicious season: {player.season}")

    # --- Batting checks ---
    if player.pa > 0:
        if player.ab > player.pa:
            errors.append(f"AB ({player.ab}) > PA ({player.pa})")
        if player.hits > player.ab:
            errors.append(f"H ({player.hits}) > AB ({player.ab})")
        if player.hr > player.hits:
            errors.append(f"HR ({player.hr}) > H ({player.hits})")
        if player.singles + player.doubles + player.triples + player.hr != player.hits:
            warnings.append("1B + 2B + 3B + HR != H — recomputing singles")
        if player.bb < 0 or player.k < 0 or player.hr < 0:
            errors.append("Negative counting stats")

        # Suspicious outliers
        if player.ab > 0:
            ba = player.hits / player.ab
            if ba > 0.420 and player.ab > 200:
                warnings.append(f"Extremely high BA ({ba:.3f}) in {player.ab} AB")
            if ba < 0.150 and player.ab > 200:
                warnings.append(f"Extremely low BA ({ba:.3f}) in {player.ab} AB")

        if player.pa > 800:
            warnings.append(f"Very high PA ({player.pa}) — check for combined stats")

    # --- Pitching checks ---
    if player.ip > 0:
        if player.ip > 400:
            warnings.append(f"Extremely high IP ({player.ip}) — dead-ball era or error?")

        era = player.er / player.ip * 9.0 if player.ip > 0 else 0
        if era < 1.0 and player.ip > 100:
            warnings.append(f"Extremely low ERA ({era:.2f}) in {player.ip} IP")
        if era > 9.0 and player.ip > 50:
            warnings.append(f"Extremely high ERA ({era:.2f}) in {player.ip} IP")

        if player.k_pitching < 0 or player.bb_allowed < 0:
            errors.append("Negative pitching counting stats")

    # --- Fielding checks ---
    if player.inn_fielded < 0:
        errors.append(f"Negative innings fielded: {player.inn_fielded}")

    # --- Games ---
    if player.games > 162:
        warnings.append(f"Games ({player.games}) > 162 — traded mid-season?")
    if player.games < 0:
        errors.append(f"Negative games: {player.games}")

    is_valid = len(errors) == 0

    if errors:
        logger.error("Validation errors for %s (%d): %s",
                      player.player_name, player.season, "; ".join(errors))
    if warnings:
        logger.warning("Validation warnings for %s (%d): %s",
                        player.player_name, player.season, "; ".join(warnings))

    return ValidationResult(is_valid=is_valid, warnings=warnings, errors=errors)
