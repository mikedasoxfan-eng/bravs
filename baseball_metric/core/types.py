"""Core data types for the BRAVS framework."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class ComponentResult:
    """Result of a single BRAVS component computation.

    Every component produces a posterior distribution over runs above FAT,
    summarized by mean, credible intervals, and optional raw samples.
    """

    name: str
    runs_mean: float
    runs_var: float
    ci_50: tuple[float, float] = (0.0, 0.0)
    ci_90: tuple[float, float] = (0.0, 0.0)
    samples: NDArray[np.floating[object]] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def runs_sd(self) -> float:
        return float(np.sqrt(self.runs_var))

    def wins(self, rpw: float) -> float:
        """Convert runs to wins using dynamic RPW."""
        return self.runs_mean / rpw

    def wins_ci_90(self, rpw: float) -> tuple[float, float]:
        return (self.ci_90[0] / rpw, self.ci_90[1] / rpw)


@dataclass
class PlayerSeason:
    """Container for a single player-season's raw statistics.

    All fields are optional to support pitchers, position players,
    catchers, two-way players, and historical data gaps.
    """

    player_id: str
    player_name: str
    season: int
    team: str
    position: str  # primary position

    # Hitting stats
    pa: int = 0
    ab: int = 0
    hits: int = 0
    singles: int = 0
    doubles: int = 0
    triples: int = 0
    hr: int = 0
    bb: int = 0
    ibb: int = 0
    hbp: int = 0
    k: int = 0
    sf: int = 0
    sh: int = 0
    sb: int = 0
    cs: int = 0
    gidp: int = 0
    games: int = 0

    # Pitching stats
    ip: float = 0.0
    er: int = 0
    hits_allowed: int = 0
    hr_allowed: int = 0
    bb_allowed: int = 0
    hbp_allowed: int = 0
    k_pitching: int = 0
    games_pitched: int = 0
    games_started: int = 0
    saves: int = 0
    holds: int = 0

    # Fielding stats (if available)
    inn_fielded: float = 0.0
    uzr: float | None = None
    drs: float | None = None
    oaa: float | None = None
    total_zone: float | None = None  # TotalZone defensive runs (pre-2002 historical)

    # Catcher stats (if applicable)
    framing_runs: float | None = None
    blocking_runs: float | None = None
    throwing_runs: float | None = None
    catcher_pitches: int = 0
    game_calling_runs: float | None = None  # WOWY-based game-calling estimate
    game_calling_years: int = 1  # years of WOWY data (more = more reliable)

    # Leverage / context
    avg_leverage_index: float = 1.0  # gmLI

    # Baserunning advanced (if available)
    extra_bases_taken: int = 0
    extra_base_opportunities: int = 0
    outs_on_bases: int = 0

    # Approach / pitch-level (if available — Statcast era)
    pitches_seen: int = 0
    chase_rate: float | None = None  # O-Swing%
    zone_contact_rate: float | None = None  # Z-Contact%
    aqi_raw: float | None = None  # pre-computed AQI if available

    # Pitching Statcast (if available — 2015+)
    xwoba_against: float | None = None  # expected wOBA against on contact
    contact_rate: float | None = None  # 1 - K% for the pitcher (rate balls in play)

    # Multi-position support: dict of position -> games at that position
    positions_played: dict[str, int] | None = None

    # Context
    league: str = "MLB"
    park_factor: float = 1.0  # multi-dimensional park factor (overall)
    league_rpg: float = 4.5  # league runs per game
    season_games: int = 162  # actual games in the season (60 for 2020)

    def __post_init__(self) -> None:
        """Auto-compute singles if hits and extra-base hits are provided."""
        if self.hits > 0 and self.singles == 0:
            computed = self.hits - self.doubles - self.triples - self.hr
            if computed >= 0:
                self.singles = computed

    @property
    def is_pitcher(self) -> bool:
        return self.ip >= 20.0

    @property
    def is_two_way(self) -> bool:
        return self.ip >= 20.0 and self.pa >= 100

    @property
    def is_catcher(self) -> bool:
        return self.position == "C"

    @property
    def ubb(self) -> int:
        """Unintentional walks."""
        return self.bb - self.ibb


@dataclass
class BRAVSResult:
    """Complete BRAVS valuation for a player-season.

    Contains the total valuation plus all component decompositions.
    """

    player: PlayerSeason
    components: dict[str, ComponentResult] = field(default_factory=dict)
    total_runs_mean: float = 0.0
    total_runs_var: float = 0.0
    rpw: float = 9.8  # runs-per-win used
    leverage_multiplier: float = 1.0

    # Posterior samples for total BRAVS (wins)
    total_samples: NDArray[np.floating[object]] | None = None

    # Calibration factor: scales raw BRAVS to WAR-equivalent range.
    # v3.7: position-specific in GPU engine (0.625 pitchers, 0.690 hitters)
    # This single factor is used by the Python engine as a compromise.
    WAR_CALIBRATION_FACTOR: float = 0.67

    # Fixed modern RPW for era-standardized comparison
    STANDARD_RPW: float = 5.90  # ~2023 run environment

    @property
    def bravs(self) -> float:
        """Point estimate of BRAVS (wins above FAT) — context-dependent."""
        return self.total_runs_mean / self.rpw

    @property
    def bravs_era_standardized(self) -> float:
        """Era-standardized BRAVS using fixed modern RPW.

        Removes the RPW effect that inflates dead-ball/pitcher's era seasons.
        Gibson 1968 still gets credit for his runs saved, but each run
        converts to wins at the same rate as a modern player.
        """
        return self.total_runs_mean / self.STANDARD_RPW

    @property
    def bravs_calibrated(self) -> float:
        """WAR-equivalent calibrated BRAVS for direct comparison with fWAR/bWAR."""
        return self.bravs * self.WAR_CALIBRATION_FACTOR

    @property
    def bravs_ci_50(self) -> tuple[float, float]:
        """50% credible interval for BRAVS."""
        if self.total_samples is not None:
            from baseball_metric.utils.math_helpers import credible_interval
            return credible_interval(self.total_samples / self.rpw, 0.50)
        sd = float(np.sqrt(self.total_runs_var)) / self.rpw
        return (self.bravs - 0.67 * sd, self.bravs + 0.67 * sd)

    @property
    def bravs_ci_90(self) -> tuple[float, float]:
        """90% credible interval for BRAVS."""
        if self.total_samples is not None:
            from baseball_metric.utils.math_helpers import credible_interval
            return credible_interval(self.total_samples / self.rpw, 0.90)
        sd = float(np.sqrt(self.total_runs_var)) / self.rpw
        return (self.bravs - 1.645 * sd, self.bravs + 1.645 * sd)

    def summary(self) -> str:
        """Human-readable summary of the BRAVS result."""
        ci90 = self.bravs_ci_90
        cal = self.WAR_CALIBRATION_FACTOR
        lines = [
            f"{'='*60}",
            f"BRAVS Report: {self.player.player_name} ({self.player.season})",
            f"{'='*60}",
            f"Total BRAVS: {self.bravs:.1f} wins  [90% CI: {ci90[0]:.1f} to {ci90[1]:.1f}]",
            f"Era-std:     {self.bravs_era_standardized:.1f} wins  "
            f"(fixed RPW={self.STANDARD_RPW})",
            f"WAR-equiv:   {self.bravs_calibrated:.1f} wins  "
            f"[{ci90[0]*cal:.1f} to {ci90[1]*cal:.1f}]",
            f"Total Runs Above FAT: {self.total_runs_mean:.1f}",
            f"Runs Per Win: {self.rpw:.2f}",
            f"Leverage Multiplier: {self.leverage_multiplier:.3f}",
            f"",
            f"Component Decomposition (runs):",
            f"{'-'*40}",
        ]
        for name, comp in sorted(self.components.items()):
            ci = comp.ci_90
            lines.append(f"  {name:20s}: {comp.runs_mean:+7.1f}  [{ci[0]:+.1f}, {ci[1]:+.1f}]")

        lines.append(f"{'='*60}")
        return "\n".join(lines)
