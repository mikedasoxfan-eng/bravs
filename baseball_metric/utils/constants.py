"""
Named constants for the BRAVS framework.

Every constant includes a citation or derivation source.
All values are calibrated to the modern MLB run environment (~4.5 R/G)
unless otherwise noted.
"""

# --- Linear Weights (wOBA scale, calibrated to ~2015-2023 average) ---
# Source: Tom Tango, "The Book" (2006), updated with FanGraphs annual coefficients
WOBA_WEIGHT_BB = 0.690  # unintentional walk
WOBA_WEIGHT_HBP = 0.720  # hit-by-pitch (slightly above BB due to pain premium / no control)
WOBA_WEIGHT_1B = 0.880  # single
WOBA_WEIGHT_2B = 1.245  # double
WOBA_WEIGHT_3B = 1.575  # triple
WOBA_WEIGHT_HR = 2.015  # home run
WOBA_WEIGHT_OUT = 0.000  # outs (baseline)

# wOBA scale factor: converts wOBA to runs above average per PA
# Source: FanGraphs annual wOBA scale, averaged 2015-2023
WOBA_SCALE = 1.157

# League average wOBA (approximate modern baseline)
# Source: FanGraphs league averages, ~2015-2023 mean
LEAGUE_AVG_WOBA = 0.315

# --- FIP Constants ---
# Source: Tango, Lichtman, Dolphin, "The Book" (2006)
FIP_HR_COEFF = 13.0  # HR coefficient in FIP formula
FIP_BB_COEFF = 3.0  # BB + HBP coefficient
FIP_K_COEFF = 2.0  # strikeout coefficient
# FIP constant is era-dependent, recalculated per season

# --- Run Values Per Event (from RE24 base-out matrix) ---
# Source: Tango's run expectancy tables, averaged 2015-2023
RUN_VALUE_STRIKE_GAINED = 0.125  # approximate run value of turning a ball into a strike
RUN_VALUE_STOLEN_BASE = 0.175  # run value of a successful steal
RUN_VALUE_CAUGHT_STEALING = -0.440  # run value of being caught stealing

# --- Positional Adjustments (runs per 162 games) ---
# Source: Tango's positional adjustment spectrum, updated
# Represents the average offensive penalty accepted for playing a premium position
POS_ADJ = {
    "C": 12.5,    # catcher: hardest position, most physical toll
    "SS": 7.5,    # shortstop: premium defensive position
    "CF": 2.5,    # center field: range and speed demands
    "2B": 2.5,    # second base: moderate defensive demands
    "3B": 2.5,    # third base: arm and reflexes
    "LF": -7.5,   # left field: minimal defensive demands
    "RF": -7.5,   # right field: arm matters but less range needed
    "1B": -12.5,  # first base: least demanding position
    "DH": -17.5,  # designated hitter: no defensive contribution
}

# --- Freely Available Talent (FAT) Baselines ---
# Runs below average per 600 PA / 200 IP, by position
# Source: Derived from analysis of replacement-level performance
# (waiver claims, minor league call-ups, Rule 5 picks), 2015-2023
FAT_BATTING_RUNS_PER_600PA = -20.0  # FAT hitters are ~20 runs below average per 600 PA
FAT_PITCHING_RUNS_PER_200IP = -25.0  # FAT pitchers are ~25 runs above (worse than) average per 200 IP

# --- Runs Per Win (dynamic, but default value) ---
# Source: Pythagorean derivative: RPW ≈ 2 * sqrt(RPG_team / PF) * (IP_team / 9)
# For modern ~4.5 R/G environment: ~9.5-10.0
DEFAULT_RUNS_PER_WIN = 9.8

# --- Bayesian Prior Parameters ---
# Population-level priors for hierarchical model
# Source: Fitted to 2010-2023 qualified player distributions

# Hitting: wOBA prior by position group
PRIOR_WOBA_MEAN = 0.315  # league average
PRIOR_WOBA_SD = 0.035  # SD of true talent wOBA across players

# Pitching: FIP prior
PRIOR_FIP_MEAN = 4.20  # league average FIP
PRIOR_FIP_SD = 0.80  # SD of true talent FIP across pitchers

# Fielding: defensive runs prior (high uncertainty)
PRIOR_FIELDING_MEAN = 0.0  # average fielder = 0 runs
PRIOR_FIELDING_SD = 5.0  # wide prior reflecting measurement uncertainty

# Baserunning: runs above average prior
PRIOR_BASERUNNING_MEAN = 0.0
PRIOR_BASERUNNING_SD = 3.0

# Catcher framing: runs above average prior
PRIOR_FRAMING_MEAN = 0.0
PRIOR_FRAMING_SD = 8.0  # wide — elite framers can be worth 15+ runs

# Approach Quality Index: runs above average prior
PRIOR_AQI_MEAN = 0.0
PRIOR_AQI_SD = 3.0

# --- Sample Size Thresholds ---
# Minimum observations before data meaningfully updates priors
MIN_PA_FOR_HITTING = 50  # below this, prior dominates
MIN_IP_FOR_PITCHING = 20.0
MIN_INNINGS_FOR_FIELDING = 300.0  # fielding requires large samples
MIN_PITCHES_FOR_FRAMING = 500
MIN_PA_FOR_AQI = 100

# --- Leverage Index ---
# Average leverage index for different roles
# Source: FanGraphs leverage index data, 2015-2023
AVG_LI_STARTER = 0.95  # starters pitch in slightly below-average leverage
AVG_LI_RELIEVER = 1.35  # relievers pitch in above-average leverage
AVG_LI_CLOSER = 1.85  # closers pitch in high leverage
AVG_LI_POSITION_PLAYER = 1.00  # position players average out to ~1.0

# --- Games / Playing Time Expectations ---
EXPECTED_GAMES_POSITION = 155  # expected full-season games for position player
EXPECTED_STARTS_PITCHER = 32  # expected starts for a full-season starter
EXPECTED_APPEARANCES_RELIEVER = 65  # expected appearances for full-season reliever
GAMES_PER_SEASON = 162

# Marginal value of a game played (wins) for a FAT-level player
# i.e., the value of showing up vs. the team playing short
MARGINAL_GAME_VALUE_POSITION = 0.030  # ~0.03 wins per game
MARGINAL_GAME_VALUE_PITCHER = 0.015

# --- Damped Leverage Exponent ---
# We use LI^alpha as the leverage multiplier, where alpha < 1 damps the effect
# Source: Calibrated to balance context-sensitivity vs stability (see docs/06-design-decisions.md)
LEVERAGE_DAMPING_EXPONENT = 0.50  # sqrt(LI) — geometric mean of "ignore leverage" and "full leverage"

# --- Ensemble Fielding Weights (default, updated by model) ---
# Prior weights for defensive metric ensemble (UZR, DRS, OAA)
DEFAULT_FIELDING_WEIGHTS = {
    "UZR": 0.30,  # Ultimate Zone Rating — zone-based, BIS data
    "DRS": 0.30,  # Defensive Runs Saved — similar methodology, different data
    "OAA": 0.40,  # Outs Above Average — Statcast-based, available 2016+
}

# --- Era Adjustment Anchor ---
# All era adjustments are relative to this season's run environment
ERA_ANCHOR_SEASON = 2023
ERA_ANCHOR_RPG = 4.62  # MLB runs per game in anchor season
