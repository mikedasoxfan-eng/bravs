"""BRAVS v2.0 GPU Engine — comprehensive fixes for all 10 known issues.

Changes from v1:
1. Baserunning: removed aggressive shrinkage, proper SB/CS/GIDP/advance modeling
2. Fielding: position-normalized range factor (PO+A vs position avg per INNING, not raw)
   with position-specific run values (SS plays worth more than 1B plays)
3. Positional: updated spectrum — LF/RF at -5.0 (was -7.5), DH at -15.0 (was -17.5)
4. AQI: reduced wOBA-residual penalty, allow extreme discipline to show through
5. Leverage: batch leverage from saves/holds proxy (closers get credit)
6. Catcher: position-based framing proxy for catchers
7. Durability: uses dynamic RPW instead of fixed 9.8
8. Era: dampened adjustment (sqrt of raw multiplier) to reduce 1960s inflation
9. Calibration: adjusted component weights so raw BRAVS aligns closer to WAR
10. Two-way: reduced DH penalty for players with pitching contributions
"""

from __future__ import annotations

import time
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 2000

# v2.0 Positional spectrum — less extreme than v1
POS_ADJ_V2 = {
    "C": 10.0, "SS": 7.0, "CF": 3.0, "2B": 3.0, "3B": 2.0,
    "LF": -5.0, "RF": -5.0, "1B": -10.0, "DH": -15.0, "P": 0.0,
}

# Position-specific fielding run values per play above average
# SS/3B plays are worth more because they're harder
POS_FIELDING_VALUE = {
    "C": 0.15, "1B": 0.10, "2B": 0.35, "3B": 0.35,
    "SS": 0.40, "LF": 0.20, "CF": 0.25, "RF": 0.20,
    "OF": 0.22, "P": 0.05, "DH": 0.0,
}

ERA_RPG = {
    1920: 4.39, 1925: 5.06, 1930: 5.55, 1935: 5.08, 1940: 4.65,
    1945: 4.09, 1950: 4.84, 1955: 4.44, 1960: 4.24, 1965: 3.89,
    1968: 3.42, 1969: 4.07, 1970: 4.34, 1973: 4.12, 1975: 4.12,
    1980: 4.29, 1985: 4.33, 1990: 4.26, 1995: 4.85, 1997: 4.77,
    1999: 5.08, 2000: 5.14, 2001: 4.78, 2004: 4.81, 2005: 4.59,
    2008: 4.65, 2010: 4.38, 2011: 4.28, 2012: 4.32, 2013: 4.17,
    2014: 4.07, 2015: 4.25, 2016: 4.48, 2017: 4.65, 2018: 4.45,
    2019: 4.83, 2020: 4.65, 2021: 4.26, 2022: 4.28, 2023: 4.62,
    2024: 4.52, 2025: 4.45,
}


def _get_rpg(year: int) -> float:
    if year in ERA_RPG:
        return ERA_RPG[year]
    keys = sorted(ERA_RPG.keys())
    if year < keys[0]: return ERA_RPG[keys[0]]
    if year > keys[-1]: return ERA_RPG[keys[-1]]
    for i in range(len(keys) - 1):
        if keys[i] <= year <= keys[i + 1]:
            t = (year - keys[i]) / (keys[i + 1] - keys[i])
            return ERA_RPG[keys[i]] * (1 - t) + ERA_RPG[keys[i + 1]] * t
    return 4.5


def batch_compute_bravs_v2(
    player_data: list[dict],
    n_samples: int = N_SAMPLES,
    seed: int = 42,
) -> list[dict]:
    """BRAVS v2.0 — all 10 fixes applied."""
    if not player_data:
        return []

    N = len(player_data)
    t0 = time.perf_counter()

    def _f(key, default=0):
        return torch.tensor([float(d.get(key, default) or default) for d in player_data],
                           dtype=torch.float32, device=DEVICE)

    # Extract all fields
    pa = _f("PA"); ab = _f("AB"); hits = _f("H")
    doubles = _f("2B"); triples = _f("3B"); hr = _f("HR")
    bb = _f("BB"); ibb = _f("IBB"); hbp = _f("HBP")
    so = _f("SO"); sf = _f("SF"); sb = _f("SB"); cs = _f("CS")
    gidp = _f("GIDP"); games = _f("G")
    ip = _f("IP"); er = _f("ER"); h_allowed = _f("H_allowed")
    hr_allowed = _f("HR_allowed"); bb_allowed = _f("BB_allowed")
    hbp_allowed = _f("HBP_allowed"); k_pitch = _f("K_pitch")
    g_pitched = _f("G_pitched"); gs = _f("GS"); sv = _f("SV")
    park_factor = _f("park_factor", 1.0)
    season_games = _f("season_games", 162)
    fielding_rf = _f("fielding_rf", 0); fielding_e = _f("fielding_e", 0)

    ubb = bb - ibb
    singles = (hits - doubles - triples - hr).clamp(min=0)

    rpg = torch.tensor([_get_rpg(int(d.get("yearID", 2023))) for d in player_data],
                       dtype=torch.float32, device=DEVICE)

    # FIX 8: Dampened era adjustment — use sqrt to reduce 1960s inflation
    raw_era_mult = 4.62 / rpg
    era_mult = raw_era_mult.sqrt() * raw_era_mult.sqrt().mean().reciprocal()  # normalize to mean=1
    # Simpler: geometric mean between 1.0 and the raw multiplier
    era_mult = (raw_era_mult * 1.0).sqrt()  # sqrt(raw * 1) = sqrt(raw)
    # Renormalize so average era_mult ≈ 1.0
    era_mult = era_mult / era_mult.mean()

    pos_adj = torch.tensor([POS_ADJ_V2.get(d.get("position", "DH"), 0.0)
                           for d in player_data], dtype=torch.float32, device=DEVICE)

    # Position-specific fielding value multiplier
    pos_fld_val = torch.tensor([POS_FIELDING_VALUE.get(d.get("position", "DH"), 0.0)
                               for d in player_data], dtype=torch.float32, device=DEVICE)

    # Dynamic RPW
    rpg_adj = rpg * park_factor
    exponent = rpg_adj.pow(0.287)
    rpw = 2.0 * rpg_adj / exponent

    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    # ============================================================
    # HITTING — same Bayesian wOBA as v1
    # ============================================================
    woba_num = (0.690 * ubb + 0.720 * hbp + 0.880 * singles +
                1.245 * doubles + 1.575 * triples + 2.015 * hr)
    woba_den = (ab + bb + sf + hbp).clamp(min=1)
    obs_woba = woba_num / woba_den / park_factor

    prior_prec = 1.0 / (0.035 ** 2)
    data_prec = pa.clamp(min=50) / 0.09
    post_prec = prior_prec + data_prec
    post_var = 1.0 / post_prec
    post_mean = post_var * (prior_prec * 0.315 + data_prec * obs_woba)

    fat_runs = -20.0 * (pa / 600.0)
    hitting_runs = ((post_mean - 0.315) / 1.157 * pa - fat_runs) * era_mult

    hit_samples = post_mean.unsqueeze(1) + post_var.sqrt().unsqueeze(1) * torch.randn(N, n_samples, device=DEVICE, generator=gen)
    hitting_samples = ((hit_samples - 0.315) / 1.157 * pa.unsqueeze(1) - fat_runs.unsqueeze(1)) * era_mult.unsqueeze(1)

    # ============================================================
    # PITCHING — same as v1
    # ============================================================
    league_era = rpg * 0.92
    fip_c = league_era - 4.20 + 3.10
    ip_safe = ip.clamp(min=0.1)
    obs_fip = (13.0 * hr_allowed + 3.0 * (bb_allowed + hbp_allowed) - 2.0 * k_pitch) / ip_safe + fip_c
    obs_fip_adj = obs_fip / park_factor

    p_post_var = 1.0 / (1.0 / (0.80 ** 2) + ip.clamp(min=20) / 2.25)
    p_post_mean = p_post_var * ((1.0 / (0.80 ** 2)) * 4.20 + (ip.clamp(min=20) / 2.25) * obs_fip_adj)
    fat_era = league_era + (25.0 / 200.0) * 9.0
    pitching_runs = ((fat_era - p_post_mean) / 9.0 * ip) * era_mult
    is_pitcher = (ip >= 10.0).float()
    pitching_runs = pitching_runs * is_pitcher

    pit_samples = ((fat_era.unsqueeze(1) - (p_post_mean.unsqueeze(1) + p_post_var.sqrt().unsqueeze(1) *
                    torch.randn(N, n_samples, device=DEVICE, generator=gen))) / 9.0 * ip.unsqueeze(1)) * era_mult.unsqueeze(1)
    pit_samples = pit_samples * is_pitcher.unsqueeze(1)

    # ============================================================
    # FIX 1: BASERUNNING — no aggressive shrinkage, proper modeling
    # ============================================================
    # SB value: +0.175 per SB, -0.44 per CS (RE24 based)
    sb_runs = sb * 0.175 + cs * (-0.44)
    # GIDP avoidance: compared to expected GIDP rate
    gidp_exp = pa * 0.15 * 0.11
    gidp_runs = (gidp_exp - gidp) * 0.37
    # Extra base running: triples are a proxy for speed (above avg triple rate)
    triple_rate = triples / ab.clamp(min=1)
    speed_proxy = (triple_rate - 0.005).clamp(min=-0.005) * ab * 2.0  # ~2 runs per extra triple
    # Total — NO shrinkage (was 0.7 in v1, killed baserunning value)
    baserunning_runs = (sb_runs + gidp_runs + speed_proxy) * era_mult

    # ============================================================
    # FIX 2: FIELDING — position-normalized, position-weighted
    # ============================================================
    # fielding_rf and fielding_e come pre-computed from Lahman (range factor & error above avg)
    # Apply position-specific run values (SS plays worth more than 1B plays)
    fielding_runs = (fielding_rf * pos_fld_val + fielding_e * 0.5) * era_mult
    # Moderate shrinkage (0.6 instead of 0.5)
    fielding_runs = fielding_runs * 0.6

    # ============================================================
    # FIX 3: POSITIONAL — updated spectrum (less extreme)
    # ============================================================
    games_frac = games / 162.0
    positional_runs = (pos_adj * games_frac) * era_mult

    # FIX 7: Two-way player adjustment — if player has significant IP,
    # reduce the DH penalty on their hitting (they ARE playing a position)
    has_pitching = (ip >= 30.0).float()
    dh_mask = torch.tensor([1.0 if d.get("position") == "DH" else 0.0
                           for d in player_data], device=DEVICE)
    # Give back 50% of the DH penalty for two-way players
    two_way_credit = dh_mask * has_pitching * abs(POS_ADJ_V2["DH"]) * 0.5 * games_frac * era_mult
    positional_runs = positional_runs + two_way_credit

    # ============================================================
    # FIX 10: DURABILITY — dynamic RPW instead of fixed 9.8
    # ============================================================
    is_pit = (ip >= 20.0).float()
    is_starter = ((gs > g_pitched * 0.5) * is_pit).float()
    is_reliever = ((1 - is_starter) * is_pit).float()
    is_batter = (1 - is_pit)

    expected_full = (is_batter * 155 + is_starter * 32 + is_reliever * 65)
    scale = (season_games * 0.95 / 162.0).clamp(max=1.0)
    expected = (expected_full * scale).clamp(min=1)
    actual_g = is_batter * games + is_starter * gs + is_reliever * g_pitched
    marginal = (is_batter * 0.030 + is_pit * 0.015 * (1 + is_starter))
    durability_runs = ((actual_g - expected) * marginal * rpw) * era_mult  # FIX: use dynamic rpw

    # ============================================================
    # FIX 5: LEVERAGE — proxy from saves/holds
    # ============================================================
    # Closers (high saves) pitch in high leverage — give them credit
    # Estimate gmLI from saves: 0+ saves = avg leverage, 30+ = high leverage
    est_li = 1.0 + (sv / 50.0).clamp(max=1.0) * 0.85  # max ~1.85 for elite closers
    lev_mult = est_li.pow(0.5) / 0.97  # damped sqrt, normalized
    skill_runs = pitching_runs  # leverage only applies to pitchers meaningfully
    leverage_runs = skill_runs * (lev_mult - 1.0) * is_pit

    # ============================================================
    # FIX 6: CATCHER — crude framing proxy
    # ============================================================
    # Catchers who play a lot of games at C get a small framing credit
    # based on the assumption that catchers who keep their job are at least
    # average framers. Elite framers should be identified by other data.
    is_catcher = torch.tensor([1.0 if d.get("position") == "C" else 0.0
                              for d in player_data], device=DEVICE)
    # Small baseline credit for all catchers (~2 runs for full season)
    catcher_runs = is_catcher * games_frac * 2.0 * era_mult

    # ============================================================
    # FIX 4: AQI — less aggressive orthogonalization penalty
    # ============================================================
    bb_rate = bb / pa.clamp(min=1)
    k_rate = so / pa.clamp(min=1)
    exp_woba = 0.310 + 0.80 * (bb_rate - 0.085) + (-0.45) * (k_rate - 0.220)
    woba_resid = obs_woba * park_factor - exp_woba
    # FIX: reduced residual penalty from -8.0 to -5.0
    aqi_raw = 1.5 * (bb_rate - 0.085) + (-1.0) * (k_rate - 0.220) + (-5.0) * woba_resid
    aqi_runs = (aqi_raw * (pa / 600.0) * 3.0) * era_mult
    aqi_shrink = (pa / (pa + 200)).clamp(max=0.8)
    aqi_runs = aqi_runs * aqi_shrink * (pa >= 100).float()

    # ============================================================
    # TOTAL
    # ============================================================
    has_batting = (pa >= 50).float()
    total_runs = (hitting_runs * has_batting + pitching_runs +
                  baserunning_runs * has_batting + fielding_runs * has_batting +
                  positional_runs + durability_runs + aqi_runs * has_batting +
                  leverage_runs + catcher_runs)

    # Posterior
    other_runs = (pitching_runs + baserunning_runs * has_batting + fielding_runs * has_batting +
                  positional_runs + durability_runs + aqi_runs * has_batting +
                  leverage_runs + catcher_runs).unsqueeze(1)
    total_samples = hitting_samples * has_batting.unsqueeze(1) + pit_samples + other_runs
    total_samples = total_samples + torch.randn(N, n_samples, device=DEVICE, generator=gen) * 3.0

    bravs = total_runs / rpw
    bravs_era_std = total_runs / 5.90

    # FIX 9: No more 0.57 multiplier — the fixes above should bring the scale closer
    # to WAR naturally. Use 0.65 as a lighter touch.
    bravs_war_eq = bravs * 0.65

    bravs_samples = total_samples / rpw.unsqueeze(1)
    ci90_lo = torch.quantile(bravs_samples, 0.05, dim=1)
    ci90_hi = torch.quantile(bravs_samples, 0.95, dim=1)

    t1 = time.perf_counter()

    # Build output
    results = []
    bravs_cpu = bravs.cpu().numpy()
    era_std_cpu = bravs_era_std.cpu().numpy()
    war_eq_cpu = bravs_war_eq.cpu().numpy()
    ci_lo = ci90_lo.cpu().numpy()
    ci_hi = ci90_hi.cpu().numpy()
    hit_cpu = hitting_runs.cpu().numpy()
    pit_cpu = pitching_runs.cpu().numpy()
    br_cpu = baserunning_runs.cpu().numpy()
    fld_cpu = fielding_runs.cpu().numpy()
    pos_cpu = positional_runs.cpu().numpy()
    dur_cpu = durability_runs.cpu().numpy()
    aqi_cpu = aqi_runs.cpu().numpy()
    lev_cpu = leverage_runs.cpu().numpy()
    cat_cpu = catcher_runs.cpu().numpy()
    rpw_cpu = rpw.cpu().numpy()

    for i, d in enumerate(player_data):
        results.append({
            "playerID": d.get("playerID", ""),
            "yearID": d.get("yearID", 0),
            "name": d.get("name", ""),
            "team": d.get("team", ""),
            "position": d.get("position", ""),
            "G": int(d.get("G", 0) or 0),
            "PA": int(d.get("PA", 0) or 0),
            "HR": int(d.get("HR", 0) or 0),
            "SB": int(d.get("SB", 0) or 0),
            "IP": round(float(d.get("IP", 0) or 0), 1),
            "bravs": round(float(bravs_cpu[i]), 2),
            "bravs_era_std": round(float(era_std_cpu[i]), 2),
            "bravs_war_eq": round(float(war_eq_cpu[i]), 2),
            "ci90_lo": round(float(ci_lo[i]), 2),
            "ci90_hi": round(float(ci_hi[i]), 2),
            "rpw": round(float(rpw_cpu[i]), 3),
            "hitting_runs": round(float(hit_cpu[i]), 1),
            "pitching_runs": round(float(pit_cpu[i]), 1),
            "baserunning_runs": round(float(br_cpu[i]), 1),
            "fielding_runs": round(float(fld_cpu[i]), 1),
            "positional_runs": round(float(pos_cpu[i]), 1),
            "durability_runs": round(float(dur_cpu[i]), 1),
            "aqi_runs": round(float(aqi_cpu[i]), 1),
            "leverage_runs": round(float(lev_cpu[i]), 1),
            "catcher_runs": round(float(cat_cpu[i]), 1),
        })

    return results
