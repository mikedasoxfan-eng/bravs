use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

// ---- Constants ----
const WOBA_W_BB: f64 = 0.690;
const WOBA_W_HBP: f64 = 0.720;
const WOBA_W_1B: f64 = 0.880;
const WOBA_W_2B: f64 = 1.245;
const WOBA_W_3B: f64 = 1.575;
const WOBA_W_HR: f64 = 2.015;
const WOBA_SCALE: f64 = 1.157;
const LEAGUE_AVG_WOBA: f64 = 0.315;
const FAT_BAT_PER_600: f64 = -20.0;
const FAT_PITCH_PER_200: f64 = -25.0;
const FIP_HR_C: f64 = 13.0;
const FIP_BB_C: f64 = 3.0;
const FIP_K_C: f64 = 2.0;
const PRIOR_WOBA_MEAN: f64 = 0.315;
const PRIOR_WOBA_SD: f64 = 0.035;
const PRIOR_FIP_MEAN: f64 = 4.20;
const PRIOR_FIP_SD: f64 = 0.80;
const PRIOR_FIELD_SD: f64 = 5.0;
const PRIOR_BR_SD: f64 = 3.0;
const PRIOR_AQI_SD: f64 = 3.0;
const PRIOR_FRAME_SD: f64 = 8.0;
const WOBA_OBS_VAR: f64 = 0.09;
const FIP_OBS_VAR: f64 = 2.25;
const FIELD_OBS_VAR: f64 = 60.0;
const TOTALZONE_OBS_VAR: f64 = 150.0;
const BR_OBS_VAR: f64 = 9.0;
const AQI_OBS_VAR: f64 = 12.0;
const FRAME_OBS_VAR: f64 = 25.0;
const RV_SB: f64 = 0.175;
const RV_CS: f64 = -0.440;
const RV_GIDP: f64 = -0.37;
const LEVERAGE_EXP: f64 = 0.50;
const AVG_DAMPED_LI: f64 = 0.97;
const GAME_MARGINAL_POS: f64 = 0.030;
const GAME_MARGINAL_PIT: f64 = 0.015;
const N_SAMPLES_FULL: usize = 10000;
const N_SAMPLES_FAST: usize = 2000;
const WAR_CAL: f64 = 0.62;
const STD_RPW: f64 = 5.90;

static POS_ADJ: &[(&str, f64)] = &[
    ("C", 12.5), ("SS", 7.5), ("CF", 2.5), ("2B", 2.5), ("3B", 2.5),
    ("LF", -7.5), ("RF", -7.5), ("1B", -12.5), ("DH", -17.5),
];

// Historical RPG lookup
fn get_rpg(season: i32) -> f64 {
    match season {
        1908 => 3.37, 1912 => 4.10, 1913 => 3.93, 1915 => 3.65,
        1920 => 4.39, 1927 => 5.06, 1930 => 5.55, 1940 => 4.65,
        1950 => 4.84, 1955 => 4.44, 1960 => 4.24, 1965 => 3.89,
        1968 => 3.42, 1969 => 4.07, 1970 => 4.34, 1973 => 4.12,
        1975 => 4.12, 1980 => 4.29, 1985 => 4.33, 1987 => 4.52,
        1990 => 4.26, 1995 => 4.85, 1997 => 4.77, 1999 => 5.08,
        2000 => 5.14, 2001 => 4.78, 2004 => 4.81, 2005 => 4.59,
        2008 => 4.65, 2010 => 4.38, 2011 => 4.28, 2012 => 4.32,
        2013 => 4.17, 2014 => 4.07, 2015 => 4.25, 2016 => 4.48,
        2017 => 4.65, 2018 => 4.45, 2019 => 4.83, 2020 => 4.65,
        2021 => 4.26, 2022 => 4.28, 2023 => 4.62, 2024 => 4.52,
        2025 => 4.45,
        _ => {
            // Interpolate or default
            if season < 1908 { 3.80 }
            else if season > 2025 { 4.50 }
            else { 4.50 }
        }
    }
}

fn pos_adj_value(pos: &str) -> f64 {
    POS_ADJ.iter().find(|(p, _)| *p == pos).map(|(_, v)| *v).unwrap_or(0.0)
}

// ---- Math ----
#[inline]
fn bayesian_update(prior_mean: f64, prior_var: f64, data_mean: f64, data_var: f64, n: usize) -> (f64, f64) {
    if n == 0 { return (prior_mean, prior_var); }
    let prec_prior = 1.0 / prior_var;
    let prec_data = n as f64 / data_var;
    let post_prec = prec_prior + prec_data;
    let post_var = 1.0 / post_prec;
    let post_mean = post_var * (prec_prior * prior_mean + prec_data * data_mean);
    (post_mean, post_var)
}

fn sample_normal(rng: &mut ChaCha8Rng, mean: f64, var: f64, n: usize) -> Vec<f64> {
    let sd = var.sqrt().max(1e-10);
    let dist = Normal::new(mean, sd).unwrap();
    (0..n).map(|_| dist.sample(rng)).collect()
}

fn credible_interval(samples: &[f64], level: f64) -> (f64, f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let alpha = 1.0 - level;
    let lo = sorted[(sorted.len() as f64 * alpha / 2.0) as usize];
    let hi = sorted[(sorted.len() as f64 * (1.0 - alpha / 2.0)) as usize];
    (lo, hi)
}

fn pythagorean_rpw(rpg: f64, park_factor: f64) -> f64 {
    let rpg_adj = rpg * park_factor;
    let exp = rpg_adj.powf(0.287);
    2.0 * rpg_adj / exp
}

fn compute_woba(bb: i32, hbp: i32, s1b: i32, s2b: i32, s3b: i32, hr: i32, ab: i32, sf: i32) -> f64 {
    let denom = (ab + bb + sf + hbp) as f64;
    if denom == 0.0 { return 0.0; }
    (WOBA_W_BB * bb as f64 + WOBA_W_HBP * hbp as f64 + WOBA_W_1B * s1b as f64
     + WOBA_W_2B * s2b as f64 + WOBA_W_3B * s3b as f64 + WOBA_W_HR * hr as f64) / denom
}

fn compute_fip(hr: i32, bb: i32, hbp: i32, k: i32, ip: f64, constant: f64) -> f64 {
    if ip <= 0.0 { return constant + 5.0; }
    (FIP_HR_C * hr as f64 + FIP_BB_C * (bb + hbp) as f64 - FIP_K_C * k as f64) / ip + constant
}

// ---- Component results ----
struct CompResult {
    name: String,
    runs_mean: f64,
    runs_var: f64,
    ci90: (f64, f64),
    samples: Vec<f64>,
}

// ---- Components ----
fn comp_hitting(pa: i32, ab: i32, bb: i32, ibb: i32, hbp: i32, singles: i32,
                doubles: i32, triples: i32, hr: i32, sf: i32,
                park_factor: f64, rng: &mut ChaCha8Rng) -> CompResult {
    if pa < 1 {
        return CompResult { name: "hitting".into(), runs_mean: 0.0, runs_var: 0.0,
                            ci90: (0.0, 0.0), samples: vec![0.0; N_SAMPLES_FULL] };
    }
    let ubb = bb - ibb;
    let obs_woba = compute_woba(ubb, hbp, singles, doubles, triples, hr, ab, sf) / park_factor;
    let (post_mean, post_var) = bayesian_update(
        PRIOR_WOBA_MEAN, PRIOR_WOBA_SD * PRIOR_WOBA_SD, obs_woba, WOBA_OBS_VAR, pa.max(50) as usize);
    let fat_runs = FAT_BAT_PER_600 * (pa as f64 / 600.0);
    let runs_above_avg = (post_mean - LEAGUE_AVG_WOBA) / WOBA_SCALE * pa as f64;
    let total_mean = runs_above_avg - fat_runs;
    let total_var = (post_var / (WOBA_SCALE * WOBA_SCALE)) * (pa as f64 * pa as f64);
    let samples_woba = sample_normal(rng, post_mean, post_var, N_SAMPLES_FULL);
    let samples: Vec<f64> = samples_woba.iter()
        .map(|w| (w - LEAGUE_AVG_WOBA) / WOBA_SCALE * pa as f64 - fat_runs).collect();
    let ci90 = credible_interval(&samples, 0.90);
    CompResult { name: "hitting".into(), runs_mean: total_mean, runs_var: total_var, ci90, samples }
}

fn comp_pitching(ip: f64, er: i32, hr: i32, bb: i32, hbp: i32, k: i32,
                 gs: i32, gp: i32, park_factor: f64, league_rpg: f64,
                 rng: &mut ChaCha8Rng) -> CompResult {
    if ip < 1.0 {
        return CompResult { name: "pitching".into(), runs_mean: 0.0, runs_var: 0.0,
                            ci90: (0.0, 0.0), samples: vec![0.0; N_SAMPLES_FULL] };
    }
    let league_era = league_rpg * 0.92;
    let fip_c = league_era - PRIOR_FIP_MEAN + 3.10;
    let obs_fip = compute_fip(hr, bb, hbp, k, ip, fip_c) / park_factor;
    let eff_n = ip.max(20.0) as usize;
    let (post_mean, post_var) = bayesian_update(
        PRIOR_FIP_MEAN, PRIOR_FIP_SD * PRIOR_FIP_SD, obs_fip, FIP_OBS_VAR, eff_n);
    let fat_era = league_era + (-FAT_PITCH_PER_200 / 200.0) * 9.0;
    let total_mean = (fat_era - post_mean) / 9.0 * ip;
    let total_var = (ip / 9.0).powi(2) * post_var;
    let fip_samples = sample_normal(rng, post_mean, post_var, N_SAMPLES_FULL);
    let samples: Vec<f64> = fip_samples.iter().map(|f| (fat_era - f) / 9.0 * ip).collect();
    let ci90 = credible_interval(&samples, 0.90);
    CompResult { name: "pitching".into(), runs_mean: total_mean, runs_var: total_var, ci90, samples }
}

fn comp_baserunning(pa: i32, sb: i32, cs: i32, gidp: i32, rng: &mut ChaCha8Rng) -> CompResult {
    let sb_runs = sb as f64 * RV_SB + cs as f64 * RV_CS;
    let gidp_exp = pa as f64 * 0.15 * 0.11;
    let gidp_runs = (gidp_exp - gidp as f64) * RV_GIDP.abs();
    let raw = sb_runs + gidp_runs;
    let pa_scale = (pa as f64 / 600.0).max(0.01);
    let obs = raw / pa_scale;
    let eff_n = (pa_scale * 10.0).max(1.0) as usize;
    let (post_mean, post_var) = bayesian_update(0.0, PRIOR_BR_SD * PRIOR_BR_SD, obs, BR_OBS_VAR, eff_n);
    let total_mean = post_mean * pa_scale;
    let total_var = post_var * pa_scale * pa_scale;
    let samples = sample_normal(rng, total_mean, total_var, N_SAMPLES_FULL);
    let ci90 = credible_interval(&samples, 0.90);
    CompResult { name: "baserunning".into(), runs_mean: total_mean, runs_var: total_var, ci90, samples }
}

fn comp_fielding(inn_fielded: f64, uzr: Option<f64>, drs: Option<f64>, oaa: Option<f64>,
                 total_zone: Option<f64>, position: &str, rng: &mut ChaCha8Rng) -> CompResult {
    if position == "DH" || position == "P" || inn_fielded < 50.0 {
        let samples = sample_normal(rng, 0.0, PRIOR_FIELD_SD * PRIOR_FIELD_SD, N_SAMPLES_FULL);
        return CompResult { name: "fielding".into(), runs_mean: 0.0,
                            runs_var: PRIOR_FIELD_SD * PRIOR_FIELD_SD, ci90: (-8.2, 8.2), samples };
    }
    // Ensemble or TotalZone
    let (obs, obs_var) = if uzr.is_some() || drs.is_some() || oaa.is_some() {
        let mut sum = 0.0; let mut wt = 0.0;
        if let Some(v) = uzr { sum += 0.30 * v; wt += 0.30; }
        if let Some(v) = drs { sum += 0.30 * v; wt += 0.30; }
        if let Some(v) = oaa { sum += 0.40 * v; wt += 0.40; }
        (sum / wt, FIELD_OBS_VAR)
    } else if let Some(tz) = total_zone {
        (tz, TOTALZONE_OBS_VAR)
    } else {
        let samples = sample_normal(rng, 0.0, PRIOR_FIELD_SD * PRIOR_FIELD_SD, N_SAMPLES_FULL);
        return CompResult { name: "fielding".into(), runs_mean: 0.0,
                            runs_var: PRIOR_FIELD_SD * PRIOR_FIELD_SD, ci90: (-8.2, 8.2), samples };
    };
    let eff_n = (inn_fielded / 300.0).max(1.0) as usize;
    let (post_mean, post_var) = bayesian_update(0.0, PRIOR_FIELD_SD * PRIOR_FIELD_SD, obs, obs_var, eff_n);
    let samples = sample_normal(rng, post_mean, post_var, N_SAMPLES_FULL);
    let ci90 = credible_interval(&samples, 0.90);
    CompResult { name: "fielding".into(), runs_mean: post_mean, runs_var: post_var, ci90, samples }
}

fn comp_positional(position: &str, games: i32, rng: &mut ChaCha8Rng) -> CompResult {
    let adj = pos_adj_value(position);
    let frac = games as f64 / 162.0;
    let runs = adj * frac;
    let samples = sample_normal(rng, runs, 1.0, N_SAMPLES_FULL);
    let ci90 = credible_interval(&samples, 0.90);
    CompResult { name: "positional".into(), runs_mean: runs, runs_var: 1.0, ci90, samples }
}

fn comp_aqi(pa: i32, bb: i32, k: i32, ubb: i32, hbp: i32, singles: i32,
            doubles: i32, triples: i32, hr: i32, ab: i32, sf: i32,
            chase_rate: Option<f64>, rng: &mut ChaCha8Rng) -> CompResult {
    if pa < 100 {
        let samples = sample_normal(rng, 0.0, PRIOR_AQI_SD * PRIOR_AQI_SD, N_SAMPLES_FULL);
        return CompResult { name: "approach_quality".into(), runs_mean: 0.0,
                            runs_var: PRIOR_AQI_SD * PRIOR_AQI_SD, ci90: (-4.9, 4.9), samples };
    }
    let bb_rate = bb as f64 / pa as f64;
    let k_rate = k as f64 / pa as f64;
    let actual_woba = compute_woba(ubb, hbp, singles, doubles, triples, hr, ab, sf);
    let exp_woba = 0.310 + 0.80 * (bb_rate - 0.085) + (-0.45) * (k_rate - 0.220);
    let woba_resid = actual_woba - exp_woba;
    let mut aqi = 0.0;
    if let Some(cr) = chase_rate { aqi += -10.0 * (cr - 0.30); }
    else { aqi += 1.5 * (bb_rate - 0.085) + (-1.0) * (k_rate - 0.220); }
    aqi += -8.0 * woba_resid;
    let obs = aqi * (pa as f64 / 600.0) * 3.0;
    let obs_var = if chase_rate.is_some() { AQI_OBS_VAR } else { AQI_OBS_VAR * 2.0 };
    let pa_scale = (pa as f64 / 600.0).max(0.1);
    let eff_n = (pa_scale * 5.0).max(1.0) as usize;
    let (post_mean, post_var) = bayesian_update(0.0, PRIOR_AQI_SD * PRIOR_AQI_SD, obs, obs_var, eff_n);
    let samples = sample_normal(rng, post_mean, post_var, N_SAMPLES_FULL);
    let ci90 = credible_interval(&samples, 0.90);
    CompResult { name: "approach_quality".into(), runs_mean: post_mean, runs_var: post_var, ci90, samples }
}

fn comp_durability(games: i32, is_pitcher: bool, gs: i32, gp: i32, season_games: i32,
                   rng: &mut ChaCha8Rng) -> CompResult {
    let (expected_full, actual, marginal) = if is_pitcher {
        if gs > gp / 2 { (32, gs, GAME_MARGINAL_PIT * 2.0) }
        else { (65, gp, GAME_MARGINAL_PIT) }
    } else {
        (155, games, GAME_MARGINAL_POS)
    };
    let expected = if season_games < 162 {
        ((expected_full as f64) * (season_games as f64 * 0.95 / 162.0)).max(1.0) as i32
    } else { expected_full };
    let delta = actual - expected;
    let runs = delta as f64 * marginal * 9.8;
    let samples = sample_normal(rng, runs, 2.0, N_SAMPLES_FULL);
    let ci90 = credible_interval(&samples, 0.90);
    CompResult { name: "durability".into(), runs_mean: runs, runs_var: 2.0, ci90, samples }
}

fn comp_catcher(framing: Option<f64>, blocking: Option<f64>, throwing: Option<f64>,
                pitches: i32, rng: &mut ChaCha8Rng) -> CompResult {
    let framing_obs = framing.unwrap_or(0.0);
    let eff_n = (pitches / 500).max(1) as usize;
    let (fr_mean, fr_var) = bayesian_update(0.0, PRIOR_FRAME_SD * PRIOR_FRAME_SD,
                                             framing_obs, FRAME_OBS_VAR, eff_n);
    let bl = blocking.unwrap_or(0.0);
    let th = throwing.unwrap_or(0.0);
    let gc_var = 16.0; // game calling prior variance
    let total = fr_mean + bl + th;
    let total_var = fr_var + gc_var;
    let fr_samples = sample_normal(rng, fr_mean, fr_var, N_SAMPLES_FULL);
    let gc_samples = sample_normal(rng, 0.0, gc_var, N_SAMPLES_FULL);
    let samples: Vec<f64> = fr_samples.iter().zip(gc_samples.iter())
        .map(|(f, g)| f + bl + th + g).collect();
    let ci90 = credible_interval(&samples, 0.90);
    CompResult { name: "catcher".into(), runs_mean: total, runs_var: total_var, ci90, samples }
}

// ---- Main compute ----
#[pyfunction]
#[pyo3(signature = (pa=0, ab=0, hits=0, doubles=0, triples=0, hr=0, bb=0, ibb=0, hbp=0,
    k=0, sf=0, sb=0, cs=0, gidp=0, games=0,
    ip=0.0, er=0, hits_allowed=0, hr_allowed=0, bb_allowed=0, hbp_allowed=0,
    k_pitching=0, games_pitched=0, games_started=0, saves=0,
    inn_fielded=0.0, uzr=None, drs=None, oaa=None, total_zone=None,
    framing_runs=None, blocking_runs=None, throwing_runs=None, catcher_pitches=0,
    avg_leverage_index=1.0, position="DH", season=2024, park_factor=1.0,
    league_rpg=0.0, season_games=162, seed=42, fast=false))]
fn compute_bravs_fast(
    py: Python<'_>,
    pa: i32, ab: i32, hits: i32, doubles: i32, triples: i32, hr: i32,
    bb: i32, ibb: i32, hbp: i32, k: i32, sf: i32, sb: i32, cs: i32, gidp: i32, games: i32,
    ip: f64, er: i32, hits_allowed: i32, hr_allowed: i32, bb_allowed: i32, hbp_allowed: i32,
    k_pitching: i32, games_pitched: i32, games_started: i32, saves: i32,
    inn_fielded: f64, uzr: Option<f64>, drs: Option<f64>, oaa: Option<f64>, total_zone: Option<f64>,
    framing_runs: Option<f64>, blocking_runs: Option<f64>, throwing_runs: Option<f64>,
    catcher_pitches: i32,
    avg_leverage_index: f64, position: &str, season: i32, park_factor: f64,
    league_rpg: f64, season_games: i32, seed: u64, fast: bool,
) -> PyResult<Py<PyDict>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_samp = if fast { N_SAMPLES_FAST } else { N_SAMPLES_FULL };
    let rpg = if league_rpg > 0.0 { league_rpg } else { get_rpg(season) };
    let rpw = pythagorean_rpw(rpg, park_factor);
    let era_mult = 4.62 / rpg; // era anchor = 2023

    let singles = if hits > 0 { (hits - doubles - triples - hr).max(0) } else { 0 };
    let ubb = bb - ibb;
    let is_pitcher = ip >= 20.0;
    let is_catcher = position == "C";

    // Compute components
    let mut components: Vec<CompResult> = Vec::with_capacity(10);

    if pa >= 1 {
        let mut c = comp_hitting(pa, ab, ubb, 0, hbp, singles, doubles, triples, hr, sf, park_factor, &mut rng);
        c.runs_mean *= era_mult; c.runs_var *= era_mult * era_mult;
        for s in c.samples.iter_mut() { *s *= era_mult; }
        c.ci90 = credible_interval(&c.samples, 0.90);
        components.push(c);
    }
    if ip >= 1.0 {
        let mut c = comp_pitching(ip, er, hr_allowed, bb_allowed, hbp_allowed, k_pitching,
                                   games_started, games_pitched, park_factor, rpg, &mut rng);
        c.runs_mean *= era_mult; c.runs_var *= era_mult * era_mult;
        for s in c.samples.iter_mut() { *s *= era_mult; }
        c.ci90 = credible_interval(&c.samples, 0.90);
        components.push(c);
    }
    if pa >= 1 {
        let mut c = comp_baserunning(pa, sb, cs, gidp, &mut rng);
        c.runs_mean *= era_mult; c.runs_var *= era_mult * era_mult;
        for s in c.samples.iter_mut() { *s *= era_mult; }
        c.ci90 = credible_interval(&c.samples, 0.90);
        components.push(c);
    }
    {
        let mut c = comp_fielding(inn_fielded, uzr, drs, oaa, total_zone, position, &mut rng);
        c.runs_mean *= era_mult; c.runs_var *= era_mult * era_mult;
        for s in c.samples.iter_mut() { *s *= era_mult; }
        c.ci90 = credible_interval(&c.samples, 0.90);
        components.push(c);
    }
    if is_catcher {
        let mut c = comp_catcher(framing_runs, blocking_runs, throwing_runs, catcher_pitches, &mut rng);
        c.runs_mean *= era_mult; c.runs_var *= era_mult * era_mult;
        for s in c.samples.iter_mut() { *s *= era_mult; }
        c.ci90 = credible_interval(&c.samples, 0.90);
        components.push(c);
    }
    {
        let mut c = comp_positional(position, games, &mut rng);
        c.runs_mean *= era_mult; c.runs_var *= era_mult * era_mult;
        for s in c.samples.iter_mut() { *s *= era_mult; }
        c.ci90 = credible_interval(&c.samples, 0.90);
        components.push(c);
    }
    if pa >= 1 {
        let mut c = comp_aqi(pa, bb, k, ubb, hbp, singles, doubles, triples, hr, ab, sf, None, &mut rng);
        c.runs_mean *= era_mult; c.runs_var *= era_mult * era_mult;
        for s in c.samples.iter_mut() { *s *= era_mult; }
        c.ci90 = credible_interval(&c.samples, 0.90);
        components.push(c);
    }
    {
        let mut c = comp_durability(games, is_pitcher, games_started, games_pitched, season_games, &mut rng);
        c.runs_mean *= era_mult; c.runs_var *= era_mult * era_mult;
        for s in c.samples.iter_mut() { *s *= era_mult; }
        c.ci90 = credible_interval(&c.samples, 0.90);
        components.push(c);
    }

    // Leverage
    let lev_mult = (avg_leverage_index.max(0.01).powf(LEVERAGE_EXP)) / AVG_DAMPED_LI;
    let skill_names = ["hitting", "pitching", "baserunning", "fielding", "catcher", "approach_quality"];
    let skill_runs: f64 = components.iter()
        .filter(|c| skill_names.contains(&c.name.as_str()))
        .map(|c| c.runs_mean).sum();
    let lev_runs = skill_runs * (lev_mult - 1.0);
    let lev_samples = sample_normal(&mut rng, lev_runs, skill_runs.abs().max(1.0) * (lev_mult - 1.0).abs() * 0.1, N_SAMPLES_FULL);
    let lev_ci = credible_interval(&lev_samples, 0.90);
    components.push(CompResult {
        name: "leverage".into(), runs_mean: lev_runs, runs_var: 1.0, ci90: lev_ci, samples: lev_samples,
    });

    // Sum posterior
    let mut total_samples = vec![0.0f64; N_SAMPLES_FULL];
    for comp in &components {
        for (i, s) in comp.samples.iter().enumerate() {
            if i < N_SAMPLES_FULL { total_samples[i] += s; }
        }
    }
    let total_mean: f64 = total_samples.iter().sum::<f64>() / N_SAMPLES_FULL as f64;
    let total_var: f64 = total_samples.iter().map(|s| (s - total_mean).powi(2)).sum::<f64>() / N_SAMPLES_FULL as f64;
    let bravs = total_mean / rpw;
    let bravs_ci = credible_interval(&total_samples.iter().map(|s| s / rpw).collect::<Vec<_>>(), 0.90);

    // Build result dict
    let dict = PyDict::new(py);
    dict.set_item("bravs", (bravs * 10.0).round() / 10.0)?;
    dict.set_item("bravs_era_std", ((total_mean / STD_RPW) * 10.0).round() / 10.0)?;
    dict.set_item("bravs_war_eq", ((bravs * WAR_CAL) * 10.0).round() / 10.0)?;
    dict.set_item("ci90_lo", (bravs_ci.0 * 10.0).round() / 10.0)?;
    dict.set_item("ci90_hi", (bravs_ci.1 * 10.0).round() / 10.0)?;
    dict.set_item("total_runs", (total_mean * 10.0).round() / 10.0)?;
    dict.set_item("rpw", (rpw * 100.0).round() / 100.0)?;
    dict.set_item("leverage_mult", (lev_mult * 1000.0).round() / 1000.0)?;

    // Components list
    let comp_list: Vec<HashMap<String, f64>> = components.iter().map(|c| {
        let mut m = HashMap::new();
        m.insert("runs".into(), (c.runs_mean * 10.0).round() / 10.0);
        m.insert("ci_lo".into(), (c.ci90.0 * 10.0).round() / 10.0);
        m.insert("ci_hi".into(), (c.ci90.1 * 10.0).round() / 10.0);
        m
    }).collect();
    let comp_names: Vec<String> = components.iter().map(|c| c.name.clone()).collect();

    let comps_py = pyo3::types::PyList::empty(py);
    for (i, c) in comp_list.iter().enumerate() {
        let d = PyDict::new(py);
        d.set_item("name", &comp_names[i])?;
        d.set_item("runs", c["runs"])?;
        d.set_item("ci_lo", c["ci_lo"])?;
        d.set_item("ci_hi", c["ci_hi"])?;
        comps_py.append(d)?;
    }
    dict.set_item("components", comps_py)?;

    Ok(dict.into())
}

#[pymodule]
fn bravs_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_bravs_fast, m)?)?;
    Ok(())
}
