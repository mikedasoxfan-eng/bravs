"""Compute BRAVS for all MiLB player-seasons (2005-2026).

Adapts the MLB GPU engine for minor league levels with:
- Level adjustment factors (AAA closer to MLB, Rk furthest)
- Adjusted run environments per level
- MiLB-specific positional adjustments
- Translation rates from each level to MLB

Output: data/bravs_milb_seasons.csv in same format as bravs_all_seasons.csv
"""

import sys, os, logging, time, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════
# Level translation rates — how much of MiLB performance translates to MLB
# Based on "The Book" and Tom Tango's research on MiLB translations
# ═══════════════════════════════════════════════════════════════════
LEVEL_TRANSLATION = {
    "AAA": 0.75,   # 75% of AAA stats translate to MLB
    "AA":  0.60,   # 60% — AA is "the real test"
    "A+":  0.50,   # High-A
    "A":   0.42,   # Single-A
    "A-":  0.35,   # Short-season A / Low-A
    "Rk":  0.25,   # Rookie ball
    "WIN": 0.55,   # Winter leagues (mature but lower competition)
}

# Run environment adjustment per level (relative to MLB ≈ 4.5 R/G)
LEVEL_RPG = {
    "AAA": 5.10,   # AAA hitter-friendly (smaller parks, thinner pitching)
    "AA":  4.40,   # AA closest to MLB run environment
    "A+":  4.60,   # High-A slightly inflated
    "A":   4.50,   # Single-A varies widely
    "A-":  4.30,   # Lower-A
    "Rk":  4.80,   # Rookie ball — lots of scoring
    "WIN": 3.80,   # Winter leagues — pitcher-friendly, shorter seasons
}

MLB_RPG = 4.50  # baseline MLB runs per game

# wOBA weights (standard)
W_BB = 0.69
W_HBP = 0.72
W_1B = 0.88
W_2B = 1.24
W_3B = 1.56
W_HR = 2.01
LG_WOBA = 0.315
WOBA_SCALE = 1.25


def load_all_milb_batting() -> pd.DataFrame:
    """Load and consolidate all MiLB batting CSVs."""
    log.info("Loading MiLB batting data...")
    files = sorted(glob.glob("data/milb/batting/*.csv"))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                dfs.append(df)
        except Exception:
            pass
    batting = pd.concat(dfs, ignore_index=True)
    log.info("  Loaded %d batting records from %d files", len(batting), len(dfs))
    return batting


def load_all_milb_pitching() -> pd.DataFrame:
    """Load and consolidate all MiLB pitching CSVs."""
    log.info("Loading MiLB pitching data...")
    files = sorted(glob.glob("data/milb/pitching/*.csv"))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                dfs.append(df)
        except Exception:
            pass
    pitching = pd.concat(dfs, ignore_index=True)
    log.info("  Loaded %d pitching records from %d files", len(pitching), len(dfs))
    return pitching


def compute_milb_bravs_batting(batting: pd.DataFrame) -> pd.DataFrame:
    """Compute BRAVS for MiLB batters on GPU."""
    log.info("Computing MiLB batting BRAVS on %s...", DEVICE)
    t0 = time.perf_counter()

    n = len(batting)
    if n == 0:
        return pd.DataFrame()

    # Extract needed columns (with safe fallbacks)
    def _col(name, default=0):
        if name in batting.columns:
            return batting[name].fillna(default).values.astype(np.float32)
        return np.full(n, default, dtype=np.float32)

    season = _col("season")
    pa = _col("batting_PA")
    ab = _col("batting_AB")
    h = _col("batting_H")
    doubles = _col("batting_2B")
    triples = _col("batting_3B")
    hr = _col("batting_HR")
    bb = _col("batting_BB")
    hbp = _col("batting_HBP")
    sf = _col("batting_SF")
    sb = _col("batting_SB")
    cs = _col("batting_CS")
    so = _col("batting_SO")
    g = _col("G")

    singles = h - doubles - triples - hr

    # Level info
    level_col = batting.get("team_level_abv", pd.Series(["AAA"] * n))
    level_str = level_col.fillna("AAA").astype(str).str.upper().str.strip().values

    # Map level to translation factor and RPG
    translation = np.array([LEVEL_TRANSLATION.get(l, 0.5) for l in level_str], dtype=np.float32)
    rpg = np.array([LEVEL_RPG.get(l, 4.5) for l in level_str], dtype=np.float32)

    # Move to GPU
    t_pa = torch.tensor(pa, device=DEVICE)
    t_ab = torch.tensor(ab, device=DEVICE)
    t_singles = torch.tensor(singles, device=DEVICE)
    t_doubles = torch.tensor(doubles, device=DEVICE)
    t_triples = torch.tensor(triples, device=DEVICE)
    t_hr = torch.tensor(hr, device=DEVICE)
    t_bb = torch.tensor(bb, device=DEVICE)
    t_hbp = torch.tensor(hbp, device=DEVICE)
    t_sf = torch.tensor(sf, device=DEVICE)
    t_sb = torch.tensor(sb, device=DEVICE)
    t_cs = torch.tensor(cs, device=DEVICE)
    t_g = torch.tensor(g, device=DEVICE)
    t_trans = torch.tensor(translation, device=DEVICE)
    t_rpg = torch.tensor(rpg, device=DEVICE)

    # ─── wOBA ───
    denom = t_ab + t_bb + t_sf + t_hbp
    denom = denom.clamp(min=1)
    woba = (W_BB * t_bb + W_HBP * t_hbp + W_1B * t_singles +
            W_2B * t_doubles + W_3B * t_triples + W_HR * t_hr) / denom

    # ─── Run environment adjustment ───
    # Scale wOBA to the level's run environment vs MLB
    rpg_adj = MLB_RPG / t_rpg.clamp(min=3.0)
    woba_adj = woba * rpg_adj

    # ─── Hitting runs (wRAA equivalent) ───
    runs_per_pa = (woba_adj - LG_WOBA) / WOBA_SCALE
    hitting_runs_raw = runs_per_pa * t_pa

    # Apply level translation
    hitting_runs = hitting_runs_raw * t_trans

    # ─── Baserunning ───
    br_runs = (t_sb * 0.2 - t_cs * 0.4) * t_trans

    # ─── Positional adjustment (simplified for MiLB) ───
    # Use position from data
    pos_col = batting.get("player_position", pd.Series(["DH"] * n))
    pos_str = pos_col.fillna("DH").astype(str).values

    pos_adj_map = {"C": 12.5, "SS": 7.5, "CF": 2.5, "2B": 3.0, "3B": 2.0,
                   "LF": -7.0, "RF": -7.0, "1B": -12.5, "DH": -17.5}

    # Handle multi-position strings (e.g., "3B/2B")
    def _get_pos_adj(p):
        primary = p.split("/")[0].strip().upper()
        return pos_adj_map.get(primary, 0.0)

    pos_adj = np.array([_get_pos_adj(p) for p in pos_str], dtype=np.float32)
    t_pos_adj = torch.tensor(pos_adj, device=DEVICE)

    positional_runs = t_pos_adj * t_g / 162.0

    # ─── Fielding estimate (rough, MiLB has no detailed fielding data) ───
    fielding_runs = positional_runs * 0.3  # weak proxy

    # ─── Total BRAVS ───
    total_runs = hitting_runs + br_runs + positional_runs + fielding_runs

    # ─── WAR-eq ───
    # Dynamic RPW based on level run environment
    rpw = t_rpg * 2.0 / 2.0  # simplified: RPW ≈ RPG
    rpw = rpw.clamp(min=8.0, max=12.0)
    war_eq = total_runs / rpw

    elapsed = time.perf_counter() - t0
    log.info("  Computed %d batting BRAVS in %.2fs on %s", n, elapsed, DEVICE)

    # Build output DataFrame
    result = pd.DataFrame({
        "playerID": batting.get("player_id", pd.Series(range(n))).values,
        "yearID": season.astype(int),
        "name": batting.get("player_full_name", pd.Series(["?"] * n)).values,
        "team": batting.get("team_abv", pd.Series(["?"] * n)).values,
        "team_name": batting.get("team_name", pd.Series(["?"] * n)).values,
        "lgID": batting.get("team_league", pd.Series(["?"] * n)).values,
        "level": level_str,
        "position": pos_str,
        "G": g.astype(int),
        "PA": pa.astype(int),
        "AB": ab.astype(int),
        "H": h.astype(int),
        "HR": hr.astype(int),
        "SB": sb.astype(int),
        "CS": cs.astype(int),
        "BB": bb.astype(int),
        "SO": so.astype(int),
        "wOBA": woba.cpu().numpy().round(3),
        "wOBA_adj": woba_adj.cpu().numpy().round(3),
        "hitting_runs": hitting_runs.cpu().numpy().round(1),
        "baserunning_runs": br_runs.cpu().numpy().round(1),
        "fielding_runs": fielding_runs.cpu().numpy().round(1),
        "positional_runs": positional_runs.cpu().numpy().round(1),
        "total_runs": total_runs.cpu().numpy().round(1),
        "bravs_war_eq": war_eq.cpu().numpy().round(2),
        "translation_rate": translation,
        "level_rpg": rpg,
    })

    return result


def compute_milb_bravs_pitching(pitching: pd.DataFrame) -> pd.DataFrame:
    """Compute BRAVS for MiLB pitchers on GPU."""
    log.info("Computing MiLB pitching BRAVS on %s...", DEVICE)
    t0 = time.perf_counter()

    n = len(pitching)
    if n == 0:
        return pd.DataFrame()

    def _col(name, default=0):
        if name in pitching.columns:
            return pitching[name].fillna(default).values.astype(np.float32)
        return np.full(n, default, dtype=np.float32)

    season = _col("season")
    g = _col("G")
    gs = _col("pitching_GS")
    ip = _col("pitching_IP")
    h = _col("pitching_H")
    er = _col("pitching_ER")
    hr = _col("pitching_HR")
    bb = _col("pitching_BB")
    so = _col("pitching_SO")
    hbp = _col("pitching_HBP")

    # Level info
    level_col = pitching.get("team_level_abv", pd.Series(["AAA"] * n))
    level_str = level_col.fillna("AAA").astype(str).str.upper().str.strip().values
    translation = np.array([LEVEL_TRANSLATION.get(l, 0.5) for l in level_str], dtype=np.float32)
    rpg = np.array([LEVEL_RPG.get(l, 4.5) for l in level_str], dtype=np.float32)

    # Move to GPU
    t_ip = torch.tensor(ip, device=DEVICE).clamp(min=0.1)
    t_hr = torch.tensor(hr, device=DEVICE)
    t_bb = torch.tensor(bb, device=DEVICE)
    t_hbp = torch.tensor(hbp, device=DEVICE)
    t_so = torch.tensor(so, device=DEVICE)
    t_er = torch.tensor(er, device=DEVICE)
    t_trans = torch.tensor(translation, device=DEVICE)
    t_rpg = torch.tensor(rpg, device=DEVICE)

    # ─── FIP ───
    fip_constant = 3.10
    fip = (13.0 * t_hr + 3.0 * (t_bb + t_hbp) - 2.0 * t_so) / t_ip + fip_constant

    # ─── ERA ───
    era = t_er * 9.0 / t_ip

    # ─── Run environment adjustment ───
    rpg_adj = t_rpg / MLB_RPG  # level inflator
    fip_adj = fip / rpg_adj    # deflate FIP to MLB scale

    # ─── Pitching runs above average ───
    lg_era = torch.tensor(4.50, device=DEVICE)
    replacement_era = torch.tensor(5.50, device=DEVICE)

    # Runs above replacement, translated to MLB
    pitching_runs_raw = (replacement_era - fip_adj) / 9.0 * t_ip
    pitching_runs = pitching_runs_raw * t_trans

    # ─── WAR-eq ───
    rpw = t_rpg.clamp(min=8.0, max=12.0)
    war_eq = pitching_runs / 10.0  # ~10 runs per win

    elapsed = time.perf_counter() - t0
    log.info("  Computed %d pitching BRAVS in %.2fs on %s", n, elapsed, DEVICE)

    result = pd.DataFrame({
        "playerID": pitching.get("player_id", pd.Series(range(n))).values,
        "yearID": season.astype(int),
        "name": pitching.get("player_full_name", pd.Series(["?"] * n)).values,
        "team": pitching.get("team_abv", pd.Series(["?"] * n)).values,
        "team_name": pitching.get("team_name", pd.Series(["?"] * n)).values,
        "lgID": pitching.get("team_league", pd.Series(["?"] * n)).values,
        "level": level_str,
        "position": "P",
        "G": g.astype(int),
        "GS": gs.astype(int),
        "IP": ip.round(1),
        "H": h.astype(int),
        "ER": er.astype(int),
        "HR": hr.astype(int),
        "BB": bb.astype(int),
        "SO": so.astype(int),
        "HBP": hbp.astype(int),
        "ERA": era.cpu().numpy().round(2),
        "FIP": fip.cpu().numpy().round(2),
        "FIP_adj": fip_adj.cpu().numpy().round(2),
        "pitching_runs": pitching_runs.cpu().numpy().round(1),
        "bravs_war_eq": war_eq.cpu().numpy().round(2),
        "translation_rate": translation,
        "level_rpg": rpg,
    })

    return result


def main():
    print("=" * 70)
    print("  BRAVS MiLB Engine — Computing BRAVS for all minor leaguers")
    print("=" * 70)

    # Load data
    batting = load_all_milb_batting()
    pitching = load_all_milb_pitching()

    # Filter to players with meaningful playing time
    batting = batting[batting["batting_PA"] >= 20].copy()
    pitching_qualified = pitching[
        (pitching.get("pitching_IP", pd.Series()) >= 5) |
        (pitching.get("G", pd.Series()) >= 3)
    ].copy()

    log.info("After filtering: %d batters, %d pitchers", len(batting), len(pitching_qualified))

    # Compute BRAVS
    bat_bravs = compute_milb_bravs_batting(batting)
    pit_bravs = compute_milb_bravs_pitching(pitching_qualified)

    # Save
    os.makedirs("data/milb", exist_ok=True)
    bat_bravs.to_csv("data/milb/bravs_milb_batting.csv", index=False)
    pit_bravs.to_csv("data/milb/bravs_milb_pitching.csv", index=False)

    # Combined seasons file
    # For batters, add placeholder pitching columns
    bat_bravs["IP"] = 0
    bat_bravs["ERA"] = 0
    bat_bravs["FIP"] = 0
    bat_bravs["pitching_runs"] = 0

    # For pitchers, add placeholder batting columns
    pit_bravs["PA"] = 0
    pit_bravs["wOBA"] = 0
    pit_bravs["hitting_runs"] = 0
    pit_bravs["baserunning_runs"] = 0
    pit_bravs["fielding_runs"] = 0
    pit_bravs["positional_runs"] = 0

    # Combine and save
    common_cols = ["playerID", "yearID", "name", "team", "team_name", "lgID",
                   "level", "position", "G", "bravs_war_eq", "translation_rate"]
    all_seasons = pd.concat([
        bat_bravs[common_cols + ["PA", "HR", "SB", "wOBA", "hitting_runs",
                                  "baserunning_runs", "fielding_runs", "positional_runs"]],
        pit_bravs[common_cols + ["IP", "ERA", "FIP", "SO", "BB", "pitching_runs"]],
    ], ignore_index=True)
    all_seasons = all_seasons.sort_values(["yearID", "bravs_war_eq"], ascending=[True, False])
    all_seasons.to_csv("data/bravs_milb_seasons.csv", index=False)

    # Summary stats
    print(f"\n  Results:")
    print(f"    Batters:  {len(bat_bravs):,} player-seasons")
    print(f"    Pitchers: {len(pit_bravs):,} player-seasons")
    print(f"    Total:    {len(all_seasons):,} player-seasons")
    print(f"    Year range: {int(all_seasons.yearID.min())}-{int(all_seasons.yearID.max())}")

    # Top MiLB seasons ever
    print(f"\n  Top 15 MiLB Batting Seasons (by translated BRAVS WAR-eq):")
    top_bat = bat_bravs.nlargest(15, "bravs_war_eq")
    for _, r in top_bat.iterrows():
        print(f"    {r['name']:<22} {int(r['yearID'])} {r['level']:<4} {r['team']:<5} "
              f"WAR={r['bravs_war_eq']:+.1f}  wOBA={r['wOBA']:.3f}  PA={int(r['PA'])}")

    print(f"\n  Top 15 MiLB Pitching Seasons:")
    top_pit = pit_bravs.nlargest(15, "bravs_war_eq")
    for _, r in top_pit.iterrows():
        print(f"    {r['name']:<22} {int(r['yearID'])} {r['level']:<4} {r['team']:<5} "
              f"WAR={r['bravs_war_eq']:+.1f}  FIP={r['FIP']:.2f}  IP={r['IP']:.0f}")

    # Level averages
    print(f"\n  Average WAR-eq by Level (batters, PA >= 100):")
    qualified_bat = bat_bravs[bat_bravs.PA >= 100]
    for level in ["AAA", "AA", "A+", "A", "A-", "RK"]:
        level_data = qualified_bat[qualified_bat.level == level]
        if len(level_data) > 0:
            print(f"    {level:<4}: avg={level_data.bravs_war_eq.mean():.2f}, "
                  f"median={level_data.bravs_war_eq.median():.2f}, "
                  f"n={len(level_data):,}")

    # Saved files
    bat_size = os.path.getsize("data/milb/bravs_milb_batting.csv") / 1024 / 1024
    pit_size = os.path.getsize("data/milb/bravs_milb_pitching.csv") / 1024 / 1024
    all_size = os.path.getsize("data/bravs_milb_seasons.csv") / 1024 / 1024
    print(f"\n  Saved:")
    print(f"    data/milb/bravs_milb_batting.csv   ({bat_size:.1f} MB)")
    print(f"    data/milb/bravs_milb_pitching.csv  ({pit_size:.1f} MB)")
    print(f"    data/bravs_milb_seasons.csv        ({all_size:.1f} MB)")

    print("\n" + "=" * 70)
    print("  MiLB BRAVS computation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
