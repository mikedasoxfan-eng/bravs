"""Compute BRAVS for every qualified player-season in MLB history using GPU.

RTX 5060 Ti: processes 80,000+ player-seasons in seconds, not hours.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import torch

from baseball_metric.data import lahman
from baseball_metric.core.gpu_engine_v2 import batch_compute_bravs_v2 as batch_compute_bravs, DEVICE


def main():
    t_start = time.perf_counter()
    os.makedirs("data", exist_ok=True)

    print("=" * 80)
    print("  COMPUTING BRAVS FOR EVERY PLAYER-SEASON IN MLB HISTORY")
    print(f"  Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    bat = lahman._batting()
    pit = lahman._pitching()
    app = lahman._appearances()
    ppl = lahman._people()

    # Build a lookup for names
    name_map = dict(zip(ppl.playerID, ppl.nameFirst + " " + ppl.nameLast))

    # Get all player-seasons with meaningful playing time (1920-2025)
    bat_season = bat[bat.yearID >= 1920].groupby(["playerID", "yearID"]).agg({
        "AB": "sum", "H": "sum", "2B": "sum", "3B": "sum", "HR": "sum",
        "BB": "sum", "SO": "sum", "SB": "sum", "CS": "sum", "G": "sum",
        "IBB": "sum", "HBP": "sum", "SF": "sum", "SH": "sum", "GIDP": "sum",
        "teamID": "last", "lgID": "last",
    }).reset_index()
    bat_season["PA"] = bat_season.AB + bat_season.BB + bat_season.HBP.fillna(0) + bat_season.SF.fillna(0) + bat_season.SH.fillna(0)
    bat_season = bat_season[bat_season.PA >= 50]

    pit_season = pit[pit.yearID >= 1920].groupby(["playerID", "yearID"]).agg({
        "IPouts": "sum", "ER": "sum", "H": "sum", "HR": "sum",
        "BB": "sum", "SO": "sum", "G": "sum", "GS": "sum", "SV": "sum",
        "HBP": "sum", "teamID": "last", "lgID": "last",
    }).reset_index()
    pit_season["IP"] = pit_season.IPouts / 3.0
    pit_season = pit_season[pit_season.IP >= 20]

    # Get positions from appearances table
    pos_lookup = {}
    pos_cols = {"G_c": "C", "G_1b": "1B", "G_2b": "2B", "G_3b": "3B",
                "G_ss": "SS", "G_lf": "LF", "G_cf": "CF", "G_rf": "RF",
                "G_dh": "DH", "G_p": "P"}
    for _, row in app.iterrows():
        pid = row.playerID
        yr = row.yearID
        best_pos = "DH"
        best_g = 0
        for col, pos in pos_cols.items():
            v = row.get(col, 0)
            g = int(v) if pd.notna(v) else 0
            if g > best_g:
                best_g = g
                best_pos = pos
        pos_lookup[(pid, yr)] = best_pos

    # Compute crude fielding runs from Lahman fielding table
    # Method: compare range factor (PO+A per inning) and error rate to position average
    fld = pd.read_csv(lahman.DATA_DIR / "Fielding.csv")
    if "InnOuts" in fld.columns:
        fld["Inn"] = fld.InnOuts / 3.0
    else:
        fld["Inn"] = fld.G * 9.0  # fallback: estimate innings from games

    fld["RF_per_inn"] = (fld.PO.fillna(0) + fld.A.fillna(0)) / fld.Inn.clip(lower=1)
    fld["E_per_inn"] = fld.E.fillna(0) / fld.Inn.clip(lower=1)

    # Position-year averages
    pos_avg = fld[fld.Inn >= 100].groupby(["yearID", "POS"]).agg(
        avg_rf=("RF_per_inn", "mean"),
        avg_e=("E_per_inn", "mean"),
    ).reset_index()

    # Merge to get above-average fielding
    fld_merged = fld.merge(pos_avg, on=["yearID", "POS"], how="left")
    fld_merged["rf_above_avg"] = (fld_merged.RF_per_inn - fld_merged.avg_rf.fillna(0)) * fld_merged.Inn
    fld_merged["e_above_avg"] = -(fld_merged.E_per_inn - fld_merged.avg_e.fillna(0)) * fld_merged.Inn  # negative errors = good

    # Convert to run values — very conservative to avoid positional bias
    # Range factor above avg overstates value at high-putout positions (1B, C, CF)
    # Use 0.3 runs per play above avg (vs the naive 0.8)
    fld_merged["fielding_rf_runs"] = fld_merged.rf_above_avg * 0.3
    fld_merged["fielding_e_runs"] = fld_merged.e_above_avg * 0.5

    # Aggregate per player-year (sum across positions)
    fld_agg = fld_merged.groupby(["playerID", "yearID"]).agg(
        fielding_rf=("fielding_rf_runs", "sum"),
        fielding_e=("fielding_e_runs", "sum"),
    ).reset_index()
    fld_lookup = {(r.playerID, int(r.yearID)): (r.fielding_rf, r.fielding_e) for _, r in fld_agg.iterrows()}

    print(f"  Fielding estimates computed for {len(fld_lookup):,} player-seasons")

    # Merge batting and pitching into unified player-season records
    all_records = {}

    for _, r in bat_season.iterrows():
        key = (r.playerID, int(r.yearID))
        pos = pos_lookup.get(key, "DH")
        season_games = {2020: 60, 1994: 115, 1995: 144, 1981: 107}.get(int(r.yearID), 162)
        all_records[key] = {
            "playerID": r.playerID, "yearID": int(r.yearID),
            "name": name_map.get(r.playerID, r.playerID),
            "team": r.teamID, "lgID": str(r.get("lgID", "")), "position": pos,
            "PA": int(r.PA), "AB": int(r.AB), "H": int(r.H),
            "2B": int(r["2B"]), "3B": int(r["3B"]), "HR": int(r.HR),
            "BB": int(r.BB), "IBB": int(r.get("IBB", 0) or 0),
            "HBP": int(r.get("HBP", 0) or 0), "SO": int(r.SO),
            "SF": int(r.get("SF", 0) or 0), "SH": int(r.get("SH", 0) or 0),
            "SB": int(r.SB), "CS": int(r.get("CS", 0) or 0),
            "GIDP": int(r.get("GIDP", 0) or 0), "G": int(r.G),
            "IP": 0, "ER": 0, "H_allowed": 0, "HR_allowed": 0,
            "BB_allowed": 0, "HBP_allowed": 0, "K_pitch": 0,
            "G_pitched": 0, "GS": 0, "SV": 0,
            "park_factor": 1.0, "season_games": season_games,
            "fielding_rf": 0.0, "fielding_e": 0.0,
        }
        # Add fielding data
        fld_key = (r.playerID, int(r.yearID))
        if fld_key in fld_lookup:
            all_records[key]["fielding_rf"] = round(fld_lookup[fld_key][0], 1)
            all_records[key]["fielding_e"] = round(fld_lookup[fld_key][1], 1)

    for _, r in pit_season.iterrows():
        key = (r.playerID, int(r.yearID))
        season_games = {2020: 60, 1994: 115, 1995: 144, 1981: 107}.get(int(r.yearID), 162)
        if key in all_records:
            # Two-way player: add pitching to existing record
            all_records[key].update({
                "IP": round(float(r.IP), 1), "ER": int(r.ER),
                "H_allowed": int(r.H), "HR_allowed": int(r.HR),
                "BB_allowed": int(r.BB), "HBP_allowed": int(r.get("HBP", 0) or 0),
                "K_pitch": int(r.SO), "G_pitched": int(r.G),
                "GS": int(r.GS), "SV": int(r.get("SV", 0) or 0),
            })
        else:
            pos = pos_lookup.get(key, "P")
            all_records[key] = {
                "playerID": r.playerID, "yearID": int(r.yearID),
                "name": name_map.get(r.playerID, r.playerID),
                "team": r.teamID, "lgID": str(r.get("lgID", "")), "position": pos,
                "PA": 0, "AB": 0, "H": 0, "2B": 0, "3B": 0, "HR": 0,
                "BB": 0, "IBB": 0, "HBP": 0, "SO": 0, "SF": 0, "SH": 0,
                "SB": 0, "CS": 0, "GIDP": 0, "G": int(r.G),
                "IP": round(float(r.IP), 1), "ER": int(r.ER),
                "H_allowed": int(r.H), "HR_allowed": int(r.HR),
                "BB_allowed": int(r.BB), "HBP_allowed": int(r.get("HBP", 0) or 0),
                "K_pitch": int(r.SO), "G_pitched": int(r.G),
                "GS": int(r.GS), "SV": int(r.get("SV", 0) or 0),
                "park_factor": 1.0, "season_games": season_games,
                "fielding_rf": 0.0, "fielding_e": 0.0,
            }
            fld_key = (r.playerID, int(r.yearID))
            if fld_key in fld_lookup:
                all_records[(r.playerID, int(r.yearID))]["fielding_rf"] = round(fld_lookup[fld_key][0], 1)
                all_records[(r.playerID, int(r.yearID))]["fielding_e"] = round(fld_lookup[fld_key][1], 1)

    player_data = list(all_records.values())
    total = len(player_data)

    print(f"\n  Player-seasons to compute: {total:,}")
    print(f"  Year range: 1920-2025")
    print(f"\n  Launching GPU computation...")

    t_gpu = time.perf_counter()
    results = batch_compute_bravs(player_data, n_samples=2000, seed=42)
    t_done = time.perf_counter()

    gpu_time = t_done - t_gpu
    print(f"\n  GPU computation: {gpu_time:.2f}s for {total:,} player-seasons")
    print(f"  Rate: {total / gpu_time:,.0f} player-seasons/second")
    print(f"  Per player: {gpu_time / total * 1_000_000:.0f} microseconds")

    # Save results
    df = pd.DataFrame(results)

    df.to_csv("data/bravs_all_seasons.csv", index=False)
    print(f"\n  Saved: data/bravs_all_seasons.csv ({len(df):,} rows)")

    df_bat = df[df.PA >= 100].copy()
    df_bat.to_csv("data/bravs_all_batting.csv", index=False)
    print(f"  Saved: data/bravs_all_batting.csv ({len(df_bat):,} rows)")

    df_pit = df[df.IP >= 30].copy()
    df_pit.to_csv("data/bravs_all_pitching.csv", index=False)
    print(f"  Saved: data/bravs_all_pitching.csv ({len(df_pit):,} rows)")

    # Career totals
    careers = df.groupby("playerID").agg(
        name=("name", "last"),
        seasons=("yearID", "count"),
        first_year=("yearID", "min"),
        last_year=("yearID", "max"),
        total_G=("G", "sum"),
        total_PA=("PA", "sum"),
        total_HR=("HR", "sum"),
        total_IP=("IP", "sum"),
        career_bravs=("bravs", "sum"),
        career_era_std=("bravs_era_std", "sum"),
        career_war_eq=("bravs_war_eq", "sum"),
        peak_bravs=("bravs", "max"),
    ).reset_index()

    peak5 = df.groupby("playerID").bravs.apply(
        lambda x: x.nlargest(5).sum()
    ).reset_index().rename(columns={"bravs": "peak5_bravs"})
    careers = careers.merge(peak5, on="playerID", how="left")

    hof_ids = set(lahman.get_hof_inducted())
    careers["hof"] = careers.playerID.isin(hof_ids)
    careers = careers.round(2)
    careers.to_csv("data/bravs_careers.csv", index=False)
    print(f"  Saved: data/bravs_careers.csv ({len(careers):,} rows)")

    # Print leaderboards
    print(f"\n{'=' * 80}")
    print(f"  TOP 25 CAREERS BY WAR-EQUIVALENT")
    print(f"{'=' * 80}")
    top = careers.nlargest(25, "career_war_eq")
    print(f"  {'Rank':<5}{'Player':<25}{'Years':>12}{'WAR-eq':>8}{'BRAVS':>8}{'Peak':>7}{'HOF':>5}")
    print(f"  {'-' * 72}")
    for i, (_, r) in enumerate(top.iterrows(), 1):
        h = "Y" if r.hof else ""
        print(f"  {i:<5}{r['name']:<25}{r.first_year:.0f}-{r.last_year:.0f}"
              f"{r.career_war_eq:>8.1f}{r.career_bravs:>8.1f}{r.peak_bravs:>7.1f}{h:>5}")

    print(f"\n{'=' * 80}")
    print(f"  TOP 25 SINGLE SEASONS")
    print(f"{'=' * 80}")
    top_s = df.nlargest(25, "bravs_war_eq")
    print(f"  {'Rank':<5}{'Player':<22}{'Year':>5}{'Pos':<5}{'WAR-eq':>8}{'BRAVS':>8}")
    print(f"  {'-' * 55}")
    for i, (_, r) in enumerate(top_s.iterrows(), 1):
        print(f"  {i:<5}{r['name']:<22}{r.yearID:>5.0f} {r.position:<4}{r.bravs_war_eq:>8.1f}{r.bravs:>8.1f}")

    total_time = time.perf_counter() - t_start
    print(f"\n  Total runtime: {total_time:.1f}s")
    print(f"  GPU compute: {gpu_time:.2f}s")
    print(f"  Data prep: {t_gpu - t_start:.1f}s")


if __name__ == "__main__":
    main()
