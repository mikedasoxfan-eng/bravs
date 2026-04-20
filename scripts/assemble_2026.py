"""2026 Season CSV Assembler — runs headlessly at 2 AM EST daily.

Fetches current 2026 stats from MLB Stats API, computes BRAVS via GPU,
and outputs CSVs in the same format as the Lahman-based historical data.

Memory efficient: processes in chunks, clears intermediate DataFrames,
uses minimal GPU memory (2000 samples, float32).

Output files:
  data/2026/batting_2026.csv     — raw batting stats (Lahman format)
  data/2026/pitching_2026.csv    — raw pitching stats (Lahman format)
  data/2026/bravs_2026.csv       — computed BRAVS (same format as bravs_all_seasons.csv)
  data/2026/careers_2026.csv     — career totals including 2026

Setup (run once):
  python scripts/assemble_2026.py --setup-cron

Manual run:
  python scripts/assemble_2026.py
"""

import sys, os, json, time, logging, argparse, subprocess, platform
from datetime import datetime, date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
import pandas as pd
import numpy as np

_log_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "logs")
os.makedirs(_log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(_log_dir, "assemble_2026.log"), mode="a"),
    ],
)
log = logging.getLogger(__name__)

# Always run from the project root, regardless of how the script is invoked
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)

MLB_API = "https://statsapi.mlb.com/api/v1"
YEAR = 2026
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "2026")
CHUNK_SIZE = 200  # players per GPU batch to limit memory


def _get(url, params=None, timeout=15):
    """Fetch from MLB API with retry."""
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 2:
                log.warning("API failed after 3 attempts: %s %s", url, e)
                return None
            time.sleep(2 ** attempt)


def _safe(val, default=0):
    try:
        return int(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _parse_ip(val):
    if not val:
        return 0.0
    s = str(val)
    if "." in s:
        parts = s.split(".")
        return int(parts[0]) + int(parts[1] or 0) / 3.0
    return float(s)


def fetch_team_games():
    """Get number of games each team has played so far."""
    data = _get(f"{MLB_API}/standings", {"leagueId": "103,104", "season": YEAR})
    team_games = {}
    if data and "records" in data:
        for rec in data["records"]:
            for team in rec.get("teamRecords", []):
                abbrev = team.get("team", {}).get("name", "")[:3]
                team_id = team.get("team", {}).get("id", 0)
                g = _safe(team.get("gamesPlayed"))
                team_games[team_id] = g
    return team_games


def fetch_all_stats():
    """Fetch batting + pitching stats for all 2026 players."""
    log.info("Fetching 2026 batting stats...")
    batting_rows = []
    pitching_rows = []

    # Get all teams
    teams_data = _get(f"{MLB_API}/teams", {"season": YEAR, "sportId": 1})
    if not teams_data or "teams" not in teams_data:
        log.error("Failed to fetch teams")
        return [], []

    team_ids = [t["id"] for t in teams_data["teams"]]
    team_games = fetch_team_games()

    for tid in team_ids:
        # Fetch roster
        roster = _get(f"{MLB_API}/teams/{tid}/roster",
                      {"season": YEAR, "rosterType": "fullSeason"})
        if not roster or "roster" not in roster:
            continue

        player_ids = [p["person"]["id"] for p in roster["roster"]]
        team_abbrev = next(
            (t.get("abbreviation", "UNK") for t in teams_data["teams"] if t["id"] == tid),
            "UNK"
        )
        season_games = team_games.get(tid, 162)

        for pid in player_ids:
            # Hitting stats
            h_data = _get(f"{MLB_API}/people/{pid}/stats",
                          {"stats": "season", "season": YEAR, "group": "hitting", "gameType": "R"})
            if h_data and "stats" in h_data:
                for g in h_data["stats"]:
                    for split in g.get("splits", []):
                        if _safe(split.get("numTeams", 1)) > 1 or len(g["splits"]) == 1:
                            s = split.get("stat", {})
                            if _safe(s.get("plateAppearances")) < 1:
                                continue
                            batting_rows.append({
                                "playerID": str(pid),
                                "yearID": YEAR,
                                "teamID": team_abbrev,
                                "lgID": split.get("league", {}).get("name", "")[:2].upper() or "AL",
                                "G": _safe(s.get("gamesPlayed")),
                                "AB": _safe(s.get("atBats")),
                                "R": _safe(s.get("runs")),
                                "H": _safe(s.get("hits")),
                                "2B": _safe(s.get("doubles")),
                                "3B": _safe(s.get("triples")),
                                "HR": _safe(s.get("homeRuns")),
                                "RBI": _safe(s.get("rbi")),
                                "SB": _safe(s.get("stolenBases")),
                                "CS": _safe(s.get("caughtStealing")),
                                "BB": _safe(s.get("baseOnBalls")),
                                "SO": _safe(s.get("strikeOuts")),
                                "IBB": _safe(s.get("intentionalWalks")),
                                "HBP": _safe(s.get("hitByPitch")),
                                "SH": _safe(s.get("sacBunts")),
                                "SF": _safe(s.get("sacFlies")),
                                "GIDP": _safe(s.get("groundIntoDoublePlay")),
                                "PA": _safe(s.get("plateAppearances")),
                                "season_games": season_games,
                                "player_name": split.get("player", {}).get("fullName",
                                               f"Player {pid}"),
                            })
                            break  # take first valid split

            # Pitching stats
            p_data = _get(f"{MLB_API}/people/{pid}/stats",
                          {"stats": "season", "season": YEAR, "group": "pitching", "gameType": "R"})
            if p_data and "stats" in p_data:
                for g in p_data["stats"]:
                    for split in g.get("splits", []):
                        if _safe(split.get("numTeams", 1)) > 1 or len(g["splits"]) == 1:
                            s = split.get("stat", {})
                            ip = _parse_ip(s.get("inningsPitched", 0))
                            if ip < 1:
                                continue
                            pitching_rows.append({
                                "playerID": str(pid),
                                "yearID": YEAR,
                                "teamID": team_abbrev,
                                "lgID": split.get("league", {}).get("name", "")[:2].upper() or "AL",
                                "G": _safe(s.get("gamesPlayed")),
                                "GS": _safe(s.get("gamesStarted")),
                                "IP": round(ip, 1),
                                "H": _safe(s.get("hits")),
                                "ER": _safe(s.get("earnedRuns")),
                                "HR": _safe(s.get("homeRuns")),
                                "BB": _safe(s.get("baseOnBalls")),
                                "SO": _safe(s.get("strikeOuts")),
                                "HBP": _safe(s.get("hitBatsmen")),
                                "SV": _safe(s.get("saves")),
                                "season_games": season_games,
                                "player_name": split.get("player", {}).get("fullName",
                                               f"Player {pid}"),
                            })
                            break

        log.info("  Team %s: %d batters, %d pitchers fetched",
                 team_abbrev, len([r for r in batting_rows if r["teamID"] == team_abbrev]),
                 len([r for r in pitching_rows if r["teamID"] == team_abbrev]))

        # Be polite to the API
        time.sleep(0.5)

    return batting_rows, pitching_rows


def compute_bravs_batch(player_data):
    """Compute BRAVS via GPU in memory-efficient chunks."""
    try:
        from baseball_metric.core.gpu_engine_v3 import batch_compute_bravs_v3, DEVICE
        log.info("Using GPU engine on %s", DEVICE)
    except ImportError:
        log.warning("GPU engine not available, using CPU fallback")
        # Fallback: compute one at a time with Python engine
        from baseball_metric.core.model import compute_bravs
        from baseball_metric.core.types import PlayerSeason
        results = []
        for d in player_data:
            ps = PlayerSeason(
                player_id=d["playerID"], player_name=d.get("name", ""),
                season=YEAR, team=d.get("team", ""), position=d.get("position", "DH"),
                pa=d.get("PA", 0), ab=d.get("AB", 0), hits=d.get("H", 0),
                doubles=d.get("2B", 0), triples=d.get("3B", 0), hr=d.get("HR", 0),
                bb=d.get("BB", 0), ibb=d.get("IBB", 0), hbp=d.get("HBP", 0),
                k=d.get("SO", 0), sf=d.get("SF", 0), sb=d.get("SB", 0),
                cs=d.get("CS", 0), gidp=d.get("GIDP", 0), games=d.get("G", 0),
                ip=d.get("IP", 0), er=d.get("ER", 0),
                hr_allowed=d.get("HR_allowed", 0), bb_allowed=d.get("BB_allowed", 0),
                k_pitching=d.get("K_pitch", 0), games_pitched=d.get("G_pitched", 0),
                games_started=d.get("GS", 0), saves=d.get("SV", 0),
                park_factor=1.0, league_rpg=4.45,
                season_games=d.get("season_games", 162),
            )
            r = compute_bravs(ps, fast=True)
            results.append({
                "playerID": d["playerID"], "yearID": YEAR,
                "name": d.get("name", ""), "team": d.get("team", ""),
                "lgID": d.get("lgID", ""), "position": d.get("position", ""),
                "bravs": round(r.bravs, 2), "bravs_war_eq": round(r.bravs_calibrated, 2),
            })
        return results

    # GPU path: process in chunks
    all_results = []
    for i in range(0, len(player_data), CHUNK_SIZE):
        chunk = player_data[i:i + CHUNK_SIZE]
        chunk_results = batch_compute_bravs_v3(chunk, n_samples=2000, seed=42)
        all_results.extend(chunk_results)
        # Free GPU memory between chunks
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


def assemble():
    """Main assembly: fetch, compute, save."""
    t0 = time.perf_counter()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    log.info("=== 2026 CSV Assembly started at %s ===", datetime.now().isoformat())

    # Fetch all stats
    batting_rows, pitching_rows = fetch_all_stats()
    log.info("Fetched %d batting rows, %d pitching rows", len(batting_rows), len(pitching_rows))

    if not batting_rows and not pitching_rows:
        log.error("No data fetched, aborting")
        return

    # Save raw stats in Lahman format
    bat_df = pd.DataFrame(batting_rows)
    bat_df.to_csv(f"{OUT_DIR}/batting_2026.csv", index=False, encoding="utf-8-sig")
    log.info("Saved %s (%d rows)", f"{OUT_DIR}/batting_2026.csv", len(bat_df))

    pit_df = pd.DataFrame(pitching_rows)
    pit_df.to_csv(f"{OUT_DIR}/pitching_2026.csv", index=False, encoding="utf-8-sig")
    log.info("Saved %s (%d rows)", f"{OUT_DIR}/pitching_2026.csv", len(pit_df))

    # Build GPU-compatible player data
    # Merge batting + pitching for two-way players
    player_data = []
    seen = set()

    for _, r in bat_df.iterrows():
        pid = r.playerID
        key = pid
        if key in seen:
            continue
        seen.add(key)

        # Check for pitching stats
        pit_match = pit_df[pit_df.playerID == pid]
        ip = float(pit_match.IP.iloc[0]) if not pit_match.empty else 0.0

        player_data.append({
            "playerID": pid, "yearID": YEAR,
            "name": r.get("player_name", ""), "team": r.teamID,
            "lgID": r.lgID, "position": "DH",  # will be overridden if fielding data exists
            "PA": int(r.PA), "AB": int(r.AB), "H": int(r.H),
            "2B": int(r["2B"]), "3B": int(r["3B"]), "HR": int(r.HR),
            "BB": int(r.BB), "IBB": int(r.get("IBB", 0)),
            "HBP": int(r.get("HBP", 0)), "SO": int(r.SO),
            "SF": int(r.get("SF", 0)), "SH": int(r.get("SH", 0)),
            "SB": int(r.SB), "CS": int(r.get("CS", 0)),
            "GIDP": int(r.get("GIDP", 0)), "G": int(r.G),
            "IP": ip,
            "ER": int(pit_match.ER.iloc[0]) if not pit_match.empty else 0,
            "H_allowed": int(pit_match.H.iloc[0]) if not pit_match.empty else 0,
            "HR_allowed": int(pit_match.HR.iloc[0]) if not pit_match.empty else 0,
            "BB_allowed": int(pit_match.BB.iloc[0]) if not pit_match.empty else 0,
            "HBP_allowed": int(pit_match.HBP.iloc[0]) if not pit_match.empty else 0,
            "K_pitch": int(pit_match.SO.iloc[0]) if not pit_match.empty else 0,
            "G_pitched": int(pit_match.G.iloc[0]) if not pit_match.empty else 0,
            "GS": int(pit_match.GS.iloc[0]) if not pit_match.empty else 0,
            "SV": int(pit_match.SV.iloc[0]) if not pit_match.empty else 0,
            "park_factor": 1.0,
            "season_games": int(r.get("season_games", 162)),
            "fielding_rf": 0, "fielding_e": 0, "gold_glove": 0, "all_star": 0,
        })

    # Add pitcher-only players
    for _, r in pit_df.iterrows():
        if r.playerID in seen:
            continue
        seen.add(r.playerID)
        player_data.append({
            "playerID": r.playerID, "yearID": YEAR,
            "name": r.get("player_name", ""), "team": r.teamID,
            "lgID": r.lgID, "position": "P",
            "PA": 0, "AB": 0, "H": 0, "2B": 0, "3B": 0, "HR": 0,
            "BB": 0, "IBB": 0, "HBP": 0, "SO": 0, "SF": 0, "SH": 0,
            "SB": 0, "CS": 0, "GIDP": 0, "G": int(r.G),
            "IP": float(r.IP), "ER": int(r.ER), "H_allowed": int(r.H),
            "HR_allowed": int(r.HR), "BB_allowed": int(r.BB),
            "HBP_allowed": int(r.get("HBP", 0)),
            "K_pitch": int(r.SO), "G_pitched": int(r.G),
            "GS": int(r.GS), "SV": int(r.get("SV", 0)),
            "park_factor": 1.0,
            "season_games": int(r.get("season_games", 162)),
            "fielding_rf": 0, "fielding_e": 0, "gold_glove": 0, "all_star": 0,
        })

    log.info("Built %d player records for GPU computation", len(player_data))

    # Compute BRAVS
    results = compute_bravs_batch(player_data)
    log.info("Computed BRAVS for %d players", len(results))

    # Save BRAVS CSV
    bravs_df = pd.DataFrame(results)
    bravs_df.to_csv(f"{OUT_DIR}/bravs_2026.csv", index=False, encoding="utf-8-sig")
    log.info("Saved %s (%d rows)", f"{OUT_DIR}/bravs_2026.csv", len(bravs_df))

    # Merge with historical careers
    try:
        hist = pd.read_csv("data/bravs_careers.csv")
        # Update or add 2026 data
        for _, r in bravs_df.iterrows():
            pid = r.get("playerID", "")
            match = hist[hist.playerID == pid]
            if not match.empty:
                idx = match.index[0]
                hist.at[idx, "career_bravs"] = hist.at[idx, "career_bravs"] + r.get("bravs", 0)
                hist.at[idx, "career_war_eq"] = hist.at[idx, "career_war_eq"] + r.get("bravs_war_eq", 0)
                hist.at[idx, "last_year"] = YEAR
                hist.at[idx, "seasons"] = hist.at[idx, "seasons"] + 1
                if r.get("bravs", 0) > hist.at[idx, "peak_bravs"]:
                    hist.at[idx, "peak_bravs"] = r.get("bravs", 0)
        hist.to_csv(f"{OUT_DIR}/careers_2026.csv", index=False, encoding="utf-8-sig")
        log.info("Saved %s with 2026 updates", f"{OUT_DIR}/careers_2026.csv")
    except Exception as e:
        log.warning("Could not update careers: %s", e)

    elapsed = time.perf_counter() - t0
    log.info("=== Assembly complete in %.0fs (%d players) ===", elapsed, len(results))


def setup_cron():
    """Set up a 2 AM EST daily cron job (or Windows Task Scheduler)."""
    script_path = os.path.abspath(__file__)
    python_path = sys.executable

    if platform.system() == "Windows":
        # Windows Task Scheduler
        task_name = "BRAVS_2026_Assembly"
        cmd = (
            f'schtasks /create /tn "{task_name}" /tr '
            f'"{python_path} {script_path}" '
            f'/sc daily /st 02:00 /f'
        )
        log.info("Creating Windows scheduled task...")
        log.info("Command: %s", cmd)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            log.info("Task created: %s runs daily at 2:00 AM", task_name)
        else:
            log.error("Failed to create task: %s", result.stderr)
            log.info("Run manually: schtasks /create /tn \"%s\" /tr \"%s %s\" /sc daily /st 02:00 /f",
                     task_name, python_path, script_path)
    else:
        # Linux/Mac cron
        cron_line = f"0 2 * * * cd {os.path.dirname(script_path)}/.. && PYTHONPATH=. {python_path} {script_path} >> logs/assemble_2026.log 2>&1"
        log.info("Add this to crontab (crontab -e):")
        log.info("  %s", cron_line)


def main():
    parser = argparse.ArgumentParser(description="2026 Season CSV Assembler")
    parser.add_argument("--setup-cron", action="store_true", help="Set up daily 2 AM scheduled task")
    parser.add_argument("--dry-run", action="store_true", help="Fetch data but don't compute BRAVS")
    args = parser.parse_args()

    if args.setup_cron:
        setup_cron()
    else:
        assemble()


if __name__ == "__main__":
    main()
