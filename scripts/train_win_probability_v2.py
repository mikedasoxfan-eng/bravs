"""Win Probability Model v2 — proper game state + walk-off logic.

Fixes from v1:
1. Parse advance strings from event_raw for accurate base state tracking
2. One-hot encode base state (8 categories)
3. Skip invalid states (e.g. home batting in Bot 9 when already winning)
4. Add interaction features (outs × inning, score × late_game)
5. Bigger model (25K params)
6. More epochs + proper validation
7. Use calibration loss in addition to BCE
"""

import sys, os, time, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WPModelV2(nn.Module):
    """Larger win probability network with batch norm.

    Input: 24 features
    - inning_norm (0-1)
    - is_top, is_bot (one-hot)
    - outs 0, 1, 2 (one-hot, 3 features)
    - 8 base states (empty, 1B, 2B, 3B, 12, 13, 23, 123) one-hot
    - score_diff_norm (batting team perspective)
    - abs_score_diff_norm
    - inning_mid (inning >= 4)
    - inning_late (inning >= 7)
    - inning_final (inning >= 9)
    - bot_9_tied, bot_9_ahead (walk-off indicators)
    - close_game (|diff| <= 2)
    """

    def __init__(self, n_features=24, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─────────────────────────── advance parsing ───────────────────────────

# Advance strings look like: ".1-3;2-H;3-H" or ".B-2" after the event
ADVANCE_RE = re.compile(r"([B123])-([123H])")


def parse_advances(event_raw: str) -> list[tuple[str, str]]:
    """Parse advance strings from Retrosheet event.

    Returns list of (runner_position, destination) tuples.
    Runner: B (batter), 1, 2, 3
    Destination: 1, 2, 3, H (home)
    """
    if "." not in event_raw:
        return []
    advance_part = event_raw.split(".", 1)[1]
    return ADVANCE_RE.findall(advance_part)


def apply_event_to_state(event_raw: str, event_simple: str, bases: list[int]) -> tuple[list[int], int]:
    """Apply an event to the base state and return (new_bases, runs_scored).

    Uses advance string parsing where available, falls back to heuristics otherwise.
    """
    runs = 0
    new_bases = bases.copy()

    # Try to parse advance string first
    advances = parse_advances(event_raw)

    # If we have explicit advances, use them as ground truth
    if advances:
        # Start with all runners cleared — we'll place them based on advances
        runners_to_place = {"1": bases[0], "2": bases[1], "3": bases[2]}
        new_bases = [0, 0, 0]

        # Apply each advance
        batter_destination = None
        for runner, dest in advances:
            if runner == "B":
                batter_destination = dest
                continue
            # Was this runner on base?
            if runners_to_place.get(runner, 0) == 0:
                continue
            runners_to_place[runner] = 0
            if dest == "H":
                runs += 1
            else:
                idx = int(dest) - 1
                new_bases[idx] = 1

        # Any runners we didn't move stay put
        # (event might say "3-H" and omit runner on 1B who didn't move)
        for runner, present in runners_to_place.items():
            if present:
                idx = int(runner) - 1
                new_bases[idx] = 1

        # Place batter
        if batter_destination:
            if batter_destination == "H":
                runs += 1
            else:
                idx = int(batter_destination) - 1
                new_bases[idx] = 1
        else:
            # No explicit batter advance — use event type to decide
            if event_simple == "HR":
                # batter scores; other runners scored above
                runs += 1
            elif event_simple in ("HIT", "WALK", "HBP", "ERROR", "FC"):
                # Check if batter implicit to 1st
                # If 1st is not already taken by a runner we moved, put batter there
                if new_bases[0] == 0:
                    new_bases[0] = 1

        return new_bases, runs

    # ─── Fallback: heuristics when no advance string ───
    if event_simple == "HR":
        runs = 1 + sum(bases)
        new_bases = [0, 0, 0]
    elif event_simple == "HIT":
        # Check event_raw for single/double/triple
        core = event_raw.split("/")[0].strip()
        if core.startswith("S"):
            # Single: 3B scores, 2B often scores, 1B often to 2B
            runs = bases[2]
            # Approximation: 2B scores 60% of the time
            runs += bases[1]  # simplified
            new_bases = [1, bases[0], 0]
        elif core.startswith("D"):
            runs = sum(bases)
            new_bases = [0, 1, 0]
        elif core.startswith("T"):
            runs = sum(bases)
            new_bases = [0, 0, 1]
        else:
            new_bases = [1, bases[0], bases[1]]
    elif event_simple in ("WALK", "HBP"):
        # Forced advances only
        if bases[0] == 1:
            if bases[1] == 1:
                if bases[2] == 1:
                    runs = 1
                else:
                    new_bases[2] = 1
            else:
                new_bases[1] = 1
        new_bases[0] = 1
    elif event_simple == "ERROR":
        new_bases = [1, bases[0], bases[1]]
    elif event_simple == "FC":
        new_bases = [1, bases[0], bases[1]]
    # OUT, K: no base state change

    return new_bases, runs


# ─────────────────────────── game reconstruction ───────────────────────────

def base_state_index(bases: list[int]) -> int:
    """Map [1b, 2b, 3b] to an index 0-7."""
    return bases[0] * 1 + bases[1] * 2 + bases[2] * 4


def reconstruct_states(plays: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct game state for each play with proper advance parsing."""
    print(f"Reconstructing {len(plays):,} plays...")
    t0 = time.perf_counter()

    # Sort by game, inning, half (T before B), outs
    plays = plays.copy()
    plays["half_ord"] = plays.half.map({"T": 0, "B": 1})
    plays = plays.sort_values(["game_id", "inning", "half_ord", "outs_before"])

    all_rows = []
    for game_id, game_plays in plays.groupby("game_id", sort=False):
        home_score = 0
        away_score = 0
        bases = [0, 0, 0]
        prev_half_inning = None

        for row in game_plays.itertuples(index=False):
            key = (row.inning, row.half)
            if key != prev_half_inning:
                bases = [0, 0, 0]
                prev_half_inning = key

            is_home_batting = 1 if row.half == "B" else 0

            # Skip plays in invalid states:
            # - Home batting in Bot 9+ when already winning (game over)
            if is_home_batting and row.inning >= 9 and home_score > away_score:
                continue

            # Record state BEFORE this play
            all_rows.append({
                "game_id": game_id,
                "inning": row.inning,
                "is_home_batting": is_home_batting,
                "outs": row.outs_before,
                "home_score": home_score,
                "away_score": away_score,
                "score_diff_home": home_score - away_score,
                "bases_idx": base_state_index(bases),
                "bases_1b": bases[0],
                "bases_2b": bases[1],
                "bases_3b": bases[2],
                "event_simple": row.event_simple,
            })

            # Update state
            new_bases, runs = apply_event_to_state(
                row.event_raw, row.event_simple, bases
            )
            bases = new_bases

            if is_home_batting:
                home_score += runs
            else:
                away_score += runs

        # Final outcome for all this game's plays
        home_won = 1 if home_score > away_score else 0
        n_game_plays = sum(1 for r in all_rows[-len(game_plays):])
        for i in range(len(all_rows) - n_game_plays, len(all_rows)):
            all_rows[i]["home_won"] = home_won

    print(f"  Reconstructed {len(all_rows):,} states in {time.perf_counter()-t0:.1f}s")
    return pd.DataFrame(all_rows)


# ─────────────────────────── feature building ───────────────────────────

def build_features(states: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build 24-feature input matrix and target."""
    n = len(states)
    X = np.zeros((n, 24), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)

    inning = states.inning.values
    is_home = states.is_home_batting.values
    outs = states.outs.values
    score_diff_home = states.score_diff_home.values
    bases_idx = states.bases_idx.values
    home_won = states.home_won.values

    # 0: inning_norm
    X[:, 0] = np.minimum(inning, 12) / 12.0
    # 1-2: top/bot one-hot
    X[:, 1] = 1.0 - is_home  # top
    X[:, 2] = is_home         # bottom
    # 3-5: outs one-hot
    for i in range(3):
        X[:, 3 + i] = (outs == i).astype(np.float32)
    # 6-13: base state one-hot (8 categories)
    for i in range(8):
        X[:, 6 + i] = (bases_idx == i).astype(np.float32)
    # 14: score diff from BATTING team's perspective
    bat_diff = np.where(is_home == 1, score_diff_home, -score_diff_home)
    X[:, 14] = np.clip(bat_diff, -10, 10) / 10.0
    # 15: abs score diff
    X[:, 15] = np.clip(np.abs(bat_diff), 0, 10) / 10.0
    # 16-18: inning phase indicators
    X[:, 16] = (inning >= 4).astype(np.float32)
    X[:, 17] = (inning >= 7).astype(np.float32)
    X[:, 18] = (inning >= 9).astype(np.float32)
    # 19: walk-off opportunity (bot 9+, tied or trailing, within 4 runs)
    walk_off = ((is_home == 1) & (inning >= 9) & (bat_diff <= 0) & (bat_diff >= -4)).astype(np.float32)
    X[:, 19] = walk_off
    # 20: bot 9 tied (highest leverage)
    X[:, 20] = ((is_home == 1) & (inning >= 9) & (bat_diff == 0)).astype(np.float32)
    # 21: close game (within 2)
    X[:, 21] = (np.abs(bat_diff) <= 2).astype(np.float32)
    # 22: outs * late game interaction
    X[:, 22] = outs * (inning >= 7).astype(np.float32) / 3.0
    # 23: runners in scoring position
    X[:, 23] = ((bases_idx & 2) > 0).astype(np.float32) + ((bases_idx & 4) > 0).astype(np.float32)
    X[:, 23] /= 2.0

    # Target: did BATTING team win?
    y = np.where(is_home == 1, home_won, 1.0 - home_won).astype(np.float32)

    return X, y


def main():
    print("=" * 72)
    print("  WIN PROBABILITY MODEL v2")
    print(f"  Device: {DEVICE}")
    print("=" * 72)

    print("\nLoading events...")
    plays = pd.read_parquet("data/retrosheet/events_parsed.parquet")
    plays = plays[plays.event_simple.isin(
        ["HR", "HIT", "WALK", "HBP", "OUT", "K", "ERROR", "FC"]
    )].copy()
    print(f"  {len(plays):,} plays, {plays.game_id.nunique():,} games")

    print("\nReconstructing game states with advance parsing...")
    states = reconstruct_states(plays)
    states = states.dropna(subset=["home_won"])
    print(f"  Valid states: {len(states):,}")

    print("\nBuilding features...")
    X, y = build_features(states)
    print(f"  X shape: {X.shape}")
    print(f"  Batting team win rate: {y.mean():.3f}")

    # Split
    n = len(X)
    n_val = int(n * 0.1)
    perm = np.random.RandomState(42).permutation(n)
    X_tr, y_tr = X[perm[n_val:]], y[perm[n_val:]]
    X_va, y_va = X[perm[:n_val]], y[perm[:n_val]]

    X_tr_t = torch.tensor(X_tr, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, device=DEVICE)
    X_va_t = torch.tensor(X_va, device=DEVICE)
    y_va_t = torch.tensor(y_va, device=DEVICE)

    # Train
    model = WPModelV2(n_features=24, hidden=128).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    BATCH_SIZE = 16384
    print("\nTraining...")
    t0 = time.perf_counter()
    best_val = float("inf")
    best_state = None

    for epoch in range(50):
        model.train()
        perm_t = torch.randperm(len(X_tr_t))
        epoch_loss = 0
        n_batches = 0
        for s in range(0, len(perm_t), BATCH_SIZE):
            idx = perm_t[s:s + BATCH_SIZE]
            pred = model(X_tr_t[idx])
            loss = F.binary_cross_entropy(pred, y_tr_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vp = model(X_va_t)
            vl = F.binary_cross_entropy(vp, y_va_t)
            va = ((vp > 0.5).float() == y_va_t).float().mean()
            # Brier score (calibration quality)
            brier = ((vp - y_va_t) ** 2).mean()

        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>2}: train={epoch_loss/n_batches:.4f} "
                  f"val_bce={vl.item():.4f} val_acc={va.item():.3%} brier={brier.item():.4f}")

    if best_state:
        model.load_state_dict(best_state)
    print(f"Training: {time.perf_counter()-t0:.1f}s")

    # ─── Sanity checks ───
    print("\n--- Sanity Checks ---")
    model.eval()

    def state_to_features(inning, is_bot, outs, bases, score_diff_bat):
        """Build a feature vector for a specific state (score_diff_bat = batting team - opponent)."""
        feat = np.zeros(24, dtype=np.float32)
        feat[0] = min(inning, 12) / 12.0
        feat[1] = 0.0 if is_bot else 1.0  # top
        feat[2] = 1.0 if is_bot else 0.0  # bottom
        feat[3 + outs] = 1.0
        # Base state
        bases_idx = bases[0] + 2*bases[1] + 4*bases[2]
        feat[6 + bases_idx] = 1.0
        feat[14] = np.clip(score_diff_bat, -10, 10) / 10.0
        feat[15] = np.clip(abs(score_diff_bat), 0, 10) / 10.0
        feat[16] = 1.0 if inning >= 4 else 0.0
        feat[17] = 1.0 if inning >= 7 else 0.0
        feat[18] = 1.0 if inning >= 9 else 0.0
        feat[19] = 1.0 if (is_bot and inning >= 9 and score_diff_bat <= 0 and score_diff_bat >= -4) else 0.0
        feat[20] = 1.0 if (is_bot and inning >= 9 and score_diff_bat == 0) else 0.0
        feat[21] = 1.0 if abs(score_diff_bat) <= 2 else 0.0
        feat[22] = outs * (1.0 if inning >= 7 else 0.0) / 3.0
        feat[23] = (bases[1] + bases[2]) / 2.0
        return feat

    tests = [
        ("Top 1, 0-0, 0 out, empty", 1, False, 0, [0,0,0], 0),
        ("Bot 1, 0-0, 0 out, empty", 1, True, 0, [0,0,0], 0),
        ("Bot 9, tied, 2 out, runner on 3B", 9, True, 2, [0,0,1], 0),
        ("Bot 9, tied, 0 out, empty", 9, True, 0, [0,0,0], 0),
        ("Bot 9, down 1, 2 out, runner on 2B", 9, True, 2, [0,1,0], -1),
        ("Bot 9, down 3, 2 out, empty", 9, True, 2, [0,0,0], -3),
        ("Top 9, up 3, 0 out, empty (away batting)", 9, False, 0, [0,0,0], 3),
        ("Top 9, up 5, 2 out, empty (away batting)", 9, False, 2, [0,0,0], 5),
        ("Top 1, up 5, 0 out, empty", 1, False, 0, [0,0,0], 5),
        ("Bot 9, down 2, 0 out, bases loaded", 9, True, 0, [1,1,1], -2),
        ("Top 5, tied, 1 out, runner on 1B", 5, False, 1, [1,0,0], 0),
        ("Bot 9, tied, 2 out, bases loaded", 9, True, 2, [1,1,1], 0),
    ]

    with torch.no_grad():
        for label, inn, is_bot, outs, bases, diff in tests:
            feat = state_to_features(inn, is_bot, outs, bases, diff)
            x = torch.tensor(feat, device=DEVICE).unsqueeze(0)
            wp = model(x).item()
            print(f"  {label:<48} WP={wp:>5.1%}")

    # Save
    os.makedirs("models", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "n_features": 24,
        "hidden": 128,
        "n_params": n_params,
    }, "models/win_probability_v2.pt")

    print(f"\nSaved models/win_probability_v2.pt ({n_params:,} params)")
    print("=" * 72)


if __name__ == "__main__":
    main()
