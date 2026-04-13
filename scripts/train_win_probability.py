"""Win Probability Model — trained on 1.72M Retrosheet plays (2015-2024).

Reconstructs game state (inning, score differential, outs, base state)
for each plate appearance, then trains a neural net to predict P(home team wins)
given the state.

This is the foundational model for:
- Live WPA during games
- Leverage-weighted player values
- Manager decision evaluation (pulling a pitcher with 2 outs and 1 base, etc.)
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WPModel(nn.Module):
    """Win probability network.

    Input features:
    - inning (1-12+), normalized
    - half (top=0, bottom=1)
    - outs (0-2)
    - score_diff (home - away), clamped
    - bases loaded state (7-dim one-hot: empty, 1st, 2nd, 3rd, 12, 13, 23, 123)
    Total: 12 features.
    """

    def __init__(self, n_features=12, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def reconstruct_game_states(plays: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct game state (score, base state, final home win flag) for each play.

    Since we don't have direct score/base state in parsed data, we approximate:
    - Score: accumulate runs based on event outcomes + explicit advance tokens
    - Base state: reset per half-inning, updated by events
    - Final outcome: compute from last play of game

    For simplicity, we use a RUN ESTIMATE per event rather than parsing advances.
    This is an approximation — better would be full event-string parsing.
    """
    print(f"Reconstructing game states for {len(plays):,} plays...")

    # Sort by game, inning, half, outs_before
    plays = plays.sort_values(["game_id", "inning", "half", "outs_before"]).copy()

    # Simple run-per-event estimates (avg run value from Tango's linear weights)
    # These are approximate since we're not parsing advance strings
    RUN_VALUE = {
        "HR": 1.40,   # Always scores 1+
        "HIT": 0.47,  # Average of 1B/2B/3B
        "WALK": 0.33,
        "HBP": 0.34,
        "OUT": -0.28,
        "K": -0.30,
        "FC": -0.25,
        "ERROR": 0.50,
        "OTHER": 0.0,
    }

    # Parse HR_SCORE: HR events explicitly score runs in the event string
    # e.g. "HR/9" scores 1 (+ any baserunners, but we'll estimate)
    # For simplicity, estimate runs from event_raw using a heuristic

    # Group by game to process each game independently
    game_groups = list(plays.groupby("game_id"))
    print(f"  {len(game_groups)} unique games")

    all_states = []
    for game_id, game_plays in game_groups:
        game_plays = game_plays.sort_values(["inning", "half", "outs_before"]).reset_index(drop=True)

        home_score = 0
        away_score = 0
        bases = [0, 0, 0]  # 1st, 2nd, 3rd
        prev_half_inning = None

        for _, play in game_plays.iterrows():
            key = (play.inning, play.half)
            if key != prev_half_inning:
                bases = [0, 0, 0]
                prev_half_inning = key

            # State BEFORE this play (what model uses to predict)
            is_home_batting = 1 if play.half == "B" else 0
            score_diff_home = home_score - away_score  # From home's perspective
            bases_state = tuple(bases)

            all_states.append({
                "game_id": game_id,
                "inning": play.inning,
                "half": play.half,
                "is_home_batting": is_home_batting,
                "outs": play.outs_before,
                "home_score": home_score,
                "away_score": away_score,
                "score_diff": home_score - away_score,
                "bases": str(bases_state),
                "bases_1b": bases[0],
                "bases_2b": bases[1],
                "bases_3b": bases[2],
                "event_simple": play.event_simple,
            })

            # Update state based on event (approximate)
            ev = str(play.event_raw)
            event_simple = play.event_simple

            # Estimate runs scored on this play
            runs = 0

            # HR always scores at least the batter + any on base
            if event_simple == "HR":
                runs = 1 + sum(bases)
                bases = [0, 0, 0]
            elif event_simple == "HIT":
                # Parse to determine bases: S=1B, D=2B, T=3B from event_raw
                core = ev.split("/")[0].split(".")[0]
                if core.startswith("S"):
                    # Single: runners on 2B/3B score, 1B -> 2B usually
                    runs = bases[1] + bases[2]  # 2B + 3B score
                    bases = [1, bases[0], 0]
                elif core.startswith("D"):
                    # Double: 1B, 2B, 3B all score
                    runs = sum(bases)
                    bases = [0, 1, 0]
                elif core.startswith("T"):
                    # Triple: everyone scores
                    runs = sum(bases)
                    bases = [0, 0, 1]
                else:
                    bases = [1, bases[0], bases[1]]
            elif event_simple == "WALK":
                # Walk: forces runners only if 1B occupied
                if bases[0] == 1:
                    if bases[1] == 1:
                        if bases[2] == 1:
                            runs = 1  # Bases loaded walk
                        else:
                            bases[2] = 1
                    else:
                        bases[1] = 1
                bases[0] = 1
            elif event_simple == "HBP":
                # Same as walk
                if bases[0] == 1:
                    if bases[1] == 1:
                        if bases[2] == 1:
                            runs = 1
                        else:
                            bases[2] = 1
                    else:
                        bases[1] = 1
                bases[0] = 1
            elif event_simple == "ERROR":
                # Treat like a single
                runs = bases[2]
                bases = [1, bases[0], bases[1]]
            elif event_simple == "FC":
                # Fielder's choice - batter safe, lead runner out
                if bases[2] == 1:
                    pass  # runner stayed or scored
                # Simplify: just shift
                bases = [1, bases[0], bases[1]]
            # OUT, K: no run change, bases mostly same (ignore productive outs)

            # Add runs to scoring team
            if is_home_batting:
                home_score += runs
            else:
                away_score += runs

        # Mark final outcome (did home win?)
        home_won = 1 if home_score > away_score else 0

        # Add home_won to all states for this game
        for i in range(len(all_states) - len(game_plays), len(all_states)):
            all_states[i]["home_won"] = home_won

    states_df = pd.DataFrame(all_states)
    print(f"  Reconstructed {len(states_df):,} states")
    return states_df


def main():
    print("=" * 72)
    print("  WIN PROBABILITY MODEL")
    print(f"  Training on {DEVICE}")
    print("=" * 72)

    print("\nLoading parsed events...")
    plays = pd.read_parquet("data/retrosheet/events_parsed.parquet")
    print(f"  {len(plays):,} plays, {plays.game_id.nunique():,} games")

    # Only reconstruct games that have valid data
    plays = plays[plays.event_simple.isin(
        ["HR", "HIT", "WALK", "HBP", "OUT", "K", "ERROR", "FC"]
    )].copy()

    print("\nReconstructing game states...")
    t0 = time.perf_counter()
    states = reconstruct_game_states(plays)
    print(f"  Took {time.perf_counter()-t0:.1f}s")

    # Drop games where home_won wasn't set
    states = states.dropna(subset=["home_won"])
    print(f"  Valid states: {len(states):,}")

    # Build feature matrix
    print("\nBuilding features...")

    # Features: inning_norm, is_top, is_bot, outs_0, outs_1, outs_2,
    # score_diff_norm (from BATTING team), bases 1-hot (8 states)
    X = np.zeros((len(states), 12), dtype=np.float32)
    y = np.zeros(len(states), dtype=np.float32)

    for i, r in enumerate(states.itertuples()):
        X[i, 0] = min(r.inning, 12) / 12.0
        X[i, 1] = 1.0 if r.half == "T" else 0.0  # top
        X[i, 2] = 1.0 if r.half == "B" else 0.0  # bottom

        # Outs one-hot
        X[i, 3 + r.outs] = 1.0

        # Score differential FROM THE BATTING TEAM'S perspective
        # If top (away bats): score_diff from away = away - home = -r.score_diff
        # If bottom (home bats): from home = r.score_diff
        if r.half == "T":
            bat_diff = -r.score_diff
        else:
            bat_diff = r.score_diff
        X[i, 6] = np.clip(bat_diff, -10, 10) / 10.0

        # Bases
        X[i, 7] = r.bases_1b
        X[i, 8] = r.bases_2b
        X[i, 9] = r.bases_3b

        # Late-game indicator
        X[i, 10] = 1.0 if r.inning >= 7 else 0.0
        X[i, 11] = 1.0 if r.inning >= 9 else 0.0

        # Target: did batting team win?
        if r.half == "T":
            y[i] = 1.0 - r.home_won  # away team
        else:
            y[i] = r.home_won  # home team

    print(f"  X shape: {X.shape}")
    print(f"  Batting team win rate (base rate): {y.mean():.3f}")

    # Split
    n = len(X)
    n_val = int(n * 0.1)
    perm = np.random.RandomState(42).permutation(n)
    X_train, y_train = X[perm[n_val:]], y[perm[n_val:]]
    X_val, y_val = X[perm[:n_val]], y[perm[:n_val]]

    X_train_t = torch.tensor(X_train, device=DEVICE)
    y_train_t = torch.tensor(y_train, device=DEVICE)
    X_val_t = torch.tensor(X_val, device=DEVICE)
    y_val_t = torch.tensor(y_val, device=DEVICE)

    # Train
    model = WPModel(n_features=12, hidden=64).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    BATCH_SIZE = 8192
    print("\nTraining...")
    t0 = time.perf_counter()

    for epoch in range(30):
        model.train()
        perm_t = torch.randperm(len(X_train_t))
        epoch_loss = 0
        n_batches = 0
        for s in range(0, len(perm_t), BATCH_SIZE):
            idx = perm_t[s:s + BATCH_SIZE]
            pred = model(X_train_t[idx])
            loss = F.binary_cross_entropy(pred, y_train_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = F.binary_cross_entropy(val_pred, y_val_t)
            val_acc = ((val_pred > 0.5).float() == y_val_t).float().mean()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>2}: train={epoch_loss/n_batches:.4f} "
                  f"val={val_loss.item():.4f} val_acc={val_acc:.3%}")

    print(f"Training: {time.perf_counter()-t0:.1f}s")

    # Final eval — key situation lookups
    print("\n--- Sanity Check: Key Game States ---")
    test_states = [
        ("Top 1, 0-0, 0 out, bases empty", [1/12, 1, 0, 1, 0, 0, 0.0, 0, 0, 0, 0, 0]),
        ("Bot 9, tied, 2 out, runner on 3", [9/12, 0, 1, 0, 0, 1, 0.0, 0, 0, 1, 1, 1]),
        ("Top 9, away up 3, 0 out, empty", [9/12, 1, 0, 1, 0, 0, 0.3, 0, 0, 0, 1, 1]),
        ("Bot 9, home down 1, 2 out, empty", [9/12, 0, 1, 0, 0, 1, -0.1, 0, 0, 0, 1, 1]),
        ("Bot 9, home down 3, 2 out, empty", [9/12, 0, 1, 0, 0, 1, -0.3, 0, 0, 0, 1, 1]),
        ("Bot 9, home up 1, 2 out, runner on 2", [9/12, 0, 1, 0, 0, 1, 0.1, 0, 1, 0, 1, 1]),
        ("Top 1, away up 5, 0 out, empty", [1/12, 1, 0, 1, 0, 0, 0.5, 0, 0, 0, 0, 0]),
    ]

    model.eval()
    with torch.no_grad():
        for label, features in test_states:
            x = torch.tensor([features], dtype=torch.float32, device=DEVICE)
            wp = model(x).item()
            print(f"  {label:<45} WP(batting team): {wp:.1%}")

    # Save
    os.makedirs("models", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "n_features": 12,
        "n_params": n_params,
    }, "models/win_probability.pt")

    print(f"\nSaved models/win_probability.pt ({n_params} params)")
    print(f"Trained on {len(X_train):,} plays, validated on {len(X_val):,}")
    print("=" * 72)


if __name__ == "__main__":
    main()
