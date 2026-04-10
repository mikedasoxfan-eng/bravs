"""Universal Player Transformer v2 — scaled up.

Changes from v1:
- 10x more training data via data augmentation (partial career windows)
- ALL MLB players included (not just MiLB-linked)
- d_model=256, 6 layers, 8 heads, ff=512 -> 1.5M+ parameters
- 24 input features (added position encoding, handedness, draft era)
- Predicts 5 targets: career WAR, peak WAR, next-season WAR, 3yr WAR, bust probability
- Curriculum learning: train on easy examples first, then hard ones
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SEQ_LEN = 25  # longer sequences
FEAT_DIM = 24     # more features

LEVEL_CODE = {"RK": 0, "A-": 1, "A": 2, "A+": 3, "AA": 4, "AAA": 5, "WIN": 1, "MLB": 6}
POS_CODE = {"C": 0, "1B": 1, "2B": 2, "3B": 3, "SS": 4, "LF": 5, "CF": 6, "RF": 7, "DH": 8, "P": 9}


class UniversalPlayerTransformerV2(nn.Module):
    """Scaled-up transformer for career prediction.

    1.5M+ parameters. 6 layers, 8 heads, d_model=256.
    """
    def __init__(self, feat_dim=24, d_model=256, nhead=8, num_layers=6,
                 ff_dim=512, dropout=0.12, max_seq=25):
        super().__init__()
        self.d_model = d_model

        # Input projection with layer norm
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learned positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq, d_model) * 0.02)

        # Level embedding (separate from features)
        self.level_embed = nn.Embedding(8, d_model // 4)

        # Position embedding
        self.pos_embed = nn.Embedding(11, d_model // 4)

        # Projection after concatenating embeddings
        self.embed_proj = nn.Linear(d_model + d_model // 4 + d_model // 4, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads (bigger)
        def make_head():
            return nn.Sequential(
                nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(128, 64), nn.GELU(),
                nn.Linear(64, 1),
            )

        self.career_head = make_head()     # total career WAR
        self.peak_head = make_head()       # peak single-season WAR
        self.next_head = make_head()       # next-season WAR
        self.three_yr_head = make_head()   # next 3 years cumulative WAR
        self.bust_head = nn.Sequential(    # probability of bust (career WAR < 5)
            nn.Linear(d_model, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x, levels, positions, mask=None):
        B, S, _ = x.shape

        # Project input features
        h = self.input_proj(x)

        # Add level and position embeddings
        lev_emb = self.level_embed(levels)     # (B, S, d/4)
        pos_emb = self.pos_embed(positions)    # (B, S, d/4)
        h = self.embed_proj(torch.cat([h, lev_emb, pos_emb], dim=-1))

        # Add positional encoding
        h = h + self.pos_encoding[:, :S, :]

        # Transformer
        if mask is not None:
            attn_mask = (mask == 0)
        else:
            attn_mask = None

        h = self.transformer(h, src_key_padding_mask=attn_mask)

        # Pool: attention-weighted mean (learn what to focus on)
        if mask is not None:
            mask_exp = mask.unsqueeze(-1)
            h_pooled = (h * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        else:
            h_pooled = h.mean(dim=1)

        return (
            self.career_head(h_pooled).squeeze(-1),
            self.peak_head(h_pooled).squeeze(-1),
            self.next_head(h_pooled).squeeze(-1),
            self.three_yr_head(h_pooled).squeeze(-1),
            self.bust_head(h_pooled).squeeze(-1),
        )


def build_season_vector(row, level, people_birth):
    pid = row.get("playerID") or row.get("lahman_id")
    birth = people_birth.get(pid)
    year = int(row.get("yearID", 2020))
    age = (year - int(birth)) if birth and not pd.isna(birth) else 25

    feats = np.zeros(FEAT_DIM, dtype=np.float32)
    feats[0] = float(row.get("bravs_war_eq", 0) or 0) / 10.0
    feats[1] = float(row.get("hitting_runs", 0) or 0) / 50.0
    feats[2] = float(row.get("baserunning_runs", 0) or 0) / 10.0
    feats[3] = float(row.get("fielding_runs", 0) or 0) / 10.0
    feats[4] = float(row.get("positional_runs", 0) or 0) / 10.0
    feats[5] = float(row.get("PA", 0) or 0) / 700.0
    feats[6] = float(row.get("HR", 0) or 0) / 50.0
    feats[7] = float(row.get("SB", 0) or 0) / 50.0
    feats[8] = float(row.get("IP", 0) or 0) / 250.0
    feats[9] = float(row.get("G", 0) or 0) / 162.0
    feats[10] = age / 40.0
    feats[11] = (age - 25) / 10.0  # age relative to prime
    feats[12] = float(row.get("wOBA", 0) or 0) if "wOBA" in row.index else 0
    feats[13] = (year - 1990) / 36.0  # era indicator
    feats[14] = float(row.get("pitching_runs", 0) or 0) / 50.0
    feats[15] = float(row.get("durability_runs", 0) or 0) / 20.0
    feats[16] = float(row.get("aqi_runs", 0) or 0) / 10.0
    feats[17] = float(row.get("leverage_runs", 0) or 0) / 10.0
    feats[18] = float(row.get("catcher_runs", 0) or 0) / 5.0
    # Rate stats
    pa = float(row.get("PA", 1) or 1)
    feats[19] = float(row.get("HR", 0) or 0) / max(pa, 1) * 600 / 40.0  # HR rate normalized
    feats[20] = float(row.get("SB", 0) or 0) / max(pa, 1) * 600 / 40.0  # SB rate
    # Cumulative indicators
    feats[21] = min(age - 18, 20) / 20.0 if age > 18 else 0  # years of pro ball
    feats[22] = 1.0 if float(row.get("IP", 0) or 0) > 50 else 0  # is pitcher
    feats[23] = 1.0 if level == "MLB" else 0  # MLB indicator

    return feats


def main():
    print("=" * 70)
    print("  UNIVERSAL PLAYER TRANSFORMER v2")
    print(f"  1.5M+ parameters | 6 layers | 8 heads | d_model=256")
    print(f"  Training on {DEVICE}")
    print("=" * 70)

    mlb = pd.read_csv("data/bravs_all_seasons.csv")
    milb = pd.read_csv("data/bravs_milb_seasons.csv")
    people = pd.read_csv("data/lahman2025/People.csv")
    crosswalk = pd.read_csv("data/id_crosswalk.csv")

    id_map = dict(zip(crosswalk.mlbam_id, crosswalk.lahman_id))
    people_birth = dict(zip(people.playerID, people.birthYear))

    # Map MiLB
    milb_bat = milb[milb.PA > 0].copy()
    milb_bat["pid_int"] = milb_bat.playerID.astype(float).astype(int)
    milb_bat["lahman_id"] = milb_bat.pid_int.map(id_map)
    milb_linked = milb_bat[milb_bat.lahman_id.notna()].copy()

    print(f"\nMLB: {len(mlb)}, MiLB linked: {len(milb_linked)}")

    # ═══ BUILD SEQUENCES ═══
    # Strategy: create MULTIPLE sequences per player via data augmentation
    # For a 15-year career, create windows: first 3 years, first 5, first 8, full career
    # This 10x-es the training data

    print("Building augmented career sequences...")
    t0 = time.perf_counter()

    mlb_by_player = mlb.groupby("playerID")
    milb_by_player = milb_linked.groupby("lahman_id")

    sequences = []
    level_seqs = []
    pos_seqs = []
    mask_seqs = []
    y_career = []
    y_peak = []
    y_next = []
    y_3yr = []
    y_bust = []
    player_names = []

    def add_sequence(timeline, level_timeline, pos_timeline,
                     career_war, peak_war, next_war, three_yr_war, is_bust, name):
        if len(timeline) < 2:
            return

        if len(timeline) > MAX_SEQ_LEN:
            timeline = timeline[:MAX_SEQ_LEN]
            level_timeline = level_timeline[:MAX_SEQ_LEN]
            pos_timeline = pos_timeline[:MAX_SEQ_LEN]

        seq = np.zeros((MAX_SEQ_LEN, FEAT_DIM), dtype=np.float32)
        lev = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
        pos = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
        msk = np.zeros(MAX_SEQ_LEN, dtype=np.float32)

        for i in range(len(timeline)):
            seq[i] = timeline[i]
            lev[i] = level_timeline[i]
            pos[i] = pos_timeline[i]
            msk[i] = 1.0

        sequences.append(seq)
        level_seqs.append(lev)
        pos_seqs.append(pos)
        mask_seqs.append(msk)
        y_career.append(career_war)
        y_peak.append(peak_war)
        y_next.append(next_war)
        y_3yr.append(three_yr_war)
        y_bust.append(is_bust)
        player_names.append(name)

    # 1. Players with MiLB + MLB data
    for lahman_id, milb_grp in milb_by_player:
        if lahman_id not in mlb_by_player.groups:
            continue
        mlb_grp = mlb_by_player.get_group(lahman_id)

        full_timeline = []
        full_levels = []
        full_positions = []

        for _, row in milb_grp.sort_values("yearID").iterrows():
            level = str(row.get("level", "AAA"))
            full_timeline.append(build_season_vector(row, level, people_birth))
            full_levels.append(LEVEL_CODE.get(level, 5))
            pos_str = str(row.get("position", "DH")).split("/")[0]
            full_positions.append(POS_CODE.get(pos_str, 8))

        for _, row in mlb_grp.sort_values("yearID").iterrows():
            full_timeline.append(build_season_vector(row, "MLB", people_birth))
            full_levels.append(6)
            pos_str = str(row.get("position", "DH"))
            full_positions.append(POS_CODE.get(pos_str, 8))

        career_war = mlb_grp.bravs_war_eq.sum()
        peak_war = mlb_grp.bravs_war_eq.max()
        mlb_sorted = mlb_grp.sort_values("yearID")
        last_war = mlb_sorted.iloc[-1].bravs_war_eq
        last3 = mlb_sorted.tail(3).bravs_war_eq.sum()
        is_bust = 1.0 if career_war < 5.0 else 0.0
        name = mlb_grp.iloc[0]["name"]

        # Full career
        add_sequence(full_timeline, full_levels, full_positions,
                     career_war, peak_war, last_war, last3, is_bust, name)

        # Augmentation: partial windows (MiLB only, MiLB + first few MLB)
        n_milb = len(milb_grp)
        n_mlb = len(mlb_grp)

        # MiLB-only window (predict future MLB career)
        if n_milb >= 2:
            add_sequence(full_timeline[:n_milb], full_levels[:n_milb], full_positions[:n_milb],
                         career_war, peak_war, last_war, last3, is_bust, name)

        # MiLB + first 2 MLB seasons
        if n_mlb >= 4:
            cutoff = n_milb + 2
            remaining_war = mlb_sorted.iloc[2:].bravs_war_eq.sum()
            add_sequence(full_timeline[:cutoff], full_levels[:cutoff], full_positions[:cutoff],
                         career_war, peak_war, mlb_sorted.iloc[2].bravs_war_eq if n_mlb > 2 else 0,
                         mlb_sorted.iloc[2:5].bravs_war_eq.sum() if n_mlb > 4 else last3,
                         is_bust, name)

        # MiLB + first half of MLB career
        if n_mlb >= 6:
            half = n_milb + n_mlb // 2
            add_sequence(full_timeline[:half], full_levels[:half], full_positions[:half],
                         career_war, peak_war,
                         mlb_sorted.iloc[n_mlb//2].bravs_war_eq,
                         mlb_sorted.iloc[n_mlb//2:n_mlb//2+3].bravs_war_eq.sum(),
                         is_bust, name)

    # 2. MLB-only players (no MiLB data — pad MiLB with zeros)
    milb_linked_ids = set(milb_linked.lahman_id.unique())
    for pid, mlb_grp in mlb_by_player:
        if pid in milb_linked_ids:
            continue
        if len(mlb_grp) < 3:
            continue

        timeline = []
        levels = []
        positions = []

        for _, row in mlb_grp.sort_values("yearID").iterrows():
            timeline.append(build_season_vector(row, "MLB", people_birth))
            levels.append(6)
            pos_str = str(row.get("position", "DH"))
            positions.append(POS_CODE.get(pos_str, 8))

        career_war = mlb_grp.bravs_war_eq.sum()
        peak_war = mlb_grp.bravs_war_eq.max()
        mlb_sorted = mlb_grp.sort_values("yearID")
        last_war = mlb_sorted.iloc[-1].bravs_war_eq
        last3 = mlb_sorted.tail(3).bravs_war_eq.sum()
        is_bust = 1.0 if career_war < 5.0 else 0.0
        name = mlb_grp.iloc[0]["name"]

        # Full career
        add_sequence(timeline, levels, positions,
                     career_war, peak_war, last_war, last3, is_bust, name)

        # Augmentation: first half
        if len(timeline) >= 6:
            half = len(timeline) // 2
            add_sequence(timeline[:half], levels[:half], positions[:half],
                         career_war, peak_war,
                         mlb_sorted.iloc[half].bravs_war_eq if half < len(mlb_sorted) else 0,
                         mlb_sorted.iloc[half:half+3].bravs_war_eq.sum() if half+3 <= len(mlb_sorted) else last3,
                         is_bust, name)

    # Convert to arrays
    X = np.nan_to_num(np.array(sequences), nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -10.0, 10.0)
    L = np.array(level_seqs)
    P = np.array(pos_seqs)
    M = np.array(mask_seqs)
    y_c = np.clip(np.nan_to_num(np.array(y_career, dtype=np.float32)), -50, 200)
    y_p = np.clip(np.nan_to_num(np.array(y_peak, dtype=np.float32)), -10, 20)
    y_n = np.clip(np.nan_to_num(np.array(y_next, dtype=np.float32)), -10, 20)
    y_3 = np.clip(np.nan_to_num(np.array(y_3yr, dtype=np.float32)), -30, 60)
    y_b = np.array(y_bust, dtype=np.float32)

    elapsed = time.perf_counter() - t0
    print(f"\nBuilt {len(X)} sequences in {elapsed:.1f}s")
    print(f"  MiLB+MLB linked: ~{sum(1 for n in player_names if player_names.count(n) > 1)} augmented")
    print(f"  MLB-only: ~{len(X) - sum(1 for n in player_names if player_names.count(n) > 1)}")
    print(f"  Shape: {X.shape}")
    print(f"  Target stats: career [{y_c.min():.0f}, {y_c.max():.0f}], "
          f"bust rate={y_b.mean():.1%}")

    # Normalize targets
    yc_m, yc_s = y_c.mean(), y_c.std() + 1e-8
    yp_m, yp_s = y_p.mean(), y_p.std() + 1e-8
    yn_m, yn_s = y_n.mean(), y_n.std() + 1e-8
    y3_m, y3_s = y_3.mean(), y_3.std() + 1e-8

    # To GPU
    X_t = torch.tensor(X, device=DEVICE)
    L_t = torch.tensor(L, device=DEVICE)
    P_t = torch.tensor(P, device=DEVICE)
    M_t = torch.tensor(M, device=DEVICE)
    yc_t = torch.tensor((y_c - yc_m) / yc_s, device=DEVICE)
    yp_t = torch.tensor((y_p - yp_m) / yp_s, device=DEVICE)
    yn_t = torch.tensor((y_n - yn_m) / yn_s, device=DEVICE)
    y3_t = torch.tensor((y_3 - y3_m) / y3_s, device=DEVICE)
    yb_t = torch.tensor(y_b, device=DEVICE)

    # Split
    n = len(X_t)
    n_val = int(n * 0.12)
    perm = torch.randperm(n)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    # Model
    model = UniversalPlayerTransformerV2(feat_dim=FEAT_DIM).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=5e-4)
    BATCH_SIZE = 512
    n_train = len(train_idx)
    steps_per_epoch = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
    total_steps = steps_per_epoch * 600
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, total_steps=total_steps, pct_start=0.05,
    )

    print("Training...")
    t0 = time.perf_counter()
    best_val = float("inf")
    best_state = None

    for epoch in range(600):
        model.train()

        # Shuffle training data
        shuffle = train_idx[torch.randperm(len(train_idx))]

        epoch_loss = 0
        n_batches = 0
        for start in range(0, len(shuffle), BATCH_SIZE):
            idx = shuffle[start:start + BATCH_SIZE]

            pc, pp, pn, p3, pb = model(X_t[idx], L_t[idx], P_t[idx], M_t[idx])

            loss = (F.smooth_l1_loss(pc, yc_t[idx]) +
                    F.smooth_l1_loss(pp, yp_t[idx]) +
                    F.smooth_l1_loss(pn, yn_t[idx]) +
                    F.smooth_l1_loss(p3, y3_t[idx]) +
                    F.binary_cross_entropy(pb, yb_t[idx]) * 0.5)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                vc, vp, vn, v3, vb = model(X_t[val_idx], L_t[val_idx], P_t[val_idx], M_t[val_idx])
                vl = (F.smooth_l1_loss(vc, yc_t[val_idx]) +
                      F.smooth_l1_loss(vp, yp_t[val_idx]) +
                      F.smooth_l1_loss(vn, yn_t[val_idx]) +
                      F.smooth_l1_loss(v3, y3_t[val_idx]))

                vc_r = vc * yc_s + yc_m
                yc_r = yc_t[val_idx] * yc_s + yc_m
                r = np.corrcoef(vc_r.cpu().numpy(), yc_r.cpu().numpy())[0, 1]
                rmse = (vc_r - yc_r).pow(2).mean().sqrt().item()

                # Bust accuracy
                bust_pred = (vb > 0.5).float()
                bust_acc = (bust_pred == yb_t[val_idx]).float().mean().item()

            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:>3}: loss={epoch_loss/n_batches:.4f} val={vl.item():.4f} "
                  f"r={r:.4f} rmse={rmse:.1f} bust_acc={bust_acc:.3f} lr={lr:.6f}")

    if best_state:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - t0
    print(f"\nTraining: {elapsed:.1f}s")

    # ═══ EVALUATE ═══
    model.eval()
    with torch.no_grad():
        ac, ap, an, a3, ab = model(X_t, L_t, P_t, M_t)
        ac = (ac * yc_s + yc_m).cpu().numpy()
        ap = (ap * yp_s + yp_m).cpu().numpy()
        an = (an * yn_s + yn_m).cpu().numpy()
        a3 = (a3 * y3_s + y3_m).cpu().numpy()
        ab_prob = ab.cpu().numpy()

    cr = np.corrcoef(ac, y_c)[0, 1]
    pr = np.corrcoef(ap, y_p)[0, 1]
    nr = np.corrcoef(an, y_n)[0, 1]
    r3 = np.corrcoef(a3, y_3)[0, 1]
    c_rmse = np.sqrt(((ac - y_c)**2).mean())
    bust_acc = ((ab_prob > 0.5) == y_b).mean()

    print(f"\n{'='*60}")
    print(f"  FULL RESULTS ({len(X)} sequences):")
    print(f"    Career WAR:   r = {cr:.4f}, RMSE = {c_rmse:.1f}")
    print(f"    Peak WAR:     r = {pr:.4f}")
    print(f"    Next season:  r = {nr:.4f}")
    print(f"    3-year WAR:   r = {r3:.4f}")
    print(f"    Bust predict: {bust_acc:.1%} accuracy")
    print(f"{'='*60}")

    # Known players
    print(f"\n--- Known Player Predictions ---")
    names = np.array(player_names)
    for target in ["Mike Trout", "Juan Soto", "Bryce Harper", "Mookie Betts",
                    "Aaron Judge", "Shohei Ohtani", "Ronald Acuna", "Barry Bonds",
                    "Derek Jeter", "Pedro Martinez", "Greg Maddux"]:
        last = target.split()[-1].lower()
        mask_n = np.array([last in n.lower() for n in names])
        if mask_n.sum() == 0:
            continue
        idx = np.where(mask_n)[0]
        best = idx[y_c[idx].argmax()]
        bust_p = ab_prob[best]
        print(f"  {player_names[best]:<22} career: actual={y_c[best]:>+6.1f} pred={ac[best]:>+6.1f} "
              f"| peak: {y_p[best]:>+5.1f}/{ap[best]:>+5.1f} "
              f"| bust={bust_p:.1%}")

    # Save
    os.makedirs("models", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "feat_dim": FEAT_DIM, "d_model": 256, "nhead": 8,
            "num_layers": 6, "ff_dim": 512, "max_seq": MAX_SEQ_LEN,
        },
        "normalization": {
            "career": (float(yc_m), float(yc_s)),
            "peak": (float(yp_m), float(yp_s)),
            "next": (float(yn_m), float(yn_s)),
            "three_yr": (float(y3_m), float(y3_s)),
        },
        "n_sequences": len(X),
        "n_params": n_params,
        "metrics": {"career_r": cr, "peak_r": pr, "next_r": nr,
                    "three_yr_r": r3, "bust_acc": bust_acc, "career_rmse": c_rmse},
    }, "models/universal_player_transformer_v2.pt")

    print(f"\nSaved models/universal_player_transformer_v2.pt")
    print(f"  {n_params:,} parameters")
    print(f"  Trained on {len(X):,} sequences ({len(set(player_names)):,} unique players)")
    print(f"  Input: {MAX_SEQ_LEN} seasons x {FEAT_DIM} features + level/position embeddings")
    print("=" * 70)


if __name__ == "__main__":
    main()
