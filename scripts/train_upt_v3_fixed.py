"""Universal Player Transformer v3 (fixed) — proper scale-up.

Lessons from the collapsed v3:
- MiLB-only busts (22K players, 0 career WAR) poisoned training
- OneCycleLR with max_lr=0.0012 was too aggressive
- Uniform sample weighting let busts dominate

v3-fixed:
- EXCLUDE pure MiLB busts (keep only players with MLB appearances)
- Keep aggressive augmentation for linked MiLB+MLB paths (multiple windows per player)
- Simple cosine LR schedule, lower peak (0.0005)
- Sample weight inversely proportional to bust likelihood
- 6 transformer layers, d_model=256, ff=512 — similar to v2 but wider
- Target ~5M params (less than failed v3's 7M but more than v2's 3.5M)
- Should produce ~35K training sequences (more than v2's 18K)
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SEQ_LEN = 28
FEAT_DIM = 26

LEVEL_CODE = {"RK": 0, "A-": 1, "A": 2, "A+": 3, "AA": 4, "AAA": 5, "WIN": 1, "MLB": 6}
POS_CODE = {"C": 0, "1B": 1, "2B": 2, "3B": 3, "SS": 4, "LF": 5,
            "CF": 6, "RF": 7, "DH": 8, "P": 9}


class UPTv3Fixed(nn.Module):
    """Scaled transformer with same architecture as v2 but wider."""

    def __init__(self, feat_dim=26, d_model=256, nhead=8, num_layers=6,
                 ff_dim=512, dropout=0.12, max_seq=28):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq, d_model) * 0.02)
        self.level_embed = nn.Embedding(8, d_model // 4)
        self.pos_embed = nn.Embedding(11, d_model // 4)
        self.embed_proj = nn.Linear(d_model + d_model // 4 + d_model // 4, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        def make_head():
            return nn.Sequential(
                nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(128, 64), nn.GELU(),
                nn.Linear(64, 1),
            )

        self.career_head = make_head()
        self.peak_head = make_head()
        self.next_head = make_head()
        self.three_yr_head = make_head()
        self.bust_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x, levels, positions, mask=None):
        B, S, _ = x.shape
        h = self.input_proj(x)

        lev_emb = self.level_embed(levels)
        pos_emb = self.pos_embed(positions)
        h = self.embed_proj(torch.cat([h, lev_emb, pos_emb], dim=-1))
        h = h + self.pos_encoding[:, :S, :]

        attn_mask = (mask == 0) if mask is not None else None
        h = self.transformer(h, src_key_padding_mask=attn_mask)

        if mask is not None:
            m = mask.unsqueeze(-1)
            h_pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
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
    feats[11] = (age - 25) / 10.0
    feats[12] = float(row.get("wOBA", 0) or 0) if "wOBA" in row.index else 0
    feats[13] = (year - 1990) / 36.0
    feats[14] = float(row.get("pitching_runs", 0) or 0) / 50.0
    feats[15] = float(row.get("durability_runs", 0) or 0) / 20.0
    feats[16] = float(row.get("aqi_runs", 0) or 0) / 10.0
    feats[17] = float(row.get("leverage_runs", 0) or 0) / 10.0
    feats[18] = float(row.get("catcher_runs", 0) or 0) / 5.0
    pa = float(row.get("PA", 1) or 1)
    feats[19] = float(row.get("HR", 0) or 0) / max(pa, 1) * 600 / 40.0
    feats[20] = float(row.get("SB", 0) or 0) / max(pa, 1) * 600 / 40.0
    feats[21] = min(age - 18, 20) / 20.0 if age > 18 else 0
    feats[22] = 1.0 if float(row.get("IP", 0) or 0) > 50 else 0
    feats[23] = 1.0 if level == "MLB" else 0
    feats[24] = 1.0 if level in ("A", "A+") else 0
    feats[25] = float(row.get("ERA", 0) or 0) / 10.0 if "ERA" in row.index else 0
    return feats


def main():
    print("=" * 72)
    print("  UNIVERSAL PLAYER TRANSFORMER v3 (FIXED)")
    print("  Proper scale-up: no bust poisoning, conservative LR")
    print(f"  Training on {DEVICE}")
    print("=" * 72)

    mlb = pd.read_csv("data/bravs_all_seasons.csv")
    milb = pd.read_csv("data/bravs_milb_seasons.csv")
    people = pd.read_csv("data/lahman2025/People.csv")
    crosswalk = pd.read_csv("data/id_crosswalk.csv")

    id_map = dict(zip(crosswalk.mlbam_id, crosswalk.lahman_id))
    people_birth = dict(zip(people.playerID, people.birthYear))

    milb_bat = milb[milb.PA > 0].copy()
    milb_bat["pid_int"] = milb_bat.playerID.astype(float).astype(int)
    milb_bat["lahman_id"] = milb_bat.pid_int.map(id_map)
    milb_linked = milb_bat[milb_bat.lahman_id.notna()].copy()

    print(f"\nMLB seasons: {len(mlb)}")
    print(f"MiLB linked to MLB: {len(milb_linked)}")

    print("\nBuilding sequences (EXCLUDING pure MiLB busts)...")
    t0 = time.perf_counter()

    sequences, level_seqs, pos_seqs, mask_seqs = [], [], [], []
    y_career, y_peak, y_next, y_3yr, y_bust = [], [], [], [], []
    player_names = []
    sample_weights = []  # NEW: per-sample weight based on value

    def add_seq(timeline, levels, positions,
                career, peak, next_w, three_yr, bust, name, weight=1.0):
        if len(timeline) < 2:
            return
        if len(timeline) > MAX_SEQ_LEN:
            timeline = timeline[:MAX_SEQ_LEN]
            levels = levels[:MAX_SEQ_LEN]
            positions = positions[:MAX_SEQ_LEN]

        seq = np.zeros((MAX_SEQ_LEN, FEAT_DIM), dtype=np.float32)
        lev = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
        pos = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
        msk = np.zeros(MAX_SEQ_LEN, dtype=np.float32)

        for i in range(len(timeline)):
            seq[i] = timeline[i]
            lev[i] = levels[i]
            pos[i] = positions[i]
            msk[i] = 1.0

        sequences.append(seq)
        level_seqs.append(lev)
        pos_seqs.append(pos)
        mask_seqs.append(msk)
        y_career.append(career)
        y_peak.append(peak)
        y_next.append(next_w)
        y_3yr.append(three_yr)
        y_bust.append(bust)
        player_names.append(name)
        sample_weights.append(weight)

    mlb_by_player = mlb.groupby("playerID")
    milb_by_player = milb_linked.groupby("lahman_id")

    # 1. Players with BOTH MiLB and MLB (linked path, aggressive augmentation)
    for lahman_id, milb_grp in milb_by_player:
        if lahman_id not in mlb_by_player.groups:
            continue
        mlb_grp = mlb_by_player.get_group(lahman_id)

        full_timeline, full_levels, full_positions = [], [], []

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

        career = mlb_grp.bravs_war_eq.sum()
        peak = mlb_grp.bravs_war_eq.max()
        mlb_sorted = mlb_grp.sort_values("yearID")
        last = mlb_sorted.iloc[-1].bravs_war_eq
        last3 = mlb_sorted.tail(3).bravs_war_eq.sum()
        is_bust = 1.0 if career < 5.0 else 0.0
        name = mlb_grp.iloc[0]["name"]

        # Weight: stars get more attention
        # Star (50+ WAR): weight 2.0
        # All-star (20-50): weight 1.5
        # Regular (5-20): weight 1.0
        # Bust (<5): weight 0.5
        if career >= 50:
            base_weight = 2.0
        elif career >= 20:
            base_weight = 1.5
        elif career >= 5:
            base_weight = 1.0
        else:
            base_weight = 0.5

        # Full career view
        add_seq(full_timeline, full_levels, full_positions,
                career, peak, last, last3, is_bust, name, base_weight)

        # Augmentation: partial windows
        n_milb = len(milb_grp)
        n_mlb = len(mlb_grp)

        # MiLB only
        if n_milb >= 2:
            add_seq(full_timeline[:n_milb], full_levels[:n_milb], full_positions[:n_milb],
                    career, peak, last, last3, is_bust, name, base_weight * 0.8)

        # MiLB + first 2 MLB
        if n_mlb >= 4:
            cutoff = n_milb + 2
            if cutoff < len(full_timeline):
                add_seq(full_timeline[:cutoff], full_levels[:cutoff], full_positions[:cutoff],
                        career, peak,
                        mlb_sorted.iloc[2].bravs_war_eq if n_mlb > 2 else 0,
                        mlb_sorted.iloc[2:5].bravs_war_eq.sum() if n_mlb > 4 else last3,
                        is_bust, name, base_weight * 0.7)

        # MiLB + first half of MLB
        if n_mlb >= 6:
            half = n_milb + n_mlb // 2
            if half < len(full_timeline):
                add_seq(full_timeline[:half], full_levels[:half], full_positions[:half],
                        career, peak,
                        mlb_sorted.iloc[n_mlb//2].bravs_war_eq,
                        mlb_sorted.iloc[n_mlb//2:n_mlb//2+3].bravs_war_eq.sum(),
                        is_bust, name, base_weight * 0.7)

    # 2. MLB-only players (pad MiLB with zeros conceptually — just use MLB)
    milb_linked_ids = set(milb_linked.lahman_id.unique())
    for pid, mlb_grp in mlb_by_player:
        if pid in milb_linked_ids:
            continue
        if len(mlb_grp) < 3:
            continue

        timeline, levels, positions = [], [], []
        for _, row in mlb_grp.sort_values("yearID").iterrows():
            timeline.append(build_season_vector(row, "MLB", people_birth))
            levels.append(6)
            pos_str = str(row.get("position", "DH"))
            positions.append(POS_CODE.get(pos_str, 8))

        career = mlb_grp.bravs_war_eq.sum()
        peak = mlb_grp.bravs_war_eq.max()
        mlb_sorted = mlb_grp.sort_values("yearID")
        last = mlb_sorted.iloc[-1].bravs_war_eq
        last3 = mlb_sorted.tail(3).bravs_war_eq.sum()
        is_bust = 1.0 if career < 5.0 else 0.0
        name = mlb_grp.iloc[0]["name"]

        if career >= 50:
            base_weight = 2.0
        elif career >= 20:
            base_weight = 1.5
        elif career >= 5:
            base_weight = 1.0
        else:
            base_weight = 0.5

        add_seq(timeline, levels, positions,
                career, peak, last, last3, is_bust, name, base_weight)

        # Augmentation: first-half window
        if len(timeline) >= 6:
            half = len(timeline) // 2
            add_seq(timeline[:half], levels[:half], positions[:half],
                    career, peak,
                    mlb_sorted.iloc[half].bravs_war_eq if half < len(mlb_sorted) else 0,
                    mlb_sorted.iloc[half:half+3].bravs_war_eq.sum() if half+3 <= len(mlb_sorted) else last3,
                    is_bust, name, base_weight * 0.7)

    # NOTE: We EXCLUDE pure MiLB-only busts (22K players who never made MLB)
    # Those poisoned v3. The model should learn from people who actually
    # played MLB, whether they were stars or fringe guys.

    X = np.nan_to_num(np.array(sequences), nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -10.0, 10.0)
    L = np.array(level_seqs)
    P = np.array(pos_seqs)
    M = np.array(mask_seqs)
    y_c = np.clip(np.nan_to_num(np.array(y_career, dtype=np.float32)), -50, 200)
    y_p = np.clip(np.nan_to_num(np.array(y_peak, dtype=np.float32)), -10, 25)
    y_n = np.clip(np.nan_to_num(np.array(y_next, dtype=np.float32)), -10, 20)
    y_3 = np.clip(np.nan_to_num(np.array(y_3yr, dtype=np.float32)), -30, 60)
    y_b = np.array(y_bust, dtype=np.float32)
    w = np.array(sample_weights, dtype=np.float32)

    elapsed = time.perf_counter() - t0
    print(f"  Built {len(X):,} sequences in {elapsed:.1f}s")
    print(f"  Shape: {X.shape}")
    print(f"  Bust rate: {y_b.mean():.1%} (v3 failed had 67%)")
    print(f"  Career WAR: mean={y_c.mean():.1f}, std={y_c.std():.1f}")
    print(f"  Star samples (50+ WAR): {(y_c >= 50).sum()}")

    # Normalize targets
    yc_m, yc_s = y_c.mean(), y_c.std() + 1e-8
    yp_m, yp_s = y_p.mean(), y_p.std() + 1e-8
    yn_m, yn_s = y_n.mean(), y_n.std() + 1e-8
    y3_m, y3_s = y_3.mean(), y_3.std() + 1e-8

    X_t = torch.tensor(X, device=DEVICE)
    L_t = torch.tensor(L, device=DEVICE)
    P_t = torch.tensor(P, device=DEVICE)
    M_t = torch.tensor(M, device=DEVICE)
    yc_t = torch.tensor((y_c - yc_m) / yc_s, device=DEVICE)
    yp_t = torch.tensor((y_p - yp_m) / yp_s, device=DEVICE)
    yn_t = torch.tensor((y_n - yn_m) / yn_s, device=DEVICE)
    y3_t = torch.tensor((y_3 - y3_m) / y3_s, device=DEVICE)
    yb_t = torch.tensor(y_b, device=DEVICE)
    w_t = torch.tensor(w, device=DEVICE)

    n = len(X_t)
    n_val = int(n * 0.12)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    model = UPTv3Fixed().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    # Conservative optimizer — no OneCycleLR this time
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    BATCH_SIZE = 512
    print("Training (conservative cosine decay)...")
    t0 = time.perf_counter()
    best_val = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(500):
        model.train()
        shuf = train_idx[torch.randperm(len(train_idx))]
        epoch_loss = 0
        n_batches = 0
        for start in range(0, len(shuf), BATCH_SIZE):
            idx = shuf[start:start + BATCH_SIZE]
            pc, pp, pn, p3, pb = model(X_t[idx], L_t[idx], P_t[idx], M_t[idx])

            # Weighted Huber loss for regression targets
            weights = w_t[idx]
            loss_c = (F.smooth_l1_loss(pc, yc_t[idx], reduction="none") * weights).mean()
            loss_p = (F.smooth_l1_loss(pp, yp_t[idx], reduction="none") * weights).mean()
            loss_n = (F.smooth_l1_loss(pn, yn_t[idx], reduction="none") * weights).mean()
            loss_3 = (F.smooth_l1_loss(p3, y3_t[idx], reduction="none") * weights).mean()
            loss_b = F.binary_cross_entropy(pb, yb_t[idx]) * 0.3

            loss = loss_c + loss_p + loss_n + loss_3 + loss_b

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        if (epoch + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                vc, vp, vn, v3, vb = model(X_t[val_idx], L_t[val_idx], P_t[val_idx], M_t[val_idx])
                vloss = (F.smooth_l1_loss(vc, yc_t[val_idx]) +
                         F.smooth_l1_loss(vp, yp_t[val_idx]) +
                         F.smooth_l1_loss(vn, yn_t[val_idx]) +
                         F.smooth_l1_loss(v3, y3_t[val_idx]))
                vc_r = vc * yc_s + yc_m
                yc_r = yc_t[val_idx] * yc_s + yc_m
                r = np.corrcoef(vc_r.cpu().numpy(), yc_r.cpu().numpy())[0, 1]
                rmse = (vc_r - yc_r).pow(2).mean().sqrt().item()

            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1

            print(f"  Epoch {epoch+1:>3}: train={epoch_loss/n_batches:.3f} "
                  f"val={vloss.item():.3f} career_r={r:.4f} rmse={rmse:.2f}")

    print(f"\nBest val loss at epoch {best_epoch}")
    if best_state:
        model.load_state_dict(best_state)
    print(f"Training: {time.perf_counter()-t0:.1f}s")

    # Full eval
    model.eval()
    with torch.no_grad():
        # Chunked to avoid OOM
        all_c = []
        all_p = []
        for s in range(0, len(X_t), 2048):
            e = s + 2048
            vc, vp, _, _, _ = model(X_t[s:e], L_t[s:e], P_t[s:e], M_t[s:e])
            all_c.append((vc * yc_s + yc_m).cpu().numpy())
            all_p.append((vp * yp_s + yp_m).cpu().numpy())
    ac = np.concatenate(all_c)
    ap = np.concatenate(all_p)

    cr = np.corrcoef(ac, y_c)[0, 1]
    pr = np.corrcoef(ap, y_p)[0, 1]
    c_rmse = np.sqrt(((ac - y_c) ** 2).mean())

    print(f"\nFULL DATASET:")
    print(f"  Career WAR: r = {cr:.4f}, RMSE = {c_rmse:.2f}")
    print(f"  Peak WAR:   r = {pr:.4f}")

    # Known players
    print(f"\n--- Known Players ---")
    names = np.array(player_names)
    for target in ["Mike Trout", "Juan Soto", "Bryce Harper", "Mookie Betts",
                    "Aaron Judge", "Shohei Ohtani", "Barry Bonds", "Derek Jeter",
                    "Pedro Martinez", "Greg Maddux", "Ronald Acuna"]:
        last = target.split()[-1].lower()
        m = np.array([last in str(n).lower() for n in names])
        if m.sum() == 0:
            continue
        idx = np.where(m)[0]
        best = idx[y_c[idx].argmax()]
        print(f"  {player_names[best]:<24} actual={y_c[best]:+6.1f} pred={ac[best]:+6.1f} "
              f"(err={ac[best]-y_c[best]:+.1f})")

    # Save
    os.makedirs("models", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"feat_dim": FEAT_DIM, "d_model": 256, "nhead": 8,
                   "num_layers": 6, "ff_dim": 512, "max_seq": MAX_SEQ_LEN},
        "normalization": {
            "career": (float(yc_m), float(yc_s)),
            "peak": (float(yp_m), float(yp_s)),
            "next": (float(yn_m), float(yn_s)),
            "three_yr": (float(y3_m), float(y3_s)),
        },
        "n_sequences": len(X),
        "n_params": n_params,
        "full_career_r": float(cr),
        "full_peak_r": float(pr),
        "full_career_rmse": float(c_rmse),
    }, "models/universal_player_transformer_v3_fixed.pt")

    print(f"\nSaved models/universal_player_transformer_v3_fixed.pt")
    print(f"  {n_params:,} parameters, {len(X):,} sequences")
    print("=" * 72)


if __name__ == "__main__":
    main()
