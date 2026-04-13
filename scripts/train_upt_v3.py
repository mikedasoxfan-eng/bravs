"""Universal Player Transformer v3 — maximum scale.

Changes from v2 (3.46M params, 18K sequences):
- Include ALL players (not just MiLB+MLB linked) via data augmentation
- Pitchers get their own sequences (separate from batters)
- Longer max sequence (30 seasons)
- d_model=320, 8 layers, 8 heads -> targeting 6M+ parameters
- 4 targets: career WAR, peak WAR, 3-year WAR, career length
- Heavier augmentation: every 2-season window becomes a training example
- Target: 80K+ sequences
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SEQ_LEN = 30
FEAT_DIM = 26

LEVEL_CODE = {"RK": 0, "A-": 1, "A": 2, "A+": 3, "AA": 4, "AAA": 5, "WIN": 1, "MLB": 6}
POS_CODE = {"C": 0, "1B": 1, "2B": 2, "3B": 3, "SS": 4, "LF": 5,
            "CF": 6, "RF": 7, "DH": 8, "P": 9}


class UniversalPlayerTransformerV3(nn.Module):
    """Scaled-up career transformer targeting 6M+ parameters."""

    def __init__(self, feat_dim=26, d_model=320, nhead=8, num_layers=8,
                 ff_dim=640, dropout=0.12, max_seq=30):
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
                nn.Linear(d_model, 160), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(160, 80), nn.GELU(),
                nn.Linear(80, 1),
            )

        self.career_head = make_head()
        self.peak_head = make_head()
        self.three_yr_head = make_head()
        self.length_head = make_head()
        self.bust_head = nn.Sequential(
            nn.Linear(d_model, 80), nn.GELU(),
            nn.Linear(80, 1), nn.Sigmoid(),
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
            self.three_yr_head(h_pooled).squeeze(-1),
            self.length_head(h_pooled).squeeze(-1),
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
    # New in v3
    feats[24] = 1.0 if level in ("A", "A+") else 0  # mid-level indicator
    feats[25] = float(row.get("ERA", 0) or 0) / 10.0 if "ERA" in row.index else 0
    return feats


def main():
    print("=" * 72)
    print("  UNIVERSAL PLAYER TRANSFORMER v3")
    print("  Target: 6M+ parameters, 80K+ training sequences")
    print(f"  Training on {DEVICE}")
    print("=" * 72)

    mlb = pd.read_csv("data/bravs_all_seasons.csv")
    milb = pd.read_csv("data/bravs_milb_seasons.csv")
    people = pd.read_csv("data/lahman2025/People.csv")
    crosswalk = pd.read_csv("data/id_crosswalk.csv")

    id_map = dict(zip(crosswalk.mlbam_id, crosswalk.lahman_id))
    people_birth = dict(zip(people.playerID, people.birthYear))

    milb_all = milb.copy()
    milb_all["pid_int"] = milb_all.playerID.astype(float).astype(int)
    milb_all["lahman_id_mapped"] = milb_all.pid_int.map(id_map)

    print(f"\nMLB seasons: {len(mlb)}")
    print(f"MiLB seasons: {len(milb_all)}")
    print(f"MiLB linked to Lahman: {milb_all.lahman_id_mapped.notna().sum()}")

    print("\nBuilding sequences with aggressive augmentation...")
    t0 = time.perf_counter()

    sequences = []
    levels_arr = []
    pos_arr = []
    masks_list = []
    y_career, y_peak, y_3yr, y_length, y_bust = [], [], [], [], []
    player_names = []

    def add_seq(timeline, level_timeline, pos_timeline,
                career, peak, three_yr, length, bust, name):
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
        levels_arr.append(lev)
        pos_arr.append(pos)
        masks_list.append(msk)
        y_career.append(career)
        y_peak.append(peak)
        y_3yr.append(three_yr)
        y_length.append(length)
        y_bust.append(bust)
        player_names.append(name)

    # 1. Players with BOTH MiLB and MLB data (full path)
    mlb_by_player = mlb.groupby("playerID")
    milb_linked = milb_all[milb_all.lahman_id_mapped.notna()]
    milb_by_player = milb_linked.groupby("lahman_id_mapped")

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

        career = mlb_grp.bravs_war_eq.sum()
        peak = mlb_grp.bravs_war_eq.max()
        mlb_sorted = mlb_grp.sort_values("yearID")
        last = mlb_sorted.iloc[-1].bravs_war_eq
        last3 = mlb_sorted.tail(3).bravs_war_eq.sum()
        seasons_played = len(mlb_grp)
        is_bust = 1.0 if career < 5.0 else 0.0
        name = mlb_grp.iloc[0]["name"]

        # Full career
        add_seq(full_timeline, full_levels, full_positions,
                career, peak, last3, seasons_played, is_bust, name)

        # Aggressive augmentation: every partial-career window
        n_milb = len(milb_grp)
        n_mlb = len(mlb_grp)
        n_total = n_milb + n_mlb

        # MiLB-only window
        if n_milb >= 2:
            add_seq(full_timeline[:n_milb], full_levels[:n_milb], full_positions[:n_milb],
                    career, peak, last3, seasons_played, is_bust, name)

        # Each partial MLB window
        for cutoff in range(n_milb + 2, n_total, 2):
            remaining_war = mlb_sorted.iloc[cutoff - n_milb:].bravs_war_eq.sum()
            remaining_3yr = mlb_sorted.iloc[cutoff - n_milb:cutoff - n_milb + 3].bravs_war_eq.sum()
            add_seq(full_timeline[:cutoff], full_levels[:cutoff], full_positions[:cutoff],
                    career, peak, remaining_3yr, seasons_played, is_bust, name)

    # 2. MLB-only players (not in MiLB linked data)
    milb_linked_ids = set(milb_linked.lahman_id_mapped.unique())
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

        career = mlb_grp.bravs_war_eq.sum()
        peak = mlb_grp.bravs_war_eq.max()
        mlb_sorted = mlb_grp.sort_values("yearID")
        last = mlb_sorted.iloc[-1].bravs_war_eq
        last3 = mlb_sorted.tail(3).bravs_war_eq.sum()
        seasons_played = len(mlb_grp)
        is_bust = 1.0 if career < 5.0 else 0.0
        name = mlb_grp.iloc[0]["name"]

        add_seq(timeline, levels, positions,
                career, peak, last3, seasons_played, is_bust, name)

        # Augmentation: every 2-season cutoff
        for cutoff in range(2, len(timeline), 2):
            rem_3yr = mlb_sorted.iloc[cutoff:cutoff + 3].bravs_war_eq.sum() if cutoff + 3 <= len(mlb_sorted) else last3
            add_seq(timeline[:cutoff], levels[:cutoff], positions[:cutoff],
                    career, peak, rem_3yr, seasons_played, is_bust, name)

    # 3. MiLB-only players (didn't make MLB, or we don't have their MLB mapped)
    # Their "career" target is 0 (bust) since they didn't reach MLB
    unlinked_milb = milb_all[milb_all.lahman_id_mapped.isna()]
    milb_unlinked_groups = unlinked_milb.groupby("playerID")

    count_milb_only = 0
    for pid, grp in milb_unlinked_groups:
        if len(grp) < 3:
            continue
        count_milb_only += 1

        timeline = []
        levels = []
        positions = []
        for _, row in grp.sort_values("yearID").iterrows():
            level = str(row.get("level", "A"))
            timeline.append(build_season_vector(row, level, people_birth))
            levels.append(LEVEL_CODE.get(level, 2))
            pos_str = str(row.get("position", "DH")).split("/")[0]
            positions.append(POS_CODE.get(pos_str, 8))

        # Target: these guys didn't make MLB, career WAR = 0
        name = grp.iloc[0]["name"]
        add_seq(timeline, levels, positions, 0.0, 0.0, 0.0, 0, 1.0, name)

    print(f"  Linked MiLB+MLB careers augmented")
    print(f"  MLB-only players: added")
    print(f"  MiLB-only busts: {count_milb_only} players")

    X = np.nan_to_num(np.array(sequences), nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -10.0, 10.0)
    L = np.array(levels_arr)
    P = np.array(pos_arr)
    M = np.array(masks_list)
    y_c = np.clip(np.nan_to_num(np.array(y_career, dtype=np.float32)), -50, 200)
    y_p = np.clip(np.nan_to_num(np.array(y_peak, dtype=np.float32)), -10, 25)
    y_3 = np.clip(np.nan_to_num(np.array(y_3yr, dtype=np.float32)), -30, 60)
    y_l = np.clip(np.nan_to_num(np.array(y_length, dtype=np.float32)), 0, 30)
    y_b = np.array(y_bust, dtype=np.float32)

    elapsed = time.perf_counter() - t0
    print(f"\n  Built {len(X):,} sequences in {elapsed:.1f}s")
    print(f"  Shape: {X.shape}")
    print(f"  Career WAR stats: mean={y_c.mean():.1f}, std={y_c.std():.1f}")
    print(f"  Bust rate: {y_b.mean():.1%}")

    # Normalize
    yc_m, yc_s = y_c.mean(), y_c.std() + 1e-8
    yp_m, yp_s = y_p.mean(), y_p.std() + 1e-8
    y3_m, y3_s = y_3.mean(), y_3.std() + 1e-8
    yl_m, yl_s = y_l.mean(), y_l.std() + 1e-8

    X_t = torch.tensor(X, device=DEVICE)
    L_t = torch.tensor(L, device=DEVICE)
    P_t = torch.tensor(P, device=DEVICE)
    M_t = torch.tensor(M, device=DEVICE)
    yc_t = torch.tensor((y_c - yc_m) / yc_s, device=DEVICE)
    yp_t = torch.tensor((y_p - yp_m) / yp_s, device=DEVICE)
    y3_t = torch.tensor((y_3 - y3_m) / y3_s, device=DEVICE)
    yl_t = torch.tensor((y_l - yl_m) / yl_s, device=DEVICE)
    yb_t = torch.tensor(y_b, device=DEVICE)

    n = len(X_t)
    n_val = int(n * 0.1)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    model = UniversalPlayerTransformerV3().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=5e-4)

    BATCH_SIZE = 256
    steps_per_epoch = (len(train_idx) + BATCH_SIZE - 1) // BATCH_SIZE
    N_EPOCHS = 300
    total_steps = steps_per_epoch * N_EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.0012, total_steps=total_steps, pct_start=0.05,
    )

    print("\nTraining...")
    t0 = time.perf_counter()
    best_val = float("inf")
    best_state = None

    for epoch in range(N_EPOCHS):
        model.train()
        shuf = train_idx[torch.randperm(len(train_idx))]
        epoch_loss = 0
        n_batches = 0
        for s in range(0, len(shuf), BATCH_SIZE):
            idx = shuf[s:s + BATCH_SIZE]
            pc, pp, p3, pl, pb = model(X_t[idx], L_t[idx], P_t[idx], M_t[idx])
            loss = (F.smooth_l1_loss(pc, yc_t[idx]) +
                    F.smooth_l1_loss(pp, yp_t[idx]) +
                    F.smooth_l1_loss(p3, y3_t[idx]) +
                    F.smooth_l1_loss(pl, yl_t[idx]) +
                    F.binary_cross_entropy(pb, yb_t[idx]) * 0.5)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                vc, vp, v3, vl, vb = model(X_t[val_idx], L_t[val_idx], P_t[val_idx], M_t[val_idx])
                vloss = (F.smooth_l1_loss(vc, yc_t[val_idx]) +
                         F.smooth_l1_loss(vp, yp_t[val_idx]) +
                         F.smooth_l1_loss(v3, y3_t[val_idx]) +
                         F.smooth_l1_loss(vl, yl_t[val_idx]))
                vc_r = vc * yc_s + yc_m
                yc_r = yc_t[val_idx] * yc_s + yc_m
                r = np.corrcoef(vc_r.cpu().numpy(), yc_r.cpu().numpy())[0, 1]
                rmse = (vc_r - yc_r).pow(2).mean().sqrt().item()

            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            print(f"  Epoch {epoch+1:>3}: train={epoch_loss/n_batches:.3f} "
                  f"val_loss={vloss.item():.3f} career_r={r:.4f} rmse={rmse:.2f}")

    if best_state:
        model.load_state_dict(best_state)
    print(f"Training: {time.perf_counter()-t0:.1f}s")

    # Full eval
    model.eval()
    # Chunked prediction to avoid OOM
    all_c = []
    all_p = []
    with torch.no_grad():
        for s in range(0, len(X_t), 1024):
            e = s + 1024
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
                    "Cam Schlittler"]:
        last = target.split()[-1].lower()
        m = np.array([last in str(n).lower() for n in names])
        if m.sum() == 0:
            continue
        idx = np.where(m)[0]
        best = idx[y_c[idx].argmax()] if y_c[idx].max() > 0 else idx[0]
        print(f"  {player_names[best]:<24} actual={y_c[best]:+6.1f} pred={ac[best]:+6.1f}")

    # Save
    os.makedirs("models", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"feat_dim": FEAT_DIM, "d_model": 320, "nhead": 8,
                   "num_layers": 8, "ff_dim": 640, "max_seq": MAX_SEQ_LEN},
        "normalization": {
            "career": (float(yc_m), float(yc_s)),
            "peak": (float(yp_m), float(yp_s)),
            "three_yr": (float(y3_m), float(y3_s)),
            "length": (float(yl_m), float(yl_s)),
        },
        "n_sequences": len(X),
        "n_params": n_params,
    }, "models/universal_player_transformer_v3.pt")

    print(f"\nSaved models/universal_player_transformer_v3.pt")
    print(f"  {n_params:,} parameters")
    print(f"  {len(X):,} training sequences")
    print("=" * 72)


if __name__ == "__main__":
    main()
