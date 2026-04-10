"""Train the Universal Player Transformer — the biggest BRAVS model.

Reads a player's full MiLB + MLB career timeline as a sequence,
then predicts career WAR, peak WAR, and next-season WAR using
a 4-layer transformer encoder with multi-head attention.

This is a sequence model: it sees minor league stats as early tokens
and MLB stats as later tokens, learning the progression patterns
that distinguish future stars from busts.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SEQ_LEN = 20
FEAT_DIM = 18

LEVEL_CODE = {"RK": 0, "A-": 1, "A": 2, "A+": 3, "AA": 4, "AAA": 5, "WIN": 1, "MLB": 6}


class UniversalPlayerTransformer(nn.Module):
    """Transformer that reads a player's full MiLB+MLB timeline
    and predicts career outcomes.

    Architecture:
    - Input projection: 18-dim season vectors -> 128-dim embeddings
    - Learned positional encoding (up to 20 seasons)
    - 4-layer transformer encoder with 4-head self-attention
    - Mean pooling over valid positions
    - 3 output heads: career WAR, peak WAR, next-season WAR

    The attention mechanism learns which seasons are most predictive:
    - A dominant AA season at age 21 gets high attention weight
    - A mediocre AAA season at age 27 gets low weight
    - The model learns age-relative-to-level patterns automatically
    """

    def __init__(self, feat_dim=18, d_model=128, nhead=4, num_layers=4, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, MAX_SEQ_LEN, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.career_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        self.peak_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        self.next_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x, mask=None):
        h = self.input_proj(x) + self.pos_encoding[:, :x.shape[1], :]

        if mask is not None:
            attn_mask = (mask == 0)
        else:
            attn_mask = None

        h = self.transformer(h, src_key_padding_mask=attn_mask)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            h_pooled = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            h_pooled = h.mean(dim=1)

        career = self.career_head(h_pooled).squeeze(-1)
        peak = self.peak_head(h_pooled).squeeze(-1)
        next_s = self.next_head(h_pooled).squeeze(-1)

        return career, peak, next_s


def build_season_vector(row, level, people_birth):
    pid = row.get("playerID") or row.get("lahman_id")
    birth = people_birth.get(pid)
    year = int(row.get("yearID", 2020))
    age = (year - int(birth)) if birth and not pd.isna(birth) else 25

    feats = np.zeros(FEAT_DIM, dtype=np.float32)
    feats[0] = float(row.get("bravs_war_eq", 0) or 0)
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
    feats[11] = LEVEL_CODE.get(level, 6) / 6.0
    feats[12] = float(row.get("wOBA", 0) or 0) if "wOBA" in row.index else 0
    feats[13] = year / 2026.0
    feats[14] = float(row.get("pitching_runs", 0) or 0) / 50.0
    feats[15] = float(row.get("durability_runs", 0) or 0) / 20.0
    feats[16] = float(row.get("aqi_runs", 0) or 0) / 10.0
    feats[17] = float(row.get("leverage_runs", 0) or 0) / 10.0
    return feats


def main():
    print("=" * 70)
    print("  BRAVS UNIVERSAL PLAYER TRANSFORMER")
    print(f"  Training on {DEVICE}")
    print("=" * 70)

    mlb = pd.read_csv("data/bravs_all_seasons.csv")
    milb = pd.read_csv("data/bravs_milb_seasons.csv")
    people = pd.read_csv("data/lahman2025/People.csv")
    crosswalk = pd.read_csv("data/id_crosswalk.csv")

    id_map = dict(zip(crosswalk.mlbam_id, crosswalk.lahman_id))
    people_birth = dict(zip(people.playerID, people.birthYear))

    milb_bat = milb[milb.PA > 0].copy()
    milb_bat["pid_int"] = milb_bat.playerID.astype(float).astype(int)
    milb_bat["lahman_id"] = milb_bat.pid_int.map(id_map)
    milb_bat = milb_bat[milb_bat.lahman_id.notna()].copy()

    print(f"\nMLB seasons: {len(mlb)}, MiLB linked: {len(milb_bat)}")

    # Build career sequences
    print("Building career sequences...")
    t0 = time.perf_counter()

    mlb_by_player = mlb.groupby("playerID")
    sequences, masks_list = [], []
    y_career, y_peak, y_next = [], [], []
    player_names = []

    for lahman_id, milb_group in milb_bat.groupby("lahman_id"):
        if lahman_id not in mlb_by_player.groups:
            continue
        mlb_group = mlb_by_player.get_group(lahman_id)

        timeline = []
        for _, row in milb_group.sort_values("yearID").iterrows():
            timeline.append(build_season_vector(row, row.get("level", "AAA"), people_birth))

        for _, row in mlb_group.sort_values("yearID").iterrows():
            timeline.append(build_season_vector(row, "MLB", people_birth))

        if len(timeline) < 3:
            continue

        if len(timeline) > MAX_SEQ_LEN:
            timeline = timeline[:MAX_SEQ_LEN]

        seq = np.zeros((MAX_SEQ_LEN, FEAT_DIM), dtype=np.float32)
        mask = np.zeros(MAX_SEQ_LEN, dtype=np.float32)
        for i, vec in enumerate(timeline):
            seq[i] = vec
            mask[i] = 1.0

        sequences.append(seq)
        masks_list.append(mask)
        y_career.append(mlb_group.bravs_war_eq.sum())
        y_peak.append(mlb_group.bravs_war_eq.max())
        y_next.append(mlb_group.sort_values("yearID").iloc[-1].bravs_war_eq)
        player_names.append(mlb_group.iloc[0]["name"])

    X = np.array(sequences)
    M = np.array(masks_list)
    y_c = np.array(y_career, dtype=np.float32)
    y_p = np.array(y_peak, dtype=np.float32)
    y_n = np.array(y_next, dtype=np.float32)

    print(f"Built {len(X)} sequences in {time.perf_counter()-t0:.1f}s")
    print(f"Shape: {X.shape}")

    # Normalize targets
    yc_m, yc_s = y_c.mean(), y_c.std() + 1e-8
    yp_m, yp_s = y_p.mean(), y_p.std() + 1e-8
    yn_m, yn_s = y_n.mean(), y_n.std() + 1e-8

    # Clean NaN/inf from data
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    M = np.nan_to_num(M, nan=0.0)
    y_c = np.nan_to_num(y_c, nan=0.0)
    y_p = np.nan_to_num(y_p, nan=0.0)
    y_n = np.nan_to_num(y_n, nan=0.0)

    # Clamp extreme values
    X = np.clip(X, -10.0, 10.0)
    y_c = np.clip(y_c, -50, 200)
    y_p = np.clip(y_p, -10, 20)
    y_n = np.clip(y_n, -10, 20)

    print(f"Data stats: X range [{X.min():.2f}, {X.max():.2f}], "
          f"y_career range [{y_c.min():.1f}, {y_c.max():.1f}]")
    print(f"Any NaN in X: {np.isnan(X).any()}, y_c: {np.isnan(y_c).any()}")

    X_t = torch.tensor(X, device=DEVICE)
    M_t = torch.tensor(M, device=DEVICE)
    yc_t = torch.tensor((y_c - yc_m) / yc_s, device=DEVICE)
    yp_t = torch.tensor((y_p - yp_m) / yp_s, device=DEVICE)
    yn_t = torch.tensor((y_n - yn_m) / yn_s, device=DEVICE)

    # Verify no NaN in tensors
    assert not torch.isnan(X_t).any(), "NaN in X_t!"
    assert not torch.isnan(yc_t).any(), "NaN in yc_t!"

    n = len(X_t)
    n_val = int(n * 0.15)
    perm = torch.randperm(n)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    model = UniversalPlayerTransformer(dropout=0.1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")
    print("Training...")

    t0 = time.perf_counter()
    best_val = float("inf")
    best_state = None

    for epoch in range(500):
        model.train()
        pc, pp, pn = model(X_t[train_idx], M_t[train_idx])

        # Huber loss is more robust to outliers than MSE
        loss = F.smooth_l1_loss(pc, yc_t[train_idx]) + F.smooth_l1_loss(pp, yp_t[train_idx]) + F.smooth_l1_loss(pn, yn_t[train_idx])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                vc, vp, vn = model(X_t[val_idx], M_t[val_idx])
                vl = F.mse_loss(vc, yc_t[val_idx]) + F.mse_loss(vp, yp_t[val_idx]) + F.mse_loss(vn, yn_t[val_idx])

                vc_r = vc * yc_s + yc_m
                yc_r = yc_t[val_idx] * yc_s + yc_m
                r = np.corrcoef(vc_r.cpu().numpy(), yc_r.cpu().numpy())[0, 1]
                rmse = (vc_r - yc_r).pow(2).mean().sqrt().item()

            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            print(f"  Epoch {epoch+1}: loss={loss.item():.4f} val={vl.item():.4f} career_r={r:.3f} rmse={rmse:.1f}")

    if best_state:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - t0
    print(f"Training: {elapsed:.1f}s")

    # Full eval
    model.eval()
    with torch.no_grad():
        ac, ap, an = model(X_t, M_t)
        ac = (ac * yc_s + yc_m).cpu().numpy()
        ap = (ap * yp_s + yp_m).cpu().numpy()
        an = (an * yn_s + yn_m).cpu().numpy()

    cr = np.corrcoef(ac, y_c)[0, 1]
    pr = np.corrcoef(ap, y_p)[0, 1]
    nr = np.corrcoef(an, y_n)[0, 1]
    c_rmse = np.sqrt(((ac - y_c)**2).mean())

    print(f"\nFULL RESULTS:")
    print(f"  Career WAR: r={cr:.4f}, RMSE={c_rmse:.1f}")
    print(f"  Peak WAR:   r={pr:.4f}")
    print(f"  Next season: r={nr:.4f}")

    # Known players
    print(f"\n--- Known Player Predictions ---")
    names = np.array(player_names)
    for target in ["Mike Trout", "Juan Soto", "Bryce Harper", "Mookie Betts",
                    "Aaron Judge", "Shohei Ohtani", "Ronald Acuna"]:
        last = target.split()[-1].lower()
        mask = np.array([last in n.lower() for n in names])
        if mask.sum() == 0:
            continue
        idx = np.where(mask)[0]
        best = idx[y_c[idx].argmax()]
        print(f"  {player_names[best]:<22} actual career={y_c[best]:+.1f} "
              f"pred={ac[best]:+.1f} | peak actual={y_p[best]:+.1f} pred={ap[best]:+.1f}")

    # Save
    os.makedirs("models", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"feat_dim": FEAT_DIM, "d_model": 128, "nhead": 4, "num_layers": 4},
        "normalization": {
            "career": (float(yc_m), float(yc_s)),
            "peak": (float(yp_m), float(yp_s)),
            "next": (float(yn_m), float(yn_s)),
        },
        "n_sequences": len(X),
        "n_params": n_params,
    }, "models/universal_player_transformer.pt")

    print(f"\nSaved models/universal_player_transformer.pt")
    print(f"  {n_params:,} parameters")
    print(f"  Trained on {len(X)} MiLB+MLB career sequences")
    print(f"  Input: {MAX_SEQ_LEN} seasons x {FEAT_DIM} features")
    print("=" * 70)


if __name__ == "__main__":
    main()
