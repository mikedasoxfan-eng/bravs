"""Pixel-perfect Daily Pitcher Performance Card — matches prototype exactly."""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from io import BytesIO
import requests
from pybaseball import statcast, cache
cache.enable()

# ── Colors exactly matching prototype ──
C = {
    "FF": "#c4363a", "SI": "#1a6b54", "CH": "#d4a017", "ST": "#2eaa60",
    "SL": "#2e9ec4", "CU": "#7b44bc", "FC": "#d06830", "FS": "#8c564b",
    "KC": "#7b44bc", "SV": "#2eaa60", "KN": "#888888",
}
NAMES = {
    "FF": "4-Seam", "SI": "Sinker", "CH": "Changeup", "ST": "Sweeper",
    "SL": "Slider", "CU": "Curveball", "FC": "Cutter", "FS": "Splitter",
    "KC": "Knuckle Curve", "SV": "Slurve", "KN": "Knuckleball",
}
TEAM_IDS = {
    "NYY": 147, "BOS": 111, "LAD": 119, "NYM": 121, "CHC": 112,
    "ATL": 144, "HOU": 117, "PHI": 143, "SF": 137, "STL": 138,
    "CLE": 114, "DET": 116, "MIN": 142, "SEA": 136, "TOR": 141,
    "ARI": 109, "BAL": 110, "CIN": 113, "COL": 115, "KC": 118,
    "LAA": 108, "MIA": 146, "MIL": 158, "OAK": 133, "PIT": 134,
    "SD": 135, "TB": 139, "TEX": 140, "WSH": 120, "CWS": 145,
    "ATH": 133, "CHW": 145,
}
FONT = "DejaVu Sans"


def fetch(pitcher_id, date):
    df = statcast(date, date)
    if df.empty: raise ValueError(f"No data for {date}")
    p = df[df.pitcher == pitcher_id]
    if p.empty: raise ValueError(f"Pitcher {pitcher_id} not found on {date}")
    return p.copy()


def find_pitcher(name, date):
    df = statcast(date, date)
    if df.empty: raise ValueError(f"No data for {date}")
    nl = name.lower()
    m = df[df.player_name.str.lower().str.contains(nl.split()[-1], na=False)]
    if m.empty: raise ValueError(f"'{name}' not found on {date}")
    pid = m.pitcher.value_counts().index[0]
    return pid, m[m.pitcher == pid].player_name.iloc[0]


def _zone(ax, sub, title):
    """Strike zone chart with density blobs."""
    ax.set_xlim(-2.0, 2.0); ax.set_ylim(0.8, 4.2)
    ax.set_aspect('equal')
    # Zone box
    ax.add_patch(patches.Rectangle((-0.83, 1.5), 1.66, 2.0, lw=1.0, ec='#333', fc='none', zorder=3))
    for x in [-0.28, 0.28]:
        ax.plot([x, x], [1.5, 3.5], c='#d0d0d0', lw=0.4, zorder=2)
    for z in [2.17, 2.83]:
        ax.plot([-0.83, 0.83], [z, z], c='#d0d0d0', lw=0.4, zorder=2)
    # Density blobs (big transparent circles)
    for pt in sub.pitch_type.unique():
        s = sub[sub.pitch_type == pt]
        px, pz = s.plate_x.dropna(), s.plate_z.dropna()
        if len(px) < 2: continue
        col = C.get(pt, "#999")
        ax.scatter(px.mean(), pz.mean(), s=2500, c=[to_rgba(col, 0.15)], ec='none', zorder=1)
    # Dots
    for pt in sub.pitch_type.unique():
        s = sub[sub.pitch_type == pt]
        col = C.get(pt, "#999")
        ax.scatter(s.plate_x, s.plate_z, c=col, s=14, alpha=0.9, ec='white', lw=0.2, zorder=4)
    n = len(sub)
    ax.set_title(f"{title}\n({n} Pitches)", fontsize=6.5, fontfamily=FONT, fontweight='bold', pad=2, linespacing=1.4)
    ax.tick_params(length=0, labelsize=0)
    for sp in ax.spines.values(): sp.set_visible(False)


def _breaks(ax, df, p_throws):
    """Movement chart with arm angle line."""
    for pt in df.pitch_type.unique():
        s = df[df.pitch_type == pt]
        hb, ivb = s.pfx_x.dropna() * 12, s.pfx_z.dropna() * 12
        ax.scatter(hb, ivb, c=C.get(pt, "#999"), s=14, alpha=0.9, ec='white', lw=0.2, zorder=4)
    ax.axhline(0, c='#e0e0e0', lw=0.4); ax.axvline(0, c='#e0e0e0', lw=0.4)
    # Arm angle
    rx, rz = df.release_pos_x.mean(), df.release_pos_z.mean()
    if pd.notna(rx) and pd.notna(rz):
        ang = np.degrees(np.arctan2(rz, abs(rx)))
        rad = np.radians(ang)
        dx = 28 * np.cos(rad) * (1 if p_throws == "R" else -1)
        ax.plot([0, dx], [0, 28 * np.sin(rad)], '--', c='#aaa', lw=0.6, zorder=2)
        ax.set_title(f"Pitch Breaks · Arm Angle: {ang:.0f}°", fontsize=6.5, fontfamily=FONT, fontweight='bold', pad=2)
    else:
        ax.set_title("Pitch Breaks", fontsize=6.5, fontfamily=FONT, fontweight='bold', pad=2)
    ax.set_xlim(-22, 22); ax.set_ylim(-22, 22)
    ax.set_xlabel("Horizontal Break (in)", fontsize=5, fontfamily=FONT, labelpad=1)
    ax.set_ylabel("Induced Vert. Break (in)", fontsize=5, fontfamily=FONT, labelpad=1)
    ax.tick_params(labelsize=4.5, length=2, pad=1)


def _woba_bg(v):
    if v <= 0.001: return "#3ddc84"
    if v <= 0.280: return "#3ddc84"
    if v <= 0.350: return "#f4d03f"
    return "#e74c3c"


def generate_card(pitcher_id, date, output_path=None):
    print(f"Fetching pitcher {pitcher_id} on {date}...")
    df = fetch(pitcher_id, date)
    print(f"  {len(df)} pitches")

    # Player info
    nm = df.player_name.iloc[0]
    player_name = " ".join(reversed(nm.split(", "))) if ", " in nm else nm
    p_throws = df.p_throws.iloc[0]
    hand = "LHP" if p_throws == "L" else "RHP"
    home, away = df.home_team.iloc[0], df.away_team.iloc[0]
    pitcher_team = home if df.inning_topbot.iloc[0] == "Top" else away
    opponent = away if pitcher_team == home else home
    year = int(df.game_year.iloc[0])

    # Box score
    pa = df.at_bat_number.nunique()
    k = len(df[df.events.isin(['strikeout', 'strikeout_double_play'])])
    bb = len(df[df.events.isin(['walk'])])
    h = len(df[df.events.isin(['single', 'double', 'triple', 'home_run'])])
    hbp = len(df[df.events.isin(['hit_by_pitch'])])
    er = len(df[df.events.isin(['home_run'])])
    strikes = len(df[df.type.isin(['S', 'X'])])
    spct = strikes / len(df) * 100 if len(df) else 0
    whiffs = len(df[df.description.isin(['swinging_strike', 'swinging_strike_blocked', 'foul_tip'])])
    outs = 0
    for _, r in df[df.events.notna()].iterrows():
        e = r.events
        if e in ['strikeout', 'strikeout_double_play', 'field_out', 'force_out',
                 'grounded_into_double_play', 'fielders_choice_out', 'sac_fly',
                 'sac_bunt', 'double_play', 'field_error']:
            outs += 1
            if 'double_play' in e: outs += 1
    ip = outs / 3

    # ════════════════════════════════════════════════════
    # FIGURE — 10×11 at 150 DPI = 1500×1650 px
    # ════════════════════════════════════════════════════
    fig = plt.figure(figsize=(10, 11), dpi=150, facecolor='white')

    # ── HEADER ──
    # Headshot
    try:
        url = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/{pitcher_id}/headshot/67/current"
        from PIL import Image
        img = Image.open(BytesIO(requests.get(url, timeout=5).content))
        axh = fig.add_axes([0.04, 0.895, 0.11, 0.085])
        axh.imshow(img); axh.axis('off')
    except: pass
    # Team logo
    try:
        tid = TEAM_IDS.get(pitcher_team, 0)
        if tid:
            logo = Image.open(BytesIO(requests.get(f"https://midfield.mlbstatic.com/v1/team/{tid}/spots/72", timeout=5).content))
            axl = fig.add_axes([0.83, 0.895, 0.12, 0.085])
            axl.imshow(logo); axl.axis('off')
    except: pass

    fig.text(0.50, 0.965, player_name, ha='center', fontsize=20, fontweight='bold', fontfamily=FONT)
    fig.text(0.50, 0.948, hand, ha='center', fontsize=8, color='#888', fontfamily=FONT)
    fig.text(0.50, 0.932, "Daily Pitching Summary", ha='center', fontsize=12, fontweight='bold', fontfamily=FONT)
    fig.text(0.50, 0.917, f"{year} MLB Season", ha='center', fontsize=7, color='#aaa', fontfamily=FONT,
             bbox=dict(boxstyle='round,pad=0.25', fc='#f0f0f0', ec='none'))

    # Game info
    fig.text(0.50, 0.895, f"{date} vs {opponent}", ha='center', fontsize=10, fontweight='bold',
             fontfamily=FONT, bbox=dict(boxstyle='round,pad=0.35', fc='#f8f8f8', ec='#ccc', lw=0.6))

    # ── BOX SCORE BAR ──
    by = 0.870
    # Top line
    fig.add_axes([0.06, by + 0.010, 0.88, 0.0005]).axhline(0, c='#ccc', lw=0.6); plt.gca().axis('off')
    items = [("IP", f"{ip:.1f}"), ("PA", f"{pa:02d}"), ("ER", str(er)), ("H", str(h)),
             ("K", str(k)), ("BB", str(bb)), ("HBP", str(hbp)), ("Strike%", f"{spct:.1f}%"), ("Whiffs", str(whiffs))]
    w = 0.88 / len(items)
    for i, (lbl, val) in enumerate(items):
        cx = 0.06 + w * i + w / 2
        fig.text(cx, by + 0.006, lbl, ha='center', fontsize=5.5, color='#999', fontfamily=FONT)
        fig.text(cx, by - 0.008, val, ha='center', fontsize=8.5, fontweight='bold', fontfamily=FONT)
        if i < len(items) - 1:
            sx = 0.06 + w * (i + 1)
            fig.add_axes([sx - 0.001, by - 0.012, 0.0005, 0.025]).axvline(0, c='#ddd', lw=0.4); plt.gca().axis('off')
    # Bottom line
    fig.add_axes([0.06, by - 0.016, 0.88, 0.0005]).axhline(0, c='#ccc', lw=0.6); plt.gca().axis('off')

    # ── THREE CHARTS ──
    chart_y, chart_h = 0.575, 0.255
    ax1 = fig.add_axes([0.05, chart_y, 0.27, chart_h])
    _zone(ax1, df[df.stand == "L"], "Pitch Locations vs LHH")

    ax2 = fig.add_axes([0.37, chart_y, 0.26, chart_h])
    _breaks(ax2, df, p_throws)

    ax3 = fig.add_axes([0.69, chart_y, 0.27, chart_h])
    _zone(ax3, df[df.stand == "R"], "Pitch Locations vs RHH")

    # ── LEGEND ──
    used = [pt for pt in df.pitch_type.value_counts().index if pt in NAMES]
    lx = 0.5 - len(used) * 0.05
    for i, pt in enumerate(used):
        x = lx + i * 0.10
        fig.text(x, 0.560, "●", ha='center', fontsize=9, color=C.get(pt, "#999"), fontfamily=FONT)
        fig.text(x + 0.015, 0.560, NAMES.get(pt, pt), ha='left', fontsize=6, fontfamily=FONT, color='#444')

    # ── STATS TABLE ──
    hdrs = ["Pitch Name", "Count", "Pitch%", "Velocity", "CSW", "BB", "SwStr%",
            "MAX V", "MAX S", "MAX IVB", "Ext", "CSp", "SwngM%", "Zone%", "Chase%", "Heart%", "Shadow%", "wOBA"]
    # Column x positions and widths
    ncol = len(hdrs)
    tbl_left, tbl_right = 0.04, 0.96
    tbl_w = tbl_right - tbl_left
    # First column wider for pitch name
    cw = [0.08] + [tbl_w * 0.054] * (ncol - 1)
    # Adjust to fill
    remaining = tbl_w - sum(cw)
    cw[0] += remaining

    cx_starts = []
    x = tbl_left
    for w in cw:
        cx_starts.append(x)
        x += w

    ty = 0.530  # table top y
    rh = 0.026  # row height
    hh = 0.020  # header height

    # Header
    fig.add_axes([tbl_left, ty - 0.001, tbl_w, 0.0005]).axhline(0, c='#bbb', lw=0.5); plt.gca().axis('off')
    for i, (hdr, cx) in enumerate(zip(hdrs, cx_starts)):
        fig.text(cx + cw[i] / 2, ty + 0.006, hdr, ha='center', va='center',
                fontsize=4.5, fontweight='bold', color='#666', fontfamily=FONT)
    fig.add_axes([tbl_left, ty - hh + 0.004, tbl_w, 0.0005]).axhline(0, c='#bbb', lw=0.5); plt.gca().axis('off')

    # Compute and draw rows
    pitch_types = [pt for pt in df.pitch_type.value_counts().index if pt in NAMES]
    for j, pt in enumerate(pitch_types):
        sub = df[df.pitch_type == pt]
        n = len(sub)
        vy = sub.release_speed.dropna()
        sp = sub.release_spin_rate.dropna()
        pz = sub.pfx_z.dropna() * 12
        ext = sub.release_extension.dropna()
        cs = len(sub[sub.description == 'called_strike'])
        ss = len(sub[sub.description.isin(['swinging_strike', 'swinging_strike_blocked', 'foul_tip'])])
        csw = (cs + ss) / n * 100 if n else 0
        swm = ss / n * 100 if n else 0
        zn = len(sub[sub.zone.between(1, 9)])
        zpct = zn / n * 100 if n else 0
        oz = sub[~sub.zone.between(1, 9)]
        ch = len(oz[oz.description.isin(['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip',
                                          'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'])])
        cpct = ch / len(oz) * 100 if len(oz) else 0
        ht = len(sub[sub.zone.isin([2, 5, 8])]) / n * 100 if n else 0
        sh = len(sub[sub.zone.isin([11, 12, 13, 14])]) / n * 100 if n else 0
        con = sub[sub.events.notna()]
        hts = con[con.events.isin(['single', 'double', 'triple', 'home_run'])]
        woba = 0
        if len(con) > 0:
            woba = (len(hts[hts.events == 'single']) * 0.88 + len(hts[hts.events == 'double']) * 1.25 +
                    len(hts[hts.events == 'triple']) * 1.58 + len(hts[hts.events == 'home_run']) * 2.02) / len(con)
        bls = len(sub[sub.type == 'B'])

        vals = [
            NAMES.get(pt, pt), str(n), f"{n/len(df)*100:.1f}%",
            f"{vy.mean():.1f}" if len(vy) else "-", f"{csw:.1f}", str(bls), f"{swm:.1f}",
            f"{vy.max():.1f}" if len(vy) else "-", f"{sp.max():.0f}" if len(sp) else "-",
            f"{pz.mean():.1f}" if len(pz) else "-", f"{ext.mean():.1f}" if len(ext) else "-",
            f"{cs}", f"{swm:.1f}", f"{zpct:.1f}", f"{cpct:.1f}", f"{ht:.1f}", f"{sh:.1f}", f"{woba:.3f}",
        ]

        ry = ty - hh - j * rh
        col = C.get(pt, "#999")

        # Full row tinted background
        bg = patches.FancyBboxPatch((tbl_left, ry - rh / 2 + 0.003), tbl_w, rh - 0.002,
                                     boxstyle="square,pad=0", fc=to_rgba(col, 0.08), ec='none',
                                     transform=fig.transFigure, clip_on=False)
        fig.add_artist(bg)

        # Pitch name cell — stronger colored background
        name_bg = patches.FancyBboxPatch((cx_starts[0], ry - rh / 2 + 0.003), cw[0], rh - 0.002,
                                          boxstyle="square,pad=0", fc=to_rgba(col, 0.55), ec='none',
                                          transform=fig.transFigure, clip_on=False)
        fig.add_artist(name_bg)

        # wOBA cell colored
        try:
            wv = float(vals[-1])
            woba_cell = patches.FancyBboxPatch((cx_starts[-1], ry - rh / 2 + 0.003), cw[-1], rh - 0.002,
                                                boxstyle="square,pad=0", fc=to_rgba(_woba_bg(wv), 0.40), ec='none',
                                                transform=fig.transFigure, clip_on=False)
            fig.add_artist(woba_cell)
        except: pass

        for i, (val, cx, w) in enumerate(zip(vals, cx_starts, cw)):
            fc = 'white' if i == 0 else '#222'
            fw = 'bold' if i == 0 else 'normal'
            fig.text(cx + w / 2, ry, val, ha='center', va='center',
                    fontsize=5, fontfamily=FONT, fontweight=fw, color=fc)

    # Totals row
    tot_y = ty - hh - len(pitch_types) * rh
    fig.add_axes([tbl_left, tot_y + rh / 2 - 0.001, tbl_w, 0.0005]).axhline(0, c='#bbb', lw=0.5); plt.gca().axis('off')
    fig.text(cx_starts[0] + cw[0] / 2, tot_y - 0.003, "All", ha='center', va='center',
            fontsize=5.5, fontweight='bold', fontfamily=FONT)
    fig.text(cx_starts[1] + cw[1] / 2, tot_y - 0.003, str(len(df)), ha='center', va='center',
            fontsize=5, fontweight='bold', fontfamily=FONT)
    fig.text(cx_starts[2] + cw[2] / 2, tot_y - 0.003, "100.0%", ha='center', va='center',
            fontsize=5, fontfamily=FONT)

    # Footer
    fig.text(0.95, 0.005, "Data: MLB", ha='right', fontsize=5, color='#bbb', fontfamily=FONT, style='italic')

    # Border
    fig.add_artist(patches.FancyBboxPatch((0.015, 0.003), 0.97, 0.993,
                   boxstyle="round,pad=0.005", lw=1.0, ec='#ccc', fc='none',
                   transform=fig.transFigure, clip_on=False))

    # Save
    if not output_path:
        safe = player_name.replace(" ", "_").replace(",", "")
        output_path = f"output/{safe}_{date}.png"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pitcher", type=str)
    parser.add_argument("--pitcher-id", type=int)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    if args.pitcher_id:
        pid = args.pitcher_id
    elif args.pitcher:
        pid, found = find_pitcher(args.pitcher, args.date)
        print(f"Found: {found} (ID: {pid})")
    else:
        parser.error("Need --pitcher or --pitcher-id")
    generate_card(pid, args.date, args.output)


if __name__ == "__main__":
    main()
