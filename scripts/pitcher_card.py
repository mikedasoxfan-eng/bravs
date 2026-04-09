"""Daily Pitcher Performance Card Generator — matches prototype exactly.

Usage:
    python scripts/pitcher_card.py --pitcher-id 543037 --date 2024-06-19
    python scripts/pitcher_card.py --pitcher "Skubal" --date 2024-08-01
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from matplotlib.table import Table
from io import BytesIO
import requests
from pybaseball import statcast, cache
cache.enable()

PITCH_COLORS = {
    "FF": "#c4363a", "SI": "#1a6b54", "CH": "#dba816", "ST": "#2ecc71",
    "SL": "#32a4c8", "CU": "#7b44bc", "FC": "#e07020", "FS": "#8c564b",
    "KC": "#a0a028", "SV": "#2ecc71", "KN": "#7f7f7f",
}
PITCH_NAMES = {
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
    "ATH": 133,
}


def fetch_game_data(pitcher_id, date):
    df = statcast(date, date)
    if df.empty:
        raise ValueError(f"No Statcast data for {date}")
    pitcher_df = df[df.pitcher == pitcher_id].copy()
    if pitcher_df.empty:
        raise ValueError(f"Pitcher {pitcher_id} did not pitch on {date}")
    return pitcher_df


def find_pitcher_id(name, date):
    df = statcast(date, date)
    if df.empty:
        raise ValueError(f"No data for {date}")
    name_lower = name.lower()
    # Statcast names are "Last, First" — search both orders
    matches = df[
        df.player_name.str.lower().str.contains(name_lower, na=False) |
        df.player_name.str.lower().str.replace(", ", " ").str.contains(name_lower, na=False)
    ]
    if matches.empty:
        last = name.split()[-1].lower()
        matches = df[df.player_name.str.lower().str.startswith(last, na=False)]
    if matches.empty:
        raise ValueError(f"No pitcher matching '{name}' on {date}")
    # Pick the one who threw the most pitches
    pid = matches.pitcher.value_counts().index[0]
    return pid, matches[matches.pitcher == pid].player_name.iloc[0]


def draw_zone(ax, df_subset, title):
    """Draw strike zone with scatter dots and density blobs."""
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(0.5, 4.5)
    ax.set_aspect('equal')

    # Strike zone box
    zone = patches.Rectangle((-0.83, 1.5), 1.66, 2.0,
                             lw=1.2, edgecolor='#333333', facecolor='none', zorder=3)
    ax.add_patch(zone)
    # Inner grid
    for x in [-0.28, 0.28]:
        ax.plot([x, x], [1.5, 3.5], color='#cccccc', lw=0.5, zorder=2)
    for z in [2.17, 2.83]:
        ax.plot([-0.83, 0.83], [z, z], color='#cccccc', lw=0.5, zorder=2)

    # Density blobs per pitch type (using KDE-like gaussian scatter)
    for pt in df_subset.pitch_type.unique():
        sub = df_subset[df_subset.pitch_type == pt]
        px = sub.plate_x.dropna()
        pz = sub.plate_z.dropna()
        if len(px) < 3:
            continue
        color = PITCH_COLORS.get(pt, "#999")
        rgba = to_rgba(color, alpha=0.12)
        # Draw filled scatter with large size as density proxy
        ax.scatter(px, pz, s=350, c=[rgba], edgecolors='none', zorder=1)

    # Actual dots
    for pt in df_subset.pitch_type.unique():
        sub = df_subset[df_subset.pitch_type == pt]
        color = PITCH_COLORS.get(pt, "#999")
        ax.scatter(sub.plate_x, sub.plate_z, c=color, s=18, alpha=0.85,
                   edgecolors='white', linewidth=0.3, zorder=4)

    ax.set_title(title, fontsize=7, fontfamily='sans-serif', fontweight='bold', pad=4)
    ax.tick_params(labelsize=0, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_breaks(ax, df, p_throws):
    """Draw pitch movement chart with arm angle line."""
    for pt in df.pitch_type.unique():
        sub = df[df.pitch_type == pt]
        hb = sub.pfx_x.dropna() * 12
        ivb = sub.pfx_z.dropna() * 12
        color = PITCH_COLORS.get(pt, "#999")
        ax.scatter(hb, ivb, c=color, s=18, alpha=0.85,
                   edgecolors='white', linewidth=0.3, zorder=4)

    ax.axhline(0, color='#dddddd', lw=0.5, zorder=1)
    ax.axvline(0, color='#dddddd', lw=0.5, zorder=1)

    # Arm angle line (estimated from release point)
    rel_x = df.release_pos_x.mean()
    rel_z = df.release_pos_z.mean()
    if pd.notna(rel_x) and pd.notna(rel_z):
        angle_deg = np.degrees(np.arctan2(rel_z, abs(rel_x)))
        angle_rad = np.radians(angle_deg)
        line_len = 30
        ax.plot([0, line_len * np.cos(angle_rad) * (1 if p_throws == "R" else -1)],
                [0, line_len * np.sin(angle_rad)],
                '--', color='#999999', lw=0.8, zorder=2)
        ax.set_title(f"Pitch Breaks · Arm Angle: {angle_deg:.0f}°",
                     fontsize=7, fontfamily='sans-serif', fontweight='bold', pad=4)
    else:
        ax.set_title("Pitch Breaks", fontsize=7, fontfamily='sans-serif', fontweight='bold', pad=4)

    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_xlabel("Horizontal Break (in)", fontsize=5, fontfamily='sans-serif')
    ax.set_ylabel("Induced Vert. Break (in)", fontsize=5, fontfamily='sans-serif')
    ax.tick_params(labelsize=5)


def woba_color(val):
    """Color code wOBA: green=good for pitcher (low), red=bad (high)."""
    if val <= 0.250:
        return "#2ecc71"
    elif val <= 0.350:
        return "#f0c808"
    else:
        return "#e74c3c"


def generate_card(pitcher_id, date, output_path=None):
    print(f"Fetching data for pitcher {pitcher_id} on {date}...")
    df = fetch_game_data(pitcher_id, date)
    print(f"  {len(df)} pitches")

    name_raw = df.player_name.iloc[0]
    player_name = " ".join(reversed(name_raw.split(", "))) if ", " in name_raw else name_raw
    p_throws = df.p_throws.iloc[0]
    hand = "LHP" if p_throws == "L" else "RHP"
    home = df.home_team.iloc[0]
    away = df.away_team.iloc[0]
    pitcher_team = home if df.inning_topbot.iloc[0] == "Top" else away
    opponent = away if pitcher_team == home else home
    year = int(df.game_year.iloc[0])

    # Box score
    pa = df.at_bat_number.nunique()
    k = len(df[df.events.isin(['strikeout', 'strikeout_double_play'])])
    bb = len(df[df.events.isin(['walk'])])
    hits = len(df[df.events.isin(['single', 'double', 'triple', 'home_run'])])
    hbp_count = len(df[df.events.isin(['hit_by_pitch'])])
    er = len(df[df.events.isin(['home_run'])])  # crude
    strikes = len(df[df.type.isin(['S', 'X'])])
    strike_pct = strikes / len(df) * 100 if len(df) else 0
    whiffs = len(df[df.description.isin(['swinging_strike', 'swinging_strike_blocked', 'foul_tip'])])
    outs = 0
    for _, r in df[df.events.notna()].iterrows():
        e = r.events
        if e in ['strikeout', 'strikeout_double_play', 'field_out', 'force_out',
                 'grounded_into_double_play', 'fielders_choice_out', 'sac_fly',
                 'sac_bunt', 'double_play', 'field_error', 'caught_stealing_2b']:
            outs += 1
            if 'double_play' in e:
                outs += 1
    ip = outs / 3

    # ====== FIGURE ======
    fig = plt.figure(figsize=(12, 12), facecolor='white', dpi=150)

    # -- HEADER --
    # Headshot
    try:
        url = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/{pitcher_id}/headshot/67/current"
        from PIL import Image
        img = Image.open(BytesIO(requests.get(url, timeout=5).content))
        ax_head = fig.add_axes([0.03, 0.88, 0.12, 0.10])
        ax_head.imshow(img)
        ax_head.axis('off')
    except Exception:
        pass

    # Team logo
    try:
        tid = TEAM_IDS.get(pitcher_team, 0)
        if tid:
            logo_url = f"https://www.mlbstatic.com/team-logos/{tid}.svg"
            # PNG fallback
            logo_url = f"https://www.mlbstatic.com/team-logos/team-cap-on-light/{tid}.svg"
            from PIL import Image
            logo_img = Image.open(BytesIO(requests.get(
                f"https://midfield.mlbstatic.com/v1/team/{tid}/spots/72", timeout=5).content))
            ax_logo = fig.add_axes([0.82, 0.885, 0.13, 0.095])
            ax_logo.imshow(logo_img)
            ax_logo.axis('off')
    except Exception:
        pass

    fig.text(0.50, 0.96, player_name, ha='center', fontsize=24, fontweight='bold', fontfamily='sans-serif')
    fig.text(0.50, 0.94, f"{hand}", ha='center', fontsize=9, color='#777777', fontfamily='sans-serif')
    fig.text(0.50, 0.925, "Daily Pitching Summary", ha='center', fontsize=13, fontweight='bold', fontfamily='sans-serif')
    fig.text(0.50, 0.91, f"{year} MLB Season", ha='center', fontsize=8, color='#999999', fontfamily='sans-serif',
             bbox=dict(boxstyle='round,pad=0.3', fc='#eeeeee', ec='none'))

    # Game info
    fig.text(0.50, 0.885, f"{date} vs {opponent}", ha='center', fontsize=11, fontweight='bold',
             fontfamily='sans-serif', bbox=dict(boxstyle='round,pad=0.4', fc='#f5f5f5', ec='#cccccc', lw=0.8))

    # Box score bar
    box_y = 0.86
    box_items = [
        ("IP", f"{ip:.1f}"), ("PA", str(pa)), ("ER", str(er)), ("H", str(hits)),
        ("K", str(k)), ("BB", str(bb)), ("HBP", str(hbp_count)),
        ("Strike%", f"{strike_pct:.1f}%"), ("Whiffs", str(whiffs)),
    ]
    n_items = len(box_items)
    for i, (label, val) in enumerate(box_items):
        x = 0.08 + i * (0.84 / n_items)
        fig.text(x, box_y + 0.008, label, ha='center', fontsize=6.5, color='#999999', fontfamily='sans-serif')
        fig.text(x, box_y - 0.008, val, ha='center', fontsize=9, fontweight='bold', fontfamily='sans-serif')
        if i < n_items - 1:
            sep_x = x + 0.84 / n_items / 2
            line_ax = fig.add_axes([sep_x, box_y - 0.015, 0.001, 0.03])
            line_ax.axvline(0, color='#dddddd', lw=0.5)
            line_ax.axis('off')

    # -- THREE CHARTS --
    ax_lhh = fig.add_axes([0.04, 0.55, 0.28, 0.26])
    draw_zone(ax_lhh, df[df.stand == "L"], "Pitch Locations vs LHH")

    ax_brk = fig.add_axes([0.37, 0.55, 0.26, 0.26])
    draw_breaks(ax_brk, df, p_throws)

    ax_rhh = fig.add_axes([0.68, 0.55, 0.28, 0.26])
    draw_zone(ax_rhh, df[df.stand == "R"], "Pitch Locations vs RHH")

    # -- LEGEND --
    used = [pt for pt in df.pitch_type.value_counts().index if pt in PITCH_NAMES]
    lx_start = 0.5 - len(used) * 0.055
    for i, pt in enumerate(used):
        x = lx_start + i * 0.11
        fig.text(x, 0.535, "●", ha='center', fontsize=11, color=PITCH_COLORS.get(pt, "#999"))
        fig.text(x + 0.018, 0.535, PITCH_NAMES.get(pt, pt), ha='left', fontsize=7, fontfamily='sans-serif')

    # -- STATS TABLE --
    # Compute stats per pitch type
    pitch_types = [pt for pt in df.pitch_type.value_counts().index if pt in PITCH_NAMES]
    table_data = []
    for pt in pitch_types:
        sub = df[df.pitch_type == pt]
        n = len(sub)
        velo = sub.release_speed.dropna()
        spin = sub.release_spin_rate.dropna()
        pfx_z_in = sub.pfx_z.dropna() * 12
        ext = sub.release_extension.dropna()
        cs = len(sub[sub.description == 'called_strike'])
        ss = len(sub[sub.description.isin(['swinging_strike', 'swinging_strike_blocked', 'foul_tip'])])
        csw = (cs + ss) / n * 100 if n else 0
        swm = ss / n * 100 if n else 0
        zone_in = len(sub[sub.zone.between(1, 9)])
        zone_pct = zone_in / n * 100 if n else 0
        out_zone = sub[~sub.zone.between(1, 9)]
        chases = len(out_zone[out_zone.description.isin(['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip',
                                                          'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'])])
        chase_pct = chases / len(out_zone) * 100 if len(out_zone) else 0
        heart = len(sub[sub.zone.isin([2, 5, 8])]) / n * 100 if n else 0
        shadow = len(sub[sub.zone.isin([11, 12, 13, 14])]) / n * 100 if n else 0

        contact = sub[sub.events.notna()]
        h_events = contact[contact.events.isin(['single', 'double', 'triple', 'home_run'])]
        woba = 0
        if len(contact) > 0:
            woba = (len(h_events[h_events.events == 'single']) * 0.88 +
                    len(h_events[h_events.events == 'double']) * 1.25 +
                    len(h_events[h_events.events == 'triple']) * 1.58 +
                    len(h_events[h_events.events == 'home_run']) * 2.02) / len(contact)

        table_data.append({
            "pt": pt, "name": PITCH_NAMES.get(pt, pt), "n": n,
            "pct": f"{n / len(df) * 100:.1f}%",
            "velo": f"{velo.mean():.1f}" if len(velo) else "-",
            "csw": f"{csw:.1f}", "swm": f"{swm:.1f}",
            "maxv": f"{velo.max():.1f}" if len(velo) else "-",
            "spin": f"{spin.mean():.0f}" if len(spin) else "-",
            "ivb": f"{pfx_z_in.mean():.1f}" if len(pfx_z_in) else "-",
            "ext": f"{ext.mean():.1f}" if len(ext) else "-",
            "zone": f"{zone_pct:.1f}", "chase": f"{chase_pct:.1f}",
            "heart": f"{heart:.1f}", "shadow": f"{shadow:.1f}",
            "woba": f"{woba:.3f}",
            "color": PITCH_COLORS.get(pt, "#999"),
        })

    # Draw table manually with colored rows
    headers = ["Pitch Name", "Count", "Pitch%", "Velo", "CSW", "SwStr%",
               "MaxV", "Spin", "IVB", "Ext", "Zone%", "Chase%", "Heart%", "Shadow%", "wOBA"]
    col_widths = [0.10, 0.05, 0.06, 0.05, 0.05, 0.06, 0.05, 0.06, 0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.06]
    x_start = 0.05
    y_start = 0.50
    row_h = 0.030
    hdr_h = 0.025

    # Header row
    cx = x_start
    for hdr, w in zip(headers, col_widths):
        fig.text(cx + w / 2, y_start, hdr, ha='center', va='center',
                fontsize=5.5, fontweight='bold', fontfamily='sans-serif', color='#555555')
        cx += w

    # Header separator
    hdr_line = fig.add_axes([x_start, y_start - hdr_h / 2, sum(col_widths), 0.001])
    hdr_line.axhline(0, color='#aaaaaa', lw=0.8)
    hdr_line.axis('off')

    # Data rows
    for j, row in enumerate(table_data):
        y = y_start - hdr_h - j * row_h
        color = row["color"]
        bg_rgba = to_rgba(color, alpha=0.15)

        # Background stripe
        bg = patches.FancyBboxPatch((x_start, y - row_h / 2), sum(col_widths), row_h,
                                     boxstyle="round,pad=0.002", fc=bg_rgba, ec='none',
                                     transform=fig.transFigure, clip_on=False)
        fig.add_artist(bg)

        # Pitch name cell with stronger color
        name_bg = patches.FancyBboxPatch((x_start, y - row_h / 2), col_widths[0], row_h,
                                          boxstyle="round,pad=0.002", fc=to_rgba(color, 0.35),
                                          ec='none', transform=fig.transFigure, clip_on=False)
        fig.add_artist(name_bg)

        values = [row["name"], str(row["n"]), row["pct"], row["velo"], row["csw"], row["swm"],
                  row["maxv"], row["spin"], row["ivb"], row["ext"],
                  row["zone"], row["chase"], row["heart"], row["shadow"], row["woba"]]

        cx = x_start
        for i, (val, w) in enumerate(zip(values, col_widths)):
            font_color = '#222222'
            fw = 'bold' if i == 0 else 'normal'
            if i == 0:
                font_color = 'white'
            # Color code wOBA
            if i == len(values) - 1:
                try:
                    wv = float(val)
                    woba_bg = patches.FancyBboxPatch((cx, y - row_h / 2), w, row_h,
                                                      boxstyle="round,pad=0.002",
                                                      fc=to_rgba(woba_color(wv), 0.4),
                                                      ec='none', transform=fig.transFigure, clip_on=False)
                    fig.add_artist(woba_bg)
                except ValueError:
                    pass

            fig.text(cx + w / 2, y, val, ha='center', va='center',
                    fontsize=5.5, fontfamily='sans-serif', fontweight=fw, color=font_color)
            cx += w

    # Totals row
    total_y = y_start - hdr_h - len(table_data) * row_h - 0.005
    tot_line = fig.add_axes([x_start, total_y + row_h / 2, sum(col_widths), 0.001])
    tot_line.axhline(0, color='#aaaaaa', lw=0.8)
    tot_line.axis('off')

    total_vals = ["All", str(len(df)), "100.0%"] + ["-"] * 12
    cx = x_start
    for val, w in zip(total_vals, col_widths):
        fig.text(cx + w / 2, total_y, val, ha='center', va='center',
                fontsize=5.5, fontfamily='sans-serif', fontweight='bold', color='#333333')
        cx += w

    # Footer
    fig.text(0.95, 0.005, "Data: MLB", ha='right', fontsize=6, color='#bbbbbb',
            fontfamily='sans-serif', style='italic')

    # Border
    border = patches.FancyBboxPatch((0.01, 0.003), 0.98, 0.993,
                                     boxstyle="round,pad=0.008", lw=1.2,
                                     edgecolor='#cccccc', facecolor='none',
                                     transform=fig.transFigure, clip_on=False)
    fig.add_artist(border)

    # Save
    if output_path is None:
        safe = player_name.replace(" ", "_").replace(",", "")
        output_path = f"output/{safe}_{date}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Pitcher Performance Card")
    parser.add_argument("--pitcher", type=str, help="Pitcher name")
    parser.add_argument("--pitcher-id", type=int, help="MLB pitcher ID")
    parser.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--output", type=str, help="Output path")
    args = parser.parse_args()

    if args.pitcher_id:
        pid = args.pitcher_id
    elif args.pitcher:
        pid, found_name = find_pitcher_id(args.pitcher, args.date)
        print(f"Found: {found_name} (ID: {pid})")
    else:
        parser.error("Need --pitcher or --pitcher-id")

    generate_card(pid, args.date, args.output)


if __name__ == "__main__":
    main()
