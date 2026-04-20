"""Logic for leaderboards, YoY deltas, stat filter, season counter."""

from __future__ import annotations

import pandas as pd

from . import data


def leaderboard(kind: str, stat_key: str, year_from: int, year_to: int,
                n: int = 20, asc: bool = False, min_pa: int = 300, min_ip: int = 40,
                pos: str | None = None) -> list[dict]:
    st = data.stat_by_key(kind, stat_key)
    if st is None:
        return []
    df = data.master(kind)
    df = df[(df["yearID"] >= year_from) & (df["yearID"] <= year_to)]
    if kind == "batter":
        df = df[df["PA"] >= min_pa]
        if pos:
            df = df[df["position"] == pos]
    else:
        df = df[df["IP"] >= min_ip]
    if stat_key not in df.columns:
        return []
    df = df.dropna(subset=[stat_key])
    df = df.sort_values(stat_key, ascending=asc or (st.direction == "low"))
    df = df.head(n)

    out = []
    for i, (_, r) in enumerate(df.iterrows()):
        mid = int(r["mlbam_id"]) if pd.notna(r.get("mlbam_id")) else None
        out.append({
            "rank": i + 1,
            "name": r.get("full_name", r.get("playerID")),
            "playerID": r["playerID"],
            "mlbam_id": mid,
            "portrait": data.portrait_url(mid),
            "year": int(r["yearID"]),
            "team": r.get("teamID") or "—",
            "value": data.fmt_value(st, r[stat_key]),
            "raw": None if pd.isna(r[stat_key]) else float(r[stat_key]),
        })
    return out


def yoy_delta(kind: str, stat_key: str, year_a: int, year_b: int,
              n: int = 20, min_pa: int = 300, min_ip: int = 40,
              improvers: bool = True) -> list[dict]:
    st = data.stat_by_key(kind, stat_key)
    if st is None:
        return []
    df = data.master(kind)
    a = df[df["yearID"] == year_a][["playerID", "full_name", "teamID", stat_key, "PA" if kind == "batter" else "IP", "mlbam_id"]]
    b = df[df["yearID"] == year_b][["playerID", stat_key, "teamID"]]
    a = a.rename(columns={stat_key: "a_val", "teamID": "team_a"})
    b = b.rename(columns={stat_key: "b_val", "teamID": "team_b"})
    m = a.merge(b, on="playerID", how="inner")
    vol_col = "PA" if kind == "batter" else "IP"
    vol_min = min_pa if kind == "batter" else min_ip
    m = m[m[vol_col] >= vol_min]
    m["delta"] = m["b_val"] - m["a_val"]
    m = m.dropna(subset=["delta"])

    # "Improvers" means better direction — higher for high-stats, lower for low-stats
    if st.direction == "low":
        m = m.sort_values("delta", ascending=improvers)
    else:
        m = m.sort_values("delta", ascending=not improvers)
    m = m.head(n)

    out = []
    for i, (_, r) in enumerate(m.iterrows()):
        mid = int(r["mlbam_id"]) if pd.notna(r.get("mlbam_id")) else None
        out.append({
            "rank": i + 1,
            "name": r.get("full_name", r.get("playerID")),
            "playerID": r["playerID"],
            "mlbam_id": mid,
            "portrait": data.portrait_url(mid),
            "team_a": r.get("team_a") or "—",
            "team_b": r.get("team_b") or "—",
            "a": data.fmt_value(st, r["a_val"]),
            "b": data.fmt_value(st, r["b_val"]),
            "delta": data.fmt_value(st, r["delta"]),
            "delta_sign": "up" if r["delta"] > 0 else ("down" if r["delta"] < 0 else "flat"),
        })
    return out


def stat_filter(kind: str, year_from: int, year_to: int,
                filters: list[dict], n: int = 30,
                sort_key: str | None = None,
                min_pa: int = 300, min_ip: int = 40) -> list[dict]:
    """filters = [{'stat': 'AVG', 'op': '>=', 'value': 0.300}, ...]"""
    df = data.master(kind)
    df = df[(df["yearID"] >= year_from) & (df["yearID"] <= year_to)]
    if kind == "batter":
        df = df[df["PA"] >= min_pa]
    else:
        df = df[df["IP"] >= min_ip]

    for f in filters:
        k, op, v = f["stat"], f["op"], f["value"]
        if k not in df.columns or v in ("", None):
            continue
        try:
            v = float(v)
        except Exception:
            continue
        col = df[k]
        if op == ">=": df = df[col >= v]
        elif op == ">": df = df[col > v]
        elif op == "<=": df = df[col <= v]
        elif op == "<": df = df[col < v]
        elif op == "=": df = df[col == v]

    if sort_key and sort_key in df.columns:
        st = data.stat_by_key(kind, sort_key)
        df = df.sort_values(sort_key, ascending=(st.direction == "low"))

    df = df.head(n)
    stats_to_show = [f["stat"] for f in filters if f["stat"] in df.columns]
    if sort_key and sort_key not in stats_to_show:
        stats_to_show.insert(0, sort_key)

    out = []
    for i, (_, r) in enumerate(df.iterrows()):
        mid = int(r["mlbam_id"]) if pd.notna(r.get("mlbam_id")) else None
        row = {
            "rank": i + 1,
            "name": r.get("full_name", r.get("playerID")),
            "playerID": r["playerID"],
            "mlbam_id": mid,
            "portrait": data.portrait_url(mid),
            "year": int(r["yearID"]),
            "team": r.get("teamID") or "—",
            "stats": {},
        }
        for k in stats_to_show:
            st = data.stat_by_key(kind, k)
            row["stats"][k] = data.fmt_value(st, r[k]) if st else str(r[k])
        out.append(row)
    return out


def season_counter(kind: str, filters: list[dict], year_from: int, year_to: int,
                   n: int = 25) -> list[dict]:
    """How many seasons did each player meet all filter conditions?"""
    df = data.master(kind)
    df = df[(df["yearID"] >= year_from) & (df["yearID"] <= year_to)]

    for f in filters:
        k, op, v = f["stat"], f["op"], f["value"]
        if k not in df.columns or v in ("", None):
            continue
        try:
            v = float(v)
        except Exception:
            continue
        col = df[k]
        if op == ">=": df = df[col >= v]
        elif op == ">": df = df[col > v]
        elif op == "<=": df = df[col <= v]
        elif op == "<": df = df[col < v]
        elif op == "=": df = df[col == v]

    counts = df.groupby(["playerID", "full_name"], dropna=False).agg(
        seasons=("yearID", "nunique"),
        first=("yearID", "min"),
        last=("yearID", "max"),
    ).reset_index()
    counts = counts.sort_values("seasons", ascending=False).head(n)

    out = []
    for i, (_, r) in enumerate(counts.iterrows()):
        out.append({
            "rank": i + 1,
            "name": r["full_name"] or r["playerID"],
            "playerID": r["playerID"],
            "seasons": int(r["seasons"]),
            "span": f"{int(r['first'])}–{int(r['last'])}",
        })
    return out


def team_roster(team_id: str, year: int, kind: str = "batter") -> list[dict]:
    df = data.master(kind)
    sub = df[(df["yearID"] == year) & (df["teamID"] == team_id)]
    if kind == "batter":
        sub = sub[sub["PA"] >= 50].sort_values("PA", ascending=False)
    else:
        sub = sub[sub["IP"] >= 10].sort_values("IP", ascending=False)

    out = []
    for _, r in sub.iterrows():
        mid = int(r["mlbam_id"]) if pd.notna(r.get("mlbam_id")) else None
        if kind == "batter":
            out.append({
                "name": r.get("full_name") or r["playerID"],
                "mlbam_id": mid,
                "portrait": data.portrait_url(mid),
                "pos": r.get("position") or "—",
                "stats": [
                    ("PA", f"{int(r['PA'])}"),
                    ("HR", f"{int(r['HR'])}"),
                    ("AVG", f"{r['AVG']:.3f}" if pd.notna(r.get("AVG")) else "—"),
                    ("OPS", f"{r['OPS']:.3f}" if pd.notna(r.get("OPS")) else "—"),
                    ("BRAVS", f"{r['bravs']:.2f}" if pd.notna(r.get("bravs")) else "—"),
                ],
            })
        else:
            out.append({
                "name": r.get("full_name") or r["playerID"],
                "mlbam_id": mid,
                "portrait": data.portrait_url(mid),
                "pos": "SP" if r.get("GS", 0) >= 10 else "RP",
                "stats": [
                    ("IP", f"{r['IP']:.1f}"),
                    ("ERA", f"{r['ERA']:.2f}" if pd.notna(r.get("ERA")) else "—"),
                    ("K%", f"{r['K_pct']:.1f}" if pd.notna(r.get("K_pct")) else "—"),
                    ("BRAVS", f"{r['bravs']:.2f}" if pd.notna(r.get("bravs")) else "—"),
                ],
            })
    return out


def team_list(year: int) -> list[dict]:
    df = data.master("batter")
    sub = df[df["yearID"] == year][["teamID", "lgID"]].dropna(subset=["teamID"])
    counts = sub["teamID"].value_counts()
    out = []
    for t, _n in counts.items():
        lg = sub[sub["teamID"] == t]["lgID"].iloc[0]
        out.append({"teamID": t, "lgID": lg})
    return sorted(out, key=lambda x: x["teamID"])
