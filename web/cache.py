"""SQLite cache for MLB API responses and computed BRAVS results."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

# DB lives alongside this module in web/bravs_cache.db
_DB_PATH = Path(__file__).resolve().parent / "bravs_cache.db"

# Default TTLs (seconds)
_API_TTL = 86400       # 24 hours for generic API responses
_CURRENT_SEASON_TTL = 3600  # 1 hour for current-season BRAVS results

# Module-level lock for connection management
_lock = threading.Lock()

# Per-thread connection cache (sqlite3 connections aren't safe across threads)
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection (created on first access)."""
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        _local.conn = conn
    return conn


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS api_cache (
    url       TEXT PRIMARY KEY,
    response  TEXT NOT NULL,
    expires   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS bravs_cache (
    player_id   INTEGER NOT NULL,
    season      INTEGER NOT NULL,
    result      TEXT NOT NULL,
    computed_at INTEGER NOT NULL,
    PRIMARY KEY (player_id, season)
);
"""


def init_cache() -> None:
    """Create cache tables if they do not already exist."""
    conn = _get_conn()
    with _lock:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()


# ---------------------------------------------------------------------------
# API cache helpers
# ---------------------------------------------------------------------------

def get_api(url: str) -> dict[str, Any] | None:
    """Return cached API response for *url* if it has not expired, else None."""
    conn = _get_conn()
    now = int(time.time())
    row = conn.execute(
        "SELECT response FROM api_cache WHERE url = ? AND expires > ?",
        (url, now),
    ).fetchone()
    if row is None:
        return None
    return json.loads(row["response"])


def set_api(url: str, response: dict[str, Any], ttl_seconds: int = _API_TTL) -> None:
    """Cache an API *response* for *url* with the given TTL (default 24 h)."""
    conn = _get_conn()
    expires = int(time.time()) + ttl_seconds
    with _lock:
        conn.execute(
            "INSERT OR REPLACE INTO api_cache (url, response, expires) VALUES (?, ?, ?)",
            (url, json.dumps(response, default=str), expires),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# BRAVS result cache helpers
# ---------------------------------------------------------------------------

def _current_year() -> int:
    """Return the current calendar year."""
    return time.localtime().tm_year


def get_bravs(player_id: int, season: int) -> dict[str, Any] | None:
    """Return cached BRAVS result for a player-season, respecting expiry rules.

    Historical seasons (before the current year) never expire.
    Current-season results expire after 1 hour.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT result, computed_at FROM bravs_cache WHERE player_id = ? AND season = ?",
        (player_id, season),
    ).fetchone()
    if row is None:
        return None

    # Historical seasons never expire
    if season < _current_year():
        return json.loads(row["result"])

    # Current season: honour 1-hour TTL
    age = int(time.time()) - row["computed_at"]
    if age > _CURRENT_SEASON_TTL:
        return None

    return json.loads(row["result"])


def set_bravs(player_id: int, season: int, result: dict[str, Any]) -> None:
    """Cache a BRAVS result for a player-season.

    Historical seasons are stored with no effective expiry.
    Current-season results are timestamped so ``get_bravs`` can enforce
    the 1-hour TTL.
    """
    conn = _get_conn()
    now = int(time.time())
    with _lock:
        conn.execute(
            "INSERT OR REPLACE INTO bravs_cache (player_id, season, result, computed_at) "
            "VALUES (?, ?, ?, ?)",
            (player_id, season, json.dumps(result, default=str), now),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

def clear_expired() -> None:
    """Remove expired entries from both cache tables."""
    conn = _get_conn()
    now = int(time.time())
    current_year = _current_year()
    cutoff = now - _CURRENT_SEASON_TTL

    with _lock:
        # Expired API rows
        conn.execute("DELETE FROM api_cache WHERE expires <= ?", (now,))

        # Current-season BRAVS rows older than the TTL
        conn.execute(
            "DELETE FROM bravs_cache WHERE season = ? AND computed_at <= ?",
            (current_year, cutoff),
        )

        conn.commit()
