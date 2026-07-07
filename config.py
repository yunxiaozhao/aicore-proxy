"""
Configuration, authentication, and database for aicore-proxy.

Loads settings from env vars and /etc/aicore-proxy/config.json (hot-reloaded).
Manages API key validation and SQLite usage logging.
"""

import os
import json
import time
import hashlib
import sqlite3
import secrets
import threading


# ---------------------------------------------------------------------------
# Config file loading — /etc/aicore-proxy/config.json (optional, volume-mounted)
# Priority: env var > config file. Config file api_keys hot-reloaded every 60s.
# ---------------------------------------------------------------------------
_CONFIG_PATH = os.environ.get("CONFIG_PATH", "/etc/aicore-proxy/config.json")
_config_file_data = {}
_config_file_mtime = 0
_config_file_lock = threading.Lock()
_config_last_check = 0


def _load_config_file():
    """Load config from JSON file if it exists. Returns cached data if file unchanged."""
    global _config_file_data, _config_file_mtime, _config_last_check
    now = time.time()
    with _config_file_lock:
        if now - _config_last_check < 60:
            return _config_file_data
        _config_last_check = now
    try:
        mtime = os.path.getmtime(_CONFIG_PATH)
        with _config_file_lock:
            if mtime == _config_file_mtime:
                return _config_file_data
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        with _config_file_lock:
            _config_file_data = data
            _config_file_mtime = mtime
        print(f"[proxy] Config file loaded/reloaded: {_CONFIG_PATH}", flush=True)
        return data
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[proxy] Config file error: {e}", flush=True)
        with _config_file_lock:
            return _config_file_data


def _cfg(env_key, config_key=None, default=None):
    """Get config value: env var takes priority, then config file, then default."""
    val = os.environ.get(env_key)
    if val is not None:
        return val
    if config_key is None:
        config_key = env_key.lower()
    cfg = _load_config_file()
    return cfg.get(config_key, default)


# ---------------------------------------------------------------------------
# Configuration — env vars take priority, then config file, then defaults
# ---------------------------------------------------------------------------
# Load config file once at startup for non-hot-reloaded fields
_startup_config = _load_config_file()

CLIENT_ID = _cfg("SAP_CLIENT_ID", "sap_client_id")
CLIENT_SECRET = _cfg("SAP_CLIENT_SECRET", "sap_client_secret")
AUTH_URL = _cfg("SAP_AUTH_URL", "sap_auth_url")
AI_API_URL = _cfg("SAP_AI_API_URL", "sap_ai_api_url")


def _parse_dep_list(val):
    """Normalize a deployment-id setting into a list of trimmed strings.

    Accepts a list of strings, a comma-separated string, or any scalar coerced
    to string. Returns [] for None / empty inputs.
    """
    if val is None:
        return []
    if isinstance(val, list):
        return [d.strip() for d in val if isinstance(d, str) and d.strip()]
    return [d.strip() for d in str(val).split(",") if d.strip()]


# ---------------------------------------------------------------------------
# Deployment IDs — two modes:
#
# 1. Flat (legacy):       SAP_DEPLOYMENT_ID="id1,id2"  /  "sap_deployment_id": ["id1","id2"]
#    All requests are load-balanced across the same pool, ignoring the client's
#    "model" field.
#
# 2. Model-aware (opt-in): per-model env vars OR a dict in the config file:
#       SAP_DEPLOYMENT_ID_OPUS="id1,id2"
#       SAP_DEPLOYMENT_ID_SONNET="id3"
#       SAP_DEPLOYMENT_ID_HAIKU="id4"
#       "sap_deployment_id": {"opus": ["id1","id2"], "sonnet": "id3", "haiku": "id4"}
#    The proxy inspects the request's "model" string for "opus"/"sonnet"/"haiku"
#    and routes to the matching pool. If no keyword matches (or that pool is
#    empty) it falls back to the union of all configured deployments.
#
# DEPLOYMENT_IDS is always the union (used for active-count bookkeeping and the
# fallback pool). DEPLOYMENT_IDS_BY_MODEL is non-empty only in mode 2.
# ---------------------------------------------------------------------------
MODEL_KEYWORDS = ("opus", "sonnet", "haiku")

DEPLOYMENT_IDS_BY_MODEL = {}

# Per-model env vars (highest priority) — if any is set, model-aware mode kicks in.
for _kw in MODEL_KEYWORDS:
    _env = os.environ.get(f"SAP_DEPLOYMENT_ID_{_kw.upper()}")
    if _env:
        DEPLOYMENT_IDS_BY_MODEL[_kw] = _parse_dep_list(_env)

_dep_raw = _cfg("SAP_DEPLOYMENT_ID", "sap_deployment_id", "")
if isinstance(_dep_raw, dict):
    # Config file specifies per-model dict; env vars (above) override matching keys.
    for _k, _v in _dep_raw.items():
        _kk = _k.lower()
        if _kk in MODEL_KEYWORDS and _kk not in DEPLOYMENT_IDS_BY_MODEL:
            DEPLOYMENT_IDS_BY_MODEL[_kk] = _parse_dep_list(_v)
    DEPLOYMENT_IDS = sorted({d for lst in DEPLOYMENT_IDS_BY_MODEL.values() for d in lst})
elif DEPLOYMENT_IDS_BY_MODEL:
    # Only per-model env vars supplied; no legacy list to merge.
    DEPLOYMENT_IDS = sorted({d for lst in DEPLOYMENT_IDS_BY_MODEL.values() for d in lst})
else:
    DEPLOYMENT_IDS = _parse_dep_list(_dep_raw)

RESOURCE_GROUP = _cfg("SAP_RESOURCE_GROUP", "sap_resource_group", "default")
VERBOSE = str(_cfg("VERBOSE", "verbose", "false")).lower() in ("true", "1", "yes")
ENABLE_STATS = str(_cfg("ENABLE_STATS", "enable_stats", "false")).lower() in ("true", "1", "yes")
# Admin API token — required to hit /admin/*. If unset, admin routes return 403.
ADMIN_TOKEN = _cfg("ADMIN_TOKEN", "admin_token", "") or ""

if not CLIENT_ID or not CLIENT_SECRET or not AUTH_URL or not AI_API_URL:
    raise ValueError("SAP credentials required: set SAP_CLIENT_ID, SAP_CLIENT_SECRET, SAP_AUTH_URL, SAP_AI_API_URL via env vars or config file")
if not DEPLOYMENT_IDS:
    raise ValueError("SAP_DEPLOYMENT_ID must contain at least one deployment ID")

if DEPLOYMENT_IDS_BY_MODEL:
    print(f"[proxy] Configured {len(DEPLOYMENT_IDS)} deployment(s) in model-aware mode: "
          f"{ {k: v for k, v in DEPLOYMENT_IDS_BY_MODEL.items()} }", flush=True)
else:
    print(f"[proxy] Configured {len(DEPLOYMENT_IDS)} deployment(s): {DEPLOYMENT_IDS}", flush=True)


# ---------------------------------------------------------------------------
# API Key Authentication
#
# Keys are never stored in plaintext. Env/config supply plaintext keys which
# are hashed at load time; the DB stores only sha256(key), never the raw key.
# The plaintext of a DB-managed key is shown exactly once, when it's created.
#
# Keys from three sources (merged): API_KEYS env var, config file, SQLite DB.
# No keys configured → auth disabled (backward compatible).
# Config file api_keys hot-reloaded every 60s.
# ---------------------------------------------------------------------------
_api_key_hashes = set()          # sha256 hex digests of all active keys
_api_keys_lock = threading.Lock()
_api_keys_last_refresh = 0


def hash_key(key):
    """Return the sha256 hex digest of an API key. Empty/None → ''."""
    if not key:
        return ""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def key_prefix(key):
    """Return a masked prefix suitable for display: first 12 chars + '…'."""
    if not key:
        return ""
    return (key[:12] + "…") if len(key) > 12 else key


def _refresh_api_keys():
    """Refresh the merged set of API-key hashes from all sources."""
    global _api_key_hashes, _api_keys_last_refresh
    now = time.time()
    with _api_keys_lock:
        if now - _api_keys_last_refresh < 60:
            return
        _api_keys_last_refresh = now

    hashes = set()
    # Source 1: env var (plaintext → hash)
    env_keys = os.environ.get("API_KEYS", "")
    for k in env_keys.split(","):
        k = k.strip()
        if k:
            hashes.add(hash_key(k))

    # Source 2: config file (plaintext → hash, hot-reloaded)
    cfg = _load_config_file()
    for k in cfg.get("api_keys", []):
        if isinstance(k, str) and k.strip():
            hashes.add(hash_key(k.strip()))

    # Source 3: SQLite DB (already hashed)
    if ENABLE_STATS and _db:
        try:
            cur = _db.execute("SELECT key_hash FROM api_keys WHERE enabled = 1")
            for row in cur.fetchall():
                if row[0]:
                    hashes.add(row[0])
        except Exception:
            pass

    with _api_keys_lock:
        _api_key_hashes = hashes


def auth_enabled():
    """Check if any API keys are configured."""
    _refresh_api_keys()
    with _api_keys_lock:
        return len(_api_key_hashes) > 0


def validate_api_key(key):
    """Validate an API key by hashing it and looking up the digest."""
    if not key:
        return False
    _refresh_api_keys()
    h = hash_key(key)
    with _api_keys_lock:
        return h in _api_key_hashes


def reset_api_keys_cache():
    """Force refresh of API keys cache on next check."""
    global _api_keys_last_refresh
    with _api_keys_lock:
        _api_keys_last_refresh = 0


def admin_token_configured():
    """True iff ADMIN_TOKEN is set to a non-empty value."""
    return bool(ADMIN_TOKEN)


def validate_admin_token(token):
    """Constant-time comparison against ADMIN_TOKEN. False if unconfigured."""
    if not ADMIN_TOKEN or not token:
        return False
    return secrets.compare_digest(str(token), str(ADMIN_TOKEN))


# ---------------------------------------------------------------------------
# SQLite Usage Tracking (opt-in via ENABLE_STATS=true)
#
# Schema (v2 — no plaintext keys):
#   api_keys(key_hash PK, key_prefix, name, created_at, enabled)
#   usage_log(id PK, key_hash, key_prefix, timestamp, deployment_id, …)
#
# If a v1 schema is detected (api_keys.key column exists), it's migrated in
# place: each row is rehashed into key_hash+key_prefix and the old key column
# is dropped so no plaintext survives on disk.
# ---------------------------------------------------------------------------
_db = None
_db_lock = threading.Lock()


def _column_names(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _migrate_api_keys_v1_to_v2(conn):
    """If api_keys still has a plaintext `key` column, rehash and drop it."""
    cols = _column_names(conn, "api_keys")
    if "key_hash" in cols and "key" not in cols:
        return  # already migrated

    if "key" in cols and "key_hash" not in cols:
        print("[proxy] Migrating api_keys: hashing plaintext keys → key_hash", flush=True)
        conn.execute("ALTER TABLE api_keys ADD COLUMN key_hash TEXT")
        conn.execute("ALTER TABLE api_keys ADD COLUMN key_prefix TEXT")
        rows = conn.execute("SELECT rowid, key FROM api_keys").fetchall()
        for rowid, raw in rows:
            if not raw:
                continue
            conn.execute(
                "UPDATE api_keys SET key_hash = ?, key_prefix = ? WHERE rowid = ?",
                (hash_key(raw), key_prefix(raw), rowid),
            )
        # Rebuild the table without the plaintext `key` column and with key_hash as PK.
        conn.execute("""CREATE TABLE api_keys_new (
            key_hash TEXT PRIMARY KEY,
            key_prefix TEXT,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            enabled BOOLEAN DEFAULT 1
        )""")
        conn.execute(
            "INSERT OR IGNORE INTO api_keys_new (key_hash, key_prefix, name, created_at, enabled) "
            "SELECT key_hash, key_prefix, name, created_at, enabled FROM api_keys "
            "WHERE key_hash IS NOT NULL AND key_hash != ''"
        )
        conn.execute("DROP TABLE api_keys")
        conn.execute("ALTER TABLE api_keys_new RENAME TO api_keys")
        conn.commit()
        # Reclaim the pages that used to hold plaintext keys.
        try:
            conn.execute("VACUUM")
        except sqlite3.OperationalError:
            pass
        print("[proxy] Migration complete: plaintext keys removed from DB", flush=True)


def _migrate_usage_log_v1_to_v2(conn):
    """Replace plaintext api_key with key_hash + key_prefix (best-effort backfill)."""
    cols = _column_names(conn, "usage_log")
    if "key_hash" in cols and "api_key" not in cols:
        return

    if "api_key" in cols and "key_hash" not in cols:
        print("[proxy] Migrating usage_log: hashing api_key → key_hash + key_prefix", flush=True)
        conn.execute("ALTER TABLE usage_log ADD COLUMN key_hash TEXT")
        conn.execute("ALTER TABLE usage_log ADD COLUMN key_prefix TEXT")
        # Backfill by hashing whatever the api_key column held. Rows whose api_key
        # was "anonymous" get hash_key("anonymous")=<some-hash>, which is fine — it
        # groups anonymous usage together without leaking a real key.
        rows = conn.execute(
            "SELECT rowid, api_key FROM usage_log WHERE key_hash IS NULL"
        ).fetchall()
        for rowid, raw in rows:
            conn.execute(
                "UPDATE usage_log SET key_hash = ?, key_prefix = ? WHERE rowid = ?",
                (hash_key(raw or ""), key_prefix(raw or ""), rowid),
            )
        # Rebuild without the plaintext api_key column.
        conn.execute("""CREATE TABLE usage_log_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_hash TEXT NOT NULL,
            key_prefix TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            deployment_id TEXT,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            status_code INTEGER,
            stream BOOLEAN,
            duration_ms INTEGER
        )""")
        conn.execute(
            "INSERT INTO usage_log_new (id, key_hash, key_prefix, timestamp, deployment_id, "
            "input_tokens, output_tokens, status_code, stream, duration_ms) "
            "SELECT id, key_hash, key_prefix, timestamp, deployment_id, "
            "input_tokens, output_tokens, status_code, stream, duration_ms FROM usage_log"
        )
        conn.execute("DROP TABLE usage_log")
        conn.execute("ALTER TABLE usage_log_new RENAME TO usage_log")
        conn.commit()
        try:
            conn.execute("VACUUM")
        except sqlite3.OperationalError:
            pass
        print("[proxy] Migration complete: usage_log rehashed", flush=True)


if ENABLE_STATS:
    _db_path = os.environ.get("DB_PATH", "/etc/aicore-proxy/proxy.db")
    os.makedirs(os.path.dirname(_db_path), exist_ok=True)
    _db = sqlite3.connect(_db_path, check_same_thread=False)
    _db.execute("PRAGMA journal_mode=WAL")
    # Fresh installs get the v2 schema directly. Existing v1 DBs get migrated below.
    _db.execute("""CREATE TABLE IF NOT EXISTS api_keys (
        key_hash TEXT PRIMARY KEY,
        key_prefix TEXT,
        name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        enabled BOOLEAN DEFAULT 1
    )""")
    _db.execute("""CREATE TABLE IF NOT EXISTS usage_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key_hash TEXT NOT NULL,
        key_prefix TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        deployment_id TEXT,
        input_tokens INTEGER DEFAULT 0,
        output_tokens INTEGER DEFAULT 0,
        status_code INTEGER,
        stream BOOLEAN,
        duration_ms INTEGER
    )""")
    _migrate_api_keys_v1_to_v2(_db)
    _migrate_usage_log_v1_to_v2(_db)
    _db.execute("CREATE INDEX IF NOT EXISTS idx_usage_key_ts ON usage_log(key_hash, timestamp)")
    _db.commit()
    print(f"[proxy] Stats enabled, SQLite DB: {_db_path}", flush=True)
    if not ADMIN_TOKEN:
        print("[proxy] WARNING: ENABLE_STATS is on but ADMIN_TOKEN is unset — /admin/* endpoints will refuse all requests.", flush=True)


def log_usage(api_key, deployment_id, input_tokens, output_tokens, status_code, stream, duration_ms):
    """Log a request to usage_log. Stores only the hash+prefix of the caller's key."""
    if not ENABLE_STATS or not _db:
        return
    kh = hash_key(api_key) if api_key else hash_key("anonymous")
    kp = key_prefix(api_key) if api_key else "anonymous"
    try:
        with _db_lock:
            _db.execute(
                "INSERT INTO usage_log (key_hash, key_prefix, deployment_id, input_tokens, output_tokens, status_code, stream, duration_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (kh, kp, deployment_id, input_tokens, output_tokens, status_code, stream, duration_ms),
            )
            _db.commit()
    except Exception as e:
        print(f"[proxy] Usage log error: {e}", flush=True)


def db_execute(query, params=()):
    """Execute a DB query and return all rows. Thread-safe."""
    with _db_lock:
        return _db.execute(query, params).fetchall()


def db_execute_write(query, params=()):
    """Execute a DB write query and commit. Returns the cursor. Thread-safe."""
    with _db_lock:
        cur = _db.execute(query, params)
        _db.commit()
        return cur
