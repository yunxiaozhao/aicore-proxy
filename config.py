"""
Configuration, authentication, and database for aicore-proxy.

Loads settings from env vars and /etc/aicore-proxy/config.json (hot-reloaded).
Manages API key validation and SQLite usage logging.
"""

import os
import json
import time
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
# Keys from three sources (merged): API_KEYS env var, config file, SQLite DB.
# No keys configured → auth disabled (backward compatible).
# Config file api_keys hot-reloaded every 60s.
# ---------------------------------------------------------------------------
_api_keys_cache = set()
_api_keys_lock = threading.Lock()
_api_keys_last_refresh = 0


def _refresh_api_keys():
    """Refresh the merged set of API keys from all sources."""
    global _api_keys_cache, _api_keys_last_refresh
    now = time.time()
    with _api_keys_lock:
        if now - _api_keys_last_refresh < 60:
            return
        _api_keys_last_refresh = now

    keys = set()
    # Source 1: env var
    env_keys = os.environ.get("API_KEYS", "")
    for k in env_keys.split(","):
        k = k.strip()
        if k:
            keys.add(k)

    # Source 2: config file (hot-reloaded)
    cfg = _load_config_file()
    for k in cfg.get("api_keys", []):
        if isinstance(k, str) and k.strip():
            keys.add(k.strip())

    # Source 3: SQLite DB (if stats enabled)
    if ENABLE_STATS and _db:
        try:
            cur = _db.execute("SELECT key FROM api_keys WHERE enabled = 1")
            for row in cur.fetchall():
                keys.add(row[0])
        except Exception:
            pass

    with _api_keys_lock:
        _api_keys_cache = keys


def auth_enabled():
    """Check if any API keys are configured."""
    _refresh_api_keys()
    with _api_keys_lock:
        return len(_api_keys_cache) > 0


def validate_api_key(key):
    """Validate an API key against the merged key set."""
    if not key:
        return False
    _refresh_api_keys()
    with _api_keys_lock:
        return key in _api_keys_cache


def reset_api_keys_cache():
    """Force refresh of API keys cache on next check."""
    global _api_keys_last_refresh
    with _api_keys_lock:
        _api_keys_last_refresh = 0


# ---------------------------------------------------------------------------
# SQLite Usage Tracking (opt-in via ENABLE_STATS=true)
# ---------------------------------------------------------------------------
_db = None
_db_lock = threading.Lock()

if ENABLE_STATS:
    _db_path = os.environ.get("DB_PATH", "/etc/aicore-proxy/proxy.db")
    os.makedirs(os.path.dirname(_db_path), exist_ok=True)
    _db = sqlite3.connect(_db_path, check_same_thread=False)
    _db.execute("PRAGMA journal_mode=WAL")
    _db.execute("""CREATE TABLE IF NOT EXISTS api_keys (
        key TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        enabled BOOLEAN DEFAULT 1
    )""")
    _db.execute("""CREATE TABLE IF NOT EXISTS usage_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        api_key TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        deployment_id TEXT,
        input_tokens INTEGER DEFAULT 0,
        output_tokens INTEGER DEFAULT 0,
        status_code INTEGER,
        stream BOOLEAN,
        duration_ms INTEGER
    )""")
    _db.execute("CREATE INDEX IF NOT EXISTS idx_usage_key_ts ON usage_log(api_key, timestamp)")
    _db.commit()
    print(f"[proxy] Stats enabled, SQLite DB: {_db_path}", flush=True)


def log_usage(api_key, deployment_id, input_tokens, output_tokens, status_code, stream, duration_ms):
    """Log a request to the usage_log table (only when ENABLE_STATS=true)."""
    if not ENABLE_STATS or not _db:
        return
    try:
        with _db_lock:
            _db.execute(
                "INSERT INTO usage_log (api_key, deployment_id, input_tokens, output_tokens, status_code, stream, duration_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (api_key or "anonymous", deployment_id, input_tokens, output_tokens, status_code, stream, duration_ms),
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
