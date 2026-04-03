"""
Lightweight reverse proxy: standard Anthropic Messages API -> SAP AI Core.

Accepts:  POST /v1/messages  (Anthropic Messages API format)
Forwards: SAP AI Core deployment endpoint with OAuth2 bearer token.
          - Non-streaming: /v2/inference/deployments/{id}/invoke
          - Streaming:     /v2/inference/deployments/{id}/invoke-with-response-stream

Architecture:
  Client (Claude Code / SDK)
    |
    |  POST /v1/messages (Anthropic format)
    v
  [This Proxy]  --- OAuth2 client_credentials ---> SAP XSUAA
    |
    |  POST /v2/inference/deployments/{id}/invoke[-with-response-stream]
    v
  SAP AI Core (Claude model deployment)
"""

import os
import json
import time
import sqlite3
import secrets
import threading
import requests as req_lib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from flask import Flask, request, Response, jsonify


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
_dep_str = _cfg("SAP_DEPLOYMENT_ID", "sap_deployment_id", "")
DEPLOYMENT_IDS = [d.strip() for d in _dep_str.split(",") if d.strip()]
RESOURCE_GROUP = _cfg("SAP_RESOURCE_GROUP", "sap_resource_group", "default")
VERBOSE = str(_cfg("VERBOSE", "verbose", "false")).lower() in ("true", "1", "yes")
ENABLE_STATS = str(_cfg("ENABLE_STATS", "enable_stats", "false")).lower() in ("true", "1", "yes")

if not CLIENT_ID or not CLIENT_SECRET or not AUTH_URL or not AI_API_URL:
    raise ValueError("SAP credentials required: set SAP_CLIENT_ID, SAP_CLIENT_SECRET, SAP_AUTH_URL, SAP_AI_API_URL via env vars or config file")
if not DEPLOYMENT_IDS:
    raise ValueError("SAP_DEPLOYMENT_ID must contain at least one deployment ID")

print(f"[proxy] Configured {len(DEPLOYMENT_IDS)} deployment(s): {DEPLOYMENT_IDS}", flush=True)

# Least-connections deployment selector (thread-safe)
_deployment_active = {dep_id: 0 for dep_id in DEPLOYMENT_IDS}
_deployment_lock = threading.Lock()


def _next_deployment():
    """Pick the deployment with the fewest active requests (least-connections).

    Increments the active count atomically before returning.
    Caller MUST call _release_deployment() when the request completes.
    """
    with _deployment_lock:
        dep = min(DEPLOYMENT_IDS, key=lambda d: _deployment_active[d])
        _deployment_active[dep] += 1
        return dep


def _release_deployment(dep_id):
    """Decrement active request count when a request completes."""
    with _deployment_lock:
        _deployment_active[dep_id] = max(0, _deployment_active[dep_id] - 1)


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


def _auth_enabled():
    """Check if any API keys are configured."""
    _refresh_api_keys()
    with _api_keys_lock:
        return len(_api_keys_cache) > 0


def _validate_api_key(key):
    """Validate an API key against the merged key set."""
    if not key:
        return False
    _refresh_api_keys()
    with _api_keys_lock:
        return key in _api_keys_cache


# ---------------------------------------------------------------------------
# SQLite Usage Tracking (opt-in via ENABLE_STATS=true)
# ---------------------------------------------------------------------------
_db = None
_db_lock = threading.Lock()

if ENABLE_STATS:
    _db_path = os.environ.get("DB_PATH", "/app/data/proxy.db")
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


def _log_usage(api_key, deployment_id, input_tokens, output_tokens, status_code, stream, duration_ms):
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


# ---------------------------------------------------------------------------
# OAuth2 Token Management
#
# - A background daemon thread auto-refreshes the token 5 minutes before expiry
# - If the thread hasn't obtained a token yet, the request thread fetches one synchronously
# - On refresh failure, retries with exponential backoff (30s -> 60s -> ... -> 300s cap)
# ---------------------------------------------------------------------------
_token = None
_token_expires = 0
_token_lock = threading.Lock()
_refresh_started = False
_refresh_lock = threading.Lock()
_last_token_error = None

# Retry-enabled session for token requests to handle transient SSL/timeout errors
_auth_session = req_lib.Session()
_auth_retry = Retry(
    total=3,
    backoff_factor=2,           # waits 0s, 2s, 4s between retries
    status_forcelist=[502, 503, 504],
    allowed_methods=["POST"],
)
_auth_session.mount("https://", HTTPAdapter(max_retries=_auth_retry))
_auth_session.mount("http://", HTTPAdapter(max_retries=_auth_retry))

# Connection-pooled session for upstream API requests (avoids per-request TCP setup)
_api_session = req_lib.Session()
_api_adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20)
_api_session.mount("https://", _api_adapter)
_api_session.mount("http://", _api_adapter)


def _fetch_token():
    """Request an OAuth2 access_token from SAP XSUAA (client_credentials grant).

    On success, stores the token and expiry time in global variables.
    Expiry is set 300s early to ensure refresh completes before actual expiration.
    Returns the new token string.
    Uses a retry-enabled session to handle transient SSL/timeout errors.
    """
    global _token, _token_expires, _last_token_error
    resp = _auth_session.post(
        f"{AUTH_URL}/oauth/token?grant_type=client_credentials",
        auth=(CLIENT_ID, CLIENT_SECRET),
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    expires_in = data.get("expires_in", 43200)
    new_token = data["access_token"]
    with _token_lock:
        _token = new_token
        _token_expires = time.time() + expires_in - 300
        _last_token_error = None
    print(f"[proxy] Token refreshed, expires in {expires_in}s", flush=True)
    return new_token


def _refresh_loop():
    """Background daemon thread loop: periodically refreshes the token.

    Normal flow: obtain token -> sleep until near expiry -> refresh again.
    Error flow: exponential backoff retry (30s / 60s / 120s / 300s cap).
    """
    global _last_token_error
    retry_delay = 30
    while True:
        try:
            _fetch_token()
            retry_delay = 30
        except Exception as e:
            print(f"[proxy] Token refresh error: {e}", flush=True)
            with _token_lock:
                _last_token_error = str(e)
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 300)
            continue
        with _token_lock:
            sleep_time = max(60, _token_expires - time.time())
        time.sleep(sleep_time)


def _ensure_refresh_thread():
    """Ensure the token refresh daemon thread is started (only once)."""
    global _refresh_started
    with _refresh_lock:
        if not _refresh_started:
            _refresh_started = True
            t = threading.Thread(target=_refresh_loop, daemon=True)
            t.start()


def _get_token():
    """Get the current valid access_token.

    1. Ensure the background refresh thread is started
    2. If token is None or expired, fetch one synchronously (blocking)
    3. Return the token (may be None; caller must handle this)
    """
    _ensure_refresh_thread()
    with _token_lock:
        token, expires = _token, _token_expires
    if token is None or time.time() > expires:
        try:
            return _fetch_token()
        except Exception as e:
            with _token_lock:
                global _last_token_error
                _last_token_error = str(e)
            print(f"[proxy] Synchronous token fetch error: {e}", flush=True)
            return None
    return token


def _forward_to_sap(headers, body, stream):
    """Forward the request to SAP AI Core; auto-retry once on 401 after refreshing the token.

    Args:
        headers: Request headers (including Authorization)
        body:    Request body dict
        stream:  Whether to use streaming

    Returns:
        (requests.Response, deployment_id) tuple

    Raises:
        req_lib.Timeout: Upstream timeout
        req_lib.RequestException: Other network errors
    """
    deployment_id = _next_deployment()
    subpath = "invoke-with-response-stream" if stream else "invoke"
    target_url = f"{AI_API_URL}/v2/inference/deployments/{deployment_id}/{subpath}"
    if VERBOSE:
        print(f"[proxy] -> deployment: {deployment_id}", flush=True)
    try:
        sap_resp = _api_session.post(target_url, headers=headers, json=body, stream=stream, timeout=300)
        if sap_resp.status_code == 401:
            new_token = _fetch_token()
            headers["Authorization"] = f"Bearer {new_token}"
            sap_resp = _api_session.post(target_url, headers=headers, json=body, stream=stream, timeout=300)
    except Exception:
        _release_deployment(deployment_id)
        raise
    return sap_resp, deployment_id


def _inject_sse_events(sap_resp, deployment_id, client_key="", req_start=None):
    """Convert SAP/Bedrock SSE stream to standard Anthropic SSE format.

    SAP returns:   data: {"type":"message_start",...}\\n\\n
    Anthropic:     event: message_start\\ndata: {"type":"message_start",...}\\n\\n

    Reads upstream SSE line by line, injecting an event: line before each data: line.
    Accumulates token usage from message_start and message_delta events.
    Ensures the upstream response is closed and deployment released when done.
    """
    input_tokens = 0
    output_tokens = 0
    try:
        for raw_line in sap_resp.iter_lines():
            line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
            if line.startswith("data: "):
                try:
                    payload = json.loads(line[6:])
                    event_type = payload.get("type", "")
                except (json.JSONDecodeError, ValueError):
                    payload = {}
                    event_type = ""
                # Accumulate token usage from SSE events
                if event_type == "message_start":
                    usage = payload.get("message", {}).get("usage", {})
                    input_tokens += usage.get("input_tokens", 0)
                elif event_type == "message_delta":
                    usage = payload.get("usage", {})
                    output_tokens += usage.get("output_tokens", 0)
                if event_type:
                    yield f"event: {event_type}\n{line}\n\n".encode("utf-8")
                else:
                    yield f"{line}\n\n".encode("utf-8")
            elif line.strip():
                yield f"{line}\n".encode("utf-8")
    except req_lib.exceptions.ChunkedEncodingError:
        print("[proxy] Upstream stream ended prematurely (ChunkedEncodingError)", flush=True)
    except req_lib.exceptions.ConnectionError as e:
        print(f"[proxy] Upstream connection lost during streaming: {e}", flush=True)
    finally:
        sap_resp.close()
        _release_deployment(deployment_id)
        duration_ms = int((time.time() - req_start) * 1000) if req_start else 0
        _log_usage(client_key, deployment_id, input_tokens, output_tokens, 200, True, duration_ms)


def _strip_cache_control(obj):
    """Recursively strip cache_control fields from dicts/lists.

    SAP AI Core / Bedrock does not support Anthropic's prompt caching extensions.
    VS Code Claude extension sends cache_control with extra fields (e.g. scope)
    that cause 400 errors.
    """
    if isinstance(obj, dict):
        obj.pop("cache_control", None)
        for v in obj.values():
            _strip_cache_control(v)
    elif isinstance(obj, list):
        for item in obj:
            _strip_cache_control(item)


def _adapt_body(body):
    """Adapt Anthropic request body to SAP AI Core compatible format.

    Returns (body, is_stream). Modifies body dict in place.
    """
    if "anthropic_version" not in body:
        body["anthropic_version"] = "bedrock-2023-05-31"

    # SAP AI Core does not accept these fields
    body.pop("model", None)
    is_stream = body.pop("stream", False)
    body.pop("context_management", None)

    # Strip cache_control from system/messages/tools — unsupported by SAP AI Core
    for key in ("system", "messages", "tools"):
        if key in body:
            _strip_cache_control(body[key])

    # Bedrock does not support Anthropic built-in tools (web_search_20250305, text_editor_20250124, etc.);
    # keep only standard custom tools (type is "custom" or absent)
    if "tools" in body:
        body["tools"] = [t for t in body["tools"]
                         if t.get("type", "custom") == "custom"]
        if not body["tools"]:
            del body["tools"]
            body.pop("tool_choice", None)

    return body, is_stream


# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/v1/messages", methods=["POST"])
def messages():
    """Main proxy endpoint: receive Anthropic Messages API requests, forward to SAP AI Core."""
    # --- API key authentication ---
    client_key = request.headers.get("x-api-key") or ""
    if not client_key:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            client_key = auth_header[7:].strip()
    if _auth_enabled() and not _validate_api_key(client_key):
        return jsonify({"type": "error", "error": {"type": "authentication_error", "message": "Invalid API key"}}), 401

    token = _get_token()
    if not token:
        return jsonify({"error": "No SAP token available yet, try again shortly"}), 503

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Invalid JSON body"}), 400

    body, is_stream = _adapt_body(body)
    req_start = time.time()

    headers = {
        "Authorization": f"Bearer {token}",
        "ai-resource-group": RESOURCE_GROUP,
        "Content-Type": "application/json",
    }

    if VERBOSE:
        msg_count = len(body.get("messages", []))
        tool_count = len(body.get("tools", []))
        print(f"[proxy] >>> stream={is_stream}, messages={msg_count}, tools={tool_count}, "
              f"body keys: {list(body.keys())}", flush=True)

    try:
        sap_resp, dep_id = _forward_to_sap(headers, body, stream=is_stream)
    except req_lib.Timeout:
        _log_usage(client_key, None, 0, 0, 504, is_stream, int((time.time() - req_start) * 1000))
        return jsonify({"error": "Upstream SAP AI Core timeout"}), 504
    except req_lib.RequestException as e:
        _log_usage(client_key, None, 0, 0, 502, is_stream, int((time.time() - req_start) * 1000))
        return jsonify({"error": f"Upstream SAP AI Core request failed: {e}"}), 502

    if VERBOSE:
        print(f"[proxy] <<< status: {sap_resp.status_code}", flush=True)

    # Always log non-200 errors regardless of VERBOSE setting for easier debugging
    if sap_resp.status_code != 200:
        error_body = sap_resp.content.decode("utf-8", errors="replace")
        print(f"[proxy] <<< {sap_resp.status_code} error: {error_body[:2000]}", flush=True)
        _release_deployment(dep_id)
        _log_usage(client_key, dep_id, 0, 0, sap_resp.status_code, is_stream, int((time.time() - req_start) * 1000))
        return Response(sap_resp.content, status=sap_resp.status_code,
                        content_type=sap_resp.headers.get("Content-Type", "application/json"))

    # Success: streaming
    if is_stream:
        # Deployment released in _inject_sse_events finally block
        return Response(
            _inject_sse_events(sap_resp, dep_id, client_key, req_start),
            status=200, headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            })

    # Success: non-streaming — extract usage from response
    _release_deployment(dep_id)
    duration_ms = int((time.time() - req_start) * 1000)
    input_tokens = output_tokens = 0
    try:
        resp_data = json.loads(sap_resp.content)
        usage = resp_data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
    except (json.JSONDecodeError, AttributeError):
        pass
    _log_usage(client_key, dep_id, input_tokens, output_tokens, 200, False, duration_ms)
    return Response(sap_resp.content, status=200,
                    content_type=sap_resp.headers.get("Content-Type", "application/json"))


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint. Returns token status for Docker healthcheck and monitoring."""
    with _token_lock:
        has_token = _token is not None
        token_error = _last_token_error
    result = {
        "status": "ok",
        "has_token": has_token,
        "token_error": token_error,
        "deployments": len(DEPLOYMENT_IDS),
        "auth_enabled": _auth_enabled(),
        "stats_enabled": ENABLE_STATS,
    }
    if ENABLE_STATS and _db:
        try:
            with _db_lock:
                row = _db.execute(
                    "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0) FROM usage_log"
                ).fetchone()
            result["total_requests"] = row[0]
            result["total_input_tokens"] = row[1]
            result["total_output_tokens"] = row[2]
        except Exception:
            pass
    return jsonify(result)


@app.route("/stats", methods=["GET"])
def stats():
    """Lightweight stats: per-deployment active connections and global counters."""
    with _deployment_lock:
        active = dict(_deployment_active)
    result = {"deployments_active": active}
    if ENABLE_STATS and _db:
        try:
            with _db_lock:
                row = _db.execute(
                    "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0) FROM usage_log"
                ).fetchone()
            result["total_requests"] = row[0]
            result["total_input_tokens"] = row[1]
            result["total_output_tokens"] = row[2]
        except Exception:
            pass
    return jsonify(result)


# ---------------------------------------------------------------------------
# Admin API (requires ENABLE_STATS=true)
# ---------------------------------------------------------------------------

@app.route("/admin/keys", methods=["GET"])
def admin_list_keys():
    """List all API keys in the database."""
    if not ENABLE_STATS or not _db:
        return jsonify({"error": "Stats not enabled"}), 404
    with _db_lock:
        rows = _db.execute("SELECT key, name, created_at, enabled FROM api_keys ORDER BY created_at").fetchall()
    return jsonify([{"key": r[0], "name": r[1], "created_at": r[2], "enabled": bool(r[3])} for r in rows])


@app.route("/admin/keys", methods=["POST"])
def admin_create_key():
    """Create a new API key. Body: {"name": "...", "key": "..."} (key is optional, auto-generated if missing)."""
    if not ENABLE_STATS or not _db:
        return jsonify({"error": "Stats not enabled"}), 404
    data = request.get_json(silent=True) or {}
    name = data.get("name", "unnamed")
    key = data.get("key") or f"sk-{secrets.token_urlsafe(32)}"
    try:
        with _db_lock:
            _db.execute("INSERT INTO api_keys (key, name) VALUES (?, ?)", (key, name))
            _db.commit()
        # Force refresh key cache
        global _api_keys_last_refresh
        with _api_keys_lock:
            _api_keys_last_refresh = 0
        return jsonify({"key": key, "name": name}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Key already exists"}), 409


@app.route("/admin/keys/<key>", methods=["DELETE"])
def admin_delete_key(key):
    """Delete (disable) an API key."""
    if not ENABLE_STATS or not _db:
        return jsonify({"error": "Stats not enabled"}), 404
    with _db_lock:
        cur = _db.execute("DELETE FROM api_keys WHERE key = ?", (key,))
        _db.commit()
    if cur.rowcount == 0:
        return jsonify({"error": "Key not found"}), 404
    global _api_keys_last_refresh
    with _api_keys_lock:
        _api_keys_last_refresh = 0
    return jsonify({"deleted": key})


@app.route("/admin/usage", methods=["GET"])
def admin_usage():
    """Usage summary. Query params: ?key=xxx, ?days=7."""
    if not ENABLE_STATS or not _db:
        return jsonify({"error": "Stats not enabled"}), 404
    key_filter = request.args.get("key")
    days = request.args.get("days", type=int)
    query = "SELECT api_key, COUNT(*) as requests, COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), COALESCE(AVG(duration_ms),0) FROM usage_log"
    params = []
    conditions = []
    if key_filter:
        conditions.append("api_key = ?")
        params.append(key_filter)
    if days:
        conditions.append("timestamp >= datetime('now', ?)")
        params.append(f"-{days} days")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " GROUP BY api_key ORDER BY requests DESC"
    with _db_lock:
        rows = _db.execute(query, params).fetchall()
    return jsonify([{
        "api_key": r[0], "requests": r[1],
        "input_tokens": r[2], "output_tokens": r[3],
        "avg_duration_ms": round(r[4]),
    } for r in rows])


if __name__ == "__main__":
    _get_token()
    app.run(host="0.0.0.0", port=6655)
