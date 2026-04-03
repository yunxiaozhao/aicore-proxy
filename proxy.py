"""
Core proxy logic: deployment selection, OAuth2 token management,
SAP AI Core request forwarding, SSE stream processing, and body adaptation.

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

import json
import time
import threading
import requests as req_lib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    CLIENT_ID, CLIENT_SECRET, AUTH_URL, AI_API_URL,
    DEPLOYMENT_IDS, RESOURCE_GROUP, VERBOSE, ENABLE_STATS,
    log_usage,
)


# ---------------------------------------------------------------------------
# Least-connections deployment selector (thread-safe)
# ---------------------------------------------------------------------------
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


def get_deployment_active():
    """Return a snapshot of per-deployment active request counts."""
    with _deployment_lock:
        return dict(_deployment_active)


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
    """Request an OAuth2 access_token from SAP XSUAA (client_credentials grant)."""
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
    """Background daemon thread loop: periodically refreshes the token."""
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


def get_token():
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


def get_token_status():
    """Return (has_token, last_error) for health check."""
    with _token_lock:
        return _token is not None, _last_token_error


# ---------------------------------------------------------------------------
# SAP AI Core request forwarding
# ---------------------------------------------------------------------------

def forward_to_sap(headers, body, stream):
    """Forward the request to SAP AI Core; auto-retry once on 401 after refreshing the token.

    Returns (requests.Response, deployment_id) tuple.
    Raises req_lib.Timeout or req_lib.RequestException on failure.
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


def release_deployment(dep_id):
    """Public wrapper for releasing a deployment slot."""
    _release_deployment(dep_id)


# ---------------------------------------------------------------------------
# SSE stream processing
# ---------------------------------------------------------------------------

def inject_sse_events(sap_resp, deployment_id, client_key="", req_start=None):
    """Convert SAP/Bedrock SSE stream to standard Anthropic SSE format.

    SAP returns:   data: {"type":"message_start",...}\\n\\n
    Anthropic:     event: message_start\\ndata: {"type":"message_start",...}\\n\\n
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
        log_usage(client_key, deployment_id, input_tokens, output_tokens, 200, True, duration_ms)


# ---------------------------------------------------------------------------
# Request body adaptation
# ---------------------------------------------------------------------------

def _strip_cache_control(obj):
    """Recursively strip cache_control fields from dicts/lists.

    SAP AI Core / Bedrock does not support Anthropic's prompt caching extensions.
    """
    if isinstance(obj, dict):
        obj.pop("cache_control", None)
        for v in obj.values():
            _strip_cache_control(v)
    elif isinstance(obj, list):
        for item in obj:
            _strip_cache_control(item)


def adapt_body(body):
    """Adapt Anthropic request body to SAP AI Core compatible format.

    Returns (body, is_stream). Modifies body dict in place.
    """
    if "anthropic_version" not in body:
        body["anthropic_version"] = "bedrock-2023-05-31"

    body.pop("model", None)
    is_stream = body.pop("stream", False)
    body.pop("context_management", None)

    for key in ("system", "messages", "tools"):
        if key in body:
            _strip_cache_control(body[key])

    if "tools" in body:
        body["tools"] = [t for t in body["tools"]
                         if t.get("type", "custom") == "custom"]
        if not body["tools"]:
            del body["tools"]
            body.pop("tool_choice", None)

    return body, is_stream
