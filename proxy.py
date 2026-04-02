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
import itertools
import threading
import requests as req_lib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from flask import Flask, request, Response, jsonify

# ---------------------------------------------------------------------------
# Configuration — from docker-compose.yml environment
# ---------------------------------------------------------------------------
CLIENT_ID = os.environ["SAP_CLIENT_ID"]
CLIENT_SECRET = os.environ["SAP_CLIENT_SECRET"]
AUTH_URL = os.environ["SAP_AUTH_URL"]         # XSUAA token endpoint base URL
AI_API_URL = os.environ["SAP_AI_API_URL"]     # SAP AI Core API base URL
DEPLOYMENT_IDS = [d.strip() for d in os.environ["SAP_DEPLOYMENT_ID"].split(",") if d.strip()]
if not DEPLOYMENT_IDS:
    raise ValueError("SAP_DEPLOYMENT_ID must contain at least one deployment ID")
RESOURCE_GROUP = os.environ.get("SAP_RESOURCE_GROUP", "default")
VERBOSE = os.environ.get("VERBOSE", "false").lower() in ("true", "1", "yes")

print(f"[proxy] Configured {len(DEPLOYMENT_IDS)} deployment(s): {DEPLOYMENT_IDS}", flush=True)

# Round-robin deployment selector (thread-safe)
_deployment_cycle = itertools.cycle(DEPLOYMENT_IDS)
_deployment_lock = threading.Lock()


def _next_deployment():
    """Return the next deployment ID in round-robin order (thread-safe)."""
    with _deployment_lock:
        return next(_deployment_cycle)


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
        requests.Response object

    Raises:
        req_lib.Timeout: Upstream timeout
        req_lib.RequestException: Other network errors
    """
    deployment_id = _next_deployment()
    subpath = "invoke-with-response-stream" if stream else "invoke"
    target_url = f"{AI_API_URL}/v2/inference/deployments/{deployment_id}/{subpath}"
    if VERBOSE:
        print(f"[proxy] -> deployment: {deployment_id}", flush=True)
    sap_resp = _api_session.post(target_url, headers=headers, json=body, stream=stream, timeout=300)
    if sap_resp.status_code == 401:
        new_token = _fetch_token()
        headers["Authorization"] = f"Bearer {new_token}"
        sap_resp = _api_session.post(target_url, headers=headers, json=body, stream=stream, timeout=300)
    return sap_resp


def _inject_sse_events(sap_resp):
    """Convert SAP/Bedrock SSE stream to standard Anthropic SSE format.

    SAP returns:   data: {"type":"message_start",...}\\n\\n
    Anthropic:     event: message_start\\ndata: {"type":"message_start",...}\\n\\n

    Reads upstream SSE line by line, injecting an event: line before each data: line.
    Ensures the upstream response is closed when the generator exits (client disconnect, etc.).
    """
    try:
        for raw_line in sap_resp.iter_lines():
            line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
            if line.startswith("data: "):
                try:
                    payload = json.loads(line[6:])
                    event_type = payload.get("type", "")
                except (json.JSONDecodeError, ValueError):
                    event_type = ""
                if event_type:
                    yield f"event: {event_type}\n{line}\n\n".encode("utf-8")
                else:
                    yield f"{line}\n\n".encode("utf-8")
            elif line.strip():
                yield f"{line}\n".encode("utf-8")
    except req_lib.exceptions.ChunkedEncodingError as e:
        print(f"[proxy] Upstream stream ended prematurely (ChunkedEncodingError): {e}", flush=True)
        err_payload = json.dumps({
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Upstream stream ended prematurely"},
        })
        yield f"event: error\ndata: {err_payload}\n\n".encode("utf-8")
    except req_lib.exceptions.ConnectionError as e:
        print(f"[proxy] Upstream connection lost during streaming: {e}", flush=True)
        err_payload = json.dumps({
            "type": "error",
            "error": {"type": "overloaded_error", "message": f"Upstream connection lost: {e}"},
        })
        yield f"event: error\ndata: {err_payload}\n\n".encode("utf-8")
    finally:
        sap_resp.close()


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
    """Main proxy endpoint: receive Anthropic Messages API requests, forward to SAP AI Core.

    Flow:
    1. Obtain OAuth2 token
    2. Validate and adapt request body (remove unsupported fields, filter built-in tools)
    3. Forward to SAP AI Core (streaming via /invoke-with-response-stream, otherwise /invoke)
    4. For streaming: inject SSE event lines and pass through; for non-streaming: return full JSON
    """
    token = _get_token()
    if not token:
        return jsonify({"error": "No SAP token available yet, try again shortly"}), 503

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Invalid JSON body"}), 400

    body, is_stream = _adapt_body(body)

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
        sap_resp = _forward_to_sap(headers, body, stream=is_stream)
    except req_lib.Timeout:
        return jsonify({"error": "Upstream SAP AI Core timeout"}), 504
    except req_lib.RequestException as e:
        return jsonify({"error": f"Upstream SAP AI Core request failed: {e}"}), 502

    if VERBOSE:
        print(f"[proxy] <<< status: {sap_resp.status_code}", flush=True)

    # Always log non-200 errors regardless of VERBOSE setting for easier debugging
    if sap_resp.status_code != 200:
        error_body = sap_resp.content.decode("utf-8", errors="replace")
        print(f"[proxy] <<< {sap_resp.status_code} error: {error_body[:2000]}", flush=True)
        return Response(sap_resp.content, status=sap_resp.status_code,
                        content_type=sap_resp.headers.get("Content-Type", "application/json"))

    # Success response
    if is_stream:
        return Response(_inject_sse_events(sap_resp), status=200, headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })

    # Non-streaming: read full response body; retry once on truncated chunked transfer
    for attempt in range(2):
        try:
            content = sap_resp.content
            break
        except req_lib.exceptions.ChunkedEncodingError as e:
            print(f"[proxy] Non-streaming response truncated (attempt {attempt + 1}): {e}", flush=True)
            sap_resp.close()
            if attempt == 0:
                try:
                    sap_resp = _forward_to_sap(headers, body, stream=False)
                except req_lib.Timeout:
                    return jsonify({"error": "Upstream SAP AI Core timeout on retry"}), 504
                except req_lib.RequestException as e2:
                    return jsonify({"error": f"Upstream retry failed: {e2}"}), 502
                if sap_resp.status_code != 200:
                    err_body = sap_resp.content.decode("utf-8", errors="replace")
                    print(f"[proxy] <<< {sap_resp.status_code} error on retry: {err_body[:2000]}", flush=True)
                    return Response(sap_resp.content, status=sap_resp.status_code,
                                    content_type=sap_resp.headers.get("Content-Type", "application/json"))
            else:
                return jsonify({"error": "Upstream response truncated after retry"}), 502
    return Response(content, status=200,
                    content_type=sap_resp.headers.get("Content-Type", "application/json"))


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint. Returns token status for Docker healthcheck and monitoring.

    Read-only: does NOT trigger token refresh to avoid side-effects from periodic healthchecks.
    """
    with _token_lock:
        has_token = _token is not None
        token_error = _last_token_error
    return jsonify({
        "status": "ok",
        "has_token": has_token,
        "token_error": token_error,
        "deployments": len(DEPLOYMENT_IDS),
    })


if __name__ == "__main__":
    _get_token()
    app.run(host="0.0.0.0", port=6655)
