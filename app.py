"""
Flask application — all HTTP route handlers for aicore-proxy.

Routes:
  POST /v1/messages  — main proxy endpoint (Anthropic Messages API)
  GET  /health       — health check
  GET  /stats        — deployment active connections + global counters
  GET  /admin/keys   — list API keys
  POST /admin/keys   — create API key
  DELETE /admin/keys/<key> — delete API key
  GET  /admin/usage  — usage summary
"""

import json
import time
import sqlite3
import requests as req_lib
from flask import Flask, request, Response, jsonify

from functools import wraps

from config import (
    DEPLOYMENT_IDS, RESOURCE_GROUP, VERBOSE, ENABLE_STATS,
    auth_enabled, validate_api_key, reset_api_keys_cache,
    admin_token_configured, validate_admin_token,
    hash_key, key_prefix,
    log_usage, db_execute, db_execute_write,
    _db,
)
from proxy import (
    get_token, get_token_status, get_deployment_active,
    forward_to_sap, release_deployment, inject_sse_events, adapt_body,
)


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
    if auth_enabled() and not validate_api_key(client_key):
        return jsonify({"type": "error", "error": {"type": "authentication_error", "message": "Invalid API key"}}), 401

    token = get_token()
    if not token:
        return jsonify({"error": "No SAP token available yet, try again shortly"}), 503

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Invalid JSON body"}), 400

    # Snapshot the raw incoming body BEFORE adapt_body mutates it, so debug
    # logs show exactly what the client sent (useful when isolating a 400).
    if VERBOSE:
        try:
            raw_body_dump = json.dumps(body, ensure_ascii=False)
        except Exception as _e:
            raw_body_dump = f"<unserializable: {_e}>"
        raw_client_headers = {k: v for k, v in request.headers.items()
                              if k.lower() not in ("authorization", "x-api-key", "cookie")}
        print(f"[proxy] === incoming request ===\n"
              f"  client headers: {json.dumps(raw_client_headers, ensure_ascii=False)}\n"
              f"  raw body keys: {list(body.keys())}\n"
              f"  raw body: {raw_body_dump}", flush=True)

    body, is_stream, req_model = adapt_body(body)
    req_start = time.time()

    headers = {
        "Authorization": f"Bearer {token}",
        "ai-resource-group": RESOURCE_GROUP,
        "Content-Type": "application/json",
    }

    if VERBOSE:
        msg_count = len(body.get("messages", []))
        tool_count = len(body.get("tools", []))
        try:
            adapted_body_dump = json.dumps(body, ensure_ascii=False)
        except Exception as _e:
            adapted_body_dump = f"<unserializable: {_e}>"
        # Field-by-field dump so each top-level key is grep-able even when
        # the full body is huge.
        per_field = {k: json.dumps(v, ensure_ascii=False) for k, v in body.items()}
        per_field_dump = "\n".join(f"    [{k}] = {v}" for k, v in per_field.items())
        outbound_headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
        print(f"[proxy] >>> model={req_model!r}, stream={is_stream}, messages={msg_count}, tools={tool_count}, "
              f"body keys: {list(body.keys())}\n"
              f"  outbound headers (Authorization redacted): {json.dumps(outbound_headers, ensure_ascii=False)}\n"
              f"  adapted body: {adapted_body_dump}\n"
              f"  adapted body per field:\n{per_field_dump}", flush=True)

    try:
        sap_resp, dep_id = forward_to_sap(headers, body, stream=is_stream, model_hint=req_model)
    except req_lib.Timeout:
        log_usage(client_key, None, 0, 0, 504, is_stream, int((time.time() - req_start) * 1000))
        return jsonify({"error": "Upstream SAP AI Core timeout"}), 504
    except req_lib.RequestException as e:
        log_usage(client_key, None, 0, 0, 502, is_stream, int((time.time() - req_start) * 1000))
        return jsonify({"error": f"Upstream SAP AI Core request failed: {e}"}), 502

    if VERBOSE:
        try:
            resp_headers_dump = json.dumps(dict(sap_resp.headers), ensure_ascii=False)
        except Exception as _e:
            resp_headers_dump = f"<unserializable: {_e}>"
        print(f"[proxy] <<< status: {sap_resp.status_code}\n"
              f"  response headers: {resp_headers_dump}", flush=True)

    if sap_resp.status_code != 200:
        error_body = sap_resp.content.decode("utf-8", errors="replace")
        try:
            resp_headers_dump = json.dumps(dict(sap_resp.headers), ensure_ascii=False)
        except Exception as _e:
            resp_headers_dump = f"<unserializable: {_e}>"
        # Full error body (not truncated) + response headers — SAP/Bedrock
        # sometimes puts the real error type in headers like x-amzn-errortype.
        print(f"[proxy] <<< {sap_resp.status_code} error on deployment {dep_id}\n"
              f"  response headers: {resp_headers_dump}\n"
              f"  full error body: {error_body}", flush=True)
        release_deployment(dep_id)
        log_usage(client_key, dep_id, 0, 0, sap_resp.status_code, is_stream, int((time.time() - req_start) * 1000))
        return Response(sap_resp.content, status=sap_resp.status_code,
                        content_type=sap_resp.headers.get("Content-Type", "application/json"))

    # Success: streaming
    if is_stream:
        return Response(
            inject_sse_events(sap_resp, dep_id, client_key, req_start),
            status=200, headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            })

    # Success: non-streaming — retry once on truncated chunked transfer
    for attempt in range(2):
        try:
            content = sap_resp.content
            break
        except req_lib.exceptions.ChunkedEncodingError as e:
            print(f"[proxy] Non-streaming response truncated (attempt {attempt + 1}): {e}", flush=True)
            sap_resp.close()
            if attempt == 0:
                try:
                    sap_resp, dep_id2 = forward_to_sap(headers, body, stream=False, model_hint=req_model)
                except req_lib.Timeout:
                    release_deployment(dep_id)
                    log_usage(client_key, dep_id, 0, 0, 504, False, int((time.time() - req_start) * 1000))
                    return jsonify({"error": "Upstream SAP AI Core timeout on retry"}), 504
                except req_lib.RequestException as e2:
                    release_deployment(dep_id)
                    log_usage(client_key, dep_id, 0, 0, 502, False, int((time.time() - req_start) * 1000))
                    return jsonify({"error": f"Upstream retry failed: {e2}"}), 502
                release_deployment(dep_id)
                dep_id = dep_id2
                if sap_resp.status_code != 200:
                    err_body = sap_resp.content.decode("utf-8", errors="replace")
                    print(f"[proxy] <<< {sap_resp.status_code} error on retry: {err_body[:2000]}", flush=True)
                    release_deployment(dep_id)
                    log_usage(client_key, dep_id, 0, 0, sap_resp.status_code, False, int((time.time() - req_start) * 1000))
                    return Response(sap_resp.content, status=sap_resp.status_code,
                                    content_type=sap_resp.headers.get("Content-Type", "application/json"))
            else:
                release_deployment(dep_id)
                log_usage(client_key, dep_id, 0, 0, 502, False, int((time.time() - req_start) * 1000))
                return jsonify({"error": "Upstream response truncated after retry"}), 502

    release_deployment(dep_id)
    duration_ms = int((time.time() - req_start) * 1000)
    input_tokens = output_tokens = 0
    try:
        resp_data = json.loads(content)
        usage = resp_data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
    except (json.JSONDecodeError, AttributeError):
        pass
    log_usage(client_key, dep_id, input_tokens, output_tokens, 200, False, duration_ms)
    return Response(content, status=200,
                    content_type=sap_resp.headers.get("Content-Type", "application/json"))


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint. Returns token status for Docker healthcheck and monitoring."""
    has_token, token_error = get_token_status()
    result = {
        "status": "ok",
        "has_token": has_token,
        "token_error": token_error,
        "deployments": len(DEPLOYMENT_IDS),
        "auth_enabled": auth_enabled(),
        "stats_enabled": ENABLE_STATS,
    }
    if ENABLE_STATS and _db:
        try:
            row = db_execute(
                "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0) FROM usage_log"
            )
            result["total_requests"] = row[0][0]
            result["total_input_tokens"] = row[0][1]
            result["total_output_tokens"] = row[0][2]
        except Exception:
            pass
    return jsonify(result)


@app.route("/stats", methods=["GET"])
def stats():
    """Lightweight stats: per-deployment active connections and global counters."""
    result = {"deployments_active": get_deployment_active()}
    if ENABLE_STATS and _db:
        try:
            row = db_execute(
                "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0) FROM usage_log"
            )
            result["total_requests"] = row[0][0]
            result["total_input_tokens"] = row[0][1]
            result["total_output_tokens"] = row[0][2]
        except Exception:
            pass
    return jsonify(result)


# ---------------------------------------------------------------------------
# Admin API (requires ENABLE_STATS=true AND a valid ADMIN_TOKEN)
#
# Auth: send the admin token in the `X-Admin-Token` header, or in
#       `Authorization: Bearer <token>`. If ADMIN_TOKEN isn't configured, every
#       /admin/* route returns 403 — the interface is off by default.
#
# Responses NEVER include a stored key in plaintext. The plaintext of a newly
# created key is returned exactly once, in the POST /admin/keys response, and
# never persisted anywhere the API can read back. Existing keys are identified
# by their key_prefix (first 12 chars + '…') for display, and by the full
# key_hash (sha256 hex) for filtering.
# ---------------------------------------------------------------------------

def require_admin(fn):
    """Gate an admin route on ENABLE_STATS + ADMIN_TOKEN header check."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not ENABLE_STATS or not _db:
            return jsonify({"error": "Stats not enabled"}), 404
        if not admin_token_configured():
            return jsonify({"error": "Admin API disabled (ADMIN_TOKEN not set)"}), 403
        token = request.headers.get("X-Admin-Token", "")
        if not token:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:].strip()
        if not validate_admin_token(token):
            return jsonify({"error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper


@app.route("/admin/keys", methods=["GET"])
@require_admin
def admin_list_keys():
    """List all API keys. Only the masked key_prefix is returned — never the raw key."""
    rows = db_execute(
        "SELECT key_hash, key_prefix, name, created_at, enabled FROM api_keys ORDER BY created_at"
    )
    return jsonify([{
        "key_hash": r[0],
        "key_prefix": r[1],
        "name": r[2],
        "created_at": r[3],
        "enabled": bool(r[4]),
    } for r in rows])


@app.route("/admin/keys", methods=["POST"])
@require_admin
def admin_create_key():
    """Create a new API key. Body: {"name": "..."}. The plaintext key is returned once here."""
    import secrets
    data = request.get_json(silent=True) or {}
    name = data.get("name", "unnamed")
    # Allow the caller to pass in a chosen key (e.g. for migration), but strongly prefer
    # server-generated ones — we can't guarantee externally-supplied keys have high entropy.
    key = data.get("key") or f"sk-{secrets.token_urlsafe(32)}"
    kh = hash_key(key)
    kp = key_prefix(key)
    try:
        db_execute_write(
            "INSERT INTO api_keys (key_hash, key_prefix, name) VALUES (?, ?, ?)",
            (kh, kp, name),
        )
        reset_api_keys_cache()
        # This is the ONLY time we ever return the plaintext key.
        return jsonify({
            "key": key,
            "key_hash": kh,
            "key_prefix": kp,
            "name": name,
            "warning": "Store this key now — it will not be shown again.",
        }), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Key already exists"}), 409


@app.route("/admin/keys/<identifier>", methods=["DELETE"])
@require_admin
def admin_delete_key(identifier):
    """Delete an API key. `identifier` is either the full key_hash or (fallback) a raw key.

    Passing the raw key still works so admins can revoke a leaked key without first
    looking up its hash, but we hash it locally rather than storing it.
    """
    # If it looks like a sha256 hex digest, try that first; otherwise treat as a raw key.
    candidates = []
    if len(identifier) == 64 and all(c in "0123456789abcdef" for c in identifier.lower()):
        candidates.append(identifier.lower())
    candidates.append(hash_key(identifier))
    for kh in candidates:
        cur = db_execute_write("DELETE FROM api_keys WHERE key_hash = ?", (kh,))
        if cur.rowcount:
            reset_api_keys_cache()
            return jsonify({"deleted": kh})
    return jsonify({"error": "Key not found"}), 404


@app.route("/admin/usage", methods=["GET"])
@require_admin
def admin_usage():
    """Usage summary. Query params: ?key_hash=xxx | ?key=xxx, ?days=7, ?group_by=day.

    Passing `key=<raw>` hashes it server-side and filters on the hash — the raw
    value is never echoed back in the response.
    """
    key_hash_filter = request.args.get("key_hash")
    raw_key_filter = request.args.get("key")
    if raw_key_filter and not key_hash_filter:
        key_hash_filter = hash_key(raw_key_filter)
    days = request.args.get("days", type=int)
    group_by = request.args.get("group_by", "")
    params = []
    conditions = []
    if key_hash_filter:
        conditions.append("key_hash = ?")
        params.append(key_hash_filter)
    if days:
        conditions.append("timestamp >= datetime('now', ?)")
        params.append(f"-{days} days")
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

    if group_by == "day":
        query = (
            f"SELECT DATE(timestamp) as date, key_hash, "
            f"COALESCE(MAX(key_prefix), '') as prefix, COUNT(*) as requests, "
            f"COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), "
            f"COALESCE(AVG(duration_ms),0) FROM usage_log{where} "
            f"GROUP BY date, key_hash ORDER BY date DESC, requests DESC"
        )
        rows = db_execute(query, params)
        return jsonify([{
            "date": r[0], "key_hash": r[1], "key_prefix": r[2],
            "requests": r[3], "input_tokens": r[4], "output_tokens": r[5],
            "avg_duration_ms": round(r[6]),
        } for r in rows])

    query = (
        f"SELECT key_hash, COALESCE(MAX(key_prefix), '') as prefix, "
        f"COUNT(*) as requests, COALESCE(SUM(input_tokens),0), "
        f"COALESCE(SUM(output_tokens),0), COALESCE(AVG(duration_ms),0) FROM usage_log{where} "
        f"GROUP BY key_hash ORDER BY requests DESC"
    )
    rows = db_execute(query, params)
    return jsonify([{
        "key_hash": r[0], "key_prefix": r[1], "requests": r[2],
        "input_tokens": r[3], "output_tokens": r[4],
        "avg_duration_ms": round(r[5]),
    } for r in rows])


if __name__ == "__main__":
    get_token()
    app.run(host="0.0.0.0", port=6655)
