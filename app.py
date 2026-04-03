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

from config import (
    DEPLOYMENT_IDS, RESOURCE_GROUP, VERBOSE, ENABLE_STATS,
    auth_enabled, validate_api_key, reset_api_keys_cache,
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

    body, is_stream = adapt_body(body)
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
        sap_resp, dep_id = forward_to_sap(headers, body, stream=is_stream)
    except req_lib.Timeout:
        log_usage(client_key, None, 0, 0, 504, is_stream, int((time.time() - req_start) * 1000))
        return jsonify({"error": "Upstream SAP AI Core timeout"}), 504
    except req_lib.RequestException as e:
        log_usage(client_key, None, 0, 0, 502, is_stream, int((time.time() - req_start) * 1000))
        return jsonify({"error": f"Upstream SAP AI Core request failed: {e}"}), 502

    if VERBOSE:
        print(f"[proxy] <<< status: {sap_resp.status_code}", flush=True)

    if sap_resp.status_code != 200:
        error_body = sap_resp.content.decode("utf-8", errors="replace")
        print(f"[proxy] <<< {sap_resp.status_code} error: {error_body[:2000]}", flush=True)
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

    # Success: non-streaming — extract usage from response
    release_deployment(dep_id)
    duration_ms = int((time.time() - req_start) * 1000)
    input_tokens = output_tokens = 0
    try:
        resp_data = json.loads(sap_resp.content)
        usage = resp_data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
    except (json.JSONDecodeError, AttributeError):
        pass
    log_usage(client_key, dep_id, input_tokens, output_tokens, 200, False, duration_ms)
    return Response(sap_resp.content, status=200,
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
# Admin API (requires ENABLE_STATS=true)
# ---------------------------------------------------------------------------

@app.route("/admin/keys", methods=["GET"])
def admin_list_keys():
    """List all API keys in the database."""
    if not ENABLE_STATS or not _db:
        return jsonify({"error": "Stats not enabled"}), 404
    rows = db_execute("SELECT key, name, created_at, enabled FROM api_keys ORDER BY created_at")
    return jsonify([{"key": r[0], "name": r[1], "created_at": r[2], "enabled": bool(r[3])} for r in rows])


@app.route("/admin/keys", methods=["POST"])
def admin_create_key():
    """Create a new API key. Body: {"name": "...", "key": "..."} (key is optional, auto-generated if missing)."""
    if not ENABLE_STATS or not _db:
        return jsonify({"error": "Stats not enabled"}), 404
    import secrets
    data = request.get_json(silent=True) or {}
    name = data.get("name", "unnamed")
    key = data.get("key") or f"sk-{secrets.token_urlsafe(32)}"
    try:
        db_execute_write("INSERT INTO api_keys (key, name) VALUES (?, ?)", (key, name))
        reset_api_keys_cache()
        return jsonify({"key": key, "name": name}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Key already exists"}), 409


@app.route("/admin/keys/<key>", methods=["DELETE"])
def admin_delete_key(key):
    """Delete (disable) an API key."""
    if not ENABLE_STATS or not _db:
        return jsonify({"error": "Stats not enabled"}), 404
    cur = db_execute_write("DELETE FROM api_keys WHERE key = ?", (key,))
    if cur.rowcount == 0:
        return jsonify({"error": "Key not found"}), 404
    reset_api_keys_cache()
    return jsonify({"deleted": key})


@app.route("/admin/usage", methods=["GET"])
def admin_usage():
    """Usage summary. Query params: ?key=xxx, ?days=7, ?group_by=day."""
    if not ENABLE_STATS or not _db:
        return jsonify({"error": "Stats not enabled"}), 404
    key_filter = request.args.get("key")
    days = request.args.get("days", type=int)
    group_by = request.args.get("group_by", "")
    params = []
    conditions = []
    if key_filter:
        conditions.append("api_key = ?")
        params.append(key_filter)
    if days:
        conditions.append("timestamp >= datetime('now', ?)")
        params.append(f"-{days} days")
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

    if group_by == "day":
        query = (
            f"SELECT DATE(timestamp) as date, api_key, COUNT(*) as requests, "
            f"COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), "
            f"COALESCE(AVG(duration_ms),0) FROM usage_log{where} "
            f"GROUP BY date, api_key ORDER BY date DESC, requests DESC"
        )
        rows = db_execute(query, params)
        return jsonify([{
            "date": r[0], "api_key": r[1], "requests": r[2],
            "input_tokens": r[3], "output_tokens": r[4],
            "avg_duration_ms": round(r[5]),
        } for r in rows])

    query = (
        f"SELECT api_key, COUNT(*) as requests, COALESCE(SUM(input_tokens),0), "
        f"COALESCE(SUM(output_tokens),0), COALESCE(AVG(duration_ms),0) FROM usage_log{where} "
        f"GROUP BY api_key ORDER BY requests DESC"
    )
    rows = db_execute(query, params)
    return jsonify([{
        "api_key": r[0], "requests": r[1],
        "input_tokens": r[2], "output_tokens": r[3],
        "avg_duration_ms": round(r[4]),
    } for r in rows])


if __name__ == "__main__":
    get_token()
    app.run(host="0.0.0.0", port=6655)
