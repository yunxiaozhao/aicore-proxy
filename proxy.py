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
import threading
import requests as req_lib
from flask import Flask, request, Response, jsonify

# ---------------------------------------------------------------------------
# Configuration — from docker-compose.yml environment
# ---------------------------------------------------------------------------
CLIENT_ID = os.environ["SAP_CLIENT_ID"]
CLIENT_SECRET = os.environ["SAP_CLIENT_SECRET"]
AUTH_URL = os.environ["SAP_AUTH_URL"]         # XSUAA token endpoint base URL
AI_API_URL = os.environ["SAP_AI_API_URL"]     # SAP AI Core API base URL
DEPLOYMENT_ID = os.environ["SAP_DEPLOYMENT_ID"]
RESOURCE_GROUP = os.environ.get("SAP_RESOURCE_GROUP", "default")
VERBOSE = os.environ.get("VERBOSE", "false").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# OAuth2 Token Management
#
# - 后台守护线程自动刷新 token，在过期前 5 分钟续期
# - 如果线程还没拿到 token，请求线程会同步获取一次
# - 刷新失败时指数退避重试（30s → 60s → … → 300s）
# ---------------------------------------------------------------------------
_token = None
_token_expires = 0
_token_lock = threading.Lock()
_refresh_started = False
_refresh_lock = threading.Lock()
_last_token_error = None


def _fetch_token():
    """向 SAP XSUAA 请求 OAuth2 access_token (client_credentials 模式)。

    成功后将 token 和过期时间写入全局变量；
    过期时间提前 300s，确保在真正失效前完成刷新。
    """
    global _token, _token_expires, _last_token_error
    resp = req_lib.post(
        f"{AUTH_URL}/oauth/token?grant_type=client_credentials",
        auth=(CLIENT_ID, CLIENT_SECRET),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    expires_in = data.get("expires_in", 43200)
    with _token_lock:
        _token = data["access_token"]
        _token_expires = time.time() + expires_in - 300
        _last_token_error = None
    print(f"[proxy] Token refreshed, expires in {expires_in}s", flush=True)


def _refresh_loop():
    """后台守护线程主循环：周期性刷新 token。

    正常流程：拿到 token → 睡眠到快过期 → 再次刷新。
    异常流程：指数退避重试（30s / 60s / 120s / 300s 上限）。
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
    """确保 token 刷新守护线程已启动（仅启动一次）。"""
    global _refresh_started
    with _refresh_lock:
        if not _refresh_started:
            _refresh_started = True
            t = threading.Thread(target=_refresh_loop, daemon=True)
            t.start()


def _get_token():
    """获取当前有效的 access_token。

    1. 确保后台刷新线程已启动
    2. 如果 token 为空或已过期，同步阻塞获取一次
    3. 返回 token（可能为 None，调用方需处理）
    """
    _ensure_refresh_thread()
    if _token is None or time.time() > _token_expires:
        try:
            _fetch_token()
        except Exception as e:
            with _token_lock:
                global _last_token_error
                _last_token_error = str(e)
            print(f"[proxy] Synchronous token fetch error: {e}", flush=True)
    with _token_lock:
        return _token


def _forward_to_sap(headers, body, stream):
    """将请求转发到 SAP AI Core，遇 401 自动刷新 token 重试一次。

    Args:
        headers: 转发请求头（含 Authorization）
        body:    请求体 dict
        stream:  是否使用流式传输

    Returns:
        requests.Response 对象

    Raises:
        req_lib.Timeout: 上游超时
        req_lib.RequestException: 其他网络错误
    """
    subpath = "invoke-with-response-stream" if stream else "invoke"
    target_url = f"{AI_API_URL}/v2/inference/deployments/{DEPLOYMENT_ID}/{subpath}"
    sap_resp = req_lib.post(target_url, headers=headers, json=body, stream=stream, timeout=300)
    if sap_resp.status_code == 401:
        _fetch_token()
        with _token_lock:
            headers["Authorization"] = f"Bearer {_token}"
        sap_resp = req_lib.post(target_url, headers=headers, json=body, stream=stream, timeout=300)
    return sap_resp


def _inject_sse_events(sap_resp):
    """将 SAP/Bedrock 的 SSE 流转换为 Anthropic 标准 SSE 格式。

    SAP 返回:   data: {"type":"message_start",...}\\n\\n
    Anthropic:  event: message_start\\ndata: {"type":"message_start",...}\\n\\n

    逐行读取上游 SSE，为每个 data: 行前插入对应的 event: 行。
    """
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


def _adapt_body(body):
    """适配 Anthropic 请求体为 SAP AI Core 兼容格式。

    返回 (body, is_stream)。就地修改 body dict。
    """
    if "anthropic_version" not in body:
        body["anthropic_version"] = "bedrock-2023-05-31"

    # SAP AI Core 不接受以下字段
    body.pop("model", None)
    is_stream = body.pop("stream", False)
    body.pop("context_management", None)

    # Bedrock 不支持 Anthropic 内置工具（web_search_20250305、text_editor_20250124 等），
    # 只保留标准自定义工具（type 为 "custom" 或无 type 字段）
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
    """核心代理端点：接收 Anthropic Messages API 请求，转发至 SAP AI Core。

    处理流程：
    1. 获取 OAuth2 token
    2. 校验并适配请求体（移除不支持的字段、过滤内置工具）
    3. 转发到 SAP AI Core（streaming 走 /invoke-with-response-stream，否则走 /invoke）
    4. 流式请求注入 SSE event 行后透传；非流式返回完整 JSON
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
        print(f"[proxy] >>> stream={is_stream}, body keys: {list(body.keys())}", flush=True)
        print(f"[proxy] >>> body: {json.dumps(body, ensure_ascii=False, default=str)[:2000]}", flush=True)

    try:
        sap_resp = _forward_to_sap(headers, body, stream=is_stream)
    except req_lib.Timeout:
        return jsonify({"error": "Upstream SAP AI Core timeout"}), 504
    except req_lib.RequestException as e:
        return jsonify({"error": f"Upstream SAP AI Core request failed: {e}"}), 502

    if VERBOSE:
        print(f"[proxy] <<< status: {sap_resp.status_code}", flush=True)

    # 非 200 统一记录错误（无论 VERBOSE 是否开启都打印，方便调试）
    if sap_resp.status_code != 200:
        error_body = sap_resp.content.decode("utf-8", errors="replace")
        print(f"[proxy] <<< {sap_resp.status_code} error: {error_body[:2000]}", flush=True)
        return Response(sap_resp.content, status=sap_resp.status_code,
                        content_type=sap_resp.headers.get("Content-Type", "application/json"))

    # 成功响应
    if is_stream:
        return Response(_inject_sse_events(sap_resp), status=200, content_type="text/event-stream")

    return Response(sap_resp.content, status=200,
                    content_type=sap_resp.headers.get("Content-Type", "application/json"))


@app.route("/health", methods=["GET"])
def health():
    """健康检查端点，返回 token 状态，供 Docker healthcheck 和监控使用。"""
    token = _get_token()
    with _token_lock:
        token_error = _last_token_error
    return jsonify({"status": "ok", "has_token": token is not None, "token_error": token_error})


if __name__ == "__main__":
    _get_token()
    app.run(host="0.0.0.0", port=6655)
