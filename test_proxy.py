"""Unit tests for aicore-proxy — load balancing, body adaptation, SSE injection, auth, and stats."""

import json
import os
import threading
import time
from unittest.mock import patch, MagicMock

# Set required env vars before importing modules
os.environ.setdefault("SAP_CLIENT_ID", "test-id")
os.environ.setdefault("SAP_CLIENT_SECRET", "test-secret")
os.environ.setdefault("SAP_AUTH_URL", "https://auth.example.com")
os.environ.setdefault("SAP_AI_API_URL", "https://api.example.com")
os.environ.setdefault("SAP_DEPLOYMENT_ID", "dep-a,dep-b,dep-c")

import pytest
import config
import proxy
import app as app_module


# ---------------------------------------------------------------------------
# Least-connections load balancing
# ---------------------------------------------------------------------------

class TestLeastConnections:
    def setup_method(self):
        with proxy._deployment_lock:
            for dep_id in proxy._deployment_active:
                proxy._deployment_active[dep_id] = 0

    def test_picks_least_active(self):
        with proxy._deployment_lock:
            proxy._deployment_active["dep-a"] = 5
            proxy._deployment_active["dep-b"] = 1
            proxy._deployment_active["dep-c"] = 3
        dep = proxy._next_deployment()
        assert dep == "dep-b"
        assert proxy._deployment_active["dep-b"] == 2

    def test_round_robin_on_tie(self):
        dep = proxy._next_deployment()
        assert dep == "dep-a"
        assert proxy._deployment_active["dep-a"] == 1

    def test_release_decrements(self):
        with proxy._deployment_lock:
            proxy._deployment_active["dep-a"] = 3
        proxy._release_deployment("dep-a")
        assert proxy._deployment_active["dep-a"] == 2

    def test_release_floor_zero(self):
        proxy._release_deployment("dep-a")
        assert proxy._deployment_active["dep-a"] == 0

    def test_concurrent_distribution(self):
        deps = [proxy._next_deployment() for _ in range(3)]
        assert sorted(deps) == ["dep-a", "dep-b", "dep-c"]

    def test_concurrent_threads(self):
        results = []

        def acquire_and_release():
            dep = proxy._next_deployment()
            results.append(dep)
            time.sleep(0.01)
            proxy._release_deployment(dep)

        threads = [threading.Thread(target=acquire_and_release) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 30
        for dep_id in proxy._deployment_active:
            assert proxy._deployment_active[dep_id] == 0


# ---------------------------------------------------------------------------
# Body adaptation
# ---------------------------------------------------------------------------

class TestAdaptBody:
    def test_strips_model_and_stream(self):
        body = {"model": "claude-3", "stream": True, "messages": []}
        adapted, is_stream = proxy.adapt_body(body)
        assert "model" not in adapted
        assert "stream" not in adapted
        assert is_stream is True

    def test_adds_anthropic_version(self):
        body = {"messages": []}
        adapted, _ = proxy.adapt_body(body)
        assert adapted["anthropic_version"] == "bedrock-2023-05-31"

    def test_preserves_existing_anthropic_version(self):
        body = {"anthropic_version": "custom-ver", "messages": []}
        adapted, _ = proxy.adapt_body(body)
        assert adapted["anthropic_version"] == "custom-ver"

    def test_strips_context_management(self):
        body = {"context_management": {"mode": "auto"}, "messages": []}
        adapted, _ = proxy.adapt_body(body)
        assert "context_management" not in adapted

    def test_strips_cache_control(self):
        body = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
                ]}
            ],
            "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        }
        adapted, _ = proxy.adapt_body(body)
        assert "cache_control" not in adapted["messages"][0]["content"][0]
        assert "cache_control" not in adapted["system"][0]

    def test_filters_builtin_tools(self):
        body = {
            "messages": [],
            "tools": [
                {"name": "my_tool", "description": "custom"},
                {"type": "web_search_20250305", "name": "web_search"},
                {"type": "text_editor_20250124", "name": "text_editor"},
                {"type": "custom", "name": "another_tool", "description": "also custom"},
            ],
            "tool_choice": {"type": "auto"},
        }
        adapted, _ = proxy.adapt_body(body)
        assert len(adapted["tools"]) == 2
        assert adapted["tools"][0]["name"] == "my_tool"
        assert adapted["tools"][1]["name"] == "another_tool"

    def test_removes_tools_key_when_all_filtered(self):
        body = {
            "messages": [],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "tool_choice": {"type": "auto"},
        }
        adapted, _ = proxy.adapt_body(body)
        assert "tools" not in adapted
        assert "tool_choice" not in adapted

    def test_default_stream_false(self):
        body = {"messages": []}
        _, is_stream = proxy.adapt_body(body)
        assert is_stream is False


# ---------------------------------------------------------------------------
# SSE event injection
# ---------------------------------------------------------------------------

class TestInjectSSEEvents:
    def _make_mock_resp(self, lines):
        resp = MagicMock()
        resp.iter_lines.return_value = [line.encode("utf-8") for line in lines]
        return resp

    def _reset_active(self):
        with proxy._deployment_lock:
            for dep_id in proxy._deployment_active:
                proxy._deployment_active[dep_id] = 0

    def test_injects_event_line(self):
        resp = self._make_mock_resp([
            'data: {"type":"message_start","message":{"id":"msg_1"}}',
        ])
        chunks = list(proxy.inject_sse_events(resp, "dep-a"))
        output = b"".join(chunks).decode("utf-8")
        assert "event: message_start\n" in output
        assert 'data: {"type":"message_start"' in output

    def test_no_event_for_non_typed_data(self):
        resp = self._make_mock_resp(['data: {"key":"value"}'])
        chunks = list(proxy.inject_sse_events(resp, "dep-a"))
        output = b"".join(chunks).decode("utf-8")
        assert "event:" not in output
        assert 'data: {"key":"value"}' in output

    def test_passthrough_non_data_lines(self):
        resp = self._make_mock_resp([": ping"])
        chunks = list(proxy.inject_sse_events(resp, "dep-a"))
        output = b"".join(chunks).decode("utf-8")
        assert ": ping" in output

    def test_skips_blank_lines(self):
        resp = self._make_mock_resp(["", "  "])
        chunks = list(proxy.inject_sse_events(resp, "dep-a"))
        assert chunks == []

    def test_releases_deployment_on_completion(self):
        self._reset_active()
        with proxy._deployment_lock:
            proxy._deployment_active["dep-a"] = 1
        resp = self._make_mock_resp(['data: {"type":"message_stop"}'])
        list(proxy.inject_sse_events(resp, "dep-a"))
        assert proxy._deployment_active["dep-a"] == 0
        resp.close.assert_called_once()

    def test_releases_deployment_on_error(self):
        self._reset_active()
        with proxy._deployment_lock:
            proxy._deployment_active["dep-b"] = 1
        resp = MagicMock()
        resp.iter_lines.side_effect = Exception("connection lost")
        try:
            list(proxy.inject_sse_events(resp, "dep-b"))
        except Exception:
            pass
        assert proxy._deployment_active["dep-b"] == 0

    def test_accumulates_streaming_tokens(self):
        """SSE events should accumulate input/output token counts."""
        self._reset_active()
        resp = self._make_mock_resp([
            'data: {"type":"message_start","message":{"usage":{"input_tokens":150}}}',
            'data: {"type":"content_block_delta","delta":{"text":"hi"}}',
            'data: {"type":"message_delta","usage":{"output_tokens":42}}',
        ])
        with patch("proxy.log_usage") as mock_log:
            list(proxy.inject_sse_events(resp, "dep-a", "sk-test", time.time()))
            mock_log.assert_called_once()
            args = mock_log.call_args[0]
            assert args[0] == "sk-test"   # client_key
            assert args[2] == 150         # input_tokens
            assert args[3] == 42          # output_tokens


# ---------------------------------------------------------------------------
# Config file loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    def test_cfg_env_var_priority(self):
        """Env var should take priority over config file."""
        os.environ["SAP_CLIENT_ID"] = "test-id"
        assert config._cfg("SAP_CLIENT_ID", "sap_client_id") == "test-id"

    def test_cfg_default(self):
        """Should return default when neither env var nor config file has the key."""
        assert config._cfg("NONEXISTENT_VAR_12345", "nonexistent", "fallback") == "fallback"

    def test_load_config_file_missing(self):
        """Missing config file should return empty dict."""
        with patch("config._CONFIG_PATH", "/nonexistent/path.json"):
            config._config_last_check = 0
            result = config._load_config_file()
            assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# API Key Authentication
# ---------------------------------------------------------------------------

class TestApiKeyAuth:
    def setup_method(self):
        """Reset key cache before each test."""
        with config._api_keys_lock:
            config._api_keys_cache = set()
            config._api_keys_last_refresh = 0

    def test_no_keys_means_no_auth(self):
        """When no keys are configured, auth should be disabled."""
        with patch.dict(os.environ, {"API_KEYS": ""}, clear=False):
            with patch("config._load_config_file", return_value={}):
                with config._api_keys_lock:
                    config._api_keys_last_refresh = 0
                assert not config.auth_enabled()

    def test_env_var_keys(self):
        with patch.dict(os.environ, {"API_KEYS": "sk-abc,sk-def"}, clear=False):
            with patch("config._load_config_file", return_value={}):
                with config._api_keys_lock:
                    config._api_keys_last_refresh = 0
                assert config.auth_enabled()
                assert config.validate_api_key("sk-abc")
                assert config.validate_api_key("sk-def")
                assert not config.validate_api_key("sk-wrong")

    def test_config_file_keys(self):
        with patch.dict(os.environ, {"API_KEYS": ""}, clear=False):
            with patch("config._load_config_file", return_value={"api_keys": ["sk-from-config"]}):
                with config._api_keys_lock:
                    config._api_keys_last_refresh = 0
                assert config.auth_enabled()
                assert config.validate_api_key("sk-from-config")

    def test_merged_keys(self):
        with patch.dict(os.environ, {"API_KEYS": "sk-env"}, clear=False):
            with patch("config._load_config_file", return_value={"api_keys": ["sk-cfg"]}):
                with config._api_keys_lock:
                    config._api_keys_last_refresh = 0
                assert config.validate_api_key("sk-env")
                assert config.validate_api_key("sk-cfg")

    def test_empty_key_rejected(self):
        assert not config.validate_api_key("")
        assert not config.validate_api_key(None)


# ---------------------------------------------------------------------------
# Flask endpoint tests
# ---------------------------------------------------------------------------

class TestMessagesEndpoint:
    @pytest.fixture
    def client(self):
        app_module.app.config["TESTING"] = True
        with app_module.app.test_client() as c:
            yield c

    def setup_method(self):
        with proxy._deployment_lock:
            for dep_id in proxy._deployment_active:
                proxy._deployment_active[dep_id] = 0
        with config._api_keys_lock:
            config._api_keys_cache = set()
            config._api_keys_last_refresh = 0

    @patch("app.get_token", return_value="fake-token")
    @patch("proxy._api_session")
    @patch("app.auth_enabled", return_value=False)
    def test_non_streaming_success(self, mock_auth, mock_session, mock_token, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = json.dumps({"type": "message", "content": [], "usage": {"input_tokens": 10, "output_tokens": 5}}).encode()
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_session.post.return_value = mock_resp

        resp = client.post("/v1/messages", json={
            "model": "claude-3", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200
        for dep_id in proxy._deployment_active:
            assert proxy._deployment_active[dep_id] == 0

    @patch("app.get_token", return_value=None)
    @patch("app.auth_enabled", return_value=False)
    def test_no_token_returns_503(self, mock_auth, mock_token, client):
        resp = client.post("/v1/messages", json={"messages": []})
        assert resp.status_code == 503

    @patch("app.auth_enabled", return_value=False)
    def test_invalid_body_returns_400(self, mock_auth, client):
        with patch("app.get_token", return_value="fake"):
            resp = client.post("/v1/messages", data="not json",
                               content_type="application/json")
            assert resp.status_code == 400

    @patch("app.auth_enabled", return_value=True)
    @patch("app.validate_api_key", return_value=False)
    def test_auth_rejects_invalid_key(self, mock_validate, mock_auth, client):
        resp = client.post("/v1/messages", json={"messages": []},
                           headers={"x-api-key": "bad-key"})
        assert resp.status_code == 401

    @patch("app.get_token", return_value="fake-token")
    @patch("proxy._api_session")
    @patch("app.auth_enabled", return_value=True)
    @patch("app.validate_api_key", return_value=True)
    def test_auth_accepts_valid_key(self, mock_validate, mock_auth, mock_session, mock_token, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = json.dumps({"type": "message", "content": []}).encode()
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_session.post.return_value = mock_resp

        resp = client.post("/v1/messages", json={
            "model": "claude-3", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
        }, headers={"x-api-key": "sk-valid"})
        assert resp.status_code == 200

    @patch("app.get_token", return_value="fake-token")
    @patch("proxy._api_session")
    @patch("app.auth_enabled", return_value=True)
    @patch("app.validate_api_key", return_value=True)
    def test_auth_via_bearer_header(self, mock_validate, mock_auth, mock_session, mock_token, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = json.dumps({"type": "message", "content": []}).encode()
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_session.post.return_value = mock_resp

        resp = client.post("/v1/messages", json={
            "model": "claude-3", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
        }, headers={"Authorization": "Bearer sk-valid"})
        assert resp.status_code == 200


class TestHealthEndpoint:
    @pytest.fixture
    def client(self):
        app_module.app.config["TESTING"] = True
        with app_module.app.test_client() as c:
            yield c

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["deployments"] == 3
        assert "stats_enabled" in data
        assert "auth_enabled" in data


class TestStatsEndpoint:
    @pytest.fixture
    def client(self):
        app_module.app.config["TESTING"] = True
        with app_module.app.test_client() as c:
            yield c

    def test_stats_returns_active_counts(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "deployments_active" in data
        assert "dep-a" in data["deployments_active"]
