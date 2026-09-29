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
# Model-aware routing
# ---------------------------------------------------------------------------

class TestModelAwareRouting:
    def setup_method(self):
        # Snapshot and patch the per-model config so tests are isolated.
        self._orig_by_model = proxy.DEPLOYMENT_IDS_BY_MODEL
        self._orig_kw = proxy.MODEL_KEYWORDS
        proxy.DEPLOYMENT_IDS_BY_MODEL = {
            "opus": ["dep-a"],
            "sonnet": ["dep-b"],
            "haiku": ["dep-c"],
        }
        with proxy._deployment_lock:
            for dep_id in proxy._deployment_active:
                proxy._deployment_active[dep_id] = 0

    def teardown_method(self):
        proxy.DEPLOYMENT_IDS_BY_MODEL = self._orig_by_model
        with proxy._deployment_lock:
            for dep_id in proxy._deployment_active:
                proxy._deployment_active[dep_id] = 0

    def test_routes_opus(self):
        assert proxy._next_deployment("claude-opus-4-8") == "dep-a"

    def test_routes_sonnet(self):
        assert proxy._next_deployment("claude-sonnet-4-6") == "dep-b"

    def test_routes_haiku(self):
        assert proxy._next_deployment("claude-haiku-4-5-20251001") == "dep-c"

    def test_unknown_model_falls_back_to_full_pool(self):
        # No keyword match → falls back to DEPLOYMENT_IDS (least-active overall).
        dep = proxy._next_deployment("gpt-4")
        assert dep in proxy.DEPLOYMENT_IDS

    def test_no_hint_falls_back_to_full_pool(self):
        dep = proxy._next_deployment(None)
        assert dep in proxy.DEPLOYMENT_IDS

    def test_flat_mode_ignores_hint(self):
        # When no per-model config is set, hint is ignored entirely.
        proxy.DEPLOYMENT_IDS_BY_MODEL = {}
        dep = proxy._next_deployment("claude-opus-4-8")
        assert dep in proxy.DEPLOYMENT_IDS

    def test_empty_pool_falls_back(self):
        proxy.DEPLOYMENT_IDS_BY_MODEL = {"opus": []}
        dep = proxy._next_deployment("claude-opus-4-8")
        assert dep in proxy.DEPLOYMENT_IDS


# ---------------------------------------------------------------------------
# Body adaptation
# ---------------------------------------------------------------------------

class TestAdaptBody:
    def test_strips_model_and_stream(self):
        body = {"model": "claude-3", "stream": True, "messages": []}
        adapted, is_stream, model = proxy.adapt_body(body)
        assert "model" not in adapted
        assert "stream" not in adapted
        assert is_stream is True
        assert model == "claude-3"

    def test_adds_anthropic_version(self):
        body = {"messages": []}
        adapted, _, _ = proxy.adapt_body(body)
        assert adapted["anthropic_version"] == "bedrock-2023-05-31"

    def test_preserves_existing_anthropic_version(self):
        body = {"anthropic_version": "custom-ver", "messages": []}
        adapted, _, _ = proxy.adapt_body(body)
        assert adapted["anthropic_version"] == "custom-ver"

    def test_strips_context_management(self):
        body = {"context_management": {"mode": "auto"}, "messages": []}
        adapted, _, _ = proxy.adapt_body(body)
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
        adapted, _, _ = proxy.adapt_body(body)
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
        adapted, _, _ = proxy.adapt_body(body)
        assert len(adapted["tools"]) == 2
        assert adapted["tools"][0]["name"] == "my_tool"
        assert adapted["tools"][1]["name"] == "another_tool"

    def test_removes_tools_key_when_all_filtered(self):
        body = {
            "messages": [],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "tool_choice": {"type": "auto"},
        }
        adapted, _, _ = proxy.adapt_body(body)
        assert "tools" not in adapted
        assert "tool_choice" not in adapted

    def test_default_stream_false(self):
        body = {"messages": []}
        _, is_stream, _ = proxy.adapt_body(body)
        assert is_stream is False

    def test_returns_none_model_when_absent(self):
        body = {"messages": []}
        _, _, model = proxy.adapt_body(body)
        assert model is None


# ---------------------------------------------------------------------------
# SSE event injection
# ---------------------------------------------------------------------------

class TestInjectSSEEvents:
    def _make_mock_resp(self, lines):
        resp = MagicMock()
        # inject_sse_events now uses iter_content with SSE "\n\n" event
        # boundaries — feed one full event (line + blank line) per chunk.
        chunks = [(line + "\n\n").encode("utf-8") for line in lines]
        resp.iter_content.return_value = chunks
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
        resp.iter_content.side_effect = Exception("connection lost")
        try:
            list(proxy.inject_sse_events(resp, "dep-b"))
        except Exception:
            pass
        assert proxy._deployment_active["dep-b"] == 0

    def test_flushes_event_split_across_chunks(self):
        """Bytes belonging to one event can arrive in multiple socket chunks;
        the event must still be emitted as a single, complete SSE block."""
        resp = MagicMock()
        # One logical event split into 3 raw byte chunks (mid-JSON and
        # straddling the "\n\n" boundary).
        resp.iter_content.return_value = [
            b'data: {"type":"content_block_delta"',
            b',"delta":{"text":"hello"}}',
            b'\n\ndata: {"type":"message_stop"}\n\n',
        ]
        chunks = list(proxy.inject_sse_events(resp, "dep-a"))
        output = b"".join(chunks).decode("utf-8")
        # Both events must be present, in order, each with its event: header.
        first = output.index("event: content_block_delta")
        second = output.index("event: message_stop")
        assert first < second
        assert '"text":"hello"' in output

    def test_flushes_trailing_event_without_blank_line(self):
        """If upstream closes without a final blank line, the last buffered
        event must still be flushed rather than silently dropped."""
        resp = MagicMock()
        resp.iter_content.return_value = [
            b'data: {"type":"message_stop"}',   # note: no trailing \n\n
        ]
        chunks = list(proxy.inject_sse_events(resp, "dep-a"))
        output = b"".join(chunks).decode("utf-8")
        assert "event: message_stop" in output

    def test_smooth_stream_splits_text_delta(self):
        """When SMOOTH_STREAM is on, one text_delta is split into multiple
        smaller content_block_delta events preserving index and total text."""
        resp = self._make_mock_resp([
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"abcdefgh"}}',
        ])
        with patch.object(proxy, "SMOOTH_STREAM", True), \
             patch.object(proxy, "SMOOTH_STREAM_CHARS", 2), \
             patch.object(proxy, "SMOOTH_STREAM_DELAY_MS", 0):
            chunks = list(proxy.inject_sse_events(resp, "dep-a"))
        output = b"".join(chunks).decode("utf-8")
        # 8 chars / 2 per delta = 4 emitted content_block_delta events.
        assert output.count("event: content_block_delta") == 4
        # Concatenating the emitted texts must equal the original.
        import re
        pieces = re.findall(r'"text":\s*"([^"]*)"', output)
        assert "".join(pieces) == "abcdefgh"
        # Each split retains the original index.
        assert output.count('"index": 0') + output.count('"index":0') == 4

    def test_smooth_stream_leaves_input_json_delta_intact(self):
        """Tool-call JSON deltas must NEVER be split — partial JSON would
        break the client's accumulator."""
        payload = ('data: {"type":"content_block_delta","index":1,'
                   '"delta":{"type":"input_json_delta","partial_json":"{\\"x\\":1}"}}')
        resp = self._make_mock_resp([payload])
        with patch.object(proxy, "SMOOTH_STREAM", True), \
             patch.object(proxy, "SMOOTH_STREAM_CHARS", 2), \
             patch.object(proxy, "SMOOTH_STREAM_DELAY_MS", 0):
            chunks = list(proxy.inject_sse_events(resp, "dep-a"))
        output = b"".join(chunks).decode("utf-8")
        # Exactly one event emitted, with the JSON string intact.
        assert output.count("event: content_block_delta") == 1
        assert '"partial_json":"{\\"x\\":1}"' in output

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
            config._api_key_hashes = set()
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
            config._api_key_hashes = set()
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
        # Non-verbose response must NOT leak deployment-level detail — that's
        # what ?verbose=1 is for.
        assert "deployments_active" not in data
        assert "models_cache" not in data

    def test_health_verbose_exposes_diagnostics(self, client):
        _reset_models_cache()
        with models_module._cache_lock:
            models_module._cache["models"] = [models_module._enrich("claude-opus-5-5")]
            models_module._cache["fetched_at"] = time.time()
            models_module._cache["source"] = "sap"
        resp = client.get("/health?verbose=1")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "deployments_active" in data
        assert "resource_group" in data
        assert "models_cache" in data
        assert data["models_cache"]["source"] in ("sap", "fallback", "none")


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


# ---------------------------------------------------------------------------
# Models module — SAP-derived catalog with cache + fallback
# ---------------------------------------------------------------------------

import models as models_module


def _fake_sap_deployments(*model_specs):
    """Build a mock GET /v2/lm/deployments payload.

    Each spec is (raw_model_name, version, status, created_at). Version and
    created_at may be None.
    """
    resources = []
    for name, version, status, created_at in model_specs:
        model_block = {"name": name}
        if version:
            model_block["version"] = version
        resources.append({
            "id": f"dep-{name}",
            "status": status,
            "createdAt": created_at,
            "details": {
                "resources": {"backend_details": {"model": model_block}},
            },
        })
    return {"count": len(resources), "resources": resources}


def _reset_models_cache():
    """Wipe the models module's cache between tests so each starts clean."""
    with models_module._cache_lock:
        models_module._cache["models"] = []
        models_module._cache["fetched_at"] = 0.0
        models_module._cache["last_error"] = None
        models_module._cache["source"] = "none"


class TestModelsUnit:
    """Unit tests for the model-name normalizer and deployment extractor."""

    def test_normalize_strips_anthropic_double_dash(self):
        assert models_module._normalize_id("anthropic--claude-4.6-opus") == "claude-4.6-opus"

    def test_normalize_strips_bedrock_dot_prefix(self):
        assert models_module._normalize_id("anthropic.claude-3-5-sonnet-20240620") == "claude-3-5-sonnet-20240620"

    def test_normalize_lowercases(self):
        assert models_module._normalize_id("Claude-OPUS-5-5") == "claude-opus-5-5"

    def test_normalize_empty(self):
        assert models_module._normalize_id("") == ""
        assert models_module._normalize_id(None) == ""

    def test_family_detection(self):
        assert models_module._family_of("claude-opus-5-5") == "opus"
        assert models_module._family_of("claude-sonnet-5") == "sonnet"
        assert models_module._family_of("claude-haiku-4-5") == "haiku"
        assert models_module._family_of("gpt-4") == ""

    def test_extract_model_from_deployment(self):
        dep = {
            "details": {"resources": {"backend_details": {
                "model": {"name": "anthropic--claude-4.6-opus", "version": "1.0"}
            }}},
            "createdAt": "2025-01-01T00:00:00Z",
        }
        name, version, created_at = models_module._extract_model_from_deployment(dep)
        assert name == "anthropic--claude-4.6-opus"
        assert version == "1.0"
        assert created_at == "2025-01-01T00:00:00Z"

    def test_extract_from_deployment_missing_model(self):
        assert models_module._extract_model_from_deployment({}) == (None, None, None)
        assert models_module._extract_model_from_deployment({"details": {}}) == (None, None, None)

    def test_enrich_known_model_uses_metadata(self):
        entry = models_module._enrich("claude-opus-5-5", created_at="2026-01-01T00:00:00Z")
        assert entry["display_name"] == "Claude Opus 5.5"
        assert entry["max_input_tokens"] == 1_000_000
        assert entry["capabilities"]["effort"]["supported"] is True

    def test_enrich_haiku_uses_haiku_caps(self):
        entry = models_module._enrich("claude-haiku-4-5")
        assert entry["capabilities"]["effort"]["supported"] is False
        assert entry["capabilities"]["thinking"]["types"]["adaptive"]["supported"] is False

    def test_enrich_unknown_model_falls_back_to_family(self):
        entry = models_module._enrich("claude-haiku-99-9")
        # Not in metadata table, but family match => haiku caps
        assert entry["capabilities"]["effort"]["supported"] is False
        assert entry["type"] == "model"
        assert entry["id"] == "claude-haiku-99-9"


class TestModelsFetch:
    """Tests for the SAP-fetch path (with the network mocked out)."""

    def setup_method(self):
        _reset_models_cache()

    @patch("proxy._api_session")
    @patch("proxy.get_token", return_value="fake-token")
    def test_fetch_extracts_running_deployments(self, mock_token, mock_session):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _fake_sap_deployments(
            ("anthropic--claude-opus-5-5", None, "RUNNING", "2026-09-22T00:00:00Z"),
            ("anthropic--claude-sonnet-5-5", None, "RUNNING", "2026-09-28T00:00:00Z"),
        )
        mock_session.get.return_value = mock_resp

        models = models_module._fetch_from_sap()
        ids = [m["id"] for m in models]
        assert "claude-opus-5-5" in ids
        assert "claude-sonnet-5-5" in ids
        # Newest createdAt first.
        assert ids[0] == "claude-sonnet-5-5"

    @patch("proxy._api_session")
    @patch("proxy.get_token", return_value="fake-token")
    def test_fetch_skips_non_running(self, mock_token, mock_session):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _fake_sap_deployments(
            ("anthropic--claude-opus-5-5", None, "PENDING", None),
            ("anthropic--claude-sonnet-5-5", None, "RUNNING", None),
        )
        mock_session.get.return_value = mock_resp

        models = models_module._fetch_from_sap()
        ids = [m["id"] for m in models]
        assert ids == ["claude-sonnet-5-5"]

    @patch("proxy._api_session")
    @patch("proxy.get_token", return_value="fake-token")
    def test_fetch_dedupes_same_model_across_deployments(self, mock_token, mock_session):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # Same model, three RUNNING deployments — should collapse to one entry.
        mock_resp.json.return_value = _fake_sap_deployments(
            ("anthropic--claude-opus-5-5", None, "RUNNING", "2026-01-01T00:00:00Z"),
            ("anthropic--claude-opus-5-5", None, "RUNNING", "2026-02-01T00:00:00Z"),
            ("anthropic--claude-opus-5-5", None, "RUNNING", "2026-03-01T00:00:00Z"),
        )
        mock_session.get.return_value = mock_resp

        models = models_module._fetch_from_sap()
        assert len(models) == 1
        # Keep the newest createdAt of the group.
        assert models[0]["created_at"] == "2026-03-01T00:00:00Z"

    @patch("proxy._api_session")
    @patch("proxy.get_token", return_value="fake-token")
    def test_fetch_appends_version_suffix(self, mock_token, mock_session):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _fake_sap_deployments(
            ("anthropic--claude-haiku-4-5", "20251001", "RUNNING", "2025-10-01T00:00:00Z"),
        )
        mock_session.get.return_value = mock_resp

        models = models_module._fetch_from_sap()
        assert models[0]["id"] == "claude-haiku-4-5-20251001"

    @patch("proxy._api_session")
    @patch("proxy.get_token", return_value="fake-token")
    def test_fetch_ignores_version_latest(self, mock_token, mock_session):
        # "latest" is a pointer, not a snapshot — don't leak it into the id.
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _fake_sap_deployments(
            ("anthropic--claude-opus-5-5", "latest", "RUNNING", None),
        )
        mock_session.get.return_value = mock_resp

        models = models_module._fetch_from_sap()
        assert models[0]["id"] == "claude-opus-5-5"

    @patch("proxy._api_session")
    @patch("proxy.get_token", return_value="fake-token")
    def test_refresh_uses_cache_within_ttl(self, mock_token, mock_session):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _fake_sap_deployments(
            ("anthropic--claude-opus-5-5", None, "RUNNING", None),
        )
        mock_session.get.return_value = mock_resp

        models_module._refresh_cache(force=True)
        models_module._refresh_cache()  # within TTL — should not re-call
        # get called exactly once (the force refresh), not again.
        assert mock_session.get.call_count == 1

    @patch("proxy._api_session")
    @patch("proxy.get_token", return_value=None)
    def test_refresh_falls_back_on_missing_token(self, mock_token, mock_session):
        # No token available — fetch raises, refresh keeps prior cache empty
        # and (with static fallback enabled) fills in the built-in list.
        with patch.object(models_module, "_STATIC_FALLBACK", True):
            models = models_module._refresh_cache(force=True)
        assert len(models) > 0
        status = models_module.get_cache_status()
        assert status["source"] == "fallback"
        assert status["last_error"] is not None

    @patch("proxy._api_session")
    @patch("proxy.get_token", return_value=None)
    def test_refresh_returns_empty_when_fallback_disabled(self, mock_token, mock_session):
        with patch.object(models_module, "_STATIC_FALLBACK", False):
            models = models_module._refresh_cache(force=True)
        assert models == []


class TestModelsEndpoint:
    @pytest.fixture
    def client(self):
        app_module.app.config["TESTING"] = True
        with app_module.app.test_client() as c:
            yield c

    def setup_method(self):
        _reset_models_cache()

    def _preload_cache(self, ids):
        """Skip the SAP fetch by preloading the cache with known ids."""
        with models_module._cache_lock:
            models_module._cache["models"] = [models_module._enrich(i) for i in ids]
            models_module._cache["fetched_at"] = time.time()
            models_module._cache["source"] = "sap"
            models_module._cache["last_error"] = None

    @patch("app.auth_enabled", return_value=False)
    def test_list_returns_data_array(self, mock_auth, client):
        self._preload_cache(["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5"])
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data["data"], list)
        assert len(data["data"]) == 3
        for m in data["data"]:
            assert m["type"] == "model"
            assert "id" in m and isinstance(m["id"], str)
            assert "display_name" in m
            assert "created_at" in m
            assert "capabilities" in m
        assert "has_more" in data
        assert "first_id" in data
        assert "last_id" in data

    @patch("app.auth_enabled", return_value=False)
    def test_list_pagination_limit(self, mock_auth, client):
        self._preload_cache(["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5"])
        resp = client.get("/v1/models?limit=2")
        data = resp.get_json()
        assert len(data["data"]) == 2
        assert data["has_more"] is True

    @patch("app.auth_enabled", return_value=False)
    def test_list_pagination_after_id(self, mock_auth, client):
        self._preload_cache(["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5"])
        first = client.get("/v1/models?limit=2").get_json()
        after = first["last_id"]
        resp = client.get(f"/v1/models?limit=2&after_id={after}")
        data = resp.get_json()
        assert data["data"][0]["id"] != after
        assert data["first_id"] != after

    @patch("app.auth_enabled", return_value=False)
    def test_get_single_model(self, mock_auth, client):
        self._preload_cache(["claude-opus-5-5"])
        resp = client.get("/v1/models/claude-opus-5-5")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["id"] == "claude-opus-5-5"
        assert data["type"] == "model"

    @patch("app.auth_enabled", return_value=False)
    def test_get_unknown_model_returns_404(self, mock_auth, client):
        self._preload_cache(["claude-opus-5-5"])
        resp = client.get("/v1/models/nonsuch-model")
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "not_found_error"

    @patch("app.auth_enabled", return_value=True)
    @patch("app.validate_api_key", return_value=False)
    def test_list_rejects_invalid_key(self, mock_validate, mock_auth, client):
        resp = client.get("/v1/models", headers={"x-api-key": "bad"})
        assert resp.status_code == 401

    @patch("app.auth_enabled", return_value=True)
    @patch("app.validate_api_key", return_value=True)
    def test_list_accepts_valid_key(self, mock_validate, mock_auth, client):
        self._preload_cache(["claude-opus-5-5"])
        resp = client.get("/v1/models", headers={"x-api-key": "good"})
        assert resp.status_code == 200

    @patch("proxy._api_session")
    @patch("proxy.get_token", return_value="fake-token")
    @patch("app.auth_enabled", return_value=False)
    def test_endpoint_calls_sap_on_cold_cache(self, mock_auth, mock_token, mock_session, client):
        """A cold cache should trigger a real fetch through the endpoint path."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _fake_sap_deployments(
            ("anthropic--claude-opus-5-5", None, "RUNNING", "2026-09-22T00:00:00Z"),
        )
        mock_session.get.return_value = mock_resp

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert mock_session.get.called
        data = resp.get_json()
        assert data["data"][0]["id"] == "claude-opus-5-5"


class TestCountTokensEndpoint:
    """POST /v1/messages/count_tokens — Anthropic Token Count API.

    Implementation is a max_tokens=1 probe against SAP AI Core, so the mocked
    upstream response mirrors what a real deployment would return.
    """

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
            config._api_key_hashes = set()
            config._api_keys_last_refresh = 0

    def _mock_upstream(self, session_mock, input_tokens=42, output_tokens=1, status=200):
        """Wire up a fake SAP AI Core /invoke response with the given usage."""
        r = MagicMock()
        r.status_code = status
        r.content = json.dumps({
            "id": "msg_probe",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "."}],
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }).encode()
        r.headers = {"Content-Type": "application/json"}
        session_mock.post.return_value = r
        return r

    @patch("app.get_token", return_value="fake-token")
    @patch("proxy._api_session")
    @patch("app.auth_enabled", return_value=False)
    def test_returns_input_tokens(self, mock_auth, mock_session, mock_token, client):
        self._mock_upstream(mock_session, input_tokens=123)
        resp = client.post("/v1/messages/count_tokens", json={
            "model": "claude-opus-5-5",
            "messages": [{"role": "user", "content": "Hello, world"}],
        })
        assert resp.status_code == 200
        data = resp.get_json()
        # Anthropic-spec shape: exactly one field, `input_tokens`.
        assert data == {"input_tokens": 123}

    @patch("app.get_token", return_value="fake-token")
    @patch("proxy._api_session")
    @patch("app.auth_enabled", return_value=False)
    def test_forces_max_tokens_one(self, mock_auth, mock_session, mock_token, client):
        """Client-supplied max_tokens must be overridden to 1 (minimize output cost)."""
        self._mock_upstream(mock_session, input_tokens=10)
        client.post("/v1/messages/count_tokens", json={
            "model": "claude-opus-5-5",
            "max_tokens": 4096,  # client-supplied — should be squashed
            "messages": [{"role": "user", "content": "hi"}],
        })
        sent_body = mock_session.post.call_args.kwargs["json"]
        assert sent_body["max_tokens"] == 1

    @patch("app.get_token", return_value="fake-token")
    @patch("proxy._api_session")
    @patch("app.auth_enabled", return_value=False)
    def test_ignores_stream_flag(self, mock_auth, mock_session, mock_token, client):
        """count_tokens has no streaming — the upstream call is always non-stream."""
        self._mock_upstream(mock_session)
        client.post("/v1/messages/count_tokens", json={
            "model": "claude-opus-5-5",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        })
        # `stream` must not leak into the upstream body and the request must be
        # non-streaming (adapt_body strips it and we always pass stream=False).
        sent_body = mock_session.post.call_args.kwargs["json"]
        assert "stream" not in sent_body
        assert mock_session.post.call_args.kwargs["stream"] is False

    @patch("app.get_token", return_value="fake-token")
    @patch("proxy._api_session")
    @patch("app.auth_enabled", return_value=False)
    def test_forwards_system_and_tools(self, mock_auth, mock_session, mock_token, client):
        """count_tokens accepts the same shape as /v1/messages — system + tools included."""
        self._mock_upstream(mock_session, input_tokens=200)
        resp = client.post("/v1/messages/count_tokens", json={
            "model": "claude-opus-5-5",
            "system": "You are helpful.",
            "tools": [{"name": "search", "description": "web search",
                       "input_schema": {"type": "object", "properties": {}}}],
            "messages": [{"role": "user", "content": "find a thing"}],
        })
        assert resp.status_code == 200
        sent_body = mock_session.post.call_args.kwargs["json"]
        assert "system" in sent_body
        assert "tools" in sent_body

    @patch("app.get_token", return_value=None)
    @patch("app.auth_enabled", return_value=False)
    def test_no_token_returns_503(self, mock_auth, mock_token, client):
        resp = client.post("/v1/messages/count_tokens", json={
            "model": "claude-opus-5-5",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 503

    @patch("app.get_token", return_value="fake-token")
    @patch("app.auth_enabled", return_value=False)
    def test_invalid_body_returns_400(self, mock_auth, mock_token, client):
        resp = client.post("/v1/messages/count_tokens",
                           data="not json", content_type="application/json")
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "invalid_request_error"

    @patch("app.auth_enabled", return_value=True)
    @patch("app.validate_api_key", return_value=False)
    def test_rejects_invalid_key(self, mock_validate, mock_auth, client):
        resp = client.post("/v1/messages/count_tokens",
                           json={"messages": []}, headers={"x-api-key": "bad"})
        assert resp.status_code == 401

    @patch("app.get_token", return_value="fake-token")
    @patch("proxy._api_session")
    @patch("app.auth_enabled", return_value=False)
    def test_upstream_error_is_forwarded(self, mock_auth, mock_session, mock_token, client):
        """A non-200 from SAP must be surfaced with its own status + body."""
        r = MagicMock()
        r.status_code = 400
        r.content = b'{"error":"bad request"}'
        r.headers = {"Content-Type": "application/json"}
        mock_session.post.return_value = r
        resp = client.post("/v1/messages/count_tokens", json={
            "model": "claude-opus-5-5",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 400
        assert b"bad request" in resp.data

    @patch("app.get_token", return_value="fake-token")
    @patch("proxy._api_session")
    @patch("app.auth_enabled", return_value=False)
    def test_releases_deployment(self, mock_auth, mock_session, mock_token, client):
        """count_tokens must release the deployment slot even after success."""
        self._mock_upstream(mock_session)
        client.post("/v1/messages/count_tokens", json={
            "model": "claude-opus-5-5",
            "messages": [{"role": "user", "content": "hi"}],
        })
        for dep_id in proxy._deployment_active:
            assert proxy._deployment_active[dep_id] == 0
