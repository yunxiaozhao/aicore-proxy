# SAP AI Core Proxy for Anthropic Messages API

A lightweight reverse proxy that translates the standard **Anthropic Messages API** (`/v1/messages`) into SAP AI Core deployment calls, with automatic OAuth2 token management.

This allows tools like **Claude Code**, the **Anthropic Python/TypeScript SDK**, and any client speaking the Anthropic Messages API to work seamlessly with Claude models deployed on SAP AI Core.

```
Client (Claude Code / SDK)
  │  POST /v1/messages
  ▼
[Proxy]  ── OAuth2 ──▶  SAP XSUAA
  │
  │  POST /v2/inference/deployments/{id}/invoke[-with-response-stream]
  ▼
SAP AI Core (Claude model)
```

## Features

- **Streaming support** — uses SAP AI Core's `/invoke-with-response-stream` endpoint and converts Bedrock SSE format to standard Anthropic SSE format (`event:` + `data:` lines)
- **Non-streaming support** — uses `/invoke` endpoint, returns full JSON response
- **Auto OAuth2 token management** — background thread refreshes tokens before expiry with exponential backoff on failure
- **Request adaptation** — automatically handles field differences between Anthropic API and SAP AI Core (removes `model`, `stream`, `context_management`; adds `anthropic_version`)
- **Built-in tool filtering** — strips Anthropic server-side tools (`web_search`, `text_editor`, etc.) that SAP AI Core doesn't support
- **401 auto-retry** — transparently refreshes token and retries on authentication failure
- **Least-connections load balancing** — distributes requests across multiple SAP AI Core deployments, routing each request to the deployment with the fewest active connections; ideal for concurrent subagent workloads. Configure via comma-separated `SAP_DEPLOYMENT_ID`
- **Model-aware routing (optional)** — when separate `SAP_DEPLOYMENT_ID_OPUS` / `_SONNET` / `_HAIKU` env vars (or a per-model dict in the config file) are configured, the proxy inspects the client's `model` string for `opus`/`sonnet`/`haiku` and routes to the matching pool, falling back to the full pool for unknown models. With only the legacy `SAP_DEPLOYMENT_ID` set, behavior is unchanged: the request's `model` field is ignored.
- **API key authentication** — optional client API key validation via env var, config file, or database; disabled when no keys configured (backward compatible). DB-managed keys are stored only as SHA-256 hashes, never plaintext.
- **Config file support** — settings can be provided via `/etc/aicore-proxy/config.json` (volume-mounted), with env vars taking priority; `api_keys` field is hot-reloaded every 60s
- **Usage statistics** — optional per-key request and token usage tracking with SQLite (enable via `ENABLE_STATS=true`); the log stores only the key's hash and display prefix, not the raw key
- **Admin API** — manage API keys and query usage stats via REST endpoints (requires `ENABLE_STATS=true` **and** a configured `ADMIN_TOKEN`)
- **Health check & stats endpoints** — `GET /health` for Docker healthcheck, `GET /stats` for deployment active connections
- **Live models catalog** — `GET /v1/models` and `GET /v1/models/{id}` return the same shape as Anthropic's own Models API (with cursor pagination). The list is built by calling SAP AI Core's `GET /v2/lm/deployments` and reading each RUNNING deployment's `details.resources.backend_details.model` — no hard-coded catalog. Results are cached for 60s (tunable via `MODELS_CACHE_TTL`)
- **Token counting** — `POST /v1/messages/count_tokens` implements Anthropic's standard [Token Count API](https://platform.claude.com/docs/en/api/messages/count_tokens) shape (`{"input_tokens": <n>}`). SAP has no native pre-invocation counter, so each call issues a `max_tokens=1` upstream probe (cost: 1 output token per call)

## Quick Start

### 1. Prerequisites

- A running Claude model deployment on SAP AI Core
- SAP AI Core service key credentials (client ID, client secret, auth URL, API URL)
- Docker

### 2. Get SAP AI Core Credentials

The credentials come from a **service key** of your SAP AI Core instance on BTP:

1. Go to **BTP Cockpit** → your subaccount → **Instances and Subscriptions**
2. Find your **AI Core** service instance → click **Create Service Key** (or view an existing one)
3. The service key JSON contains the values you need:

| Service Key Field | Environment Variable |
|---|---|
| `clientid` | `SAP_CLIENT_ID` |
| `clientsecret` | `SAP_CLIENT_SECRET` |
| `url` | `SAP_AUTH_URL` |
| `serviceurls.AI_API_URL` | `SAP_AI_API_URL` |

### 3. Find Your Deployment ID

The `SAP_DEPLOYMENT_ID` identifies which model deployment to route requests to. You can list all deployments via the SAP AI Core API:

```bash
# Get an OAuth token
TOKEN=$(curl -s -X POST "$SAP_AUTH_URL/oauth/token" \
  -u "$SAP_CLIENT_ID:$SAP_CLIENT_SECRET" \
  -d "grant_type=client_credentials" | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# List all deployments
curl -s "$SAP_AI_API_URL/v2/lm/deployments" \
  -H "Authorization: Bearer $TOKEN" \
  -H "AI-Resource-Group: $SAP_RESOURCE_GROUP" | python3 -m json.tool
```

Look for the deployment with the desired model (e.g. `anthropic--claude-4.6-opus`) and `"status": "RUNNING"`, then copy its `id` field.

### 4. Configure

```bash
cp docker-compose.example.yml docker-compose.yml
```

Edit `docker-compose.yml` and fill in your SAP AI Core credentials:

| Variable | Description |
|---|---|
| `SAP_CLIENT_ID` | OAuth2 client ID from service key |
| `SAP_CLIENT_SECRET` | OAuth2 client secret from service key |
| `SAP_AUTH_URL` | XSUAA token endpoint base URL |
| `SAP_AI_API_URL` | SAP AI Core API base URL |
| `SAP_DEPLOYMENT_ID` | Deployment ID(s), comma-separated for load balancing |
| `SAP_DEPLOYMENT_ID_OPUS` / `_SONNET` / `_HAIKU` | Optional. Per-model deployment pools — when any is set, the proxy routes based on the client's `model` field (opus/sonnet/haiku keyword) instead of round-robining across `SAP_DEPLOYMENT_ID`. Unknown models fall back to `SAP_DEPLOYMENT_ID`. |
| `SAP_RESOURCE_GROUP` | Resource group (default: `default`) |
| `VERBOSE` | Enable detailed request/response logging (default: `false`) |
| `API_KEYS` | Optional: comma-separated API keys for client authentication. Values are hashed in memory and never persisted in plaintext. |
| `ENABLE_STATS` | Optional: enable per-key usage tracking with SQLite (default: `false`) |
| `ADMIN_TOKEN` | Required to use `/admin/*` endpoints. If unset, the admin API is disabled and returns 403. |

All settings can also be provided via a config file (see below).

### Config File (Optional)

Mount a directory to `/etc/aicore-proxy` and create `config.json`:

```bash
mkdir -p ./aicore-proxy
cat > ./aicore-proxy/config.json << 'EOF'
{
  "sap_client_id": "sb-xxx",
  "sap_client_secret": "xxx",
  "sap_auth_url": "https://...",
  "sap_ai_api_url": "https://...",
  "sap_deployment_id": "id1,id2",
  "//": "Or, for model-aware routing, replace the line above with a dict:",
  "//example": "\"sap_deployment_id\": {\"opus\": [\"opus-id-1\"], \"sonnet\": [\"sonnet-id-1\"], \"haiku\": [\"haiku-id-1\"]}",
  "api_keys": ["sk-key1", "sk-key2"],
  "enable_stats": true
}
EOF
```

- **Env vars take priority** over config file values
- The `api_keys` field is **hot-reloaded** every 60s — change keys without restarting
- All fields are optional — only override what you need

### 5. Run

```bash
docker compose up -d
```

The proxy listens on port **6655**.

### 6. Use with Claude Code

```bash
# Set the API base URL to point to your proxy
export ANTHROPIC_BASE_URL=http://localhost:6655
export ANTHROPIC_API_KEY=dummy  # any non-empty value works

claude
```

### 7. Use with Anthropic SDK

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:6655",
    api_key="dummy",  # any non-empty value, auth is handled by the proxy
)

# Non-streaming
message = client.messages.create(
    model="claude-sonnet-4-20250514",  # model field is ignored, deployment ID determines the model
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)

# Streaming
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

## Additional Anthropic API endpoints

Beyond `/v1/messages`, two more standard Anthropic endpoints are proxied so existing clients and SDKs work without special-casing this proxy. Both are auth-gated the same way as `/v1/messages` — if `API_KEYS` is set, missing/invalid keys return `401` with the same `{"type":"error","error":{...}}` shape.

### `GET /v1/models` — list available models

Lists the models this proxy can actually reach, built by calling SAP AI Core's `GET /v2/lm/deployments` and extracting `details.resources.backend_details.model.{name,version}` from each **RUNNING** deployment. Non-running deployments are skipped; duplicates (same model in multiple deployments) collapse to a single entry with the newest `createdAt`.

The response mirrors the [Anthropic Models API](https://platform.claude.com/docs/en/api/models/list): `data[]` entries with `type/id/display_name/created_at/max_input_tokens/max_tokens/capabilities`, plus `first_id`, `last_id`, `has_more`. Supports `?limit=1..1000` (default 20) and `?after_id` / `?before_id` cursor pagination. Use `GET /v1/models/<model_id>` to fetch one entry (404 if not deployed).

```bash
curl http://localhost:6655/v1/models -H "x-api-key: $KEY"
curl http://localhost:6655/v1/models/claude-opus-5-5 -H "x-api-key: $KEY"
```

**Naming**: SAP model names like `anthropic--claude-opus-5-5` are normalized to the public Anthropic IDs (`claude-opus-5-5`) so clients can send them straight into `/v1/messages` without transformation. When SAP reports a pinned `version`, it's appended as a suffix (`claude-haiku-4-5-20251001`); `version: "latest"` is ignored.

**Caching**: results are cached for 60s to avoid hammering the deployments endpoint. Tune with `MODELS_CACHE_TTL=<seconds>`. To force a refetch right after adding or removing a deployment in SAP, call `POST /admin/models/refresh` (see [Admin API](#admin-api)).

**Fallback**: if the SAP fetch fails on a cold cache, the proxy serves a small built-in list of well-known Claude IDs so `/v1/models` still works during an upstream outage. Set `MODELS_STATIC_FALLBACK=false` to make failures visible (returns an empty list instead). `GET /health?verbose=1` includes a `models_cache` block showing `source: "sap"` vs `"fallback"` and the last error, if any.

### `POST /v1/messages/count_tokens` — pre-flight token counting

Standard [Anthropic Token Count API](https://platform.claude.com/docs/en/api/messages/count_tokens) — accepts the same request shape as `/v1/messages` (`messages`, `system`, `tools`, `model`) and returns `{"input_tokens": <n>}` for how many input tokens the prompt would use.

```bash
curl -X POST http://localhost:6655/v1/messages/count_tokens \
     -H "x-api-key: $KEY" -H "Content-Type: application/json" \
     -d '{"model":"claude-opus-5-5","messages":[{"role":"user","content":"Hello, world"}]}'
# {"input_tokens": 12}
```

**Implementation note**: SAP AI Core / Bedrock does not expose a native pre-invocation token counter. The proxy issues a `max_tokens=1` upstream call and reads `usage.input_tokens` from the response, so **each `count_tokens` call costs 1 output token** on the underlying deployment. That's cheap (typically <$0.0001) but not free — usage is logged the same as a regular `/v1/messages` call so it shows up in `/admin/usage`.

Client-supplied `max_tokens`, `stream`, `temperature`, etc. are ignored: count_tokens has no generation semantics. Upstream errors (invalid `model`, tool schema issues, etc.) are surfaced verbatim with their original status code, so a 400 here means the same request would 400 against `/v1/messages` too — handy for validating a prompt before spending on a real call.

## API Key Authentication

When `API_KEYS` is set (env var or config file), clients must provide a valid key:

```bash
export ANTHROPIC_BASE_URL=http://localhost:6655
export ANTHROPIC_API_KEY=sk-key1  # must match a configured key

claude
```

Keys can be provided via `x-api-key` header or `Authorization: Bearer <key>` header.

If no keys are configured, auth is disabled (backward compatible).

## Usage Statistics

Enable with `ENABLE_STATS=true` to track per-key request counts and token usage in SQLite. The DB stores only `sha256(key)` and a short `key_prefix` for display — plaintext keys are never persisted.

### Admin API

Every `/admin/*` route requires `ADMIN_TOKEN` and rejects the request otherwise. Send it either as an `X-Admin-Token` header or as `Authorization: Bearer <ADMIN_TOKEN>`.

```bash
export ADMIN=$ADMIN_TOKEN

# Create a key — the plaintext is returned EXACTLY ONCE, store it now.
curl -X POST http://localhost:6655/admin/keys \
     -H "X-Admin-Token: $ADMIN" -H "Content-Type: application/json" \
     -d '{"name": "dev-team"}'
# {"key":"sk-...","key_hash":"...","key_prefix":"sk-abcdefgh…","name":"dev-team",
#  "warning":"Store this key now — it will not be shown again."}

# List keys — only masked prefixes are returned, never the raw key.
curl -H "X-Admin-Token: $ADMIN" http://localhost:6655/admin/keys

# Revoke a key — pass its key_hash (preferred) or the raw key (hashed locally).
curl -X DELETE -H "X-Admin-Token: $ADMIN" \
     http://localhost:6655/admin/keys/<key_hash-or-raw-key>

# Usage summary — filter by hash or (raw key, hashed server-side); by time range; group by day.
curl -H "X-Admin-Token: $ADMIN" http://localhost:6655/admin/usage
curl -H "X-Admin-Token: $ADMIN" "http://localhost:6655/admin/usage?key_hash=<hash>&days=7"
curl -H "X-Admin-Token: $ADMIN" "http://localhost:6655/admin/usage?key=sk-key1&days=7"
curl -H "X-Admin-Token: $ADMIN" "http://localhost:6655/admin/usage?group_by=day"

# Force-refetch the /v1/models catalog from SAP (bypasses the 60s TTL cache).
# Useful right after adding or removing a deployment in SAP AI Core.
curl -X POST -H "X-Admin-Token: $ADMIN" http://localhost:6655/admin/models/refresh

# Public (no admin token) — deployment stats and health.
curl http://localhost:6655/stats
curl http://localhost:6655/health
```

### Upgrading from a previous version

Older proxy versions stored API keys and per-request logs in plaintext. On first boot, the proxy detects the old schema, rehashes every stored key into `key_hash` + `key_prefix`, drops the plaintext columns, and `VACUUM`s the DB so the old pages are reclaimed. **The plaintext of DB-managed keys is unrecoverable after this migration** — clients continue to work (they still send the same plaintext key; the proxy hashes it on each request), but the admin API can only display prefixes going forward. If you lost track of a key, revoke it via `DELETE /admin/keys/<key_hash>` and issue a new one.

## Build from Source

```bash
docker build -t aicore-proxy .
```

## Limitations

- **No web search** — Anthropic's built-in server-side tools (`web_search`, `text_editor`) are not supported by SAP AI Core / Bedrock. The proxy silently filters them out.
- **No failover** — if a deployment returns an error, the proxy does not automatically retry on another deployment.

## License

MIT
