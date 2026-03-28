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
- **Health check endpoint** — `GET /health` for Docker healthcheck and monitoring

## Quick Start

### 1. Prerequisites

- A running Claude model deployment on SAP AI Core
- SAP AI Core service key credentials (client ID, client secret, auth URL, API URL)
- Docker

### 2. Configure

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
| `SAP_DEPLOYMENT_ID` | Your Claude model deployment ID |
| `SAP_RESOURCE_GROUP` | Resource group (default: `default`) |
| `VERBOSE` | Enable detailed request/response logging (default: `false`) |

### 3. Run

```bash
docker compose up -d
```

The proxy listens on port **6655**.

### 4. Use with Claude Code

```bash
# Set the API base URL to point to your proxy
export ANTHROPIC_BASE_URL=http://localhost:6655
export ANTHROPIC_API_KEY=dummy  # any non-empty value works

claude
```

### 5. Use with Anthropic SDK

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

## Build from Source

```bash
docker build -t aicore-proxy .
```

## Limitations

- **No web search** — Anthropic's built-in server-side tools (`web_search`, `text_editor`) are not supported by SAP AI Core / Bedrock. The proxy silently filters them out.
- **Single deployment** — all requests are routed to one deployment ID. To use multiple models, run multiple proxy instances.

## License

MIT
