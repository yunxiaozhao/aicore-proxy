"""
Model catalog served by GET /v1/models and GET /v1/models/{model_id}.

Primary source: live SAP AI Core deployment list. On every request (subject to
a short TTL cache), we call `GET /v2/lm/deployments`, keep only RUNNING ones,
and pull `details.resources.backend_details.model.{name,version}` from each
resource. That gives the exact set of models this proxy can actually reach.

Response shape mirrors Anthropic's Models API
(https://platform.claude.com/docs/en/api/models/list): each entry has
type/id/display_name/created_at/max_input_tokens/max_tokens and a capabilities
object. SAP doesn't return capabilities, so we look them up by model name (or
family, when the exact name isn't in our table).

Fallback: if the SAP fetch fails or returns no Claude deployments, we serve a
small built-in list of well-known Anthropic Claude IDs so the endpoint still
works during upstream outages. Turn the fallback off with
MODELS_STATIC_FALLBACK=false to make failures visible.
"""

import os
import time
import threading
import requests as req_lib


# ---------------------------------------------------------------------------
# Capability presets — SAP AI Core doesn't advertise per-model capabilities,
# so we fill them in from the model name using the same shape Anthropic's own
# /v1/models returns.
# ---------------------------------------------------------------------------
_FULL_CAPS = {
    "batch": {"supported": True},
    "citations": {"supported": True},
    "code_execution": {"supported": True},
    "context_management": {
        "clear_thinking_20251015": {"supported": True},
        "clear_tool_uses_20250919": {"supported": True},
        "compact_20260112": {"supported": True},
        "supported": True,
    },
    "effort": {
        "high": {"supported": True},
        "low": {"supported": True},
        "max": {"supported": True},
        "medium": {"supported": True},
        "supported": True,
        "xhigh": {"supported": True},
    },
    "image_input": {"supported": True},
    "pdf_input": {"supported": True},
    "structured_outputs": {"supported": True},
    "thinking": {
        "supported": True,
        "types": {
            "adaptive": {"supported": True},
            "enabled": {"supported": True},
        },
    },
}

# Haiku family: extended (not adaptive) thinking, effort param unsupported.
_HAIKU_CAPS = {
    **_FULL_CAPS,
    "effort": {
        "high": {"supported": False},
        "low": {"supported": False},
        "max": {"supported": False},
        "medium": {"supported": False},
        "supported": False,
        "xhigh": {"supported": False},
    },
    "thinking": {
        "supported": True,
        "types": {
            "adaptive": {"supported": False},
            "enabled": {"supported": True},
        },
    },
}


# Per-model metadata table used to enrich raw SAP model names into full
# Anthropic-format entries. Keyed by the lowercased SAP `model.name` value
# (which for Bedrock-hosted Claude deployments looks like "anthropic--claude-*"
# or just "claude-*"). Anything not in this table gets sensible defaults.
_MODEL_META = {
    # Current lineup
    "claude-fable-5-1":         {"display": "Claude Fable 5.1",         "max_in": 1_000_000, "max_out": 128_000, "caps": _FULL_CAPS,  "family": "fable"},
    "claude-opus-5-5":          {"display": "Claude Opus 5.5",          "max_in": 1_000_000, "max_out": 128_000, "caps": _FULL_CAPS,  "family": "opus"},
    "claude-sonnet-5-5":        {"display": "Claude Sonnet 5.5",        "max_in": 1_000_000, "max_out": 128_000, "caps": _FULL_CAPS,  "family": "sonnet"},
    "claude-haiku-4-5":         {"display": "Claude Haiku 4.5",         "max_in":   200_000, "max_out":  64_000, "caps": _HAIKU_CAPS, "family": "haiku"},
    "claude-haiku-4-5-20251001":{"display": "Claude Haiku 4.5 (2025-10-01)", "max_in": 200_000, "max_out": 64_000, "caps": _HAIKU_CAPS, "family": "haiku"},
    # Legacy still-available
    "claude-opus-5":            {"display": "Claude Opus 5",            "max_in": 1_000_000, "max_out": 128_000, "caps": _FULL_CAPS,  "family": "opus"},
    "claude-sonnet-5":          {"display": "Claude Sonnet 5",          "max_in": 1_000_000, "max_out": 128_000, "caps": _FULL_CAPS,  "family": "sonnet"},
    "claude-fable-5":           {"display": "Claude Fable 5",           "max_in": 1_000_000, "max_out": 128_000, "caps": _FULL_CAPS,  "family": "fable"},
    "claude-opus-4-8":          {"display": "Claude Opus 4.8",          "max_in":   200_000, "max_out":  64_000, "caps": _FULL_CAPS,  "family": "opus"},
    "claude-opus-4-7":          {"display": "Claude Opus 4.7",          "max_in": 1_000_000, "max_out":  64_000, "caps": _FULL_CAPS,  "family": "opus"},
    "claude-opus-4-6":          {"display": "Claude Opus 4.6",          "max_in":   200_000, "max_out":  64_000, "caps": _FULL_CAPS,  "family": "opus"},
    "claude-opus-4-5":          {"display": "Claude Opus 4.5",          "max_in":   200_000, "max_out":  64_000, "caps": _FULL_CAPS,  "family": "opus"},
    "claude-sonnet-4-6":        {"display": "Claude Sonnet 4.6",        "max_in": 1_000_000, "max_out":  64_000, "caps": _FULL_CAPS,  "family": "sonnet"},
    "claude-sonnet-4-5":        {"display": "Claude Sonnet 4.5",        "max_in": 1_000_000, "max_out":  64_000, "caps": _FULL_CAPS,  "family": "sonnet"},
}


# Static fallback served only when the SAP fetch produces zero entries — either
# the upstream is down at boot time or `MODELS_STATIC_FALLBACK=false` is off
# (in which case an empty list is served instead, making the outage visible).
_STATIC_FALLBACK_IDS = [
    "claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5",
    "claude-haiku-4-5", "claude-haiku-4-5-20251001",
    "claude-opus-5", "claude-sonnet-5", "claude-opus-4-8",
    "claude-sonnet-4-6",
]


# ---------------------------------------------------------------------------
# Config knobs (read from env; sensible defaults)
# ---------------------------------------------------------------------------
_TTL = int(os.environ.get("MODELS_CACHE_TTL", "60"))  # seconds
_STATIC_FALLBACK = os.environ.get("MODELS_STATIC_FALLBACK", "true").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Cache — refreshed at most once per _TTL seconds by whichever request touches
# it after expiry. Not preloaded at boot; first /v1/models hit pays the fetch.
# ---------------------------------------------------------------------------
_cache_lock = threading.Lock()
_cache = {
    "models": [],           # list of Anthropic-format dicts, newest first
    "fetched_at": 0.0,      # unix ts of last successful fetch
    "last_error": None,     # str, most recent fetch failure (for diagnostics)
    "source": "none",       # "sap" | "fallback" | "none"
}


# ---------------------------------------------------------------------------
# Model-name normalization + enrichment
# ---------------------------------------------------------------------------

def _normalize_id(raw_name):
    """Strip Bedrock-style prefixes so IDs match Anthropic's public model IDs.

    SAP AI Core Claude deployments typically expose model names like
    "anthropic--claude-4.6-opus" or "anthropic.claude-3-5-sonnet-20240620";
    Anthropic's own API uses "claude-opus-4-6" / "claude-3-5-sonnet-20240620".
    We lowercase, strip a leading "anthropic--" or "anthropic." prefix, and
    leave the rest untouched — good enough to match the metadata table and
    close enough to what clients expect to see back.
    """
    if not raw_name:
        return ""
    s = str(raw_name).strip().lower()
    for prefix in ("anthropic--", "anthropic."):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s


def _family_of(model_id):
    """Return 'opus'/'sonnet'/'haiku'/'fable' or '' — used for fallback caps."""
    lid = model_id.lower()
    for fam in ("opus", "sonnet", "haiku", "fable"):
        if fam in lid:
            return fam
    return ""


def _enrich(model_id, created_at=None):
    """Turn a normalized model_id into a full Anthropic-format entry.

    Looks up the metadata table first; falls back to family-based defaults so
    a brand-new model SAP exposes still renders a well-formed entry.
    """
    meta = _MODEL_META.get(model_id)
    if meta:
        return {
            "type": "model",
            "id": model_id,
            "display_name": meta["display"],
            "created_at": created_at or "1970-01-01T00:00:00Z",
            "max_input_tokens": meta["max_in"],
            "max_tokens": meta["max_out"],
            "capabilities": meta["caps"],
        }
    fam = _family_of(model_id)
    caps = _HAIKU_CAPS if fam == "haiku" else _FULL_CAPS
    # Pretty-ish display name from the raw id (e.g. "claude-foo-bar" -> "Claude Foo Bar").
    display = " ".join(p.capitalize() for p in model_id.split("-")) if model_id else "Unknown"
    return {
        "type": "model",
        "id": model_id,
        "display_name": display,
        "created_at": created_at or "1970-01-01T00:00:00Z",
        # Conservative defaults for an unknown model — clients that actually
        # care should check the capability object rather than these numbers.
        "max_input_tokens": 200_000,
        "max_tokens": 64_000,
        "capabilities": caps,
    }


# ---------------------------------------------------------------------------
# SAP AI Core deployment fetch
# ---------------------------------------------------------------------------

def _extract_model_from_deployment(dep):
    """Pull a model name (raw string) out of one deployment resource.

    SAP AI Core stores it at details.resources.backend_details.model.name for
    generative-AI-hub Claude deployments. `version` is optional and, when
    present, gets suffixed to the id so the same model at different snapshot
    versions renders as separate entries.
    """
    if not isinstance(dep, dict):
        return None, None, None
    details = dep.get("details") or {}
    resources = details.get("resources") or {}
    backend = resources.get("backend_details") or {}
    model = backend.get("model") or {}
    name = model.get("name")
    version = model.get("version")
    created_at = dep.get("createdAt") or dep.get("startTime")
    return name, version, created_at


def _fetch_from_sap():
    """Call GET /v2/lm/deployments; return a list of Anthropic-format entries.

    Only RUNNING deployments contribute — a pending/failed deployment can't
    serve requests, and listing it would misrepresent what this proxy can
    actually reach. Deduplicates by model id (multiple deployments of the
    same model → one entry). Sorts newest-first by createdAt when available.

    Raises on network / auth failure; caller decides whether to fall back.
    """
    # Imported lazily to break the models <-> proxy <-> config import cycle.
    from proxy import get_token, _api_session
    from config import AI_API_URL, RESOURCE_GROUP

    token = get_token()
    if not token:
        raise RuntimeError("no SAP token available")

    resp = _api_session.get(
        f"{AI_API_URL}/v2/lm/deployments",
        headers={
            "Authorization": f"Bearer {token}",
            "ai-resource-group": RESOURCE_GROUP,
        },
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()
    resources = payload.get("resources", []) if isinstance(payload, dict) else []

    # Group entries by normalized model id; keep the newest createdAt per id
    # (multiple deployments of the same model → one advertised entry).
    seen = {}
    for dep in resources:
        if dep.get("status") != "RUNNING":
            continue
        raw_name, version, created_at = _extract_model_from_deployment(dep)
        if not raw_name:
            continue
        model_id = _normalize_id(raw_name)
        if version and version not in ("latest", ""):
            # Append version as a suffix so pinned snapshots show up separately;
            # skip when the id already ends with the version to avoid dup suffix.
            if not model_id.endswith(version.lower()):
                model_id = f"{model_id}-{version.lower()}"
        if not model_id:
            continue
        existing = seen.get(model_id)
        if existing is None or (created_at and created_at > (existing.get("created_at") or "")):
            seen[model_id] = _enrich(model_id, created_at=created_at)

    # Newest-first (RFC-3339 strings sort lexicographically the right way).
    models = sorted(seen.values(), key=lambda m: m.get("created_at", ""), reverse=True)
    return models


def _refresh_cache(force=False):
    """Refresh the cache from SAP if TTL expired (or `force`). Thread-safe.

    On failure, leaves any prior cached data in place — a transient upstream
    hiccup shouldn't blank out /v1/models between refreshes. `last_error` is
    recorded either way for /v1/test to surface.
    """
    now = time.time()
    with _cache_lock:
        if not force and (now - _cache["fetched_at"]) < _TTL and _cache["source"] != "none":
            return _cache["models"]

    try:
        models = _fetch_from_sap()
    except Exception as e:
        # Do not evict cached data on failure — one bad refresh shouldn't
        # break /v1/models. On the very first attempt, fall through to the
        # static fallback so the endpoint still returns something useful.
        print(f"[proxy] models: SAP deployment fetch failed: {e}", flush=True)
        with _cache_lock:
            _cache["last_error"] = str(e)
            _cache["fetched_at"] = now  # rate-limit failing calls to once per TTL
            if _cache["source"] == "none" and _STATIC_FALLBACK:
                _cache["models"] = [_enrich(mid) for mid in _STATIC_FALLBACK_IDS]
                _cache["source"] = "fallback"
            return _cache["models"]

    if not models and _STATIC_FALLBACK:
        # SAP returned no Claude deployments — better to show the standard
        # Anthropic catalog than nothing at all. Marked as "fallback" so
        # callers can tell it isn't live data.
        with _cache_lock:
            _cache["models"] = [_enrich(mid) for mid in _STATIC_FALLBACK_IDS]
            _cache["fetched_at"] = now
            _cache["last_error"] = None
            _cache["source"] = "fallback"
            return _cache["models"]

    with _cache_lock:
        _cache["models"] = models
        _cache["fetched_at"] = now
        _cache["last_error"] = None
        _cache["source"] = "sap"
        return models


def get_cache_status():
    """Diagnostic snapshot for /v1/test — never contains model bodies."""
    with _cache_lock:
        return {
            "source": _cache["source"],
            "count": len(_cache["models"]),
            "fetched_at": _cache["fetched_at"],
            "last_error": _cache["last_error"],
            "ttl_seconds": _TTL,
        }


# ---------------------------------------------------------------------------
# Public helpers used by app.py
# ---------------------------------------------------------------------------

def get_models():
    """Return the current model list. Triggers a refresh if the cache is cold or stale."""
    return _refresh_cache()


def get_model(model_id):
    """Fetch one model entry by id, or None if not in the current catalog."""
    for m in get_models():
        if m["id"] == model_id:
            return m
    return None


def refresh_now():
    """Force-bypass the TTL and refetch from SAP. Used by admin/debug callers."""
    return _refresh_cache(force=True)


def paginate(items, after_id=None, before_id=None, limit=20):
    """Cursor-paginate a list the way Anthropic's Models API does.

    - `after_id` returns items strictly after that id (forward paging).
    - `before_id` returns items strictly before that id (backward paging).
    - `limit` is capped to [1, 1000] with default 20.
    Returns (page, has_more, first_id, last_id).
    """
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(1000, limit))

    ids = [m["id"] for m in items]
    start, end = 0, len(items)
    if after_id and after_id in ids:
        start = ids.index(after_id) + 1
    elif before_id and before_id in ids:
        end = ids.index(before_id)

    window = items[start:end]
    page = window[:limit]
    has_more = len(window) > limit
    first_id = page[0]["id"] if page else None
    last_id = page[-1]["id"] if page else None
    return page, has_more, first_id, last_id
