"""Transparent Hermes inference routing through a Headroom proxy.

Hermes keeps resolving the provider, credentials, model, and logical upstream
URL.  A request hook rewrites only the final HTTP request so Headroom can
optimize it and forward it to the upstream Hermes selected.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Any, Mapping


DEFAULT_HEADROOM_URL = "http://127.0.0.1:8787"
_ROUTING_MARKER = "_hermes_headroom_routing"


class HeadroomRoutingError(RuntimeError):
    """Raised when strict Headroom routing cannot be installed safely."""


@dataclass(frozen=True)
class HeadroomRoutingSettings:
    enabled: bool = False
    url: str = DEFAULT_HEADROOM_URL
    strict: bool = True


def _as_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if value is None:
        return default
    return bool(value)


def headroom_settings_from_config(config: Mapping[str, Any] | None) -> HeadroomRoutingSettings:
    raw = config.get("headroom") if isinstance(config, Mapping) else None
    if not isinstance(raw, Mapping):
        raw = {}

    settings = HeadroomRoutingSettings(
        enabled=_as_bool(raw.get("enabled"), default=False),
        url=str(raw.get("url") or DEFAULT_HEADROOM_URL).strip().rstrip("/"),
        strict=_as_bool(raw.get("strict"), default=True),
    )
    if settings.enabled:
        _validate_settings(settings)
    return settings


def load_headroom_routing_settings() -> HeadroomRoutingSettings:
    """Load the profile-aware routing block without mutating config on disk."""
    try:
        from hermes_cli.config import load_config_readonly

        return headroom_settings_from_config(load_config_readonly())
    except HeadroomRoutingError:
        raise
    except Exception:
        # A missing/unreadable optional config must not change historical
        # behavior. When the block is present and enabled, validation above is
        # deliberately strict and its error is allowed through.
        return HeadroomRoutingSettings()


def _validate_settings(settings: HeadroomRoutingSettings) -> None:
    try:
        import httpx

        proxy_url = httpx.URL(settings.url)
    except Exception as exc:
        raise HeadroomRoutingError(
            f"Invalid headroom.url {settings.url!r}; expected an http(s) proxy origin"
        ) from exc

    if proxy_url.scheme not in {"http", "https"} or not proxy_url.host:
        raise HeadroomRoutingError(
            f"Invalid headroom.url {settings.url!r}; expected an http(s) proxy origin"
        )
    if proxy_url.username or proxy_url.password or proxy_url.query or proxy_url.fragment:
        raise HeadroomRoutingError(
            "headroom.url must be a bare proxy origin without credentials, query, or fragment"
        )
    if proxy_url.path not in {"", "/"}:
        raise HeadroomRoutingError(
            "headroom.url must not contain a path; use e.g. http://127.0.0.1:8787"
        )


def _effective_port(url: Any) -> int | None:
    if url.port is not None:
        return int(url.port)
    return 443 if url.scheme == "https" else 80 if url.scheme == "http" else None


def _same_origin(left: Any, right: Any) -> bool:
    return (
        left.scheme.lower(),
        (left.host or "").lower(),
        _effective_port(left),
    ) == (
        right.scheme.lower(),
        (right.host or "").lower(),
        _effective_port(right),
    )


def _is_loopback_host(host: str | None) -> bool:
    normalized = str(host or "").strip().lower().strip("[]")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _url_without_query(url: Any, *, path: str) -> str:
    normalized_path = path or "/"
    return str(url.copy_with(path=normalized_path, query=None, fragment=None)).rstrip("/")


def _is_chatgpt_codex_request(url: Any) -> bool:
    host = str(url.host or "").lower()
    return host == "chatgpt.com" and url.path.startswith("/backend-api/codex/")


def route_httpx_request(request: Any, settings: HeadroomRoutingSettings) -> None:
    """Rewrite one prepared httpx request through Headroom in place."""
    if not settings.enabled:
        return

    import httpx

    upstream = request.url
    proxy = httpx.URL(settings.url)

    if _same_origin(upstream, proxy):
        return

    if upstream.scheme not in {"http", "https"}:
        if settings.strict:
            raise HeadroomRoutingError(
                f"Headroom cannot route non-HTTP inference URL {str(upstream)!r}"
            )
        return

    # Match Headroom's own transparent OpenCode transport: do not silently
    # capture unrelated loopback services. Strict mode makes that exclusion
    # explicit instead of bypassing Headroom unnoticed.
    if _is_loopback_host(upstream.host):
        if settings.strict:
            raise HeadroomRoutingError(
                f"Headroom strict routing refuses direct loopback upstream {str(upstream)!r}"
            )
        return

    upstream_path = upstream.path or "/"
    proxy_path = upstream_path
    base_url: str | None = None
    original_path: str | None = None

    if upstream_path.endswith("/chat/completions"):
        proxy_path = "/v1/chat/completions"
        base_url = _url_without_query(upstream, path="/")
        original_path = upstream_path
    elif upstream_path.endswith("/responses"):
        proxy_path = "/v1/responses"
        if not _is_chatgpt_codex_request(upstream):
            base_url = _url_without_query(upstream, path="/")
            original_path = upstream_path
    elif upstream_path.endswith("/v1/messages"):
        proxy_path = "/v1/messages"
        base_path = upstream_path[: -len("/v1/messages")]
        base_url = _url_without_query(upstream, path=base_path)
    else:
        # Headroom's catch-all passthrough preserves non-chat provider routes
        # (model metadata, Gemini-native paths, etc.). Compression remains the
        # responsibility of Headroom's version-matched route surface.
        base_url = _url_without_query(upstream, path="/")

    if base_url:
        request.headers["X-Headroom-Base-Url"] = base_url
    else:
        request.headers.pop("X-Headroom-Base-Url", None)
    if original_path:
        request.headers["X-Headroom-Original-Path"] = original_path
    else:
        request.headers.pop("X-Headroom-Original-Path", None)

    request.url = proxy.copy_with(
        path=proxy_path,
        query=upstream.query or None,
        fragment=None,
    )
    request.headers["Host"] = request.url.netloc.decode("ascii")


def _request_hook(settings: HeadroomRoutingSettings, *, async_mode: bool):
    if async_mode:
        async def _route_async(request: Any) -> None:
            route_httpx_request(request, settings)

        return _route_async

    def _route_sync(request: Any) -> None:
        route_httpx_request(request, settings)

    return _route_sync


def install_headroom_httpx_hook(
    http_client: Any,
    *,
    settings: HeadroomRoutingSettings | None = None,
) -> bool:
    """Install the transparent request hook on an httpx client once."""
    settings = settings or load_headroom_routing_settings()
    if not settings.enabled:
        return False
    if http_client is None:
        if settings.strict:
            raise HeadroomRoutingError(
                "Headroom routing is enabled but Hermes could not build an httpx client"
            )
        return False

    # A client is bound to the routing settings that were active when it was
    # built. Re-installing with a later config snapshot would leave two hooks
    # fighting over the same prepared request; the caller must rebuild the
    # client to adopt a different proxy URL or strictness policy.
    marker = (settings.url, settings.strict)
    if getattr(http_client, _ROUTING_MARKER, None) is not None:
        return True

    hooks = getattr(http_client, "event_hooks", None)
    if not isinstance(hooks, dict):
        if settings.strict:
            raise HeadroomRoutingError(
                "Headroom routing requires an httpx client with request event hooks"
            )
        return False

    try:
        import httpx

        async_mode = isinstance(http_client, httpx.AsyncClient)
        hooks.setdefault("request", []).append(_request_hook(settings, async_mode=async_mode))
        setattr(http_client, _ROUTING_MARKER, marker)
    except Exception as exc:
        if settings.strict:
            raise HeadroomRoutingError("Failed to install strict Headroom HTTP routing") from exc
        return False
    return True


def ensure_headroom_httpx_client(http_client: Any) -> bool:
    """Ensure an existing SDK http client is routed when the feature is enabled."""
    return install_headroom_httpx_hook(http_client)


def build_headroom_httpx_client(*, timeout: Any) -> Any | None:
    """Build the direct local httpx client needed by the Anthropic SDK."""
    settings = load_headroom_routing_settings()
    if not settings.enabled:
        return None

    import httpx

    client = httpx.Client(timeout=timeout, trust_env=False)
    install_headroom_httpx_hook(client, settings=settings)
    return client


def headroom_openai_http_client_kwargs(
    base_url: str,
    *,
    async_mode: bool = False,
) -> dict[str, Any]:
    """Return an SDK ``http_client`` kwarg only when Headroom is enabled."""
    settings = load_headroom_routing_settings()
    if not settings.enabled:
        return {}

    from agent.process_bootstrap import build_keepalive_http_client

    client = build_keepalive_http_client(base_url, async_mode=async_mode)
    if client is None:
        if settings.strict:
            raise HeadroomRoutingError(
                "Headroom routing is enabled but Hermes could not build an httpx client"
            )
        return {}
    return {"http_client": client}


__all__ = [
    "DEFAULT_HEADROOM_URL",
    "HeadroomRoutingError",
    "HeadroomRoutingSettings",
    "build_headroom_httpx_client",
    "ensure_headroom_httpx_client",
    "headroom_settings_from_config",
    "headroom_openai_http_client_kwargs",
    "install_headroom_httpx_hook",
    "load_headroom_routing_settings",
    "route_httpx_request",
]
