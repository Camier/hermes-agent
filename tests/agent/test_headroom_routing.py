from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import httpx
import pytest

from agent import (
    agent_runtime_helpers,
    anthropic_adapter,
    auxiliary_client,
    gemini_native_adapter,
)
from agent import headroom_routing as routing


@pytest.fixture
def enabled_settings() -> routing.HeadroomRoutingSettings:
    return routing.HeadroomRoutingSettings(
        enabled=True,
        url="http://127.0.0.1:8787",
        strict=True,
    )


@pytest.mark.parametrize(
    ("upstream", "original_path"),
    [
        ("https://openrouter.ai/api/v1/chat/completions", "/api/v1/chat/completions"),
        ("https://api.deepseek.com/v1/chat/completions", "/v1/chat/completions"),
        (
            "https://inference-api.nousresearch.com/v1/chat/completions",
            "/v1/chat/completions",
        ),
        ("https://opencode.ai/zen/go/v1/chat/completions", "/zen/go/v1/chat/completions"),
        ("https://ollama.com/v1/chat/completions", "/v1/chat/completions"),
    ],
)
def test_openai_compatible_providers_keep_exact_upstream_path(
    enabled_settings: routing.HeadroomRoutingSettings,
    upstream: str,
    original_path: str,
) -> None:
    request = httpx.Request(
        "POST",
        upstream,
        headers={"Authorization": "Bearer provider-secret"},
    )

    routing.route_httpx_request(request, enabled_settings)

    assert str(request.url) == "http://127.0.0.1:8787/v1/chat/completions"
    assert request.headers["x-headroom-base-url"] == str(
        httpx.URL(upstream).copy_with(path="/", query=None, fragment=None)
    ).rstrip("/")
    assert request.headers["x-headroom-original-path"] == original_path
    assert "x-client" not in request.headers
    assert request.headers["authorization"] == "Bearer provider-secret"
    assert request.headers["host"] == "127.0.0.1:8787"


@pytest.mark.parametrize(
    ("upstream", "base_url"),
    [
        ("https://api.kimi.com/coding/v1/messages", "https://api.kimi.com/coding"),
        ("https://api.minimax.io/anthropic/v1/messages", "https://api.minimax.io/anthropic"),
        ("https://opencode.ai/zen/go/v1/messages", "https://opencode.ai/zen/go"),
    ],
)
def test_anthropic_messages_providers_keep_upstream_base(
    enabled_settings: routing.HeadroomRoutingSettings,
    upstream: str,
    base_url: str,
) -> None:
    request = httpx.Request("POST", upstream, headers={"x-api-key": "provider-secret"})

    routing.route_httpx_request(request, enabled_settings)

    assert str(request.url) == "http://127.0.0.1:8787/v1/messages"
    assert request.headers["x-headroom-base-url"] == base_url
    assert "x-headroom-original-path" not in request.headers
    assert request.headers["x-api-key"] == "provider-secret"


def test_codex_oauth_uses_headroom_native_responses_route(
    enabled_settings: routing.HeadroomRoutingSettings,
) -> None:
    request = httpx.Request(
        "POST",
        "https://chatgpt.com/backend-api/codex/responses?stream=true",
        headers={
            "Authorization": "Bearer codex-oauth-token",
            "ChatGPT-Account-ID": "acct-123",
            "originator": "codex_cli_rs",
        },
    )

    routing.route_httpx_request(request, enabled_settings)

    assert str(request.url) == "http://127.0.0.1:8787/v1/responses?stream=true"
    assert "x-headroom-base-url" not in request.headers
    assert "x-headroom-original-path" not in request.headers
    assert request.headers["authorization"] == "Bearer codex-oauth-token"
    assert request.headers["chatgpt-account-id"] == "acct-123"


def test_openai_responses_keeps_its_explicit_upstream(
    enabled_settings: routing.HeadroomRoutingSettings,
) -> None:
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/responses?stream=true",
        headers={"Authorization": "Bearer provider-secret"},
    )

    routing.route_httpx_request(request, enabled_settings)

    assert str(request.url) == "http://127.0.0.1:8787/v1/responses?stream=true"
    assert request.headers["x-headroom-base-url"] == "https://api.openai.com"
    assert request.headers["x-headroom-original-path"] == "/v1/responses"
    assert request.headers["authorization"] == "Bearer provider-secret"


def test_gemini_native_path_uses_generic_headroom_passthrough(
    enabled_settings: routing.HeadroomRoutingSettings,
) -> None:
    request = httpx.Request(
        "POST",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?alt=sse",
        headers={"x-goog-api-key": "provider-secret"},
    )

    routing.route_httpx_request(request, enabled_settings)

    assert str(request.url) == (
        "http://127.0.0.1:8787/v1beta/models/gemini-3.5-flash:generateContent?alt=sse"
    )
    assert request.headers["x-headroom-base-url"] == (
        "https://generativelanguage.googleapis.com"
    )
    assert request.headers["x-goog-api-key"] == "provider-secret"


def test_same_proxy_origin_is_not_double_routed(
    enabled_settings: routing.HeadroomRoutingSettings,
) -> None:
    request = httpx.Request(
        "POST",
        "http://127.0.0.1:8787/v1/messages",
        headers={"X-Headroom-Base-Url": "https://api.kimi.com/coding"},
    )

    routing.route_httpx_request(request, enabled_settings)

    assert str(request.url) == "http://127.0.0.1:8787/v1/messages"
    assert request.headers["x-headroom-base-url"] == "https://api.kimi.com/coding"


def test_strict_mode_rejects_unrelated_loopback_upstream(
    enabled_settings: routing.HeadroomRoutingSettings,
) -> None:
    request = httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions")

    with pytest.raises(routing.HeadroomRoutingError, match="loopback upstream"):
        routing.route_httpx_request(request, enabled_settings)


def test_relaxed_mode_leaves_unrelated_loopback_direct() -> None:
    settings = routing.HeadroomRoutingSettings(enabled=True, strict=False)
    request = httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions")

    routing.route_httpx_request(request, settings)

    assert str(request.url) == "http://127.0.0.1:11434/v1/chat/completions"


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("http://127.0.0.1:8787/v1", "must not contain a path"),
        ("http://user:secret@127.0.0.1:8787", "bare proxy origin"),
        ("http://127.0.0.1:8787?mode=proxy", "bare proxy origin"),
    ],
)
def test_invalid_enabled_proxy_url_fails_during_config_resolution(
    url: str,
    message: str,
) -> None:
    with pytest.raises(routing.HeadroomRoutingError, match=message):
        routing.headroom_settings_from_config(
            {"headroom": {"enabled": True, "url": url}}
        )


def test_hook_installs_once_on_sync_and_async_clients(
    enabled_settings: routing.HeadroomRoutingSettings,
) -> None:
    sync_client = httpx.Client()
    async_client = httpx.AsyncClient()
    try:
        assert routing.install_headroom_httpx_hook(sync_client, settings=enabled_settings)
        assert routing.install_headroom_httpx_hook(sync_client, settings=enabled_settings)
        assert routing.install_headroom_httpx_hook(
            sync_client,
            settings=routing.HeadroomRoutingSettings(
                enabled=True,
                url="http://127.0.0.1:9999",
                strict=False,
            ),
        )
        assert len(sync_client.event_hooks["request"]) == 1

        assert routing.install_headroom_httpx_hook(async_client, settings=enabled_settings)
        assert len(async_client.event_hooks["request"]) == 1
        assert asyncio.iscoroutinefunction(async_client.event_hooks["request"][0])
    finally:
        sync_client.close()
        asyncio.run(async_client.aclose())


def test_installed_sync_hook_routes_a_prepared_httpx_request(
    enabled_settings: routing.HeadroomRoutingSettings,
) -> None:
    captured: dict = {}

    def _respond(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(_respond))
    try:
        routing.install_headroom_httpx_hook(client, settings=enabled_settings)
        response = client.post(
            "https://openrouter.ai/api/v1/chat/completions?trace=1",
            headers={"Authorization": "Bearer provider-secret"},
        )
    finally:
        client.close()

    assert response.status_code == 200
    request = captured["request"]
    assert str(request.url) == "http://127.0.0.1:8787/v1/chat/completions?trace=1"
    assert request.headers["x-headroom-base-url"] == "https://openrouter.ai"
    assert request.headers["x-headroom-original-path"] == "/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer provider-secret"
    assert "x-client" not in request.headers


def test_installed_async_hook_routes_a_prepared_httpx_request(
    enabled_settings: routing.HeadroomRoutingSettings,
) -> None:
    captured: dict = {}

    async def _exercise() -> httpx.Response:
        def _respond(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            return httpx.Response(200, json={"ok": True})

        async with httpx.AsyncClient(transport=httpx.MockTransport(_respond)) as client:
            routing.install_headroom_httpx_hook(client, settings=enabled_settings)
            return await client.post(
                "https://api.kimi.com/coding/v1/messages",
                headers={"x-api-key": "provider-secret"},
            )

    response = asyncio.run(_exercise())

    assert response.status_code == 200
    request = captured["request"]
    assert str(request.url) == "http://127.0.0.1:8787/v1/messages"
    assert request.headers["x-headroom-base-url"] == "https://api.kimi.com/coding"
    assert request.headers["x-api-key"] == "provider-secret"


def test_openai_client_helper_fails_closed_when_builder_returns_none(
    monkeypatch: pytest.MonkeyPatch,
    enabled_settings: routing.HeadroomRoutingSettings,
) -> None:
    monkeypatch.setattr(routing, "load_headroom_routing_settings", lambda: enabled_settings)
    monkeypatch.setattr(
        "agent.process_bootstrap.build_keepalive_http_client",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(routing.HeadroomRoutingError, match="could not build an httpx client"):
        routing.headroom_openai_http_client_kwargs("https://openrouter.ai/api/v1")


def test_primary_openai_chokepoint_installs_routing_hook(
    monkeypatch: pytest.MonkeyPatch,
    enabled_settings: routing.HeadroomRoutingSettings,
) -> None:
    captured: dict = {}

    def _openai(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(routing, "load_headroom_routing_settings", lambda: enabled_settings)
    monkeypatch.setattr(
        agent_runtime_helpers,
        "_ra",
        lambda: SimpleNamespace(OpenAI=_openai, logger=logging.getLogger("test")),
    )
    http_client = httpx.Client()
    agent = SimpleNamespace(
        provider="openrouter",
        _client_log_context=lambda: "test",
        _build_keepalive_http_client=lambda *args, **kwargs: None,
    )
    try:
        agent_runtime_helpers.create_openai_client(
            agent,
            {
                "api_key": "test-key",
                "base_url": "https://openrouter.ai/api/v1",
                "http_client": http_client,
            },
            reason="test",
            shared=False,
        )
        assert len(captured["http_client"].event_hooks["request"]) == 1
    finally:
        http_client.close()


def test_auxiliary_openai_chokepoint_installs_routing_hook(
    monkeypatch: pytest.MonkeyPatch,
    enabled_settings: routing.HeadroomRoutingSettings,
) -> None:
    captured: dict = {}

    def _openai(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(routing, "load_headroom_routing_settings", lambda: enabled_settings)
    monkeypatch.setattr(auxiliary_client, "OpenAI", _openai)
    http_client = httpx.Client()
    try:
        auxiliary_client._create_openai_client(
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            http_client=http_client,
        )
        assert len(captured["http_client"].event_hooks["request"]) == 1
    finally:
        http_client.close()


def test_anthropic_client_chokepoint_preserves_provider_headers(
    monkeypatch: pytest.MonkeyPatch,
    enabled_settings: routing.HeadroomRoutingSettings,
) -> None:
    captured: dict = {}

    def _anthropic(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(routing, "load_headroom_routing_settings", lambda: enabled_settings)
    monkeypatch.setattr(
        anthropic_adapter,
        "_get_anthropic_sdk",
        lambda: SimpleNamespace(Anthropic=_anthropic),
    )

    anthropic_adapter.build_anthropic_client(
        "test-key",
        "https://api.kimi.com/coding",
    )
    try:
        assert captured["base_url"] == "https://api.kimi.com/coding"
        assert captured["default_headers"]["User-Agent"] == "claude-code/0.1.0"
        assert len(captured["http_client"].event_hooks["request"]) == 1
    finally:
        captured["http_client"].close()


def test_direct_gemini_client_uses_local_headroom_without_env_proxy(
    monkeypatch: pytest.MonkeyPatch,
    enabled_settings: routing.HeadroomRoutingSettings,
) -> None:
    captured: dict = {}
    real_client = httpx.Client

    def _client(**kwargs):
        captured.update(kwargs)
        return real_client(**kwargs)

    monkeypatch.setattr(routing, "load_headroom_routing_settings", lambda: enabled_settings)
    monkeypatch.setattr(gemini_native_adapter.httpx, "Client", _client)

    client = gemini_native_adapter.GeminiNativeClient(api_key="test-key")
    try:
        assert captured["trust_env"] is False
        assert len(client._http.event_hooks["request"]) == 1
    finally:
        client.close()
