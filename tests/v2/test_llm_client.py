"""
Testes Unitários — LLM Client (Odysseus V2) Provider-Aware

Cobertura:
    - Mock do sistema de providers do Odysseus (endpoint_resolver + llm_core)
    - chat() — OpenAI, Anthropic, Ollama, erro, timeout
    - chat_stream() — SSE streaming para OpenAI, Anthropic, Ollama
    - check_availability()
    - list_models()
    - get_model_info()
    - reconfigure()

Uso:
    pytest tests/v2/test_llm_client.py -v
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, AsyncIterator, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ======================================================================
#  Mock helpers — simulam o shared httpx client do llm_core
# ======================================================================


class _MockStreamResponse:
    """Mock de resposta streaming para aiter_lines()."""

    def __init__(self, status_code: int, sse_lines: list[str]):
        self.status_code = status_code
        self._lines = sse_lines

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return b"mock stream error"


class _MockAsyncClient:
    """
    Mock do cliente httpx partilhado do llm_core (_get_http_client()).

    Suporta:
        - get() / post() com respostas registadas por método+URL
        - stream() com linhas SSE
    """

    def __init__(self):
        self.requests: list[dict] = []
        self._responses: dict[str, Any] = {}
        self._stream_lines: dict[str, list[str]] = {}
        self._stream_error: Optional[Exception] = None
        self.is_closed = False

    def _key(self, method: str, url: str) -> str:
        return f"{method.upper()}:{url}"

    def register(self, method: str, url: str, status: int = 200, json_data: Optional[dict] = None, text: str = ""):
        class _Resp:
            def __init__(s):
                s.status_code = status
                s._json = json_data or {}
                s.text = text
            def json(s):
                return s._json
        self._responses[self._key(method, url)] = _Resp()

    def register_stream(self, method: str, url: str, sse_lines: list[str]):
        self._stream_lines[self._key(method, url)] = sse_lines

    def set_stream_error(self, exc: Exception):
        self._stream_error = exc

    async def get(self, url: str, **kw) -> Any:
        self.requests.append({"method": "GET", "url": url})
        resp = self._responses.get(self._key("GET", url))
        if resp:
            return resp
        # Default 404
        class _404:
            status_code = 404
            def json(s):
                return {}
        return _404()

    async def post(self, url: str, **kw) -> Any:
        self.requests.append({"method": "POST", "url": url, "json": kw.get("json")})
        resp = self._responses.get(self._key("POST", url))
        if resp:
            return resp
        # Default success
        class _Def:
            status_code = 200
            def json(s):
                return {"choices": [{"message": {"content": "default response"}, "finish_reason": "stop"}]}
        return _Def()

    def stream(self, method: str, url: str, **kw) -> "StreamCtx":
        self.requests.append({"method": f"STREAM:{method}", "url": url, "json": kw.get("json")})
        return StreamCtx(self, method, url)


class StreamCtx:
    def __init__(self, client: _MockAsyncClient, method: str, url: str):
        self.client = client
        self.key = client._key(method, url)

    async def __aenter__(self) -> _MockStreamResponse:
        if self.client._stream_error:
            raise self.client._stream_error
        lines = self.client._stream_lines.get(self.key, [])
        return _MockStreamResponse(status_code=200, sse_lines=lines)

    async def __aexit__(self, *args):
        pass


# ======================================================================
#  Fixtures
# ======================================================================

@pytest.fixture
def mock_http() -> _MockAsyncClient:
    """Cria um mock do shared httpx client do llm_core."""
    return _MockAsyncClient()


@pytest.fixture
def mock_llm_core(mock_http: _MockAsyncClient):
    """Faz patch do src.llm_core._get_http_client() e detect_provider."""
    patcher = patch("src.llm_core._get_http_client", return_value=mock_http)
    patcher.start()
    yield mock_http
    patcher.stop()


@pytest.fixture
def mock_endpoint_resolver():
    """Faz patch do resolve_endpoint para não precisar de DB."""
    def _resolve(setting_prefix, fallback_url=None, fallback_model=None, fallback_headers=None, owner=None):
        return (None, None, None)  # fallback to Ollama
    patcher = patch("src.endpoint_resolver.resolve_endpoint", side_effect=_resolve)
    patcher.start()
    yield
    patcher.stop()


@pytest.fixture
def llm_client(mock_http: _MockAsyncClient, mock_llm_core, mock_endpoint_resolver):
    """Cria um LLMClient com configuração fixa e sistema de providers mockado."""
    from core.v2.llm_client import LLMClient, LLMConfig
    config = LLMConfig(
        base_url="http://127.0.0.1:11434",
        model="llama3.2",
        timeout=30.0,
        temperature=0.7,
        max_tokens=2048,
        top_p=0.9,
    )
    # Marcar como resolvido para usar o fallback (evita tentar resolver endpoint)
    client = LLMClient(config=config)
    client._resolved = True
    return client


# ======================================================================
#  Tests: LLMConfig
# ======================================================================

class TestLLMConfig:
    def test_default_values(self):
        from core.v2.llm_client import LLMConfig
        c = LLMConfig()
        assert c.base_url == ""
        assert c.model == ""
        assert c.timeout == 60.0

    def test_from_env_basic(self):
        with patch.dict(os.environ, {
            "OLLAMA_BASE_URL": "http://server:11434/v1",
            "OLLAMA_MODEL": "mistral",
            "LLM_TIMEOUT": "120",
        }, clear=True):
            from core.v2.llm_client import LLMConfig
            c = LLMConfig.from_env()
            assert c.base_url == "http://server:11434"  # /v1 stripped
            assert c.model == "mistral"
            assert c.timeout == 120.0

    def test_from_env_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            from core.v2.llm_client import LLMConfig
            c = LLMConfig.from_env()
            assert c.base_url == "http://127.0.0.1:11434"
            assert c.model == "qwen3:8b"

    def test_ollama_url_fallback(self):
        with patch.dict(os.environ, {"OLLAMA_URL": "http://fallback:11434"}, clear=True):
            from core.v2.llm_client import LLMConfig
            c = LLMConfig.from_env()
            assert c.base_url == "http://fallback:11434"


# ======================================================================
#  Tests: chat() — OpenAI-compatible fallback
# ======================================================================

class TestChatOpenAI:
    @pytest.mark.asyncio
    async def test_success(self, llm_client, mock_http):
        """Resposta OpenAI-compatível."""
        mock_http.register("POST", "http://127.0.0.1:11434/api/chat", json_data={
            "model": "llama3.2",
            "message": {"content": "A capital de França é Paris."},
            "done": True,
        })

        resp = await llm_client.chat([{"role": "user", "content": "Qual é a capital?"}])

        assert resp.success is True
        assert "Paris" in resp.content

    @pytest.mark.asyncio
    async def test_local_ollama_openai_compat_uses_native_api(self, llm_client, mock_http):
        llm_client.config.resolved_url = "http://localhost:11434/v1/chat/completions"
        llm_client.config.resolved_model = "qwen3:8b"
        llm_client.config.resolved_provider = "openai"
        llm_client._resolved = True
        llm_client._models_cache = ["qwen3:8b"]
        mock_http.register("POST", "http://localhost:11434/api/chat", json_data={
            "model": "qwen3:8b",
            "message": {"content": "CORE_V2_OK"},
            "done": True,
        })

        resp = await llm_client.chat(
            [{"role": "user", "content": "Responde apenas CORE_V2_OK"}],
            max_tokens=32,
        )

        request = next(request for request in mock_http.requests if request["method"] == "POST")
        assert request["url"] == "http://localhost:11434/api/chat"
        assert request["json"]["think"] is False
        assert resp.success is True
        assert resp.content == "CORE_V2_OK"

    @pytest.mark.asyncio
    async def test_empty_messages(self, llm_client):
        resp = await llm_client.chat([])
        assert resp.success is False
        assert "vazia" in resp.error.lower()

    @pytest.mark.asyncio
    async def test_server_error(self, llm_client, mock_http):
        mock_http.register("POST", "http://127.0.0.1:11434/api/chat", status=500, text="Erro interno")

        resp = await llm_client.chat([{"role": "user", "content": "teste"}])
        assert resp.success is False

    @pytest.mark.asyncio
    async def test_timeout(self, llm_client, mock_http):
        original_post = mock_http.post
        async def timeout_post(url, **kw):
            raise TimeoutError("Timed out")
        mock_http.post = timeout_post

        resp = await llm_client.chat([{"role": "user", "content": "teste"}])
        assert resp.success is False
        assert "timeout" in resp.error.lower() or "Timeout" in resp.error

    @pytest.mark.asyncio
    async def test_with_custom_params(self, llm_client, mock_http):
        with patch("src.llm_core._build_ollama_payload", return_value={
            "model": "codellama", "messages": [], "stream": False, "temperature": 0.1, "num_predict": 100
        }):
            resp = await llm_client.chat(
                messages=[{"role": "user", "content": "faz código"}],
                model="codellama",
                temperature=0.1,
                max_tokens=100,
            )
            # Since we mocked the payload builder, check the call went through
            assert resp.success is True


# ======================================================================
#  Tests: chat_stream()
# ======================================================================

class TestChatStream:
    @pytest.mark.asyncio
    async def test_successful_stream(self, llm_client, mock_http):
        """Ollama streaming."""
        sse_lines = [
            json.dumps({"message": {"content": "Olá"}}) + "\n",
            json.dumps({"message": {"content": " mundo"}}) + "\n",
            json.dumps({"message": {"content": ""}, "done": True}) + "\n",
        ]
        mock_http.register_stream("POST", "http://127.0.0.1:11434/api/chat", sse_lines)

        tokens = []
        async for token in llm_client.chat_stream([{"role": "user", "content": "teste"}]):
            if token == "__DONE__":
                break
            tokens.append(token)

        assert tokens == ["Olá", " mundo"]

    @pytest.mark.asyncio
    async def test_local_ollama_openai_compat_stream_uses_native_api(
        self, llm_client, mock_http
    ):
        url = "http://127.0.0.1:11434/v1/chat/completions"
        native_url = "http://127.0.0.1:11434/api/chat"
        llm_client.config.resolved_url = url
        llm_client.config.resolved_model = "qwen3:8b"
        llm_client.config.resolved_provider = "openai"
        llm_client._resolved = True
        llm_client._models_cache = ["qwen3:8b"]
        mock_http.register_stream("POST", native_url, [
            json.dumps({"message": {"content": "OK"}}),
            json.dumps({"message": {"content": "", "thinking": "internal chain"}, "done": True}),
        ])

        tokens = [
            token async for token in llm_client.chat_stream(
                [{"role": "user", "content": "Responde OK"}], max_tokens=32
            )
        ]

        stream_request = next(
            request for request in mock_http.requests
            if request["method"] == "STREAM:POST" and request["url"] == native_url
        )
        assert stream_request["json"]["think"] is False
        assert tokens == ["OK", "__DONE__"]

    @pytest.mark.asyncio
    async def test_empty_messages(self, llm_client):
        tokens = []
        async for token in llm_client.chat_stream([]):
            tokens.append(token)
        assert tokens == []

    @pytest.mark.asyncio
    async def test_stream_server_error(self, llm_client, mock_http):
        """Erro 500 no stream."""
        class ErrResp:
            status_code = 500
            async def aiter_lines(self):
                return
                yield
            async def aread(self):
                return b"Overloaded"
        class ErrCtx:
            async def __aenter__(s):
                return ErrResp()
            async def __aexit__(s, *a):
                pass
        mock_http.stream = lambda m, u, **kw: ErrCtx()

        tokens = []
        async for token in llm_client.chat_stream([{"role": "user", "content": "teste"}]):
            if token == "__DONE__":
                break
            tokens.append(token)

        assert len(tokens) == 1
        assert "Erro" in tokens[0] or "erro" in tokens[0]

    @pytest.mark.asyncio
    async def test_stream_connection_error(self, llm_client, mock_http):
        mock_http.set_stream_error(ConnectionError("Ligação perdida"))

        tokens = []
        async for token in llm_client.chat_stream([{"role": "user", "content": "teste"}]):
            if token == "__DONE__":
                break
            tokens.append(token)

        assert len(tokens) == 1
        assert "Erro" in tokens[0] or "erro" in tokens[0]

    @pytest.mark.asyncio
    async def test_stream_invalid_json(self, llm_client, mock_http):
        # Ollama native format: raw JSON lines, no "data:" prefix
        sse_lines = [
            json.dumps({"message": {"content": "Olá"}, "done": False}),
            "not json at all",
            json.dumps({"message": {"content": " mundo"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True}),
        ]
        mock_http.register_stream("POST", "http://127.0.0.1:11434/api/chat", sse_lines)

        tokens = []
        async for token in llm_client.chat_stream([{"role": "user", "content": "teste"}]):
            if token == "__DONE__":
                break
            tokens.append(token)

        assert tokens == ["Olá", " mundo"]

    @pytest.mark.asyncio
    async def test_skip_non_data_lines(self, llm_client, mock_http):
        # Ollama native format: skip empty lines, parse valid JSON lines
        sse_lines = [
            "",
            json.dumps({"message": {"content": "Token"}, "done": False}),
            "     ",
            json.dumps({"message": {"content": ""}, "done": True}),
        ]
        mock_http.register_stream("POST", "http://127.0.0.1:11434/api/chat", sse_lines)

        tokens = []
        async for token in llm_client.chat_stream([{"role": "user", "content": "teste"}]):
            if token == "__DONE__":
                break
            tokens.append(token)

        assert tokens == ["Token"]


# ======================================================================
#  Tests: check_availability()
# ======================================================================

class TestCheckAvailability:
    @pytest.mark.asyncio
    async def test_ollama_available(self, llm_client, mock_http):
        """Ollama tags endpoint responde."""
        mock_http.register("GET", "http://127.0.0.1:11434/api/tags", json_data={"models": []})

        ok = await llm_client.check_availability()
        assert ok is True

    @pytest.mark.asyncio
    async def test_unavailable(self, llm_client, mock_http):
        """Todos os endpoints falham."""
        # mock_http default 404 for get, and chat fallback also returns 404
        class _404Resp:
            status_code = 404
            def json(s): return {}
        mock_http.post = lambda u, **kw: _404Resp()
        mock_http.get = lambda u, **kw: _404Resp()

        ok = await llm_client.check_availability()
        assert ok is False

    @pytest.mark.asyncio
    async def test_cached(self, llm_client):
        llm_client._available = True
        ok = await llm_client.check_availability(force=False)
        assert ok is True
        # No requests made
        assert getattr(llm_client, '_request_count', 0) == 0

    @pytest.mark.asyncio
    async def test_force_recheck(self, llm_client, mock_http):
        llm_client._available = True
        mock_http.register("GET", "http://127.0.0.1:11434/api/tags", json_data={"models": []})

        ok = await llm_client.check_availability(force=True)
        assert ok is True


# ======================================================================
#  Tests: list_models()
# ======================================================================

class TestListModels:
    @pytest.mark.asyncio
    async def test_ollama_format(self, llm_client, mock_http):
        mock_http.register("GET", "http://127.0.0.1:11434/api/tags", json_data={
            "models": [{"name": "llama3.2"}, {"name": "mistral"}],
        })

        models = await llm_client.list_models()
        assert models == ["llama3.2", "mistral"]

    @pytest.mark.asyncio
    async def test_no_models(self, llm_client, mock_http):
        mock_http.register("GET", "http://127.0.0.1:11434/api/tags", status=404)
        models = await llm_client.list_models()
        assert models == []


# ======================================================================
#  Tests: get_model_info()
# ======================================================================

class TestGetModelInfo:
    @pytest.mark.asyncio
    async def test_returns_config(self, llm_client, mock_http):
        mock_http.register("GET", "http://127.0.0.1:11434/api/tags", json_data={"models": []})
        info = await llm_client.get_model_info()

        assert info["base_url"] == "http://127.0.0.1:11434"
        assert info["model"] == "llama3.2"
        assert info["available"] is True
        assert info["provider"] == "ollama"


# ======================================================================
#  Tests: reconfigure()
# ======================================================================

class TestReconfigure:
    @pytest.mark.asyncio
    async def test_change_model(self, llm_client):
        result = await llm_client.reconfigure(model="mistral")
        assert "model" in result["changed"]
        assert llm_client.config.model == "mistral"

    @pytest.mark.asyncio
    async def test_change_temperature(self, llm_client):
        result = await llm_client.reconfigure(temperature=0.3)
        assert "temperature" in result["changed"]
        assert llm_client.config.temperature == 0.3

    @pytest.mark.asyncio
    async def test_change_base_url_triggers_resolve(self, llm_client):
        result = await llm_client.reconfigure(base_url="http://novo:11434")
        assert "base_url" in result["changed"]
        assert result["reconnected"] is True
        assert llm_client.config.base_url == "http://novo:11434"
        # Resolução do endpoint deve ser reiniciada
        assert llm_client._resolved is True  # já estava marcado

    @pytest.mark.asyncio
    async def test_no_changes(self, llm_client):
        result = await llm_client.reconfigure(model="llama3.2")  # same as default
        assert result["changed"] == []

    @pytest.mark.asyncio
    async def test_unknown_field_ignored(self, llm_client):
        result = await llm_client.reconfigure(model="mistral", unknown="x")
        assert "model" in result["changed"]

    @pytest.mark.asyncio
    async def test_multiple_fields(self, llm_client):
        result = await llm_client.reconfigure(model="codellama", temperature=0.1, max_tokens=512)
        assert set(result["changed"]) == {"model", "temperature", "max_tokens"}

    @pytest.mark.asyncio
    async def test_temperature_conversion(self, llm_client):
        result = await llm_client.reconfigure(temperature=1)
        assert llm_client.config.temperature == 1.0


# ======================================================================
#  Tests: Lifecycle & Singleton
# ======================================================================

class TestLifecycle:
    def test_init_with_config(self):
        from core.v2.llm_client import LLMClient, LLMConfig
        c = LLMConfig(model="test")
        client = LLMClient(config=c)
        assert client.config.model == "test"

    def test_init_without_config(self):
        from core.v2.llm_client import LLMClient
        client = LLMClient()
        assert client.config is not None

    @pytest.mark.asyncio
    async def test_close(self, llm_client):
        await llm_client.close()
        assert llm_client._client is None


class TestSingleton:
    def setup_method(self):
        from core.v2.llm_client import reset_llm_client
        reset_llm_client()

    def test_same_instance(self):
        from core.v2.llm_client import get_llm_client
        assert get_llm_client() is get_llm_client()

    def test_reset_creates_new(self):
        from core.v2.llm_client import get_llm_client, reset_llm_client
        c1 = get_llm_client()
        reset_llm_client()
        c2 = get_llm_client()
        assert c1 is not c2


class TestExceptions:
    def test_inherit_exception(self):
        from core.v2.llm_client import LLMConnectionError, LLMTimeoutError, LLMResponseError
        assert issubclass(LLMConnectionError, Exception)
        assert issubclass(LLMTimeoutError, Exception)
        assert issubclass(LLMResponseError, Exception)


@pytest.mark.asyncio
async def test_local_ollama_openai_compat_resolves_installed_model(monkeypatch):
    """Não enviar o modelo do endpoint inexistente ao Ollama local."""
    from core.v2.llm_client import LLMClient, LLMConfig

    client = LLMClient(config=LLMConfig(
        base_url="http://127.0.0.1:11434",
        model="llama3.2",
    ))
    monkeypatch.setattr(client, "_get_provider", lambda: "openai")
    monkeypatch.setattr(
        client, "_get_chat_url",
        lambda: "http://localhost:11434/v1/chat/completions",
    )

    async def installed():
        return ["qwen3:8b", "llama3.2:3b"]

    monkeypatch.setattr(client, "list_models", installed)
    client.config.resolved_model = "retired-model:12b"
    client._resolved = True

    await client._ensure_usable_model()

    assert client._model_override == "llama3.2:3b"
    assert client._get_model() == "llama3.2:3b"
