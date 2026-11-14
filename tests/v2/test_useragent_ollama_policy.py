from __future__ import annotations

import asyncio
import contextvars
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.v2.llm_client import (
    LLMClient,
    LLMConfig,
    OllamaAuthorizationError,
    explicit_model_use_requested,
    useragent_ollama_request_active,
    useragent_ollama_authorization,
)
from core.v2.cache import CacheConfig, SemanticCache
from core.v2.user_agent_config import UserAgent, UserConfig
from core.v2.agents import AgentTask


class _Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = ""

    def json(self):
        return self._payload


class _HttpClient:
    def __init__(self, *, post_response=None, get_response=None):
        self.requests = []
        self.post_response = post_response or _Response(
            payload={"message": {"content": "Resposta do modelo"}}
        )
        self.get_response = get_response or _Response(
            payload={"models": [{"name": "qwen3:8b"}]}
        )

    async def post(self, url, **kwargs):
        self.requests.append(("POST", url, kwargs))
        return self.post_response

    async def get(self, url, **kwargs):
        self.requests.append(("GET", url, kwargs))
        return self.get_response


def _ollama_client(http):
    client = LLMClient(LLMConfig(
        base_url="http://127.0.0.1:11434",
        model="qwen3:8b",
    ))
    client._resolved = True
    client._client = http
    return client


def test_explicit_model_request_requires_affirmative_model_use():
    assert explicit_model_use_requested("Utiliza o modelo Ollama para responder.")
    assert explicit_model_use_requested("Usa qwen3 para responder.")
    assert explicit_model_use_requested("Use the model to answer.")
    assert not explicit_model_use_requested("O que é o Ollama?")
    assert not explicit_model_use_requested("Não uses o modelo Ollama.")
    assert not explicit_model_use_requested("Do not use the model.")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "authorized"),
    [
        ("Verifica este ficheiro e pede aprovação para o editar.", False),
        ("Usa Ollama para analisar o ficheiro e pede aprovação para o editar.", True),
    ],
)
async def test_approved_plan_resume_preserves_original_model_consent(
    query, authorized
):
    approval_id = "approval-test"
    plan = SimpleNamespace(query=query)
    orchestrator = SimpleNamespace(
        _approval_resume_lock=asyncio.Lock(),
        _pending_plans={approval_id: plan},
        safety=SimpleNamespace(
            get_approval=lambda _approval_id: {"status": "approved"}
        ),
    )
    client = _ollama_client(_HttpClient())

    async def execute_plan(*_args, **_kwargs):
        return await client.chat([{"role": "user", "content": "continua"}])

    orchestrator._execute_plan = AsyncMock(side_effect=execute_plan)
    from core.v2.orchestrator import OdysseusOrchestrator

    result = await OdysseusOrchestrator.resume_approved_plan(
        orchestrator, approval_id
    )

    assert result.success is authorized
    if authorized:
        assert len(client._client.requests) == 1
    else:
        assert client._client.requests == []


@pytest.mark.asyncio
async def test_unauthorized_useragent_scope_blocks_all_ollama_network_paths():
    http = _HttpClient()
    client = _ollama_client(http)
    messages = [{"role": "user", "content": "Olá"}]

    with useragent_ollama_authorization(False):
        response = await client.chat(messages)
        available = await client.check_availability(force=True)
        tokens = [token async for token in client.chat_stream(messages)]
        with pytest.raises(OllamaAuthorizationError):
            await client.list_models()

    assert response.success is False
    assert "bloqueada" in response.error
    assert available is False
    assert client._available is None
    assert tokens[-1] == "__DONE__"
    assert "bloqueada" in tokens[0]
    assert http.requests == []


@pytest.mark.asyncio
async def test_explicit_useragent_request_makes_one_chat_call_without_model_probe():
    http = _HttpClient()
    client = _ollama_client(http)

    with useragent_ollama_authorization(True):
        response = await client.chat([{"role": "user", "content": "Responde."}])

    assert response.success is True
    assert response.content == "Resposta do modelo"
    assert [request[0] for request in http.requests] == ["POST"]
    assert http.requests[0][1] == "http://127.0.0.1:11434/api/chat"


@pytest.mark.asyncio
async def test_denied_probe_does_not_poison_later_explicit_request():
    http = _HttpClient()
    client = _ollama_client(http)

    with useragent_ollama_authorization(False):
        assert await client.check_availability() is False
    with useragent_ollama_authorization(True):
        response = await client.chat([{"role": "user", "content": "Responde."}])

    assert response.success is True
    assert [request[0] for request in http.requests] == ["POST"]


@pytest.mark.asyncio
async def test_explicit_request_with_unavailable_ollama_does_not_retry_chat():
    http = _HttpClient(get_response=_Response(status_code=503))
    client = _ollama_client(http)

    with useragent_ollama_authorization(True):
        available = await client.check_availability(force=True)

    assert available is False
    assert [request[0] for request in http.requests] == ["GET"]
    assert http.requests[0][1] == "http://127.0.0.1:11434/api/tags"


@pytest.mark.asyncio
async def test_explicit_chat_attempt_on_unavailable_model_is_one_controlled_failure():
    http = _HttpClient(post_response=_Response(status_code=404))
    client = _ollama_client(http)

    with useragent_ollama_authorization(True):
        response = await client.chat([{"role": "user", "content": "Responde."}])

    assert response.success is False
    assert "404" in response.error
    assert [request[0] for request in http.requests] == ["POST"]


@pytest.mark.asyncio
async def test_cache_never_makes_implicit_ollama_embedding_calls():
    http = _HttpClient()
    client = _ollama_client(http)
    cache = SemanticCache(client, CacheConfig(enabled=True))

    with useragent_ollama_authorization(False):
        unauthorized_embedding = await cache._compute_embedding("normal request")
    with useragent_ollama_authorization(True):
        implicit_embedding = await cache._compute_embedding("explicit chat request")

    assert unauthorized_embedding is None
    assert implicit_embedding is None
    assert http.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "context", "expected_requests"),
    [
        ("Responde de forma simples.", {}, []),
        ("Responde de forma simples.", {"model": "qwen3:8b"}, ["POST"]),
        ("Utiliza o modelo Ollama para responder.", {}, ["POST"]),
    ],
)
async def test_useragent_scope_covers_indirect_orchestrator_calls(
    monkeypatch, question, context, expected_requests
):
    from core.v2 import agents, orchestrator, safety

    http = _HttpClient()
    client = _ollama_client(http)

    class FakeOrchestrator:
        async def process(self, _question, **_kwargs):
            llm_response = await client.chat([
                {"role": "user", "content": _question},
            ])
            return SimpleNamespace(
                message=llm_response.content or "Fallback controlado.",
                metadata={"llm_used": llm_response.success},
            )

    monkeypatch.setattr(
        safety,
        "get_safety_manager",
        lambda: SimpleNamespace(sanitize_input=lambda value: value),
    )
    monkeypatch.setattr(
        agents,
        "get_agent_factory",
        lambda: SimpleNamespace(select_worker=lambda _message: None),
    )
    monkeypatch.setattr(orchestrator, "get_orchestrator", lambda **_kwargs: FakeOrchestrator())
    monkeypatch.setattr("core.v2.llm_client.get_llm_client", lambda: client)

    agent = UserAgent(config=UserConfig())
    agent._retrieve_relevant_knowledge = lambda _query: []
    agent._retrieve_relevant_memory = lambda _query: ""

    await agent.process(question, context=context)

    assert [request[0] for request in http.requests] == expected_requests


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("context", "expected_requests"),
    [
        ({}, []),
        ({"model": "qwen3:8b"}, ["POST"]),
    ],
)
async def test_useragent_stream_scope_covers_indirect_orchestrator_calls(
    monkeypatch, context, expected_requests
):
    from core.v2 import agents, orchestrator, safety

    http = _HttpClient()
    client = _ollama_client(http)

    class FakeOrchestrator:
        async def process_stream(self, _question, **_kwargs):
            response = await client.chat([
                {"role": "user", "content": _question},
            ])
            yield response.content or response.error
            yield "__DONE__"

    monkeypatch.setattr(
        safety,
        "get_safety_manager",
        lambda: SimpleNamespace(sanitize_input=lambda value: value),
    )
    monkeypatch.setattr(
        agents,
        "get_agent_factory",
        lambda: SimpleNamespace(select_worker=lambda _message: None),
    )
    monkeypatch.setattr(
        orchestrator,
        "get_orchestrator",
        lambda **_kwargs: FakeOrchestrator(),
    )
    monkeypatch.setattr("core.v2.llm_client.get_llm_client", lambda: client)

    agent = UserAgent(config=UserConfig())
    agent._retrieve_relevant_knowledge = lambda _query: []
    agent._retrieve_relevant_memory = lambda _query: ""

    tokens = [
        token async for token in agent.process_stream(
            "Responde normalmente.", context=context
        )
    ]

    assert tokens[-1] == "__DONE__"
    assert [request[0] for request in http.requests] == expected_requests
    if not expected_requests:
        assert "bloqueada" in tokens[0]
    else:
        assert "Resposta do modelo" in "".join(tokens)


@pytest.mark.asyncio
async def test_useragent_stream_can_close_from_a_different_context(monkeypatch):
    from core.v2 import agents, orchestrator, safety

    class FakeOrchestrator:
        async def process_stream(self, _question, **_kwargs):
            yield "first token"
            await asyncio.Event().wait()

    monkeypatch.setattr(
        safety,
        "get_safety_manager",
        lambda: SimpleNamespace(sanitize_input=lambda value: value),
    )
    monkeypatch.setattr(
        agents,
        "get_agent_factory",
        lambda: SimpleNamespace(select_worker=lambda _message: None),
    )
    monkeypatch.setattr(
        orchestrator,
        "get_orchestrator",
        lambda **_kwargs: FakeOrchestrator(),
    )
    monkeypatch.setattr(
        "core.v2.llm_client.get_llm_client",
        lambda: SimpleNamespace(),
    )

    agent = UserAgent(config=UserConfig())
    agent._retrieve_relevant_knowledge = lambda _query: []
    agent._retrieve_relevant_memory = lambda _query: ""
    stream = agent.process_stream("Responde normalmente.")

    assert await anext(stream) == "first token"
    assert useragent_ollama_request_active() is False
    close_task = asyncio.create_task(
        stream.aclose(),
        context=contextvars.Context(),
    )
    await close_task


@pytest.mark.asyncio
async def test_useragent_agent_execute_blocks_implicit_ollama_probe(monkeypatch):
    from core.v2 import user_agent_config

    http = _HttpClient()
    client = _ollama_client(http)
    monkeypatch.setattr(user_agent_config, "_get_llm_client", lambda: client)

    agent = UserAgent(config=UserConfig())
    agent._learn_from_query = lambda _query, _config: None
    result = await agent.execute(AgentTask(
        id="TASK-001",
        agent_type="useragent",
        query="Olá, responde normalmente.",
    ))

    assert result.success is True
    assert result.details["llm_used"] is False
    assert http.requests == []
