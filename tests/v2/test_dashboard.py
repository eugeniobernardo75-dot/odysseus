"""
Testes Unitários — Dashboard Odysseus V2 (endpoint /api/v2/dashboard)

Cobertura:
    - Mock completo de todos os subsistemas (orquestrador, memória, agentes,
      LLM client, cache semântico, safety manager)
    - Verificação de que a resposta agrega dados de todas as fontes
    - Estrutura esperada: agents, memory, plans, llm, cache, approvals, audit
    - Simulação de falhas individuais em cada subsistema
    - Verificação de tipos e valores nos dados devolvidos
    - Teste das funções auxiliares start/stop_dashboard_broadcast

Uso:
    pytest tests/test_dashboard.py -v
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ── Ajustar path para importar o módulo core ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ======================================================================
#  Fixtures — Mocks dos subsistemas
# ======================================================================


@pytest.fixture
def mock_orchestrator():
    """Mock do orquestrador Odysseus V2."""
    mock = MagicMock()
    mock.list_plans.return_value = [
        {
            "id": "plan-001",
            "query": "Análise de mercado",
            "status": "completed",
            "steps": 3,
            "created_at": "2026-07-27T10:00:00",
        },
        {
            "id": "plan-002",
            "query": "Viabilidade financeira",
            "status": "running",
            "steps": 5,
            "created_at": "2026-07-27T10:05:00",
        },
    ]
    mock.get_audit_report.return_value = {
        "agents_available": 7,
        "plans_executed": 2,
        "principles": ["Segurança", "Eficiência", "Transparência"],
        "system": "Odysseus V2",
        "llm": {"available": True, "model": "llama3.2"},
        "memory": {
            "permanent_entries": 3,
            "conversation_length": 15,
            "pending_approvals": 0,
        },
    }
    return mock


@pytest.fixture
def mock_memory():
    """Mock do MemoryManager."""
    mock = MagicMock()
    mock.recall_all_permanent.return_value = {
        "nome_utilizador": "João",
        "linguagem_preferida": "Python",
        "projeto_atual": "Odysseus",
    }
    mock.get_conversation_context.return_value = [
        {"role": "user", "content": "Olá"},
        {"role": "assistant", "content": "Olá! Como posso ajudar?"},
    ]
    mock.get_pending_approvals.return_value = []
    return mock


@pytest.fixture
def mock_agent_factory():
    """Mock do AgentFactory."""
    mock = MagicMock()
    mock.list_agents.return_value = [
        {"name": "research", "description": "Pesquisa web e recolha de informação"},
        {"name": "code", "description": "Análise e geração de código"},
        {"name": "business", "description": "Análise de negócio e SWOT"},
        {"name": "legal", "description": "Análise jurídica e contratos"},
        {"name": "finance", "description": "Análise financeira e investimentos"},
        {"name": "documentation", "description": "Geração de documentação"},
        {"name": "science", "description": "Análise científica e papers"},
    ]
    return mock


@pytest.fixture
def mock_llm_client():
    """Mock do LLMClient (métodos retornam awaitables)."""
    mock = AsyncMock()
    # CRÍTICO: get_model_info deve ser AsyncMock para retornar uma coroutine.
    # O _collect_data() usa asyncio.run_coroutine_threadsafe() que requer
    # um awaitable, não um valor direto.
    mock.get_model_info = AsyncMock(return_value={
        "base_url": "http://127.0.0.1:11434",
        "model": "llama3.2",
        "timeout": 60.0,
        "temperature": 0.7,
        "max_tokens": 2048,
        "top_p": 0.9,
        "available": True,
    })
    mock.config = MagicMock()
    mock.config.model = "llama3.2"
    mock.config.base_url = "http://127.0.0.1:11434"
    return mock


@pytest.fixture
def mock_cache():
    """Mock do SemanticCache."""
    mock = MagicMock()
    mock.get_stats.return_value = {
        "enabled": True,
        "threshold": 0.85,
        "entries": 42,
        "hits": 150,
        "misses": 80,
        "hit_rate": 0.6522,
        "evictions": 3,
        "ttl_seconds": 3600,
        "max_entries": 500,
    }
    return mock


@pytest.fixture
def mock_safety():
    """Mock do SafetyManager."""
    mock = MagicMock()
    mock.get_pending.return_value = [
        {
            "approval_id": "appr-001",
            "message": "Executar comando: rm -rf /",
            "risk": "high",
            "status": "pending",
            "created_at": "2026-07-27T09:00:00",
        },
        {
            "approval_id": "appr-002",
            "message": "Instalar pacote: requests",
            "risk": "low",
            "status": "pending",
            "created_at": "2026-07-27T09:30:00",
        },
    ]
    mock.get_all.return_value = [
        {
            "approval_id": "appr-001",
            "message": "Executar comando: rm -rf /",
            "risk": "high",
            "status": "pending",
            "created_at": "2026-07-27T09:00:00",
        },
        {
            "approval_id": "appr-002",
            "message": "Instalar pacote: requests",
            "risk": "low",
            "status": "pending",
            "created_at": "2026-07-27T09:30:00",
        },
        {
            "approval_id": "appr-003",
            "message": "Aceder à base de dados de produção",
            "risk": "high",
            "status": "approved",
            "created_at": "2026-07-27T08:00:00",
        },
    ]
    return mock


def _sync_run_coroutine(coro):
    """
    Executa uma coroutine 'sincronamente' usando coro.send(None).

    Útil para simular loop.run_until_complete() sem precisar
    de um event loop real. A coroutine deve completar numa
    única iteração (como as que vêm de AsyncMock).
    """
    import asyncio
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value
    # Se ainda não terminou, continua iterando
    while True:
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value


@pytest.fixture
def dashboard_app(
    mock_orchestrator,
    mock_memory,
    mock_agent_factory,
    mock_llm_client,
    mock_cache,
    mock_safety,
):
    """
    Cria uma app FastAPI de teste com o router do dashboard,
    e faz patch de todos os singletons dos subsistemas.

    NOTA: O _collect_data() usa asyncio.run_coroutine_threadsafe()
    para obter dados do LLM client. Como estamos a testar com
    TestClient (que corre o handler no event loop), isto causa
    um deadlock: o event loop não consegue processar a coroutine
    agendada porque está bloqueado no fut.result().

    Para contornar isto, fazemos patch do asyncio.get_event_loop()
    para devolver um loop mock onde is_running() retorna False.
    Assim o código usa run_until_complete() em vez de
    run_coroutine_threadsafe(), e o nosso _sync_run_coroutine()
    executa a coroutine sem precisar de um event loop real.
    """
    import asyncio as _asyncio

    patches = [
        patch("core.v2.orchestrator.get_orchestrator", return_value=mock_orchestrator),
        patch("core.v2.memory.get_memory_manager", return_value=mock_memory),
        patch("core.v2.agents.get_agent_factory", return_value=mock_agent_factory),
        patch("core.v2.llm_client.get_llm_client", return_value=mock_llm_client),
        patch("core.v2.cache.get_cache", return_value=mock_cache),
        patch("core.v2.safety.get_safety_manager", return_value=mock_safety),
    ]

    for p in patches:
        p.start()

    from core.v2.dashboard_routes import setup_dashboard_routes

    app = FastAPI()
    app.include_router(setup_dashboard_routes())

    # Patch do event loop para evitar deadlock com run_coroutine_threadsafe
    # O mock loop faz is_running()=False, forçando o uso de run_until_complete,
    # que por sua vez usa _sync_run_coroutine() para executar a coroutine.
    mock_loop = MagicMock(spec=_asyncio.AbstractEventLoop)
    mock_loop.is_running.return_value = False
    mock_loop.run_until_complete.side_effect = _sync_run_coroutine
    mock_loop.create_future.side_effect = _asyncio.Future

    loop_patcher = patch("asyncio.get_event_loop", return_value=mock_loop)
    loop_patcher.start()

    yield app

    loop_patcher.stop()

    for p in patches:
        p.stop()


@pytest.fixture
def client(dashboard_app):
    """TestClient apontando para a app de teste."""
    return TestClient(dashboard_app)


# ======================================================================
#  Testes: Estrutura Geral
# ======================================================================


class TestDashboardGeneralStructure:
    """Testes de alto nível do endpoint /api/v2/dashboard."""

    def test_dashboard_returns_200(self, client):
        """Endpoint responde com 200 OK."""
        response = client.get("/api/v2/dashboard")
        assert response.status_code == 200

    def test_dashboard_has_all_top_level_keys(self, client):
        """Payload contém todas as secções esperadas."""
        response = client.get("/api/v2/dashboard")
        data = response.json()

        assert "agents" in data
        assert "memory" in data
        assert "plans" in data
        assert "llm" in data
        assert "cache" in data
        assert "approvals" in data
        assert "audit" in data
        assert "timestamp" in data
        assert "success" in data

    def test_dashboard_success_flag(self, client):
        """Flag success deve ser True."""
        response = client.get("/api/v2/dashboard")
        data = response.json()
        assert data["success"] is True

    def test_dashboard_timestamp_format(self, client):
        """Timestamp deve ser uma string ISO 8601 válida."""
        response = client.get("/api/v2/dashboard")
        data = response.json()
        assert "timestamp" in data
        # Deve terminar com Z ou ter timezone
        ts = data["timestamp"]
        assert "T" in ts
        # Deve ser parseável como datetime
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None or ts.endswith("Z")

    def test_dashboard_content_type(self, client):
        """Resposta deve ser application/json."""
        response = client.get("/api/v2/dashboard")
        assert "application/json" in response.headers.get("content-type", "")


# ======================================================================
#  Testes: Secção de Agentes
# ======================================================================


class TestDashboardAgents:
    """Testes da secção 'agents' do dashboard."""

    def test_agents_returns_list(self, client):
        """agents.list deve ser uma lista."""
        response = client.get("/api/v2/dashboard")
        agents = response.json()["agents"]
        assert isinstance(agents, dict)
        assert "list" in agents
        assert isinstance(agents["list"], list)
        assert "total" in agents

    def test_agents_has_seven_items(self, client):
        """Devem existir 7 agentes especializados."""
        response = client.get("/api/v2/dashboard")
        agents = response.json()["agents"]
        assert agents["total"] == 7
        assert len(agents["list"]) == 7

    def test_agents_have_name_and_description(self, client):
        """Cada agente deve ter 'name' e 'description'."""
        response = client.get("/api/v2/dashboard")
        agents = response.json()["agents"]
        for agent in agents["list"]:
            assert "name" in agent
            assert "description" in agent
            assert isinstance(agent["name"], str)
            assert isinstance(agent["description"], str)

    def test_agents_research_first(self, client):
        """O primeiro agente deve ser o ResearchAgent."""
        response = client.get("/api/v2/dashboard")
        agents = response.json()["agents"]
        assert agents["list"][0]["name"] == "research"

    def test_agents_science_last(self, client):
        """O último agente deve ser o ScienceAgent."""
        response = client.get("/api/v2/dashboard")
        agents = response.json()["agents"]
        assert agents["list"][-1]["name"] == "science"


# ======================================================================
#  Testes: Secção de Memória
# ======================================================================


class TestDashboardMemory:
    """Testes da secção 'memory' do dashboard."""

    def test_memory_has_permanent(self, client):
        """Memória permanente deve estar presente."""
        response = client.get("/api/v2/dashboard")
        memory = response.json()["memory"]
        assert "permanent" in memory
        assert isinstance(memory["permanent"], dict)

    def test_memory_permanent_entries_count(self, client):
        """Contagem de entradas permanentes deve ser 3."""
        response = client.get("/api/v2/dashboard")
        memory = response.json()["memory"]
        assert memory["permanent_entries"] == 3

    def test_memory_has_specific_values(self, client):
        """Valores específicos da memória devem estar corretos."""
        response = client.get("/api/v2/dashboard")
        memory = response.json()["memory"]
        assert memory["permanent"]["nome_utilizador"] == "João"
        assert memory["permanent"]["linguagem_preferida"] == "Python"
        assert memory["permanent"]["projeto_atual"] == "Odysseus"

    def test_memory_conversation_length(self, client):
        """Comprimento do contexto de conversação deve ser 2."""
        response = client.get("/api/v2/dashboard")
        memory = response.json()["memory"]
        assert memory["conversation_length"] == 2

    def test_memory_pending_storage(self, client):
        """Pending storage deve ser uma lista."""
        response = client.get("/api/v2/dashboard")
        memory = response.json()["memory"]
        assert "pending_storage" in memory
        assert isinstance(memory["pending_storage"], list)
        assert len(memory["pending_storage"]) == 0


# ======================================================================
#  Testes: Secção de Planos
# ======================================================================


class TestDashboardPlans:
    """Testes da secção 'plans' do dashboard."""

    def test_plans_has_total_and_recent(self, client):
        """Secção plans deve ter 'total' e 'recent'."""
        response = client.get("/api/v2/dashboard")
        plans = response.json()["plans"]
        assert "total" in plans
        assert "recent" in plans
        assert isinstance(plans["total"], int)
        assert isinstance(plans["recent"], list)

    def test_plans_total_count(self, client):
        """Total de planos deve ser 2."""
        response = client.get("/api/v2/dashboard")
        plans = response.json()["plans"]
        assert plans["total"] == 2

    def test_plans_recent_len(self, client):
        """Lista recent deve ter 2 planos."""
        response = client.get("/api/v2/dashboard")
        plans = response.json()["plans"]
        assert len(plans["recent"]) == 2

    def test_plans_have_required_fields(self, client):
        """Cada plano deve ter os campos essenciais."""
        response = client.get("/api/v2/dashboard")
        plans = response.json()["plans"]
        for plan in plans["recent"]:
            assert "id" in plan
            assert "query" in plan
            assert "status" in plan
            assert "steps" in plan

    def test_plans_different_statuses(self, client):
        """Planos devem ter status diferentes."""
        response = client.get("/api/v2/dashboard")
        plans = response.json()["plans"]["recent"]
        statuses = [p["status"] for p in plans]
        assert "completed" in statuses
        assert "running" in statuses


# ======================================================================
#  Testes: Secção LLM & Cache
# ======================================================================


class TestDashboardLLM:
    """Testes da secção 'llm' do dashboard."""

    def test_llm_has_config_fields(self, client):
        """LLM deve ter campos de configuração."""
        response = client.get("/api/v2/dashboard")
        llm = response.json()["llm"]
        assert "base_url" in llm
        assert "model" in llm
        assert "timeout" in llm
        assert "temperature" in llm
        assert "available" in llm

    def test_llm_config_values(self, client):
        """Valores de configuração do LLM devem estar corretos."""
        response = client.get("/api/v2/dashboard")
        llm = response.json()["llm"]
        assert llm["model"] == "llama3.2"
        assert llm["base_url"] == "http://127.0.0.1:11434"
        assert llm["timeout"] == 60.0
        assert llm["temperature"] == 0.7
        assert llm["max_tokens"] == 2048
        assert llm["top_p"] == 0.9

    def test_llm_available(self, client):
        """LLM deve estar disponível (mock)."""
        response = client.get("/api/v2/dashboard")
        llm = response.json()["llm"]
        assert llm["available"] is True


class TestDashboardCache:
    """Testes da secção 'cache' do dashboard."""

    def test_cache_has_stats_fields(self, client):
        """Cache deve ter os campos de estatísticas."""
        response = client.get("/api/v2/dashboard")
        cache = response.json()["cache"]
        assert "enabled" in cache
        assert "threshold" in cache
        assert "entries" in cache
        assert "hits" in cache
        assert "misses" in cache
        assert "hit_rate" in cache

    def test_cache_stats_values(self, client):
        """Valores das estatísticas do cache devem estar corretos."""
        response = client.get("/api/v2/dashboard")
        cache = response.json()["cache"]
        assert cache["enabled"] is True
        assert cache["threshold"] == 0.85
        assert cache["entries"] == 42
        assert cache["hits"] == 150
        assert cache["misses"] == 80
        assert cache["hit_rate"] == 0.6522
        assert cache["evictions"] == 3

    def test_cache_hit_rate_is_float(self, client):
        """Hit rate deve ser float."""
        response = client.get("/api/v2/dashboard")
        cache = response.json()["cache"]
        assert isinstance(cache["hit_rate"], float)


# ======================================================================
#  Testes: Secção de Aprovações
# ======================================================================


class TestDashboardApprovals:
    """Testes da secção 'approvals' do dashboard."""

    def test_approvals_has_counts_and_list(self, client):
        """Approvals deve ter pending_count, pending e total_count."""
        response = client.get("/api/v2/dashboard")
        approvals = response.json()["approvals"]
        assert "pending_count" in approvals
        assert "pending" in approvals
        assert "total_count" in approvals

    def test_approvals_pending_count(self, client):
        """Devem existir 2 aprovações pendentes."""
        response = client.get("/api/v2/dashboard")
        approvals = response.json()["approvals"]
        assert approvals["pending_count"] == 2

    def test_approvals_total_count(self, client):
        """Total de aprovações deve ser 3."""
        response = client.get("/api/v2/dashboard")
        approvals = response.json()["approvals"]
        assert approvals["total_count"] == 3

    def test_approvals_pending_limited_to_10(self, client):
        """Lista pending deve estar limitada a 10 itens."""
        response = client.get("/api/v2/dashboard")
        approvals = response.json()["approvals"]
        assert len(approvals["pending"]) <= 10

    def test_approvals_have_required_fields(self, client):
        """Cada approval deve ter os campos essenciais."""
        response = client.get("/api/v2/dashboard")
        approvals = response.json()["approvals"]
        for a in approvals["pending"]:
            assert "approval_id" in a
            assert "message" in a
            assert "risk" in a
            assert "status" in a

    def test_approvals_different_risks(self, client):
        """Aprovações pendentes devem ter diferentes níveis de risco."""
        response = client.get("/api/v2/dashboard")
        approvals = response.json()["approvals"]["pending"]
        risks = {a["risk"] for a in approvals}
        assert "high" in risks
        assert "low" in risks


# ======================================================================
#  Testes: Secção de Auditoria
# ======================================================================


class TestDashboardAudit:
    """Testes da secção 'audit' do dashboard."""

    def test_audit_has_agents_available(self, client):
        """Auditoria deve incluir contagem de agentes disponíveis."""
        response = client.get("/api/v2/dashboard")
        audit = response.json()["audit"]
        assert audit["agents_available"] == 7

    def test_audit_has_plans_executed(self, client):
        """Auditoria deve incluir planos executados."""
        response = client.get("/api/v2/dashboard")
        audit = response.json()["audit"]
        assert audit["plans_executed"] == 2

    def test_audit_has_principles(self, client):
        """Auditoria deve incluir princípios."""
        response = client.get("/api/v2/dashboard")
        audit = response.json()["audit"]
        assert isinstance(audit["principles"], list)
        assert len(audit["principles"]) == 3

    def test_audit_has_system_name(self, client):
        """Auditoria deve incluir nome do sistema."""
        response = client.get("/api/v2/dashboard")
        audit = response.json()["audit"]
        assert audit["system"] == "Odysseus V2"

    def test_audit_has_llm_info(self, client):
        """Auditoria deve incluir informação do LLM."""
        response = client.get("/api/v2/dashboard")
        audit = response.json()["audit"]
        assert "llm" in audit
        assert audit["llm"]["available"] is True

    def test_audit_has_memory_info(self, client):
        """Auditoria deve incluir informação da memória."""
        response = client.get("/api/v2/dashboard")
        audit = response.json()["audit"]
        assert "memory" in audit
        assert audit["memory"]["permanent_entries"] == 3
        assert audit["memory"]["conversation_length"] == 15


# ======================================================================
#  Testes: Tratamento de Erros (subsistemas a falhar)
# ======================================================================


class TestDashboardErrorHandling:
    """Testes de resiliência quando subsistemas individuais falham."""

    def test_orchestrator_failure(self, client, mock_orchestrator):
        """Falha do orquestrador → plans com erro, audit com erro."""
        mock_orchestrator.list_plans.side_effect = RuntimeError("Orquestrador em baixo")
        mock_orchestrator.get_audit_report.side_effect = RuntimeError("Auditoria indisponível")

        response = client.get("/api/v2/dashboard")
        data = response.json()

        # Plans deve ter erro
        assert "error" in data["plans"]
        assert "Orquestrador" in data["plans"]["error"]

        # Audit deve ter erro
        assert "error" in data["audit"]
        assert "Auditoria" in data["audit"]["error"]

        # Mas outras secções devem funcionar
        assert "error" not in data.get("agents", {})
        assert "error" not in data.get("memory", {})

    def test_memory_failure(self, client, mock_memory):
        """Falha da memória → memory com erro."""
        mock_memory.recall_all_permanent.side_effect = ValueError("Erro de memória")

        response = client.get("/api/v2/dashboard")
        data = response.json()

        assert "error" in data["memory"]
        assert "memória" in data["memory"]["error"].lower()

        # Outras secções intactas
        assert "error" not in data.get("agents", {})
        assert "error" not in data.get("plans", {})

    def test_llm_failure(self, client, mock_llm_client):
        """Falha do LLM client → llm com erro."""

        async def raise_error():
            raise ConnectionError("LLM servidor offline")

        mock_llm_client.get_model_info.side_effect = raise_error

        response = client.get("/api/v2/dashboard")
        data = response.json()

        assert "error" in data["llm"]
        assert "LLM" in data["llm"]["error"] or "offline" in data["llm"]["error"].lower()

    def test_cache_failure(self, client, mock_cache):
        """Falha do cache → cache com erro."""
        mock_cache.get_stats.side_effect = RuntimeError("Cache corrompido")

        response = client.get("/api/v2/dashboard")
        data = response.json()

        assert "error" in data["cache"]
        assert "Cache" in data["cache"]["error"] or "cache" in data["cache"]["error"].lower()

    def test_safety_failure(self, client, mock_safety):
        """Falha do safety manager → approvals com erro."""
        mock_safety.get_pending.side_effect = RuntimeError("Safety manager offline")

        response = client.get("/api/v2/dashboard")
        data = response.json()

        assert "error" in data["approvals"]
        assert "Safety" in data["approvals"]["error"] or "safety" in data["approvals"]["error"].lower()

    def test_multiple_subsystems_fail(self, client, mock_llm_client, mock_cache, mock_memory):
        """Múltiplos subsistemas falham em simultâneo."""
        async def llm_error():
            raise TimeoutError("LLM timeout")

        mock_llm_client.get_model_info.side_effect = llm_error
        mock_cache.get_stats.side_effect = RuntimeError("Cache em baixo")
        mock_memory.recall_all_permanent.side_effect = ValueError("Memória corrompida")

        response = client.get("/api/v2/dashboard")
        data = response.json()

        # Secções com erro
        assert "error" in data["llm"]
        assert "error" in data["cache"]
        assert "error" in data["memory"]

        # Secções sem erro
        assert "error" not in data.get("agents", {})
        assert "error" not in data.get("plans", {})

        # Timestamp e success ainda funcionam
        assert data["success"] is True
        assert "timestamp" in data


# ======================================================================
#  Testes: Dados Vazios
# ======================================================================


class TestDashboardEmptyData:
    """Testes com dados vazios (listas vazias, dicts vazios)."""

    def test_empty_agents(self, client, mock_agent_factory):
        """Nenhum agente → lista vazia, total = 0."""
        mock_agent_factory.list_agents.return_value = []

        response = client.get("/api/v2/dashboard")
        agents = response.json()["agents"]
        assert agents["total"] == 0
        assert agents["list"] == []

    def test_empty_memory(self, client, mock_memory):
        """Nenhuma memória → dict vazio."""
        mock_memory.recall_all_permanent.return_value = {}

        response = client.get("/api/v2/dashboard")
        memory = response.json()["memory"]
        assert memory["permanent_entries"] == 0
        assert memory["permanent"] == {}

    def test_empty_plans(self, client, mock_orchestrator):
        """Nenhum plano → lista vazia, total = 0."""
        mock_orchestrator.list_plans.return_value = []

        response = client.get("/api/v2/dashboard")
        plans = response.json()["plans"]
        assert plans["total"] == 0
        assert plans["recent"] == []

    def test_empty_approvals(self, client, mock_safety):
        """Nenhuma aprovação → pending vazio, total = 0."""
        mock_safety.get_pending.return_value = []
        mock_safety.get_all.return_value = []

        response = client.get("/api/v2/dashboard")
        approvals = response.json()["approvals"]
        assert approvals["pending_count"] == 0
        assert approvals["pending"] == []
        assert approvals["total_count"] == 0


# ======================================================================
#  Testes: Funções Auxiliares (start/stop broadcast)
# ======================================================================


class TestDashboardBroadcastHelpers:
    """Testes das funções start_dashboard_broadcast e stop_dashboard_broadcast."""

    @pytest.mark.asyncio
    async def test_start_dashboard_broadcast(self):
        """start_dashboard_broadcast() deve chamar start_broadcast_loop()."""
        from core.v2.dashboard_routes import start_dashboard_broadcast

        # A função importa start_broadcast_loop de .ws_broadcaster,
        # por isso o patch deve ser nesse módulo, não em dashboard_routes.
        with patch("core.v2.ws_broadcaster.start_broadcast_loop", new_callable=AsyncMock) as mock_start:
            await start_dashboard_broadcast()
            mock_start.assert_awaited_once_with(interval=5.0)

    @pytest.mark.asyncio
    async def test_stop_dashboard_broadcast(self):
        """stop_dashboard_broadcast() deve chamar stop_broadcast_loop()."""
        from core.v2.dashboard_routes import stop_dashboard_broadcast

        with patch("core.v2.ws_broadcaster.stop_broadcast_loop", new_callable=AsyncMock) as mock_stop:
            await stop_dashboard_broadcast()
            mock_stop.assert_awaited_once()


# ======================================================================
#  Testes: Content-Type e Cabeçalhos
# ======================================================================


class TestDashboardHeaders:
    """Testes de cabeçalhos HTTP e content-type."""

    def test_json_content_type(self, client):
        """Resposta deve ter Content-Type: application/json."""
        response = client.get("/api/v2/dashboard")
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type

    def test_response_is_serializable(self, client):
        """Resposta deve ser JSON serializável (parse sem erro)."""
        response = client.get("/api/v2/dashboard")
        import json
        # Isto lança exceção se não for JSON válido
        data = json.loads(response.text)
        assert isinstance(data, dict)

    def test_no_html_in_response(self, client):
        """Resposta não deve conter HTML (apenas JSON)."""
        response = client.get("/api/v2/dashboard")
        text = response.text.strip()
        assert text.startswith("{")
        assert text.endswith("}")
