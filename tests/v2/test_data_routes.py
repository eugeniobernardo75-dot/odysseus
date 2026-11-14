"""
Testes para as Rotas de Dados — Odysseus V2

Cobre os endpoints:
    GET    /v2/status              — Estado completo do sistema
    GET    /v2/agents              — Listar agentes
    GET    /v2/skills              — Listar skills
    GET    /v2/memory              — Listar memórias
    GET    /v2/memory/{key}        — Obter memória específica
    POST   /v2/memory              — Criar/atualizar memória
    DELETE /v2/memory/{key}        — Remover memória
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Adiciona o diretório raiz do Odysseus ao path
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


# ======================================================================
#  Test: setup_data_routes — router structure
# ======================================================================

class TestSetup:
    """Testa a estrutura base do router."""

    def test_router_created(self):
        """O router é criado com prefixo e tags corretos."""
        from core.v2.data_routes import setup_data_routes
        router = setup_data_routes()
        assert router.prefix == "/v2"
        assert "odysseus-data" in router.tags

    def test_all_routes_registered(self):
        """Todas as 8 rotas estão registadas."""
        from core.v2.data_routes import setup_data_routes
        router = setup_data_routes()
        routes = [(r.path, list(r.methods)[0] if r.methods else "ANY") for r in router.routes]
        route_map = {f"{method} {path}" for path, method in routes}

        expected = {
            "GET /v2/status",
            "GET /v2/settings",
            "GET /v2/agents",
            "GET /v2/skills",
            "GET /v2/memory",
            "POST /v2/memory",
            "GET /v2/memory/{key:path}",
            "DELETE /v2/memory/{key:path}",
        }
        for exp in expected:
            assert exp in route_map, f"Rota {exp} não encontrada"

    def test_exports_in_init(self):
        """setup_data_routes é exportado pelo __init__."""
        from core.v2 import setup_data_routes
        assert callable(setup_data_routes)


# ======================================================================
#  Client helper — cria um TestClient com o router montado
# ======================================================================

@pytest.fixture
def client():
    """Cria um TestClient FastAPI com as data routes montadas."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from core.v2.data_routes import setup_data_routes

    app = FastAPI()
    app.include_router(setup_data_routes())
    return TestClient(app)


# ======================================================================
#  Mocks partilhados
# ======================================================================

def _make_mock_memory():
    """Cria um MemoryManager mockado com dados de teste."""
    mem = MagicMock()
    mem.recall_permanent = MagicMock(side_effect=lambda k: {
        "user_name": "João",
        "user_email": "joao@email.com",
    }.get(k))
    mem.recall_all_permanent.return_value = {
        "user_name": "João",
        "user_email": "joao@email.com",
    }
    mem.audit_log.return_value = [
        {
            "key": "user_name",
            "value": "João",
            "category": "fact",
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:00+00:00",
            "source": "conversation",
        },
        {
            "key": "user_email",
            "value": "joao@email.com",
            "category": "fact",
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:00+00:00",
            "source": "api",
        },
    ]
    mem.get_conversation_context.return_value = []
    mem.get_pending_approvals.return_value = []
    mem.delete_permanent.return_value = True
    return mem


def _make_mock_agents():
    """Cria um AgentFactory mockado com 7 agentes."""
    agents = MagicMock()
    agents.list_agents.return_value = [
        {"name": "research", "description": "Pesquisa e validação de informação"},
        {"name": "code", "description": "Programação e desenvolvimento"},
        {"name": "business", "description": "Análise de negócios e estratégia"},
        {"name": "legal", "description": "Análise jurídica"},
        {"name": "finance", "description": "Análise financeira"},
        {"name": "documentation", "description": "Documentação e relatórios"},
        {"name": "science", "description": "Análise científica"},
    ]
    return agents


# ======================================================================
#  Test: GET /v2/status
# ======================================================================

class TestStatus:
    """Testa o endpoint GET /v2/status."""

    @patch("core.v2.orchestrator.get_orchestrator")
    @patch("core.v2.memory.get_memory_manager")
    @patch("core.v2.agents.get_agent_factory")
    @patch("core.v2.llm_client.get_llm_client")
    @patch("core.v2.cache.get_cache")
    @patch("core.v2.safety.get_safety_manager")
    @patch("core.v2.tool_registry.get_tool_registry")
    def test_status_basic_structure(
        self,
        mock_get_tr, mock_get_safety, mock_get_cache,
        mock_get_llm, mock_get_agents, mock_get_mem, mock_get_orch,
        client,
    ):
        """Resposta contém campos essenciais."""
        # Orquestrador
        orch = MagicMock()
        orch.list_plans.return_value = []
        orch.get_audit_report.return_value = {}
        mock_get_orch.return_value = orch

        # Memória
        mem = MagicMock()
        mem.recall_all_permanent.return_value = {}
        mem.get_conversation_context.return_value = []
        mem.get_pending_approvals.return_value = []
        mock_get_mem.return_value = mem

        # Agentes
        agents = MagicMock()
        agents.list_agents.return_value = []
        mock_get_agents.return_value = agents

        # LLM
        llm = MagicMock()
        llm.get_model_info = AsyncMock(return_value={"model": "test", "available": True})
        mock_get_llm.return_value = llm

        # Cache
        cache = MagicMock()
        cache.get_stats.return_value = {"entries": 0, "hits": 0, "misses": 0}
        mock_get_cache.return_value = cache

        # Safety
        safety = MagicMock()
        safety.get_pending.return_value = []
        safety.get_all.return_value = []
        mock_get_safety.return_value = safety

        # Tool registry
        tr = MagicMock()
        tr.count.return_value = 0
        tr.list.return_value = []
        mock_get_tr.return_value = tr

        response = client.get("/v2/status")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "system" in data
        assert data["system"]["name"] == "Odysseus V2"
        assert "timestamp" in data["system"]
        assert "llm" in data
        assert "memory" in data
        assert "agents" in data
        assert "cache" in data
        assert "approvals" in data
        assert "plans" in data
        assert "tools" in data
        assert "workspace" in data

    @patch("core.v2.orchestrator.get_orchestrator")
    def test_status_error_handling(self, mock_get_orch, client):
        """Erro num subsistema não quebra os outros."""
        mock_get_orch.side_effect = Exception("Orchestrator error")

        response = client.get("/v2/status")

        assert response.status_code == 200
        data = response.json()
        # Mesmo com erro do orquestrador, o success deve ser True
        # porque o endpoint em si funcionou
        assert data["success"] is True


# ======================================================================
#  Test: GET /v2/settings
# ======================================================================

class TestSettings:
    """Testa o endpoint GET /v2/settings."""

    def test_settings_basic_structure(self, client):
        """Settings retorna estrutura esperada."""
        response = client.get("/v2/settings")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "settings" in data
        assert "system" in data["settings"]
        assert data["settings"]["system"]["version"] == "2.0.0"


# ======================================================================
#  Test: GET /v2/agents
# ======================================================================

class TestAgents:
    """Testa o endpoint GET /v2/agents."""

    @patch("core.v2.agents.get_agent_factory")
    def test_list_agents_success(self, mock_get_agents, client):
        """Lista todos os agentes com sucesso."""
        mock_get_agents.return_value = _make_mock_agents()

        response = client.get("/v2/agents")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total"] == 7
        assert len(data["agents"]) == 7
        assert data["agents"][0]["name"] == "research"
        assert data["agents"][0]["type"] == "specialized"

    @patch("core.v2.agents.get_agent_factory")
    def test_list_agents_empty(self, mock_get_agents, client):
        """Lista vazia quando não há agentes."""
        empty = MagicMock()
        empty.list_agents.return_value = []
        mock_get_agents.return_value = empty

        response = client.get("/v2/agents")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total"] == 0
        assert data["agents"] == []

    @patch("core.v2.agents.get_agent_factory")
    def test_list_agents_error(self, mock_get_agents, client):
        """Erro ao listar agentes retorna resposta de erro."""
        mock_get_agents.side_effect = Exception("Erro de teste")

        response = client.get("/v2/agents")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        assert data["agents"] == []


# ======================================================================
#  Test: GET /v2/skills
# ======================================================================

class TestSkills:
    """Testa o endpoint GET /v2/skills."""

    @patch("core.v2.agents.get_agent_factory")
    @patch("core.v2.tool_registry.get_tool_registry")
    @patch("core.v2.tool_chaining.get_chain_registry")
    def test_list_skills_all(self, mock_get_cr, mock_get_tr, mock_get_agents, client):
        """Lista skills de todas as categorias (chains + tools + agents)."""
        # Mock chain registry (0 chains — evitamos estado real)
        cr = MagicMock()
        cr.count.return_value = 0
        cr.list.return_value = []
        mock_get_cr.return_value = cr

        # Mock agents
        mock_get_agents.return_value = _make_mock_agents()

        # Mock tool registry (2 tools)
        tool1 = MagicMock()
        tool1.name = "web.fetch"
        tool1.definition = MagicMock(
            description="Fetch de páginas web",
            category="web",
            capabilities={"web", "fetch"},
            parameters={"properties": {"url": {"type": "string"}}},
        )
        tool2 = MagicMock()
        tool2.name = "file.read"
        tool2.definition = MagicMock(
            description="Leitura de ficheiros",
            category="file",
            capabilities={"file", "read"},
            parameters={"properties": {"path": {"type": "string"}}},
        )
        tr = MagicMock()
        tr.count.return_value = 2
        tr.list.return_value = [tool1, tool2]
        mock_get_tr.return_value = tr

        response = client.get("/v2/skills")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        # 2 tools + 7 agents = 9 (no chains since mocked to 0)
        assert data["total"] == 9
        categories = {s["category"] for s in data["skills"]}
        assert "tool" in categories
        assert "agent" in categories

    @patch("core.v2.tool_registry.get_tool_registry")
    def test_list_skills_filter_tool(self, mock_get_tr, client):
        """Filtro por categoria 'tool'."""
        tool = MagicMock()
        tool.name = "test.tool"
        tool.definition = MagicMock(
            description="Test tool",
            category="test",
            capabilities={"test"},
            parameters={"properties": {}},
        )
        tr = MagicMock()
        tr.count.return_value = 1
        tr.list.return_value = [tool]
        mock_get_tr.return_value = tr

        response = client.get("/v2/skills?category=tool")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["category_filter"] == "tool"
        if data["skills"]:
            for s in data["skills"]:
                assert s["category"] == "tool"

    @patch("core.v2.agents.get_agent_factory")
    def test_list_skills_filter_agent(self, mock_get_agents, client):
        """Filtro por categoria 'agent'."""
        mock_get_agents.return_value = _make_mock_agents()

        response = client.get("/v2/skills?category=agent")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["category_filter"] == "agent"
        if data["skills"]:
            for s in data["skills"]:
                assert s["category"] == "agent"


# ======================================================================
#  Test: GET /v2/memory
# ======================================================================

class TestListMemory:
    """Testa o endpoint GET /v2/memory."""

    @patch("core.v2.memory.get_memory_manager")
    def test_list_all(self, mock_get_mem, client):
        """Lista todas as memórias."""
        mock_get_mem.return_value = _make_mock_memory()

        response = client.get("/v2/memory")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total"] == 2
        assert data["total_all"] == 2
        assert len(data["entries"]) == 2

    @patch("core.v2.memory.get_memory_manager")
    def test_filter_by_category(self, mock_get_mem, client):
        """Filtro por categoria retorna apenas entradas dessa categoria."""
        mock_get_mem.return_value = _make_mock_memory()

        response = client.get("/v2/memory?category=fact")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        for e in data["entries"]:
            assert e["category"] == "fact"

    @patch("core.v2.memory.get_memory_manager")
    def test_search(self, mock_get_mem, client):
        """Pesquisa por texto retorna resultados correspondentes."""
        mock_get_mem.return_value = _make_mock_memory()

        response = client.get("/v2/memory?search=email")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        for e in data["entries"]:
            assert "email" in str(e.get("key", "")).lower() \
                or "email" in str(e.get("value", "")).lower()

    @patch("core.v2.memory.get_memory_manager")
    def test_empty_memory(self, mock_get_mem, client):
        """Memória vazia retorna lista vazia."""
        mem = MagicMock()
        mem.audit_log.return_value = []
        mem.recall_all_permanent.return_value = {}
        mock_get_mem.return_value = mem

        response = client.get("/v2/memory")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total"] == 0

    @patch("core.v2.memory.get_memory_manager")
    def test_error(self, mock_get_mem, client):
        """Erro ao listar retorna resposta de erro."""
        mock_get_mem.side_effect = Exception("Erro de teste")

        response = client.get("/v2/memory")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


# ======================================================================
#  Test: GET /v2/memory/{key}
# ======================================================================

class TestGetMemory:
    """Testa o endpoint GET /v2/memory/{key}."""

    @patch("core.v2.memory.get_memory_manager")
    def test_get_existing(self, mock_get_mem, client):
        """Obter memória existente."""
        mock_get_mem.return_value = _make_mock_memory()

        response = client.get("/v2/memory/user_name")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["entry"]["key"] == "user_name"
        assert data["entry"]["value"] == "João"

    @patch("core.v2.memory.get_memory_manager")
    def test_get_not_found(self, mock_get_mem, client):
        """Memória não encontrada retorna 404."""
        mem = _make_mock_memory()
        mem.recall_permanent = MagicMock(return_value=None)
        mock_get_mem.return_value = mem

        response = client.get("/v2/memory/nonexistent")

        assert response.status_code == 404


# ======================================================================
#  Test: POST /v2/memory
# ======================================================================

class TestCreateMemory:
    """Testa o endpoint POST /v2/memory."""

    @patch("core.v2.memory.get_memory_manager")
    @patch("core.v2.safety.get_safety_manager")
    def test_create_new(self, mock_get_safety, mock_get_mem, client):
        """Criar uma nova memória com sucesso."""
        mem = _make_mock_memory()
        mem.recall_permanent = MagicMock(return_value=None)  # não existe
        mock_get_mem.return_value = mem

        safety = MagicMock()
        safety.check_action.return_value = {
            "allowed": True,
            "requires_approval": False,
            "approval_id": None,
            "risk": "low",
            "message": "Permitido",
        }
        mock_get_safety.return_value = safety

        response = client.post(
            "/v2/memory",
            json={"key": "user_test", "value": "Teste", "category": "fact"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["action"] in ("created", "updated")
        assert data["entry"]["key"] == "user_test"

    @patch("core.v2.memory.get_memory_manager")
    @patch("core.v2.safety.get_safety_manager")
    def test_create_missing_key(self, mock_get_safety, mock_get_mem, client):
        """Criar memória sem key retorna erro."""
        mock_get_mem.return_value = _make_mock_memory()
        mock_get_safety.return_value = MagicMock()

        response = client.post(
            "/v2/memory",
            json={"value": "Teste"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "key" in data["error"].lower()

    @patch("core.v2.memory.get_memory_manager")
    @patch("core.v2.safety.get_safety_manager")
    def test_create_missing_value(self, mock_get_safety, mock_get_mem, client):
        """Criar memória sem value retorna erro."""
        mock_get_mem.return_value = _make_mock_memory()
        mock_get_safety.return_value = MagicMock()

        response = client.post(
            "/v2/memory",
            json={"key": "test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "value" in data["error"].lower()

    @patch("core.v2.memory.get_memory_manager")
    @patch("core.v2.safety.get_safety_manager")
    def test_create_invalid_category(self, mock_get_safety, mock_get_mem, client):
        """Categoria inválida retorna erro."""
        mock_get_mem.return_value = _make_mock_memory()
        mock_get_safety.return_value = MagicMock()

        response = client.post(
            "/v2/memory",
            json={"key": "test", "value": "v", "category": "invalid_cat"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "categoria" in data["error"].lower()

    @patch("core.v2.memory.get_memory_manager")
    @patch("core.v2.safety.get_safety_manager")
    def test_create_with_approval(self, mock_get_safety, mock_get_mem, client):
        """Quando safety requer aprovação, retorna pending."""
        mem = _make_mock_memory()
        mem.recall_permanent = MagicMock(return_value=None)
        mock_get_mem.return_value = mem

        safety = MagicMock()
        safety.check_action.return_value = {
            "allowed": False,
            "requires_approval": True,
            "approval_id": "APPROVAL-0001",
            "risk": "high",
            "message": "Requires approval",
        }
        mock_get_safety.return_value = safety

        response = client.post(
            "/v2/memory",
            json={"key": "sensitive", "value": "secret", "category": "fact"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["requires_approval"] is True
        assert data["approval_id"] == "APPROVAL-0001"


# ======================================================================
#  Test: DELETE /v2/memory/{key}
# ======================================================================

class TestDeleteMemory:
    """Testa o endpoint DELETE /v2/memory/{key}."""

    @patch("core.v2.memory.get_memory_manager")
    @patch("core.v2.safety.get_safety_manager")
    def test_delete_existing(self, mock_get_safety, mock_get_mem, client):
        """Apagar memória existente com force=true."""
        mem = _make_mock_memory()
        mem.recall_permanent = MagicMock(return_value="João")  # existe
        mock_get_mem.return_value = mem
        mock_get_safety.return_value = MagicMock()

        response = client.delete("/v2/memory/user_name?force=true")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["action"] == "deleted"
        assert data["key"] == "user_name"

    @patch("core.v2.memory.get_memory_manager")
    @patch("core.v2.safety.get_safety_manager")
    def test_delete_not_found(self, mock_get_safety, mock_get_mem, client):
        """Apagar memória inexistente retorna 404."""
        mem = _make_mock_memory()
        mem.recall_permanent = MagicMock(return_value=None)  # não existe
        mock_get_mem.return_value = mem
        mock_get_safety.return_value = MagicMock()

        response = client.delete("/v2/memory/nonexistent")

        assert response.status_code == 404

    @patch("core.v2.memory.get_memory_manager")
    @patch("core.v2.safety.get_safety_manager")
    def test_delete_with_approval(self, mock_get_safety, mock_get_mem, client):
        """Sem force=true, requer aprovação."""
        mem = _make_mock_memory()
        mem.recall_permanent = MagicMock(return_value="João")  # existe
        mock_get_mem.return_value = mem

        safety = MagicMock()
        safety.check_action.return_value = {
            "allowed": False,
            "requires_approval": True,
            "approval_id": "APPROVAL-0001",
            "risk": "medium",
            "message": "Requires approval for delete",
        }
        mock_get_safety.return_value = safety

        response = client.delete("/v2/memory/user_name")

        assert response.status_code == 200
        data = response.json()
        assert data["requires_approval"] is True
        assert data["approval_id"] == "APPROVAL-0001"


# ======================================================================
#  Test: Integração com v2_server e app.py
# ======================================================================

class TestIntegration:
    """Testa que as rotas são carregadas nos servidores."""

    def test_routes_exist_in_v2_server(self):
        """Verifica que o v2_server inclui as data routes."""
        import ast

        with open(str(_root / "v2_server.py"), "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = getattr(node.func, "attr", None)
                if func == "include_router":
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            fname = (
                                getattr(child.func, "id", "")
                                if isinstance(child.func, ast.Name)
                                else ""
                            )
                            if "setup_data_routes" in fname:
                                found = True

        assert found, "setup_data_routes não encontrado no v2_server.py"

    def test_routes_exist_in_app(self):
        """Verifica que o app.py inclui as data routes."""
        import ast

        with open(str(_root / "app.py"), "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = getattr(node.func, "attr", None)
                if func == "include_router":
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            fname = (
                                getattr(child.func, "id", "")
                                if isinstance(child.func, ast.Name)
                                else ""
                            )
                            if "setup_data_routes" in fname:
                                found = True

        assert found, "setup_data_routes não encontrado no app.py"
