"""
Testes RC1 — Bridge V2 (v2_server.py) + Integração (src/v2_bridge_integration.py)
==================================================================================

Valida os requisitos da fase RC1:

  1. /v1/chat/completions usa o OrchestratorAdapter (Orchestrator V2) POR
     OMISSÃO — o V2LLMWrapper nunca é o fluxo principal.
  2. Quando o Orchestrator não está disponível, o bridge DEGRADA para o LLM
     puro (a interface nunca fica sem resposta) com aviso nos logs.
  3. use_orchestrator=false força o modo LLM puro (debug/legado).
  4. /v1/models expõe sempre o modelo virtual "odysseus-v2".
  5. src/v2_bridge_integration.py:
       - ensure_v2_model_endpoint() regista o endpoint "Odysseus V2" de forma
         idempotente (nunca duplica, atualiza pinned/cached quando necessário);
       - ensure_v2_bridge_running() só arranca o bridge quando necessário e
         respeita ODYSSEUS_V2_BRIDGE_AUTOSTART=0;
       - start_v2_bridge_integration() nunca levanta exceções.
"""

from __future__ import annotations

from core.v2.constants import V2_BRIDGE_PORT

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Adiciona o diretório raiz do Odysseus ao path
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def test_bridge_binds_loopback_by_default(monkeypatch):
    """O bridge não tem autenticação própria: não pode nascer exposto na rede."""
    import importlib
    import core.v2.constants as constants

    monkeypatch.delenv("V2_BRIDGE_HOST", raising=False)
    reloaded = importlib.reload(constants)
    try:
        assert reloaded.V2_BRIDGE_HOST == "127.0.0.1"
        assert reloaded.V2_BRIDGE_PORT == 8001
    finally:
        importlib.reload(constants)


def _openai_response(content: str = "Resposta do orquestrador") -> dict:
    """Resposta OpenAI fake para o adapter mock."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1704067200,
        "model": "odysseus-v2",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
    }


async def _fake_stream(_body: dict | None = None):
    """Async generator fake para o streaming (SSE OpenAI)."""
    payload = (
        'data: {"id":"chatcmpl-test","object":"chat.completion.chunk",'
        '"choices":[{"index":0,"delta":{"role":"assistant","content":"olá"}}]}\n\n'
    )
    yield payload.encode("utf-8")
    yield b"data: [DONE]\n\n"


# ======================================================================
#  Fixtures partilhadas do bridge — /v1/chat/completions (Tarefa 1)
# ======================================================================

@pytest.fixture
def bridge(monkeypatch):
    """App bridge com OrchestratorAdapter disponível (mock)."""
    import v2_server
    from core.v2 import orchestrator_adapter

    # OrchestratorAdapter mock — disponível
    adapter = MagicMock()
    adapter.is_available = True
    adapter.chat_completion = AsyncMock(return_value=_openai_response())
    adapter.chat_completion_stream = MagicMock(side_effect=_fake_stream)
    monkeypatch.setattr(
        orchestrator_adapter, "create_adapter_from_config", lambda config: adapter
    )

    # LLM wrapper mock — deve ser chamado APENAS em modo degradado/debug
    mock_llm = MagicMock()
    mock_llm.list_models = AsyncMock(return_value=[])
    mock_llm.chat_completion = AsyncMock(
        return_value={"error": {"message": "LLM puro NÃO deve ser o fluxo padrão"}}
    )
    mock_llm.chat_completion_stream = MagicMock(side_effect=_fake_stream)
    monkeypatch.setattr(v2_server, "V2LLMWrapper", lambda config: mock_llm)
    monkeypatch.setattr(v2_server, "V2OrchestratorWrapper", lambda config: MagicMock())

    from fastapi.testclient import TestClient

    config = v2_server.BridgeConfig(port=V2_BRIDGE_PORT)
    app = v2_server.create_app(config)
    return {
        "client": TestClient(app),
        "adapter": adapter,
        "mock_llm": mock_llm,
    }


class TestChatCompletionsOrchestratorDefault:
    """O Orchestrator é o caminho padrão; o LLM puro é só legado/fallback."""

    def test_default_request_goes_through_orchestrator(self, bridge):
        """Sem parâmetros, o pedido passa pelo OrchestratorAdapter, não pelo LLM."""
        client = bridge["client"]
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "odysseus-v2",
                "messages": [{"role": "user", "content": "Olá!"}],
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "Resposta do orquestrador"

        bridge["adapter"].chat_completion.assert_awaited_once()
        bridge["mock_llm"].chat_completion.assert_not_awaited()

    def test_default_streaming_goes_through_orchestrator(self, bridge):
        """Streaming SSE por omissão usa chat_completion_stream do adapter."""
        client = bridge["client"]
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "odysseus-v2",
                "messages": [{"role": "user", "content": "Conta uma história"}],
                "stream": True,
            },
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert "olá" in resp.text
        assert "data: [DONE]" in resp.text

        bridge["adapter"].chat_completion_stream.assert_called_once()
        bridge["mock_llm"].chat_completion_stream.assert_not_called()

    def test_use_orchestrator_false_forces_llm_puro(self, bridge):
        """?use_orchestrator=false força o modo LLM puro (debug/legado)."""
        client = bridge["client"]
        resp = client.post(
            "/v1/chat/completions?use_orchestrator=false",
            json={
                "model": "odysseus-v2",
                "messages": [{"role": "user", "content": "Debug"}],
            },
        )

        # O LLM mock devolve erro — mas é ele o handler em modo forçado
        assert resp.status_code == 400
        bridge["mock_llm"].chat_completion.assert_awaited_once()
        bridge["adapter"].chat_completion.assert_not_awaited()


@pytest.fixture
def bridge_unavailable(monkeypatch):
    """App bridge com OrchestratorAdapter indisponível (mock)."""
    import v2_server
    from core.v2 import orchestrator_adapter

    # OrchestratorAdapter indisponível → create_app define orch_adapter=None
    adapter = MagicMock()
    adapter.is_available = False
    monkeypatch.setattr(
        orchestrator_adapter, "create_adapter_from_config", lambda config: adapter
    )

    mock_llm = MagicMock()
    mock_llm.list_models = AsyncMock(return_value=[])
    mock_llm.chat_completion = AsyncMock(
        return_value=_openai_response("Resposta degradada (LLM puro)")
    )
    mock_llm.chat_completion_stream = MagicMock(side_effect=_fake_stream)
    monkeypatch.setattr(v2_server, "V2LLMWrapper", lambda config: mock_llm)
    monkeypatch.setattr(v2_server, "V2OrchestratorWrapper", lambda config: MagicMock())

    from fastapi.testclient import TestClient

    config = v2_server.BridgeConfig(port=V2_BRIDGE_PORT)
    app = v2_server.create_app(config)
    return {
        "client": TestClient(app),
        "mock_llm": mock_llm,
        "adapter": adapter,
    }


class TestChatCompletionsDegradedFallback:
    """Orchestrator indisponível → degrada para LLM puro (nunca sem resposta)."""

    def test_degraded_fallback_uses_llm(self, bridge_unavailable):
        """Sem Orchestrator, o pedido degrada para o LLM puro com resposta válida."""
        client = bridge_unavailable["client"]
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "odysseus-v2",
                "messages": [{"role": "user", "content": "Tudo bem?"}],
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "Resposta degradada (LLM puro)"
        bridge_unavailable["mock_llm"].chat_completion.assert_awaited_once()
        bridge_unavailable["adapter"].chat_completion.assert_not_called()

    def test_degraded_response_is_marked(self, bridge_unavailable):
        """O fallback degradado é sinalizado (X-Odysseus-Mode) para não mascarar falhas."""
        client = bridge_unavailable["client"]
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "odysseus-v2",
                "messages": [{"role": "user", "content": "Olá"}],
            },
        )
        assert resp.headers.get("x-odysseus-mode") == "degraded"

    def test_debug_mode_is_not_marked_degraded(self, bridge):
        """use_orchestrator=false é um modo explícito — não é sinalizado como degradado."""
        client = bridge["client"]
        resp = client.post(
            "/v1/chat/completions?use_orchestrator=false",
            json={
                "model": "odysseus-v2",
                "messages": [{"role": "user", "content": "Debug"}],
            },
        )
        assert "x-odysseus-mode" not in resp.headers


class TestModelsEndpoint:
    """/v1/models expõe sempre o modelo virtual odysseus-v2."""

    def test_odysseus_v2_always_listed(self, monkeypatch):
        import v2_server

        mock_llm = MagicMock()
        mock_llm.list_models = AsyncMock(return_value=[])
        monkeypatch.setattr(v2_server, "V2LLMWrapper", lambda config: mock_llm)
        monkeypatch.setattr(v2_server, "V2OrchestratorWrapper", lambda config: MagicMock())
        monkeypatch.setattr(
            "core.v2.orchestrator_adapter.create_adapter_from_config",
            lambda config: MagicMock(is_available=False),
        )

        from fastapi.testclient import TestClient

        app = v2_server.create_app(v2_server.BridgeConfig(port=V2_BRIDGE_PORT))
        client = TestClient(app)
        resp = client.get("/v1/models")

        assert resp.status_code == 200
        model_ids = [m["id"] for m in resp.json()["data"]]
        assert "odysseus-v2" in model_ids

    def test_health_endpoint(self, monkeypatch):
        """GET /health devolve status ok (usado pelo healthcheck da integração)."""
        import v2_server

        monkeypatch.setattr(v2_server, "V2LLMWrapper", lambda config: MagicMock())
        monkeypatch.setattr(v2_server, "V2OrchestratorWrapper", lambda config: MagicMock())
        monkeypatch.setattr(
            "core.v2.orchestrator_adapter.create_adapter_from_config",
            lambda config: MagicMock(is_available=False),
        )

        from fastapi.testclient import TestClient

        app = v2_server.create_app(v2_server.BridgeConfig(port=V2_BRIDGE_PORT))
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ======================================================================
#  Testes da integração — seed idempotente (Tarefas 2 e 4)
# ======================================================================

class TestEnsureV2ModelEndpoint:
    """ensure_v2_model_endpoint regista "Odysseus V2" sem duplicar."""

    @pytest.fixture
    def db_session(self, monkeypatch):
        """SQLite real temporário ligado a core.database.SessionLocal."""
        from tests.helpers.sqlite_db import make_temp_sqlite
        import core.database

        SessionLocal, engine, _tmp = make_temp_sqlite(core.database.Base.metadata)
        monkeypatch.setattr(core.database, "SessionLocal", SessionLocal)
        return SessionLocal, engine

    def test_creates_endpoint_once(self, db_session, monkeypatch):
        """A primeira chamada cria o endpoint; a segunda não duplica."""
        from src.v2_bridge_integration import ensure_v2_model_endpoint
        from core.database import ModelEndpoint

        SessionLocal, engine = db_session
        try:
            first_id = ensure_v2_model_endpoint()
            second_id = ensure_v2_model_endpoint()

            assert first_id == second_id == "odysseus-v2"

            with SessionLocal() as s:
                rows = s.query(ModelEndpoint).all()
                assert len(rows) == 1
                ep = rows[0]
                assert ep.name == "Odysseus V2"
                assert ep.base_url == f"http://127.0.0.1:{V2_BRIDGE_PORT}/v1"
                assert ep.is_enabled is True
        finally:
            engine.dispose()

    def test_pins_and_caches_odysseus_v2(self, db_session):
        """O modelo odysseus-v2 fica pinned + cached para aparecer no picker."""
        from src.v2_bridge_integration import ensure_v2_model_endpoint
        from core.database import ModelEndpoint
        import json

        SessionLocal, engine = db_session
        try:
            ensure_v2_model_endpoint()
            with SessionLocal() as s:
                ep = s.query(ModelEndpoint).filter_by(id="odysseus-v2").one()
                assert "odysseus-v2" in json.loads(ep.pinned_models or "[]")
                assert "odysseus-v2" in json.loads(ep.cached_models or "[]")
        finally:
            engine.dispose()

    def test_updates_existing_endpoint_without_duplicating(self, db_session):
        """Endpoint já existente: não duplica nem re-ativa desativação do utilizador."""
        from src.v2_bridge_integration import ensure_v2_model_endpoint
        from core.database import ModelEndpoint
        import json

        SessionLocal, engine = db_session
        try:
            # Pré-existir endpoint desativado pelo utilizador, sem pinned
            with SessionLocal() as s:
                ep = ModelEndpoint(
                    id="odysseus-v2",
                    name="Odysseus V2",
                    base_url=f"http://127.0.0.1:{V2_BRIDGE_PORT}/v1",
                    is_enabled=False,
                    model_type="llm",
                    endpoint_kind="local",
                    model_refresh_mode="manual",
                )
                s.add(ep)
                s.commit()

            ensure_v2_model_endpoint()

            with SessionLocal() as s:
                rows = s.query(ModelEndpoint).all()
                assert len(rows) == 1  # nunca duplica
                ep = s.query(ModelEndpoint).filter_by(id="odysseus-v2").one()
                # Escolha do utilizador respeitada: continua desativado
                assert ep.is_enabled is False
                # Mas o modelo fica pinned para quando o utilizador o ativar
                assert "odysseus-v2" in json.loads(ep.pinned_models or "[]")
        finally:
            engine.dispose()


# ======================================================================
#  Testes da integração — autostart do bridge (Tarefa 2)
# ======================================================================

class TestEnsureV2BridgeRunning:
    """ensure_v2_bridge_running arranca o bridge só quando necessário."""

    def test_noop_when_already_reachable(self, monkeypatch):
        """Bridge já ativo → não arranca subprocesso."""
        from src.v2_bridge_integration import ensure_v2_bridge_running

        monkeypatch.setattr(
            "src.v2_bridge_integration.is_v2_bridge_reachable", lambda timeout=1.0: True
        )
        popen = MagicMock()
        monkeypatch.setattr("src.v2_bridge_integration.subprocess.Popen", popen)

        assert ensure_v2_bridge_running() is True
        popen.assert_not_called()

    def test_autostart_disabled_does_not_spawn(self, monkeypatch):
        """ODYSSEUS_V2_BRIDGE_AUTOSTART=0 → não arranca, devolve False."""
        from src.v2_bridge_integration import ensure_v2_bridge_running

        monkeypatch.setattr(
            "src.v2_bridge_integration.is_v2_bridge_reachable", lambda timeout=1.0: False
        )
        monkeypatch.setenv("ODYSSEUS_V2_BRIDGE_AUTOSTART", "0")
        popen = MagicMock()
        monkeypatch.setattr("src.v2_bridge_integration.subprocess.Popen", popen)

        assert ensure_v2_bridge_running() is False
        popen.assert_not_called()

    def test_starts_bridge_when_offline(self, monkeypatch):
        """Bridge offline + autostart ativo → lança v2_server.py e espera /health."""
        from src.v2_bridge_integration import ensure_v2_bridge_running

        # Primeira verificação falha (offline), mas depois de spawn o bridge
        # fica saudável — simula o boot real do v2_server.py.
        reachable_results = iter([False, True])
        monkeypatch.setattr(
            "src.v2_bridge_integration.is_v2_bridge_reachable",
            lambda timeout=1.0: next(reachable_results, False),
        )
        monkeypatch.delenv("ODYSSEUS_V2_BRIDGE_AUTOSTART", raising=False)
        monkeypatch.setenv("ODYSSEUS_V2_APPROVAL_TOKEN", "test-token-not-for-logs")
        monkeypatch.setenv("V2_TEST_INHERITED_SETTING", "preserved")
        popen = MagicMock()
        popen.return_value.pid = 4242
        monkeypatch.setattr("src.v2_bridge_integration.subprocess.Popen", popen)

        assert ensure_v2_bridge_running() is True
        popen.assert_called_once()
        script = popen.call_args[0][0][-1]
        assert script.endswith("v2_server.py")
        child_env = popen.call_args.kwargs["env"]
        assert child_env["ODYSSEUS_V2_APPROVAL_TOKEN"] == "test-token-not-for-logs"
        assert child_env["V2_TEST_INHERITED_SETTING"] == "preserved"

    def test_reads_existing_windows_user_token_when_process_environment_is_stale(
        self, monkeypatch
    ):
        """Recupera o mesmo token User sem o expor quando a app herdou ambiente antigo."""
        import sys
        from types import SimpleNamespace

        import src.v2_bridge_integration as integration

        class RegistryKey:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        token = "user-scope-token-not-for-logs"
        fake_winreg = SimpleNamespace(
            HKEY_CURRENT_USER="HKCU",
            OpenKey=lambda root, path: RegistryKey(),
            QueryValueEx=lambda key, name: (token, 1),
        )
        monkeypatch.setattr(integration.os, "name", "nt")
        monkeypatch.delenv("ODYSSEUS_V2_APPROVAL_TOKEN", raising=False)
        monkeypatch.setitem(sys.modules, "winreg", fake_winreg)

        child_env = integration._bridge_process_environment()

        assert child_env["ODYSSEUS_V2_APPROVAL_TOKEN"] == token
        assert "ODYSSEUS_V2_APPROVAL_TOKEN" not in integration.os.environ

    def test_waits_for_health_after_spawn(self, monkeypatch):
        """Após o spawn, o bridge só é dado como pronto quando /health responde."""
        from src.v2_bridge_integration import ensure_v2_bridge_running

        reachable_results = iter([False, False, True])  # offline, booting, online
        monkeypatch.setattr(
            "src.v2_bridge_integration.is_v2_bridge_reachable",
            lambda timeout=1.0: next(reachable_results, True),
        )
        monkeypatch.delenv("ODYSSEUS_V2_BRIDGE_AUTOSTART", raising=False)
        monkeypatch.setattr("src.v2_bridge_integration.subprocess.Popen", MagicMock())
        monkeypatch.setattr("src.v2_bridge_integration.time.sleep", lambda s: None)

        assert ensure_v2_bridge_running() is True


class TestStartV2BridgeIntegration:
    """start_v2_bridge_integration orquestra seed + autostart sem levantar exceções."""

    @pytest.mark.asyncio
    async def test_returns_state_dict(self, monkeypatch):
        from src.v2_bridge_integration import start_v2_bridge_integration

        monkeypatch.setattr(
            "src.v2_bridge_integration.ensure_v2_model_endpoint",
            lambda db=None: "odysseus-v2",
        )
        monkeypatch.setattr(
            "src.v2_bridge_integration.ensure_v2_bridge_running",
            lambda force=False: True,
        )

        state = await start_v2_bridge_integration()
        assert state == {"endpoint_id": "odysseus-v2", "bridge_running": True}

    @pytest.mark.asyncio
    async def test_never_raises_on_failures(self, monkeypatch):
        """Falhas internas degradam para logs — o arranque da app nunca bloqueia."""
        from src.v2_bridge_integration import start_v2_bridge_integration

        def boom(*args, **kwargs):
            raise RuntimeError("DB indisponível")

        monkeypatch.setattr(
            "src.v2_bridge_integration.ensure_v2_model_endpoint", boom
        )
        monkeypatch.setattr(
            "src.v2_bridge_integration.ensure_v2_bridge_running", boom
        )

        state = await start_v2_bridge_integration()
        assert state == {"endpoint_id": None, "bridge_running": False}
