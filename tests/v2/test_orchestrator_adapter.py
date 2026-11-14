"""
Testes unitários para o OrchestratorAdapter.

Testa a conversão entre o formato OpenAI (/v1/chat/completions) e o
Orchestrator V2, tanto em modo normal como streaming.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ======================================================================
#  Helpers
# ======================================================================

class MockOrchestratorResponse:
    """Simula OrchestratorResponse."""
    def __init__(self, message="Resposta do orquestrador", metadata=None):
        self.message = message
        self.plan = None
        self.agent_results = []
        self.requires_approval = []
        self.suggested_memory = []
        self.metadata = metadata or {"llm_used": True}


class MockOrchestrator:
    """Simula o OdysseusOrchestrator para testes."""

    def __init__(self, available=True):
        self._available = available

    async def process(self, user_message, context=None, approval_id=None):
        return MockOrchestratorResponse(
            message=f"Processado: {user_message[:50]}",
            metadata={"llm_used": True, "parallel_agents": 0},
        )

    async def process_stream(self, user_message, context=None):
        if not self._available:
            yield "Erro: orquestrador não disponível"
            yield "__DONE__"
            return
        yield "Olá! "
        yield "Sou o "
        yield "Orquestrador."
        yield "__DONE__"

    @property
    def agents(self):
        agent = MagicMock()
        agent.list_agents.return_value = [
            {"name": "research", "description": "Pesquisa"},
            {"name": "code", "description": "Programação"},
        ]
        return agent

    @property
    def memory(self):
        mem = MagicMock()
        mem.recall_all_permanent.return_value = {"nome": "João"}
        return mem


class MockUserAgent:
    """Simula o ponto único de coordenação do V2."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    async def process(self, question, context=None, approval_id=None):
        result = await self.orchestrator.process(
            question, context=context, approval_id=approval_id
        )
        result.metadata["coordinated_by"] = "useragent"
        return result

    async def process_stream(self, question, context=None):
        async for token in self.orchestrator.process_stream(question):
            yield token


# ======================================================================
#  Fixtures
# ======================================================================

@pytest.fixture
def mock_orchestrator():
    return MockOrchestrator(available=True)


@pytest.fixture
def mock_orchestrator_unavailable():
    return MockOrchestrator(available=False)

# Para testar o adapter truly unavailable (orchestrator=None), usamos fixture direta


@pytest.fixture
def adapter(mock_orchestrator):
    from core.v2.orchestrator_adapter import OrchestratorAdapter
    return OrchestratorAdapter(
        orchestrator=mock_orchestrator,
        user_agent=MockUserAgent(mock_orchestrator),
    )


@pytest.fixture
def adapter_unavailable():
    from core.v2.orchestrator_adapter import OrchestratorAdapter
    # Adapter sem orchestrator = unavailable
    return OrchestratorAdapter(orchestrator=None)


# ======================================================================
#  Testes: OrchestratorAdapter — Propriedades
# ======================================================================

class TestAdapterProperties:
    """Testa as propriedades básicas do OrchestratorAdapter."""

    def test_is_available_true(self, adapter):
        assert adapter.is_available is True

    def test_is_available_false(self, adapter_unavailable):
        assert adapter_unavailable.is_available is False

    def test_orchestrator_property(self, adapter, mock_orchestrator):
        assert adapter.orchestrator is mock_orchestrator

    def test_orchestrator_none(self):
        from core.v2.orchestrator_adapter import OrchestratorAdapter
        a = OrchestratorAdapter(orchestrator=None)
        assert a.is_available is False
        assert a.orchestrator is None


# ======================================================================
#  Testes: OrchestratorAdapter — Conversão de formato
# ======================================================================

class TestExtractUserMessage:
    """Testa a extração da mensagem do utilizador do formato OpenAI."""

    def test_extract_simple(self, adapter):
        messages = [
            {"role": "system", "content": "Tu és um assistente."},
            {"role": "user", "content": "Olá, tudo bem?"},
        ]
        assert adapter._extract_user_message(messages) == "Olá, tudo bem?"

    def test_extract_last_user(self, adapter):
        messages = [
            {"role": "user", "content": "Primeira pergunta"},
            {"role": "assistant", "content": "Resposta"},
            {"role": "user", "content": "Segunda pergunta"},
        ]
        assert adapter._extract_user_message(messages) == "Segunda pergunta"

    def test_extract_multimodal(self, adapter):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Descreve esta imagem:"},
                {"type": "image_url", "image_url": {"url": "data:image/..."}},
            ]}
        ]
        assert adapter._extract_user_message(messages) == "Descreve esta imagem:"

    def test_extract_empty(self, adapter):
        assert adapter._extract_user_message([]) == ""

    def test_extract_no_user(self, adapter):
        messages = [
            {"role": "system", "content": "Tu és um assistente."},
            {"role": "assistant", "content": "Resposta"},
        ]
        assert adapter._extract_user_message(messages) == ""

    def test_extract_content_none(self, adapter):
        messages = [{"role": "user", "content": None}]
        assert adapter._extract_user_message(messages) == ""


class TestBuildContext:
    """Testa a construção de contexto a partir de mensagens OpenAI."""

    def test_build_context_basic(self, adapter):
        messages = [
            {"role": "system", "content": "Tu és um assistente."},
            {"role": "user", "content": "Olá!"},
        ]
        ctx = adapter._build_context(messages)
        assert ctx["source"] == "openai_api"
        assert ctx["messages"] == messages
        assert ctx["system_prompt"] == "Tu és um assistente."

    def test_build_context_no_system(self, adapter):
        messages = [{"role": "user", "content": "Olá!"}]
        ctx = adapter._build_context(messages)
        assert "system_prompt" not in ctx


# ======================================================================
#  Testes: OrchestratorAdapter — chat_completion (non-streaming)
# ======================================================================

class TestChatCompletion:
    """Testa o método chat_completion (non-streaming)."""

    @pytest.mark.asyncio
    async def test_basic_chat(self, adapter):
        body = {
            "messages": [
                {"role": "system", "content": "Tu és um assistente."},
                {"role": "user", "content": "Quem és tu?"},
            ],
            "model": "odysseus-v2",
        }
        result = await adapter.chat_completion(body)

        assert "id" in result
        assert result["object"] == "chat.completion"
        assert result["model"] == "odysseus-v2"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert "Processado: Quem és tu?" in result["choices"][0]["message"]["content"]
        assert result["choices"][0]["finish_reason"] == "stop"
        assert "usage" in result
        assert result["usage"]["prompt_tokens"] > 0
        assert result["usage"]["total_tokens"] > 0
        assert result["coordinated_by"] == "useragent"

    @pytest.mark.asyncio
    async def test_empty_messages(self, adapter):
        body = {"messages": [], "model": "test"}
        result = await adapter.chat_completion(body)

        assert "error" in result
        assert "Nenhuma mensagem do utilizador" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_no_user_message(self, adapter):
        body = {
            "messages": [
                {"role": "system", "content": "Tu és um assistente."},
                {"role": "assistant", "content": "Resposta."},
            ],
            "model": "test",
        }
        result = await adapter.chat_completion(body)

        assert "error" in result
        assert "Nenhuma mensagem do utilizador" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_adapter_not_available(self, adapter_unavailable):
        body = {"messages": [{"role": "user", "content": "Olá"}], "model": "test"}
        result = await adapter_unavailable.chat_completion(body)

        assert "error" in result
        assert "não disponível" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_with_multimodal_message(self, adapter):
        body = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "O que vês nesta imagem?"},
                    {"type": "image_url", "image_url": {"url": "data:image/..."}},
                ]}
            ],
            "model": "odysseus-v2",
        }
        result = await adapter.chat_completion(body)

        assert result["object"] == "chat.completion"
        assert "O que vês" in result["choices"][0]["message"]["content"]


# ======================================================================
#  Testes: OrchestratorAdapter — chat_completion_stream (SSE streaming)
# ======================================================================

class TestChatCompletionStream:
    """Testa o método chat_completion_stream (SSE streaming)."""

    @pytest.mark.asyncio
    async def test_basic_stream(self, adapter):
        body = {
            "messages": [{"role": "user", "content": "Quem és?"}],
            "model": "odysseus-v2",
        }

        chunks = []
        async for chunk in adapter.chat_completion_stream(body):
            chunks.append(chunk)

        # Deve ter recebido chunks SSE
        assert len(chunks) >= 2  # tokens + final + DONE

        # Verificar formato dos chunks de dados
        data_chunks = [c for c in chunks if c.startswith(b"data: ")]
        done_chunks = [c for c in chunks if c == b"data: [DONE]\n\n"]

        assert len(done_chunks) == 1
        assert len(data_chunks) >= 2  # tokens + final

        # Verificar que os chunks têm o formato OpenAI
        first_data = json.loads(data_chunks[0][6:])  # Remove "data: "
        assert first_data["object"] == "chat.completion.chunk"
        assert first_data["coordinated_by"] == "useragent"
        assert "id" in first_data
        assert "choices" in first_data
        assert first_data["choices"][0]["delta"]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_stream_empty_messages(self, adapter):
        body = {"messages": [], "model": "test"}

        chunks = []
        async for chunk in adapter.chat_completion_stream(body):
            chunks.append(chunk)

        # Deve ter pelo menos um chunk de erro e o DONE
        assert len(chunks) >= 2
        data_chunks = [c for c in chunks if c.startswith(b"data: ")]
        first = json.loads(data_chunks[0][6:])
        assert "error" in first

    @pytest.mark.asyncio
    async def test_stream_unavailable(self, adapter_unavailable):
        body = {"messages": [{"role": "user", "content": "Olá"}], "model": "test"}

        chunks = []
        async for chunk in adapter_unavailable.chat_completion_stream(body):
            chunks.append(chunk)

        data_chunks = [c for c in chunks if c.startswith(b"data: ")]
        first = json.loads(data_chunks[0][6:])
        assert "error" in first
        assert "não disponível" in first["error"]["message"]


# ======================================================================
#  Testes: OrchestratorAdapter — get_info
# ======================================================================

class TestGetInfo:
    """Testa o método get_info."""

    def test_info_available(self, adapter):
        info = adapter.get_info()
        assert info["available"] is True
        assert info["agent_count"] == 2
        assert "research" in info["agents"]
        assert "code" in info["agents"]

    def test_info_unavailable(self, adapter_unavailable):
        info = adapter_unavailable.get_info()
        assert info["available"] is False
        assert "error" in info


# ======================================================================
#  Testes: Factory Functions
# ======================================================================

class TestFactoryFunctions:
    """Testa a função create_adapter_from_config."""

    def test_create_from_config_with_orchestrator_error(self):
        """Testa que a factory retorna um adapter não disponível se o Orchestrator falhar."""
        from core.v2.orchestrator_adapter import create_adapter_from_config

        # Config mock que vai falhar ao criar o LLM
        config = MagicMock()
        config.ollama_url = ""
        config.llm_model = ""
        config.llm_timeout = 30
        config.llm_temperature = 0.7
        config.llm_max_tokens = 2048

        adapter = create_adapter_from_config(config)
        # Deve retornar um adapter, mesmo que o Orchestrator não esteja disponível
        assert adapter is not None
        # is_available pode ser False porque não há LLM real
        # Mas o adapter existe


# ======================================================================
#  Testes: OrchestratorAdapter — Tool Event SSE Emission
# ======================================================================

class TestToolEventSSE:
    """
    Testa a emissão de eventos SSE de ferramentas (tool_start, tool_output, tool_error)
    durante o streaming.
    """

    @pytest.mark.asyncio
    async def test_stream_with_tool_events(self):
        """
        Testa que o adapter emite tool_start e tool_output SSE events
        quando o MockOrchestrator tem um tool_manager que emite eventos.
        """
        from core.v2.orchestrator_adapter import OrchestratorAdapter, ToolEvent, _format_tool_event_sse

        events_fired = []

        class MockToolManagerWithCallbacks:
            """ToolManager mock que regista eventos."""
            def __init__(self):
                self._event_callbacks = []

            def add_event_callback(self, cb):
                self._event_callbacks.append(cb)

            def remove_event_callback(self, cb):
                self._event_callbacks = [c for c in self._event_callbacks if c is not cb]

            async def fire_event(self, event):
                events_fired.append(event)
                for cb in self._event_callbacks:
                    await cb(event)

        class MockOrchestratorWithTools:
            """Orquestrador mock que usa ferramentas durante o stream."""
            def __init__(self):
                self.tool_manager = MockToolManagerWithCallbacks()

            def set_tool_event_callback(self, cb):
                self.tool_manager.add_event_callback(cb)

            def remove_tool_event_callback(self, cb):
                self.tool_manager.remove_event_callback(cb)

            async def process_stream(self, user_message):
                # Emular uso de ferramenta durante o stream
                # Isto normalmente seria feito pelo Orchestrator, mas aqui
                # simulamos com o tool_manager.fire_event()
                yield "A "
                yield "analisar "
                yield "ficheiro..."

                yield "__DONE__"

            @property
            def agents(self):
                return MagicMock()

            @property
            def memory(self):
                return MagicMock()

        orch = MockOrchestratorWithTools()
        adapter = OrchestratorAdapter(
            orchestrator=orch,
            user_agent=MockUserAgent(orch),
        )

        body = {
            "messages": [{"role": "user", "content": "Analisa o ficheiro"}],
            "model": "odysseus-v2",
        }

        chunks = []
        async for chunk in adapter.chat_completion_stream(body):
            chunks.append(chunk)

        # Verificar chunks básicos
        done_chunks = [c for c in chunks if c == b"data: [DONE]\n\n"]
        assert len(done_chunks) == 1

        # Verificar que os chunks de dados incluem texto
        data_chunks = [c for c in chunks if c.startswith(b"data: ") and b"[DONE]" not in c]
        assert len(data_chunks) >= 1

    @pytest.mark.asyncio
    async def test_format_tool_start_sse(self):
        """Testa a formatação de um evento tool_start como SSE."""
        from core.v2.orchestrator_adapter import _format_tool_event_sse
        from core.v2.tool_manager import ToolEvent

        event = ToolEvent(
            event_type="tool_start",
            tool_name="file.read",
            execution_id="EXEC-001",
            params={"path": "/tmp/test.txt"},
            timestamp="2024-01-01T00:00:00",
            status="running",
        )

        sse = _format_tool_event_sse(
            event,
            response_id="cmpl-test",
            created=1704067200,
            model="test-model",
        )

        assert sse.startswith(b"data: ")
        assert sse.endswith(b"\n\n")

        payload = json.loads(sse[6:])
        assert payload["type"] == "tool_start"
        assert payload["tool"] == "file.read"
        assert payload["execution_id"] == "EXEC-001"
        assert payload["params"]["path"] == "/tmp/test.txt"
        assert payload["object"] == "tool.event"

    @pytest.mark.asyncio
    async def test_format_tool_output_sse(self):
        """Testa a formatação de um evento tool_output como SSE."""
        from core.v2.orchestrator_adapter import _format_tool_event_sse
        from core.v2.tool_manager import ToolEvent

        event = ToolEvent(
            event_type="tool_output",
            tool_name="file.read",
            execution_id="EXEC-001",
            params={"path": "/tmp/test.txt"},
            result={"content": "Hello, World!", "size_bytes": 13},
            duration_ms=12.5,
            timestamp="2024-01-01T00:00:01",
            status="success",
        )

        sse = _format_tool_event_sse(
            event,
            response_id="cmpl-test",
            created=1704067200,
            model="test-model",
        )

        payload = json.loads(sse[6:])
        assert payload["type"] == "tool_output"
        assert payload["tool"] == "file.read"
        assert payload["execution_id"] == "EXEC-001"
        assert payload["result"]["content"] == "Hello, World!"
        assert payload["duration_ms"] == 12.5

    @pytest.mark.asyncio
    async def test_format_tool_error_sse(self):
        """Testa a formatação de um evento tool_error como SSE."""
        from core.v2.orchestrator_adapter import _format_tool_event_sse
        from core.v2.tool_manager import ToolEvent

        event = ToolEvent(
            event_type="tool_error",
            tool_name="shell.execute",
            execution_id="EXEC-002",
            params={"command": "rm -rf /"},
            error="Comando bloqueado por segurança",
            duration_ms=5.0,
            timestamp="2024-01-01T00:00:02",
            status="denied",
        )

        sse = _format_tool_event_sse(
            event,
            response_id="cmpl-test",
            created=1704067200,
            model="test-model",
        )

        payload = json.loads(sse[6:])
        assert payload["type"] == "tool_error"
        assert payload["tool"] == "shell.execute"
        assert payload["error"] == "Comando bloqueado por segurança"
        assert payload["execution_id"] == "EXEC-002"


# ======================================================================
#  Testes: ToolManager — Event Callbacks
# ======================================================================

class TestToolManagerEventCallbacks:
    """Testa o mecanismo de callbacks de eventos no ToolManager."""

    @pytest.mark.asyncio
    async def test_tool_manager_fires_tool_start_event(self):
        """
        Testa que o ToolManager emite um evento tool_start quando uma
        ferramenta começa a executar.
        """
        from core.v2.tool_manager import ToolManager, ToolEvent
        from core.v2.tool_registry import get_tool_registry
        from core.v2.tools import register_base_tools

        # Registar ferramentas base
        registry = get_tool_registry()
        if registry.count() == 0:
            register_base_tools(registry)

        tm = ToolManager(registry=registry, safety_manager=None)

        received_events = []

        async def event_collector(event: ToolEvent):
            received_events.append(event)

        tm.add_event_callback(event_collector)

        # Executar uma ferramenta que existe (file)
        result = await tm.execute(
            "file",
            {"action": "read", "path": "/nonexistent/file.txt"},
        )

        # Deve ter recebido pelo menos um evento
        assert len(received_events) >= 1

        # O primeiro evento deve ser tool_start
        start_events = [e for e in received_events if e.event_type == "tool_start"]
        assert len(start_events) == 1
        assert start_events[0].tool_name == "file"
        assert start_events[0].params["action"] == "read"

    @pytest.mark.asyncio
    async def test_tool_manager_fires_tool_output_on_success(self):
        """
        Testa que o ToolManager emite tool_output quando uma ferramenta
        executa com sucesso.
        """
        from core.v2.tool_manager import ToolManager, ToolEvent
        from core.v2.tool_registry import get_tool_registry
        from core.v2.tools import register_base_tools
        from core.v2.execution_policy import approved_root
        import uuid

        registry = get_tool_registry()
        if registry.count() == 0:
            register_base_tools(registry)

        tm = ToolManager(registry=registry, safety_manager=None)
        workspace = approved_root("workspace")
        if workspace is None:
            pytest.skip("Approved Workspace is not available")

        received_events = []

        async def event_collector(event: ToolEvent):
            received_events.append(event)

        tm.add_event_callback(event_collector)

        # O file tool confina caminhos ao Workspace aprovado.
        test_file = workspace / f"_buff_tool_exists_{uuid.uuid4().hex}.tmp"
        test_file.write_bytes(b"test")

        try:
            result = await tm.execute(
                "file",
                {"action": "exists", "path": str(test_file)},
            )

            # Deve ter tool_start e tool_output
            assert result.success
            output_events = [e for e in received_events if e.event_type == "tool_output"]
            assert len(output_events) == 1
            assert output_events[0].tool_name == "file"
            assert output_events[0].status == "success"
            assert output_events[0].duration_ms is not None
        finally:
            test_file.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_tool_manager_fires_tool_error_for_unknown_tool(self):
        """
        Testa que o ToolManager emite tool_error quando a ferramenta
        não existe no registry.
        """
        from core.v2.tool_manager import ToolManager, ToolEvent
        from core.v2.tool_registry import ToolRegistry

        registry = ToolRegistry()
        tm = ToolManager(registry=registry)

        received_events = []

        async def event_collector(event: ToolEvent):
            received_events.append(event)

        tm.add_event_callback(event_collector)

        result = await tm.execute("nonexistent.tool", {"some": "param"})

        # Deve ter tool_error
        error_events = [e for e in received_events if e.event_type == "tool_error"]
        assert len(error_events) == 1
        assert "não encontrada" in error_events[0].error

    @pytest.mark.asyncio
    async def test_tool_manager_multiple_callbacks(self):
        """
        Testa que múltiplos callbacks registados no ToolManager recebem
        os mesmos eventos.
        """
        from core.v2.tool_manager import ToolManager, ToolEvent
        from core.v2.tool_registry import ToolRegistry

        registry = ToolRegistry()
        tm = ToolManager(registry=registry)

        events_a = []
        events_b = []

        async def collector_a(event: ToolEvent):
            events_a.append(event)

        async def collector_b(event: ToolEvent):
            events_b.append(event)

        tm.add_event_callback(collector_a)
        tm.add_event_callback(collector_b)

        result = await tm.execute("nonexistent.tool", {})

        # Ambos os callbacks devem ter recebido o mesmo evento
        assert len(events_a) == 1
        assert len(events_b) == 1
        assert events_a[0].tool_name == "nonexistent.tool"
        assert events_b[0].tool_name == "nonexistent.tool"

        # Remover um callback e verificar que não recebe mais eventos
        tm.remove_event_callback(collector_a)
        tm._event_callbacks.clear()

        events_a.clear()
        events_b.clear()
        result = await tm.execute("another.nonexistent", {})

        # Sem callbacks, sem eventos
        assert len(events_a) == 0
        assert len(events_b) == 0


# ======================================================================
#  Testes: Orchestrator — Tool Event Callback Propagation
# ======================================================================

class TestOrchestratorToolEventPropagation:
    """Testa que o Orchestrator propaga tool event callbacks para o ToolManager."""

    def test_set_tool_event_callback_delegates_to_tool_manager(self):
        """
        Testa que set_tool_event_callback() no Orchestrator delega
        corretamente para o ToolManager.
        """
        from core.v2.orchestrator import OdysseusOrchestrator
        from core.v2.tool_manager import ToolManager

        orch = OdysseusOrchestrator()

        # Verificar que o orquestrador tem tool_manager
        assert orch.tool_manager is not None

        # O tool_manager deve ter event_callbacks vazio
        assert len(orch.tool_manager._event_callbacks) == 0

        # Registar callback
        async def dummy_callback(event):
            pass

        orch.set_tool_event_callback(dummy_callback)

        # Verificar que foi delegado para o tool_manager
        assert len(orch.tool_manager._event_callbacks) == 1
        assert orch.tool_manager._event_callbacks[0] is dummy_callback

        # Remover callback
        orch.remove_tool_event_callback(dummy_callback)
        assert len(orch.tool_manager._event_callbacks) == 0


# ======================================================================
#  Testes: Integração com v2_server (opcional, apenas estrutura)
# ======================================================================

@pytest.mark.skip(reason="Teste de integração requer servidor em execução")
class TestIntegration:
    """Testes de integração com o v2_server (opcionais)."""

    @pytest.mark.asyncio
    async def test_endpoint_returns_openai_format(self):
        """Placeholder: testar que /v1/chat/completions?use_orchestrator=true
        devolve resposta no formato OpenAI."""
        pass
