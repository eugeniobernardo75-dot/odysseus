"""
OrchestratorAdapter — Ponte entre OpenAI API e o Orchestrator V2
=================================================================

Converte pedidos no formato OpenAI (/v1/chat/completions) para o
Orchestrator V2 e vice-versa.

O Orchestrator não sabe que existe Odysseus (ou qualquer outra interface).
Ele processa um pedido e devolve um resultado estruturado. Cabe ao Adapter
traduzir entre os dois mundos.

Fluxo:
    messages[] (OpenAI) ─► extract user_message ─► Orchestrator.process()
        ─► OrchestratorResponse ─► OpenAI format response

Streaming:
    messages[] (OpenAI) ─► extract user_message ─► Orchestrator.process_stream()
        ─► tokens ─► SSE (data: {"delta": "..."}\n\n)

    Durante o streaming, se o Orchestrator executar ferramentas via ToolManager,
    o Adapter emite eventos SSE adicionais:

        data: {"type": "tool_start", "tool": "...", "params": {...}, "execution_id": "..."}
        data: {"type": "tool_output", "tool": "...", "result": {...}, "execution_id": "..."}
        data: {"type": "tool_error", "tool": "...", "error": "...", "execution_id": "..."}

Uso:
    adapter = OrchestratorAdapter(get_orchestrator(llm_client=...))
    result = await adapter.chat_completion({"messages": [...], "model": "..."})
    # or
    async for chunk in adapter.chat_completion_stream({"messages": [...], "model": "..."}):
        yield chunk
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

from .tool_manager import ToolEvent

logger = logging.getLogger(__name__)


# ======================================================================
#  OpenAI Helpers
# ======================================================================

def _build_openai_response(
    content: str,
    model: str = "odysseus-v2",
    finish_reason: str = "stop",
    usage: dict | None = None,
) -> dict:
    """Constrói resposta no formato OpenAI chat completion."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _build_openai_error(message: str) -> dict:
    """Constrói resposta de erro no formato OpenAI."""
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": None,
            "code": None,
        }
    }


# ======================================================================
#  Eventos SSE para ferramentas
# ======================================================================

def _format_tool_event_sse(
    event: ToolEvent,
    response_id: str,
    created: int,
    model: str,
) -> bytes:
    """
    Converte um ToolEvent num chunk SSE de tipo próprio.

    Formatos:
        tool_start:
            data: {"type":"tool_start","id": "...","object":"tool.event",
                   "tool":"file.read","params":{...},"execution_id":"...",
                   "timestamp":"..."}\n\n

        tool_output:
            data: {"type":"tool_output","id": "...","object":"tool.event",
                   "tool":"file.read","result":{...},"execution_id":"...",
                   "duration_ms":123.4,"timestamp":"..."}\n\n

        tool_error:
            data: {"type":"tool_error","id": "...","object":"tool.event",
                   "tool":"file.read","error":"...","execution_id":"...",
                   "timestamp":"..."}\n\n
    """
    payload = {
        "id": response_id,
        "object": "tool.event",
        "type": event.event_type,
        "tool": event.tool_name,
        "execution_id": event.execution_id,
        "timestamp": event.timestamp or datetime.now(timezone.utc).isoformat(),
    }

    if event.event_type == "tool_start":
        payload["params"] = event.params or {}
    elif event.event_type == "tool_output":
        payload["result"] = _serialize_result(event.result)
        if event.duration_ms is not None:
            payload["duration_ms"] = event.duration_ms
    elif event.event_type == "tool_error":
        payload["error"] = event.error or "Erro desconhecido"
        payload["params"] = event.params or {}
        if event.duration_ms is not None:
            payload["duration_ms"] = event.duration_ms

    return b"data: " + json.dumps(payload, ensure_ascii=False, default=str).encode() + b"\n\n"


def _serialize_result(result: Any) -> Any:
    """Serializa o resultado de uma ferramenta para JSON."""
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if hasattr(result, "__dict__"):
        return result.__dict__
    if isinstance(result, (str, int, float, bool, list, dict)):
        return result
    try:
        json.dumps(result)
        return result
    except (TypeError, ValueError):
        return str(result)


# ======================================================================
#  OrchestratorAdapter
# ======================================================================

class OrchestratorAdapter:
    """
    Adaptador que converte entre o formato OpenAI (/v1/chat/completions)
    e o Orchestrator V2.

    Funciona como um wrapper em torno do Orchestrator, expondo uma interface
    compatível com a API OpenAI para que o Odysseus (ou qualquer outro cliente)
    possa comunicar com o Orchestrator sem conhecer os seus detalhes internos.

    O Orchestrator não sabe da existência do Adapter nem do formato OpenAI.

    Durante o streaming, o Adapter regista um callback no ToolManager do
    Orchestrator para receber eventos de ferramentas (tool_start, tool_output,
    tool_error) e emiti-los como SSE events no stream.
    """

    def __init__(self, orchestrator, user_agent=None):
        """
        Args:
            orchestrator: Instância de OdysseusOrchestrator (ou compatível).
            user_agent: Coordenador opcional, usado sobretudo em testes.
        """
        self._orchestrator = orchestrator
        self._user_agent = user_agent
        self._available = orchestrator is not None

    # ── Propriedades ──

    @property
    def orchestrator(self):
        return self._orchestrator

    @property
    def is_available(self) -> bool:
        return self._available and self._orchestrator is not None

    def _get_user_agent(self):
        if self._user_agent is None:
            from .user_agent_config import get_user_agent
            self._user_agent = get_user_agent()
        return self._user_agent

    # ── Conversão de formato ──

    def _extract_user_message(self, messages: list[dict]) -> str:
        """
        Extrai a última mensagem do utilizador do formato OpenAI.

        Percorre as mensagens da mais recente para a mais antiga
        e retorna o content da primeira mensagem com role='user'.
        """
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # content pode ser string ou lista de partes (multimodal)
                if isinstance(content, list):
                    # Extrair texto de partes multimodais
                    text_parts = [
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    return " ".join(filter(None, text_parts))
                return str(content) if content else ""
        return ""

    def _build_context(self, messages: list[dict]) -> dict:
        """
        Constrói contexto a partir das mensagens OpenAI para passar
        ao Orchestrator.

        Inclui:
        - messages: lista completa de mensagens (para o LLM)
        - system_prompt: primeira mensagem system, se existir
        - context_window: as últimas N mensagens (para memory)
        """
        context = {
            "messages": messages,
            "source": "openai_api",
        }

        # Extrair system prompt se existir
        for msg in messages:
            if msg.get("role") == "system":
                context["system_prompt"] = msg.get("content", "")
                break

        return context

    def _estimate_usage(self, messages: list[dict], response: str) -> dict:
        """
        Estimativa simples de uso de tokens.
        Útil quando o Orchestrator não devolve métricas de tokens.
        """
        prompt_text = json.dumps(messages, ensure_ascii=False)
        # Estimativa grosseira: ~4 chars por token
        prompt_tokens = max(1, len(prompt_text) // 4)
        completion_tokens = max(1, len(response) // 4)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    # ── API Principais ──

    async def chat_completion(self, body: dict) -> dict:
        """
        Processa um pedido no formato OpenAI e devolve resposta no formato OpenAI.

        Args:
            body: Dicionário no formato OpenAI:
                {
                    "messages": [{"role": "user", "content": "..."}],
                    "model": "odysseus-v2",
                    ...
                }

        Returns:
            Resposta no formato OpenAI chat completion.
        """
        if not self.is_available:
            return _build_openai_error("Orchestrator V2 não disponível.")

        messages = body.get("messages", [])
        model = body.get("model", "odysseus-v2")

        # Extrair mensagem do utilizador
        user_message = self._extract_user_message(messages)
        if not user_message:
            return _build_openai_error(
                "Nenhuma mensagem do utilizador encontrada. "
                "É necessário pelo menos uma mensagem com role='user'."
            )

        # Construir contexto — incluir o modelo pedido (modo manual)
        context = self._build_context(messages)
        if model and str(model).strip() not in ("", "odysseus-v2"):
            context["model"] = str(model).strip()

        # O UserAgent é a única entrada de coordenação; o Orchestrator fica
        # atrás dele como processador interno.
        try:
            result = await self._get_user_agent().process(
                question=user_message,
                context=context,
                approval_id=body.get("approval_id"),
            )

            # Converter resposta para formato OpenAI
            response_content = result.message or ""

            # Estimar uso de tokens
            usage = self._estimate_usage(messages, response_content)

            # Incluir metadados do Orchestrator quando disponíveis
            if result.metadata:
                if "llm_used" in result.metadata:
                    usage["llm_used"] = result.metadata["llm_used"]
                if "parallel_agents" in result.metadata:
                    usage["parallel_agents"] = result.metadata["parallel_agents"]

            response_payload = _build_openai_response(
                content=response_content,
                model=model,
                usage=usage,
            )
            response_payload["coordinated_by"] = "useragent"
            if result.metadata and result.metadata.get("knowledge_sources"):
                response_payload["knowledge_sources"] = result.metadata[
                    "knowledge_sources"
                ]
            return response_payload

        except Exception as e:
            logger.error("Erro no OrchestratorAdapter.chat_completion: %s", e, exc_info=True)
            return _build_openai_error(str(e))

    async def chat_completion_stream(
        self,
        body: dict,
    ) -> AsyncGenerator[bytes, None]:
        """
        Processa um pedido em streaming e devolve SSE tokens no formato OpenAI,
        incluindo eventos de ferramentas (tool_start, tool_output, tool_error).

        Quando o Orchestrator executa ferramentas via ToolManager, o Adapter
        emite eventos SSE entre os tokens de texto:

            data: {"type":"tool_start","tool":"file.read","params":{...},...}\n\n
            data: {"type":"tool_output","tool":"file.read","result":{...},...}\n\n
            data: {"type":"tool_error","tool":"file.read","error":"...",...}\n\n

        Yields:
            bytes: Chunks SSE no formato:
                data: {"id": "...", "object": "chat.completion.chunk", ...}\n\n
                data: {"type": "tool_start", ...}\n\n
                data: {"type": "tool_output", ...}\n\n
                data: [DONE]\n\n
        """
        if not self.is_available:
            error_chunk = _build_openai_error("Orchestrator V2 não disponível.")
            yield b"data: " + json.dumps(error_chunk, ensure_ascii=False).encode() + b"\n\n"
            yield b"data: [DONE]\n\n"
            return

        messages = body.get("messages", [])
        model = body.get("model", "odysseus-v2")

        user_message = self._extract_user_message(messages)
        if not user_message:
            error_chunk = _build_openai_error(
                "Nenhuma mensagem do utilizador encontrada."
            )
            yield b"data: " + json.dumps(error_chunk, ensure_ascii=False).encode() + b"\n\n"
            yield b"data: [DONE]\n\n"
            return

        # Só passa contexto com modelo quando há uma escolha manual — mantém
        # compatibilidade com mocks/orquestradores que não aceitam o kwarg.
        stream_context = None
        if model and str(model).strip() not in ("", "odysseus-v2"):
            stream_context = self._build_context(messages)
            stream_context["model"] = str(model).strip()

        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        first = True

        # ── Queue para eventos de ferramentas ──
        # O ToolManager do Orchestrator notifica o Adapter via callback.
        # Este callback coloca os eventos numa fila asyncio, que é
        # consumida entre tokens de texto no loop de streaming.
        tool_event_queue: asyncio.Queue[ToolEvent] = asyncio.Queue()

        async def _tool_event_handler(event: ToolEvent) -> None:
            """Callback registado no ToolManager do Orchestrator."""
            try:
                await tool_event_queue.put(event)
            except Exception as e:
                logger.debug(
                    "OrchestratorAdapter: erro ao colocar evento %s/%s na queue: %s",
                    event.event_type, event.tool_name, e,
                )

        # Registar callback no ToolManager através do Orchestrator
        try:
            self._orchestrator.set_tool_event_callback(_tool_event_handler)
        except Exception as e:
            logger.debug(
                "OrchestratorAdapter: não foi possível registar tool event callback: %s",
                e,
            )

        try:
            full_response = ""

            async def _drain_tool_events():
                """Emit tool events pendentes na queue como SSE."""
                while not tool_event_queue.empty():
                    try:
                        event = tool_event_queue.get_nowait()
                        yield _format_tool_event_sse(
                            event, response_id, created, model,
                        )
                    except asyncio.QueueEmpty:
                        break

            stream_iter = self._get_user_agent().process_stream(
                user_message, context=stream_context
            )
            async for token in stream_iter:
                # ── Antes de cada token de texto, drenar eventos pendentes ──
                async for sse_chunk in _drain_tool_events():
                    yield sse_chunk

                if token == "__DONE__":
                    break

                full_response += token

                # Chunk SSE no formato OpenAI
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "coordinated_by": "useragent",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant" if first else "",
                                "content": token,
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                first = False
                yield b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\n\n"

            # Drenar eventos restantes (após o stream acabar)
            async for sse_chunk in _drain_tool_events():
                yield sse_chunk

            # Chunk final com finish_reason
            final_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "coordinated_by": "useragent",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield b"data: " + json.dumps(final_chunk, ensure_ascii=False).encode() + b"\n\n"
            yield b"data: [DONE]\n\n"

        except Exception as e:
            logger.error("Erro no OrchestratorAdapter.chat_completion_stream: %s", e, exc_info=True)
            error_chunk = _build_openai_error(str(e))
            yield b"data: " + json.dumps(error_chunk, ensure_ascii=False).encode() + b"\n\n"
            yield b"data: [DONE]\n\n"

        finally:
            # Limpar callback para não vazar listeners entre pedidos
            try:
                self._orchestrator.remove_tool_event_callback(_tool_event_handler)
            except Exception:
                pass

    # ── Informação ──

    def get_info(self) -> dict:
        """Retorna informação sobre o estado do Adapter e Orchestrator."""
        if not self.is_available:
            return {"available": False, "error": "Orchestrator não disponível"}

        try:
            agents = self._orchestrator.agents.list_agents()
            return {
                "available": True,
                "agents": [a["name"] for a in agents],
                "agent_count": len(agents),
            }
        except Exception as e:
            return {"available": True, "error": str(e)}


# ======================================================================
#  Factory functions
# ======================================================================

def create_adapter_from_config(config) -> OrchestratorAdapter:
    """
    Cria um OrchestratorAdapter a partir de uma BridgeConfig.

    Inicializa o LLMClient e o Orchestrator, e retorna o Adapter pronto a usar.

    Args:
        config: BridgeConfig com as definições de ligação.

    Returns:
        OrchestratorAdapter pronto a usar.
    """
    try:
        from .orchestrator import get_orchestrator
        from .llm_client import LLMClient, LLMConfig

        # Configurar LLM
        llm_config = LLMConfig.from_env()
        if config.ollama_url:
            llm_config.base_url = config.ollama_url.rstrip("/")
        if config.llm_model:
            llm_config.model = config.llm_model
        llm_config.timeout = config.llm_timeout
        llm_config.temperature = config.llm_temperature
        llm_config.max_tokens = config.llm_max_tokens

        llm_client = LLMClient(config=llm_config)

        # Inicializar Orchestrator
        orchestrator = get_orchestrator(llm_client=llm_client)

        adapter = OrchestratorAdapter(orchestrator)

        agents = orchestrator.agents.list_agents()
        logger.info(
            "✅ OrchestratorAdapter criado. Agentes: %s",
            ", ".join(a["name"] for a in agents),
        )

        return adapter

    except Exception as e:
        logger.warning("⚠️  Não foi possível criar OrchestratorAdapter: %s", e)
        return OrchestratorAdapter(orchestrator=None)
