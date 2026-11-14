from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import logging
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse

# ── Caminhos ──
# Adiciona o diretório raiz do Odysseus ao PYTHONPATH para importar os módulos V2
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv

load_dotenv(os.path.join(_project_root, ".env"), override=False)

from core.v2.constants import V2_BRIDGE_HOST, V2_BRIDGE_PORT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("v2-server")

# ======================================================================
#  Config
# ======================================================================

class BridgeConfig:
    """Configuração do servidor bridge V2."""

    def __init__(
        self,
        *,
        host: str = V2_BRIDGE_HOST,
        port: int = V2_BRIDGE_PORT,
        ollama_url: str = "",
        llm_model: str = "",
        llm_timeout: float = 120.0,
        llm_temperature: float = 0.7,
        llm_max_tokens: int = 4096,
    ):
        self.host = host
        self.port = port
        self.ollama_url = ollama_url
        self.llm_model = llm_model
        self.llm_timeout = llm_timeout
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.started_at = datetime.now(timezone.utc)
        self.request_count = 0

    @property
    def uptime_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.started_at).total_seconds()


# ======================================================================
#  LLM Client Wrapper
# ======================================================================

class V2LLMWrapper:
    """
    Wrapper que tenta usar o LLMClient V2 do Odysseus.
    Se não conseguir importar (ex: dependências em falta), cai para
    chamadas HTTP diretas ao Ollama/OpenCode.
    """

    def __init__(self, config: BridgeConfig):
        self.config = config
        self._v2_client = None
        self._v2_available = False
        self._httpx_client = None
        self._init_v2()

    def _init_v2(self):
        """Tenta inicializar o cliente V2 nativo do Odysseus."""
        try:
            # Tenta importar o LLMClient V2
            from core.v2.llm_client import LLMClient, LLMConfig, get_llm_client

            # Configurar baseado nas opções fornecidas
            llm_config = LLMConfig.from_env()
            if self.config.ollama_url:
                llm_config.base_url = self.config.ollama_url.rstrip("/")
            if self.config.llm_model:
                llm_config.model = self.config.llm_model
            llm_config.timeout = self.config.llm_timeout
            llm_config.temperature = self.config.llm_temperature
            llm_config.max_tokens = self.config.llm_max_tokens

            self._v2_client = LLMClient(config=llm_config)
            self._v2_available = True
            logger.info("✅ LLMClient V2 inicializado com sucesso!")
            logger.info("   Base URL: %s", llm_config.base_url)
            logger.info("   Model:    %s", llm_config.model)
        except Exception as e:
            logger.warning(
                "⚠️  Não foi possível inicializar o LLMClient V2: %s", e
            )
            logger.warning("   A usar fallback HTTP direto para Ollama.")
            self._v2_available = False

    async def _get_httpx(self):
        """Obtém (ou cria) um cliente HTTPX."""
        if self._httpx_client is None:
            import httpx
            self._httpx_client = httpx.AsyncClient(timeout=30.0)
        return self._httpx_client

    async def list_models(self) -> list[dict]:
        """Lista os modelos disponíveis no formato OpenAI."""
        models = []

        # Tentar V2 nativo primeiro
        if self._v2_available:
            try:
                v2_models = await self._v2_client.list_models()
                for m in v2_models:
                    models.append({
                        "id": m,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "odysseus-v2",
                    })
                if models:
                    return models
            except Exception as e:
                logger.debug("V2 list_models failed: %s", e)

        # Fallback: HTTP direto ao Ollama
        base_url = self.config.ollama_url or "http://localhost:11434"
        try:
            client = await self._get_httpx()
            tags_url = base_url.rstrip("/") + "/api/tags"
            resp = await client.get(tags_url, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                for m in data.get("models", []):
                    name = m.get("name", "")
                    if name:
                        models.append({
                            "id": name,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "ollama",
                        })
        except Exception as e:
            logger.warning("Falha ao listar modelos via HTTP: %s", e)

        # Se não há modelos, pelo menos mostrar o configurado
        if not models and self.config.llm_model:
            models.append({
                "id": self.config.llm_model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "odysseus-v2",
            })

        return models

    async def chat_completion(
        self,
        body: dict,
    ) -> dict:
        """
        Processa um pedido de chat completion no formato OpenAI.
        Retorna resposta no formato OpenAI.
        """
        messages = body.get("messages", [])
        model = body.get("model", self.config.llm_model or "llama3.2")
        temperature = body.get("temperature", self.config.llm_temperature)
        max_tokens = body.get("max_tokens", self.config.llm_max_tokens)
        stream = body.get("stream", False)

        # Tentar V2 nativo
        if self._v2_available:
            try:
                response = await self._v2_client.chat(
                    messages=messages,
                    model=model or None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )

                if response.success:
                    return _build_openai_response(
                        content=response.content,
                        model=response.model or model,
                        finish_reason=response.finish_reason,
                        usage=response.usage or {},
                    )
                else:
                    # Erro: devolver como resposta de erro OpenAI
                    logger.warning("V2 chat failed: %s", response.error)
                    return _build_openai_error(response.error or "Erro desconhecido")
            except Exception as e:
                logger.warning("V2 chat exception: %s", e)
                return _build_openai_error(str(e))

        # Fallback HTTP direto ao Ollama
        return await self._fallback_chat(messages, model, temperature, max_tokens)

    async def chat_completion_stream(
        self,
        body: dict,
    ) -> AsyncGenerator[bytes, None]:
        """
        Processa um pedido de chat completion em streaming.
        Yields SSE chunks no formato OpenAI.
        """
        messages = body.get("messages", [])
        model = body.get("model", self.config.llm_model or "llama3.2")
        temperature = body.get("temperature", self.config.llm_temperature)
        max_tokens = body.get("max_tokens", self.config.llm_max_tokens)

        # Tentar V2 nativo
        if self._v2_available:
            try:
                response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                created = int(time.time())
                first = True

                async for token in self._v2_client.chat_stream(
                    messages=messages,
                    model=model or None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    if token == "__DONE__":
                        # Enviar chunk final [DONE]
                        yield b"data: [DONE]\n\n"
                        return

                    chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
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

                # Chunk final
                final_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
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
                return
            except Exception as e:
                logger.warning("V2 stream error: %s", e)
                error_chunk = {
                    "error": {"message": str(e), "type": "server_error"},
                }
                yield b"data: " + json.dumps(error_chunk, ensure_ascii=False).encode() + b"\n\n"
                yield b"data: [DONE]\n\n"
                return

        # Fallback HTTP direto ao Ollama
        async for chunk in self._fallback_stream(messages, model, temperature, max_tokens):
            yield chunk

    # ── Fallbacks HTTP diretos ao Ollama ──

    async def _fallback_chat(self, messages, model, temperature, max_tokens) -> dict:
        """Fallback: chamada HTTP direta ao Ollama."""
        base_url = self.config.ollama_url or "http://localhost:11434"
        chat_url = base_url.rstrip("/") + "/api/chat"

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            client = await self._get_httpx()
            resp = await client.post(chat_url, json=payload, timeout=120.0)
            if resp.status_code == 200:
                data = resp.json()
                content = ""
                msg = data.get("message", {})
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                return _build_openai_response(
                    content=content or "",
                    model=model,
                    usage={
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                        "total_tokens": (data.get("prompt_eval_count", 0)
                                         + data.get("eval_count", 0)),
                    },
                )
            else:
                return _build_openai_error(
                    f"Ollama HTTP {resp.status_code}: {resp.text[:300]}"
                )
        except Exception as e:
            return _build_openai_error(str(e))

    async def _fallback_stream(self, messages, model, temperature, max_tokens) -> AsyncGenerator[bytes, None]:
        """Fallback: streaming HTTP direto ao Ollama."""
        base_url = self.config.ollama_url or "http://localhost:11434"
        chat_url = base_url.rstrip("/") + "/api/chat"

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            first = True

            client = await self._get_httpx()
            async with client.stream("POST", chat_url, json=payload) as resp:
                if resp.status_code != 200:
                    error_chunk = {
                        "error": {"message": f"Ollama HTTP {resp.status_code}"},
                    }
                    yield b"data: " + json.dumps(error_chunk).encode() + b"\n\n"
                    yield b"data: [DONE]\n\n"
                    return

                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk_data = json.loads(line)
                    except Exception:
                        continue

                    content = ""
                    msg = chunk_data.get("message", {})
                    if isinstance(msg, dict):
                        content = msg.get("content", "")

                    if content:
                        chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant" if first else "",
                                        "content": content,
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        first = False
                        yield b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\n\n"

                    if chunk_data.get("done"):
                        break

            # Chunk final
            final = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield b"data: " + json.dumps(final, ensure_ascii=False).encode() + b"\n\n"
            yield b"data: [DONE]\n\n"

        except Exception as e:
            err = {"error": {"message": str(e)}}
            yield b"data: " + json.dumps(err).encode() + b"\n\n"
            yield b"data: [DONE]\n\n"

# ======================================================================
#  V2 Orchestrator Wrapper (com agentes, memória, segurança)
# ======================================================================

class V2OrchestratorWrapper:
    """
    Wrapper para as rotas legadas /v2/chat (orquestrador V2 completo).

    Inicialização LAZY: o orquestrador (com LLM client, memória, chromadb) só é
    construído no primeiro acesso, para não duplicar a instanciação que o
    OrchestratorAdapter já faz no arranque do bridge (o caminho RC1).

    Usado pelo endpoint /v2/chat para responder com o pipeline completo:
    Analisar -> Planear -> Executar (agentes) -> Validar -> Melhorar
    """

    def __init__(self, config: BridgeConfig):
        self.config = config
        self._orchestrator = None
        self._orchestrator_available = False
        self._initialized = False
        self._init_lock = threading.Lock()

    def _ensure_init(self):
        """
        Inicializa o orquestrador na primeira utilização.

        Protegido por lock para que pedidos concorrentes não inicializem duas
        vezes o mesmo orquestrador. Se a inicialização falhar (ex: LLM ainda a
        arrancar), permite tentar novamente no próximo acesso — nunca fica
        preso num estado "morto".
        """
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            try:
                # Importar o cliente LLM
                from core.v2.llm_client import LLMClient, LLMConfig

                llm_config = LLMConfig.from_env()
                if self.config.ollama_url:
                    llm_config.base_url = self.config.ollama_url.rstrip("/")
                if self.config.llm_model:
                    llm_config.model = self.config.llm_model
                llm_config.timeout = self.config.llm_timeout
                llm_config.temperature = self.config.llm_temperature
                llm_config.max_tokens = self.config.llm_max_tokens

                llm_client = LLMClient(config=llm_config)

                # Importar e inicializar o orquestrador
                from core.v2.orchestrator import get_orchestrator
                self._orchestrator = get_orchestrator(llm_client=llm_client)

                # Verificar se agentes estão registados
                agents = self._orchestrator.agents.list_agents()
                agent_names = [a["name"] for a in agents]

                self._orchestrator_available = True
                self._initialized = True
                logger.info("✅ Orquestrador V2 inicializado (lazy) com sucesso!")
                logger.info("   Agentes disponíveis: %s", ", ".join(agent_names))
            except Exception as e:
                logger.warning(
                    "⚠️  Não foi possível inicializar o orquestrador V2: %s", e
                )
                self._orchestrator_available = False

    @property
    def is_available(self) -> bool:
        self._ensure_init()
        return self._orchestrator_available

    async def chat(self, question: str, context: Optional[dict] = None) -> dict:
        """
        Processa uma mensagem através do orquestrador V2 completo.

        Args:
            question: Mensagem do utilizador.
            context: Contexto adicional (opcional).

        Returns:
            Resposta estruturada com:
            - response: texto da resposta
            - agent_results: resultados de agentes usados
            - plan: plano executado (se aplicável)
            - suggested_memory: sugestões de memória
        """
        self._ensure_init()
        if not self._orchestrator_available:
            return {
                "success": False,
                "response": "Orquestrador V2 não disponível. Usa /v1/chat/completions como alternativa.",
                "error": "orchestrator_not_available",
            }

        try:
            from core.v2.api_bridge import process_request
            result = await process_request(
                question=question,
                context=context or {},
            )
            return result
        except Exception as e:
            logger.error("Erro no orquestrador V2: %s", e, exc_info=True)
            return {
                "success": False,
                "response": f"Erro ao processar pedido: {e}",
                "error": str(e),
            }

    async def chat_stream(
        self,
        question: str,
        context: Optional[dict] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Processa uma mensagem em streaming através do orquestrador V2.

        Yields tokens individuais da resposta, terminando com __DONE__.
        """
        self._ensure_init()
        if not self._orchestrator_available:
            yield "Orquestrador V2 não disponível. Usa /v1/chat/completions como alternativa."
            yield "__DONE__"
            return

        try:
            from core.v2.user_agent_config import get_user_agent

            async for token in get_user_agent().process_stream(
                question, context=context
            ):
                yield token
        except Exception as e:
            logger.error("Erro no streaming do orquestrador: %s", e)
            yield f"Erro: {e}"
            yield "__DONE__"

    def get_info(self) -> dict:
        """Retorna info sobre o estado do orquestrador."""
        self._ensure_init()
        info = {
            "available": self._orchestrator_available,
        }

        if self._orchestrator_available:
            try:
                agents = self._orchestrator.agents.list_agents()
                info["agents"] = [a["name"] for a in agents]
                info["agent_count"] = len(agents)
                info["memory_entries"] = len(
                    self._orchestrator.memory.recall_all_permanent()
                )
            except Exception as e:
                info["error"] = str(e)

        return info


# ======================================================================
#  Helpers OpenAI
# ======================================================================

def _build_openai_response(
    content: str,
    model: str = "unknown",
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


async def _read_json_body(request: Request) -> Optional[dict]:
    """Lê o corpo JSON de forma robusta (UTF-8, fallback Latin-1).

    Alguns clientes enviam o corpo em Latin-1/cp1252 (ex.: texto acentuado
    em português), o que fazia `await request.json()` rebentar com
    UnicodeDecodeError -> 500. Aqui tentamos UTF-8 primeiro e, se falhar,
    Latin-1 (que nunca falha). Se mesmo assim não for JSON válido,
    devolvemos None para o handler responder 400 limpo.
    """
    raw = await request.body()
    if not raw:
        return None
    text = None
    for encoding in ("utf-8", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        return None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ======================================================================
#  FastAPI App
# ======================================================================

def create_app(config: BridgeConfig | None = None) -> FastAPI:
    """Cria e configura a aplicação FastAPI."""
    if config is None:
        config = BridgeConfig()

    app = FastAPI(
        title="Odysseus V2 Bridge",
        description="Bridge API OpenAI-compatível para o Odysseus V2",
        version="2.0.0",
    )

    # CORS — permite qualquer origem (útil para Docker/proxies)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Instância do wrapper LLM
    llm = V2LLMWrapper(config)

    # Instância do orquestrador V2 (com agentes, memória, segurança)
    orch = V2OrchestratorWrapper(config)

    # ── OrchestratorAdapter (substituto do V2LLMWrapper, usa o Orchestrator em vez do LLM puro) ──
    orch_adapter = None
    try:
        from core.v2.orchestrator_adapter import create_adapter_from_config
        orch_adapter = create_adapter_from_config(config)
        if orch_adapter and orch_adapter.is_available:
            logger.info("✅ OrchestratorAdapter inicializado com sucesso!")
        else:
            logger.warning("⚠️  OrchestratorAdapter não disponível.")
            orch_adapter = None
    except Exception as e:
        logger.warning("⚠️  OrchestratorAdapter não pôde ser criado: %s", e)
        orch_adapter = None

    # Aprovação do mesmo processo do bridge, protegida por token dedicado.
    # Sem token configurado, falha fechada; nunca transforma localhost em bypass.
    approval_token = os.getenv("ODYSSEUS_V2_APPROVAL_TOKEN", "").strip()

    def _require_approval_token(authorization: str | None) -> None:
        expected = approval_token
        supplied = (authorization or "").strip()
        if not expected or supplied != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="Aprovação autenticada necessária.")

    @app.get("/v2/approvals/pending")
    async def bridge_pending_approvals(authorization: str | None = Header(default=None)):
        _require_approval_token(authorization)
        if orch_adapter is None or not orch_adapter.is_available:
            raise HTTPException(status_code=503, detail="Orchestrator indisponível.")
        orchestrator = orch_adapter.orchestrator
        pending = [
            {
                **approval,
                "plan_id": orchestrator.get_pending_plan_id(approval["approval_id"]),
            }
            for approval in orchestrator.safety.get_pending()
        ]
        return {"success": True, "count": len(pending), "pending": pending}

    @app.post("/v2/approve/{approval_id}")
    async def bridge_approve(approval_id: str, authorization: str | None = Header(default=None)):
        _require_approval_token(authorization)
        if orch_adapter is None or not orch_adapter.is_available:
            raise HTTPException(status_code=503, detail="Orchestrator indisponível.")
        orchestrator = orch_adapter.orchestrator
        approval = orchestrator.safety.get_approval(approval_id)
        if approval is None:
            raise HTTPException(status_code=404, detail="Aprovação não encontrada.")
        if not orchestrator.has_pending_plan(approval_id):
            raise HTTPException(
                status_code=409,
                detail="Aprovação não tem um plano de ferramenta pendente para retomar.",
            )
        if approval["status"] == "pending":
            success, message = orchestrator.safety.approve(approval_id)
            if not success:
                raise HTTPException(
                    status_code=409,
                    detail={"message": message, "approval_id": approval_id},
                )
        elif approval["status"] == "approved":
            message = "Aprovação já concedida; a retomar o plano pendente."
        else:
            raise HTTPException(
                status_code=409,
                detail={"message": "A aprovação já foi negada.", "approval_id": approval_id},
            )

        try:
            result = await orchestrator.resume_approved_plan(approval_id)
        except (KeyError, PermissionError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        executions = []
        for agent_result in result.agent_results:
            details = agent_result.details or {}
            tool_result = details.get("tool_result")
            if tool_result is None:
                continue
            metadata = tool_result.get("metadata") or {}
            data = tool_result.get("data")
            return_code = metadata.get("return_code")
            if return_code is None and isinstance(data, dict):
                return_code = data.get("return_code")
            executions.append({
                "tool": details.get("tool"),
                "success": agent_result.success,
                "return_code": return_code,
                "isolated": metadata.get("isolated"),
                "isolation_backend": metadata.get("isolation_backend"),
            })

        return {
            "success": result.plan is not None and result.plan.status == "completed",
            "approval_id": approval_id,
            "status": "approved",
            "message": message,
            "resumed": True,
            "plan_id": result.plan.id if result.plan else None,
            "plan_status": result.plan.status if result.plan else None,
            "executions": executions,
            "response": result.message,
        }

    @app.post("/v2/deny/{approval_id}")
    async def bridge_deny(approval_id: str, authorization: str | None = Header(default=None)):
        _require_approval_token(authorization)
        if orch_adapter is None or not orch_adapter.is_available:
            raise HTTPException(status_code=503, detail="Orchestrator indisponível.")
        orchestrator = orch_adapter.orchestrator
        success, message = orchestrator.safety.deny(approval_id)
        if not success:
            raise HTTPException(status_code=409, detail={"message": message, "approval_id": approval_id})
        plan_id = orchestrator.get_pending_plan_id(approval_id)
        orchestrator.discard_pending_plan(approval_id)
        return {
            "success": True,
            "approval_id": approval_id,
            "status": "denied",
            "message": message,
            "resumed": False,
            "plan_id": plan_id,
            "executions": [],
        }

    # ── Rotas ──

    @app.get("/")
    async def root():
        """Página de estado HTML."""
        models = await llm.list_models()
        orch_info = orch.get_info()
        model_list = "\n".join(
            f'        <div class="model-item"><code>{m["id"]}</code>'
            f' <span class="owner">{m["owned_by"]}</span></div>'
            for m in models[:20]
        )

        return HTMLResponse(f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Odysseus V2 Bridge — Status</title>
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a1a; color: #e2e2ec; line-height:1.6;
            min-height:100vh; display:flex; align-items:center; justify-content:center;
        }}
        .card {{
            background: #12122a; border:1px solid #2d2d50; border-radius:16px;
            padding:40px; max-width:640px; width:90%; margin:40px auto;
        }}
        h1 {{ font-size:24px; margin-bottom:8px; }}
        h1 span {{ color:#6c5ce7; }}
        .badge {{
            display:inline-block; padding:4px 12px; border-radius:999px;
            font-size:12px; font-weight:600; margin-bottom:16px;
        }}
        .badge.online {{ background:rgba(0,184,148,0.15); color:#55efc4; }}
        .meta {{ color:#7c7c96; font-size:14px; margin-bottom:24px; }}
        .meta strong {{ color:#e2e2ec; }}
        .section {{ margin-top:24px; }}
        .section h3 {{ font-size:14px; color:#7c7c96; text-transform:uppercase;
                       letter-spacing:0.5px; margin-bottom:12px; }}
        .endpoint {{ display:flex; align-items:center; gap:8px;
                     padding:8px 12px; background:#1a1a34; border-radius:8px;
                     font-family:monospace; font-size:13px; margin-bottom:6px; }}
        .endpoint .method {{ color:#6c5ce7; font-weight:600; }}
        .endpoint .path {{ color:#e2e2ec; }}
        .endpoint .desc {{ color:#7c7c96; margin-left:auto; font-size:11px; }}
        .model-item {{ display:flex; align-items:center; gap:8px;
                       padding:6px 0; border-bottom:1px solid #1a1a34; }}
        .model-item code {{ font-size:13px; }}
        .model-item .owner {{ font-size:11px; color:#7c7c96; }}
        .models-more {{ color:#7c7c96; font-size:12px; margin-top:8px; }}
        .footer {{ margin-top:24px; padding-top:16px; border-top:1px solid #2d2d50;
                   font-size:12px; color:#55556e; text-align:center; }}
        .footer a {{ color:#6c5ce7; text-decoration:none; }}
    </style>
</head>
<body>
    <div class="card">
        <div class="badge online">● Online</div>
        <h1>Odysseus <span>V2</span> Bridge</h1>
        <p class="meta">
            A correr há <strong>{config.uptime_seconds:.0f}s</strong> ·
            Pedidos: <strong>{config.request_count}</strong> ·
            Modelos: <strong>{len(models)}</strong>
        </p>

        <div class="section">
            <h3>Endpoints</h3>
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/v1/models</span>
                <span class="desc">Listar modelos</span>
            </div>
            <div class="endpoint">
                <span class="method">POST</span>
                <span class="path">/v1/chat/completions</span>
                <span class="desc">Chat completion</span>
            </div>
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/v2/status</span>
                <span class="desc">Estado do servidor</span>
            </div>
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/v2/config</span>
                <span class="desc">Configuração</span>
            </div>
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/v2/chat?q=...</span>
                <span class="desc">Chat com orquestrador + agentes</span>
            </div>
            <div class="endpoint">
                <span class="method">POST</span>
                <span class="path">/v2/chat</span>
                <span class="desc">Chat com orquestrador + agentes</span>
            </div>
            <div class="endpoint">
                <span class="method">POST</span>
                <span class="path">/v2/chat/stream</span>
                <span class="desc">Streaming do orquestrador</span>
            </div>
        </div>

        <div class="section">
            <h3>Orquestrador</h3>
            <p style="margin-bottom:8px;">
                <span class="badge online" style="background:{"rgba(0,184,148,0.15)" if orch_info.get("available") else "rgba(255,107,107,0.15)"}; color:{"#55efc4" if orch_info.get("available") else "#ff6b6b"};">
                    {"● Online" if orch_info.get("available") else "● Offline"}
                </span>
            </p>
            {f'<p class="meta" style="margin-bottom:8px;">Agentes: <strong>{", ".join(orch_info.get("agents", []))}</strong></p>' if orch_info.get("agents") else ""}
            {f'<p class="meta">Memória: <strong>{orch_info.get("memory_entries", 0)}</strong> entradas</p>' if orch_info.get("available") else ""}
        </div>

        <div class="section">
            <h3>Modelos disponíveis ({len(models)})</h3>
            {model_list}
            {f'<div class="models-more">... e mais {len(models)-20}</div>' if len(models) > 20 else ''}
        </div>

        <div class="footer">
            Registado na interface Odysseus como modelo "Odysseus V2".
            <br>
            Bridge: <code style="color:#6c5ce7">http://localhost:{config.port}</code>
        </div>
    </div>
</body>
</html>""")

    @app.get("/v1/models")
    async def list_models():
        """Lista modelos disponíveis (formato OpenAI).

        Inclui sempre o modelo virtual "odysseus-v2" (Orchestrator V2), para que
        o picker da interface Odysseus o encontre mesmo sem modelos Ollama
        carregados no host.
        """
        models = await llm.list_models()
        if not any(m.get("id") == "odysseus-v2" for m in models):
            models.insert(0, {
                "id": "odysseus-v2",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "odysseus-v2",
            })
        return {"object": "list", "data": models}

    # --- Endpoints de descoberta/probe usados pela app principal ---
    # A app principal (model_discovery / model_context) sonda o bridge para
    # identificar o tipo de servidor (LM Studio /api/v1/models, llama.cpp
    # /props e /slots, e a base /v1). Responder 200 com payloads neutros
    # elimina os 404s repetidos nos logs sem falsear a deteção: como nao
    # devolvemos os marcadores (key/architecture, total_slots...), o bridge
    # continua a ser classificado como servidor OpenAI-compatível.

    @app.get("/api/v1/models")
    async def list_models_lmstudio():
        """Probe LM Studio (/api/v1/models) — lista vazia (não é LM Studio)."""
        return {"models": []}

    @app.get("/props")
    async def llama_props():
        """Probe llama.cpp (/props) — payload vazio (não é llama-server)."""
        return {}

    @app.get("/slots")
    async def llama_slots():
        """Probe llama.cpp (/slots) — sem slots (não é llama-server)."""
        return []

    @app.get("/v1")
    async def v1_root():
        """Base OpenAI-compatível (/v1) — responde 200 a sondagens de base."""
        return {"object": "list", "data": []}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        use_orchestrator: bool = Query(
            True,
            description="Se true (padrão), usa o Orchestrator V2 (agentes+memória+"
                        "segurança+skills+aprendizagem). Se false, força o LLM puro.",
        ),
    ):
        """Chat completion no formato OpenAI.

        Desde o RC1, o caminho padrão é o OrchestratorAdapter — o cérebro do
        Odysseus V2. Memória, seleção de agentes, execução de ferramentas,
        skills, aprendizagem e streaming passam todos pelo Orchestrator; o LLM
        é apenas o motor de inferência.

        Se o Orchestrator não estiver disponível (ex: LLM inacessível), degrada
        para o LLM puro com um aviso nos logs — a interface nunca fica sem
        resposta. `use_orchestrator=false` força o modo LLM puro (debug).

        Query params:
            use_orchestrator: bool — usar Orchestrator V2 (padrão) em vez de LLM puro
        """
        config.request_count += 1
        body = await _read_json_body(request)
        if body is None:
            return JSONResponse(status_code=400, content=_build_openai_error(
                "Corpo do pedido não é JSON válido (encoding inválido)."
            ))
        stream = body.get("stream", False)

        # ── Modo Orchestrator (padrão — cérebro do Odysseus V2) ──
        if use_orchestrator and orch_adapter is not None and orch_adapter.is_available:
            if stream:
                return StreamingResponse(
                    orch_adapter.chat_completion_stream(body),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )

            result = await orch_adapter.chat_completion(body)
            if "error" in result:
                return JSONResponse(status_code=400, content=result)
            return result

        if use_orchestrator:
            # Orquestrador pedido mas indisponível → degrada para LLM puro.
            logger.warning(
                "⚠️  Orchestrator V2 não disponível — a responder em modo degradado "
                "(LLM puro). Verifica os logs do servidor."
            )

        # ── Modo LLM puro (degradado quando o Orchestrator falha, ou debug) ──
        # Em modo degradado (orchestrator indisponível mas pedido), adiciona o
        # header X-Odysseus-Mode: degraded para a interface/UI conseguir
        # distinguir uma resposta sem o cérebro V2 — o fallback nunca mascara
        # uma falha do Orchestrator em silêncio.
        degraded_headers = {
            "X-Odysseus-Mode": "degraded",
        } if use_orchestrator else None

        if stream:
            return StreamingResponse(
                llm.chat_completion_stream(body),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    **(degraded_headers or {}),
                },
            )

        result = await llm.chat_completion(body)
        if "error" in result:
            return JSONResponse(status_code=400, content=result, headers=degraded_headers)
        return JSONResponse(content=result, headers=degraded_headers)

    @app.get("/v2/status")
    async def v2_status():
        """Estado do servidor bridge V2."""
        models = await llm.list_models()
        return {
            "server": "Odysseus V2 Bridge",
            "version": "2.0.0",
            "status": "online",
            "uptime_seconds": config.uptime_seconds,
            "requests_served": config.request_count,
            "v2_client_available": llm._v2_available,
            "models_count": len(models),
            "started_at": config.started_at.isoformat(),
        }

    # ── Data Routes (Memory, Skills, Agents) ──
    try:
        from core.v2.data_routes import setup_data_routes
        app.include_router(setup_data_routes())
        logger.info("✅ Data routes (memory/skills/agents) registadas")
    except Exception as e:
        logger.warning("⚠️  Data routes não disponíveis: %s", e)

    @app.get("/v2/config")
    async def v2_config():
        """Configuração atual do servidor."""
        models = await llm.list_models()
        return {
            "host": config.host,
            "port": config.port,
            "ollama_url": config.ollama_url or "auto (default)",
            "default_model": config.llm_model or "auto",
            "timeout": config.llm_timeout,
            "temperature": config.llm_temperature,
            "max_tokens": config.llm_max_tokens,
            "v2_client_available": llm._v2_available,
            "available_models": [m["id"] for m in models],
        }

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ==================================================================
    #  /v2/chat — Orquestrador V2 com agentes
    # ==================================================================

    @app.get("/v2/chat")
    async def v2_chat_get(q: str = Query(..., description="A tua pergunta ou mensagem")):
        """
        Chat com o orquestrador Odysseus V2 (via GET).

        Usa o pipeline completo: Analisar → Planear → Executar (agentes) → Validar.

        Exemplo:
            GET /v2/chat?q=Quais%20s%C3%A3o%20os%20agentes%20dispon%C3%ADveis
        """
        config.request_count += 1
        result = await orch.chat(q)
        return result

    @app.post("/v2/chat")
    async def v2_chat_post(request: Request):
        """
        Chat com o orquestrador Odysseus V2 (via POST).

        Suporta contexto adicional e mensagens mais longas.

        Corpo (JSON):
            {
                "q": "Qual é o melhor modelo para programação?",
                "context": { "user_id": "..." }  # opcional
            }

        Resposta:
            {
                "success": true,
                "response": "Resposta do orquestrador...",
                "agent_results": [...],   # se agentes foram usados
                "plan": {...},            # se um plano foi executado
                "metadata": {...}
            }
        """
        config.request_count += 1
        body = await _read_json_body(request)
        if body is None:
            return {
                "success": False,
                "error": "Corpo do pedido não é JSON válido (encoding inválido).",
                "response": "Por favor, envia o pedido em JSON válido.",
            }
        question = body.get("message", body.get("q", ""))
        context = body.get("context", {})

        if not question:
            return {
                "success": False,
                "error": "A pergunta não pode estar vazia.",
                "response": "Por favor, escreve uma mensagem para que eu possa ajudar-te.",
            }

        result = await orch.chat(question, context=context)
        return result

    @app.post("/v2/chat/stream")
    async def v2_chat_stream(request: Request):
        """
        Chat com o orquestrador Odysseus V2 em streaming (SSE).

        Útil para respostas longas — os tokens são enviados um a um
        via Server-Sent Events.

        Corpo (JSON):
            { "q": "Faz uma análise SWOT da Tesla" }

        Formato da resposta (SSE):
            data: {"token": "Análise"}
            data: {"token": " SWOT"}
            data: {"token": "..."}
            data: {"done": true}
        """
        config.request_count += 1
        body = await _read_json_body(request)
        if body is None:
            async def error_stream():
                yield b"data: " + json.dumps({
                    "error": "Corpo do pedido não é JSON válido (encoding inválido)."
                }).encode() + b"\n\n"
                yield b"data: " + json.dumps({"done": True}).encode() + b"\n\n"
            return StreamingResponse(
                error_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        question = body.get("q", body.get("message", body.get("question", "")))

        if not question:
            async def error_stream():
                yield b"data: " + json.dumps({
                    "error": "A pergunta não pode estar vazia."
                }).encode() + b"\n\n"
                yield b"data: " + json.dumps({"done": True}).encode() + b"\n\n"
            return StreamingResponse(
                error_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        async def event_stream():
            yield b"data: " + json.dumps({
                "type": "metadata",
                "coordinated_by": "useragent",
            }).encode() + b"\n\n"
            # Registar callback para tool events e criar queue
            _tool_event_queue = None
            _cleanup_handler = None
            orch._ensure_init()
            if orch._orchestrator is not None:
                try:
                    import asyncio
                    from core.v2.tool_manager import ToolEvent
                    _tool_event_queue = asyncio.Queue()

                    async def _tool_handler(event: ToolEvent):
                        await _tool_event_queue.put(event)

                    orch._orchestrator.set_tool_event_callback(_tool_handler)
                    _cleanup_handler = _tool_handler
                except Exception:
                    _tool_event_queue = None

            try:
                async for token in orch.chat_stream(
                    question, context=body.get("context")
                ):
                    # Drenar eventos de ferramenta antes de cada token
                    if _tool_event_queue is not None:
                        while not _tool_event_queue.empty():
                            try:
                                import asyncio
                                event = _tool_event_queue.get_nowait()
                                yield b"data: " + json.dumps({
                                    "type": event.event_type,
                                    "tool": event.tool_name,
                                    "execution_id": event.execution_id,
                                    "params": event.params,
                                    "result": event.result,
                                    "error": event.error,
                                    "duration_ms": event.duration_ms,
                                    "timestamp": event.timestamp,
                                }, ensure_ascii=False, default=str).encode() + b"\n\n"
                            except asyncio.QueueEmpty:
                                break

                    if token == "__DONE__":
                        yield b"data: " + json.dumps({
                            "done": True,
                            "coordinated_by": "useragent",
                        }).encode() + b"\n\n"
                        return

                    yield b"data: " + json.dumps({
                        "token": token,
                        "coordinated_by": "useragent",
                    }, ensure_ascii=False).encode() + b"\n\n"
            finally:
                if _cleanup_handler is not None:
                    try:
                        orch._orchestrator.remove_tool_event_callback(_cleanup_handler)
                    except Exception:
                        pass

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Odysseus V2 Bridge Server — API OpenAI-compatível"
    )
    parser.add_argument(
        "--port", type=int, default=V2_BRIDGE_PORT,
        help="Porta do servidor (default: 8001, env: V2_BRIDGE_PORT)",
    )
    parser.add_argument(
        "--host", type=str, default=os.environ.get("V2_BRIDGE_HOST", V2_BRIDGE_HOST),
        help="Host do servidor (default: 127.0.0.1, env: V2_BRIDGE_HOST)",
    )
    parser.add_argument(
        "--ollama-url", type=str, default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="URL base do Ollama (default: http://localhost:11434, env: OLLAMA_BASE_URL)",
    )
    parser.add_argument(
        "--model", type=str, default=os.environ.get("LLM_MODEL", ""),
        help="Modelo padrão (ex: qwen3:8b, env: LLM_MODEL)",
    )
    parser.add_argument(
        "--timeout", type=float, default=float(os.environ.get("LLM_TIMEOUT", "120")),
        help="Timeout em segundos (default: 120, env: LLM_TIMEOUT)",
    )
    parser.add_argument(
        "--temperature", type=float,
        default=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
        help="Temperatura (default: 0.7, env: LLM_TEMPERATURE)",
    )
    parser.add_argument(
        "--max-tokens", type=int,
        default=int(os.environ.get("LLM_MAX_TOKENS", "4096")),
        help="Máximo de tokens (default: 4096, env: LLM_MAX_TOKENS)",
    )

    args = parser.parse_args()

    config = BridgeConfig(
        host=args.host,
        port=args.port,
        ollama_url=args.ollama_url,
        llm_model=args.model,
        llm_timeout=args.timeout,
        llm_temperature=args.temperature,
        llm_max_tokens=args.max_tokens,
    )

    app = create_app(config)

    server_url = f"http://localhost:{config.port}"
    docker_url = f"http://host.docker.internal:{config.port}"
    print(f"""
{'=' * 62}
   Odysseus V2 Bridge Server
{'=' * 62}
  Acessivel em:  {server_url}
  (Em Docker:    {docker_url})

  Endpoints:
    GET  /v1/models           -> Lista de modelos
    POST /v1/chat/completions -> Chat completion (OpenAI, Orchestrator por omissao)
    GET  /v2/chat?q=...       -> Chat com orquestrador + agentes
    POST /v2/chat             -> Chat com orquestrador + agentes
    POST /v2/chat/stream      -> Streaming do orquestrador
    GET  /v2/status           -> Estado do servidor
    GET  /v2/config           -> Configuracao

  Na interface Odysseus, o modelo "Odysseus V2" é registado e apontado para
  este bridge automaticamente no arranque da app principal (src/v2_bridge_integration.py).
{'=' * 62}
""")

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()