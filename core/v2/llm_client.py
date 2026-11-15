"""
Cliente LLM — Odysseus V2 (Provider-Aware)

Cliente assíncrono que usa o sistema de deteção de providers do Odysseus
em vez de estar hardcoded ao Ollama. Suporta todos os providers que o
Odysseus principal suporta:

    - Ollama (nativo + OpenAI compatível)
    - OpenAI (api.openai.com)
    - Anthropic (claude)
    - OpenRouter, Groq
    - Mistral, DeepSeek, xAI
    - ChatGPT Subscription (Codex API)
    - GitHub Copilot
    - Qualquer servidor OpenAI-compatível (LM Studio, llama.cpp, vLLM, etc.)

A configuração do endpoint é obtida através do sistema de settings do
Odysseus (resolve_endpoint("default")), que lê do ModelEndpoint no
SQLite e das configurações do utilizador. Se nenhum endpoint estiver
configurado, cai para Ollama via variáveis de ambiente (fallback).

Interface compatível com LLMClient original do V2:
    - chat(messages)         → LLMResponse
    - chat_stream(messages)  → AsyncGenerator[str] (tokens)
    - check_availability()   → bool
    - list_models()          → list[str]
    - get_model_info()       → dict
    - reconfigure(**kwargs)  → dict
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Iterator, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

_USERAGENT_OLLAMA_AUTHORIZED: ContextVar[Optional[bool]] = ContextVar(
    "useragent_ollama_authorized",
    default=None,
)

_MODEL_USE_ACTION = re.compile(
    r"\b(?:usa|usar|use|uses|using|utiliz\w*|cham\w*|execut\w*|"
    r"invoc\w*|invoke|run|call)\b",
    re.IGNORECASE,
)
_MODEL_USE_TARGET = re.compile(
    r"\b(?:ollama|llm|model(?:\s+(?:locally|for\s+language))?|"
    r"modelo(?:\s+(?:local|de\s+linguagem))?|"
    r"qwen[\w:.-]*|llama[\w:.-]*|mistral[\w:.-]*|gemma[\w:.-]*|"
    r"phi[\w:.-]*|deepseek[\w:.-]*)\b",
    re.IGNORECASE,
)
_NEGATED_MODEL_USE = re.compile(
    r"\b(?:não|nao|never|don't|do\s+not|without|sem)\b"
    r"[^.!?]{0,60}\b(?:ollama|llm|model|modelo|qwen[\w:.-]*|llama[\w:.-]*|"
    r"mistral[\w:.-]*|gemma[\w:.-]*|phi[\w:.-]*|deepseek[\w:.-]*)\b",
    re.IGNORECASE,
)


class OllamaAuthorizationError(RuntimeError):
    """Raised when a UserAgent request tries to reach Ollama without consent."""


@contextmanager
def useragent_ollama_authorization(authorized: bool) -> Iterator[None]:
    """Apply request-scoped Ollama authorization to all indirect LLM calls."""
    token = _USERAGENT_OLLAMA_AUTHORIZED.set(bool(authorized))
    try:
        yield
    finally:
        _USERAGENT_OLLAMA_AUTHORIZED.reset(token)


def explicit_model_use_requested(message: str) -> bool:
    """Require an affirmative model-use instruction in the user's own text."""
    if _NEGATED_MODEL_USE.search(message):
        return False
    actions = [match.start() for match in _MODEL_USE_ACTION.finditer(message)]
    targets = [match.start() for match in _MODEL_USE_TARGET.finditer(message)]
    return any(abs(action - target) <= 60 for action in actions for target in targets)


def _useragent_ollama_scope_active() -> bool:
    return _USERAGENT_OLLAMA_AUTHORIZED.get() is not None


def useragent_ollama_request_active() -> bool:
    """True while UserAgent request code is running, including cache hooks."""
    return _useragent_ollama_scope_active()


def _useragent_ollama_is_authorized() -> bool:
    return _USERAGENT_OLLAMA_AUTHORIZED.get() is True


# ======================================================================
#  Provider Functions imported from Odysseus core
# ======================================================================

# Lazy imports to avoid circular deps at module load time
def _get_endpoint_resolver():
    from src.endpoint_resolver import resolve_endpoint as _re
    return _re

def _get_llm_core():
    from src import llm_core as _lc
    return _lc

def _get_http_client():
    """Return the shared process-wide httpx client from llm_core."""
    return _get_llm_core()._get_http_client()

def _detect_provider(url: str) -> str:
    return _get_llm_core()._detect_provider(url)

def _provider_label(url: str) -> str:
    return _get_llm_core()._provider_label(url)

def _sanitize_llm_messages(messages):
    return _get_llm_core()._sanitize_llm_messages(messages)

def _normalize_openai_chat_url(url: str) -> str:
    return _get_llm_core()._normalize_openai_chat_url(url)

def _normalize_ollama_url(url: str) -> str:
    return _get_llm_core()._normalize_ollama_url(url)

def _normalize_anthropic_url(url: str) -> str:
    return _get_llm_core()._normalize_anthropic_url(url)

def _normalize_chatgpt_subscription_url(url: str) -> str:
    return _get_llm_core()._normalize_chatgpt_subscription_url(url)

def _build_anthropic_payload(model, messages, temperature, max_tokens, stream=False, tools=None):
    return _get_llm_core()._build_anthropic_payload(model, messages, temperature, max_tokens, stream, tools)

def _build_ollama_payload(model, messages, temperature, max_tokens, stream=False, tools=None, num_ctx=None):
    return _get_llm_core()._build_ollama_payload(model, messages, temperature, max_tokens, stream, tools, num_ctx)

def _build_chatgpt_responses_payload(model, messages, temperature, max_tokens, *, stream=False):
    return _get_llm_core()._build_chatgpt_responses_payload(model, messages, temperature, max_tokens, stream=stream)

def _build_anthropic_headers(headers):
    return _get_llm_core()._build_anthropic_headers(headers)

def _parse_anthropic_response(data):
    return _get_llm_core()._parse_anthropic_response(data)

def _parse_ollama_response(data):
    return _get_llm_core()._parse_ollama_response(data)

def _call_timeout(read_timeout):
    return _get_llm_core()._call_timeout(read_timeout)

def _stream_timeout(read_timeout):
    return _get_llm_core()._stream_timeout(read_timeout)

def _build_anthropic_messages_from_openai(messages):
    """Convert OpenAI-format messages list to Anthropic messages (excluding system)."""
    result = []
    for m in messages or []:
        role = m.get("role", "")
        if role == "system":
            continue  # system is handled separately in Anthropic payload
        if role == "tool":
            result.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", ""),
                    "content": m.get("content", ""),
                }],
            })
        elif role == "assistant":
            content = m.get("content", "")
            tool_calls = m.get("tool_calls")
            if tool_calls:
                blocks = []
                if content:
                    blocks.append({"type": "text", "text": str(content)})
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args.strip() else {}
                        except json.JSONDecodeError:
                            args = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": args,
                    })
                result.append({"role": "assistant", "content": blocks})
            else:
                result.append({"role": "assistant", "content": str(content) if content else ""})
        elif role == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                result.append({"role": "user", "content": content})
            elif isinstance(content, list):
                # Convert multimodal blocks
                from src.llm_core import _convert_openai_content_to_anthropic
                result.append({"role": "user", "content": _convert_openai_content_to_anthropic(content)})
            else:
                result.append({"role": "user", "content": str(content) if content else ""})
    return result

def _extract_anthropic_system(messages):
    """Extract system message content from an OpenAI-format message list."""
    parts = []
    for m in messages or []:
        if m.get("role") == "system":
            c = m.get("content", "")
            if c:
                parts.append(str(c))
    return "\n\n".join(parts) if parts else None


# ======================================================================
#  Configuração
# ======================================================================

# Modelos Ollama preferidos para o fallback automático (por ordem de preferência).
# O primeiro modelo desta lista que existir no servidor será usado quando o
# modelo configurado não estiver instalado (ex.: llama3.2 em falta -> qwen3:8b).
PREFERRED_OLLAMA_MODELS = [
    # Equipa virtual oficial (Orchestrator)
    "qwen3:8b",
    "qwen3",
    "llama3.2:3b",
    "llama3.2",
    "gemma3:12b",
    "gemma3:4b",
    "gemma2:9b",
    "gemma2:2b",
    "qwen2.5:7b",
    "qwen2.5:3b",
    "qwen2.5:1.5b",
    "llama3.2",
    "llama3.1",
    "llama3",
    "mistral",
    "phi4",
    "phi3",
]


@dataclass
class LLMConfig:
    """Configuração do cliente LLM do V2.

    Ao contrário da versão anterior (hardcoded a Ollama), esta configuração
    é dinâmica: o endpoint é resolvido através do sistema de providers do
    Odysseus. Os campos abaixo servem como fallback para o modo Ollama
    classico quando não há endpoint configurado.
    """
    base_url: str = ""
    model: str = ""
    timeout: float = 60.0
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9

    # Resolved endpoint from Odysseus provider system
    resolved_url: str = ""
    resolved_model: str = ""
    resolved_headers: dict = field(default_factory=dict)
    resolved_provider: str = ""  # "openai", "anthropic", "ollama", etc.

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Carrega configuração de fallback a partir de variáveis de ambiente (Ollama classico)."""
        in_docker = os.path.exists("/.dockerenv")
        if not in_docker:
            try:
                with open("/proc/1/cgroup", "r", encoding="utf-8", errors="ignore") as fh:
                    in_docker = any(m in fh.read() for m in ("docker", "containerd", "kubepods"))
            except Exception:
                pass

        default_url = (
            "http://host.docker.internal:11434"
            if in_docker
            else "http://127.0.0.1:11434"
        )

        raw_url = (
            os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_URL")
            or default_url
        ).rstrip("/")
        if raw_url.endswith("/v1"):
            raw_url = raw_url[:-3]

        return cls(
            base_url=raw_url,
            model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            timeout=float(os.getenv("LLM_TIMEOUT", "60")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            top_p=float(os.getenv("LLM_TOP_P", "0.9")),
        )


# ======================================================================
#  Resposta estruturada
# ======================================================================

@dataclass
class LLMResponse:
    """Resposta do LLM estruturada."""
    content: str
    model: str = ""
    finish_reason: str = "stop"
    usage: dict = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


# ======================================================================
#  Exceções
# ======================================================================

class LLMConnectionError(Exception):
    """Erro de conexão com o servidor LLM."""
    pass


class LLMTimeoutError(Exception):
    """Timeout na resposta do LLM."""
    pass


class LLMResponseError(Exception):
    """Resposta inválida do LLM."""
    pass


# ======================================================================
#  Cliente LLM (Provider-Aware)
# ======================================================================

class LLMClient:
    """
    Cliente assíncrono para LLM que usa o sistema de providers do Odysseus.

    Em vez de apontar diretamente para um servidor específico (ex: Ollama),
    este cliente resolve o endpoint através de `src.endpoint_resolver.resolve_endpoint("default")`,
    que lê a configuração do SQLite (ModelEndpoint) e as settings do utilizador.

    Providers suportados (via llm_core._detect_provider):
        - openai     (api.openai.com, OpenAI-compatible)
        - anthropic  (claude) 
        - ollama     (nativo /api/chat)
        - openrouter
        - groq, nvidia, moonshot, mistral, deepseek, cerebras
        - chatgpt-subscription (Codex API)
        - copilot    (GitHub Copilot)
        - opencode   (zen/go)

    Fallback: Se nenhum endpoint estiver configurado, usa Ollama via env vars
    (compatível com a versão anterior).
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self._client: Optional[Any] = None  # httpx.AsyncClient from llm_core
        self._available: Optional[bool] = None  # cached após primeira verificação
        self._resolved = False  # se o endpoint Odysseus já foi resolvido
        # Fallback automático de modelo (Ollama): quando o modelo configurado
        # não existe no servidor, escolhe um modelo instalado.
        self._model_override: Optional[str] = None
        self._models_cache: Optional[list[str]] = None

    # ------------------------------------------------------------------
    #  Resolução do endpoint Odysseus
    # ------------------------------------------------------------------

    def _resolve_endpoint(self) -> bool:
        """
        Tenta resolver o endpoint usando o sistema de providers do Odysseus.

        Se conseguir resolver, atualiza self.config com a URL, model e headers
        específicos do provider detetado.

        Returns:
            True se o endpoint foi resolvido com sucesso, False para fallback Ollama.
        """
        if self._resolved:
            return bool(self.config.resolved_url)

        try:
            # Resolver endpoint "default" (configurado nas settings do Odysseus)
            resolver = _get_endpoint_resolver()
            chat_url, model, headers = resolver(
                "default",
                fallback_url=None,
                fallback_model=None,
            )

            if chat_url and model:
                self.config.resolved_url = chat_url
                self.config.resolved_model = model
                self.config.resolved_headers = headers or {}
                self.config.resolved_provider = _detect_provider(chat_url)
                self._resolved = True
                logger.info(
                    "V2 LLM: endpoint resolvido → provider=%s model=%s url=%s",
                    self.config.resolved_provider, model, chat_url,
                )
                return True
        except Exception as e:
            logger.info(
                "V2 LLM: não foi possível resolver endpoint Odysseus: %s. "
                "A usar fallback Ollama.", e,
            )

        self._resolved = True  # marcamos como resolvido (para fallback)
        return False

    def _get_chat_url(self) -> str:
        """Retorna a URL de chat atual (resolvida ou fallback Ollama)."""
        if self._resolve_endpoint() and self.config.resolved_url:
            return self.config.resolved_url
        # Fallback: construir URL nativa Ollama
        base = self.config.base_url.rstrip("/")
        return base + "/api/chat"

    def _get_model(self) -> str:
        """Retorna o model name atual (resolvido, override automático ou fallback)."""
        if self._model_override:
            return self._model_override
        if self._resolve_endpoint() and self.config.resolved_model:
            return self.config.resolved_model
        return self.config.model

    def _get_provider(self) -> str:
        """Retorna o provider atual (resolvido ou 'ollama')."""
        if self._resolve_endpoint() and self.config.resolved_provider:
            return self.config.resolved_provider
        url = self._get_chat_url()
        return _detect_provider(url)

    def _get_request_target(self) -> tuple[str, str]:
        """Força API nativa Ollama para qualquer URL em :11434.

        Histórico: detecção via _is_ollama_openai_compat_url era
        inconsistente — o mesmo URL (http://host:11434/v1/chat/completions)
        passava para /api/chat num pedido mas caía em /v1/chat/completions
        noutro, causando 60s+ de thinking sem think:false. Detecção
        agora é feita só por porta, sem depender de heurística de path.
        """
        chat_url = self._get_chat_url()
        provider = self._get_provider()
        try:
            parsed = urlsplit(chat_url)
            if parsed.port == 11434:
                return f"{parsed.scheme}://{parsed.netloc}/api/chat", "ollama"
        except (ValueError, AttributeError):
            pass
        return chat_url, provider

    def _ollama_request_denied(self) -> bool:
        """Check the active UserAgent request policy against the resolved provider."""
        if (
            not _useragent_ollama_scope_active()
            or _useragent_ollama_is_authorized()
        ):
            return False
        _, provider = self._get_request_target()
        return provider == "ollama"

    # ------------------------------------------------------------------
    #  Seleção automática de modelo (Ollama)
    # ------------------------------------------------------------------

    async def _ensure_usable_model(self) -> None:
        """Escolhe automaticamente um modelo disponível no Ollama.

        Quando o modelo configurado (ex.: llama3.2) não está instalado no
        servidor Ollama, usa um modelo instalado -- preferindo os da lista
        PREFERRED_OLLAMA_MODELS (onde entra o qwen3:8b). O resultado fica
        em cache para não repetir o pedido a cada mensagem.
        """
        # _models_cache não vazio sinaliza que a verificação já correu nesta
        # sessão — evita um pedido /api/tags por cada mensagem.
        provider = self._get_provider()
        chat_url = self._get_chat_url()
        # Docker/local Ollama também pode expor a API compatível OpenAI
        # (`/v1/chat/completions`). Nesse caso o provider é detectado como
        # `openai`, mas a lista local de modelos continua a ser a autoridade.
        local_ollama_compat = (
            "localhost:11434" in chat_url.lower()
            or "127.0.0.1:11434" in chat_url.lower()
        )
        if (
            self._model_override is not None
            or self._models_cache is not None
            or (provider != "ollama" and not local_ollama_compat)
        ):
            return
        try:
            models = await self.list_models()
            if not models:
                return
            self._models_cache = models
            current = self._get_model()
            if current and current in models:
                return

            # Se a resolução de endpoint devolveu um modelo obsoleto, a
            # configuração explícita continua a ser a
            # fonte de escolha. Também aceitar a variante tagless instalada
            # (llama3.2 -> llama3.2:3b), evitando HTTP 404 e fallback sem LLM.
            configured = self.config.model or ""
            configured_base = configured.split(":")[0].lower()
            configured_family = [
                m for m in models if m.split(":")[0].lower() == configured_base
            ] if configured_base else []
            if configured_family:
                self._model_override = configured_family[0]
                return

            base = current.split(":")[0].lower() if current else ""
            if base:
                same_family = [m for m in models if m.split(":")[0].lower() == base]
                if same_family:
                    if current not in same_family:
                        self._model_override = same_family[0]
                        logger.info(
                            "Modelo '%s' resolvido para a variante instalada '%s'.",
                            current, same_family[0],
                        )
                    return

            normalized = {m.lower(): m for m in models}
            chosen = None
            for pref in PREFERRED_OLLAMA_MODELS:
                if pref.lower() in normalized:
                    chosen = normalized[pref.lower()]
                    break
            if chosen is None:
                chosen = models[0]

            self._model_override = chosen
            logger.warning(
                "Modelo configurado '%s' não está disponível no Ollama -- a usar "
                "'%s' (disponíveis: %s).",
                current, chosen, ", ".join(models[:8]),
            )
        except Exception as e:
            logger.debug("Auto-seleção de modelo falhou: %s", e)

    def _get_headers(self) -> dict:
        """Retorna headers HTTP para o provider atual."""
        if self._resolve_endpoint():
            base_headers = dict(self.config.resolved_headers)
            provider = self.config.resolved_provider
            if provider == "anthropic":
                return _build_anthropic_headers(base_headers)
            return base_headers
        return {"Content-Type": "application/json"}

    # ------------------------------------------------------------------
    #  Gestão do cliente HTTP (usa o shared client do llm_core)
    # ------------------------------------------------------------------

    async def _get_client(self):
        """Obtém o cliente HTTP partilhado do llm_core (lazy)."""
        if self._client is None:
            self._client = _get_http_client()
        return self._client

    async def close(self):
        """Fecha o cliente HTTP (no-op: o shared client é gerido pelo llm_core)."""
        self._client = None

    # ------------------------------------------------------------------
    #  Verificação de disponibilidade
    # ------------------------------------------------------------------

    async def check_availability(self, force: bool = False) -> bool:
        """
        Verifica se o LLM está acessível através do provider configurado.

        O resultado é em cache para evitar pings repetidos.
        Usa force=True para forçar nova verificação.
        """
        if self._ollama_request_denied():
            logger.info(
                "Verificação de disponibilidade Ollama bloqueada para pedido "
                "do UserAgent sem autorização expressa."
            )
            return False

        if self._available is not None and not force:
            return self._available

        chat_url, provider = self._get_request_target()
        model = self._get_model()

        if not chat_url or not model:
            self._available = False
            return False

        try:
            client = await self._get_client()
            headers = self._get_headers()

            if provider == "ollama":
                # Ollama: tentar /api/tags (endpoint nativo)
                base = chat_url.rstrip("/")
                if base.endswith("/api/chat"):
                    base = base[:-5]  # remove /chat
                tags_url = base + "/tags"
                resp = await client.get(tags_url, timeout=5.0)
                if resp.status_code == 200:
                    self._available = True
                    logger.info("LLM disponível (Ollama): %s", tags_url)
                    return True

                if _useragent_ollama_scope_active():
                    self._available = False
                    return False

                # Fallback: tentar endpoint /api/embeddings
                try:
                    resp = await client.post(
                        chat_url,
                        json={"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
                        timeout=5.0,
                    )
                    if resp.status_code == 200:
                        self._available = True
                        return True
                except Exception:
                    pass

            elif provider == "anthropic":
                # Anthropic: ver /v1/models
                from urllib.parse import urlparse
                from src.endpoint_resolver import build_models_url
                base = chat_url
                if base.endswith("/v1/messages"):
                    base = base[:-12]  # remove /v1/messages (12 chars)
                try:
                    resp = await client.get(base.rstrip("/") + "/v1/models", timeout=5.0)
                    if resp.status_code in (200, 401):
                        # 401 significa que a API está ativa (só precisamos de saber se responde)
                        self._available = True
                        return True
                except Exception:
                    pass

            elif provider == "chatgpt-subscription":
                # ChatGPT Codex API: tentar ping no endpoint responses
                try:
                    resp = await client.get(chat_url.rstrip("/responses"), timeout=5.0)
                    self._available = resp.status_code < 500
                    return self._available
                except Exception:
                    pass

            else:
                # OpenAI-compatible: /v1/models
                base = chat_url
                if base.endswith("/chat/completions"):
                    base = base[:-17]  # remove /chat/completions (17 chars)
                models_url = base.rstrip("/") + "/models"
                try:
                    resp = await client.get(models_url, timeout=5.0, headers=headers)
                    if resp.status_code == 200:
                        self._available = True
                        return True
                except Exception:
                    pass

                # Fallback: tentar chat completion com 1 token
                try:
                    resp = await client.post(
                        chat_url,
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": "ping"}],
                            "max_tokens": 1,
                        },
                        timeout=5.0,
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        self._available = True
                        return True
                except Exception:
                    pass

            self._available = False
            logger.info("LLM não disponível: %s (provider=%s)", chat_url, provider)
            return False

        except Exception as e:
            self._available = False
            logger.info("LLM não disponível: %s — %s", chat_url, e)
            return False

    # ------------------------------------------------------------------
    #  Chat Completion
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> LLMResponse:
        """
        Envia uma conversa para o LLM e retorna a resposta.

        Usa o provider detetado automaticamente (Anthropic, OpenAI, Ollama, etc.)
        e constrói o payload específico para cada provider.

        Args:
            messages: Lista de mensagens [{"role": "...", "content": "..."}]
            model: Nome do modelo (usa o resolvido ou config.model se não fornecido)
            temperature: Temperatura (usa config.temperature se não fornecido)
            max_tokens: Máximo de tokens (usa config.max_tokens se não fornecido)
            stream: Se True, ativa streaming (usar chat_stream() em vez disso)

        Returns:
            LLMResponse com content, model, usage, etc.
        """
        if not messages:
            return LLMResponse(
                content="",
                success=False,
                error="Lista de mensagens vazia.",
            )

        if self._ollama_request_denied():
            return LLMResponse(
                content="",
                success=False,
                error=(
                    "Chamada ao Ollama bloqueada: o pedido do UserAgent não "
                    "solicitou explicitamente a utilização de um modelo."
                ),
            )

        # The UserAgent must not silently substitute a different model or probe
        # Ollama's model list. An explicit request gets one attempt with the
        # configured/requested model; other callers retain existing behavior.
        if not _useragent_ollama_scope_active():
            await self._ensure_usable_model()

        try:
            chat_url, provider = self._get_request_target()
            effective_model = model or self._get_model()
            effective_temp = temperature if temperature is not None else self.config.temperature
            effective_max_tokens = max_tokens or self.config.max_tokens
            headers = self._get_headers()

            if not chat_url or not effective_model:
                return LLMResponse(
                    content="",
                    success=False,
                    error="Endpoint LLM não configurado. Configura um modelo nas definições.",
                )

            # Sanitizar mensagens (remove campos extra, repara tool_calls orphan)
            clean_messages = _sanitize_llm_messages(messages)

            # Construir payload específico do provider
            if provider == "anthropic":
                payload = _build_anthropic_payload(
                    effective_model, clean_messages, effective_temp,
                    effective_max_tokens, stream=False,
                )
                client = await self._get_client()

                logger.debug(
                    "LLM request (Anthropic): model=%s, messages=%d, temp=%.1f",
                    effective_model, len(clean_messages), effective_temp,
                )

                resp = await client.post(
                    chat_url if chat_url.endswith("/v1/messages") else chat_url.rstrip("/") + "/v1/messages",
                    json=payload,
                    headers=headers,
                    timeout=_call_timeout(self.config.timeout),
                )

                if resp.status_code != 200:
                    from src.llm_core import _format_upstream_error
                    return LLMResponse(
                        content="",
                        success=False,
                        error=_format_upstream_error(resp.status_code, resp.content, chat_url),
                    )

                data = resp.json()
                content = _parse_anthropic_response(data)
                usage = data.get("usage", {})
                resp_model = data.get("model", effective_model)
                finish = data.get("stop_reason", "stop")

                return LLMResponse(
                    content=content,
                    model=resp_model,
                    finish_reason=finish,
                    usage=usage,
                    success=True,
                )

            elif provider == "ollama":
                payload = _build_ollama_payload(
                    effective_model, clean_messages, effective_temp,
                    effective_max_tokens, stream=False,
                )
                client = await self._get_client()

                logger.debug(
                    "LLM request (Ollama): model=%s, messages=%d, temp=%.1f",
                    effective_model, len(clean_messages), effective_temp,
                )

                resp = await client.post(
                    chat_url,
                    json=payload,
                    timeout=_call_timeout(self.config.timeout),
                )

                if resp.status_code != 200:
                    return LLMResponse(
                        content="",
                        success=False,
                        error=f"Ollama retornou status {resp.status_code}: {resp.text[:500]}",
                    )

                data = resp.json()
                content = _parse_ollama_response(data)
                usage = {
                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                    "eval_count": data.get("eval_count", 0),
                }
                return LLMResponse(
                    content=content or "",
                    model=effective_model,
                    finish_reason=data.get("done_reason", "stop"),
                    usage=usage,
                    success=True,
                )

            elif provider == "chatgpt-subscription":
                payload = _build_chatgpt_responses_payload(
                    effective_model, clean_messages, effective_temp,
                    effective_max_tokens, stream=False,
                )
                client = await self._get_client()

                # ChatGPT Codex API usa /responses
                codex_url = chat_url if chat_url.endswith("/responses") else chat_url.rstrip("/") + "/responses"

                resp = await client.post(
                    codex_url,
                    json=payload,
                    headers=headers,
                    timeout=_call_timeout(self.config.timeout),
                )

                if resp.status_code != 200:
                    from src.llm_core import _format_chatgpt_subscription_error
                    return LLMResponse(
                        content="",
                        success=False,
                        error=_format_chatgpt_subscription_error(resp.status_code, resp.text[:500]),
                    )

                data = resp.json()
                # Parse response output
                output = data.get("output", [])
                content_parts = []
                for block in output:
                    if isinstance(block, dict):
                        for part in block.get("content", []):
                            if isinstance(part, dict) and part.get("type") == "output_text":
                                content_parts.append(part.get("text", ""))
                content = "".join(content_parts)
                return LLMResponse(
                    content=content or "",
                    model=data.get("model", effective_model),
                    usage=data.get("usage", {}),
                    success=True,
                )

            else:
                # OpenAI-compatible (OpenAI, OpenRouter, Groq, Mistral, DeepSeek, etc.)
                from src.llm_core import (
                    _normalize_openai_chat_url, _restricts_temperature,
                    _uses_max_completion_tokens,
                )

                api_url = _normalize_openai_chat_url(chat_url)
                if not api_url:
                    api_url = chat_url.rstrip("/") + "/chat/completions"

                payload = {
                    "model": effective_model,
                    "messages": clean_messages,
                    "max_tokens": effective_max_tokens,
                    "stream": False,
                }

                if not _restricts_temperature(effective_model):
                    payload["temperature"] = effective_temp
                else:
                    payload.pop("temperature", None)

                if provider in ("openai", "openrouter", "deepseek") and _uses_max_completion_tokens(effective_model):
                    payload["max_completion_tokens"] = payload.pop("max_tokens")

                # Mistral reasoning effort
                if provider == "mistral":
                    payload["reasoning_effort"] = os.getenv("ODYSSEUS_MISTRAL_REASONING_EFFORT", "high")

                # Defesa em profundidade: se 1A não apanhar (ex.: URLs com
                # hostname weird), enviamos think:false também aqui. Ollama
                # aceita campos extra no formato OpenAI-compat; outros providers
                # ignoram ou rejeitam, por isso só o fazemos para :11434.
                try:
                    if urlsplit(chat_url).port == 11434:
                        payload["think"] = False
                except (ValueError, AttributeError):
                    pass

                client = await self._get_client()

                logger.debug(
                    "LLM request (OpenAI-compat): provider=%s model=%s messages=%d",
                    provider, effective_model, len(clean_messages),
                )

                resp = await client.post(
                    api_url,
                    json=payload,
                    headers=headers,
                    timeout=_call_timeout(self.config.timeout),
                )

                if resp.status_code != 200:
                    from src.llm_core import _format_upstream_error
                    return LLMResponse(
                        content="",
                        success=False,
                        error=_format_upstream_error(resp.status_code, resp.content, chat_url),
                    )

                data = resp.json()

                # Extract content from OpenAI-format response
                content = ""
                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    msg = choice.get("message", {})
                    content = msg.get("content") or ""
                    finish_reason = choice.get("finish_reason", "stop")
                else:
                    finish_reason = "stop"

                return LLMResponse(
                    content=content,
                    model=data.get("model", effective_model),
                    finish_reason=finish_reason,
                    usage=data.get("usage", {}),
                    success=True,
                )

        except Exception as e:
            error_str = str(e)
            if any(w in error_str.lower() for w in ("timeout", "timed out", "read timeout")):
                return LLMResponse(
                    content="",
                    success=False,
                    error=f"Timeout após {self.config.timeout}s ao contactar LLM.",
                )
            return LLMResponse(
                content="",
                success=False,
                error=f"Erro de comunicação com LLM: {error_str}",
            )

    # ------------------------------------------------------------------
    #  Chat Completion com Streaming
    # ------------------------------------------------------------------

    async def chat_stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Faz uma chamada de chat completion em streaming para o LLM.

        Usa o provider detetado automaticamente. No final da transmissão,
        faz yield de "__DONE__" para indicar que a resposta terminou.

        Yields:
            str: Tokens individuais da resposta.
        """
        if not messages:
            return

        if self._ollama_request_denied():
            yield (
                "❌ Chamada ao Ollama bloqueada: o pedido do UserAgent não "
                "solicitou explicitamente a utilização de um modelo."
            )
            yield "__DONE__"
            return

        if not _useragent_ollama_scope_active():
            await self._ensure_usable_model()

        chat_url, provider = self._get_request_target()
        effective_model = model or self._get_model()
        effective_temp = temperature if temperature is not None else self.config.temperature
        effective_max_tokens = max_tokens or self.config.max_tokens
        headers = self._get_headers()

        if not chat_url or not effective_model:
            yield "❌ Endpoint LLM não configurado. Configura um modelo nas definições."
            yield "__DONE__"
            return

        try:
            clean_messages = _sanitize_llm_messages(messages)

            if provider == "anthropic":
                payload = _build_anthropic_payload(
                    effective_model, clean_messages, effective_temp,
                    effective_max_tokens, stream=True,
                )
                client = await self._get_client()
                api_url = chat_url if chat_url.endswith("/v1/messages") else chat_url.rstrip("/") + "/v1/messages"

                async with client.stream("POST", api_url, json=payload, headers=headers,
                                          timeout=_stream_timeout(max(self.config.timeout, 180.0))) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        from src.llm_core import _format_upstream_error
                        yield f"\n\n❌ {_format_upstream_error(resp.status_code, body, chat_url)}"
                        yield "__DONE__"
                        return

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except Exception:
                            continue

                        # Anthropic SSE: content_block_delta → delta.text
                        ctype = chunk.get("type", "")
                        if ctype == "content_block_delta":
                            delta = chunk.get("delta", {})
                            text = delta.get("text", "")
                            if text:
                                yield text
                        elif ctype == "message_stop":
                            break

                yield "__DONE__"
                return

            elif provider == "ollama":
                payload = _build_ollama_payload(
                    effective_model, clean_messages, effective_temp,
                    effective_max_tokens, stream=True,
                )
                client = await self._get_client()

                async with client.stream("POST", chat_url, json=payload,
                                          timeout=_stream_timeout(max(self.config.timeout, 180.0))) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield f"\n\n❌ Erro do Ollama (status {resp.status_code}): {body[:500].decode(errors='ignore')}"
                        yield "__DONE__"
                        return

                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except Exception:
                            continue
                        content = chunk.get("message", {}).get("content", "") or chunk.get("response", "")
                        if content:
                            yield content
                        if chunk.get("done"):
                            break

                yield "__DONE__"
                return

            elif provider == "chatgpt-subscription":
                payload = _build_chatgpt_responses_payload(
                    effective_model, clean_messages, effective_temp,
                    effective_max_tokens, stream=True,
                )
                client = await self._get_client()
                codex_url = chat_url if chat_url.endswith("/responses") else chat_url.rstrip("/") + "/responses"

                async with client.stream("POST", codex_url, json=payload, headers=headers,
                                          timeout=_stream_timeout(max(self.config.timeout, 180.0))) as resp:
                    if resp.status_code != 200:
                        from src.llm_core import _format_chatgpt_subscription_error
                        yield f"\n\n❌ {_format_chatgpt_subscription_error(resp.status_code, (await resp.aread())[:500])}"
                        yield "__DONE__"
                        return

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except Exception:
                            continue
                        # SSE from ChatGPT responses API
                        for part in (chunk.get("output") or []):
                            if isinstance(part, dict):
                                for block in (part.get("content") or []):
                                    if isinstance(block, dict) and block.get("type") == "output_text":
                                        yield block.get("text", "")

                yield "__DONE__"
                return

            else:
                # OpenAI-compatible streaming (OpenAI, OpenRouter, Groq, etc.)
                from src.llm_core import (
                    _normalize_openai_chat_url, _restricts_temperature,
                    _uses_max_completion_tokens,
                )

                api_url = _normalize_openai_chat_url(chat_url)
                if not api_url:
                    api_url = chat_url.rstrip("/") + "/chat/completions"

                payload = {
                    "model": effective_model,
                    "messages": clean_messages,
                    "max_tokens": effective_max_tokens,
                    "stream": True,
                }

                if not _restricts_temperature(effective_model):
                    payload["temperature"] = effective_temp

                if provider in ("openai", "openrouter", "deepseek") and _uses_max_completion_tokens(effective_model):
                    payload["max_completion_tokens"] = payload.pop("max_tokens")

                if provider == "mistral":
                    payload["reasoning_effort"] = os.getenv("ODYSSEUS_MISTRAL_REASONING_EFFORT", "high")

                # Defesa em profundidade: se 1A não apanhar (ex.: URLs com
                # hostname weird), enviamos think:false também aqui. Ollama
                # aceita campos extra no formato OpenAI-compat; outros providers
                # ignoram ou rejeitam, por isso só o fazemos para :11434.
                try:
                    if urlsplit(chat_url).port == 11434:
                        payload["think"] = False
                except (ValueError, AttributeError):
                    pass

                client = await self._get_client()

                async with client.stream("POST", api_url, json=payload, headers=headers,
                                          timeout=_stream_timeout(max(self.config.timeout, 180.0))) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        from src.llm_core import _format_upstream_error
                        yield f"\n\n❌ {_format_upstream_error(resp.status_code, body, chat_url)}"
                        yield "__DONE__"
                        return

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except Exception:
                            continue

                        # OpenAI-format SSE
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content

                yield "__DONE__"
                return

        except Exception as e:
            # TimeoutError tem str() vazio — incluir tipo para diagnóstico.
            error_str = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            logger.warning("Erro no streaming LLM: %s", error_str)
            yield f"\n\n❌ Erro de comunicação: {error_str}"
            yield "__DONE__"

    # ------------------------------------------------------------------
    #  Utilitários
    # ------------------------------------------------------------------

    async def list_models(self) -> list[str]:
        """
        Lista modelos disponíveis no provider configurado.

        Usa o sistema de descoberta do Odysseus para providers locais
        (Ollama, LM Studio, llama.cpp) e listas hardcoded para Anthropic.
        """
        if self._ollama_request_denied():
            raise OllamaAuthorizationError(
                "A listagem de modelos Ollama foi bloqueada para um pedido "
                "do UserAgent sem autorização expressa."
            )

        try:
            chat_url = self._get_chat_url()
            if not chat_url:
                return []

            provider = self._get_provider()
            headers = self._get_headers()

            if provider == "anthropic":
                # Lista hardcoded de modelos Claude
                from src.llm_core import ANTHROPIC_MODELS
                return list(ANTHROPIC_MODELS)

            if provider == "ollama":
                # Ollama: /api/tags
                base = chat_url.rstrip("/")
                if base.endswith("/api/chat"):
                    base = base[:-5]
                tags_url = base + "/tags"
                client = await self._get_client()
                resp = await client.get(tags_url, timeout=5.0)
                if resp.status_code == 200:
                    data = resp.json()
                    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                return []

            # OpenAI-compatible: /v1/models
            from src.llm_core import list_model_ids
            return list_model_ids(chat_url, timeout=int(self.config.timeout), headers=headers)

        except Exception as e:
            logger.debug("Erro ao listar modelos: %s", e)
            return []

    async def get_model_info(self) -> dict:
        """Retorna informações sobre a configuração atual do LLM."""
        try:
            self._resolve_endpoint()
        except Exception:
            pass

        return {
            "base_url": self.config.resolved_url or self.config.base_url,
            # Mostra o modelo realmente usado (inclui a seleção automática).
            "model": self._model_override or self.config.resolved_model or self.config.model,
            "provider": self.config.resolved_provider or self._get_provider(),
            "available": await self.check_availability(),
            "timeout": self.config.timeout,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

    async def _list_models(self) -> list[str]:
        """Alias para list_models (compatibilidade)."""
        return await self.list_models()

    # ------------------------------------------------------------------
    #  Configuração dinâmica
    # ------------------------------------------------------------------

    async def reconfigure(self, **kwargs) -> dict:
        """
        Atualiza campos da configuração em tempo real (sem reiniciar o servidor).

        Aceita: model, temperature, timeout, max_tokens, base_url, top_p
        Se o base_url mudar, a resolução do endpoint Odysseus é reiniciada.

        Args:
            **kwargs: Campos a atualizar.

        Returns:
            dict com o estado atual após a atualização.
        """
        changed = []
        need_reconnect = False

        valid_fields = {"model", "temperature", "timeout", "max_tokens", "base_url", "top_p"}

        for key, value in kwargs.items():
            if key not in valid_fields:
                logger.warning("Campo ignorado (não reconhecido): %s", key)
                continue

            # Converter tipos
            if key == "temperature":
                value = float(value)
            elif key == "timeout":
                value = float(value)
            elif key == "max_tokens":
                value = int(value)
            elif key == "top_p":
                value = float(value)
            else:
                value = str(value).strip()

            if key == "base_url":
                value = value.rstrip("/")
                if value.endswith("/v1"):
                    value = value[:-3]
                need_reconnect = True

            current = getattr(self.config, key, None)
            if current == value:
                continue

            setattr(self.config, key, value)
            changed.append(key)

        # Se o base_url ou model mudaram, reiniciar resolução do endpoint
        if need_reconnect or "model" in changed:
            self._resolved = False
            self.config.resolved_url = ""
            self.config.resolved_model = ""
            self.config.resolved_headers = {}
            self.config.resolved_provider = ""
            self._available = None
            # Modelo/endpoint alterado: limpar a seleção automática anterior
            # para que o novo modelo seja verificado de novo no Ollama.
            self._model_override = None
            self._models_cache = None

            # Invalidar cache semântico
            try:
                from .cache import get_cache
                cache = get_cache()
                removed = cache.invalidate()
                if removed:
                    logger.info("Cache semântico invalidado: %d entradas", removed)
            except Exception as e:
                logger.debug("Falha ao invalidar cache semântico: %s", e)

        logger.info(
            "LLM config atualizada: %s (re-resolve=%s)",
            ", ".join(changed) if changed else "nenhuma alteração",
            need_reconnect,
        )

        return {
            "success": True,
            "changed": changed,
            "reconnected": need_reconnect,
            "config": {
                "base_url": self.config.base_url,
                "model": self.config.model,
                "temperature": self.config.temperature,
                "timeout": self.config.timeout,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "resolved_url": self.config.resolved_url,
                "resolved_model": self.config.resolved_model,
                "resolved_provider": self.config.resolved_provider or self._get_provider(),
            },
        }


# ======================================================================
#  Singleton
# ======================================================================

_llm_client: Optional[LLMClient] = None


def get_llm_client(config: Optional[LLMConfig] = None) -> LLMClient:
    """Retorna a instância global do cliente LLM."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient(config=config)
    return _llm_client


def reset_llm_client() -> None:
    """Reinicia o cliente LLM (útil para testes ou reconfiguração)."""
    global _llm_client
    _llm_client = None
