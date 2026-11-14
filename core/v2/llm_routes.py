"""
Rotas de Configuração LLM — Odysseus V2

Endpoints para consultar e alterar a configuração do LLM em tempo real,
sem necessidade de reiniciar o servidor.

Endpoints:
    GET    /api/v2/llm/config    — Estado atual da configuração
    PUT    /api/v2/llm/config    — Atualizar configuração (model, temperature, timeout, ...)
    POST   /api/v2/llm/check     — Forçar verificação de disponibilidade
    GET    /api/v2/llm/models    — Listar modelos disponíveis no servidor
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Body

logger = logging.getLogger(__name__)


def setup_llm_routes(llm_client=None) -> APIRouter:
    """
    Cria e retorna o router com as rotas de configuração LLM.

    Args:
        llm_client: Instância do LLMClient (usa singleton se None)
    """
    router = APIRouter(prefix="/api/v2/llm", tags=["llm-config"])

    # Import lazy para evitar circular imports
    def _get_llm():
        nonlocal llm_client
        if llm_client is None:
            from .llm_client import get_llm_client
            llm_client = get_llm_client()
        return llm_client

    # ------------------------------------------------------------------
    #  GET /api/v2/llm/config
    # ------------------------------------------------------------------

    @router.get("/config")
    async def get_llm_config():
        """
        Retorna a configuração atual do LLM.

        Útil para o frontend mostrar o modelo ativo, temperatura,
        timeout e estado de disponibilidade.
        """
        client = _get_llm()
        info = await client.get_model_info()

        return {
            "success": True,
            "config": info,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    #  PUT /api/v2/llm/config
    # ------------------------------------------------------------------

    @router.put("/config")
    async def update_llm_config(
        data: dict = Body(...),
    ):
        """
        Atualiza a configuração do LLM em tempo real.

        Podes alterar qualquer combinação de campos. Apenas os campos
        fornecidos serão alterados — os restantes mantêm o valor atual.

        Corpo do pedido (todos opcionais):
            model (str):       Nome do modelo (ex: "llama3.2", "mistral", "codellama")
            temperature (float): Temperatura para geração (0.0 — 2.0, default: 0.7)
            timeout (float):    Timeout em segundos (default: 60)
            max_tokens (int):   Máximo de tokens na resposta (default: 2048)
            base_url (str):     URL base do servidor LLM (ex: "http://127.0.0.1:11434")
            top_p (float):      Nucleus sampling (0.0 — 1.0, default: 0.9)

        Exemplo:
            PUT /api/v2/llm/config
            { "model": "mistral", "temperature": 0.3 }
        """
        client = _get_llm()

        # Validar campos
        valid_fields = {"model", "temperature", "timeout", "max_tokens", "base_url", "top_p"}
        unknown = [k for k in data if k not in valid_fields]
        if unknown:
            return {
                "success": False,
                "error": f"Campos desconhecidos: {', '.join(unknown)}",
                "valid_fields": sorted(valid_fields),
            }

        # Validar tipos e intervalos
        if "temperature" in data:
            try:
                t = float(data["temperature"])
                if t < 0.0 or t > 2.0:
                    return {"success": False, "error": "temperature deve estar entre 0.0 e 2.0"}
            except (TypeError, ValueError):
                return {"success": False, "error": "temperature deve ser um número (float)"}

        if "timeout" in data:
            try:
                t = float(data["timeout"])
                if t < 1.0:
                    return {"success": False, "error": "timeout deve ser >= 1 segundo"}
            except (TypeError, ValueError):
                return {"success": False, "error": "timeout deve ser um número (float)"}

        if "max_tokens" in data:
            try:
                m = int(data["max_tokens"])
                if m < 1:
                    return {"success": False, "error": "max_tokens deve ser >= 1"}
            except (TypeError, ValueError):
                return {"success": False, "error": "max_tokens deve ser um número inteiro"}

        if "top_p" in data:
            try:
                p = float(data["top_p"])
                if p < 0.0 or p > 1.0:
                    return {"success": False, "error": "top_p deve estar entre 0.0 e 1.0"}
            except (TypeError, ValueError):
                return {"success": False, "error": "top_p deve ser um número (float)"}

        # Aplicar alterações
        result = await client.reconfigure(**data)

        return {
            "success": True,
            "message": "Configuração atualizada com sucesso.",
            **result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    #  POST /api/v2/llm/check
    # ------------------------------------------------------------------

    @router.post("/check")
    async def check_llm_availability():
        """
        Força uma verificação de disponibilidade do servidor LLM.

        Útil depois de alterar o base_url ou iniciar o servidor LLM.
        """
        client = _get_llm()
        available = await client.check_availability(force=True)

        return {
            "success": True,
            "available": available,
            "config": {
                "base_url": client.config.base_url,
                "model": client.config.model,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    #  Cache Semântico
    # ------------------------------------------------------------------

    def _get_cache():
        """Obtém a instância global do cache (lazy)."""
        from .cache import get_cache
        return get_cache()

    @router.get("/cache")
    async def get_cache_stats():
        """
        Retorna estatísticas e estado do cache semântico.

        Útil para monitorizar a eficácia do cache (hit rate, entradas, TTL, etc.).
        """
        cache = _get_cache()
        return {
            "success": True,
            "stats": cache.get_stats(),
            "recent_entries": cache.get_entries_snapshot(limit=10),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @router.put("/cache")
    async def update_cache_config(
        data: dict = Body(...),
    ):
        """
        Atualiza a configuração do cache semântico em tempo real.

        Corpo do pedido (todos opcionais):
            enabled (bool):      Ativar/desativar cache (true/false)
            threshold (float):   Similaridade mínima para considerar hit (0.0 - 1.0, default: 0.85)
            max_entries (int):   Número máximo de entradas (default: 500)
            ttl_seconds (float): Tempo de vida de cada entrada em segundos (default: 3600)

        Exemplo:
            PUT /api/v2/llm/cache
            { "threshold": 0.9, "enabled": true }
        """
        cache = _get_cache()
        result = cache.configure(**data)
        return {
            "success": True,
            "message": "Configuração do cache atualizada.",
            **result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @router.delete("/cache")
    async def clear_cache():
        """
        Limpa todas as entradas do cache semântico.

        Útil para forçar o sistema a re-gerar respostas do zero ou
        depois de alterar o modelo/base_url.
        """
        cache = _get_cache()
        removed = cache.clear()
        return {
            "success": True,
            "message": f"Cache limpo. {removed} entradas removidas.",
            "removed": removed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    #  GET /api/v2/llm/models
    # ------------------------------------------------------------------

    @router.get("/models")
    async def list_available_models():
        """
        Lista os modelos disponíveis no servidor LLM.

        Útil para o utilizador saber que modelos estão instalados
        e escolher um para usar.
        """
        client = _get_llm()
        models = await client.list_models()

        return {
            "success": True,
            "models": models,
            "count": len(models),
            "active_model": client.config.model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    return router
