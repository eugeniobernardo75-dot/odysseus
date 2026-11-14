"""
API Bridge — Odysseus V2

Ponte entre o endpoint HTTP /api/v2/chat e o orquestrador Odysseus V2.

Esta bridge:
    - Recebe o pedido do utilizador via HTTP.
    - Encaminha para o orquestrador Odysseus V2.
    - Processa a resposta.
    - Retorna uma resposta estruturada para o cliente.

Uso:
    from core.v2.api_bridge import process_request
    result = await process_request(question)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Optional

from .orchestrator import get_orchestrator, OrchestratorResponse
from .llm_client import get_llm_client

logger = logging.getLogger(__name__)


# ======================================================================
#  Provider OpenCode
# ======================================================================
#  O OpenCode funciona como um coding assistant terminal-based que pode
#  ser usado como modelo alternativo no bridge V2. Quando configurado
#  como provider, o OpenCode é invocado via CLI em vez da API HTTP
#  tradicional, permitindo coding assistido local com agent-loop.
#
#  Providers OpenCode suportados (detetados por heurística de URL):
#    - "opcode"   → OpenCode CLI local (zen/go)  → binário opencode
#    - "opencode" → OpenCode CLI local (zen/go)  → binário opencode
#
#  Configuração:
#    OPENCODE_BIN=/caminho/para/opencode  (default: "opencode" no PATH)
#    OPENCODE_MODEL=model_name            (modelo a usar internamente)
# ======================================================================


def is_opencode_available() -> bool:
    """Verifica se o OpenCode CLI está disponível no sistema."""
    try:
        import shutil
        bin_path = os.getenv("OPENCODE_BIN", "opencode")
        return shutil.which(bin_path) is not None or (
            os.path.isfile(bin_path) and os.access(bin_path, os.X_OK)
        )
    except Exception:
        return False


def get_opencode_info() -> dict:
    """Retorna informação sobre a configuração OpenCode."""
    return {
        "name": "opencode",
        "available": is_opencode_available(),
        "bin_path": os.getenv("OPENCODE_BIN", "opencode"),
        "model": os.getenv("OPENCODE_MODEL", ""),
        "type": "opencode_cli",
        "description": (
            "OpenCode — coding assistant local via CLI. "
            "Usa agent-loop para coding assistido no terminal."
        ),
    }


# ======================================================================
#  Interface principal
# ======================================================================

async def process_request(question: str, context: Optional[dict] = None,
                           approval_id: Optional[str] = None) -> dict:
    """
    Processa um pedido do utilizador através do orquestrador Odysseus V2.

    Args:
        question: A mensagem/pergunta do utilizador.
        context: Contexto adicional (opcional) para enriquecer a resposta.
        approval_id: ID de aprovação prévia (para re-processar ação após aprovação).

    Returns:
        Um dicionário com a resposta estruturada:
        {
            "success": bool,
            "response": str,
            "metadata": { ... },
            "agent_results": [ ... ],   # se aplicável
            "requires_approval": [ ... ], # se aplicável
            "suggested_memory": [ ... ],  # se aplicável
            "plan": { ... } | None,       # se aplicável
        }
    """
    if not question or not question.strip():
        return {
            "success": False,
            "error": "A mensagem não pode estar vazia.",
            "response": "Por favor, escreve uma mensagem para que eu possa ajudar-te.",
        }

    try:
        # Sanitizar a entrada
        question = question.strip()

        logger.info("Odysseus V2 a processar: %s...", question[:100])

        # Extrair approval_id do context se presente (enviado pelo frontend)
        effective_approval_id = approval_id
        if context and "approval_id" in context:
            effective_approval_id = context["approval_id"]

        # UserAgent coordena a resolução; o Orchestrator processa internamente.
        from .user_agent_config import get_user_agent
        user_agent = get_user_agent()
        result = await user_agent.process(question, context=context,
                                          approval_id=effective_approval_id)

        # Construir resposta estruturada
        response = _build_response(result)

        logger.info("Odysseus V2: resposta gerada com sucesso.")
        return response

    except asyncio.TimeoutError:
        logger.error("Odysseus V2: timeout ao processar pedido.")
        return {
            "success": False,
            "error": "O pedido excedeu o tempo limite.",
            "response": "Desculpa, o processamento demorou mais do que o esperado. Podes tentar novamente?",
        }

    except Exception as e:
        logger.error("Odysseus V2: erro ao processar: %s\n%s", e, traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "response": "Ocorreu um erro inesperado. Por favor, tenta novamente.",
        }


# ======================================================================
#  Funções de ferramentas
# ======================================================================

async def execute_tool(tool_name: str, params: dict,
                        context: Optional[dict] = None) -> dict:
    """
    Executa uma ferramenta através do ToolManager do orquestrador.

    Args:
        tool_name: Nome da ferramenta a executar.
        params: Parâmetros para a ferramenta.
        context: Contexto de execução (workspace, sessão, etc.).

    Returns:
        Resultado estruturado da execução.
    """
    try:
        from .orchestrator import get_orchestrator
        from .execution_policy import build_tool_context, sanitize_caller_context

        llm_client = get_llm_client()
        orch = get_orchestrator(llm_client=llm_client)

        if not orch.tool_manager:
            return {
                "success": False,
                "error": "ToolManager não inicializado no orquestrador",
            }

        # O chamador não escolhe o perímetro: as chaves reservadas são
        # descartadas e o workspace é reconstruído a partir das raízes
        # aprovadas, tal como no fluxo do Orchestrator.
        tool_context = {
            **sanitize_caller_context(context),
            **build_tool_context(
                getattr(orch, "workspace_purpose", "workspace"),
                allow_interpreters=getattr(orch, "allow_interpreters", False),
            ),
        }

        result = await orch.tool_manager.execute(
            tool_name=tool_name,
            params=params,
            context=tool_context,
        )

        return {
            "success": result.success,
            "data": result.data,
            "error": result.error,
            "execution_time": result.execution_time,
            "tool_name": result.tool_name,
            "metadata": result.metadata,
            "warnings": result.warnings,
        }

    except Exception as e:
        logger.error("Erro ao executar ferramenta '%s': %s", tool_name, e)
        return {
            "success": False,
            "error": str(e),
            "tool_name": tool_name,
        }


async def list_available_tools(capability: Optional[str] = None) -> dict:
    """
    Lista ferramentas disponíveis no sistema.

    Args:
        capability: Filtro por capacidade (opcional).

    Returns:
        Dict com lista de ferramentas disponíveis.
    """
    try:
        orch = get_orchestrator(llm_client=get_llm_client())
        if orch.tool_manager:
            tools = orch.tool_manager.get_available_tools(capability=capability)
            return {
                "success": True,
                "tools": tools,
                "count": len(tools),
                "capability_filter": capability,
            }

        return {
            "success": False,
            "error": "ToolManager não disponível",
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def execute_chain(
    chain_name: str,
    input_data: dict[str, Any],
    context: Optional[dict] = None,
) -> dict:
    """
    Executa uma ToolChain pelo nome.

    As chains são sequências predefinidas de ferramentas que se
    encadeiam automaticamente (ex: web.fetch -> code.analyze -> memory.store).

    Args:
        chain_name: Nome da chain a executar.
        input_data: Input inicial (ex: {"url": "https://..."}).
        context: Contexto de execução.

    Returns:
        Dict com resultados da chain.
    """
    try:
        from .tool_chaining import get_chain_registry, ChainRunner

        # Registar chains predefinidas se ainda não existirem
        reg = get_chain_registry()
        if reg.count() == 0:
            from .tool_chaining import register_default_chains
            register_default_chains(reg)

        # Verificar se chain existe
        chain_def = reg.get(chain_name)
        if not chain_def:
            available = [c["name"] for c in reg.list()]
            return {
                "success": False,
                "error": f"Chain '{chain_name}' não encontrada. Disponíveis: {available}",
            }

        orch = get_orchestrator(llm_client=get_llm_client())
        if not orch.tool_manager:
            return {
                "success": False,
                "error": "ToolManager não inicializado no orquestrador",
            }

        from .execution_policy import build_tool_context, sanitize_caller_context
        chain_context = {
            **sanitize_caller_context(context),
            **build_tool_context(
                getattr(orch, "workspace_purpose", "workspace"),
                allow_interpreters=getattr(orch, "allow_interpreters", False),
            ),
        }

        runner = ChainRunner(orch.tool_manager)
        result = await runner.run(chain_name, input_data, context=chain_context)

        return {
            "success": result.success,
            "chain_name": result.chain_name,
            "execution_id": result.execution_id,
            "steps": result.steps,
            "total_time_ms": result.total_time_ms,
            "error": result.error,
            "failed_step": result.failed_step,
            "n_steps": len(result.steps),
        }

    except Exception as e:
        logger.error("Erro ao executar chain '%s': %s", chain_name, e)
        return {
            "success": False,
            "error": str(e),
            "chain_name": chain_name,
        }


async def list_chains(tag: Optional[str] = None) -> dict:
    """
    Lista as chains predefinidas disponíveis.

    Args:
        tag: Filtrar por tag (opcional).

    Returns:
        Dict com lista de chains.
    """
    try:
        from .tool_chaining import get_chain_registry, register_default_chains

        reg = get_chain_registry()
        if reg.count() == 0:
            register_default_chains(reg)

        chains = reg.list(tag=tag)
        return {
            "success": True,
            "chains": chains,
            "count": len(chains),
            "tag_filter": tag,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ======================================================================
#  Funções auxiliares
# ======================================================================

def _build_response(result: OrchestratorResponse) -> dict:
    """Constrói a resposta JSON estruturada a partir do resultado do orquestrador."""
    # Adicionar info do OpenCode se disponível
    opencode_available = is_opencode_available()
    opencode_info = get_opencode_info() if opencode_available else None

    response = {
        "success": True,
        "response": result.message,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0",
            "system": "Odysseus V2",
        },
        "providers": {
            "opencode": opencode_info,
        } if opencode_available else {},
    }
    if result.metadata and "coordinated_by" in result.metadata:
        response["metadata"]["coordinated_by"] = result.metadata["coordinated_by"]

    # Adicionar resultados de agentes, se existirem
    if result.agent_results:
        response["agent_results"] = [
            {
                "task_id": r.task_id,
                "success": r.success,
                "summary": r.summary,
                "details": r.details,
                "confidence": r.confidence,
                "suggestions": r.suggestions,
            }
            for r in result.agent_results
        ]

    # Adicionar aprovações pendentes
    if result.requires_approval:
        response["requires_approval"] = result.requires_approval

    # Adicionar sugestões de memória
    if result.suggested_memory:
        response["suggested_memory"] = result.suggested_memory

    # Adicionar plano, se existir
    if result.plan:
        plan = result.plan
        response["plan"] = {
            "id": plan.id,
            "query": plan.query,
            "status": plan.status,
            "steps": plan.steps,
            "created_at": plan.created_at,
        }

    return response


# ── Síncrono (para chamadas não-assíncronas) ──

def process_request_sync(question: str, context: Optional[dict] = None,
                          approval_id: Optional[str] = None) -> dict:
    """
    Versão síncrona do process_request para contextos que não suportam async.

    Uso:
        from core.v2.api_bridge import process_request_sync
        result = process_request_sync("Olá, tudo bem?")
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(
            process_request(question, context, approval_id),
            loop,
        )
        return future.result(timeout=30)

    return loop.run_until_complete(process_request(question, context, approval_id))
