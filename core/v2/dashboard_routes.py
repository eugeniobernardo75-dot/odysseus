"""
Rotas do Dashboard — Odysseus V2

Inclui:
    - GET  /api/v2/dashboard    — JSON agregado do sistema
    - WS   /ws/dashboard        — WebSocket para atualizações em tempo real

O WebSocket transmite o mesmo payload do GET /api/v2/dashboard
periodicamente (a cada 5s) para todos os clientes conectados,
removendo a necessidade de polling.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


def setup_dashboard_routes() -> APIRouter:
    """Cria o router com endpoints REST e WebSocket do dashboard."""
    router = APIRouter(prefix="/api/v2", tags=["odysseus-dashboard"])

    # ── Lazy imports para evitar circular imports ──

    def _get_orchestrator():
        from .orchestrator import get_orchestrator
        return get_orchestrator()

    def _get_safety():
        from .safety import get_safety_manager
        return get_safety_manager()

    def _get_memory():
        from .memory import get_memory_manager
        return get_memory_manager()

    def _get_agents():
        from .agents import get_agent_factory
        return get_agent_factory()

    def _get_llm():
        from .llm_client import get_llm_client
        return get_llm_client()

    def _get_cache():
        from .cache import get_cache
        return get_cache()

    def _collect_data() -> dict:
        """
        Recolhe dados de todos os subsistemas.
        Usada tanto pelo REST endpoint como pelo broadcast WebSocket.
        """
        orchestrator = _get_orchestrator()
        memory = _get_memory()
        agents = _get_agents()
        llm_client = _get_llm()
        cache = _get_cache()
        safety = _get_safety()
        import asyncio

        data = {}

        # ── Agentes ──
        agent_list = agents.list_agents()
        data["agents"] = {"total": len(agent_list), "list": agent_list}

        # ── Memória permanente ──
        try:
            permanent = memory.recall_all_permanent()
            data["memory"] = {
                "permanent_entries": len(permanent),
                "permanent": permanent,
                "conversation_length": len(memory.get_conversation_context()),
                "pending_storage": memory.get_pending_approvals(),
            }
        except Exception as e:
            data["memory"] = {"error": str(e)}

        # ── LLM ──
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(
                    llm_client.get_model_info(), loop
                )
                data["llm"] = fut.result(timeout=5)
            else:
                data["llm"] = loop.run_until_complete(llm_client.get_model_info())
        except Exception as e:
            data["llm"] = {"error": str(e)}

        # ── Cache semântico ──
        try:
            data["cache"] = cache.get_stats()
        except Exception as e:
            data["cache"] = {"error": str(e)}

        # ── Aprovações ──
        try:
            pending = safety.get_pending()
            all_approvals = safety.get_all()
            data["approvals"] = {
                "pending_count": len(pending),
                "pending": pending[:10],
                "total_count": len(all_approvals),
            }
        except Exception as e:
            data["approvals"] = {"error": str(e)}

        # ── Planos ──
        try:
            plans = orchestrator.list_plans()
            data["plans"] = {"total": len(plans), "recent": plans[-10:]}
        except Exception as e:
            data["plans"] = {"error": str(e)}

        # ── Auditoria ──
        try:
            audit = orchestrator.get_audit_report()
            memory_data = data.get("memory", {})
            if (
                "memory" not in audit
                and isinstance(memory_data, dict)
                and "error" not in memory_data
            ):
                audit["memory"] = {
                    "permanent_entries": memory_data.get("permanent_entries", 0),
                    "conversation_length": memory_data.get("conversation_length", 0),
                    "pending_approvals": len(memory_data.get("pending_storage", [])),
                }
            data["audit"] = audit
        except Exception as e:
            data["audit"] = {"error": str(e)}

        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        data["success"] = True
        return data

    # ==================================================================
    #  GET /api/v2/dashboard  (REST — consulta única)
    # ==================================================================

    @router.get("/dashboard")
    async def get_dashboard():
        """
        Endpoint REST agregador do dashboard Odysseus V2.

        Útil para carregamento inicial ou consultas manuais.
        Para atualizações em tempo real, usa /ws/dashboard (WebSocket).
        """
        return _collect_data()

    # ==================================================================
    #  WS /ws/dashboard  (WebSocket — tempo real)
    # ==================================================================

    @router.websocket("/ws/dashboard")
    async def dashboard_websocket(ws: WebSocket):
        """
        WebSocket para receber atualizações do dashboard em tempo real.

        O servidor faz broadcast automático dos dados a cada 5s
        para todos os clientes conectados. O cliente apenas precisa
        de manter a conexão aberta e ouvir as mensagens JSON.

        Formato:
            { "agents": {...}, "memory": {...}, "llm": {...},
              "cache": {...}, "approvals": {...}, "plans": {...},
              "audit": {...}, "timestamp": "...", "success": true }
        """
        from .ws_broadcaster import get_connection_manager

        manager = get_connection_manager()
        try:
            await manager.connect(ws)
            # Manter conexão aberta — o broadcast loop envia dados
            while True:
                # Receber mensagens do cliente (pings de keepalive)
                try:
                    data = await ws.receive_text()
                    # Cliente pode enviar {"command": "ping"} para keepalive
                    if data == '{"command":"ping"}':
                        await ws.send_text('{"pong":true}')
                except WebSocketDisconnect:
                    break
        except Exception as e:
            logger.debug("WebSocket error: %s", e)
        finally:
            await manager.disconnect(ws)

    return router


# ======================================================================
#  Helper para iniciar o broadcast loop no startup da app
# ======================================================================

async def start_dashboard_broadcast() -> None:
    """Inicia o background task de broadcast do dashboard."""
    from .ws_broadcaster import start_broadcast_loop
    await start_broadcast_loop(interval=5.0)


async def stop_dashboard_broadcast() -> None:
    """Para o background task de broadcast."""
    from .ws_broadcaster import stop_broadcast_loop
    await stop_broadcast_loop()
