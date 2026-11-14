"""
Broadcaster WebSocket — Odysseus V2 Dashboard

Gerencia conexões WebSocket e transmite atualizações do dashboard
em tempo real para todos os clientes conectados, sem necessidade
de polling por parte do frontend.

Arquitetura:
    - ConnectionManager mantém um conjunto de conexões ativas.
    - broadcast() envia dados JSON para todos os clientes.
    - broadcast_loop() (background task) recolhe dados do dashboard
      a cada 5 segundos e transmite para todos os clientes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Gerencia múltiplas conexões WebSocket.

    Mantém um conjunto de websockets ativos e fornece métodos
    para conectar, desconectar e transmitir mensagens.
    """

    def __init__(self):
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        """Aceita uma nova conexão WebSocket e regista-a."""
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)
        logger.info("WebSocket conectado (%d ativos)", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        """Remove uma conexão WebSocket."""
        async with self._lock:
            self._connections.discard(ws)
        logger.info("WebSocket desconectado (%d ativos)", len(self._connections))

    async def broadcast(self, data: dict) -> int:
        """
        Envia dados JSON para TODOS os clientes conectados.

        Remove conexões que falharam (cliente desconectou).

        Returns:
            Número de clientes que receberam a mensagem.
        """
        message = json.dumps(data, ensure_ascii=False, default=str)
        dead: list[WebSocket] = []
        sent = 0

        async with self._lock:
            for ws in self._connections:
                try:
                    await ws.send_text(message)
                    sent += 1
                except Exception:
                    dead.append(ws)

            # Remover conexões mortas
            for ws in dead:
                self._connections.discard(ws)

        if dead:
            logger.debug("%d conexão(ões) WebSocket removida(s) por inatividade", len(dead))

        return sent

    @property
    def active_count(self) -> int:
        return len(self._connections)


# ── Singleton global ──
_manager: Optional[ConnectionManager] = None
_broadcast_task: Optional[asyncio.Task] = None


def get_connection_manager() -> ConnectionManager:
    """Retorna a instância global do gestor de conexões."""
    global _manager
    if _manager is None:
        _manager = ConnectionManager()
    return _manager


def reset_connection_manager() -> None:
    """Reinicia o gestor (útil para testes)."""
    global _manager, _broadcast_task
    if _manager:
        _manager._connections.clear()
    _manager = None
    _broadcast_task = None


# ======================================================================
#  Background broadcast loop
# ======================================================================

async def _collect_dashboard_data() -> dict:
    """
    Recolhe dados do dashboard (mesma lógica do endpoint /api/v2/dashboard).

    Esta função corre dentro do broadcast loop para fazer push
    de dados atualizados para todos os clientes WebSocket.
    """
    data: dict = {"success": True, "timestamp": datetime.now(timezone.utc).isoformat()}

    try:
        from .orchestrator import get_orchestrator
        orchestrator = get_orchestrator()
    except Exception:
        orchestrator = None

    try:
        from .safety import get_safety_manager
        safety = get_safety_manager()
    except Exception:
        safety = None

    try:
        from .memory import get_memory_manager
        memory = get_memory_manager()
    except Exception:
        memory = None

    try:
        from .agents import get_agent_factory
        agents = get_agent_factory()
    except Exception:
        agents = None

    try:
        from .llm_client import get_llm_client
        llm_client = get_llm_client()
    except Exception:
        llm_client = None

    try:
        from .cache import get_cache
        cache = get_cache()
    except Exception:
        cache = None

    # ── Agentes ──
    try:
        if agents:
            agent_list = agents.list_agents()
            data["agents"] = {"total": len(agent_list), "list": agent_list}
    except Exception as e:
        data["agents"] = {"error": str(e)}

    # ── Memória ──
    try:
        if memory:
            permanent = memory.recall_all_permanent()
            data["memory"] = {
                "permanent_entries": len(permanent),
                "permanent": permanent,
                "conversation_length": len(memory.get_conversation_context()),
            }
    except Exception as e:
        data["memory"] = {"error": str(e)}

    # ── LLM ──
    try:
        if llm_client:
            data["llm"] = await llm_client.get_model_info()
    except Exception as e:
        data["llm"] = {"error": str(e)}

    # ── Cache ──
    try:
        if cache:
            data["cache"] = cache.get_stats()
    except Exception as e:
        data["cache"] = {"error": str(e)}

    # ── Aprovações ──
    try:
        if safety:
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
        if orchestrator:
            plans = orchestrator.list_plans()
            data["plans"] = {"total": len(plans), "recent": plans[-10:]}
    except Exception as e:
        data["plans"] = {"error": str(e)}

    # ── Auditoria ──
    try:
        if orchestrator:
            data["audit"] = orchestrator.get_audit_report()
    except Exception as e:
        data["audit"] = {"error": str(e)}

    return data


async def start_broadcast_loop(interval: float = 5.0) -> asyncio.Task:
    """
    Inicia o background task que periodicamente recolhe dados
    do dashboard e transmite para todos os clientes WebSocket.

    Args:
        interval: Intervalo entre broadcasts em segundos (default: 5s).

    Returns:
        asyncio.Task que executa o loop.
    """
    global _broadcast_task
    if _broadcast_task and not _broadcast_task.done():
        logger.warning("Broadcast loop já está em execução.")
        return _broadcast_task

    async def _loop():
        manager = get_connection_manager()
        logger.info("Broadcast loop iniciado (intervalo=%ds)", interval)

        # Pequeno atraso inicial para o servidor estabilizar
        await asyncio.sleep(2)

        while True:
            try:
                if manager.active_count > 0:
                    data = await _collect_dashboard_data()
                    sent = await manager.broadcast(data)
                    if sent:
                        logger.debug("Broadcast enviado para %d cliente(s)", sent)
            except asyncio.CancelledError:
                logger.info("Broadcast loop cancelado.")
                break
            except Exception as e:
                logger.warning("Erro no broadcast loop: %s", e)

            await asyncio.sleep(interval)

    _broadcast_task = asyncio.create_task(_loop())
    return _broadcast_task


async def stop_broadcast_loop() -> None:
    """Para o background broadcast loop."""
    global _broadcast_task
    if _broadcast_task and not _broadcast_task.done():
        _broadcast_task.cancel()
        try:
            await _broadcast_task
        except asyncio.CancelledError:
            pass
        _broadcast_task = None
        logger.info("Broadcast loop parado.")
