"""
Tool Manager — Odysseus V2

Motor de execução de ferramentas com segurança, timeouts e auditoria.

O Tool Manager:
    - Recebe pedidos de execução de ferramentas
    - Valida parâmetros contra o ToolRegistry
    - Verifica permissões com o SafetyManager
    - Executa com timeout e logging
    - Regista todas as execuções (audit trail)
    - Retorna resultados estruturados

Integra-se com:
    - ToolRegistry: validação e descoberta de ferramentas
    - SafetyManager: verificação de permissões antes de executar
    - MemoryManager: registo de operações importantes
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Collection, Optional

from .tool_registry import ToolRegistry, ToolResult, get_tool_registry
from .execution_policy import authorized_execution, redact_params

# Import adiado para evitar circular imports
# from .tool_chaining import ChainDefinition, ChainExecutor, ChainResult

logger = logging.getLogger(__name__)


# ======================================================================
#  Estruturas
# ======================================================================

@dataclass
class ToolEvent:
    """
    Evento de ferramenta emitido durante o ciclo de execução.

    Usado pelo ToolManager para notificar listeners (ex: OrchestratorAdapter)
    quando uma ferramenta inicia, termina ou encontra um erro.
    Os listeners podem então emitir eventos SSE para o cliente.
    """
    event_type: str  # "tool_start", "tool_output", "tool_error"
    tool_name: str
    execution_id: str = ""
    params: dict | None = None
    result: Any = None
    error: str | None = None
    duration_ms: float | None = None
    timestamp: str = ""
    status: str = ""  # "pending", "running", "success", "failed", "timeout", "denied"


@dataclass
class ExecutionRecord:
    """Registo de uma execução de ferramenta (audit trail)."""
    id: str
    tool_name: str
    params: dict
    status: str  # pending, running, success, failed, denied, timeout
    started_at: str
    finished_at: Optional[str] = None
    duration_ms: float = 0.0
    result: Optional[ToolResult] = None
    error: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


# Tipo do callback de evento de ferramenta
ToolEventCallback = Callable[[ToolEvent], Awaitable[None]]


# ======================================================================
#  Tool Manager
# ======================================================================

class ToolManager:
    """
    Motor de execução de ferramentas.

    Gerencia o ciclo de vida completo da execução de uma ferramenta:
    validação → segurança → execução → auditoria.

    Características:
        - Timeout configurável por ferramenta
        - Integração com SafetyManager
        - Audit trail completo de todas as execuções
        - Execução concorrente segura
        - Estatísticas de utilização
        - Event callbacks para notificação em tempo real
    """

    def __init__(self, registry: Optional[ToolRegistry] = None,
                 safety_manager: Optional[Any] = None):
        self._registry = registry or get_tool_registry()
        self._safety = safety_manager
        self._execution_history: list[ExecutionRecord] = []
        self._active_executions: dict[str, ExecutionRecord] = {}
        self._stats: dict = {
            "total_executions": 0,
            "successful": 0,
            "failed": 0,
            "denied": 0,
            "timeouts": 0,
            "by_tool": {},
        }
        self._event_callbacks: list[ToolEventCallback] = []

    # ------------------------------------------------------------------
    #  Configuração
    # ------------------------------------------------------------------

    def set_safety_manager(self, safety_manager: Any) -> None:
        """Define o SafetyManager para verificação de permissões."""
        self._safety = safety_manager

    # ------------------------------------------------------------------
    #  Event Callbacks
    # ------------------------------------------------------------------

    def add_event_callback(self, callback: ToolEventCallback) -> None:
        """
        Regista um callback para ser chamado quando eventos de ferramenta ocorrerem.

        O callback recebe um ToolEvent com os detalhes do evento:
        - tool_start: quando a ferramenta começa a executar
        - tool_output: quando a ferramenta termina com sucesso
        - tool_error: quando a ferramenta falha

        Args:
            callback: Função assíncrona que recebe ToolEvent.
        """
        if callback not in self._event_callbacks:
            self._event_callbacks.append(callback)

    def remove_event_callback(self, callback: ToolEventCallback) -> None:
        """Remove um callback previamente registado."""
        self._event_callbacks = [
            cb for cb in self._event_callbacks if cb is not callback
        ]

    def clear_event_callbacks(self) -> None:
        """Remove todos os callbacks registados."""
        self._event_callbacks.clear()

    async def _fire_event(self, event: ToolEvent) -> None:
        """Notifica todos os listeners registados sobre um evento."""
        for callback in self._event_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.debug(
                    "ToolManager: callback falhou para evento %s: %s",
                    event.event_type, e,
                )

    # ------------------------------------------------------------------
    #  Execução
    # ------------------------------------------------------------------

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
        context: Optional[dict] = None,
        approval_id: Optional[str] = None,
        allowed_tools: Optional[Collection[str]] = None,
    ) -> ToolResult:
        """
        Executa uma ferramenta com validação, segurança e auditoria.

        Fluxo:
            1. Validar ferramenta existe no registry
            2. Validar parâmetros
            3. Verificar permissões (SafetyManager)
            4. Executar com timeout
            5. Registrar no audit trail
            6. Retornar resultado

        Args:
            tool_name: Nome da ferramenta a executar.
            params: Parâmetros para a ferramenta.
            context: Contexto de execução (workspace, sessão, etc.).
            approval_id: ID de aprovação prévia (opcional).

        Returns:
            ToolResult com o resultado da execução.
        """
        context = context or {}
        # Os parâmetros podem conter segredos (tokens, palavras-passe em
        # comandos): o que fica em registo e em eventos é a versão redigida.
        # A ferramenta recebe os parâmetros originais.
        safe_params = redact_params(params)
        record = ExecutionRecord(
            id=f"EXEC-{uuid.uuid4().hex[:8].upper()}",
            tool_name=tool_name,
            params=safe_params,
            status="pending",
            started_at=datetime.now(timezone.utc).isoformat(),
            user_id=context.get("user_id"),
            session_id=context.get("session_id"),
        )

        # ── 1. Obter ferramenta do registry ──
        tool = self._registry.get_tool(tool_name)
        if not tool:
            record.status = "failed"
            record.error = f"Ferramenta '{tool_name}' não encontrada"
            self._finalize_record(record)
            await self._fire_event(ToolEvent(
                event_type="tool_error",
                tool_name=tool_name,
                execution_id=record.id,
                error=record.error,
                timestamp=record.started_at,
                status="failed",
            ))
            return ToolResult(
                success=False,
                error=record.error,
                tool_name=tool_name,
            )

        # ── 2. Validar parâmetros ──
        validation_errors = tool.validate_params(params)
        if validation_errors:
            record.status = "failed"
            record.error = f"Parâmetros inválidos: {'; '.join(validation_errors)}"
            self._finalize_record(record)
            await self._fire_event(ToolEvent(
                event_type="tool_error",
                tool_name=tool_name,
                execution_id=record.id,
                params=safe_params,
                error=record.error,
                timestamp=record.started_at,
                status="failed",
            ))
            return ToolResult(
                success=False,
                error=record.error,
                tool_name=tool_name,
            )

        required_capability = (
            f"file.{params.get('action')}" if tool_name == "file" else tool_name
        )
        if allowed_tools is not None and required_capability not in allowed_tools:
            record.status = "denied"
            record.error = (
                f"O agente delegado não está autorizado a usar "
                f"'{required_capability}'."
            )
            record.result = ToolResult(
                success=False,
                error=record.error,
                tool_name=tool_name,
                metadata={
                    "denied": True,
                    "reason": "worker_tool_not_allowed",
                    "capability": required_capability,
                },
            )
            self._finalize_record(record)
            self._stats["denied"] += 1
            await self._fire_event(ToolEvent(
                event_type="tool_error",
                tool_name=tool_name,
                execution_id=record.id,
                params=safe_params,
                error=record.error,
                timestamp=record.started_at,
                status="denied",
            ))
            return record.result

        # ── 3. Verificar permissões ──
        # A decisão identifica a ferramenta e as permissões declaradas, em vez
        # do literal "tool_execution" — que se classificava como risco baixo e
        # nunca exigia aprovação. Qualquer insuficiência do controlo é
        # fail-closed: a ferramenta não é executada.
        approval_required = tool.definition.requires_approval or (
            tool_name == "file" and params.get("action") in {"write", "delete"}
        )
        if approval_required:
            if not self._safety:
                # Sem SafetyManager não há como obter aprovação: negar, em vez
                # de executar uma ferramenta que a exige.
                record.status = "denied"
                record.error = (
                    f"A ferramenta '{tool_name}' exige aprovação mas não existe "
                    f"SafetyManager configurado — execução negada."
                )
                record.result = ToolResult(
                    success=False,
                    error=record.error,
                    tool_name=tool_name,
                )
                self._finalize_record(record)
                self._stats["denied"] += 1
                await self._fire_event(ToolEvent(
                    event_type="tool_error",
                    tool_name=tool_name,
                    execution_id=record.id,
                    params=safe_params,
                    error=record.error,
                    timestamp=record.started_at,
                    status="denied",
                ))
                return record.result

            permissions = list(tool.definition.permissions or [])
            try:
                safety_check = self._safety.check_action(
                    action_type=f"tool:{tool_name}",
                    description=(
                        f"Executar '{tool_name}': "
                        f"{json.dumps(params, default=str)[:200]}"
                    ),
                    details={
                        "tool": tool_name,
                        "permissions": permissions,
                        "params": safe_params,
                    },
                    existing_approval_id=approval_id,
                    permissions=permissions,
                    # O hash liga a operação real (parâmetros completos); o que
                    # fica registado e visível é `details`, já redigido.
                    fingerprint_material={
                        "tool": tool_name,
                        "params": params,
                        "permissions": permissions,
                    },
                )
            except Exception as e:
                logger.error(
                    "Verificação de segurança falhou para '%s': %s", tool_name, e
                )
                safety_check = {
                    "allowed": False,
                    "requires_approval": False,
                    "message": (
                        f"Verificação de segurança indisponível ({e}); "
                        f"a ferramenta não foi executada."
                    ),
                }

            if safety_check.get("requires_approval"):
                # Ação precisa de aprovação
                record.status = "denied"
                approval_id = safety_check.get("approval_id")
                record.error = (
                    f"Requer aprovação: {safety_check.get('message', '')}"
                )
                record.result = ToolResult(
                    success=False,
                    error=record.error,
                    tool_name=tool_name,
                    metadata={
                        "approval_id": approval_id,
                        "requires_approval": True,
                    },
                )
                self._finalize_record(record)
                self._stats["denied"] += 1
                await self._fire_event(ToolEvent(
                    event_type="tool_error",
                    tool_name=tool_name,
                    execution_id=record.id,
                    params=safe_params,
                    error=record.error,
                    timestamp=record.started_at,
                    status="denied",
                ))
                return record.result

            if not safety_check.get("allowed", True):
                # Ação negada
                record.status = "denied"
                record.result = ToolResult(
                    success=False,
                    error=f"Ação negada: {safety_check.get('message', 'Permissão negada')}",
                    tool_name=tool_name,
                )
                self._finalize_record(record)
                self._stats["denied"] += 1
                await self._fire_event(ToolEvent(
                    event_type="tool_error",
                    tool_name=tool_name,
                    execution_id=record.id,
                    params=safe_params,
                    error=record.error,
                    timestamp=record.started_at,
                    status="denied",
                ))
                return record.result

        # ── 4. Executar com timeout ──
        record.status = "running"
        self._active_executions[record.id] = record

        # Emitir tool_start
        await self._fire_event(ToolEvent(
            event_type="tool_start",
            tool_name=tool_name,
            execution_id=record.id,
            params=safe_params,
            timestamp=record.started_at,
            status="running",
        ))

        timeout = tool.definition.timeout
        start = time.monotonic()
        try:
            # O marcador de execução autorizada só é activado aqui, depois de o
            # gate de aprovação ter passado. É o que permite às ferramentas que
            # exigem aprovação recusarem chamadas directas fora do ToolManager.
            with authorized_execution(tool_name, record.id):
                result = await asyncio.wait_for(
                    tool.execute(params, context=context),
                    timeout=timeout,
                )
            duration = time.monotonic() - start

            result.execution_time = duration
            result.tool_name = tool_name

            record.status = "success" if result.success else "failed"
            record.result = result
            record.duration_ms = round(duration * 1000, 1)

            if result.success:
                self._stats["successful"] += 1
            else:
                self._stats["failed"] += 1

            # Emitir tool_output (success)
            await self._fire_event(ToolEvent(
                event_type="tool_output",
                tool_name=tool_name,
                execution_id=record.id,
                params=safe_params,
                result=result.data if result.success else result.error,
                duration_ms=record.duration_ms,
                timestamp=record.finished_at or datetime.now(timezone.utc).isoformat(),
                status=record.status,
            ))

        except asyncio.TimeoutError:
            duration = time.monotonic() - start
            record.status = "timeout"
            record.error = f"Timeout após {timeout}s na ferramenta '{tool_name}'"
            record.duration_ms = round(timeout * 1000)
            self._stats["timeouts"] += 1
            result = ToolResult(
                success=False,
                error=record.error,
                tool_name=tool_name,
                execution_time=timeout,
                metadata={"timeout": timeout},
            )
            record.result = result

            # Emitir tool_error (timeout)
            await self._fire_event(ToolEvent(
                event_type="tool_error",
                tool_name=tool_name,
                execution_id=record.id,
                params=safe_params,
                error=record.error,
                duration_ms=record.duration_ms,
                timestamp=datetime.now(timezone.utc).isoformat(),
                status="timeout",
            ))

        except Exception as e:
            duration = time.monotonic() - start
            record.status = "failed"
            record.error = str(e)
            record.duration_ms = round(duration * 1000, 1)
            self._stats["failed"] += 1
            logger.error("ToolManager: erro ao executar '%s': %s\n%s",
                         tool_name, e, traceback.format_exc())
            result = ToolResult(
                success=False,
                error=f"Erro na execução: {e}",
                tool_name=tool_name,
                execution_time=duration,
                metadata={"exception": str(e)},
            )
            record.result = result

            # Emitir tool_error
            await self._fire_event(ToolEvent(
                event_type="tool_error",
                tool_name=tool_name,
                execution_id=record.id,
                params=safe_params,
                error=record.error,
                duration_ms=record.duration_ms,
                timestamp=datetime.now(timezone.utc).isoformat(),
                status="failed",
            ))

        finally:
            self._active_executions.pop(record.id, None)

        # ── 5. Finalizar e registar ──
        self._finalize_record(record)
        self._update_tool_stats(tool_name, result.success)

        return result

    # ------------------------------------------------------------------
    #  Auditoria e Estatísticas
    # ------------------------------------------------------------------

    def _finalize_record(self, record: ExecutionRecord):
        """Finaliza um registo de execução."""
        record.finished_at = datetime.now(timezone.utc).isoformat()
        self._execution_history.append(record)
        self._stats["total_executions"] += 1

        # Log
        logger.info(
            "Tool '%s' → %s (%.1fms)",
            record.tool_name, record.status, record.duration_ms,
        )

    def _update_tool_stats(self, tool_name: str, success: bool):
        """Atualiza estatísticas por ferramenta."""
        if tool_name not in self._stats["by_tool"]:
            self._stats["by_tool"][tool_name] = {
                "executions": 0,
                "successful": 0,
                "failed": 0,
            }
        stats = self._stats["by_tool"][tool_name]
        stats["executions"] += 1
        if success:
            stats["successful"] += 1
        else:
            stats["failed"] += 1

    def get_history(self, limit: int = 50, tool_name: Optional[str] = None) -> list[dict]:
        """
        Obtém o histórico de execuções.

        Args:
            limit: Número máximo de registos.
            tool_name: Filtrar por ferramenta (opcional).

        Returns:
            Lista de registos de execução.
        """
        history = self._execution_history
        if tool_name:
            history = [r for r in history if r.tool_name == tool_name]
        return [
            {
                "id": r.id,
                "tool": r.tool_name,
                "params_summary": str(r.params)[:100],
                "status": r.status,
                "duration_ms": r.duration_ms,
                "started_at": r.started_at,
                "error": r.error,
            }
            for r in history[-limit:]
        ]

    def get_stats(self) -> dict:
        """Obtém estatísticas de utilização do sistema de ferramentas."""
        return dict(self._stats)

    def get_active_executions(self) -> list[dict]:
        """Obtém lista de execuções em curso."""
        return [
            {
                "id": e.id,
                "tool": e.tool_name,
                "started_at": e.started_at,
            }
            for e in self._active_executions.values()
        ]

    def get_available_tools(self, capability: Optional[str] = None) -> list[dict]:
        """
        Lista ferramentas disponíveis, opcionalmente filtradas por capacidade.

        Args:
            capability: Capacidade específica (ex: "web.fetch").

        Returns:
            Lista de metadados das ferramentas disponíveis.
        """
        if capability:
            return [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.definition.parameters,
                    "category": t.definition.category,
                    "capabilities": t.definition.capabilities,
                }
                for t in self._registry.find_by_capability(capability)
            ]
        return self._registry.list_tools()

    async def execute_chain(
        self,
        chain: Any,
        input_data: dict[str, Any],
        context: Optional[dict] = None,
    ) -> Any:
        """
        Executa uma ToolChain completa.

        Esta é uma porta de entrada para o sistema de chaining.
        O ChainExecutor faz o trabalho pesado de resolução de variáveis
        e execução sequencial dos passos.

        Args:
            chain: ChainDefinition com os passos a executar.
            input_data: Input inicial do utilizador.
            context: Contexto de execução.

        Returns:
            ChainResult com resultados de todos os passos.
        """
        from .tool_chaining import ChainExecutor
        executor = ChainExecutor(self, registry=self._registry)
        return await executor.execute(chain, input_data, context=context)

    async def execute_batch(
        self,
        executions: list[tuple[str, dict]],
        context: Optional[dict] = None,
    ) -> list[ToolResult]:
        """
        Executa múltiplas ferramentas em sequência.

        Args:
            executions: Lista de (tool_name, params).
            context: Contexto partilhado.

        Returns:
            Lista de resultados na mesma ordem.
        """
        results = []
        for tool_name, params in executions:
            result = await self.execute(tool_name, params, context=context)
            results.append(result)
        return results

    async def execute_parallel(
        self,
        executions: list[tuple[str, dict]],
        context: Optional[dict] = None,
    ) -> list[ToolResult]:
        """
        Executa múltiplas ferramentas em paralelo.

        Args:
            executions: Lista de (tool_name, params).
            context: Contexto partilhado.

        Returns:
            Lista de resultados na mesma ordem.
        """
        tasks = [
            self.execute(tool_name, params, context=context)
            for tool_name, params in executions
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)


# ======================================================================
#  Singleton
# ======================================================================

_tool_manager: Optional[ToolManager] = None


def get_tool_manager(registry: Optional[ToolRegistry] = None,
                     safety_manager: Optional[Any] = None) -> ToolManager:
    """Retorna a instância global do ToolManager.

    Um SafetyManager fornecido depois de a instância existir é adoptado em vez
    de descartado: sem isto, um primeiro acesso sem verificador deixaria a
    aprovação desligada em silêncio para o resto do processo.
    """
    global _tool_manager
    if _tool_manager is None:
        _tool_manager = ToolManager(registry=registry, safety_manager=safety_manager)
    elif safety_manager is not None and _tool_manager._safety is None:
        _tool_manager.set_safety_manager(safety_manager)
    return _tool_manager


def reset_tool_manager() -> None:
    """Reinicia o ToolManager (útil para testes)."""
    global _tool_manager
    _tool_manager = None
