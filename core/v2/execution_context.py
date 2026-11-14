"""ExecutionContext — contexto imutável por pedido (Fase 5.1).

Substitui progressivamente o estado mutável do Orchestrator
(`workspace_purpose`, `_tool_context`) por um objecto imutável
criado no início de cada pedido.

Fase 5.1 apenas introduz a estrutura. `process` e `_execute_plan*`
continuam a usar os atributos do Orchestrator; a migração é 5.2.

Design:
- frozen: campos não podem ser reatribuídos.
- tool_context como MappingProxyType: leitura apenas.
- from_orchestrator: captura o estado actual sem o alterar.
- read_context: interface única que 5.2 vai usar; se existir
  `_current_context` já criado, devolve-o; senão cria um novo a
  partir do estado do Orchestrator (compatível com 5.1).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ExecutionContext:
    """Contexto de execução imutável, por pedido."""

    request_id: str
    purpose: str
    tool_context: Mapping[str, Any]
    approval_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Garante que o contexto interno é somente leitura."""
        if not isinstance(self.tool_context, MappingProxyType):
            object.__setattr__(
                self,
                "tool_context",
                MappingProxyType(dict(self.tool_context)),
            )

    @classmethod
    def from_orchestrator(
        cls,
        orch: Any,
        request_id: Optional[str] = None,
    ) -> "ExecutionContext":
        """Captura o estado actual do Orchestrator num ExecutionContext.

        Não altera o Orchestrator. Devolve um snapshot imutável que
        5.2 vai começar a usar em vez de `self._tool_context`.
        """
        raw = getattr(orch, "_tool_context", None) or {}
        ctx_dict = dict(raw) if isinstance(raw, dict) else {}
        return cls(
            request_id=request_id or f"REQ-{uuid.uuid4().hex[:8]}",
            purpose=str(getattr(orch, "workspace_purpose", "") or ""),
            tool_context=MappingProxyType(ctx_dict),
        )


def read_context(
    orch: Any,
    request_id: Optional[str] = None,
) -> ExecutionContext:
    """Retorna o contexto do pedido, criando um se ainda não existir.

    Fase 5.1: sempre cria a partir do estado do Orchestrator.
    Fase 5.2: se o Orchestrator tiver `_current_context` (um
    ExecutionContext), devolve-o; senão delega em from_orchestrator.
    Fase 5.3: só `_current_context` conta.
    """
    existing = getattr(orch, "_current_context", None)
    if isinstance(existing, ExecutionContext):
        return existing
    return ExecutionContext.from_orchestrator(orch, request_id=request_id)
