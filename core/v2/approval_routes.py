"""
Rotas de Aprovação — Odysseus V2

Endpoints para o utilizador aprovar/negar ações arriscadas detetadas
pelo SafetyManager.

Endpoints:
    POST   /api/v2/approve/{approval_id}   — Aprovar ação
    POST   /api/v2/deny/{approval_id}      — Negar ação
    GET    /api/v2/approvals/pending        — Listar aprovações pendentes
    GET    /api/v2/approvals/{approval_id}  — Detalhes de uma aprovação
    GET    /api/v2/approvals                — Listar todas (auditoria)

Fluxo:
    1. Utilizador envia mensagem de alto risco
    2. Orquestrador/SafetyManager bloqueia e devolve approval_id
    3. Utilizador chama POST /api/v2/approve/APPROVAL-0001
    4. SafetyManager marca como "approved"
    5. Orquestrador pode re-processar a ação com a aprovação
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Path

logger = logging.getLogger(__name__)


def setup_approval_routes(safety_manager=None) -> APIRouter:
    """
    Cria e retorna o router com as rotas de aprovação.

    Args:
        safety_manager: Instância do SafetyManager (usa singleton se None)
    """
    router = APIRouter(prefix="/api/v2", tags=["approvals"])

    # Import lazy para evitar circular imports
    def _get_safety():
        nonlocal safety_manager
        if safety_manager is None:
            from .safety import get_safety_manager
            safety_manager = get_safety_manager()
        return safety_manager

    # ------------------------------------------------------------------
    #  POST /api/v2/approve/{approval_id}
    # ------------------------------------------------------------------

    @router.post("/approve/{approval_id}")
    async def approve_action(
        approval_id: str = Path(..., description="ID da aprovação (ex: APPROVAL-0001)"),
    ):
        """
        Aprova uma ação pendente.

        Após aprovação, o orquestrador pode re-processar a ação
        original com a aprovação em cache.
        """
        sm = _get_safety()
        success, message = sm.approve(approval_id)

        if not success:
            # Verificar se existe ou já foi processada
            existing = sm.get_approval(approval_id)
            if existing is None:
                raise HTTPException(status_code=404, detail={
                    "error": "APPROVAL_NOT_FOUND",
                    "message": message,
                    "approval_id": approval_id,
                })
            else:
                raise HTTPException(status_code=409, detail={
                    "error": "APPROVAL_ALREADY_PROCESSED",
                    "message": message,
                    "approval_id": approval_id,
                    "status": existing["status"],
                })

        return {
            "success": True,
            "message": message,
            "approval_id": approval_id,
            "status": "approved",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    #  POST /api/v2/deny/{approval_id}
    # ------------------------------------------------------------------

    @router.post("/deny/{approval_id}")
    async def deny_action(
        approval_id: str = Path(..., description="ID da aprovação (ex: APPROVAL-0001)"),
    ):
        """
        Nega uma ação pendente.

        A ação bloqueada não será executada.
        """
        sm = _get_safety()
        success, message = sm.deny(approval_id)

        if not success:
            existing = sm.get_approval(approval_id)
            if existing is None:
                raise HTTPException(status_code=404, detail={
                    "error": "APPROVAL_NOT_FOUND",
                    "message": message,
                    "approval_id": approval_id,
                })
            else:
                raise HTTPException(status_code=409, detail={
                    "error": "APPROVAL_ALREADY_PROCESSED",
                    "message": message,
                    "approval_id": approval_id,
                    "status": existing["status"],
                })

        return {
            "success": True,
            "message": message,
            "approval_id": approval_id,
            "status": "denied",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    #  GET /api/v2/approvals/pending
    # ------------------------------------------------------------------

    @router.get("/approvals/pending")
    async def list_pending_approvals():
        """
        Lista todas as aprovações pendentes (ações à espera de decisão do utilizador).
        """
        sm = _get_safety()
        pending = sm.get_pending()

        return {
            "success": True,
            "count": len(pending),
            "pending": pending,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    #  GET /api/v2/approvals/{approval_id}
    # ------------------------------------------------------------------

    @router.get("/approvals/{approval_id}")
    async def get_approval_details(
        approval_id: str = Path(..., description="ID da aprovação"),
    ):
        """
        Retorna detalhes de uma aprovação específica.
        """
        sm = _get_safety()
        approval = sm.get_approval(approval_id)

        if approval is None:
            raise HTTPException(status_code=404, detail={
                "error": "APPROVAL_NOT_FOUND",
                "message": f"ID de aprovação '{approval_id}' não encontrado.",
                "approval_id": approval_id,
            })

        return {
            "success": True,
            "approval": approval,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    #  GET /api/v2/approvals
    # ------------------------------------------------------------------

    @router.get("/approvals")
    async def list_all_approvals():
        """
        Lista TODAS as aprovações (útil para auditoria).
        """
        sm = _get_safety()
        all_approvals = sm.get_all()

        return {
            "success": True,
            "count": len(all_approvals),
            "approvals": all_approvals,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    return router
