"""
Agent Config Routes — Odysseus V2
=================================

API REST para administrar os agentes personalizados do utilizador (definidos
em user_config.json) sem editar o ficheiro à mão. Todas as alterações são
gravadas automaticamente em user_config.json via UserConfig.save() e os
agentes são recarregados na AgentFactory.

Endpoints (prefixo /api/v2/agent-config):
    GET    /config                     → configuração completa + modelos + equipa virtual
    GET    /models                     → modelos disponíveis no servidor
    GET    /agents                     → lista de agentes personalizados
    POST   /agents                     → criar agente
    PUT    /agents/{name}              → atualizar agente
    DELETE /agents/{name}              → remover agente
    PATCH  /agents/{name}/toggle       → ativar/desativar
    POST   /agents/{name}/duplicate    → duplicar agente
    POST   /agents/{name}/restore      → restaurar do disco (descartar edições não gravadas)
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)


def _get_user_config():
    from .user_agent_config import UserConfig, get_user_agent
    ua = get_user_agent()
    ua.reload_config()
    return ua.user_config


def _get_agents():
    from .agents import get_agent_factory
    return get_agent_factory()


def _get_llm():
    from .llm_client import get_llm_client
    return get_llm_client()


def _reload_agent_factory() -> list[str]:
    """Recarrega os agentes personalizados na factory (hot-reload)."""
    try:
        return _get_agents().reload_custom_agents()
    except Exception as e:
        logger.warning("Erro ao recarregar agentes personalizados: %s", e)
        return []


def _find_agent(cfg, name: str) -> Optional[dict]:
    for spec in cfg.custom_agents:
        if (spec.get("name") or "").strip().lower() == name.lower():
            return spec
    return None


def _normalise_spec(spec: dict) -> dict:
    """Normaliza/valida um spec de agente personalizado."""
    name = str(spec.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="O campo 'name' é obrigatório.")
    clean = {
        "name": name,
        "description": str(spec.get("description") or "").strip(),
        "keywords": [
            str(k).strip() for k in (spec.get("keywords") or []) if str(k).strip()
        ],
        "system_prompt": str(spec.get("system_prompt") or spec.get("role") or ""),
        "model": str(spec.get("model") or "").strip() or None,
        "enabled": bool(spec.get("enabled", True)),
        "category": str(spec.get("category") or "geral").strip() or "geral",
        "avatar": str(spec.get("avatar") or "🤖").strip() or "🤖",
        "temperature": float(spec.get("temperature") or 0.7),
        "max_tokens": int(spec.get("max_tokens") or 2048),
        "tools": list(spec.get("tools") or []),
        "priority": int(spec.get("priority") or 0),
    }
    return clean


def setup_agent_config_routes() -> APIRouter:
    """Cria o router de administração de agentes personalizados."""
    router = APIRouter(prefix="/api/v2/agent-config", tags=["agent-config"])

    # ==================================================================
    #  GET /config — configuração completa
    # ==================================================================
    @router.get("/config")
    async def get_full_config():
        cfg = _get_user_config()
        models: list[str] = []
        try:
            llm = _get_llm()
            models = await llm.list_models()
        except Exception as e:
            logger.warning("Erro ao listar modelos: %s", e)

        from .virtual_team import get_virtual_team
        from .user_agent_config import get_user_agent
        return {
            "success": True,
            "config": cfg.to_dict(),
            "agents": list(cfg.custom_agents),
            "models": models,
            "virtual_team": get_virtual_team(),
            "default_model": cfg.model.get("default"),
            "useragent": get_user_agent().get_capabilities_snapshot(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ==================================================================
    #  GET /models — modelos disponíveis
    # ==================================================================
    @router.get("/models")
    async def list_models():
        models: list[str] = []
        try:
            llm = _get_llm()
            models = await llm.list_models()
        except Exception as e:
            logger.warning("Erro ao listar modelos: %s", e)
        return {"success": True, "models": models, "count": len(models)}

    # ==================================================================
    #  GET /agents — lista de agentes personalizados
    # ==================================================================
    @router.get("/agents")
    async def list_custom_agents():
        cfg = _get_user_config()
        return {
            "success": True,
            "total": len(cfg.custom_agents),
            "agents": list(cfg.custom_agents),
        }

    # ==================================================================
    #  POST /agents — criar agente
    # ==================================================================
    @router.post("/agents")
    async def create_agent(body: dict):
        cfg = _get_user_config()
        spec = _normalise_spec(body)
        if _find_agent(cfg, spec["name"]):
            raise HTTPException(
                status_code=409,
                detail=f"Já existe um agente com o nome '{spec['name']}'.",
            )

        from .safety import get_safety_manager
        check = get_safety_manager().check_action(
            action_type="create_agent",
            description=f"Criar agente '{spec['name']}' via API",
            details={"spec": spec},
            permissions=["agents.create"],
        )
        if check.get("requires_approval"):
            raise HTTPException(status_code=202, detail={
                "requires_approval": True,
                "approval_id": check["approval_id"],
                "risk": check["risk"],
                "message": check["message"],
                "next": (
                    f"POST /api/v2/approve/{check['approval_id']} e reenvia "
                    f"este pedido (a retoma automática é Fase 2)"
                ),
            })

        cfg.custom_agents.append(spec)
        if not cfg.save():
            raise HTTPException(status_code=500, detail="Erro ao gravar user_config.json.")
        registered = _reload_agent_factory()
        return {
            "success": True,
            "agent": spec,
            "registered": spec["name"] in registered,
            "message": f"Agente '{spec['name']}' criado e gravado em user_config.json.",
        }

    # ==================================================================
    #  PUT /agents/{name} — atualizar agente
    # ==================================================================
    @router.put("/agents/{name:path}")
    async def update_agent(name: str, body: dict):
        cfg = _get_user_config()
        existing = _find_agent(cfg, name)
        if existing is None:
            raise HTTPException(status_code=404, detail=f"Agente '{name}' não encontrado.")

        from .safety import get_safety_manager
        check = get_safety_manager().check_action(
            action_type="update_agent",
            description=f"Operação em '{name}' via API",
            details={"name": name},
            permissions=["agents.create"],
        )
        if check.get("requires_approval"):
            raise HTTPException(status_code=202, detail={
                "requires_approval": True,
                "approval_id": check["approval_id"],
                "risk": check["risk"],
                "message": check["message"],
            })

        spec = _normalise_spec({**existing, **body, "name": name})
        # Impede renomear para um nome já ocupado
        other = _find_agent(cfg, spec["name"])
        if other is not None and other is not existing:
            raise HTTPException(
                status_code=409,
                detail=f"Já existe um agente com o nome '{spec['name']}'.",
            )
        idx = cfg.custom_agents.index(existing)
        cfg.custom_agents[idx] = spec
        if not cfg.save():
            raise HTTPException(status_code=500, detail="Erro ao gravar user_config.json.")
        _reload_agent_factory()
        return {
            "success": True,
            "agent": spec,
            "message": f"Agente '{spec['name']}' atualizado e gravado em user_config.json.",
        }

    # ==================================================================
    #  DELETE /agents/{name} — remover agente
    # ==================================================================
    @router.delete("/agents/{name:path}")
    async def delete_agent(name: str):
        cfg = _get_user_config()
        existing = _find_agent(cfg, name)
        if existing is None:
            raise HTTPException(status_code=404, detail=f"Agente '{name}' não encontrado.")

        from .safety import get_safety_manager
        check = get_safety_manager().check_action(
            action_type="delete_agent",
            description=f"Operação em '{name}' via API",
            details={"name": name},
            permissions=["agents.create"],
        )
        if check.get("requires_approval"):
            raise HTTPException(status_code=202, detail={
                "requires_approval": True,
                "approval_id": check["approval_id"],
                "risk": check["risk"],
                "message": check["message"],
            })

        cfg.custom_agents.remove(existing)
        if not cfg.save():
            raise HTTPException(status_code=500, detail="Erro ao gravar user_config.json.")
        _reload_agent_factory()
        return {
            "success": True,
            "message": f"Agente '{name}' removido e gravado em user_config.json.",
        }

    # ==================================================================
    #  PATCH /agents/{name}/toggle — ativar/desativar
    # ==================================================================
    @router.patch("/agents/{name:path}/toggle")
    async def toggle_agent(name: str):
        cfg = _get_user_config()
        existing = _find_agent(cfg, name)
        if existing is None:
            raise HTTPException(status_code=404, detail=f"Agente '{name}' não encontrado.")

        from .safety import get_safety_manager
        check = get_safety_manager().check_action(
            action_type="toggle_agent",
            description=f"Operação em '{name}' via API",
            details={"name": name},
            permissions=["agents.create"],
        )
        if check.get("requires_approval"):
            raise HTTPException(status_code=202, detail={
                "requires_approval": True,
                "approval_id": check["approval_id"],
                "risk": check["risk"],
                "message": check["message"],
            })

        existing["enabled"] = not bool(existing.get("enabled", True))
        if not cfg.save():
            raise HTTPException(status_code=500, detail="Erro ao gravar user_config.json.")
        _reload_agent_factory()
        return {
            "success": True,
            "enabled": existing["enabled"],
            "message": (
                f"Agente '{name}' ativado." if existing["enabled"]
                else f"Agente '{name}' desativado."
            ),
        }

    # ==================================================================
    #  POST /agents/{name}/duplicate — duplicar agente
    # ==================================================================
    @router.post("/agents/{name:path}/duplicate")
    async def duplicate_agent(name: str):
        cfg = _get_user_config()
        existing = _find_agent(cfg, name)
        if existing is None:
            raise HTTPException(status_code=404, detail=f"Agente '{name}' não encontrado.")
        clone = copy.deepcopy(existing)
        base = f"{clone.get('name', 'agente')}-copia"
        new_name = base
        counter = 2
        while _find_agent(cfg, new_name):
            new_name = f"{base}-{counter}"
            counter += 1
        clone["name"] = new_name
        clone["enabled"] = True
        cfg.custom_agents.append(clone)
        if not cfg.save():
            raise HTTPException(status_code=500, detail="Erro ao gravar user_config.json.")
        _reload_agent_factory()
        return {
            "success": True,
            "agent": clone,
            "message": f"Agente '{new_name}' criado (duplicado de '{name}').",
        }

    # ==================================================================
    #  POST /agents/{name}/restore — restaurar do disco
    # ==================================================================
    @router.post("/agents/{name:path}/restore")
    async def restore_agent(name: str):
        """Restaura o agente a partir do user_config.json no disco
        (descarta edições não gravadas em memória)."""
        from .user_agent_config import UserConfig, CONFIG_FILE
        fresh = UserConfig.load()
        spec = _find_agent(fresh, name)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"Agente '{name}' não encontrado.")
        return {
            "success": True,
            "agent": spec,
            "message": f"Agente '{name}' restaurado do disco.",
        }

    return router
