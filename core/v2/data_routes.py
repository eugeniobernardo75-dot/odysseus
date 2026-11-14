"""
Rotas de Dados — Odysseus V2

Endpoints para expor os dados do sistema como uma API REST completa,
permitindo que o Audícios (ou qualquer outra interface) consuma a
inteligência do Odysseus sem depender de código interno.

Endpoints:
    GET    /v2/status         — Estado completo do sistema
    GET    /v2/settings       — Configuração do sistema
    GET    /v2/agents         — Listar agentes especializados
    GET    /v2/skills         — Listar skills disponíveis (chains + ferramentas + agentes)
    GET    /v2/memory         — Listar memórias permanentes
    POST   /v2/memory         — Criar/atualizar uma memória (com aprovação)
    GET    /v2/memory/{key}   — Obter uma memória específica
    DELETE /v2/memory/{key}   — Remover uma memória
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Path, Body, Query

logger = logging.getLogger(__name__)


def setup_data_routes() -> APIRouter:
    """
    Cria e retorna o router com as rotas de dados do Odysseus V2.

    Estas rotas expõem memória, skills, agentes e estado do sistema
    como uma API REST consumível por qualquer interface.
    """
    router = APIRouter(prefix="/v2", tags=["odysseus-data"])

    # ── Lazy imports para evitar circular imports ──

    def _get_orchestrator():
        from .orchestrator import get_orchestrator
        return get_orchestrator()

    def _get_memory():
        from .memory import get_memory_manager
        return get_memory_manager()

    def _get_agents():
        from .agents import get_agent_factory
        return get_agent_factory()

    def _get_safety():
        from .safety import get_safety_manager
        return get_safety_manager()

    def _get_llm():
        from .llm_client import get_llm_client
        return get_llm_client()

    def _get_cache():
        from .cache import get_cache
        return get_cache()

    def _get_tool_registry():
        from .tool_registry import get_tool_registry
        return get_tool_registry()

    # ==================================================================
    #  GET /v2/status — Estado completo do sistema
    # ==================================================================

    @router.get("/status")
    async def get_system_status():
        """
        Retorna o estado completo do sistema Odysseus V2.

        Inclui:
        - LLM: modelo ativo, disponibilidade, configuração
        - Memória: entradas permanentes, tamanho da conversa, pendentes
        - Agentes: total, lista
        - Cache: estatísticas
        - Aprovações: pendentes, total
        - Planos: total, recentes
        - Ferramentas: total, categorias
        - Skills: total
        - Workspace: sessões ativas
        - Sistema: versão, uptime

        Útil para:
        - Dashboard carregar estado inicial completo
        - Audícios mostrar indicadores de saúde do sistema
        - Diagnóstico remoto
        """
        # Inicializar subsistemas com fallback individual.
        # Cada um pode falhar independentemente sem quebrar os outros.
        # Usamos None como placeholder e cada secção trata o seu.
        data: dict[str, Any] = {
            "success": True,
            "system": {
                "name": "Odysseus V2",
                "version": "2.0.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }

        # ── Sub-sistemas (cada um com try/except isolado) ──
        orchestrator: Any = None
        memory: Any = None
        agents: Any = None
        llm_client: Any = None
        cache: Any = None
        safety: Any = None
        tool_registry: Any = None

        try:
            orchestrator = _get_orchestrator()
        except Exception as e:
            data["orchestrator"] = {"error": str(e)}

        try:
            memory = _get_memory()
        except Exception as e:
            data["memory"] = {"error": str(e)}

        try:
            agents = _get_agents()
        except Exception as e:
            data["agents"] = {"error": str(e)}

        try:
            llm_client = _get_llm()
        except Exception as e:
            data["llm"] = {"available": False, "error": str(e)}

        try:
            cache = _get_cache()
        except Exception as e:
            data["cache"] = {"error": str(e)}

        try:
            safety = _get_safety()
        except Exception as e:
            data["approvals"] = {"error": str(e)}

        try:
            tool_registry = _get_tool_registry()
        except Exception as e:
            data["tools"] = {"error": str(e)}

        # ── LLM (com guarda para None) ──
        if llm_client is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    fut = asyncio.run_coroutine_threadsafe(
                        llm_client.get_model_info(), loop
                    )
                    llm_info = fut.result(timeout=5)
                else:
                    llm_info = loop.run_until_complete(llm_client.get_model_info())
                data["llm"] = llm_info
            except Exception as e:
                data["llm"] = {"available": False, "error": str(e)}
        elif "llm" not in data:
            data["llm"] = {"available": False, "error": "LLM client não foi inicializado"}

        # ── Memória ──
        try:
            permanent = memory.recall_all_permanent()
            pending_storage = memory.get_pending_approvals()
            data["memory"] = {
                "permanent_entries": len(permanent),
                "permanent": {k: str(v)[:200] for k, v in permanent.items()},
                "conversation_length": len(memory.get_conversation_context()),
                "pending_storage_count": len(pending_storage),
            }
        except Exception as e:
            data["memory"] = {"error": str(e)}

        # ── Agentes ──
        try:
            agent_list = agents.list_agents()
            data["agents"] = {
                "total": len(agent_list),
                "list": agent_list,
            }
        except Exception as e:
            data["agents"] = {"error": str(e)}

        # ── Cache ──
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
            data["plans"] = {
                "total": len(plans),
                "recent": plans[-10:] if plans else [],
            }
        except Exception as e:
            data["plans"] = {"error": str(e)}

        # ── Ferramentas ──
        try:
            n_tools = tool_registry.count()
            tools_by_category = {}
            for t in tool_registry.list():
                tool_def = getattr(t, "definition", None)
                cat_name = tool_def.category if tool_def else "general"
                tools_by_category.setdefault(cat_name, 0)
                tools_by_category[cat_name] += 1

            data["tools"] = {
                "total": n_tools,
                "by_category": tools_by_category,
            }
        except Exception as e:
            data["tools"] = {"error": str(e)}

        # ── Chains (Skills) ──
        try:
            from .tool_chaining import get_chain_registry, register_default_chains
            reg = get_chain_registry()
            if reg.count() == 0:
                register_default_chains(reg)
            chains = reg.list()
            data["chains"] = {
                "total": len(chains),
                "list": [
                    {
                        "name": c.get("name"),
                        "description": c.get("description", ""),
                        "tags": c.get("tags", []),
                        "n_steps": len(c.get("steps", [])),
                    }
                    for c in chains
                ],
            }
        except Exception as e:
            data["chains"] = {"error": str(e)}

        # ── Workspace ──
        try:
            from .workspace import get_workspace_manager
            ws = get_workspace_manager()
            data["workspace"] = ws.get_stats()
        except Exception as e:
            data["workspace"] = {"error": str(e)}

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
                    "pending_approvals": memory_data.get("pending_storage_count", 0),
                }
            data["audit"] = audit
        except Exception as e:
            data["audit"] = {"error": str(e)}

        return data

    # ==================================================================
    #  GET /v2/settings — Configuração do sistema
    # ==================================================================

    @router.get("/settings")
    async def get_system_settings():
        """
        Retorna a configuração atual do sistema Odysseus V2.

        Inclui:
        - LLM: modelo, temperatura, timeout, max_tokens, base_url
        - Cache: ativo, threshold, max_entries
        - Workspace: ativo, sessões máximas
        - Sistema: versão

        Útil para o Audícios mostrar/preencher o painel de configurações
        e saber o estado atual de cada componente.

        Exemplo de resposta:
        ```json
        {
            "success": true,
            "settings": {
                "llm": {
                    "model": "qwen3:8b",
                    "temperature": 0.7,
                    "timeout": 60,
                    "max_tokens": 2048,
                    "base_url": "http://localhost:11434"
                },
                "cache": {
                    "enabled": true,
                    "threshold": 0.85,
                    "max_entries": 500
                },
                "workspace": {
                    "active_sessions": 1,
                    "max_sessions": 50
                }
            }
        }
        ```
        """
        settings: dict[str, Any] = {
            "system": {
                "version": "2.0.0",
                "name": "Odysseus V2",
            },
        }

        # ── LLM ──
        try:
            llm = _get_llm()
            if llm is not None:
                config = getattr(llm, "config", None)
                if config:
                    settings["llm"] = {
                        "model": getattr(config, "model", "unknown"),
                        "temperature": getattr(config, "temperature", 0.7),
                        "timeout": getattr(config, "timeout", 60),
                        "max_tokens": getattr(config, "max_tokens", 2048),
                        "base_url": getattr(config, "base_url", "http://localhost:11434"),
                    }
                else:
                    settings["llm"] = {"model": "unknown"}
            else:
                settings["llm"] = {"available": False}
        except Exception as e:
            settings["llm"] = {"error": str(e)}

        # ── Cache ──
        try:
            cache = _get_cache()
            stats = cache.get_stats()
            settings["cache"] = {
                "enabled": stats.get("enabled", False),
                "threshold": stats.get("threshold", 0.85),
                "max_entries": stats.get("max_entries", 500),
                "entries": stats.get("entries", 0),
                "hit_rate": stats.get("hit_rate", 0.0),
            }
        except Exception as e:
            settings["cache"] = {"error": str(e)}

        # ── Workspace ──
        try:
            from .workspace import get_workspace_manager
            ws = get_workspace_manager()
            ws_stats = ws.get_stats()
            settings["workspace"] = {
                "active_sessions": ws_stats.get("active_sessions", 0),
                "max_sessions": ws_stats.get("max_sessions", 50),
                "base_dir": ws_stats.get("base_dir", ""),
            }
        except Exception as e:
            settings["workspace"] = {"error": str(e)}

        return {
            "success": True,
            "settings": settings,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ==================================================================
    #  GET /v2/agents — Listar agentes disponíveis
    # ==================================================================

    @router.get("/agents")
    async def list_agents():
        """
        Lista todos os agentes especializados disponíveis no sistema.

        Cada agente inclui:
        - name: Identificador único
        - description: Descrição do que o agente faz
        - capabilities: Capacidades específicas (quando disponível)

        Exemplo de resposta:
        ```json
        {
            "success": true,
            "total": 7,
            "agents": [
                {
                    "name": "research",
                    "description": "Pesquisa e validação de informação...",
                    "type": "specialized"
                },
                ...
            ]
        }
        ```
        """
        try:
            agents = _get_agents()
            agent_list = agents.list_agents()

            # Adicionar metadados por agente
            enriched = []
            for a in agent_list:
                enriched.append({
                    "name": a.get("name", "unknown"),
                    "description": a.get("description", ""),
                    "type": "specialized",
                    "can_handle": a.get("keywords", []),
                })

            return {
                "success": True,
                "total": len(enriched),
                "agents": enriched,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error("Erro ao listar agentes: %s", e)
            return {
                "success": False,
                "error": str(e),
                "agents": [],
                "total": 0,
            }

    # ==================================================================
    #  GET /v2/skills — Listar skills disponíveis
    # ==================================================================

    @router.get("/skills")
    async def list_skills(
        category: Optional[str] = Query(
            None, description="Filtrar por categoria (chain, tool, agent)"
        ),
    ):
        """
        Lista todas as skills disponíveis no sistema.

        Skills são as capacidades que o Odysseus pode executar:
        - chains: Sequências automáticas de ferramentas (ex: web_fetch → analyze → store)
        - tools: Ferramentas individuais (ex: web.fetch, file.read)
        - agents: Agentes especializados (ex: research, code, business)

        Parâmetros:
        - category (opcional): Filtrar por "chain", "tool" ou "agent"

        Exemplo:
            GET /v2/skills?category=tool

        Útil para o Audícios mostrar o que o Odysseus sabe fazer.
        """
        try:
            skills = []

            # ── 1. Tool Chains ──
            if category is None or category == "chain":
                try:
                    from .tool_chaining import get_chain_registry, register_default_chains
                    reg = get_chain_registry()
                    if reg.count() == 0:
                        register_default_chains(reg)
                    for c in reg.list():
                        skills.append({
                            "id": f"chain.{c.get('name')}",
                            "name": c.get("name", ""),
                            "description": c.get("description", ""),
                            "category": "chain",
                            "tags": c.get("tags", []),
                            "n_steps": len(c.get("steps", [])),
                        })
                except Exception as e:
                    logger.debug("Erro ao carregar chains: %s", e)

            # ── 2. Tools ──
            if category is None or category == "tool":
                try:
                    registry = _get_tool_registry()
                    for t in registry.list():
                        tool_def = getattr(t, "definition", None)
                        capabilities = list(tool_def.capabilities) if tool_def else []
                        cat_name = tool_def.category if tool_def else "general"
                        skills.append({
                            "id": f"tool.{t.name}",
                            "name": t.name,
                            "description": tool_def.description if tool_def else "",
                            "category": "tool",
                            "tool_category": cat_name,
                            "capabilities": capabilities,
                            "n_parameters": len(tool_def.parameters.get("properties", {})) if tool_def else 0,
                        })
                except Exception as e:
                    logger.debug("Erro ao carregar tools: %s", e)

            # ── 3. Agents ──
            if category is None or category == "agent":
                try:
                    agents = _get_agents()
                    for a in agents.list_agents():
                        skills.append({
                            "id": f"agent.{a.get('name', '')}",
                            "name": a.get("name", ""),
                            "description": a.get("description", ""),
                            "category": "agent",
                            "type": "specialized",
                        })
                except Exception as e:
                    logger.debug("Erro ao carregar agentes: %s", e)

            return {
                "success": True,
                "total": len(skills),
                "skills": skills,
                "category_filter": category,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error("Erro ao listar skills: %s", e)
            return {
                "success": False,
                "error": str(e),
                "skills": [],
                "total": 0,
            }

    # ==================================================================
    #  GET /v2/memory — Listar memórias permanentes
    # ==================================================================

    @router.get("/memory")
    async def list_memory(
        category: Optional[str] = Query(
            None, description="Filtrar por categoria (general, fact, preference, skill, context)"
        ),
        search: Optional[str] = Query(
            None, description="Texto para pesquisar nas memórias (chave ou valor)"
        ),
        limit: int = Query(
            50, description="Número máximo de resultados", ge=1, le=500
        ),
    ):
        """
        Lista as memórias permanentes do sistema.

        Parâmetros:
        - category (opcional): Filtrar por categoria
        - search (opcional): Pesquisar texto nas chaves e valores
        - limit (opcional, default 50): Número máximo de resultados

        Exemplos:
            GET /v2/memory
            GET /v2/memory?category=preference
            GET /v2/memory?search=email

        Útil para o Audícios mostrar o painel Brain com as memórias
        guardadas.
        """
        try:
            memory = _get_memory()
            audit = memory.audit_log()

            # Filtrar por categoria
            if category:
                audit = [e for e in audit if e.get("category") == category]

            # Pesquisar texto
            if search:
                search_lower = search.lower()
                audit = [
                    e for e in audit
                    if search_lower in str(e.get("key", "")).lower()
                    or search_lower in str(e.get("value", "")).lower()
                ]

            # Ordenar por updated_at (mais recente primeiro)
            audit.sort(
                key=lambda e: e.get("updated_at", ""),
                reverse=True,
            )

            # Limitar
            entries = audit[:limit]

            return {
                "success": True,
                "total": len(entries),
                "total_all": len(memory.recall_all_permanent()),
                "entries": entries,
                "filters": {
                    "category": category,
                    "search": search,
                    "limit": limit,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error("Erro ao listar memória: %s", e)
            return {
                "success": False,
                "error": str(e),
                "entries": [],
                "total": 0,
            }

    # ==================================================================
    #  GET /v2/memory/{key} — Obter uma memória específica
    # ==================================================================

    @router.get("/memory/{key:path}")
    async def get_memory(
        key: str = Path(..., description="Chave da memória"),
    ):
        """
        Obtém uma memória permanente específica pela chave.

        Exemplo:
            GET /v2/memory/user_email

        Útil para o Audícios mostrar detalhes de uma memória
        específica no painel Brain.
        """
        try:
            memory = _get_memory()
            value = memory.recall_permanent(key)

            if value is None:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "MEMORY_NOT_FOUND",
                        "message": f"Memória '{key}' não encontrada.",
                        "key": key,
                    },
                )

            # Procurar metadados adicionais no audit
            audit = memory.audit_log()
            meta = next((e for e in audit if e.get("key") == key), {})

            return {
                "success": True,
                "entry": {
                    "key": key,
                    "value": value,
                    "category": meta.get("category", "general"),
                    "created_at": meta.get("created_at", ""),
                    "updated_at": meta.get("updated_at", ""),
                    "source": meta.get("source", "unknown"),
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Erro ao obter memória '%s': %s", key, e)
            return {
                "success": False,
                "error": str(e),
            }

    # ==================================================================
    #  POST /v2/memory — Criar/atualizar memória
    # ==================================================================

    @router.post("/memory")
    async def create_memory(
        data: dict = Body(...),
    ):
        """
        Cria ou atualiza uma memória permanente.

        Corpo do pedido (JSON):
        ```json
        {
            "key": "user_email",
            "value": "joao@email.com",
            "category": "fact",
            "source": "user_input"
        }
        ```

        Campos:
        - key (obrigatório): Chave única da memória
        - value (obrigatório): Valor a guardar (qualquer tipo JSON)
        - category (opcional, default "general"): Categoria (fact, preference, skill, context, general)
        - source (opcional, default "api"): Origem da informação

        Comportamento:
        - Se a chave já existir, atualiza o valor
        - Se a chave for nova, cria a entrada
        - Usa o SafetyManager para verificar se precisa de aprovação

        Exemplo:
            POST /v2/memory
            { "key": "user_name", "value": "João" }

        Resposta:
        ```json
        {
            "success": true,
            "message": "Memória 'user_name' criada com sucesso.",
            "action": "created",
            "entry": { "key": "user_name", ... },
            "requires_approval": false
        }
        ```
        """
        try:
            key = data.get("key", "").strip()
            value = data.get("value")
            category = data.get("category", "general")
            source = data.get("source", "api")

            if not key:
                return {
                    "success": False,
                    "error": "O campo 'key' é obrigatório.",
                }

            if value is None:
                return {
                    "success": False,
                    "error": "O campo 'value' é obrigatório.",
                }

            # Validar categoria
            valid_categories = {"general", "fact", "preference", "skill", "context"}
            if category not in valid_categories:
                return {
                    "success": False,
                    "error": f"Categoria inválida. Válidas: {', '.join(sorted(valid_categories))}",
                }

            memory = _get_memory()
            safety = _get_safety()

            # Verificar se já existe
            existing = memory.recall_permanent(key)
            action = "updated" if existing is not None else "created"

            # Verificar segurança (ação de escrita)
            safety_check = safety.check_action(
                action_type="memory_write",
                description=f"Guardar memória: {key} = {str(value)[:100]}",
                details={
                    "key": key,
                    "category": category,
                    "source": source,
                },
            )

            if safety_check["requires_approval"]:
                # Colocar como pendente em vez de guardar diretamente
                memory.request_permanent_storage(
                    key=key,
                    value=value,
                    category=category,
                    source=source,
                )
                return {
                    "success": True,
                    "message": f"Memória '{key}' requer aprovação antes de ser guardada.",
                    "action": "pending_approval",
                    "requires_approval": True,
                    "approval_id": safety_check["approval_id"],
                    "risk": safety_check["risk"],
                    "entry": {
                        "key": key,
                        "value": value,
                        "category": category,
                        "source": source,
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            # Guardar diretamente
            if existing is not None:
                memory.update_permanent(key=key, value=value, category=category)
            else:
                # Usar request + confirm para manter consistência
                memory.request_permanent_storage(key=key, value=value, category=category, source=source)
                memory.confirm_permanent_storage(key=key)

            return {
                "success": True,
                "message": f"Memória '{key}' {action} com sucesso.",
                "action": action,
                "requires_approval": False,
                "entry": {
                    "key": key,
                    "value": value,
                    "category": category,
                    "source": source,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error("Erro ao criar memória: %s", e)
            return {
                "success": False,
                "error": str(e),
            }

    # ==================================================================
    #  DELETE /v2/memory/{key} — Remover memória
    # ==================================================================

    @router.delete("/memory/{key:path}")
    async def delete_memory(
        key: str = Path(..., description="Chave da memória a remover"),
        force: bool = Query(
            False, description="Se true, ignora a verificação de segurança"
        ),
    ):
        """
        Remove uma memória permanente.

        Por segurança, esta operação requer aprovação explícita.
        Usa o parâmetro `force=true` para ignorar a verificação
        (apenas recomendado para administradores).

        Exemplo:
            DELETE /v2/memory/user_email

        Resposta:
        ```json
        {
            "success": true,
            "message": "Memória 'user_email' apagada com sucesso.",
            "action": "deleted"
        }
        ```
        """
        try:
            memory = _get_memory()
            safety = _get_safety()

            # Verificar se existe
            existing = memory.recall_permanent(key)
            if existing is None:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "MEMORY_NOT_FOUND",
                        "message": f"Memória '{key}' não encontrada.",
                        "key": key,
                    },
                )

            if not force:
                # Verificar segurança
                safety_check = safety.check_action(
                    action_type="memory_delete",
                    description=f"Apagar memória: {key}",
                    details={"key": key},
                )

                if safety_check["requires_approval"]:
                    return {
                        "success": False,
                        "message": f"A operação requer aprovação.",
                        "requires_approval": True,
                        "approval_id": safety_check["approval_id"],
                        "risk": safety_check["risk"],
                        "key": key,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

            # Apagar
            deleted = memory.delete_permanent(key)
            if not deleted:
                return {
                    "success": False,
                    "error": f"Não foi possível apagar a memória '{key}'.",
                }

            return {
                "success": True,
                "message": f"Memória '{key}' apagada com sucesso.",
                "action": "deleted",
                "key": key,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Erro ao apagar memória '%s': %s", key, e)
            return {
                "success": False,
                "error": str(e),
            }

    return router
