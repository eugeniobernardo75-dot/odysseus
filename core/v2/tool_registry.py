"""
Tool Registry — Odysseus V2

Registo central de todas as ferramentas disponíveis no sistema.

Cada ferramenta é definida por:
    - name: identificador único
    - description: descrição legível
    - parameters: schema JSON dos parâmetros
    - capabilities: lista de capacidades semânticas (ex: ["web.fetch", "file.read"])
    - permissions: permissões necessárias (ex: ["network", "filesystem.write"])
    - category: categoria funcional (web, file, code, memory, system)
    - version: versão semântica

O registry permite:
    - register / unregister tools
    - discover tools por capability, category, ou nome
    - validar parâmetros contra schema JSON
    - obter metadata completa de qualquer ferramenta
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ======================================================================
#  Schemas e Definições
# ======================================================================

@dataclass
class ToolDefinition:
    """Definição formal de uma ferramenta no registry."""
    name: str
    description: str
    parameters: dict  # JSON Schema dos parâmetros
    capabilities: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    category: str = "general"
    version: str = "1.0.0"
    requires_approval: bool = False
    timeout: float = 30.0
    author: str = "odysseus"
    tags: list[str] = field(default_factory=list)


@dataclass
class ToolResult:
    """Resultado estruturado da execução de uma ferramenta."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    tool_name: str = ""
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ======================================================================
#  Classe Base para Ferramentas
# ======================================================================

class BaseTool(ABC):
    """Classe base abstrata para todas as ferramentas do sistema."""

    def __init__(self):
        self._definition = self._build_definition()

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    @property
    def name(self) -> str:
        return self._definition.name

    @property
    def description(self) -> str:
        return self._definition.description

    @abstractmethod
    def _build_definition(self) -> ToolDefinition:
        """Cria a ToolDefinition para esta ferramenta."""
        ...

    @abstractmethod
    async def execute(self, params: dict[str, Any], context: Optional[dict] = None) -> ToolResult:
        """
        Executa a ferramenta com os parâmetros fornecidos.

        Args:
            params: Dicionário com os parâmetros (validados contra o schema).
            context: Contexto de execução (workspace path, session info, etc.).

        Returns:
            ToolResult com o resultado da execução.
        """
        ...

    def validate_params(self, params: dict) -> list[str]:
        """
        Valida os parâmetros contra o JSON Schema da definição.

        Returns:
            Lista de erros de validação (vazia se tudo OK).
        """
        errors = []
        schema = self._definition.parameters
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        # Verificar campos obrigatórios
        for field_name in required:
            if field_name not in params or params[field_name] is None:
                errors.append(f"Campo obrigatório '{field_name}' não fornecido")

        # Verificar tipos (validação básica)
        for field_name, value in params.items():
            prop = properties.get(field_name, {})
            if not prop:
                errors.append(f"Campo '{field_name}' não reconhecido")
                continue
            expected_type = prop.get("type", "string")
            if expected_type == "string" and not isinstance(value, str):
                errors.append(f"Campo '{field_name}' deve ser string")
            elif expected_type == "integer" and not isinstance(value, int):
                errors.append(f"Campo '{field_name}' deve ser inteiro")
            elif expected_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"Campo '{field_name}' deve ser número")
            elif expected_type == "array" and not isinstance(value, list):
                errors.append(f"Campo '{field_name}' deve ser lista")
            elif expected_type == "object" and not isinstance(value, dict):
                errors.append(f"Campo '{field_name}' deve ser objeto")
            elif expected_type == "boolean" and not isinstance(value, bool):
                errors.append(f"Campo '{field_name}' deve ser booleano")

        return errors


# ======================================================================
#  Tool Registry
# ======================================================================

class ToolRegistry:
    """
    Registo central de ferramentas.

    Mantém o catálogo de todas as ferramentas disponíveis no sistema,
    permitindo registo, descoberta e validação.

    Características:
        - Registo dinâmico de ferramentas
        - Descoberta por capacidade (capability), categoria, nome
        - Validação de parâmetros contra JSON Schema
        - Histórico de registo (audit trail)
        - Prevenção de duplicados
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._capability_index: dict[str, list[str]] = {}  # capability → [tool_names]
        self._category_index: dict[str, list[str]] = {}    # category → [tool_names]
        self._register_history: list[dict] = []

    # ------------------------------------------------------------------
    #  Registo
    # ------------------------------------------------------------------

    def register(self, tool: BaseTool) -> bool:
        """
        Regista uma ferramenta no catálogo.

        Se já existir uma ferramenta com o mesmo nome, substitui
        e regista no histórico.

        Returns:
            True se registou, False se substituiu uma existente.
        """
        name = tool.name
        replaced = name in self._tools

        if replaced:
            logger.info("ToolRegistry: a substituir ferramenta '%s'", name)
        else:
            logger.info("ToolRegistry: a registar ferramenta '%s'", name)

        self._tools[name] = tool

        # Atualizar índices
        self._index_tool(tool)

        # Histórico
        self._register_history.append({
            "action": "replaced" if replaced else "registered",
            "tool": name,
            "version": tool.definition.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return not replaced

    def unregister(self, name: str) -> bool:
        """
        Remove uma ferramenta do catálogo.

        Returns:
            True se removeu, False se não existia.
        """
        if name not in self._tools:
            logger.warning("ToolRegistry: tentativa de remover ferramenta '%s' inexistente", name)
            return False

        tool = self._tools.pop(name)

        # Remover dos índices
        self._deindex_tool(tool)

        self._register_history.append({
            "action": "unregistered",
            "tool": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        logger.info("ToolRegistry: ferramenta '%s' removida", name)
        return True

    # ------------------------------------------------------------------
    #  Indexação
    # ------------------------------------------------------------------

    def _index_tool(self, tool: BaseTool):
        """Indexa uma ferramenta nos índices de capacidade e categoria."""
        name = tool.name

        # Índice de capacidades
        for cap in tool.definition.capabilities:
            if cap not in self._capability_index:
                self._capability_index[cap] = []
            if name not in self._capability_index[cap]:
                self._capability_index[cap].append(name)

        # Índice de categorias
        cat = tool.definition.category
        if cat not in self._category_index:
            self._category_index[cat] = []
        if name not in self._category_index[cat]:
            self._category_index[cat].append(name)

    def _deindex_tool(self, tool: BaseTool):
        """Remove uma ferramenta dos índices."""
        name = tool.name

        for cap in self._capability_index:
            if name in self._capability_index[cap]:
                self._capability_index[cap].remove(name)

        for cat in self._category_index:
            if name in self._category_index[cat]:
                self._category_index[cat].remove(name)

    # ------------------------------------------------------------------
    #  Descoberta
    # ------------------------------------------------------------------

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Obtém uma ferramenta pelo nome."""
        return self._tools.get(name)

    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        """Obtém a definição de uma ferramenta pelo nome."""
        tool = self._tools.get(name)
        return tool.definition if tool else None

    def list_tools(self, category: Optional[str] = None) -> list[dict]:
        """
        Lista todas as ferramentas registadas, opcionalmente filtradas por categoria.

        Returns:
            Lista de dicionários com metadados das ferramentas.
        """
        tools = []
        for name, tool in self._tools.items():
            if category and tool.definition.category != category:
                continue
            tools.append({
                "name": name,
                "description": tool.description,
                "category": tool.definition.category,
                "capabilities": tool.definition.capabilities,
                "version": tool.definition.version,
                "requires_approval": tool.definition.requires_approval,
                "parameters": tool.definition.parameters,
            })
        return sorted(tools, key=lambda t: t["name"])

    def find_by_capability(self, capability: str) -> list[BaseTool]:
        """
        Encontra ferramentas que oferecem uma determinada capacidade.

        Args:
            capability: Capacidade a procurar (ex: "web.fetch", "file.read").

        Returns:
            Lista de ferramentas que oferecem essa capacidade.
        """
        tool_names = self._capability_index.get(capability, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def find_by_capabilities(self, capabilities: list[str]) -> list[BaseTool]:
        """
        Encontra ferramentas que oferecem TODAS as capacidades listadas.

        Args:
            capabilities: Lista de capacidades necessárias.

        Returns:
            Lista de ferramentas que oferecem todas as capacidades.
        """
        if not capabilities:
            return []

        result = None
        for cap in capabilities:
            tools_for_cap = set(self._capability_index.get(cap, []))
            if result is None:
                result = tools_for_cap
            else:
                result &= tools_for_cap

        return [self._tools[name] for name in (result or set()) if name in self._tools]

    def get_capabilities(self) -> list[str]:
        """Lista todas as capacidades disponíveis no sistema."""
        return sorted(self._capability_index.keys())

    def get_categories(self) -> list[str]:
        """Lista todas as categorias de ferramentas disponíveis."""
        return sorted(self._category_index.keys())

    # ------------------------------------------------------------------
    #  Utilitários
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Número de ferramentas registadas."""
        return len(self._tools)

    def get_register_history(self, limit: int = 50) -> list[dict]:
        """Obtém o histórico de registo (últimos N eventos)."""
        return self._register_history[-limit:]

    def validate_tool_execution(self, tool_name: str, params: dict) -> list[str]:
        """
        Valida se uma ferramenta existe e os parâmetros estão corretos.

        Returns:
            Lista de erros (vazia se tudo OK).
        """
        tool = self._tools.get(tool_name)
        if not tool:
            return [f"Ferramenta '{tool_name}' não encontrada no registry"]
        return tool.validate_params(params)


# ======================================================================
#  Singleton
# ======================================================================

_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Retorna a instância global do ToolRegistry."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


def reset_tool_registry() -> None:
    """Reinicia o registry (útil para testes)."""
    global _tool_registry
    _tool_registry = None
