"""
Testes Unitários — ToolRegistry, ToolDefinition, ToolResult, BaseTool

Cobertura:
    - ToolDefinition dataclass
    - ToolResult dataclass
    - BaseTool (abstrato)
    - ToolRegistry: register, unregister, re-register
    - ToolRegistry: list_tools, get_tool, get_definition
    - ToolRegistry: find_by_capability, find_by_capabilities
    - ToolRegistry: get_capabilities, get_categories
    - ToolRegistry: validate_tool_execution
    - ToolRegistry: get_register_history, count
    - ToolRegistry: singleton (get_tool_registry, reset_tool_registry)
    - BaseTool.validate_params: types, required, desconhecidos

Uso:
    pytest tests/v2/test_tool_registry.py -v
"""

from __future__ import annotations

import sys
import os
from dataclasses import fields
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ======================================================================
#  Helpers — Tool de teste concreto
# ======================================================================

class _MockTool:
    """Tool falsa para testar o registry sem depender de ferramentas reais."""

    def __init__(self, name="mock_tool", category="mock", capabilities=None,
                 requires_approval=False, timeout=30.0, params=None):
        from core.v2.tool_registry import ToolDefinition
        self._definition = ToolDefinition(
            name=name,
            description="Tool de teste",
            parameters=params or {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Input"},
                },
                "required": ["input"],
            },
            capabilities=capabilities or [f"{name}.action"],
            permissions=[f"{name}.read"],
            category=category,
            version="1.0.0",
            requires_approval=requires_approval,
            timeout=timeout,
            tags=["test"],
        )

    @property
    def definition(self):
        return self._definition

    @property
    def name(self):
        return self._definition.name

    @property
    def description(self):
        return self._definition.description

    async def execute(self, params, context=None):
        from core.v2.tool_registry import ToolResult
        return ToolResult(success=True, data={"result": "ok"})

    def validate_params(self, params):
        """Validação simplificada para testes."""
        from core.v2.tool_registry import BaseTool
        # Delegar para a implementação real de BaseTool
        return BaseTool.validate_params(self, params)


# ======================================================================
#  Tests: ToolDefinition dataclass
# ======================================================================

class TestToolDefinition:
    def test_default_values(self):
        from core.v2.tool_registry import ToolDefinition
        td = ToolDefinition(name="test", description="desc", parameters={})
        assert td.name == "test"
        assert td.description == "desc"
        assert td.parameters == {}
        assert td.capabilities == []
        assert td.permissions == []
        assert td.category == "general"
        assert td.version == "1.0.0"
        assert td.requires_approval is False
        assert td.timeout == 30.0
        assert td.author == "odysseus"
        assert td.tags == []

    def test_custom_values(self):
        from core.v2.tool_registry import ToolDefinition
        td = ToolDefinition(
            name="web",
            description="Fetch",
            parameters={"type": "object", "properties": {}},
            capabilities=["web.fetch"],
            permissions=["network"],
            category="web",
            version="2.0.0",
            requires_approval=True,
            timeout=60.0,
            author="dev",
            tags=["internet"],
        )
        assert td.capabilities == ["web.fetch"]
        assert td.requires_approval is True
        assert td.timeout == 60.0
        assert td.author == "dev"

    def test_all_fields_present(self):
        from core.v2.tool_registry import ToolDefinition
        field_names = {f.name for f in fields(ToolDefinition)}
        expected = {
            "name", "description", "parameters", "capabilities",
            "permissions", "category", "version", "requires_approval",
            "timeout", "author", "tags",
        }
        assert field_names == expected, f"Missing: {expected - field_names}"


# ======================================================================
#  Tests: ToolResult dataclass
# ======================================================================

class TestToolResult:
    def test_default_values(self):
        from core.v2.tool_registry import ToolResult
        r = ToolResult(success=True)
        assert r.success is True
        assert r.data is None
        assert r.error is None
        assert r.execution_time == 0.0
        assert r.tool_name == ""
        assert r.metadata == {}
        assert r.warnings == []

    def test_success_result(self):
        from core.v2.tool_registry import ToolResult
        r = ToolResult(success=True, data={"key": "value"}, tool_name="web",
                       execution_time=1.5)
        assert r.data["key"] == "value"
        assert r.tool_name == "web"
        assert r.execution_time == 1.5

    def test_error_result(self):
        from core.v2.tool_registry import ToolResult
        r = ToolResult(success=False, error="Falhou", metadata={"code": 500})
        assert r.success is False
        assert r.error == "Falhou"
        assert r.metadata["code"] == 500

    def test_all_fields_present(self):
        from core.v2.tool_registry import ToolResult
        field_names = {f.name for f in fields(ToolResult)}
        expected = {
            "success", "data", "error", "execution_time",
            "tool_name", "metadata", "warnings",
        }
        assert field_names == expected


# ======================================================================
#  Tests: ToolRegistry — Registo
# ======================================================================

class TestToolRegistry:
    def setup_method(self):
        from core.v2.tool_registry import ToolRegistry
        self.reg = ToolRegistry()
        self.tool_a = _MockTool(name="alpha", category="cat_a",
                                 capabilities=["alpha.do", "alpha.run"])
        self.tool_b = _MockTool(name="beta", category="cat_b",
                                 capabilities=["beta.do"])

    def test_register_new(self):
        assert self.reg.register(self.tool_a) is True
        assert self.reg.count() == 1

    def test_register_duplicate(self):
        self.reg.register(self.tool_a)
        # Substituir — deve retornar False
        new_a = _MockTool(name="alpha", category="cat_a", capabilities=["alpha.new"])
        assert self.reg.register(new_a) is False
        assert self.reg.count() == 1

    def test_register_multiple(self):
        self.reg.register(self.tool_a)
        self.reg.register(self.tool_b)
        assert self.reg.count() == 2

    def test_unregister_exists(self):
        self.reg.register(self.tool_a)
        assert self.reg.unregister("alpha") is True
        assert self.reg.count() == 0

    def test_unregister_not_exists(self):
        assert self.reg.unregister("ghost") is False
        assert self.reg.unregister("") is False

    def test_register_twice_unregister_once(self):
        self.reg.register(self.tool_a)
        self.reg.register(self.tool_b)
        self.reg.unregister("alpha")
        assert self.reg.count() == 1
        assert self.reg.get_tool("beta") is not None

    def test_register_history(self):
        self.reg.register(self.tool_a)
        self.reg.register(self.tool_b)
        self.reg.unregister("alpha")
        history = self.reg.get_register_history()
        assert len(history) == 3
        assert history[0]["action"] == "registered"
        assert history[1]["action"] == "registered"
        assert history[2]["action"] == "unregistered"
        assert history[0]["tool"] == "alpha"
        assert history[1]["tool"] == "beta"
        assert history[2]["tool"] == "alpha"

    def test_register_history_limit(self):
        for i in range(10):
            t = _MockTool(name=f"t{i}")
            self.reg.register(t)
        history = self.reg.get_register_history(limit=3)
        assert len(history) == 3

    def test_empty_registry(self):
        assert self.reg.count() == 0
        assert self.reg.list_tools() == []
        assert self.reg.get_capabilities() == []
        assert self.reg.get_categories() == []


# ======================================================================
#  Tests: ToolRegistry — Descoberta
# ======================================================================

class TestToolRegistryDiscovery:
    def setup_method(self):
        from core.v2.tool_registry import ToolRegistry
        self.reg = ToolRegistry()
        self.reg.register(_MockTool(name="web", category="web",
                                     capabilities=["web.fetch", "web.extract"]))
        self.reg.register(_MockTool(name="file", category="file",
                                     capabilities=["file.read", "file.write"]))
        self.reg.register(_MockTool(name="code", category="code",
                                     capabilities=["code.analyze", "file.read"]))

    def test_get_tool(self):
        tool = self.reg.get_tool("web")
        assert tool is not None
        assert tool.name == "web"
        assert self.reg.get_tool("ghost") is None

    def test_get_definition(self):
        td = self.reg.get_definition("web")
        assert td is not None
        assert td.name == "web"
        assert self.reg.get_definition("ghost") is None

    def test_list_tools(self):
        tools = self.reg.list_tools()
        assert len(tools) == 3
        # Ordenado por nome
        assert tools[0]["name"] == "code"
        assert tools[1]["name"] == "file"
        assert tools[2]["name"] == "web"

    def test_list_tools_by_category(self):
        tools = self.reg.list_tools(category="web")
        assert len(tools) == 1
        assert tools[0]["name"] == "web"

    def test_list_tools_empty_category(self):
        tools = self.reg.list_tools(category="database")
        assert tools == []

    def test_list_tools_fields(self):
        tools = self.reg.list_tools()
        for t in tools:
            assert "name" in t
            assert "description" in t
            assert "category" in t
            assert "capabilities" in t
            assert "version" in t
            assert "requires_approval" in t
            assert "parameters" in t

    def test_find_by_capability(self):
        tools = self.reg.find_by_capability("web.fetch")
        assert len(tools) == 1
        assert tools[0].name == "web"

    def test_find_by_capability_missing(self):
        tools = self.reg.find_by_capability("db.query")
        assert tools == []

    def test_find_by_capability_shared(self):
        tools = self.reg.find_by_capability("file.read")
        assert len(tools) == 2  # file e code têm file.read
        names = {t.name for t in tools}
        assert names == {"file", "code"}

    def test_find_by_capabilities_all(self):
        tools = self.reg.find_by_capabilities(["file.read"])
        assert len(tools) == 2

    def test_find_by_capabilities_intersection(self):
        tools = self.reg.find_by_capabilities(["file.read", "code.analyze"])
        assert len(tools) == 1
        assert tools[0].name == "code"

    def test_find_by_capabilities_empty(self):
        assert self.reg.find_by_capabilities([]) == []

    def test_find_by_capabilities_nonexistent(self):
        assert self.reg.find_by_capabilities(["does.not.exist"]) == []

    def test_get_capabilities(self):
        caps = self.reg.get_capabilities()
        expected = {"code.analyze", "file.read", "file.write", "web.extract", "web.fetch"}
        assert set(caps) == expected

    def test_get_categories(self):
        cats = self.reg.get_categories()
        assert set(cats) == {"code", "file", "web"}


# ======================================================================
#  Tests: ToolRegistry — Validação
# ======================================================================

class TestToolRegistryValidation:
    def setup_method(self):
        from core.v2.tool_registry import ToolRegistry
        self.reg = ToolRegistry()
        self.reg.register(_MockTool(name="test", category="test",
                                     capabilities=["test.do"]))

    def test_validate_tool_execution_ok(self):
        errors = self.reg.validate_tool_execution("test", {"input": "hello"})
        assert errors == []

    def test_validate_tool_execution_missing(self):
        errors = self.reg.validate_tool_execution("ghost", {})
        assert len(errors) == 1
        assert "não encontrada" in errors[0]

    def test_validate_tool_execution_empty_registry(self):
        from core.v2.tool_registry import ToolRegistry
        reg = ToolRegistry()
        errors = reg.validate_tool_execution("web", {"url": "http://x.com"})
        assert len(errors) == 1


# ======================================================================
#  Tests: BaseTool.validate_params
# ======================================================================

class TestBaseToolValidateParams:
    def test_required_field_missing(self):
        tool = _MockTool(name="required_test")
        errors = tool.validate_params({})
        assert len(errors) >= 1

    def test_required_field_present(self):
        tool = _MockTool(name="required_test")
        errors = tool.validate_params({"input": "ok"})
        assert len(errors) == 0

    def test_unknown_field(self):
        tool = _MockTool(name="unknown_test")
        errors = tool.validate_params({"input": "ok", "ghost": "value"})
        # ghost não está no schema
        unknown_errors = [e for e in errors if "não reconhecido" in e]
        assert len(unknown_errors) >= 1

    def test_field_type_mismatch(self):
        from core.v2.tool_registry import ToolDefinition, BaseTool
        # Criar tool com schema que exige integer
        class IntTool(BaseTool):
            def _build_definition(self):
                return ToolDefinition(
                    name="int_tool",
                    description="Requer inteiro",
                    parameters={
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer", "description": "Contagem"},
                        },
                        "required": ["count"],
                    },
                )
            async def execute(self, params, context=None):
                from core.v2.tool_registry import ToolResult
                return ToolResult(success=True)

        tool = IntTool()
        errors = tool.validate_params({"count": "not_an_int"})
        type_errors = [e for e in errors if "deve ser inteiro" in e]
        assert len(type_errors) >= 1

    def test_string_field_ok(self):
        tool = _MockTool(name="string_test")
        errors = tool.validate_params({"input": "hello"})
        assert errors == []

    def test_none_for_required(self):
        tool = _MockTool(name="none_test")
        errors = tool.validate_params({"input": None})
        assert len(errors) >= 1


# ======================================================================
#  Tests: Singleton
# ======================================================================

class TestToolRegistrySingleton:
    def setup_method(self):
        from core.v2.tool_registry import reset_tool_registry
        reset_tool_registry()

    def test_get_returns_same_instance(self):
        from core.v2.tool_registry import get_tool_registry
        r1 = get_tool_registry()
        r2 = get_tool_registry()
        assert r1 is r2

    def test_reset_creates_new(self):
        from core.v2.tool_registry import get_tool_registry, reset_tool_registry
        r1 = get_tool_registry()
        reset_tool_registry()
        r2 = get_tool_registry()
        assert r1 is not r2

    def test_reset_empty(self):
        from core.v2.tool_registry import get_tool_registry, reset_tool_registry
        reg = get_tool_registry()
        reg.register(_MockTool(name="temp"))
        assert reg.count() == 1
        reset_tool_registry()
        reg2 = get_tool_registry()
        assert reg2.count() == 0

    def test_singleton_independent_from_instance(self):
        from core.v2.tool_registry import ToolRegistry, get_tool_registry, reset_tool_registry
        reset_tool_registry()
        instance = ToolRegistry()
        singleton = get_tool_registry()
        assert instance is not singleton
        instance.register(_MockTool(name="direct"))
        assert singleton.count() == 0
