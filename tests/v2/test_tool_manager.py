"""
Testes Unitários — ToolManager

Cobertura:
    - execução com ferramenta válida e inválida
    - validação de parâmetros
    - integração com SafetyManager (approval, deny)
    - timeout
    - execute_batch, execute_parallel
    - execute_chain (delega ao ChainExecutor)
    - get_history, get_stats, get_active_executions
    - get_available_tools
    - singleton (get_tool_manager, reset_tool_manager)

Uso:
    pytest tests/v2/test_tool_manager.py -v
"""

from __future__ import annotations

import sys
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ======================================================================
#  Helper — Cria uma tool mockada
# ======================================================================

def _make_mock_tool(name="mock_tool", requires_approval=False, timeout=30.0,
                    required_params=None, capabilities=None):
    """Cria uma MagicMock que se comporta como uma ferramenta."""
    from core.v2.tool_registry import ToolDefinition
    tool = MagicMock()
    tool.name = name
    tool.description = "Mock " + name

    params_schema = {
        "type": "object",
        "properties": {},
        "required": required_params or [],
    }
    if required_params:
        for p in required_params:
            params_schema["properties"][p] = {"type": "string"}

    tool.definition = ToolDefinition(
        name=name, description=tool.description,
        parameters=params_schema,
        capabilities=capabilities or [f"{name}.do"],
        category="mock", timeout=timeout,
        requires_approval=requires_approval,
        permissions=["admin"] if requires_approval else [],
    )

    # validate_params — usa a BaseTool real para validação
    from core.v2.tool_registry import BaseTool
    tool.validate_params = MagicMock(side_effect=lambda p: BaseTool.validate_params(tool, p))

    # Garantir que _definition existe (BaseTool.validate_params acede a ele)
    tool._definition = tool.definition
    return tool


def _make_success_result(data=None):
    from core.v2.tool_registry import ToolResult
    return ToolResult(
        success=True, data=data or {"result": "ok"},
        error=None, execution_time=0.1, tool_name="mock_tool",
        metadata={}, warnings=[],
    )


# ======================================================================
#  Fixtures
# ======================================================================

@pytest.fixture
def mock_tool():
    tool = _make_mock_tool(required_params=["input"])
    tool.execute = AsyncMock(return_value=_make_success_result())
    return tool


@pytest.fixture
def registry(mock_tool):
    from core.v2.tool_registry import ToolRegistry
    reg = ToolRegistry()
    reg.register(mock_tool)
    return reg


@pytest.fixture
def manager(registry):
    from core.v2.tool_manager import ToolManager
    return ToolManager(registry=registry)


# ======================================================================
#  Tests: Execução
# ======================================================================

class TestToolManagerExecute:
    @pytest.mark.asyncio
    async def test_execute_success(self, manager):
        result = await manager.execute("mock_tool", {"input": "hello"})
        assert result.success is True
        assert result.data["result"] == "ok"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, manager):
        result = await manager.execute("ghost", {})
        assert result.success is False
        assert "não encontrada" in result.error

    @pytest.mark.asyncio
    async def test_execute_invalid_params(self, manager):
        result = await manager.execute("mock_tool", {"wrong_param": "x"})
        assert result.success is False
        assert "inválidos" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_tool_raises_exception(self, registry):
        from core.v2.tool_manager import ToolManager
        failing = _make_mock_tool(name="fail_tool", timeout=5.0)
        failing.execute = AsyncMock(side_effect=RuntimeError("Unexpected"))
        registry.register(failing)
        mgr = ToolManager(registry=registry)
        result = await mgr.execute("fail_tool", {})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_timeout(self, registry):
        from core.v2.tool_manager import ToolManager
        from core.v2.tool_registry import ToolDefinition, ToolResult
        slow = MagicMock()
        slow.name = "slow_tool"
        slow.definition = ToolDefinition(
            name="slow_tool", description="Slow",
            parameters={"type": "object", "properties": {}, "required": []},
            timeout=0.1, requires_approval=False,
        )

        async def slow_fn(params, context=None):
            import asyncio
            await asyncio.sleep(10)
            return ToolResult(success=True)

        slow.execute = slow_fn
        slow.validate_params = MagicMock(return_value=[])
        registry.register(slow)
        mgr = ToolManager(registry=registry)
        result = await mgr.execute("slow_tool", {})
        assert result.success is False
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_safety_requires_approval(self, registry):
        from core.v2.tool_manager import ToolManager
        appr = _make_mock_tool(name="appr_tool", requires_approval=True)
        appr.execute = AsyncMock()
        safety = MagicMock()
        safety.check_action.return_value = {
            "requires_approval": True,
            "message": "Precisa de confirmação",
            "approval_id": "appr-001",
        }
        registry.register(appr)
        mgr = ToolManager(registry=registry, safety_manager=safety)
        result = await mgr.execute("appr_tool", {})
        assert result.success is False
        assert "aprovação" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_safety_denied(self, registry):
        from core.v2.tool_manager import ToolManager
        denied = _make_mock_tool(name="denied_tool", requires_approval=True)
        denied.execute = AsyncMock()
        safety = MagicMock()
        safety.check_action.return_value = {"allowed": False, "message": "Negado"}
        registry.register(denied)
        mgr = ToolManager(registry=registry, safety_manager=safety)
        result = await mgr.execute("denied_tool", {})
        assert result.success is False
        assert "negada" in result.error.lower()


# ======================================================================
#  Tests: Batch e Paralelo
# ======================================================================

class TestToolManagerBatch:
    @pytest.mark.asyncio
    async def test_execute_batch(self, manager):
        results = await manager.execute_batch([
            ("mock_tool", {"input": "a"}),
            ("mock_tool", {"input": "b"}),
        ])
        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is True

    @pytest.mark.asyncio
    async def test_execute_batch_with_failure(self, manager):
        results = await manager.execute_batch([
            ("mock_tool", {"input": "a"}),
            ("ghost", {}),
        ])
        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False

    @pytest.mark.asyncio
    async def test_execute_batch_empty(self, manager):
        results = await manager.execute_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_execute_parallel(self, manager):
        results = await manager.execute_parallel([
            ("mock_tool", {"input": "a"}),
            ("mock_tool", {"input": "b"}),
        ])
        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is True


# ======================================================================
#  Tests: Chain
# ======================================================================

class TestToolManagerChain:
    @pytest.mark.asyncio
    async def test_execute_chain_delegates(self, manager):
        from core.v2.tool_chaining import ChainDefinition, ChainStep
        chain = ChainDefinition(
            name="test_chain", description="Test",
            steps=[ChainStep(tool="mock_tool", params={"input": "{input.x}"})],
        )
        result = await manager.execute_chain(chain, {"x": "hello"})
        assert result.chain_name == "test_chain"
        assert len(result.steps) == 1

    @pytest.mark.asyncio
    async def test_execute_chain_unknown_tool(self, manager):
        from core.v2.tool_chaining import ChainDefinition, ChainStep
        chain = ChainDefinition(
            name="bad_chain", description="Bad",
            steps=[ChainStep(tool="ghost_tool", params={})],
        )
        result = await manager.execute_chain(chain, {})
        assert result.success is False
        assert result.failed_step == 0


# ======================================================================
#  Tests: Estatísticas e Histórico
# ======================================================================

class TestToolManagerStats:
    @pytest.mark.asyncio
    async def test_stats_after_execution(self, manager):
        await manager.execute("mock_tool", {"input": "a"})
        stats = manager.get_stats()
        assert stats["total_executions"] == 1
        assert stats["successful"] == 1

    @pytest.mark.asyncio
    async def test_stats_after_failure(self, manager):
        await manager.execute("ghost", {})
        stats = manager.get_stats()
        assert stats["total_executions"] == 1
        # Ferramenta não encontrada -> execução não inicia, failed não incrementa

    @pytest.mark.asyncio
    async def test_stats_by_tool(self, manager):
        await manager.execute("mock_tool", {"input": "a"})
        await manager.execute("mock_tool", {"input": "b"})
        stats = manager.get_stats()
        assert stats["by_tool"]["mock_tool"]["executions"] == 2
        assert stats["by_tool"]["mock_tool"]["successful"] == 2

    def test_stats_initial(self, manager):
        stats = manager.get_stats()
        assert stats["total_executions"] == 0

    @pytest.mark.asyncio
    async def test_history_after_executions(self, manager):
        await manager.execute("mock_tool", {"input": "a"})
        await manager.execute("mock_tool", {"input": "b"})
        history = manager.get_history()
        assert len(history) == 2
        assert history[0]["tool"] == "mock_tool"
        assert history[0]["status"] == "success"

    @pytest.mark.asyncio
    async def test_history_limit(self, manager):
        for i in range(10):
            await manager.execute("mock_tool", {"input": str(i)})
        history = manager.get_history(limit=3)
        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_history_by_tool(self, registry):
        from core.v2.tool_manager import ToolManager
        other = _make_mock_tool(name="other")
        other.execute = AsyncMock(return_value=_make_success_result())
        registry.register(other)
        mgr = ToolManager(registry=registry)
        await mgr.execute("mock_tool", {"input": "a"})
        await mgr.execute("other", {})
        filtered = mgr.get_history(tool_name="mock_tool")
        assert len(filtered) == 1
        assert filtered[0]["tool"] == "mock_tool"

    @pytest.mark.asyncio
    async def test_get_active_executions(self, manager):
        assert manager.get_active_executions() == []

    def test_get_available_tools(self, manager):
        tools = manager.get_available_tools()
        assert len(tools) >= 1
        assert any(t["name"] == "mock_tool" for t in tools)

    def test_get_available_tools_by_capability(self, manager):
        # mock_tool tem capacidade "mock.do" por omissao
        tools = manager.get_available_tools(capability="mock_tool.do")
        assert len(tools) == 1
        assert tools[0]["name"] == "mock_tool"

    def test_get_available_tools_unknown_capability(self, manager):
        assert manager.get_available_tools(capability="x.y") == []


# ======================================================================
#  Tests: Safety Manager
# ======================================================================

class TestToolManagerSafety:
    def test_set_safety_manager(self, manager):
        s = MagicMock()
        manager.set_safety_manager(s)
        assert manager._safety is s

    def test_set_safety_manager_none(self, manager):
        manager.set_safety_manager(None)
        assert manager._safety is None


# ======================================================================
#  Tests: Singleton
# ======================================================================

class TestToolManagerSingleton:
    def setup_method(self):
        from core.v2.tool_manager import reset_tool_manager
        reset_tool_manager()

    def test_get_returns_same(self):
        from core.v2.tool_manager import get_tool_manager
        assert get_tool_manager() is get_tool_manager()

    def test_reset_creates_new(self):
        from core.v2.tool_manager import get_tool_manager, reset_tool_manager
        m1 = get_tool_manager()
        reset_tool_manager()
        assert get_tool_manager() is not m1

    def test_get_with_registry(self):
        from core.v2.tool_manager import get_tool_manager, reset_tool_manager
        from core.v2.tool_registry import ToolRegistry
        reset_tool_manager()
        m = get_tool_manager(registry=ToolRegistry())
        assert isinstance(m._registry, ToolRegistry)


# ======================================================================
#  Tests: ExecutionRecord
# ======================================================================

class TestExecutionRecord:
    def test_defaults(self):
        from core.v2.tool_manager import ExecutionRecord
        r = ExecutionRecord(
            id="EXEC-001", tool_name="test", params={},
            status="pending", started_at="2026-01-01T00:00:00Z",
        )
        assert r.id == "EXEC-001"
        assert r.status == "pending"
        assert r.duration_ms == 0.0
        assert r.result is None
        assert r.error is None

    def test_with_result(self):
        from core.v2.tool_manager import ExecutionRecord
        from core.v2.tool_registry import ToolResult
        r = ExecutionRecord(
            id="EXEC-002", tool_name="web", params={"url": "x"},
            status="success", started_at="2026-01-01T00:00:00Z",
            duration_ms=150.0,
            result=ToolResult(success=True, data={"ok": True}),
        )
        assert r.duration_ms == 150.0
        assert r.result.success is True


# ======================================================================
#  Tests: Aprovação real (SafetyManager real, sem mocks de decisão)
# ======================================================================
# Regressão coberta: a ação era classificada pelo literal "tool_execution",
# que devolve risco baixo — pelo que uma ferramenta com requires_approval=True
# (shell, python) executava sem aprovação, ou falhava com TypeError por passar
# um kwarg que o SafetyManager não aceitava.

class TestToolManagerSafetyReal:
    @pytest.fixture
    def safety(self):
        from core.v2.safety import SafetyManager
        return SafetyManager()

    def _guarded(self, registry, name="guarded_tool", requires_approval=True):
        tool = _make_mock_tool(name=name, requires_approval=requires_approval)
        tool.execute = AsyncMock(return_value=_make_success_result())
        registry.register(tool)
        return tool

    @pytest.mark.asyncio
    async def test_requires_approval_blocks_and_creates_pending(self, registry, safety):
        from core.v2.tool_manager import ToolManager
        tool = self._guarded(registry)
        mgr = ToolManager(registry=registry, safety_manager=safety)

        result = await mgr.execute("guarded_tool", {})

        assert result.success is False
        assert result.metadata.get("approval_id")
        assert safety.count_pending() == 1
        assert tool.execute.await_count == 0

    @pytest.mark.asyncio
    async def test_approval_allows_exactly_one_execution(self, registry, safety):
        from core.v2.tool_manager import ToolManager
        tool = self._guarded(registry)
        mgr = ToolManager(registry=registry, safety_manager=safety)

        blocked = await mgr.execute("guarded_tool", {})
        approval_id = blocked.metadata["approval_id"]
        approved, _ = safety.approve(approval_id)
        assert approved is True

        result = await mgr.execute("guarded_tool", {}, approval_id=approval_id)

        assert result.success is True
        assert tool.execute.await_count == 1

    @pytest.mark.asyncio
    async def test_denied_never_executes(self, registry, safety):
        from core.v2.tool_manager import ToolManager
        tool = self._guarded(registry)
        mgr = ToolManager(registry=registry, safety_manager=safety)

        blocked = await mgr.execute("guarded_tool", {})
        approval_id = blocked.metadata["approval_id"]
        safety.deny(approval_id)

        result = await mgr.execute("guarded_tool", {}, approval_id=approval_id)

        assert result.success is False
        assert tool.execute.await_count == 0

    @pytest.mark.asyncio
    async def test_safety_failure_is_fail_closed(self, registry):
        """Se o verificador falhar, a ferramenta não é executada."""
        from core.v2.tool_manager import ToolManager
        tool = self._guarded(registry)
        broken = MagicMock()
        broken.check_action.side_effect = TypeError("assinatura incompativel")
        mgr = ToolManager(registry=registry, safety_manager=broken)

        result = await mgr.execute("guarded_tool", {})

        assert result.success is False
        assert tool.execute.await_count == 0

    @pytest.mark.asyncio
    async def test_tool_without_requires_approval_still_runs(self, registry, safety):
        from core.v2.tool_manager import ToolManager
        tool = self._guarded(registry, name="free_tool", requires_approval=False)
        mgr = ToolManager(registry=registry, safety_manager=safety)

        result = await mgr.execute("free_tool", {})

        assert result.success is True
        assert tool.execute.await_count == 1
        assert safety.count_pending() == 0

    @pytest.mark.asyncio
    async def test_missing_safety_manager_is_fail_closed(self, registry):
        """Sem SafetyManager, uma ferramenta que exige aprovação não executa."""
        from core.v2.tool_manager import ToolManager
        tool = self._guarded(registry, name="needs_approval")
        mgr = ToolManager(registry=registry, safety_manager=None)

        result = await mgr.execute("needs_approval", {})

        assert result.success is False
        assert tool.execute.await_count == 0
        assert "aprov" in (result.error or "").lower()


# ======================================================================
#  Tests: Singleton — adopção tardia do SafetyManager
# ======================================================================

class TestToolManagerSingletonSafety:
    def setup_method(self):
        from core.v2.tool_manager import reset_tool_manager
        reset_tool_manager()

    def teardown_method(self):
        from core.v2.tool_manager import reset_tool_manager
        reset_tool_manager()

    def test_late_safety_manager_is_adopted(self):
        from core.v2.safety import SafetyManager
        from core.v2.tool_manager import get_tool_manager

        first = get_tool_manager()
        assert first._safety is None

        safety = SafetyManager()
        second = get_tool_manager(safety_manager=safety)

        assert second is first
        assert second._safety is safety

    def test_existing_safety_manager_is_not_replaced(self):
        from core.v2.safety import SafetyManager
        from core.v2.tool_manager import get_tool_manager

        original = SafetyManager()
        get_tool_manager(safety_manager=original)

        replacement = SafetyManager()
        manager = get_tool_manager(safety_manager=replacement)

        assert manager._safety is original
