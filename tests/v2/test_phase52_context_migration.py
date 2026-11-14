"""Testes Fase 5.2 — migração de _tool_context para ExecutionContext."""
import pytest

from core.v2.agents import AgentResult
from core.v2.orchestrator import OdysseusOrchestrator
from core.v2.planner_agent import Plan, PlanStep
from core.v2.tool_registry import ToolResult


def _make_orchestrator_with_captured_context():
    """Orchestrator mínimo com tool_manager fake que captura o context."""
    orch = OdysseusOrchestrator.__new__(OdysseusOrchestrator)
    orch.workspace_purpose = "analysis"
    orch.allow_interpreters = False
    orch._tool_context = {
        "workspace_path": "/tmp/phase52-test",
        "workspace_purpose": "analysis",
        "workspace_root": "workspace",
    }
    orch._pending_plans = {}

    captured = {}

    class FakeToolManager:
        async def execute(self, tool_name, params, context=None,
                          approval_id=None, allowed_tools=None):
            captured["tool_name"] = tool_name
            captured["params"] = params
            captured["context"] = dict(context) if context is not None else None
            captured["approval_id"] = approval_id
            return ToolResult(
                success=True,
                data={"ok": True},
                error=None,
                metadata={},
                warnings=[],
            )

    orch._tool_manager = FakeToolManager()
    orch._worker_tool_allowlist = lambda worker_name: frozenset()

    return orch, captured


@pytest.mark.asyncio
async def test_tool_step_passes_snapshot_context():
    orch, captured = _make_orchestrator_with_captured_context()
    step = PlanStep(step="t", description="test", type="tool",
                    tool="shell", tool_params={"command": "echo hi"})
    plan = Plan(id="P", query="q", steps=[step])

    result = await orch._execute_tool_step(step, plan)

    assert result.success is True
    assert captured["tool_name"] == "shell"
    assert captured["context"]["workspace_path"] == "/tmp/phase52-test"
    assert captured["context"]["workspace_purpose"] == "analysis"


@pytest.mark.asyncio
async def test_tool_step_isolated_from_orchestrator_mutation():
    orch, captured = _make_orchestrator_with_captured_context()
    step = PlanStep(step="t", description="test", type="tool",
                    tool="shell", tool_params={})
    plan = Plan(id="P", query="q", steps=[step])

    orch._tool_context["workspace_path"] = "/tmp/before-call"
    await orch._execute_tool_step(step, plan)
    assert captured["context"]["workspace_path"] == "/tmp/before-call"

    orch._tool_context["workspace_path"] = "/tmp/after-call"
    assert captured["context"]["workspace_path"] == "/tmp/before-call"


@pytest.mark.asyncio
async def test_tool_step_metadata_workspace_from_snapshot():
    orch, captured = _make_orchestrator_with_captured_context()
    step = PlanStep(step="t", description="test", type="tool",
                    tool="shell", tool_params={})
    plan = Plan(id="P", query="q", steps=[step])

    result = await orch._execute_tool_step(step, plan)

    assert result.details is not None
    assert result.details.get("workspace") == "/tmp/phase52-test"


@pytest.mark.asyncio
async def test_tool_step_empty_context_handled():
    orch, captured = _make_orchestrator_with_captured_context()
    orch._tool_context = {}
    step = PlanStep(step="t", description="test", type="tool",
                    tool="shell", tool_params={})
    plan = Plan(id="P", query="q", steps=[step])

    result = await orch._execute_tool_step(step, plan)

    assert result.success is True
    assert captured["context"] == {}


@pytest.mark.asyncio
async def test_tool_step_without_tool_name_returns_error():
    orch, _ = _make_orchestrator_with_captured_context()
    step = PlanStep(step="t", description="test", type="tool", tool=None)
    plan = Plan(id="P", query="q", steps=[step])

    result = await orch._execute_tool_step(step, plan)

    assert result.success is False
    assert "sem ferramenta" in result.summary.lower() or result.error == "tool_missing"
