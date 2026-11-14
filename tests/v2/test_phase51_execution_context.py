"""Testes Fase 5.1 — ExecutionContext."""
import pytest

from core.v2.execution_context import ExecutionContext, read_context


def test_context_is_frozen():
    ctx = ExecutionContext(
        request_id="R1",
        purpose="analysis",
        tool_context={},
    )
    with pytest.raises(Exception):
        ctx.purpose = "development"  # type: ignore


def test_tool_context_is_read_only():
    ctx = ExecutionContext(
        request_id="R1",
        purpose="analysis",
        tool_context={"workspace_path": "/tmp/x"},
    )
    with pytest.raises(TypeError):
        ctx.tool_context["new_key"] = "value"  # type: ignore


def test_default_optional_fields_none():
    ctx = ExecutionContext(
        request_id="R1",
        purpose="analysis",
        tool_context={},
    )
    assert ctx.approval_id is None
    assert ctx.session_id is None
    assert ctx.user_id is None


def test_from_orchestrator_captures_state():
    class FakeOrch:
        workspace_purpose = "analysis"
        _tool_context = {"workspace_path": "/tmp/fake", "workspace_root": "workspace"}

    ctx = ExecutionContext.from_orchestrator(FakeOrch())
    assert ctx.purpose == "analysis"
    assert ctx.tool_context["workspace_path"] == "/tmp/fake"
    assert ctx.request_id.startswith("REQ-")


def test_from_orchestrator_uses_provided_request_id():
    class FakeOrch:
        workspace_purpose = "development"
        _tool_context = {}

    ctx = ExecutionContext.from_orchestrator(FakeOrch(), request_id="CUSTOM-42")
    assert ctx.request_id == "CUSTOM-42"
    assert ctx.purpose == "development"


def test_from_orchestrator_handles_missing_attributes():
    class EmptyOrch:
        pass

    ctx = ExecutionContext.from_orchestrator(EmptyOrch())
    assert ctx.purpose == ""
    assert dict(ctx.tool_context) == {}


def test_from_orchestrator_snapshot_isolated_from_mutations():
    """Mutar `orch._tool_context` depois não altera o ctx já criado."""
    class FakeOrch:
        workspace_purpose = "analysis"
        _tool_context = {"workspace_path": "/tmp/original"}

    orch = FakeOrch()
    ctx = ExecutionContext.from_orchestrator(orch)
    orch._tool_context["workspace_path"] = "/tmp/changed"
    assert ctx.tool_context["workspace_path"] == "/tmp/original"


def test_read_context_creates_new_when_none_exist():
    class FakeOrch:
        workspace_purpose = "analysis"
        _tool_context = {"workspace_path": "/tmp/x"}

    ctx = read_context(FakeOrch(), request_id="R-5")
    assert ctx.request_id == "R-5"
    assert ctx.purpose == "analysis"


def test_read_context_returns_existing_when_present():
    class FakeOrch:
        workspace_purpose = "analysis"
        _tool_context = {}
        _current_context = ExecutionContext(
            request_id="EXISTING-1",
            purpose="development",
            tool_context={"workspace_path": "/tmp/y"},
        )

    ctx = read_context(FakeOrch(), request_id="IGNORED")
    assert ctx.request_id == "EXISTING-1"
    assert ctx.purpose == "development"


def test_read_context_ignores_non_execution_context():
    class FakeOrch:
        workspace_purpose = "analysis"
        _tool_context = {"workspace_path": "/tmp/x"}
        _current_context = "not-an-execution-context"

    ctx = read_context(FakeOrch(), request_id="R-7")
    assert ctx.request_id == "R-7"
    assert ctx.purpose == "analysis"
