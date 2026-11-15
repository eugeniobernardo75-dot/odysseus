"""Testes Fase 5.3b-1 — activacao do ContextVar em process."""
import asyncio

import pytest

from core.v2.execution_context import _current_ctx, read_context
from core.v2.orchestrator import OdysseusOrchestrator


def test_contextvar_empty_before_process():
    assert _current_ctx.get() is None


@pytest.mark.asyncio
async def test_process_activates_contextvar():
    orch = OdysseusOrchestrator.__new__(OdysseusOrchestrator)
    orch.workspace_purpose = "analysis"
    orch.allow_interpreters = False
    orch._tool_context = {"workspace_path": "/tmp/test-53b"}
    orch._pending_plans = {}

    class FakeSafety:
        def sanitize_input(self, s):
            return s

    orch.safety = FakeSafety()

    captured = {"ctx": None}

    def fake_analyse(msg):
        captured["ctx"] = read_context(orch)
        return {"type": "conversation"}

    async def fake_llm(*a, **kw):
        return "resposta"

    orch._analyse_intent = fake_analyse
    orch._handle_special_commands = lambda m: None
    orch._generate_llm_response = fake_llm
    orch._check_memory_suggestions = lambda *a, **kw: []

    try:
        await orch.process("ola")
    except Exception:
        pass

    assert captured["ctx"] is not None
    assert captured["ctx"].purpose == "analysis"
    assert captured["ctx"].tool_context["workspace_path"] == "/tmp/test-53b"


@pytest.mark.asyncio
async def test_process_resets_contextvar_after():
    orch = OdysseusOrchestrator.__new__(OdysseusOrchestrator)
    orch.workspace_purpose = "analysis"
    orch.allow_interpreters = False
    orch._tool_context = {}
    orch._pending_plans = {}

    class FakeSafety:
        def sanitize_input(self, s):
            return s

    orch.safety = FakeSafety()
    orch._analyse_intent = lambda m: {"type": "conversation"}
    orch._handle_special_commands = lambda m: None

    async def fake_llm(*a, **kw):
        return "ok"

    orch._generate_llm_response = fake_llm
    orch._check_memory_suggestions = lambda *a, **kw: []

    assert _current_ctx.get() is None
    try:
        await orch.process("teste")
    except Exception:
        pass
    assert _current_ctx.get() is None


@pytest.mark.asyncio
async def test_process_resets_contextvar_on_exception():
    orch = OdysseusOrchestrator.__new__(OdysseusOrchestrator)
    orch.workspace_purpose = "analysis"
    orch.allow_interpreters = False
    orch._tool_context = {}
    orch._pending_plans = {}

    class FakeSafety:
        def sanitize_input(self, s):
            return s

    orch.safety = FakeSafety()
    orch._analyse_intent = lambda m: {"type": "conversation"}
    orch._handle_special_commands = lambda m: None

    async def fake_llm_raises(*a, **kw):
        raise RuntimeError("boom")

    orch._generate_llm_response = fake_llm_raises
    orch._check_memory_suggestions = lambda *a, **kw: []

    with pytest.raises(RuntimeError):
        await orch.process("teste")

    assert _current_ctx.get() is None


@pytest.mark.asyncio
async def test_concurrent_processes_isolated_contextvars():
    class FakeSafety:
        def sanitize_input(self, s):
            return s

    def make_orch(purpose, workspace):
        orch = OdysseusOrchestrator.__new__(OdysseusOrchestrator)
        orch.workspace_purpose = purpose
        orch.allow_interpreters = False
        orch._tool_context = {"workspace_path": workspace}
        orch._pending_plans = {}
        orch.safety = FakeSafety()

        seen = {"purpose": None, "workspace": None}

        def fake_analyse(msg):
            ctx = read_context(orch)
            seen["purpose"] = ctx.purpose
            seen["workspace"] = ctx.tool_context.get("workspace_path")
            return {"type": "conversation"}

        async def fake_llm(*a, **kw):
            return "ok"

        orch._analyse_intent = fake_analyse
        orch._handle_special_commands = lambda m: None
        orch._generate_llm_response = fake_llm
        orch._check_memory_suggestions = lambda *a, **kw: []
        return orch, seen

    orch_a, seen_a = make_orch("analysis", "/tmp/A")
    orch_b, seen_b = make_orch("development", "/tmp/B")

    async def run_a():
        try:
            await orch_a.process("msg A")
        except Exception:
            pass

    async def run_b():
        try:
            await orch_b.process("msg B")
        except Exception:
            pass

    await asyncio.gather(run_a(), run_b())

    assert seen_a["purpose"] == "analysis"
    assert seen_a["workspace"] == "/tmp/A"
    assert seen_b["purpose"] == "development"
    assert seen_b["workspace"] == "/tmp/B"
