"""Testes Fase 5.3a — ContextVar no execution_context."""
import asyncio
import pytest

from core.v2.execution_context import (
    ExecutionContext,
    _current_ctx,
    read_context,
)


def test_contextvar_starts_empty():
    assert _current_ctx.get() is None


def test_read_context_uses_contextvar_when_active():
    """read_context prefere o ContextVar sobre o Orchestrator."""
    class FakeOrch:
        workspace_purpose = "from-orch"
        _tool_context = {"workspace_path": "/from-orch"}

    ctx = ExecutionContext(
        request_id="REQ-from-var",
        purpose="from-contextvar",
        tool_context={"workspace_path": "/from-var"},
    )
    token = _current_ctx.set(ctx)
    try:
        result = read_context(FakeOrch(), request_id="IGNORED")
        assert result.request_id == "REQ-from-var"
        assert result.purpose == "from-contextvar"
        assert result.tool_context["workspace_path"] == "/from-var"
    finally:
        _current_ctx.reset(token)


def test_read_context_falls_back_when_contextvar_empty():
    """Sem ContextVar activo, read_context usa from_orchestrator."""
    class FakeOrch:
        workspace_purpose = "analysis"
        _tool_context = {"workspace_path": "/tmp/x"}

    result = read_context(FakeOrch(), request_id="REQ-fallback")
    assert result.request_id == "REQ-fallback"
    assert result.purpose == "analysis"
    assert result.tool_context["workspace_path"] == "/tmp/x"


def test_read_context_legacy_attribute_still_works():
    """Atributo _current_context (legado 5.1) continua funcional."""
    class FakeOrch:
        workspace_purpose = "analysis"
        _tool_context = {}
        _current_context = ExecutionContext(
            request_id="REQ-legacy",
            purpose="legacy",
            tool_context={},
        )

    result = read_context(FakeOrch())
    assert result.request_id == "REQ-legacy"
    assert result.purpose == "legacy"


@pytest.mark.asyncio
async def test_concurrent_tasks_isolated_context():
    """Prova principal: dois tasks em paralelo têm ContextVars independentes."""
    results = {}

    async def worker(name: str, purpose: str, barrier: asyncio.Barrier):
        ctx = ExecutionContext(
            request_id=f"REQ-{name}",
            purpose=purpose,
            tool_context={"workspace_path": f"/tmp/{name}"},
        )
        token = _current_ctx.set(ctx)
        try:
            await barrier.wait()  # ambos os tasks chegam aqui em simultâneo
            # Após a barreira, cada task deve ver o SEU contexto
            class FakeOrch:
                workspace_purpose = "should-be-ignored"
                _tool_context = {}
            seen = read_context(FakeOrch())
            results[name] = seen
        finally:
            _current_ctx.reset(token)

    barrier = asyncio.Barrier(2)
    await asyncio.gather(
        worker("A", "purpose-A", barrier),
        worker("B", "purpose-B", barrier),
    )

    assert results["A"].purpose == "purpose-A"
    assert results["A"].tool_context["workspace_path"] == "/tmp/A"
    assert results["B"].purpose == "purpose-B"
    assert results["B"].tool_context["workspace_path"] == "/tmp/B"
    # Nenhum herdou do outro
    assert results["A"].purpose != results["B"].purpose


@pytest.mark.asyncio
async def test_contextvar_reset_after_task():
    """Após o with terminar, o ContextVar volta a None no escopo pai."""

    async def set_and_clear():
        ctx = ExecutionContext(
            request_id="R1", purpose="tmp", tool_context={}
        )
        token = _current_ctx.set(ctx)
        try:
            assert _current_ctx.get() is not None
        finally:
            _current_ctx.reset(token)

    assert _current_ctx.get() is None
    await set_and_clear()
    assert _current_ctx.get() is None


def test_contextvar_survives_read_context_multiple_calls():
    """Múltiplas chamadas a read_context dentro do mesmo with devolvem o mesmo."""
    class FakeOrch:
        workspace_purpose = "x"
        _tool_context = {}

    ctx = ExecutionContext(
        request_id="R-stable", purpose="stable", tool_context={}
    )
    token = _current_ctx.set(ctx)
    try:
        r1 = read_context(FakeOrch())
        r2 = read_context(FakeOrch())
        r3 = read_context(FakeOrch())
        assert r1.request_id == r2.request_id == r3.request_id == "R-stable"
    finally:
        _current_ctx.reset(token)
