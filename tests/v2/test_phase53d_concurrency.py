"""Testes Fase 5.3d — concorrencia end-to-end via process real.

Prova que o ContextVar activado em `process` (5.3b-1) sobrevive
todo o pipeline — nao apenas a entrada. Dois pedidos em paralelo
chegam ao `_generate_llm_response` cada um com o SEU contexto.
"""
import asyncio

import pytest

from core.v2.execution_context import read_context
from core.v2.orchestrator import OdysseusOrchestrator


class _FakeSafety:
    def sanitize_input(self, s):
        return s

    def check_action(self, **kw):
        return {
            "allowed": True,
            "requires_approval": False,
            "risk": "safe",
            "message": "ok",
            "approval_id": None,
        }


def _make_orchestrator(purpose, workspace, captured_key, captured):
    """Orchestrator minimo que captura o contexto no ponto tardio
    (_generate_llm_response) do pipeline."""
    orch = OdysseusOrchestrator.__new__(OdysseusOrchestrator)
    orch.workspace_purpose = purpose
    orch.allow_interpreters = False
    orch._tool_context = {"workspace_path": workspace, "workspace_purpose": purpose}
    orch._pending_plans = {}
    orch.safety = _FakeSafety()
    orch._cache = None

    def fake_handle_special(msg):
        return None

    def fake_analyse(msg):
        return {"type": "conversation", "complexity": "low"}

    async def fake_llm(message, model=None, useragent_knowledge=None,
                       useragent_memory=None, conversation_history=None):
        # PONTO TARDIO — depois de todo o pipeline excepto a propria chamada LLM
        ctx = read_context(orch)
        captured[captured_key] = {
            "purpose": ctx.purpose,
            "workspace": ctx.tool_context.get("workspace_path"),
            "thread_task": asyncio.current_task().get_name() if asyncio.current_task() else None,
        }
        return f"resp-{captured_key}"

    orch._handle_special_commands = fake_handle_special
    orch._analyse_intent = fake_analyse
    orch._generate_llm_response = fake_llm
    orch._check_memory_suggestions = lambda *a, **kw: []

    return orch


@pytest.mark.asyncio
async def test_two_parallel_processes_isolated_at_llm_call():
    """Dois process em paralelo: cada um ve o SEU contexto no momento
    da chamada ao LLM (ponto tardio do pipeline)."""
    captured = {}

    orch_a = _make_orchestrator("analysis", "/tmp/A", "A", captured)
    orch_b = _make_orchestrator("development", "/tmp/B", "B", captured)

    async def run_a():
        return await orch_a.process("msg A")

    async def run_b():
        return await orch_b.process("msg B")

    result_a, result_b = await asyncio.gather(run_a(), run_b())

    assert result_a is not None
    assert result_b is not None

    # Capturas no ponto tardio
    assert "A" in captured, "orch A nao chegou ao LLM"
    assert "B" in captured, "orch B nao chegou ao LLM"

    # Cada um viu o SEU contexto
    assert captured["A"]["purpose"] == "analysis"
    assert captured["A"]["workspace"] == "/tmp/A"
    assert captured["B"]["purpose"] == "development"
    assert captured["B"]["workspace"] == "/tmp/B"

    # E nao herdou o do outro
    assert captured["A"]["purpose"] != captured["B"]["purpose"]
    assert captured["A"]["workspace"] != captured["B"]["workspace"]


@pytest.mark.asyncio
async def test_sequential_processes_do_not_leak_context():
    """Dois process sequenciais: o segundo nao herda o primeiro."""
    captured = {}

    orch_a = _make_orchestrator("analysis", "/tmp/A", "A", captured)
    orch_b = _make_orchestrator("development", "/tmp/B", "B", captured)

    await orch_a.process("msg A")
    await orch_b.process("msg B")

    assert captured["A"]["purpose"] == "analysis"
    assert captured["B"]["purpose"] == "development"
    assert captured["B"]["workspace"] == "/tmp/B"


@pytest.mark.asyncio
async def test_three_parallel_processes_isolated():
    """Stress: tres em paralelo, cada um ve o seu."""
    captured = {}

    orches = [
        _make_orchestrator(f"purpose-{i}", f"/tmp/p{i}", f"p{i}", captured)
        for i in range(3)
    ]

    async def run(orch):
        return await orch.process("msg")

    await asyncio.gather(*[run(o) for o in orches])

    for i in range(3):
        key = f"p{i}"
        assert key in captured, f"orch {key} nao chegou ao LLM"
        assert captured[key]["purpose"] == f"purpose-{i}"
        assert captured[key]["workspace"] == f"/tmp/p{i}"


@pytest.mark.asyncio
async def test_contextvar_isolated_between_interleaved_calls():
    """Interleaving real: um task cede controlo ao outro (await),
    e cada um continua a ver o seu contexto depois do await."""
    captured = {}
    barrier = asyncio.Barrier(2)

    async def run_a():
        orch = _make_orchestrator("analysis", "/tmp/A", "A", captured)

        async def fake_llm_slow(message, model=None, **kw):
            # Cede controlo ao outro task
            await barrier.wait()
            # Depois da barreira, ainda deve ver o SEU contexto
            ctx = read_context(orch)
            captured["A"] = {
                "purpose": ctx.purpose,
                "workspace": ctx.tool_context.get("workspace_path"),
            }
            return "resp-A"

        orch._generate_llm_response = fake_llm_slow
        return await orch.process("msg A")

    async def run_b():
        orch = _make_orchestrator("development", "/tmp/B", "B", captured)

        async def fake_llm_slow(message, model=None, **kw):
            await barrier.wait()
            ctx = read_context(orch)
            captured["B"] = {
                "purpose": ctx.purpose,
                "workspace": ctx.tool_context.get("workspace_path"),
            }
            return "resp-B"

        orch._generate_llm_response = fake_llm_slow
        return await orch.process("msg B")

    await asyncio.gather(run_a(), run_b())

    assert captured["A"]["purpose"] == "analysis"
    assert captured["B"]["purpose"] == "development"


@pytest.mark.asyncio
async def test_contextvar_cleared_after_parallel_batch():
    """Depois de gather completo, ninguem deixou contexto pendente."""
    from core.v2.execution_context import _current_ctx

    captured = {}

    orch_a = _make_orchestrator("analysis", "/tmp/A", "A", captured)
    orch_b = _make_orchestrator("development", "/tmp/B", "B", captured)

    await asyncio.gather(
        orch_a.process("msg A"),
        orch_b.process("msg B"),
    )

    # ContextVar vazio no scope pai (fora dos with terminados)
    assert _current_ctx.get() is None
