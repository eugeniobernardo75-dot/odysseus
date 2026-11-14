from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_workers_are_registered_and_selected_by_useragent_factory():
    from core.v2.agents import AgentFactory

    factory = AgentFactory()
    assert {
        "secretaria", "tecnico", "director-geral"
    }.issubset({agent["name"] for agent in factory.list_agents()})
    assert factory.select_worker("Cria um ficheiro de teste").name == "tecnico"
    assert factory.select_worker(
        "Como Director-Geral, organiza as prioridades"
    ).name == "director-geral"
    assert factory.select_worker("Prepara uma carta formal para a reunião").name == "secretaria"
    assert factory.select_worker("Pesquisa fontes sobre a história de Roma") is None
    assert factory.select_agent("Pesquisa fontes sobre a história de Roma").name == "research"
    assert factory.select_worker("Executa git status").name == "tecnico"


def test_research_plan_does_not_add_director_without_a_worker_match():
    from core.v2.agents import AgentFactory
    from core.v2.orchestrator import OdysseusOrchestrator
    from core.v2.planner_agent import Plan, PlanStep

    orchestrator = object.__new__(OdysseusOrchestrator)
    orchestrator.agents = AgentFactory()
    plan = Plan(
        id="PLAN-RESEARCH-ONLY",
        query="Pesquisa fontes sobre a história de Roma",
        steps=[
            PlanStep(
                step="research",
                description="Pesquisar fontes relevantes",
                type="agent",
                agent="research",
            )
        ],
    )

    orchestrator._bind_delegated_worker(plan, None)

    assert plan.delegated_worker is None
    assert [step.agent for step in plan.steps] == ["research"]


def test_explicit_technical_change_selects_development_purpose():
    from core.v2.user_agent_config import UserAgent

    assert UserAgent._task_purpose(
        "Cria um ficheiro de teste", "tecnico"
    ) == "development"
    assert UserAgent._task_purpose(
        "Analisa este log", "tecnico"
    ) is None


def test_operational_worker_requests_enter_task_pipeline():
    from core.v2.agents import AgentFactory
    from core.v2.memory import DictMemoryBackend, MemoryManager
    from core.v2.orchestrator import OdysseusOrchestrator

    orchestrator = OdysseusOrchestrator(
        memory=MemoryManager(backend=DictMemoryBackend()),
        llm_client=None,
        agents=AgentFactory(),
    )

    assert orchestrator._analyse_intent(
        "Como Secretária, prepara uma carta formal"
    )["type"] == "task"
    assert orchestrator._analyse_intent(
        "Como Director-Geral, organiza as prioridades"
    )["type"] == "task"


@pytest.mark.asyncio
async def test_director_cannot_execute_shell():
    from core.v2.tool_manager import ToolManager
    from core.v2.tool_registry import ToolRegistry, ToolResult
    from core.v2.tools import ShellTool
    from core.v2.workers import DirectorWorker

    registry = ToolRegistry()
    shell = ShellTool()
    executed = False

    async def execute_shell(params, context=None):
        nonlocal executed
        executed = True
        return ToolResult(success=True)

    shell.execute = execute_shell
    registry.register(shell)
    manager = ToolManager(registry=registry)

    result = await manager.execute(
        "shell",
        {"command": "echo must-not-run"},
        allowed_tools=DirectorWorker.allowed_tools,
    )

    assert result.success is False
    assert result.metadata["reason"] == "worker_tool_not_allowed"
    assert executed is False


@pytest.mark.asyncio
async def test_orchestrator_applies_director_allowlist_to_tool_step():
    from core.v2.memory import DictMemoryBackend, MemoryManager
    from core.v2.orchestrator import OdysseusOrchestrator
    from core.v2.planner_agent import Plan, PlanStep

    orchestrator = OdysseusOrchestrator(
        memory=MemoryManager(backend=DictMemoryBackend()),
        llm_client=None,
    )
    plan = Plan(
        id="PLAN-DIRECTOR-DENY",
        query="Tenta executar shell",
        delegated_worker="director-geral",
        steps=[
            PlanStep(
                step="forbidden_shell",
                description="Deve ser bloqueado pela allowlist",
                type="tool",
                tool="shell",
                tool_params={"command": "echo must-not-run"},
            )
        ],
    )

    result = await orchestrator._execute_tool_step(plan.steps[0], plan)

    assert result.success is False
    assert "não está autorizado" in result.details["tool_result"]["error"]


@pytest.mark.asyncio
async def test_streaming_worker_task_uses_coordinated_process():
    from unittest.mock import AsyncMock

    from core.v2.agents import AgentFactory
    from core.v2.memory import DictMemoryBackend, MemoryManager
    from core.v2.orchestrator import OdysseusOrchestrator, OrchestratorResponse

    orchestrator = OdysseusOrchestrator(
        memory=MemoryManager(backend=DictMemoryBackend()),
        llm_client=None,
        agents=AgentFactory(),
    )
    orchestrator.process = AsyncMock(
        return_value=OrchestratorResponse(message="Aguardar aprovação.")
    )

    tokens = [
        token async for token in orchestrator.process_stream(
            "Como Técnico, cria um ficheiro de teste",
            delegated_worker="tecnico",
            purpose="development",
        )
    ]

    assert tokens == ["Aguardar aprovação.", "__DONE__"]
    orchestrator.process.assert_awaited_once()
    assert orchestrator.process.await_args.kwargs["delegated_worker"] == "tecnico"
    assert orchestrator.process.await_args.kwargs["purpose"] == "development"


@pytest.mark.asyncio
async def test_file_write_requires_human_approval_then_executes(tmp_path):
    from core.v2.safety import SafetyManager
    from core.v2.tool_manager import ToolManager
    from core.v2.tool_registry import ToolRegistry
    from core.v2.tools import FileTool
    from core.v2.workers import TecnicoWorker

    registry = ToolRegistry()
    registry.register(FileTool())
    safety = SafetyManager()
    manager = ToolManager(registry=registry, safety_manager=safety)
    params = {
        "action": "write",
        "path": "teste_seguro.txt",
        "content": "OK",
    }
    context = {"workspace_path": str(tmp_path)}

    pending = await manager.execute(
        "file",
        params,
        context=context,
        allowed_tools=TecnicoWorker.allowed_tools,
    )
    approval_id = pending.metadata["approval_id"]
    assert pending.success is False
    assert not (tmp_path / "teste_seguro.txt").exists()

    approved, _ = safety.approve(approval_id)
    assert approved is True
    completed = await manager.execute(
        "file",
        params,
        context=context,
        approval_id=approval_id,
        allowed_tools=TecnicoWorker.allowed_tools,
    )
    assert completed.success is True
    assert (tmp_path / "teste_seguro.txt").read_text(encoding="utf-8") == "OK"


@pytest.mark.asyncio
async def test_technician_cannot_delete_files():
    from core.v2.tool_manager import ToolManager
    from core.v2.tool_registry import ToolRegistry
    from core.v2.tools import FileTool
    from core.v2.workers import TecnicoWorker

    registry = ToolRegistry()
    registry.register(FileTool())
    manager = ToolManager(registry=registry)

    result = await manager.execute(
        "file",
        {"action": "delete", "path": "not-to-be-deleted.txt"},
        allowed_tools=TecnicoWorker.allowed_tools,
    )

    assert result.success is False
    assert result.metadata["reason"] == "worker_tool_not_allowed"


@pytest.mark.asyncio
async def test_technician_cannot_read_outside_workspace(tmp_path):
    from core.v2.tool_manager import ToolManager
    from core.v2.tool_registry import ToolRegistry
    from core.v2.tools import FileTool
    from core.v2.workers import TecnicoWorker

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("private", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(FileTool())
    manager = ToolManager(registry=registry)

    result = await manager.execute(
        "file",
        {"action": "read", "path": str(outside_file)},
        context={"workspace_path": str(workspace)},
        allowed_tools=TecnicoWorker.allowed_tools,
    )

    assert result.success is False
    assert "fora do workspace" in result.error.lower()
