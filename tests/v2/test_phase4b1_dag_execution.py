"""Testes Fase 4B.1 — motor DAG de execução de planos."""
import pytest

from core.v2.agents import AgentResult
from core.v2.orchestrator import OdysseusOrchestrator
from core.v2.planner_agent import Plan, PlanStep


def _make_orchestrator(agent_results_by_step=None, tool_results_by_step=None):
    agent_results_by_step = agent_results_by_step or {}
    tool_results_by_step = tool_results_by_step or {}

    orch = OdysseusOrchestrator.__new__(OdysseusOrchestrator)
    orch.workspace_purpose = "analysis"
    orch.allow_interpreters = False
    orch._tool_context = {
        "workspace_path": "/tmp/fake",
        "workspace_purpose": "analysis",
        "workspace_root": "workspace",
    }
    orch._pending_plans = {}

    exec_order = []

    async def fake_agent_step(step, plan, useragent_knowledge=None):
        exec_order.append(step.step)
        ok = agent_results_by_step.get(step.step, True)
        return AgentResult(
            task_id=f"T-{step.step}",
            success=ok,
            summary=f"{step.step} {'ok' if ok else 'fail'}",
        )

    async def fake_tool_step(step, plan, approval_id=None):
        exec_order.append(step.step)
        outcome = tool_results_by_step.get(
            step.step, {"success": True, "requires_approval": False}
        )
        if outcome.get("requires_approval"):
            return AgentResult(
                task_id=f"T-{step.step}",
                success=False,
                summary=f"{step.step} requires approval",
                details={
                    "requires_approval": True,
                    "approval_id": outcome.get("approval_id", "APPROVAL-0001"),
                    "tool": step.tool,
                },
            )
        return AgentResult(
            task_id=f"T-{step.step}",
            success=outcome.get("success", True),
            summary=f"{step.step} tool",
            details={"tool_result": {"data": "fake-output"}},
        )

    async def fake_llm(*a, **kw):
        return "resposta consolidada"

    orch._execute_agent_step = fake_agent_step
    orch._execute_tool_step = fake_tool_step
    orch._generate_llm_response = fake_llm
    orch._build_agent_response = lambda results: "fallback consolidado"

    return orch, exec_order


@pytest.mark.asyncio
async def test_linear_chain_respects_order():
    plan = Plan(
        id="P1", query="chain",
        steps=[
            PlanStep(step="a", description="a", type="agent", agent="code"),
            PlanStep(step="b", description="b", type="agent", agent="code", dependencies=["a"]),
            PlanStep(step="c", description="c", type="agent", agent="code", dependencies=["b"]),
        ],
    )
    orch, order = _make_orchestrator()
    r = await orch._execute_plan_dag(plan)
    assert order == ["a", "b", "c"], f"ordem: {order}"
    assert plan.status == "completed"
    assert r.metadata["engine"] == "dag"


@pytest.mark.asyncio
async def test_diamond_respects_dependencies():
    plan = Plan(
        id="P2", query="diamond",
        steps=[
            PlanStep(step="a", description="a", type="agent", agent="code"),
            PlanStep(step="b", description="b", type="agent", agent="code", dependencies=["a"]),
            PlanStep(step="c", description="c", type="agent", agent="code", dependencies=["a"]),
            PlanStep(step="d", description="d", type="agent", agent="code", dependencies=["b", "c"]),
        ],
    )
    orch, order = _make_orchestrator()
    await orch._execute_plan_dag(plan)
    assert order[0] == "a"
    assert order[-1] == "d"
    assert set(order[1:3]) == {"b", "c"}
    assert plan.status == "completed"


@pytest.mark.asyncio
async def test_failed_step_blocks_dependent():
    plan = Plan(
        id="P3", query="fail-block",
        steps=[
            PlanStep(step="a", description="a", type="agent", agent="code"),
            PlanStep(step="b", description="b", type="agent", agent="code", dependencies=["a"]),
        ],
    )
    orch, order = _make_orchestrator(agent_results_by_step={"a": False})
    r = await orch._execute_plan_dag(plan)
    assert "a" in order
    assert "b" not in order, "b nao devia executar"
    assert plan.steps[1].status == "blocked"
    assert plan.status == "failed"


@pytest.mark.asyncio
async def test_failed_step_blocks_transitively():
    plan = Plan(
        id="P4", query="transitive",
        steps=[
            PlanStep(step="a", description="a", type="agent", agent="code"),
            PlanStep(step="b", description="b", type="agent", agent="code", dependencies=["a"]),
            PlanStep(step="c", description="c", type="agent", agent="code", dependencies=["b"]),
        ],
    )
    orch, order = _make_orchestrator(agent_results_by_step={"a": False})
    await orch._execute_plan_dag(plan)
    assert order == ["a"]
    assert plan.steps[1].status == "blocked"
    assert plan.steps[2].status == "blocked"


@pytest.mark.asyncio
async def test_all_success_completes_plan():
    plan = Plan(
        id="P5", query="success",
        steps=[
            PlanStep(step="a", description="a", type="agent", agent="code"),
            PlanStep(step="b", description="b", type="agent", agent="code", dependencies=["a"]),
        ],
    )
    orch, _ = _make_orchestrator()
    r = await orch._execute_plan_dag(plan)
    assert plan.status == "completed"
    assert r.metadata["steps_status"] == {"a": "completed", "b": "completed"}


@pytest.mark.asyncio
async def test_empty_plan_returns_clean():
    plan = Plan(id="P6", query="empty", steps=[])
    orch, _ = _make_orchestrator()
    r = await orch._execute_plan_dag(plan)
    assert plan.status == "completed"
    assert r.metadata["engine"] == "dag"


@pytest.mark.asyncio
async def test_cycle_detected_no_execution():
    plan = Plan(
        id="P7", query="cycle",
        steps=[
            PlanStep(step="a", description="a", type="agent", agent="code", dependencies=["b"]),
            PlanStep(step="b", description="b", type="agent", agent="code", dependencies=["a"]),
        ],
    )
    orch, order = _make_orchestrator()
    await orch._execute_plan_dag(plan)
    assert order == [], "nenhum agente devia correr"
    assert plan.status == "failed"
    assert all(s.status == "failed" for s in plan.steps)


@pytest.mark.asyncio
async def test_tool_step_requires_approval():
    plan = Plan(
        id="P8", query="tool-approval",
        steps=[
            PlanStep(step="t", description="tool", type="tool", tool="shell",
                     tool_params={"command": "git status"}),
        ],
    )
    orch, _ = _make_orchestrator(
        tool_results_by_step={
            "t": {"success": False, "requires_approval": True, "approval_id": "APPROVAL-0007"},
        }
    )
    r = await orch._execute_plan_dag(plan)
    assert plan.status == "awaiting_approval"
    assert r.requires_approval
    assert r.requires_approval[0]["approval_id"] == "APPROVAL-0007"
    assert "APPROVAL-0007" in orch._pending_plans
