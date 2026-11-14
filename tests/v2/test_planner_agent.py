"""Regressões do contrato PlannerAgent -> Plan -> ToolManager."""

from __future__ import annotations

import pytest

from core.v2.agents import AgentTask
from core.v2.llm_client import LLMResponse
from core.v2.planner_agent import PlannerAgent


class FakeLLM:
    def __init__(self, content: str, success: bool = True):
        self.content = content
        self.success = success
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return LLMResponse(
            content=self.content,
            model="fake",
            success=self.success,
            error=None if self.success else "fake failure",
        )


TOOL_PLAN = """{
  "objective": "Obter o estado do repositório",
  "complexity": "low",
  "steps": [{
    "step": "git_status",
    "description": "Executar git status no workspace autorizado",
    "type": "tool",
    "tool": "shell",
    "tool_params": {"command": "git status"},
    "dependencies": [],
    "validation_criteria": "O comando devolveu um código de saída"
  }],
  "estimated_tools": ["shell"],
  "risk_assessment": "Execução shell requer aprovação"
}"""

INVALID_GIT_PLAN = """{
  "objective": "Executar git status",
  "steps": [{"step": "git_status", "description": "Executar", "type": "git", "tool_params": {"action": "status"}}]
}"""

NO_TOOL_PLAN = """{
  "objective": "Responder à saudação",
  "complexity": "low",
  "steps": [{
    "step": "respond",
    "description": "Responder sem executar ferramentas",
    "type": "llm",
    "tool": null,
    "dependencies": [],
    "validation_criteria": "Resposta textual"
  }]
}"""


@pytest.mark.asyncio
async def test_planner_uses_injected_orchestrator_client(monkeypatch):
    injected = FakeLLM(TOOL_PLAN)
    planner = PlannerAgent(llm_client=injected)

    # Se o planeador cair no singleton global, este teste falha explicitamente.
    import core.v2.llm_client as llm_module
    monkeypatch.setattr(
        llm_module, "get_llm_client",
        lambda: (_ for _ in ()).throw(AssertionError("singleton global usado")),
    )

    result = await planner.execute(AgentTask(
        id="PLN-TEST-1",
        agent_type="planner",
        query="Executa git status",
        context={"tool_capabilities": {"capabilities": ["shell"]}},
    ))

    assert result.success is True
    assert injected.calls
    plan = result.details["plan_object"]
    assert len(plan.steps) == 1
    assert plan.steps[0].type == "tool"
    assert plan.steps[0].tool == "shell"
    assert plan.steps[0].tool_params == {"command": "git status"}


@pytest.mark.asyncio
async def test_planner_preserves_non_tool_request_without_inventing_tool():
    planner = PlannerAgent(llm_client=FakeLLM(NO_TOOL_PLAN))
    result = await planner.execute(AgentTask(
        id="PLN-TEST-2",
        agent_type="planner",
        query="Diz olá",
        context={"tool_capabilities": {"capabilities": ["shell"]}},
    ))

    assert result.success is True
    plan = result.details["plan_object"]
    assert all(step.type != "tool" for step in plan.steps)
    assert all(step.tool is None for step in plan.steps)


@pytest.mark.asyncio
async def test_invalid_tool_contract_falls_back_without_inventing_execution():
    planner = PlannerAgent(llm_client=FakeLLM(INVALID_GIT_PLAN))
    result = await planner.execute(AgentTask(
        id="PLN-TEST-INVALID-GIT",
        agent_type="planner",
        query="Executa git status",
        context={"tool_capabilities": {"capabilities": ["shell"]}},
    ))
    plan = result.details["plan_object"]
    assert all(step.type != "tool" for step in plan.steps)
    assert all(step.tool is None for step in plan.steps)


@pytest.mark.asyncio
async def test_invalid_llm_plan_falls_back_without_inventing_execution():
    planner = PlannerAgent(llm_client=FakeLLM("não é JSON"))
    result = await planner.execute(AgentTask(
        id="PLN-TEST-3",
        agent_type="planner",
        query="Executa git status",
        context={"tool_capabilities": {"capabilities": ["shell"]}},
    ))

    assert result.success is True
    plan = result.details["plan_object"]
    assert all(step.type != "tool" for step in plan.steps)
    assert all(step.tool is None for step in plan.steps)


def test_planner_tool_catalog_includes_parameter_schemas(monkeypatch):
    from core.v2 import tool_registry

    class Registry:
        def list_tools(self):
            return [{
                "name": "shell",
                "description": "Safe command execution",
                "category": "system",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            }]

    monkeypatch.setattr(tool_registry, "get_tool_registry", lambda: Registry())
    planner = PlannerAgent(llm_client=FakeLLM(NO_TOOL_PLAN))

    available_tools = planner._get_available_tools()

    assert '"command":{"type":"string"}' in available_tools
    assert '"required":["command"]' in available_tools
    assert '"action"' not in available_tools
