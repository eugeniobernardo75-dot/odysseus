from __future__ import annotations

import re
from types import SimpleNamespace

from fastapi.testclient import TestClient


def test_bridge_approval_authenticates_and_resumes_original_plan(monkeypatch):
    from core.v2.memory import DictMemoryBackend, MemoryManager
    from core.v2 import llm_client, orchestrator as orchestrator_module, user_agent_config
    from core.v2.orchestrator import OdysseusOrchestrator
    from core.v2.orchestrator_adapter import OrchestratorAdapter
    from core.v2.planner_agent import Plan, PlanStep
    from core.v2.safety import SafetyManager
    from core.v2.tool_manager import ToolManager
    from core.v2.tool_registry import (
        BaseTool,
        ToolDefinition,
        ToolRegistry,
        ToolResult,
    )
    from core.v2.workers import TecnicoWorker
    import v2_server

    approval_token = "test-approval-token"
    monkeypatch.setenv("ODYSSEUS_V2_APPROVAL_TOKEN", approval_token)
    monkeypatch.setattr(
        TecnicoWorker,
        "allowed_tools",
        TecnicoWorker.allowed_tools | {"approval_probe"},
    )
    calls = []
    plans = []

    class HarmlessWorkerLLM:
        async def chat(self, messages, max_tokens=700):
            return SimpleNamespace(
                success=True,
                content="Análise técnica concluída.",
                error=None,
            )

    class HarmlessApprovalProbe(BaseTool):
        def _build_definition(self):
            return ToolDefinition(
                name="approval_probe",
                description="Regista a execução de uma operação de teste inofensiva.",
                parameters={
                    "type": "object",
                    "properties": {"marker": {"type": "string"}},
                    "required": ["marker"],
                },
                permissions=["test.probe"],
                requires_approval=True,
            )

        async def execute(self, params, context=None):
            calls.append(params["marker"])
            return ToolResult(
                success=True,
                data={"executed": params["marker"]},
                tool_name=self.name,
            )

    safety = SafetyManager()
    registry = ToolRegistry()
    registry.register(HarmlessApprovalProbe())
    tool_manager = ToolManager(registry=registry, safety_manager=safety)
    orchestrator = OdysseusOrchestrator(
        memory=MemoryManager(backend=DictMemoryBackend()),
        safety=safety,
        tool_manager=tool_manager,
    )

    def make_plan(message, analysis):
        plan = Plan(
            id="PLAN-APPROVAL-PROBE",
            query=message,
            steps=[
                PlanStep(
                    step="run_approval_probe",
                    description="Executa a operação inofensiva de teste.",
                    type="tool",
                    tool="approval_probe",
                    tool_params={"marker": "approved-once"},
                )
            ],
        )
        plans.append(plan)
        return plan

    async def create_plan(message, analysis, **kwargs):
        return make_plan(message, analysis)

    orchestrator._analyse_intent = lambda message: {
        "type": "task",
        "complexity": "low",
        "requires_tools": True,
    }
    orchestrator._create_plan = create_plan
    user_agent = user_agent_config.UserAgent()
    user_agent._retrieve_relevant_knowledge = lambda _query: []
    user_agent._retrieve_relevant_memory = lambda _query: ""
    monkeypatch.setattr(
        orchestrator_module, "get_orchestrator", lambda **_kwargs: orchestrator
    )
    monkeypatch.setattr(llm_client, "get_llm_client", HarmlessWorkerLLM)
    monkeypatch.setattr(
        user_agent_config, "_get_memory_manager", lambda: orchestrator.memory
    )
    adapter = OrchestratorAdapter(orchestrator, user_agent=user_agent)
    monkeypatch.setattr(
        "core.v2.orchestrator_adapter.create_adapter_from_config",
        lambda config: adapter,
    )
    monkeypatch.setattr(v2_server, "V2LLMWrapper", lambda config: object())
    monkeypatch.setattr(
        v2_server,
        "V2OrchestratorWrapper",
        lambda config: type("UnusedOrchestratorWrapper", (), {})(),
    )

    app = v2_server.create_app(v2_server.BridgeConfig())
    client = TestClient(app)
    chat = client.post(
        "/v1/chat/completions",
        json={
            "model": "odysseus-v2",
            "messages": [{"role": "user", "content": "Executa o teste inofensivo."}],
        },
    )

    assert chat.status_code == 200
    response_text = chat.json()["choices"][0]["message"]["content"]
    approval_id = re.search(r"APPROVAL-\d+", response_text).group(0)
    assert calls == []
    assert len(plans) == 1
    assert orchestrator.has_pending_plan(approval_id)
    assert orchestrator._pending_plans[approval_id] is plans[0]
    assert safety.get_approval(approval_id)["status"] == "pending"

    pending = client.get("/v2/approvals/pending")
    assert pending.status_code == 401
    pending = client.get(
        "/v2/approvals/pending",
        headers={"Authorization": f"Bearer {approval_token}"},
    )
    assert pending.status_code == 200
    assert pending.json()["pending"][0]["approval_id"] == approval_id
    assert pending.json()["pending"][0]["plan_id"] == plans[0].id

    unauthorized = client.post(f"/v2/approve/{approval_id}")
    assert unauthorized.status_code == 401
    assert calls == []
    assert safety.get_approval(approval_id)["status"] == "pending"

    approved = client.post(
        f"/v2/approve/{approval_id}",
        headers={"Authorization": f"Bearer {approval_token}"},
    )
    assert approved.status_code == 200
    assert approved.json()["success"] is True
    assert approved.json()["resumed"] is True
    assert approved.json()["plan_id"] == plans[0].id
    assert approved.json()["plan_status"] == "completed"
    assert approved.json()["executions"] == [
        {
            "tool": "approval_probe",
            "success": True,
            "return_code": None,
            "isolated": None,
            "isolation_backend": None,
        }
    ]
    assert calls == ["approved-once"]
    assert len(plans) == 1
    assert plans[0].status == "completed"
    assert not orchestrator.has_pending_plan(approval_id)

    repeated = client.post(
        f"/v2/approve/{approval_id}",
        headers={"Authorization": f"Bearer {approval_token}"},
    )
    assert repeated.status_code == 409
    assert calls == ["approved-once"]

    denied_chat = client.post(
        "/v1/chat/completions",
        json={
            "model": "odysseus-v2",
            "messages": [{"role": "user", "content": "Nega o teste inofensivo."}],
        },
    )
    assert denied_chat.status_code == 200
    denied_text = denied_chat.json()["choices"][0]["message"]["content"]
    denied_id = re.search(r"APPROVAL-\d+", denied_text).group(0)
    assert len(plans) == 2
    assert calls == ["approved-once"]

    denied = client.post(
        f"/v2/deny/{denied_id}",
        headers={"Authorization": f"Bearer {approval_token}"},
    )
    assert denied.status_code == 200
    assert denied.json()["status"] == "denied"
    assert denied.json()["resumed"] is False
    assert denied.json()["plan_id"] == plans[1].id
    assert denied.json()["executions"] == []
    assert not orchestrator.has_pending_plan(denied_id)
    assert calls == ["approved-once"]

    denied_approval = client.post(
        f"/v2/approve/{denied_id}",
        headers={"Authorization": f"Bearer {approval_token}"},
    )
    assert denied_approval.status_code == 409
    assert calls == ["approved-once"]
