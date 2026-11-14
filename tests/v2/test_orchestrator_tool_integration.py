"""
Testes de integração — Orchestrator → ToolManager → ferramenta real (Etapa 4)

Percorrem o fluxo real:

    plano → passo de ferramenta → ToolManager (aprovação) → ShellTool
          → subprocesso real (git) → resultado de volta ao plano

Objectivos:
    - o workspace é resolvido pelo *servidor* (Workspace autorizado), nunca pelo
      chamador nem pelos parâmetros da tarefa;
    - sem aprovação nada executa e o pedido de aprovação regressa ao fluxo;
    - com aprovação, a ferramenta executa de verdade e o resultado volta ao plano;
    - a aprovação não é reutilizável para outra operação;
    - `workdir` para área protegida é negado.

Nota: os comandos usados são de inspeção (`git status`) e correm dentro de
`Workspace`. Nada toca Brain Vault, Mind OS, data/, logs/ ou .env.

Uso:
    pytest tests/v2/test_orchestrator_tool_integration.py -v
"""

from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _make_orchestrator():
    """Orchestrator real, sem LLM (o planeamento é fornecido pelo teste)."""
    from core.v2.memory import DictMemoryBackend, MemoryManager
    from core.v2.orchestrator import OdysseusOrchestrator

    memory = MemoryManager(backend=DictMemoryBackend())
    return OdysseusOrchestrator(memory=memory, llm_client=None)


def _tool_plan(*, tool_params: dict, step_name: str = "run_probe", plan_id: str = "PLAN-T-1"):
    from core.v2.planner_agent import Plan, PlanStep

    return Plan(
        id=plan_id,
        query="verificar o estado do repositório no workspace autorizado",
        delegated_worker="tecnico",
        steps=[
            PlanStep(
                step=step_name,
                description="executar comando de inspeção",
                type="tool",
                tool="shell",
                tool_params=tool_params,
            )
        ],
    )


# ======================================================================
#  Workspace resolvido pelo servidor
# ======================================================================

class TestServerResolvedWorkspace:
    def test_context_uses_approved_workspace(self):
        from core.v2.execution_policy import approved_root

        from core.v2.execution_policy import DEFAULT_PURPOSE

        orch = _make_orchestrator()
        assert orch._tool_context["workspace_path"] == str(approved_root("workspace"))
        # Propósito padrão restritivo (analysis), não "workspace".
        assert orch.get_purpose() == DEFAULT_PURPOSE
        assert orch._tool_context["workspace_purpose"] == DEFAULT_PURPOSE
        # Interpretadores desligados por omissão.
        assert "allow_interpreters" not in orch._tool_context
        assert orch.allow_interpreters is False

    @pytest.mark.asyncio
    async def test_caller_cannot_choose_workspace(self):
        """Um workspace_path do chamador é ignorado e substituído pelo do servidor."""
        from core.v2.execution_policy import approved_root

        orch = _make_orchestrator()
        await orch.process(
            "olá, tudo bem?",
            context={
                "workspace_path": "C:\\",
                "cwd": "C:\\Windows",
                "allow_interpreters": True,
                "session_id": "sess-1",
            },
        )
        assert orch._tool_context["workspace_path"] == str(approved_root("workspace"))
        assert "allow_interpreters" not in orch._tool_context

    def test_security_validation_uses_quarantine_root(self):
        from core.v2.execution_policy import approved_root, build_tool_context

        ctx = build_tool_context("security_validation")
        assert ctx["workspace_path"] == str(approved_root("quarantine"))
        assert ctx["workspace_path"] != str(approved_root("workspace"))

    def test_server_can_switch_purpose(self):
        orch = _make_orchestrator()
        from core.v2.execution_policy import approved_root

        assert orch.set_purpose("development") is True
        assert orch.get_purpose() == "development"
        assert orch.allow_interpreters is True
        assert orch._tool_context["workspace_path"] == str(approved_root("workspace"))

        assert orch.set_purpose("security_validation") is True
        assert orch._tool_context["workspace_path"] == str(approved_root("quarantine"))

    def test_invalid_purpose_does_not_change_state(self):
        orch = _make_orchestrator()
        before = orch.get_purpose()
        assert orch.set_purpose("root") is False
        assert orch.set_purpose("Development") is False
        assert orch.get_purpose() == before

    @pytest.mark.asyncio
    async def test_invalid_purpose_denies_request(self):
        """Propósito inválido pedido pelo servidor: nada executa."""
        orch = _make_orchestrator()
        response = await orch.process("olá", purpose="root")
        assert response.metadata.get("invalid_purpose") is True
        assert "inválido" in response.message.lower()


# ======================================================================
#  Integração real: aprovação → execução → resultado
# ======================================================================

class TestOrchestratorToolExecution:
    @pytest.fixture
    def safety(self):
        from core.v2.safety import get_safety_manager
        manager = get_safety_manager()
        manager.clear_pending()
        return manager

    @pytest.mark.asyncio
    async def test_without_approval_nothing_executes(self, safety):
        orch = _make_orchestrator()
        plan = _tool_plan(tool_params={"command": "git status"})

        response = await orch._execute_plan(plan)

        assert response.metadata.get("awaiting_approval") is True
        assert response.requires_approval
        approval = response.requires_approval[0]
        assert approval["tool"] == "shell"
        assert approval["approval_id"]
        assert plan.status == "awaiting_approval"
        assert plan.steps[0].status == "pending"
        # Nada executou: nenhum resultado de ferramenta no plano.
        assert not any(r.details and r.details.get("tool_result") for r in response.agent_results)

    @pytest.mark.asyncio
    async def test_approved_tool_step_executes_really(self, safety):
        """Execução real através do Orchestrator, com aprovação humana."""
        from core.v2.execution_policy import approved_root

        orch = _make_orchestrator()
        plan = _tool_plan(tool_params={"command": "git status"})

        first = await orch._execute_plan(plan)
        approval_id = first.requires_approval[0]["approval_id"]
        approved, _ = safety.approve(approval_id)
        assert approved is True

        second = await orch._execute_plan(plan, approval_id=approval_id)

        assert plan.status == "completed"
        assert plan.steps[0].status == "completed"

        results = [r for r in second.agent_results if r.details and r.details.get("tool_result")]
        assert len(results) == 1
        result = results[0]
        assert result.success is True

        tool_result = result.details["tool_result"]
        # O workspace usado é o do servidor, não o do chamador.
        assert result.details["workspace"] == str(approved_root("workspace"))
        # O comando correu de verdade: houve processo e código de saída real.
        assert tool_result["data"]["return_code"] == 0
        assert tool_result["metadata"]["return_code"] == 0

    @pytest.mark.asyncio
    async def test_approval_not_reusable_for_other_operation(self, safety):
        orch = _make_orchestrator()
        plan_a = _tool_plan(tool_params={"command": "git status"}, plan_id="PLAN-A")
        first = await orch._execute_plan(plan_a)
        approval_id = first.requires_approval[0]["approval_id"]
        safety.approve(approval_id)

        # Mesma aprovação, operação diferente -> negado, nada executa.
        plan_b = _tool_plan(tool_params={"command": "git log"}, plan_id="PLAN-B")
        response = await orch._execute_plan(plan_b, approval_id=approval_id)

        results = [r for r in response.agent_results if r.details and r.details.get("tool_result")]
        assert results and results[0].success is False
        assert "não corresponde" in (results[0].error or "")
        assert plan_b.steps[0].status == "failed"

        # E o comando original continua a poder executar com a sua aprovação.
        ok = await orch._execute_plan(plan_a, approval_id=approval_id)
        good = [r for r in ok.agent_results if r.details and r.details.get("tool_result")]
        assert good and good[0].success is True

    @pytest.mark.asyncio
    async def test_protected_workdir_denied_through_orchestrator(self, safety):
        orch = _make_orchestrator()
        plan = _tool_plan(
            tool_params={"command": "git status", "workdir": "Odysseus_Mind_OS"},
            plan_id="PLAN-PROT",
        )
        first = await orch._execute_plan(plan)
        approval_id = first.requires_approval[0]["approval_id"]
        safety.approve(approval_id)

        response = await orch._execute_plan(plan, approval_id=approval_id)

        results = [r for r in response.agent_results if r.details and r.details.get("tool_result")]
        assert results and results[0].success is False
        assert "workdir" in (results[0].error or "")
        assert plan.steps[0].status == "failed"

    @pytest.mark.asyncio
    async def test_disallowed_command_denied_through_orchestrator(self, safety):
        """Aprovação não transforma um comando proibido em permitido."""
        orch = _make_orchestrator()
        plan = _tool_plan(tool_params={"command": "git clone http://exemplo"}, plan_id="PLAN-DENY")
        first = await orch._execute_plan(plan)
        approval_id = first.requires_approval[0]["approval_id"]
        safety.approve(approval_id)

        response = await orch._execute_plan(plan, approval_id=approval_id)

        results = [r for r in response.agent_results if r.details and r.details.get("tool_result")]
        assert results and results[0].success is False
        assert plan.steps[0].status == "failed"

    @pytest.mark.asyncio
    async def test_interpreters_denied_by_default_through_orchestrator(self, safety):
        """`python` não está na allow-list por omissão: o passo falha na política."""
        orch = _make_orchestrator()
        plan = _tool_plan(tool_params={"command": "python --version"}, plan_id="PLAN-PY")
        first = await orch._execute_plan(plan)
        approval_id = first.requires_approval[0]["approval_id"]
        safety.approve(approval_id)

        response = await orch._execute_plan(plan, approval_id=approval_id)

        results = [r for r in response.agent_results if r.details and r.details.get("tool_result")]
        assert results and results[0].success is False
        assert "permitidos" in (results[0].error or "")
