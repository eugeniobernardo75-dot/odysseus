"""
Testes — execution_policy (Etapa 4)

Cobre:
    - raízes autorizadas (Workspace / Quarantine) e recusas;
    - contexto de ferramentas construído pelo servidor;
    - saneamento do contexto do chamador;
    - política de comandos com e sem autorização de interpretadores;
    - ligação aprovação↔operação (fingerprint);
    - redacção de segredos;
    - sonda de isolamento (estado real, sem optimismo).

Uso:
    pytest tests/v2/test_execution_policy.py -v
"""

from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def root():
    from core.v2.execution_policy import V2_PROJECT_ROOT
    return V2_PROJECT_ROOT


# ======================================================================
#  Raízes autorizadas
# ======================================================================

class TestApprovedRoots:
    def test_workspace_and_quarantine_resolve(self):
        from core.v2.execution_policy import approved_root
        workspace = approved_root("workspace")
        quarantine = approved_root("quarantine")
        assert workspace is not None and workspace.is_dir()
        assert quarantine is not None and quarantine.is_dir()
        assert workspace.name == "Workspace"
        assert quarantine.name == "Quarantine"
        assert workspace != quarantine

    def test_unknown_root_key_rejected(self):
        from core.v2.execution_policy import approved_root
        for purpose in ("", "   ", "project", "root", "bra", "../..", "Workspace"):
            assert approved_root(purpose) is None, purpose


class TestPurposePolicy:
    """Política central de propósitos: explícita, restritiva e não elevável."""

    def test_expected_purposes_exist(self):
        from core.v2.execution_policy import PURPOSES
        assert set(PURPOSES) == {
            "development", "testing", "analysis",
            "security_validation", "documentation",
        }

    def test_default_purpose_is_restrictive(self):
        from core.v2.execution_policy import DEFAULT_PURPOSE, resolve_purpose
        policy = resolve_purpose(DEFAULT_PURPOSE)
        assert policy.allow_interpreters is False
        assert policy.allow_write is False

    def test_permission_matrix(self):
        from core.v2.execution_policy import resolve_purpose
        expected = {
            "development":         ("workspace",  True,  True),
            "testing":             ("workspace",  True,  False),
            "analysis":            ("workspace",  False, False),
            "documentation":       ("workspace",  False, True),
            "security_validation": ("quarantine", False, False),
        }
        for name, (root_key, interp, write) in expected.items():
            policy = resolve_purpose(name)
            assert policy is not None, name
            assert (policy.root_key, policy.allow_interpreters, policy.allow_write) == \
                   (root_key, interp, write), name

    def test_case_variants_rejected(self):
        from core.v2.execution_policy import resolve_purpose
        for name in ("Development", "DEVELOPMENT", " development", "development ",
                     "security-validation", "Security_Validation"):
            assert resolve_purpose(name) is None, name

    def test_non_string_rejected(self):
        from core.v2.execution_policy import resolve_purpose
        for value in (None, 1, [], {}, True):
            assert resolve_purpose(value) is None, value

    def test_write_is_fail_closed_for_invalid_purpose(self):
        from core.v2.execution_policy import purpose_allows_write
        assert purpose_allows_write(None) is True          # compatibilidade
        assert purpose_allows_write("root") is False       # inválido -> nega
        assert purpose_allows_write("analysis") is False
        assert purpose_allows_write("development") is True

    def test_kill_switch_only_removes_capability(self, monkeypatch):
        """O interruptor global nunca concede: só desliga."""
        from core.v2.execution_policy import (INTERPRETER_KILL_SWITCH_ENV,
                                              purpose_allows_interpreters)
        assert purpose_allows_interpreters("development") is True
        monkeypatch.setenv(INTERPRETER_KILL_SWITCH_ENV, "1")
        assert purpose_allows_interpreters("development") is False
        assert purpose_allows_interpreters("analysis") is False

    def test_deprecated_env_does_not_grant(self, monkeypatch):
        """ODOYSSEUS_V2_ALLOW_INTERPRETERS já não concede capacidade."""
        from core.v2.execution_policy import (build_tool_context,
                                              purpose_allows_interpreters)
        monkeypatch.setenv("ODYSSEUS_V2_ALLOW_INTERPRETERS", "1")
        assert purpose_allows_interpreters("analysis") is False
        assert "allow_interpreters" not in build_tool_context("analysis")

    def test_protected_and_outside_roots_rejected(self, root):
        """Nenhuma área protegida nem caminho externo é raiz aprovada."""
        from core.v2.execution_policy import approved_root, APPROVED_WORKSPACE_ROOTS
        # A tabela só admite as duas pastas aprovadas.
        assert set(APPROVED_WORKSPACE_ROOTS) == {"workspace", "quarantine"}
        assert set(APPROVED_WORKSPACE_ROOTS.values()) == {"Workspace", "Quarantine"}
        # E as raízes aprovadas não são áreas protegidas.
        from core.v2.tools import _is_protected_path
        assert _is_protected_path(approved_root("workspace")) is False
        assert _is_protected_path(approved_root("quarantine")) is False


# ======================================================================
#  Contexto de ferramentas (servidor)
# ======================================================================

class TestToolContext:
    def test_build_uses_default_purpose(self):
        from core.v2.execution_policy import (DEFAULT_PURPOSE, approved_root,
                                              build_tool_context)
        ctx = build_tool_context()
        assert ctx["workspace_purpose"] == DEFAULT_PURPOSE
        assert ctx["workspace_path"] == str(approved_root("workspace"))
        assert "allow_interpreters" not in ctx  # analysis não permite interpretadores

    def test_build_security_validation_uses_quarantine(self):
        from core.v2.execution_policy import approved_root, build_tool_context
        ctx = build_tool_context("security_validation")
        assert ctx["workspace_path"] == str(approved_root("quarantine"))
        assert ctx["workspace_purpose"] == "security_validation"
        assert "allow_interpreters" not in ctx

    def test_invalid_purpose_yields_empty_context(self):
        """Contexto vazio => as ferramentas de execução negam (fail-closed)."""
        from core.v2.execution_policy import build_tool_context
        for purpose in ("", None, "workspace", "quarantine", "Development", "root"):
            assert build_tool_context(purpose) == {}, purpose

    def test_sanitize_strips_reserved_keys(self):
        from core.v2.execution_policy import sanitize_caller_context
        clean = sanitize_caller_context({
            "workspace_path": "C:\\",
            "workspace": "C:\\",
            "workspace_purpose": "development",
            "cwd": "C:\\Windows",
            "workdir": "..",
            "allow_interpreters": True,
            "session_id": "s1",
            "model": "m",
        })
        assert clean == {"session_id": "s1", "model": "m"}

    def test_caller_cannot_elevate_purpose(self):
        """O chamador não consegue transformar analysis em development."""
        from core.v2.execution_policy import sanitize_caller_context
        clean = sanitize_caller_context({"workspace_purpose": "development"})
        assert "workspace_purpose" not in clean

    def test_sanitize_handles_non_dict(self):
        from core.v2.execution_policy import sanitize_caller_context
        assert sanitize_caller_context(None) == {}
        assert sanitize_caller_context(["workspace_path"]) == {}


# ======================================================================
#  Política de comandos e interpretadores
# ======================================================================

class TestInterpreterPolicy:
    def test_interpreters_denied_by_default(self, root):
        from core.v2.execution_policy import check_command
        for cmd in ("python --version", "python3 -V", "pytest -q",
                    "python -m pytest"):
            result = check_command(cmd, root)
            assert result["allowed"] is False, cmd
            assert "permitidos" in result["reason"]

    def test_interpreters_allowed_with_server_flag(self, root):
        from core.v2.execution_policy import check_command
        assert check_command("python --version", root, allow_interpreters=True)["allowed"] is True
        assert check_command("python -m pytest", root, allow_interpreters=True)["allowed"] is True

    def test_inline_code_still_denied_even_when_authorized(self, root):
        """Autorizar interpretadores não autoriza código inline no ShellTool."""
        from core.v2.execution_policy import check_command
        assert check_command("python -c print(1)", root, allow_interpreters=True)["allowed"] is False

    def test_git_always_allowed_read_only(self, root):
        from core.v2.execution_policy import check_command
        assert check_command("git status", root)["allowed"] is True
        assert check_command("git push", root)["allowed"] is False


# ======================================================================
#  Fingerprint e redacção
# ======================================================================

class TestFingerprintAndRedaction:
    def test_fingerprint_binds_params(self):
        from core.v2.execution_policy import operation_fingerprint
        a = operation_fingerprint("tool:shell", {"tool": "shell", "params": {"command": "git status"}})
        b = operation_fingerprint("tool:shell", {"tool": "shell", "params": {"command": "git log"}})
        c = operation_fingerprint("tool:shell", {"tool": "shell", "params": {"command": "git status"}})
        assert a != b
        assert a == c

    def test_fingerprint_is_deterministic_across_key_order(self):
        from core.v2.execution_policy import operation_fingerprint
        a = operation_fingerprint("t", {"b": 1, "a": 2})
        b = operation_fingerprint("t", {"a": 2, "b": 1})
        assert a == b

    def test_redaction_removes_secrets(self):
        from core.v2.execution_policy import redact_params
        red = redact_params({
            "command": "git status",
            "api_key": "segredo-123",
            "nested": {"password": "p", "token": "t", "ok": "visivel"},
        })
        assert red["command"] == "git status"
        assert red["api_key"] == "<redigido>"
        assert red["nested"]["password"] == "<redigido>"
        assert red["nested"]["token"] == "<redigido>"
        assert red["nested"]["ok"] == "visivel"
        assert "segredo-123" not in str(red)

    def test_redaction_truncates_long_strings(self):
        from core.v2.execution_policy import redact_params
        red = redact_params({"code": "x" * 500}, max_string=50)
        assert len(red["code"]) < 80
        assert red["code"].endswith("…[truncado]")


# ======================================================================
#  Sonda de isolamento — estado real, sem optimismo
# ======================================================================

class TestIsolationProbe:
    def test_probe_reports_real_state(self):
        from core.v2.execution_policy import isolation_backends
        report = isolation_backends()
        # Chaves obrigatórias e coerência do veredicto.
        for key in ("windows_sandbox", "docker_cli", "docker_running",
                    "wsl_cli", "effective_isolation_available", "verdict"):
            assert key in report
        expected = bool(report["windows_sandbox"] or report["docker_running"])
        assert report["effective_isolation_available"] is expected
        # WSL nunca é apresentado como fronteira de segurança.
        assert report["wsl_is_security_boundary"] is False

    def test_verdict_does_not_claim_isolation_without_backend(self):
        from core.v2.execution_policy import isolation_backends
        report = isolation_backends()
        if not report["effective_isolation_available"]:
            assert "sem backend" in report["verdict"]
            assert "disponível" not in report["verdict"]


# ======================================================================
#  Terminação de processos (timeouts e descendentes)
# ======================================================================

class TestProcessTermination:
    """Verifica a terminação real e os caminhos de recurso, sem optimismo."""

    @pytest.mark.asyncio
    async def test_terminates_real_process(self):
        """Um processo real é terminado e fica com código de saída definido."""
        import asyncio
        import sys
        from core.v2.execution_policy import terminate_process_tree

        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(30)",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        assert proc.returncode is None

        outcome = await terminate_process_tree(proc)

        assert isinstance(outcome, bool)
        # A garantia observável: o processo já não está a correr.
        assert proc.returncode is not None

    @pytest.mark.asyncio
    async def test_finished_process_is_noop(self):
        import asyncio
        import sys
        from core.v2.execution_policy import terminate_process_tree

        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "pass",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        assert await terminate_process_tree(proc) is True

    @pytest.mark.asyncio
    async def test_fallback_kills_direct_process_without_pid(self):
        """Sem PID conhecido não há árvore: termina o processo directo e di-lo."""
        from core.v2.execution_policy import terminate_process_tree

        class _NoPidProc:
            pid = None
            returncode = None

            def __init__(self):
                self.killed = False

            def kill(self):
                self.killed = True
                self.returncode = -9

            async def wait(self):
                return self.returncode

        proc = _NoPidProc()
        outcome = await terminate_process_tree(proc)

        assert outcome is False          # não conseguiu garantir a árvore
        assert proc.killed is True
        assert proc.returncode == -9


# ======================================================================
#  Fluxo de autorização: contexto, propósito e capacidades
# ======================================================================

class TestAuthorizationFlowContext:
    """O contexto de execução é do servidor; a autorização é por operação."""

    def test_context_never_carries_caller_workspace(self):
        from core.v2.execution_policy import (approved_root, build_tool_context,
                                              sanitize_caller_context)
        caller = {"workspace_path": "C:\\Windows", "allow_interpreters": True,
                  "workspace_purpose": "development"}
        ctx = {**sanitize_caller_context(caller), **build_tool_context("analysis")}
        assert ctx["workspace_path"] == str(approved_root("workspace"))
        assert ctx["workspace_purpose"] == "analysis"
        assert "allow_interpreters" not in ctx

    def test_purpose_capabilities_are_not_in_context_by_default(self):
        from core.v2.execution_policy import build_tool_context
        for purpose in ("analysis", "documentation", "security_validation"):
            assert "allow_interpreters" not in build_tool_context(purpose), purpose
