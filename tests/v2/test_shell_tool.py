"""
Testes Unitários — ShellTool

Cobertura:
    - Definição: nome, capacidades, requires_approval
    - _check_safety: comandos bloqueados, permitidos, desconhecidos (filtro legado)
    - Política de comandos (execution_policy.check_command)
    - Autorização: chamada directa negada, execução via ToolManager
    - Confinamento do workdir ao workspace autorizado
    - Contrato de subprocesso: exec (sem shell), argv, cwd, desfechos
    - Aprovação: ausente, válida, inválida, reutilizada para outra operação

Nenhum comando real é executado: `asyncio.create_subprocess_exec` é sempre
substituído por um duplo controlado.

Uso:
    pytest tests/v2/test_shell_tool.py -v
"""

from __future__ import annotations

import sys
import os
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def shell_tool():
    from core.v2.tools import ShellTool
    return ShellTool()


@pytest.fixture
def workspace():
    """Workspace autorizado real: a raiz do projecto (não é área protegida)."""
    from core.v2.tools import V2_PROJECT_ROOT
    return V2_PROJECT_ROOT


@contextmanager
def _authorized(tool_name: str = "shell"):
    """Simula o marcador que só o ToolManager activa após aprovação."""
    from core.v2.execution_policy import authorized_execution
    with authorized_execution(tool_name, "EXEC-TEST-0001"):
        yield


def _ctx(ws):
    return {"workspace_path": str(ws)}


# ======================================================================
#  Tests: Definição
# ======================================================================

class TestShellToolDefinition:
    def test_name(self, shell_tool):
        assert shell_tool.name == "shell"

    def test_capabilities(self, shell_tool):
        assert "shell.execute" in shell_tool.definition.capabilities
        assert "shell.check" in shell_tool.definition.capabilities

    def test_requires_approval(self, shell_tool):
        assert shell_tool.definition.requires_approval is True

    def test_category(self, shell_tool):
        assert shell_tool.definition.category == "system"

    def test_timeout(self, shell_tool):
        assert shell_tool.definition.timeout == 60.0

    def test_required_params(self, shell_tool):
        required = shell_tool.definition.parameters["required"]
        assert "command" in required


# ======================================================================
#  Tests: _check_safety (filtro legado, mantido como defesa em profundidade)
# ======================================================================

class TestShellToolCheckSafety:
    def test_blocked_rm_rf(self, shell_tool):
        assert shell_tool._check_safety("rm -rf /")["safe"] is False

    def test_blocked_mkfs(self, shell_tool):
        assert shell_tool._check_safety("mkfs.ext4 /dev/sda1")["safe"] is False

    def test_blocked_dd(self, shell_tool):
        assert shell_tool._check_safety("dd if=/dev/zero of=/dev/sda")["safe"] is False

    def test_blocked_sudo(self, shell_tool):
        assert shell_tool._check_safety("sudo rm -rf")["safe"] is False

    def test_blocked_su(self, shell_tool):
        assert shell_tool._check_safety("su - root")["safe"] is False

    def test_blocked_chmod_777(self, shell_tool):
        assert shell_tool._check_safety("chmod 777 /etc/passwd")["safe"] is False

    def test_blocked_reboot(self, shell_tool):
        assert shell_tool._check_safety("reboot now")["safe"] is False

    def test_blocked_docker_rm(self, shell_tool):
        assert shell_tool._check_safety("docker rm container")["safe"] is False

    def test_allowed_ls(self, shell_tool):
        assert shell_tool._check_safety("ls -la")["safe"] is True

    def test_allowed_cat(self, shell_tool):
        assert shell_tool._check_safety("cat /etc/hosts")["safe"] is True

    def test_allowed_python(self, shell_tool):
        assert shell_tool._check_safety("python --version")["safe"] is True

    def test_allowed_git(self, shell_tool):
        assert shell_tool._check_safety("git status")["safe"] is True

    def test_allowed_grep(self, shell_tool):
        assert shell_tool._check_safety("grep -r 'foo' .")["safe"] is True

    def test_allowed_echo(self, shell_tool):
        assert shell_tool._check_safety("echo hello")["safe"] is True

    def test_unknown_command(self, shell_tool):
        assert shell_tool._check_safety("some_random_tool --danger")["safe"] is False

    def test_empty_command(self, shell_tool):
        assert shell_tool._check_safety("")["safe"] is False

    def test_case_insensitive_block(self, shell_tool):
        assert shell_tool._check_safety("RM -RF /")["safe"] is False

    def test_blocked_in_substring(self, shell_tool):
        assert shell_tool._check_safety("echo 'rm -rf /'")["safe"] is False


# ======================================================================
#  Tests: Política explícita de comandos
# ======================================================================

class TestCommandPolicy:
    def _check(self, command, workspace, platform=None):
        from core.v2.execution_policy import check_command
        return check_command(command, workspace, platform=platform)

    def test_common_commands_allowed_on_both_platforms(self, workspace):
        # `git` (leitura) é permitido em ambas as plataformas. Interpretadores
        # deixaram de estar na lista por omissão na Etapa 4.
        for cmd in ("git status", "git log --oneline"):
            for platform in ("posix", "nt"):
                assert self._check(cmd, workspace, platform)["allowed"] is True, (cmd, platform)
        for platform in ("posix", "nt"):
            assert self._check("python --version", workspace, platform)["allowed"] is False

    def test_posix_inspection_allowed_on_posix(self, workspace):
        for cmd in ("ls -la", "cat README.md", "grep README .", "wc README.md", "echo hello"):
            assert self._check(cmd, workspace, "posix")["allowed"] is True, cmd

    def test_windows_inspection_allowed_on_windows(self, workspace):
        for cmd in ("where python", "findstr README README.md", "more README.md", "tree core"):
            assert self._check(cmd, workspace, "nt")["allowed"] is True, cmd

    def test_platform_lists_are_not_interchangeable(self, workspace):
        """A lista é consciente da plataforma: nomes não portáveis não passam."""
        assert self._check("echo hello", workspace, "nt")["allowed"] is False
        assert self._check("where python", workspace, "posix")["allowed"] is False

    def test_unknown_executable_denied(self, workspace):
        assert self._check("curl http://example.com", workspace)["allowed"] is False
        assert self._check("rm -rf .", workspace)["allowed"] is False
        assert self._check("powershell -c x", workspace)["allowed"] is False

    def test_chaining_and_substitution_denied(self, workspace):
        for cmd in ("echo a; rm -rf /", "echo a && ls", "echo a | cat",
                    "echo $(whoami)", "echo `whoami`", "ls > out.txt", "ls < in.txt"):
            assert self._check(cmd, workspace)["allowed"] is False, cmd

    def test_indirect_execution_tokens_denied(self, workspace):
        from core.v2.execution_policy import check_command
        # `find -exec` cai já pela charset/allow-list; verifica o token directo.
        assert check_command("git log -exec", workspace)["allowed"] is False

    def test_denylist_blocks_before_policy(self, shell_tool):
        """A deny-list de destrutivos é aplicada antes da allow-list."""
        assert shell_tool._check_denylist("rm -rf /") is not None
        assert shell_tool._check_denylist("sudo su") is not None
        assert shell_tool._check_denylist("git status") is None
        # `rm` não está na allow-list, mas a deny-list apanha a variante perigosa
        # mesmo que a política viesse a permitir `rm`.
        assert shell_tool._check_denylist("rm -rf / --no-preserve-root") is not None

    def test_git_write_subcommands_denied(self, workspace):
        for cmd in ("git clone http://x", "git push", "git reset --hard",
                    "git checkout main", "git clean -fd"):
            assert self._check(cmd, workspace)["allowed"] is False, cmd

    def test_python_inline_code_denied(self, workspace):
        assert self._check("python -c print(1)", workspace)["allowed"] is False

    def test_absolute_path_outside_workspace_denied(self, workspace):
        assert self._check("cat ../../../Windows/win.ini", workspace, "posix")["allowed"] is False
        assert self._check("cat /etc/passwd", workspace, "posix")["allowed"] is False
        assert self._check("more ../../Windows/win.ini", workspace, "nt")["allowed"] is False

    def test_path_inside_workspace_allowed(self, workspace):
        assert self._check("cat core/v2/tools.py", workspace, "posix")["allowed"] is True
        assert self._check("more core/v2/tools.py", workspace, "nt")["allowed"] is True

    def test_protected_area_argument_denied(self, workspace):
        """Argumentos que resolvem para áreas protegidas são negados."""
        for cmd in ("cat Odysseus_Mind_OS/Home.md",
                    "cat data/app.db",
                    "cat logs/v2_bridge.log"):
            result = self._check(cmd, workspace, "posix")
            assert result["allowed"] is False, cmd
            assert "protegida" in result["reason"] or "fora do workspace" in result["reason"]

    def test_brain_vault_with_spaces_is_not_addressable(self, workspace):
        """Limitação declarada, verificada em vez de disfarçada.

        "Obsidian Brain Vault" contém espaços. As aspas são rejeitadas pela
        charset e a tokenização é feita por espaços, pelo que o caminho do vault
        não é endereçável como um único argumento: o comando resulta em três
        argumentos que não resolvem para a área protegida e falham no sistema de
        ficheiros. Nada do vault é lido.
        """
        from core.v2.tools import _resolve_confined
        result = self._check("cat Obsidian Brain Vault/README.md", workspace, "posix")
        # O token que sobra resolve dentro do workspace, não no vault:
        assert _resolve_confined("Vault/README.md", workspace) is not None
        # E não existe forma de expressar o caminho com espaços (aspas negadas):
        assert self._check('cat "Obsidian Brain Vault/README.md"', workspace, "posix")["allowed"] is False
        # O comando é aceite como três argumentos que não alcançam o vault.
        assert result["allowed"] is True
        assert result["argv"] == ["cat", "Obsidian", "Brain", "Vault/README.md"]


# ======================================================================
#  Tests: Autorização e workspace
# ======================================================================

class TestShellToolAuthorizationGuard:
    @pytest.mark.asyncio
    async def test_direct_call_is_denied(self, shell_tool, workspace, monkeypatch):
        """Chamada directa fora do ToolManager não executa."""
        import asyncio
        spawned = []

        async def fake_exec(*args, **kwargs):
            spawned.append(args)
            return _FakeProc(returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        result = await shell_tool.execute({"command": "git status"}, context=_ctx(workspace))

        assert result.success is False
        assert result.metadata.get("reason") == "not_authorized"
        assert spawned == []

    @pytest.mark.asyncio
    async def test_authorized_without_workspace_is_denied(self, shell_tool, monkeypatch):
        import asyncio
        spawned = []

        async def fake_exec(*args, **kwargs):
            spawned.append(args)
            return _FakeProc(returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        with _authorized():
            result = await shell_tool.execute({"command": "git status"})

        assert result.success is False
        assert result.metadata.get("reason") == "no_workspace"
        assert spawned == []

    @pytest.mark.asyncio
    async def test_protected_workspace_is_rejected(self, shell_tool, monkeypatch):
        import asyncio
        from core.v2.tools import V2_PROJECT_ROOT
        spawned = []

        async def fake_exec(*args, **kwargs):
            spawned.append(args)
            return _FakeProc(returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        with _authorized():
            result = await shell_tool.execute(
                {"command": "git status"},
                context=_ctx(V2_PROJECT_ROOT / "Obsidian Brain Vault"),
            )

        assert result.success is False
        assert result.metadata.get("reason") == "no_workspace"
        assert spawned == []

    @pytest.mark.asyncio
    async def test_authorization_for_other_tool_does_not_help(self, shell_tool, workspace, monkeypatch):
        import asyncio
        spawned = []

        async def fake_exec(*args, **kwargs):
            spawned.append(args)
            return _FakeProc(returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        with _authorized("python"):
            result = await shell_tool.execute({"command": "git status"}, context=_ctx(workspace))

        assert result.success is False
        assert spawned == []


# ======================================================================
#  Tests: Confinamento do workdir
# ======================================================================

class TestShellToolWorkdirConfinement:
    async def _run(self, shell_tool, workspace, monkeypatch, workdir, command="git status"):
        import asyncio
        seen = {}

        async def fake_exec(*args, **kwargs):
            seen["args"] = args
            seen["cwd"] = kwargs.get("cwd")
            return _FakeProc(stdout="ok", returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        with _authorized():
            result = await shell_tool.execute(
                {"command": command, "workdir": workdir}, context=_ctx(workspace)
            )
        return result, seen

    @pytest.mark.asyncio
    async def test_workdir_inside_workspace_is_used(self, shell_tool, workspace, monkeypatch):
        result, seen = await self._run(shell_tool, workspace, monkeypatch, "core")
        assert result.success is True
        assert seen["cwd"].replace("\\", "/").endswith("core")

    @pytest.mark.asyncio
    async def test_workdir_traversal_denied(self, shell_tool, workspace, monkeypatch):
        result, seen = await self._run(shell_tool, workspace, monkeypatch, "..")
        assert result.success is False
        assert result.metadata.get("reason") == "workdir_out_of_workspace"
        assert seen == {}

    @pytest.mark.asyncio
    async def test_workdir_absolute_outside_denied(self, shell_tool, workspace, monkeypatch):
        result, seen = await self._run(
            shell_tool, workspace, monkeypatch, str(workspace.parent)
        )
        assert result.success is False
        assert result.metadata.get("reason") == "workdir_out_of_workspace"
        assert seen == {}

    @pytest.mark.asyncio
    async def test_workdir_protected_area_denied(self, shell_tool, workspace, monkeypatch):
        result, seen = await self._run(
            shell_tool, workspace, monkeypatch, "Odysseus_Mind_OS"
        )
        assert result.success is False
        assert result.metadata.get("reason") == "workdir_out_of_workspace"
        assert seen == {}

    @pytest.mark.asyncio
    async def test_workdir_missing_denied(self, shell_tool, workspace, monkeypatch):
        result, seen = await self._run(
            shell_tool, workspace, monkeypatch, "nao_existe_xyz"
        )
        assert result.success is False
        assert result.metadata.get("reason") == "workdir_missing"
        assert seen == {}


# ======================================================================
#  Tests: Execução (duplo de processo; sem comandos reais)
# ======================================================================

class _FakeProc:
    """Duplo de processo: regista kill/wait sem executar nada."""

    def __init__(self, stdout="", stderr="", returncode=0, block=False, pid=None):
        self._stdout = stdout
        self._stderr = stderr
        # Um processo bloqueado ainda não tem código de saída — é isso que
        # distingue "não terminou" de "terminou com sucesso".
        self.returncode = None if block else returncode
        self._block = block
        self.pid = pid
        self.killed = False
        self.waited = False

    async def communicate(self):
        if self._block:
            import asyncio
            await asyncio.sleep(30)
        return self._stdout, self._stderr

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        self.waited = True
        if self.returncode is None:
            self.returncode = -9
        return self.returncode


class TestShellToolSubprocessContract:
    @pytest.mark.asyncio
    async def test_runs_with_exec_and_argv(self, shell_tool, workspace, monkeypatch):
        """Sem shell: argv explícito, cwd no workspace, sem `shell`/`args`."""
        import asyncio
        calls = []

        async def fake_exec(*args, **kwargs):
            calls.append((args, kwargs))
            return _FakeProc(stdout="ok\n", returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        with _authorized():
            result = await shell_tool.execute(
                {"command": "git status --short"}, context=_ctx(workspace)
            )

        assert result.success is True
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args == ("git", "status", "--short")
        assert "args" not in kwargs and "shell" not in kwargs
        assert kwargs.get("cwd") == str(workspace)
        assert result.data["output_captured"] is True
        assert result.data["stdout"] == "ok\n"

    @pytest.mark.asyncio
    async def test_blocked_command_never_spawns(self, shell_tool, workspace, monkeypatch):
        import asyncio
        spawned = []

        async def fake_exec(*args, **kwargs):
            spawned.append(args)
            return _FakeProc()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        with _authorized():
            result = await shell_tool.execute({"command": "rm -rf /"}, context=_ctx(workspace))

        assert result.success is False
        assert spawned == []

    @pytest.mark.asyncio
    async def test_nonzero_exit_is_failure(self, shell_tool, workspace, monkeypatch):
        import asyncio

        async def fake_exec(*args, **kwargs):
            return _FakeProc(stdout="", stderr="boom", returncode=2)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        with _authorized():
            result = await shell_tool.execute({"command": "git status"}, context=_ctx(workspace))

        assert result.success is False
        assert result.data["return_code"] == 2
        assert "boom" in result.error

    @pytest.mark.asyncio
    async def test_return_code_none_is_failure(self, shell_tool, workspace, monkeypatch):
        """Código de saída ausente nunca é tratado como sucesso."""
        import asyncio
        proc_holder = {}

        async def fake_exec(*args, **kwargs):
            proc = _FakeProc(stdout="", returncode=None)
            proc_holder["proc"] = proc
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        with _authorized():
            result = await shell_tool.execute({"command": "git status"}, context=_ctx(workspace))

        assert result.success is False
        assert result.metadata.get("return_code") is None
        assert "return_code=None" in result.error

    @pytest.mark.asyncio
    async def test_timeout_terminates_child(self, shell_tool, workspace, monkeypatch):
        import asyncio
        procs = []

        async def fake_exec(*args, **kwargs):
            proc = _FakeProc(block=True)
            procs.append(proc)
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        with _authorized():
            result = await shell_tool.execute(
                {"command": "git status", "timeout": 0.01}, context=_ctx(workspace)
            )

        assert result.success is False
        assert "timeout" in result.error.lower()
        assert len(procs) == 1
        assert procs[0].killed is True
        assert procs[0].waited is True
        assert result.metadata.get("terminated") is False  # só o processo directo

    @pytest.mark.asyncio
    async def test_spawn_error_is_captured(self, shell_tool, workspace, monkeypatch):
        import asyncio

        async def fake_exec(*args, **kwargs):
            raise OSError("spawn indisponivel")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        with _authorized():
            result = await shell_tool.execute({"command": "git status"}, context=_ctx(workspace))

        assert result.success is False
        assert result.metadata.get("spawn_failed") is True
        assert "spawn indisponivel" in result.error

    @pytest.mark.asyncio
    async def test_pipe_unavailable_falls_back_to_no_capture(
        self, shell_tool, workspace, monkeypatch
    ):
        """Ambiente sem pipes: executa sem captura e diz que não capturou."""
        import asyncio
        calls = []

        async def fake_exec(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise PermissionError("[WinError 5] Acesso negado")
            return _FakeProc(returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        with _authorized():
            result = await shell_tool.execute({"command": "git status"}, context=_ctx(workspace))

        assert len(calls) == 2
        assert "text" not in calls[1]
        assert result.success is True
        assert result.data["output_captured"] is False
        assert result.data["stdout"] is None
        assert result.warnings and "não capturado" in result.warnings[0].lower()


# ======================================================================
#  Tests: Aprovação (via ToolManager, com SafetyManager real)
# ======================================================================

class TestShellToolApprovalGate:
    def _manager(self, safety):
        from core.v2.tool_manager import ToolManager
        from core.v2.tool_registry import ToolRegistry
        from core.v2.tools import ShellTool
        registry = ToolRegistry()
        registry.register(ShellTool())
        return ToolManager(registry=registry, safety_manager=safety)

    @pytest.mark.asyncio
    async def test_no_approval_means_no_spawn(self, workspace, monkeypatch):
        import asyncio
        from core.v2.safety import SafetyManager
        spawned = []

        async def fake_exec(*args, **kwargs):
            spawned.append(args)
            return _FakeProc(returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        safety = SafetyManager()
        manager = self._manager(safety)

        result = await manager.execute(
            "shell", {"command": "git status"}, context=_ctx(workspace)
        )

        assert result.success is False
        assert result.metadata.get("approval_id")
        assert spawned == []
        assert safety.count_pending() == 1

    @pytest.mark.asyncio
    async def test_approved_command_spawns_once(self, workspace, monkeypatch):
        import asyncio
        from core.v2.safety import SafetyManager
        spawned = []

        async def fake_exec(*args, **kwargs):
            spawned.append(args)
            return _FakeProc(stdout="ok\n", returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        safety = SafetyManager()
        manager = self._manager(safety)

        blocked = await manager.execute(
            "shell", {"command": "git status"}, context=_ctx(workspace)
        )
        approval_id = blocked.metadata["approval_id"]
        approved, _ = safety.approve(approval_id)
        assert approved is True

        result = await manager.execute(
            "shell", {"command": "git status"},
            context=_ctx(workspace), approval_id=approval_id,
        )

        assert result.success is True
        assert spawned == [("git", "status")]

    @pytest.mark.asyncio
    async def test_approval_cannot_be_reused_for_another_command(self, workspace, monkeypatch):
        """A aprovação está ligada à operação concreta."""
        import asyncio
        from core.v2.safety import SafetyManager
        spawned = []

        async def fake_exec(*args, **kwargs):
            spawned.append(args)
            return _FakeProc(returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        safety = SafetyManager()
        manager = self._manager(safety)

        blocked = await manager.execute(
            "shell", {"command": "git status"}, context=_ctx(workspace)
        )
        approval_id = blocked.metadata["approval_id"]
        safety.approve(approval_id)

        # Mesma aprovação, comando diferente -> negado e nada executa.
        result = await manager.execute(
            "shell", {"command": "git log"},
            context=_ctx(workspace), approval_id=approval_id,
        )

        assert result.success is False
        assert spawned == []
        assert "não corresponde" in (result.error or "")

    @pytest.mark.asyncio
    async def test_approval_cannot_be_reused_with_different_workdir(self, workspace, monkeypatch):
        import asyncio
        from core.v2.safety import SafetyManager
        spawned = []

        async def fake_exec(*args, **kwargs):
            spawned.append(args)
            return _FakeProc(returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        safety = SafetyManager()
        manager = self._manager(safety)

        blocked = await manager.execute(
            "shell", {"command": "git status", "workdir": "core"},
            context=_ctx(workspace),
        )
        approval_id = blocked.metadata["approval_id"]
        safety.approve(approval_id)

        result = await manager.execute(
            "shell", {"command": "git status", "workdir": "tests"},
            context=_ctx(workspace), approval_id=approval_id,
        )

        assert result.success is False
        assert spawned == []

    @pytest.mark.asyncio
    async def test_safety_failure_is_fail_closed(self, workspace, monkeypatch):
        import asyncio
        spawned = []

        async def fake_exec(*args, **kwargs):
            spawned.append(args)
            return _FakeProc(returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        broken = MagicMock()
        broken.check_action.side_effect = TypeError("assinatura incompativel")
        manager = self._manager(broken)

        result = await manager.execute(
            "shell", {"command": "git status"}, context=_ctx(workspace)
        )

        assert result.success is False
        assert spawned == []
        assert "indispon" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_missing_safety_manager_is_fail_closed(self, workspace, monkeypatch):
        import asyncio
        spawned = []

        async def fake_exec(*args, **kwargs):
            spawned.append(args)
            return _FakeProc(returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        manager = self._manager(None)
        result = await manager.execute(
            "shell", {"command": "git status"}, context=_ctx(workspace)
        )

        assert result.success is False
        assert spawned == []


# ======================================================================
#  Tests: Interpretadores e separação de raízes (Etapa 4)
# ======================================================================

class TestInterpreterAuthorization:
    """`python`/`pytest` só com autorização do servidor — e ainda assim com aprovação."""

    def _manager(self, safety):
        from core.v2.tool_manager import ToolManager
        from core.v2.tool_registry import ToolRegistry
        from core.v2.tools import ShellTool
        registry = ToolRegistry()
        registry.register(ShellTool())
        return ToolManager(registry=registry, safety_manager=safety)

    @pytest.mark.asyncio
    async def test_python_denied_by_default_after_approval(self, monkeypatch):
        import asyncio
        from core.v2.safety import SafetyManager
        spawned = []

        async def fake_exec(*args, **kwargs):
            spawned.append(args)
            return _FakeProc(returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        safety = SafetyManager()
        manager = self._manager(safety)
        from core.v2.execution_policy import build_tool_context
        ctx = build_tool_context("analysis")  # servidor: propósito sem interpretadores

        blocked = await manager.execute("shell", {"command": "python --version"}, context=ctx)
        approval_id = blocked.metadata["approval_id"]
        safety.approve(approval_id)

        result = await manager.execute(
            "shell", {"command": "python --version"}, context=ctx, approval_id=approval_id
        )

        assert result.success is False
        assert "permitidos" in result.error.lower()
        assert spawned == []

    @pytest.mark.asyncio
    async def test_python_allowed_when_server_authorizes(self, monkeypatch):
        import asyncio
        from core.v2.safety import SafetyManager
        spawned = []

        async def fake_exec(*args, **kwargs):
            spawned.append(args)
            return _FakeProc(stdout="Python 3.x", returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        safety = SafetyManager()
        manager = self._manager(safety)
        from core.v2.execution_policy import build_tool_context
        ctx = build_tool_context("development")  # propósito com interpretadores

        blocked = await manager.execute("shell", {"command": "python --version"}, context=ctx)
        approval_id = blocked.metadata["approval_id"]
        safety.approve(approval_id)

        result = await manager.execute(
            "shell", {"command": "python --version"}, context=ctx, approval_id=approval_id
        )

        assert result.success is True
        assert spawned and spawned[0][0] in ("python", "python3")

    @pytest.mark.asyncio
    async def test_inline_code_denied_even_when_authorized(self, monkeypatch):
        import asyncio
        from core.v2.safety import SafetyManager
        spawned = []

        async def fake_exec(*args, **kwargs):
            spawned.append(args)
            return _FakeProc(returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        safety = SafetyManager()
        manager = self._manager(safety)
        from core.v2.execution_policy import build_tool_context
        ctx = build_tool_context("development")

        blocked = await manager.execute("shell", {"command": "python -c print(1)"}, context=ctx)
        approval_id = blocked.metadata["approval_id"]
        safety.approve(approval_id)

        result = await manager.execute(
            "shell", {"command": "python -c print(1)"}, context=ctx, approval_id=approval_id
        )

        assert result.success is False
        assert spawned == []


class TestCrossRootSeparation:
    """Workspace e Quarantine são raízes distintas: uma não alcança a outra."""

    @pytest.mark.asyncio
    async def test_workdir_quarantine_denied_from_workspace(self, monkeypatch):
        import asyncio
        from core.v2.execution_policy import approved_root, build_tool_context
        spawned = []

        async def fake_exec(*args, **kwargs):
            spawned.append(args)
            return _FakeProc(returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        ctx = build_tool_context("development")     # raiz Workspace
        quarantine = approved_root("quarantine")    # outra raiz aprovada

        with _authorized():
            result = await shell_tool_stub().execute(
                {"command": "git status", "workdir": str(quarantine)}, context=ctx
            )

        assert result.success is False
        assert result.metadata.get("reason") == "workdir_out_of_workspace"
        assert spawned == []

    @pytest.mark.asyncio
    async def test_protected_areas_denied_as_workdir(self, monkeypatch):
        import asyncio
        from core.v2.execution_policy import build_tool_context
        spawned = []

        async def fake_exec(*args, **kwargs):
            spawned.append(args)
            return _FakeProc(returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        ctx = build_tool_context("development")
        tool = shell_tool_stub()

        for target in ("Obsidian Brain Vault", "Odysseus_Mind_OS", "data", "logs"):
            with _authorized():
                result = await tool.execute(
                    {"command": "git status", "workdir": target}, context=ctx
                )
            assert result.success is False, target

        assert spawned == []


def shell_tool_stub():
    from core.v2.tools import ShellTool
    return ShellTool()


# ======================================================================
#  Tests: Validação de parâmetros
# ======================================================================

class TestShellToolValidation:
    def test_valid_params(self, shell_tool):
        errors = shell_tool.validate_params({"command": "ls -la"})
        assert errors == []

    def test_missing_command(self, shell_tool):
        errors = shell_tool.validate_params({})
        assert len(errors) >= 1
