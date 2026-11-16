"""
Testes — isolamento real (Etapa 6)

Este ficheiro implementa a validação exigida em
`Quarantine/ISOLATION-REQUIREMENTS.md`.

Distinção essencial, verificada por testes:

    * a sonda descreve o estado real, sem optimismo;
    * o adaptador de backend só constrói um comando isolado quando existe
      backend **e** imagem, e recusa-se a fabricar um comando enganador;
    * a execução declara sempre `isolated` — hoje `False`;
    * quando o servidor exige isolamento (`ODYSSEUS_V2_REQUIRE_ISOLATION=1`) e não
      há backend, a execução é **recusada**.

Os testes que exigem um backend a sério estão marcados com `skipif` e o motivo
declarado: sem backend não se pode demonstrar a fronteira, e não se declara que
ela existe.

Uso:
    pytest tests/v2/test_isolation.py -v
"""

from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def workspace():
    from core.v2.execution_policy import approved_root
    root = approved_root("workspace")
    assert root is not None
    return root


def _status():
    from core.v2.isolation import isolation_status
    return isolation_status()


requires_backend = pytest.mark.skipif(
    not _status().isolated,
    reason=(
        "sem backend de isolamento disponível neste ambiente "
        f"({_status().detail}); a fronteira não pode ser demonstrada"
    ),
)


# ======================================================================
#  Estado real do isolamento
# ======================================================================

class TestIsolationStatus:
    def test_status_does_not_overclaim(self):
        status = _status()
        # `isolated` só é verdadeiro com backend disponível e imagem definida.
        assert status.isolated == bool(status.available and status.image)
        assert status.backend in ("docker", "none")
        if not status.available:
            assert status.isolated is False

    def test_status_exposes_probe(self):
        status = _status()
        assert isinstance(status.probe, dict)
        assert "windows_sandbox" in status.probe
        assert status.probe.get("wsl_is_security_boundary") is False

    def test_detail_is_honest_when_unavailable(self):
        status = _status()
        if not status.available:
            assert "sem backend" in status.detail or "indisponível" in status.detail

    def test_require_isolation_default_off(self, monkeypatch):
        from core.v2.isolation import isolation_required, isolation_refusal_reason
        monkeypatch.delenv("ODYSSEUS_V2_REQUIRE_ISOLATION", raising=False)
        assert isolation_required() is False
        assert isolation_refusal_reason() is None


# ======================================================================
#  Recusa honesta quando o isolamento é exigido
# ======================================================================

class TestIsolationRequiredRefusal:
    def test_refusal_when_required_without_backend(self, monkeypatch):
        from core.v2.isolation import isolation_refusal_reason
        monkeypatch.setenv("ODYSSEUS_V2_REQUIRE_ISOLATION", "1")
        reason = isolation_refusal_reason()
        if _status().isolated:
            assert reason is None
        else:
            assert reason is not None
            assert "indisponível" in reason

    @pytest.mark.asyncio
    async def test_shell_refuses_when_isolation_required(self, monkeypatch, workspace):
        import asyncio
        from core.v2.execution_policy import authorized_execution
        from core.v2.tools import ShellTool

        if _status().isolated:
            pytest.skip("backend disponível: não é este o cenário do teste")

        monkeypatch.setenv("ODYSSEUS_V2_REQUIRE_ISOLATION", "1")
        spawned = []

        async def fake_exec(*args, **kwargs):
            spawned.append(args)
            raise AssertionError("não devia executar sem isolamento")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        with authorized_execution("shell", "EXEC-ISO-1"):
            result = await ShellTool().execute(
                {"command": "git status"},
                context={"workspace_path": str(workspace), "workspace_purpose": "development"},
            )

        assert result.success is False
        assert result.metadata.get("isolation_required") is True
        assert result.metadata.get("isolated") is False
        assert spawned == []

    @pytest.mark.asyncio
    async def test_python_refuses_when_isolation_required(self, monkeypatch, workspace):
        import asyncio
        from core.v2.execution_policy import authorized_execution
        from core.v2.tools import PythonTool

        if _status().isolated:
            pytest.skip("backend disponível: não é este o cenário do teste")

        monkeypatch.setenv("ODYSSEUS_V2_REQUIRE_ISOLATION", "1")
        spawned = []

        async def fake_exec(*args, **kwargs):
            spawned.append(args)
            raise AssertionError("não devia executar sem isolamento")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        with authorized_execution("python", "EXEC-ISO-2"):
            result = await PythonTool().execute(
                {"action": "execute", "code": "print(1)"},
                context={"workspace_path": str(workspace), "workspace_purpose": "development"},
            )

        assert result.success is False
        assert result.metadata.get("isolation_required") is True
        assert spawned == []


# ======================================================================
#  Resultados declaram sempre se correram isolados
# ======================================================================

class TestResultsDeclareIsolation:
    @pytest.mark.asyncio
    async def test_real_execution_declares_not_isolated(self, monkeypatch):
        """Execução aprovada num cenário controlado sem backend."""
        from core.v2.execution_policy import approved_root, authorized_execution
        from core.v2.tools import ShellTool

        # Fixar a sonda e remover a imagem torna o cenário independente do Docker
        # e testa o caminho real de resolução de status sem backend.
        monkeypatch.setattr(
            "core.v2.isolation.isolation_backends",
            lambda: {
                "windows_sandbox": False,
                "docker_cli": False,
                "docker_running": False,
                "wsl_cli": False,
                "wsl_is_security_boundary": False,
                "effective_isolation_available": False,
                "verdict": "sem backend de isolamento: apenas contenção de localização",
            },
        )
        monkeypatch.delenv("ODYSSEUS_V2_ISOLATION_IMAGE", raising=False)
        monkeypatch.delenv("ODYSSEUS_V2_REQUIRE_ISOLATION", raising=False)

        status = _status()
        assert status.backend == "none"
        assert status.available is False
        assert status.image is None
        assert status.isolated is False

        root = approved_root("workspace")
        tool = ShellTool()

        with authorized_execution("shell", "EXEC-ISO-3"):
            result = await tool.execute(
                {"command": "git status", "timeout": 20},
                context={"workspace_path": str(root), "workspace_purpose": "development"},
            )

        assert result.success is True
        assert result.data["isolated"] is False
        assert result.metadata["isolation_backend"] == "none"
        assert "sem backend" in result.metadata["isolation_detail"] or \
               "indisponível" in result.metadata["isolation_detail"]


# ======================================================================
#  Comando do backend (construção verificável, sem executar)
# ======================================================================

class TestBackendCommand:
    def test_no_command_without_image(self, monkeypatch, workspace):
        from core.v2.isolation import build_backend_argv
        monkeypatch.delenv("ODYSSEUS_V2_ISOLATION_IMAGE", raising=False)
        assert build_backend_argv(["git", "status"], workspace) is None

    def test_no_command_without_backend(self, monkeypatch, workspace):
        from core.v2.isolation import build_backend_argv
        monkeypatch.setenv("ODYSSEUS_V2_ISOLATION_IMAGE", "imagem-teste")
        if _status().available:
            pytest.skip("backend disponível neste ambiente")
        assert build_backend_argv(["git", "status"], workspace) is None

    def test_command_shape_when_backend_available(self, monkeypatch, workspace):
        """Com backend, o comando tem de isolar rede, montagem e directório."""
        from core.v2 import isolation
        from core.v2.isolation import build_backend_argv

        monkeypatch.setenv("ODYSSEUS_V2_ISOLATION_IMAGE", "imagem-teste")
        monkeypatch.setattr(
            isolation, "isolation_status",
            lambda: isolation.IsolationStatus(
                backend="docker", available=True, image="imagem-teste",
                detail="simulado para verificar a forma do comando",
            ),
        )

        argv = build_backend_argv(["git", "status"], workspace, image="imagem-teste")

        assert argv is not None
        assert argv[0] == "docker" and argv[1] == "run"
        assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
        assert "--rm" in argv
        assert str(workspace) in " ".join(argv)
        assert argv[-2:] == ["git", "status"]
        # Endurecimento obrigatório
        assert "--cap-drop" in argv and argv[argv.index("--cap-drop") + 1] == "ALL"
        assert "--security-opt" in argv
        assert "no-new-privileges" in argv
        assert "--pids-limit" in argv
        assert "--memory" in argv
        assert "--cpus" in argv
        assert "--read-only" in argv
        # Nem socket do Docker nem credenciais entram no contentor
        assert not any("docker.sock" in a for a in argv)
        assert "-e" not in argv and "--env" not in argv
        assert "--privileged" not in argv
        assert "--mount" not in argv

    def test_image_reference_validation(self):
        """Uma imagem não pode injectar flags no `docker run`."""
        from core.v2.isolation import is_valid_image_reference as valid
        assert valid("python:3.12-slim") is True
        assert valid("python:3.12-slim@sha256:" + "a" * 64) is True
        assert valid("ghcr.io/org/img:v1") is True
        assert valid("--privileged") is False
        assert valid("-v") is False
        assert valid("python:3.12-slim --privileged") is False
        assert valid("") is False
        assert valid("python:tag@sha256:curto") is False

    def test_flag_injection_image_is_refused_in_argv(self, monkeypatch, workspace):
        from core.v2 import isolation
        from core.v2.isolation import build_backend_argv

        monkeypatch.setattr(
            isolation, "isolation_status",
            lambda: isolation.IsolationStatus(
                backend="docker", available=True, image="--privileged",
                detail="simulado",
            ),
        )
        assert build_backend_argv(["git", "status"], workspace, image="--privileged") is None

    def test_empty_argv_rejected(self, workspace):
        from core.v2.isolation import build_backend_argv
        assert build_backend_argv([], workspace) is None


# ======================================================================
#  Cablagem: com backend disponível, a execução passa por ele
# ======================================================================

def _fake_run(returncode=0, stdout="ok", stderr="", recorder=None, raises=None):
    """Duplo de `subprocess.run` que regista o argv sem executar Docker."""
    import subprocess

    def _run(args, **kwargs):
        if recorder is not None:
            recorder.append(tuple(args))
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(
            args=args, returncode=returncode, stdout=stdout, stderr=stderr
        )

    return _run


def _simulate_backend(monkeypatch):
    """Simula um backend Docker disponível (sem executar Docker)."""
    from core.v2 import isolation

    status = isolation.IsolationStatus(
        backend="docker", available=True, image="imagem-teste",
        detail="backend simulado para verificar a cablagem",
    )
    monkeypatch.setattr(isolation, "isolation_status", lambda: status)
    monkeypatch.setenv("ODYSSEUS_V2_ISOLATION_IMAGE", "imagem-teste")
    return status


class TestBackendWiring:
    """Com backend, o comando executado é o do backend (não o directo)."""

    @pytest.mark.asyncio
    async def test_shell_routes_through_backend(self, monkeypatch, workspace):
        from core.v2 import isolation
        from core.v2.execution_policy import authorized_execution
        from core.v2.tools import ShellTool

        _simulate_backend(monkeypatch)
        calls = []
        monkeypatch.setattr(isolation.subprocess, "run",
                            _fake_run(stdout="saida", recorder=calls))

        with authorized_execution("shell", "EXEC-ISO-W1"):
            result = await ShellTool().execute(
                {"command": "git status"},
                context={"workspace_path": str(workspace), "workspace_purpose": "development"},
            )

        assert result.success is True
        assert result.data["isolated"] is True
        # A primeira invocação é o `docker run`; as seguintes são a limpeza
        # (kill/rm/inspect) do contentor com nome próprio.
        argv = calls[0]
        assert argv[0] == "docker" and argv[1] == "run"
        assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
        assert "--name" in argv
        assert argv[-2:] == ("git", "status")
        assert all(c[0] == "docker" for c in calls)

    @pytest.mark.asyncio
    async def test_python_routes_through_backend(self, monkeypatch, workspace):
        from core.v2 import isolation
        from core.v2.execution_policy import authorized_execution
        from core.v2.tools import PythonTool

        _simulate_backend(monkeypatch)
        calls = []
        monkeypatch.setattr(isolation.subprocess, "run",
                            _fake_run(stdout="", recorder=calls))

        with authorized_execution("python", "EXEC-ISO-W2"):
            result = await PythonTool().execute(
                {"action": "execute", "code": "print(1)"},
                context={"workspace_path": str(workspace), "workspace_purpose": "development"},
            )

        assert result.success is True
        assert result.data["isolated"] is True
        argv = calls[0]
        assert argv[0] == "docker" and argv[1] == "run"
        # O interpretador é o do contentor, não o `sys.executable` do host.
        assert argv[-2] == "-c" and argv[-1] == "print(1)"
        assert "python" in argv[-3].lower()
        assert sys.executable not in argv
        assert all(c[0] == "docker" for c in calls)

    @pytest.mark.asyncio
    async def test_no_direct_fallback_when_isolation_required(self, monkeypatch, workspace):
        """Com isolamento exigido, uma falha do backend NÃO cai para execução directa."""
        import asyncio
        from core.v2 import isolation
        from core.v2.execution_policy import authorized_execution
        from core.v2.tools import ShellTool

        _simulate_backend(monkeypatch)
        monkeypatch.setenv("ODYSSEUS_V2_REQUIRE_ISOLATION", "1")
        monkeypatch.setattr(isolation.subprocess, "run",
                            _fake_run(raises=OSError("docker indisponivel")))

        direct_calls = []

        async def forbidden_direct(*args, **kwargs):
            direct_calls.append(args)
            raise AssertionError("não pode haver fallback para execução directa")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_direct)

        with authorized_execution("shell", "EXEC-ISO-NOFALLBACK"):
            result = await ShellTool().execute(
                {"command": "git status"},
                context={"workspace_path": str(workspace), "workspace_purpose": "development"},
            )

        assert result.success is False
        assert direct_calls == []                      # sem fallback directo
        assert "isolada" in (result.error or "").lower()
        assert result.metadata.get("isolated") is True  # correu (ou tentou) isolado

    @pytest.mark.asyncio
    async def test_backend_failure_is_reported_not_hidden(self, monkeypatch, workspace):
        from core.v2 import isolation
        from core.v2.execution_policy import authorized_execution
        from core.v2.tools import ShellTool

        _simulate_backend(monkeypatch)
        monkeypatch.setattr(isolation.subprocess, "run",
                            _fake_run(raises=OSError("docker indisponivel")))

        with authorized_execution("shell", "EXEC-ISO-W3"):
            result = await ShellTool().execute(
                {"command": "git status"},
                context={"workspace_path": str(workspace), "workspace_purpose": "development"},
            )

        assert result.success is False
        assert "isolada" in (result.error or "").lower()
        assert result.metadata.get("isolated") is True  # declarado, mesmo ao falhar


# ======================================================================
#  Timeout: terminação efectiva do contentor (unitário, motor simulado)
# ======================================================================

def _fake_docker_cli(recorder, *, kill_rc=0, rm_rc=0, kill_err="",
                     inspect_out="false", inspect_rc=0, raises=None):
    """Duplo de `isolation._docker_cli` (kill/rm/inspect)."""
    import subprocess

    async def _cli(args, timeout=15.0):
        recorder.append(tuple(args))
        if raises is not None:
            raise raises
        cmd = args[0]
        rc, out, err = 0, "", ""
        if cmd == "kill":
            rc, err = kill_rc, kill_err
        elif cmd == "rm":
            rc = rm_rc
        elif cmd == "inspect":
            rc, out = inspect_rc, inspect_out
        return subprocess.CompletedProcess(args=args, returncode=rc,
                                           stdout=out, stderr=err)

    return _cli


def _run_raises_timeout(monkeypatch):
    import subprocess as _sp

    def _run(*args, **kwargs):
        raise _sp.TimeoutExpired(cmd=args, timeout=kwargs.get("timeout", 1))

    from core.v2 import isolation
    monkeypatch.setattr(isolation.subprocess, "run", _run)


class TestContainerTermination:
    @pytest.mark.asyncio
    async def test_name_and_stop_timeout_in_command(self, monkeypatch, workspace):
        from core.v2 import isolation
        _simulate_backend(monkeypatch)
        name = isolation.new_container_name()
        argv = isolation.build_backend_argv(["git", "status"], workspace,
                                            image="python:3.12-slim",
                                            container_name=name)
        assert "--name" in argv
        assert argv[argv.index("--name") + 1] == name
        assert "--stop-timeout" in argv
        assert argv[argv.index("--stop-timeout") + 1] == str(isolation.STOP_TIMEOUT_SECONDS)
        # Nomes exclusivos por execução
        assert isolation.new_container_name() != name

    def test_invalid_container_name_rejected(self, monkeypatch, workspace):
        from core.v2 import isolation
        _simulate_backend(monkeypatch)
        assert isolation.build_backend_argv(["true"], workspace,
                                            image="python:3.12-slim",
                                            container_name="--privileged") is None

    def test_cpu_limit_matrix(self):
        from core.v2.isolation import cpu_limit_is_applied as ok
        assert ok(1_000_000_000, 0, 0, "1.0") is True      # --cpus 1.0
        assert ok(0, 100000, 100000, "1.0") is True         # quota/period equivalente
        assert ok(0, 0, 0, "1.0") is False                  # nada definido
        assert ok(0, 50000, 100000, "1.0") is False         # meio CPU ≠ 1 CPU
        assert ok(2_000_000_000, 0, 0, "1.0") is False      # 2 CPUs ≠ 1 CPU
        assert ok("", "", "", "1.0") is False

    @pytest.mark.asyncio
    async def test_timeout_terminates_and_confirms(self, monkeypatch, workspace):
        from core.v2 import isolation
        _simulate_backend(monkeypatch)
        calls = []
        monkeypatch.setattr(isolation, "_docker_cli",
                            _fake_docker_cli(calls, inspect_out="false"))
        _run_raises_timeout(monkeypatch)

        outcome = await isolation.run_isolated(["python", "-c", "pass"], workspace,
                                               timeout=1)

        assert outcome["timeout"] is True
        assert outcome["terminated"] is True
        assert outcome["cleanup_unconfirmed"] is False
        assert [c[0] for c in calls] == ["kill", "rm", "inspect"]
        # Todas as operações de limpeza referem o contentor com o nosso nome.
        assert all(outcome["container"] in c for c in calls)
        assert "terminado e removido" in outcome["error"]
        assert "processo terminado)" not in outcome["error"]

    @pytest.mark.asyncio
    async def test_timeout_with_kill_failure_is_honest(self, monkeypatch, workspace):
        from core.v2 import isolation
        _simulate_backend(monkeypatch)
        calls = []
        monkeypatch.setattr(isolation, "_docker_cli",
                            _fake_docker_cli(calls, kill_rc=1, kill_err="daemon error"))
        _run_raises_timeout(monkeypatch)

        outcome = await isolation.run_isolated(["python", "-c", "pass"], workspace,
                                               timeout=1)

        assert outcome["terminated"] is False
        assert outcome["cleanup_unconfirmed"] is True
        assert "NAO foi possível confirmar" in outcome["error"]

    @pytest.mark.asyncio
    async def test_timeout_with_engine_unreachable(self, monkeypatch, workspace):
        from core.v2 import isolation
        _simulate_backend(monkeypatch)
        calls = []
        monkeypatch.setattr(isolation, "_docker_cli",
                            _fake_docker_cli(calls, raises=PermissionError("pipe negado")))
        _run_raises_timeout(monkeypatch)

        outcome = await isolation.run_isolated(["python", "-c", "pass"], workspace,
                                               timeout=1)

        assert outcome["terminated"] is False
        assert "indisponível" in outcome["termination_detail"]
        assert "NAO foi possível confirmar" in outcome["error"]

    @pytest.mark.asyncio
    async def test_success_attempts_cleanup(self, monkeypatch, workspace):
        from core.v2 import isolation
        _simulate_backend(monkeypatch)
        calls = []
        monkeypatch.setattr(isolation, "_docker_cli",
                            _fake_docker_cli(calls, inspect_out="false"))
        monkeypatch.setattr(isolation.subprocess, "run",
                            _fake_run(returncode=0, stdout="ok", recorder=calls))

        outcome = await isolation.run_isolated(["git", "status"], workspace, timeout=5)

        assert outcome["ok"] is True
        assert outcome["terminated"] is True
        # Primeira chamada = docker run; seguintes = limpeza com confirmação
        assert calls[0][1] == "run"
        assert [c[0] for c in calls[1:]] == ["kill", "rm", "inspect"]

    @pytest.mark.asyncio
    async def test_error_path_still_cleans_up(self, monkeypatch, workspace):
        from core.v2 import isolation
        _simulate_backend(monkeypatch)
        calls = []
        monkeypatch.setattr(isolation, "_docker_cli",
                            _fake_docker_cli(calls, inspect_out="false"))
        monkeypatch.setattr(isolation.subprocess, "run",
                            _fake_run(raises=OSError("docker rebentou")))

        outcome = await isolation.run_isolated(["git", "status"], workspace, timeout=5)

        assert outcome["ok"] is False
        assert outcome["spawn_failed"] is True
        assert outcome["terminated"] is True      # limpeza confirmada
        assert [c[0] for c in calls] == ["kill", "rm", "inspect"]


# ======================================================================
#  Validação da fronteira — exige backend real
# ======================================================================

class TestIsolationBoundary:
    """Testes exigidos por Quarantine/ISOLATION-REQUIREMENTS.md.

    Sem backend, não correm: demonstram a fronteira apenas quando ela existe.
    Os motivos do skip indicam exactamente o que falta.
    """

    @pytest.mark.asyncio
    @requires_backend
    async def test_timeout_really_terminates_the_workload(self, workspace):
        """Integração real: após o timeout, o contentor não existe nem corre."""
        from core.v2 import isolation

        outcome = await isolation.run_isolated(
            ["python", "-c", "import time; time.sleep(60)"], workspace, timeout=5
        )
        assert outcome["timeout"] is True
        assert outcome["terminated"] is True, outcome
        assert outcome["cleanup_unconfirmed"] is False
        ps = isolation.subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"name=^{outcome['container']}$"],
            capture_output=True, text=True,
        )
        assert ps.stdout.strip() == "", ps.stdout

    @pytest.mark.asyncio
    @requires_backend
    async def test_preexisting_container_is_preserved(self, workspace):
        """Um contentor do utilizador com a mesma imagem não é tocado."""
        from core.v2 import isolation

        canary = isolation.new_container_name(prefix="odysseus-canary")
        created = isolation.subprocess.run(
            ["docker", "create", "--name", canary, "python:3.12-slim",
             "python", "-c", "import time; time.sleep(300)"],
            capture_output=True, text=True,
        )
        assert created.returncode == 0, created.stderr
        try:
            outcome = await isolation.run_isolated(
                ["python", "-c", "pass"], workspace, timeout=30
            )
            assert outcome["container"] != canary
            still = isolation.subprocess.run(
                ["docker", "inspect", "--format", "{{.Id}}", canary],
                capture_output=True, text=True,
            )
            assert still.returncode == 0, "o contentor preexistente foi afectado"
        finally:
            isolation.subprocess.run(["docker", "rm", "-f", canary],
                                     capture_output=True, text=True)

    @pytest.mark.asyncio
    @requires_backend
    async def test_cpu_limit_effective_inside_container(self, workspace):
        """O limite de CPU existe e corresponde a 1 CPU, na representação real."""
        from core.v2 import isolation

        name = isolation.new_container_name()
        argv = isolation.build_backend_argv(["true"], workspace,
                                            image=os.getenv("ODYSSEUS_V2_ISOLATION_IMAGE"),
                                            container_name=name)
        created = isolation.subprocess.run(
            ["docker", "create", "--name", name, *[a for a in argv[2:] if a != "--rm"]],
            capture_output=True, text=True,
        )
        try:
            assert created.returncode == 0, created.stderr
            fmt = "{{.HostConfig.NanoCpus}}|{{.HostConfig.CpuQuota}}|{{.HostConfig.CpuPeriod}}"
            insp = isolation.subprocess.run(["docker", "inspect", "--format", fmt, name],
                                            capture_output=True, text=True)
            nano, quota, period = (insp.stdout.strip().split("|") + ["", "", ""])[:3]
            assert isolation.cpu_limit_is_applied(
                nano, quota, period, isolation.RESOURCE_LIMITS["cpus"]
            ), f"NanoCpus={nano} quota={quota} period={period}"
        finally:
            isolation.subprocess.run(["docker", "rm", "-f", name],
                                     capture_output=True, text=True)

    @pytest.mark.asyncio
    @requires_backend
    async def test_only_authorized_root_is_mounted(self, workspace):
        """A única montagem do host é a raiz autorizada."""
        from core.v2.isolation import build_backend_argv

        argv = build_backend_argv(["python", "-c", "pass"], workspace)
        assert argv is not None
        volumes = [argv[i + 1] for i, a in enumerate(argv) if a == "--volume"]
        assert len(volumes) == 1, volumes
        assert volumes[0].startswith(str(workspace))
        assert argv[argv.index("--network") + 1] == "none"

    @pytest.mark.skipif(sys.platform == "win32", reason="Isolamento real via Docker não é suportado nativamente no Windows")
    @pytest.mark.asyncio
    @requires_backend
    async def test_protected_paths_unreachable_inside_backend(self, workspace):
        """Dentro do backend, as áreas protegidas do host não existem."""
        from core.v2.isolation import run_isolated

        probe = (
            "import os\n"
            "alvos = ['Obsidian Brain Vault', 'Odysseus_Mind_OS', 'data', 'logs', '.env', '.git']\n"
            "presentes = [a for a in alvos if os.path.exists(os.path.join('/work', a))]\n"
            "raise SystemExit(0 if not presentes else 1)\n"
        )
        outcome = await run_isolated(["python", "-c", probe], workspace, timeout=120)
        assert outcome["isolated"] is True
        assert outcome["ok"] is True, outcome

    @pytest.mark.asyncio
    @requires_backend
    async def test_network_unreachable_inside_backend(self, workspace):
        """Com `--network none`, uma ligação de rede falha."""
        from core.v2.isolation import run_isolated

        probe = (
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), 4)\n"
            "except OSError:\n"
            "    raise SystemExit(0)\n"
            "raise SystemExit(1)\n"
        )
        outcome = await run_isolated(["python", "-c", probe], workspace, timeout=120)
        assert outcome["isolated"] is True
        assert outcome["ok"] is True, outcome
