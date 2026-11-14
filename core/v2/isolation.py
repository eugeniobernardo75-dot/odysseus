"""
Isolamento real — abstracção e adaptador do backend (Etapa 6)

Princípio: **uma pasta, um `cwd` ou um `ContextVar` não são isolamento.**
Isolamento exige que o sistema operativo imponha limites a ficheiros, processos,
recursos e rede.

Este módulo:

    - descreve o estado real do backend (`isolation_status`);
    - constrói o comando de um backend de contentor quando ele existir
      (`build_backend_argv`), sem o executar aqui;
    - permite exigir isolamento (`ODYSSEUS_V2_REQUIRE_ISOLATION=1`), caso em que a
      execução é **recusada** enquanto não houver backend — em vez de correr sem
      protecção e o chamar de isolado.

Enquanto nenhum backend estiver disponível, as ferramentas de execução declaram
`isolated: false` em todos os resultados. Nenhuma comunicação deve apresentar
essa execução como isolada.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .execution_policy import isolation_backends

logger = logging.getLogger(__name__)

# Exigir isolamento: sem backend, a execução é recusada (fail-closed).
REQUIRE_ISOLATION_ENV = "ODYSSEUS_V2_REQUIRE_ISOLATION"

# Imagem usada pelo backend de contentor. Sem valor definido, não se inventa
# uma imagem: o backend fica indisponível para execução.
ISOLATION_IMAGE_ENV = "ODYSSEUS_V2_ISOLATION_IMAGE"

# Ponto de montagem dentro do contentor.
CONTAINER_WORKDIR = "/work"

# Utilizador dentro do contentor (opcional). Sem valor, o contentor usa o seu
# utilizador por omissão — mas com todas as capacidades removidas e sem
# escalada de privilégios. Definir, por exemplo, `65534:65534` é mais restrito
# e pode impedir a escrita no workspace montado.
ISOLATION_USER_ENV = "ODYSSEUS_V2_ISOLATION_USER"
CONTAINER_USER_RE = re.compile(r"^[0-9]{1,10}(:[0-9]{1,10})?$")

# Limites de recursos do contentor.
RESOURCE_LIMITS: dict[str, str] = {
    "pids": "256",
    "memory": "1g",
    "cpus": "1.0",
}

# Graça concedida ao contentor para terminar quando é pedido o kill.
STOP_TIMEOUT_SECONDS = 2

# Referências de imagem aceites: nome[:tag][@sha256:digest]. Repudia qualquer
# coisa que comece por `-` (injecção de flags do `docker run`) ou com espaços.
_IMAGE_REF_RE = re.compile(
    r"^(?!-)[A-Za-z0-9][A-Za-z0-9._/-]*"
    r"(?::[A-Za-z0-9._-]{1,128})?"
    r"(?:@sha256:[0-9a-f]{64})?$"
)


def is_valid_image_reference(image: str) -> bool:
    """True se a referência de imagem for aceitável para o `docker run`.

    Impede injecção de flags (uma imagem começada por `-` seria interpretada
    como opção do Docker) e aceita fixação por digest (`@sha256:…`).
    """
    if not isinstance(image, str):
        return False
    candidate = image.strip()
    if not candidate or candidate != image:
        return False
    return bool(_IMAGE_REF_RE.match(candidate))


def hardening_summary() -> list[str]:
    """Controlos aplicados ao contentor, para inspecção e relatório."""
    controls = [
        "network:none",
        "cap-drop:ALL",
        "no-new-privileges",
        f"pids-limit:{RESOURCE_LIMITS['pids']}",
        f"memory:{RESOURCE_LIMITS['memory']}",
        f"cpus:{RESOURCE_LIMITS['cpus']}",
        "read-only-rootfs",
        "tmpfs:/tmp",
        "single-host-mount",
    ]
    if (os.getenv(ISOLATION_USER_ENV) or "").strip():
        controls.append(f"user:{os.getenv(ISOLATION_USER_ENV).strip()}")
    return controls


@dataclass(frozen=True)
class IsolationStatus:
    """Estado do isolamento, sem optimismo."""
    backend: str                  # "docker" | "none"
    available: bool               # backend capaz de impor limites?
    image: Optional[str] = None
    network_disabled: bool = True
    detail: str = ""
    probe: dict = field(default_factory=dict)

    @property
    def isolated(self) -> bool:
        """True apenas quando existe backend disponível *e* imagem definida."""
        return bool(self.available and self.image)


def isolation_status() -> IsolationStatus:
    """Estado real do isolamento no ambiente actual."""
    probe = isolation_backends()
    image = (os.getenv(ISOLATION_IMAGE_ENV) or "").strip() or None

    if probe.get("docker_running"):
        backend = "docker"
        available = True
    else:
        backend = "none"
        available = False

    if available and image is None:
        detail = (
            "Docker em execução mas sem imagem definida "
            f"({ISOLATION_IMAGE_ENV}); a execução isolada está indisponível."
        )
    elif available:
        detail = f"Backend Docker disponível (imagem {image})."
    else:
        detail = probe.get("verdict", "sem backend de isolamento")

    return IsolationStatus(
        backend=backend,
        available=available,
        image=image,
        network_disabled=True,
        detail=detail,
        probe=probe,
    )


def isolation_required() -> bool:
    """True quando o servidor exige isolamento para executar."""
    return os.getenv(REQUIRE_ISOLATION_ENV, "0").strip().lower() in ("1", "true", "yes")


def isolation_refusal_reason() -> Optional[str]:
    """Motivo pelo qual a execução tem de ser recusada, ou None se pode correr.

    Devolve None apenas quando o isolamento não é exigido (a execução corre com
    `isolated: false`, declarado no resultado).
    """
    if not isolation_required():
        return None
    status = isolation_status()
    if status.isolated:
        return None
    return (
        f"Isolamento exigido ({REQUIRE_ISOLATION_ENV}=1) mas indisponível: "
        f"{status.detail}. A execução não foi realizada."
    )


def new_container_name(prefix: str = "odysseus-iso") -> str:
    """Nome exclusivo por execução, para terminar e limpar o contentor certo.

    Sem nome controlado pela aplicação não é possível distinguir o contentor
    criado pelo Odysseus de um contentor preexistente do utilizador.
    """
    import uuid
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def cpu_limit_is_applied(nanocpus: Any, quota: Any, period: Any,
                         expected_cpus: str = "1.0") -> bool:
    """True se o limite de CPU pedido estiver efectivamente representado.

    O Docker representa ``--cpus`` em ``HostConfig.NanoCpus``; ``CpuQuota`` e
    ``CpuPeriod`` só são preenchidos por ``--cpu-quota``/``--cpu-period`` e a
    conversão para a quota do cgroup acontece no arranque do contentor. Aceita
    qualquer das representações **desde que corresponda ao valor pedido** — não
    basta estar preenchida.
    """
    def _num(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    try:
        wanted = float(expected_cpus)
    except (TypeError, ValueError):
        return False
    if wanted <= 0:
        return False

    nano = _num(nanocpus)
    if nano > 0:
        return abs(nano / 1_000_000_000 - wanted) < 1e-6

    q, p = _num(quota), _num(period)
    if q > 0 and p > 0:
        return abs(q / p - wanted) < 1e-6
    return False


async def _docker_cli(args: list[str], timeout: float = 15.0):
    """Executa um comando do CLI do Docker (sem shell)."""
    import asyncio

    def _run():
        return subprocess.run(["docker", *args], capture_output=True, text=True,
                              timeout=timeout)

    return await asyncio.to_thread(_run)


async def terminate_container(name: str) -> tuple[bool, str]:
    """Termina e remove o contentor *name*, confirmando o resultado.

    Devolve ``(terminado, detalhe)``. Idempotente: um contentor já inexistente
    conta como terminado. Se o motor estiver inacessível, devolve False com o
    motivo — nunca afirma terminação sem confirmação.
    """
    if not name:
        return False, "sem nome de contentor"

    steps: list[str] = []
    for args in (["kill", name], ["rm", "-f", name]):
        try:
            result = await _docker_cli(args)
        except Exception as e:                      # motor inacessível
            return False, f"{args[0]} indisponível: {type(e).__name__}: {e}"
        stderr = (result.stderr or "").strip()
        if result.returncode != 0 and "no such container" not in stderr.lower():
            return False, f"{args[0]} falhou: {stderr[:80]}"
        steps.append(f"{args[0]}=ok")

    # Confirmação explícita do estado.
    try:
        check = await _docker_cli(["inspect", "--format", "{{.State.Running}}", name])
    except Exception as e:
        return False, f"confirmação indisponível: {type(e).__name__}: {e}"
    if check.returncode != 0:
        return True, "contentor ausente após limpeza"
    running = (check.stdout or "").strip()
    if running == "false":
        return True, "contentor parado e removido"
    return False, f"contentor ainda em execução (Running={running})"


def build_backend_argv(argv: list[str], workspace: Path,
                       image: Optional[str] = None,
                       container_name: Optional[str] = None) -> Optional[list[str]]:
    """Constrói o comando que corre *argv* dentro do backend de isolamento.

    Regras aplicadas (quando o backend existir):

        - ``--network none``: sem rede;
        - ``--rm``: sem contentores órfãos;
        - ``--name <exclusivo>``: identificação inequívoca para terminar/limpar;
        - ``--stop-timeout``: limite de graça ao terminar o workload;
        - a raiz autorizada é o **único** ponto montado, em ``/work``;
        - ``--workdir /work`` fixa o directório de trabalho;
        - o processo corre sem privilégios adicionais.

    Devolve None quando não há backend/imagem ou quando o pedido é inválido —
    nunca devolve um comando que dê a impressão de estar isolado sem o estar.
    """
    if not argv:
        return None
    image = (image or os.getenv(ISOLATION_IMAGE_ENV) or "").strip()
    if not image:
        return None
    if not is_valid_image_reference(image):
        logger.error("Referência de imagem recusada: %r", image[:80])
        return None
    status = isolation_status()
    if not status.available:
        return None

    try:
        mount = Path(workspace).resolve()
    except (OSError, ValueError, RuntimeError):
        return None
    if not mount.is_dir():
        return None

    # Endurecimento: sem capacidades, sem escalada de privilégios, com limites de
    # recursos, sistema de ficheiros do contentor só de leitura (o workspace
    # montado e /tmp continuam utilizáveis).
    argv_out = [
        "docker", "run", "--rm",
        "--network", "none",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", str(RESOURCE_LIMITS["pids"]),
        "--memory", RESOURCE_LIMITS["memory"],
        "--cpus", RESOURCE_LIMITS["cpus"],
        "--stop-timeout", str(STOP_TIMEOUT_SECONDS),
        "--read-only",
        "--tmpfs", "/tmp:rw,nosuid,size=64m",
    ]
    if container_name:
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$", container_name):
            logger.error("Nome de contentor recusado: %r", container_name[:40])
            return None
        argv_out += ["--name", container_name]
    container_user = (os.getenv(ISOLATION_USER_ENV) or "").strip()
    if container_user:
        if not CONTAINER_USER_RE.match(container_user):
            logger.error("Utilizador de contentor recusado: %r", container_user[:40])
            return None
        argv_out += ["--user", container_user]
    argv_out += [
        "--workdir", CONTAINER_WORKDIR,
        # `--volume` é a única montagem do host.
        "--volume", f"{mount}:{CONTAINER_WORKDIR}",
        image,
        *argv,
    ]
    return argv_out


def describe_backend_argv(argv: list[str], workspace: Path,
                          image: Optional[str] = None) -> str:
    """Representação legível do comando isolado (para registo, sem segredos)."""
    built = build_backend_argv(argv, workspace, image=image)
    if built is None:
        return "<isolamento indisponível>"
    return " ".join(shlex.quote(part) for part in built)


async def run_isolated(argv: list[str], workspace: Path, timeout: float = 30.0,
                       capture: bool = True,
                       container_name: Optional[str] = None) -> dict:
    """Executa *argv* dentro do backend de isolamento.

    Devolve um dicionário com ``ok``, ``return_code``, ``stdout``, ``stderr``,
    ``isolated``, ``argv``, ``container``, ``terminated`` e
    ``cleanup_unconfirmed``. Quando não há backend, devolve ``ok=False`` e
    ``isolated=False`` — nunca executa fora do backend fingindo estar isolado.

    O contentor tem nome exclusivo e é terminado/removido com confirmação em
    todos os desfechos. `terminated=False` significa que a terminação **não foi
    confirmada** (por exemplo, motor inacessível) — e o resultado di-lo.
    """
    import asyncio

    status = isolation_status()
    if not status.isolated:
        return {
            "ok": False, "return_code": None, "stdout": None, "stderr": None,
            "isolated": False, "argv": None,
            "error": f"isolamento indisponível: {status.detail}",
        }

    name = container_name or new_container_name()
    backend_argv = build_backend_argv(argv, workspace, image=status.image,
                                      container_name=name)
    if backend_argv is None:
        return {
            "ok": False, "return_code": None, "stdout": None, "stderr": None,
            "isolated": False, "argv": None, "container": name,
            "error": "não foi possível construir o comando do backend",
        }

    # `subprocess.run` num thread, em vez do transporte asyncio: além de
    # capturar output onde o ambiente o permite (o transporte asyncio deste
    # sandbox recusa `text=True` com stdio não-pipe), o timeout é imposto pelo
    # próprio subprocesso.
    #
    # IMPORTANTE: matar o processo cliente NÃO termina o contentor. Por isso, em
    # todos os desfechos, o contentor com o nosso nome é terminado e a
    # terminação é confirmada — ou a incerteza é comunicada explicitamente.
    def _run() -> subprocess.CompletedProcess:
        return subprocess.run(
            backend_argv,
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.PIPE if capture else subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )

    timeout_hit = False
    completed = None
    failure: Optional[str] = None
    spawn_failed = False
    try:
        completed = await asyncio.to_thread(_run)
    except subprocess.TimeoutExpired:
        timeout_hit = True
    except PermissionError as e:
        # Ambiente que proíbe pipes: repetir sem captura, preservando o rc.
        if capture:
            outcome = await run_isolated(argv, workspace, timeout, capture=False,
                                         container_name=name)
            outcome["argv"] = backend_argv
            return outcome
        failure = f"não foi possível iniciar o backend: {e}"
        spawn_failed = True
    except Exception as e:
        failure = f"não foi possível iniciar o backend: {e}"
        spawn_failed = True

    # ── Limpeza com confirmação, em sucesso, erro e timeout ──
    terminated, detail = await terminate_container(name)

    if timeout_hit:
        return {
            "ok": False, "return_code": None, "stdout": None, "stderr": None,
            "isolated": True, "argv": backend_argv, "container": name,
            "timeout": True,
            "terminated": terminated,
            "cleanup_unconfirmed": not terminated,
            "termination_detail": detail,
            "error": (
                f"backend excedeu o timeout de {timeout}s; contentor '{name}' "
                + ("terminado e removido" if terminated
                   else f"NAO foi possível confirmar a terminação ({detail})")
            ),
        }

    if failure is not None:
        return {
            "ok": False, "return_code": None, "stdout": None, "stderr": None,
            "isolated": True, "argv": backend_argv, "container": name,
            "spawn_failed": spawn_failed,
            "terminated": terminated,
            "cleanup_unconfirmed": not terminated,
            "termination_detail": detail,
            "error": failure,
        }

    return {
        "ok": completed.returncode == 0,
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "isolated": True,
        "argv": backend_argv,
        "container": name,
        "terminated": terminated,
        "cleanup_unconfirmed": not terminated,
        "termination_detail": detail,
        "error": None,
    }


def isolation_flags() -> dict:
    """Campos a incluir em cada resultado de execução, para não induzir em erro."""
    status = isolation_status()
    return {
        "isolated": status.isolated,
        "isolation_backend": status.backend,
        "isolation_detail": status.detail,
    }
