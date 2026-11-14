"""
Política de execução — Odysseus V2
==================================

Concentra as decisões de segurança que autorizam (ou negam) execução real de
ferramentas:

    - vet_workspace      : valida o único espaço de trabalho autorizado;
    - check_command      : política explícita de comandos permitidos/negados;
    - authorized_execution / is_authorized_for
                         : marcador de execução autorizada, activado apenas
                           pelo ToolManager depois de a aprovação passar;
    - operation_fingerprint
                         : liga uma aprovação à operação concreta que a pediu;
    - redact_params      : retira segredos antes de qualquer registo.

Princípios:
    - negar por omissão: o que não estiver explicitamente permitido é negado;
    - nenhuma lista de palavras é tratada como fronteira de segurança — a
      fronteira é a aprovação humana mais o confinamento do workspace;
    - a análise textual é defesa em profundidade, não garantia.

Nota sobre importações: os helpers de contenção de caminhos vivem em
`core/v2/tools.py` (onde já estavam, e de onde os testes os importam). Para
evitar um ciclo de importação no arranque, este módulo importa-os de forma
preguiçosa, dentro das funções que os usam.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


# ======================================================================
#  Helpers de contenção (fonte única em core/v2/tools.py)
# ======================================================================
# A raiz do projecto é a mesma expressão usada em `tools.py`. Não se importa de
# lá para não criar ciclo no arranque (tools importa este módulo).

V2_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def _is_protected_path(path: Path) -> bool:
    from .tools import _is_protected_path as _impl
    return _impl(path)


def _is_within(path: Path, root: Path) -> bool:
    from .tools import _is_within as _impl
    return _impl(path, root)


def _resolve_confined(path: str, base: Path) -> Optional[Path]:
    from .tools import _resolve_confined as _impl
    return _impl(path, base)


# ======================================================================
#  Workspace autorizado
# ======================================================================

def vet_workspace(raw: Any) -> Optional[Path]:
    """Valida e devolve o workspace autorizado, ou None se for inaceitável.

    Regras (equivalentes conceptuais ao vet_workspace da V1, sem reutilizar a
    implementação, que depende de constantes e de um contextvar da V1):

        - caminho obrigatório;
        - tem de existir e ser um directório;
        - a raiz de um filesystem é recusada (tornaria tudo "interior");
        - áreas protegidas (Brain Vault, Mind OS, data/, logs/, .git) são
          recusadas;
        - devolve o caminho real, com symlinks resolvidos.
    """
    if raw is None or not str(raw).strip():
        return None
    try:
        expanded = os.path.expanduser(str(raw).strip())
        resolved = Path(os.path.realpath(expanded))
    except (OSError, ValueError, RuntimeError):
        return None

    if not resolved.is_dir():
        return None
    # A raiz é o próprio dirname (cobre "/", "C:\\" e "\\\\server\\share").
    if os.path.dirname(str(resolved)) == str(resolved):
        return None
    if _is_protected_path(resolved):
        return None
    return resolved


# ======================================================================
#  Política de comandos
# ======================================================================
# O comando NÃO é entregue a uma shell: é tokenizado aqui e executado com
# create_subprocess_exec. Sem shell não há encadeamento, pipes, redireccio-
# namentos nem substituição de comandos — a classe de ataque mais perigosa
# desaparece por construção, e a validação de argumentos passa a ser possível.

# Conjunto de caracteres aceites. Rejeita deliberadamente metacaracteres de
# shell, aspas, crases, '$', '%', '&', ';', '|', '>', '<', '(', ')', '{', '}',
# '*', '?', '~', '!', '\\' e quebras de linha.
_SAFE_COMMAND_CHARS = re.compile(r"^[A-Za-z0-9 _\-./=:,@+]+$")

_MAX_COMMAND_LENGTH = 2000

# Executáveis permitidos. Tudo o resto é negado por omissão.
#
# A lista é consciente da plataforma porque os nomes não são portáveis:
# `echo`, `ls`, `cat`, `pwd` e `which` são builtins de shell em Windows (não
# existem como executáveis), enquanto `where`, `findstr`, `more` e `tree` só
# existem em Windows. Sem esta distinção, um comando "permitido" falhava com
# WinError 2 em vez de executar — foi o que a validação real mostrou.
#
# Nota: esta lista é sobre *ferramentas de inspecção*. `python`/`pytest` correm
# código e só são aceites sob aprovação, com as restrições abaixo.

_ALLOWED_COMMON: frozenset[str] = frozenset({
    # controlo de versões (apenas leitura — ver _GIT_ALLOWED_SUBCOMMANDS)
    "git",
})

# Interpretadores: NÃO fazem parte da allow-list por omissão. Só são aceites
# quando o *servidor* marca explicitamente a tarefa como autorizada a correr
# código (`allow_interpreters` no contexto, construído pelo servidor) e, mesmo
# assim, a operação continua a exigir aprovação humana ligada ao fingerprint.
_INTERPRETERS: frozenset[str] = frozenset({"python", "python3", "pytest"})

# Ferramentas POSIX de inspecção.
_ALLOWED_POSIX: frozenset[str] = frozenset({
    "ls", "cat", "head", "tail", "grep", "wc", "sort", "uniq", "diff",
    "which", "whoami", "date", "pwd", "echo", "find",
})

# Equivalentes reais em Windows (executáveis, não builtins de shell).
_ALLOWED_WINDOWS: frozenset[str] = frozenset({
    "where", "findstr", "sort", "more", "fc", "tree",
})


def allowed_executables(platform: Optional[str] = None,
                        allow_interpreters: bool = False) -> frozenset[str]:
    """Executáveis permitidos para a plataforma indicada (por omissão, a actual).

    `allow_interpreters` só deve vir de configuração controlada pelo servidor:
    um interpretador executa código arbitrário, pelo que não pertence à lista
    por omissão. Mesmo autorizado, continua a exigir aprovação humana.
    """
    name = (platform or os.name)
    base = _ALLOWED_COMMON | (_ALLOWED_WINDOWS if name == "nt" else _ALLOWED_POSIX)
    return base | _INTERPRETERS if allow_interpreters else base

# `git` só pode ser usado para leitura do repositório.
_GIT_ALLOWED_SUBCOMMANDS: frozenset[str] = frozenset({
    "status", "log", "diff", "show", "branch", "tags", "shortlog",
    "remotes", "describe", "rev-parse", "ls-files", "blame", "whatchanged",
})

# `python` só pode ser usado para versão ou para correr pytest/scripts dentro
# do workspace. `-c` (código inline) é negado: para isso existe o PythonTool,
# que exige aprovação e tem política própria.
_PYTHON_ALLOWED_FLAGS: frozenset[str] = frozenset({"--version", "-V"})

# Tokens que permitem execução indirecta por outras vias.
_FORBIDDEN_TOKENS: frozenset[str] = frozenset({
    "-exec", "-execdir", "-delete", "-ok", "-okdir", "--exec",
    "-c", "--command", "-i", "--interactive",
})


def _binary_name(token: str) -> str:
    """Nome do executável, sem directoria e sem extensão .exe."""
    name = os.path.basename(token.replace("/", os.sep)).casefold()
    if name.endswith(".exe"):
        name = name[:-4]
    return name


def _resolve_argument(token: str, workspace: Path) -> Optional[Path]:
    """Resolve um token que aparenta ser caminho, confinando-o ao workspace."""
    from .tools import _resolve_confined as _impl
    return _impl(token, workspace)


def check_command(command: str, workspace: Path,
                  platform: Optional[str] = None,
                  allow_interpreters: bool = False) -> dict:
    """Aplica a política de comandos.

    Devolve ``{"allowed": bool, "reason": str, "argv": list[str] | None}``.
    Negar é sempre o resultado por omissão.

    `platform` existe para os testes poderem verificar ambas as listas de
    executáveis de forma determinística; em produção usa a plataforma actual.
    `allow_interpreters` autoriza `python`/`python3`/`pytest` e só deve ser
    activado por configuração do servidor para a tarefa concreta.
    """
    allowed_bins = allowed_executables(platform, allow_interpreters)
    if not isinstance(command, str) or not command.strip():
        return {"allowed": False, "reason": "Comando vazio.", "argv": None}

    raw = command.strip()
    if len(raw) > _MAX_COMMAND_LENGTH:
        return {
            "allowed": False,
            "reason": f"Comando demasiado longo ({len(raw)} > {_MAX_COMMAND_LENGTH}).",
            "argv": None,
        }

    if not _SAFE_COMMAND_CHARS.match(raw):
        offenders = sorted({c for c in raw if not re.match(r"[A-Za-z0-9 _\-./=:,@+]", c)})
        return {
            "allowed": False,
            "reason": (
                "Comando contém caracteres não permitidos "
                f"({', '.join(repr(c) for c in offenders[:8])}); "
                "encadeamento, pipes, redireccionamentos e substituição não são aceites."
            ),
            "argv": None,
        }

    argv = raw.split()
    if not argv:
        return {"allowed": False, "reason": "Comando vazio.", "argv": None}

    binary = _binary_name(argv[0])
    if binary not in allowed_bins:
        return {
            "allowed": False,
            "reason": (
                f"Executável '{argv[0]}' não está na lista de permitidos para "
                f"esta plataforma. Permitidos: {', '.join(sorted(allowed_bins))}."
            ),
            "argv": None,
        }

    lowered = [t.casefold() for t in argv[1:]]
    for token in _FORBIDDEN_TOKENS:
        if token in lowered:
            return {
                "allowed": False,
                "reason": f"Argumento '{token}' não é permitido (execução indirecta).",
                "argv": None,
            }

    # ── Regras específicas por executável ──
    if binary == "git":
        sub = next((t.casefold() for t in argv[1:] if not t.startswith("-")), "")
        if not sub:
            return {"allowed": False, "reason": "Falta o subcomando de git.", "argv": None}
        if sub not in _GIT_ALLOWED_SUBCOMMANDS:
            return {
                "allowed": False,
                "reason": (
                    f"Subcomando git '{sub}' não é de leitura. "
                    f"Permitidos: {', '.join(sorted(_GIT_ALLOWED_SUBCOMMANDS))}."
                ),
                "argv": None,
            }

    if binary in ("python", "python3"):
        args = [t.casefold() for t in argv[1:]]
        if not args:
            return {"allowed": False, "reason": "python requer um argumento.", "argv": None}
        if args[0] in _PYTHON_ALLOWED_FLAGS:
            pass  # apenas reportar versão
        elif args[0] == "-m" and len(args) > 1 and args[1] == "pytest":
            pass  # pytest é permitido; corre código, mas sob aprovação
        elif args[0].endswith(".py"):
            resolved = _resolve_argument(argv[1], workspace)
            if resolved is None:
                return {
                    "allowed": False,
                    "reason": f"Script '{argv[1]}' está fora do workspace autorizado.",
                    "argv": None,
                }
        else:
            return {
                "allowed": False,
                "reason": (
                    "python só é permitido como '--version'/'-V', '-m pytest' ou "
                    "um script .py dentro do workspace (para código inline usa o "
                    "PythonTool, que exige aprovação)."
                ),
                "argv": None,
            }

    # ── Argumentos que aparentam ser caminhos têm de ficar no workspace ──
    for token in argv[1:]:
        if token.startswith("-"):
            continue
        looks_like_path = (
            token.startswith(("/", "./", "../", "~"))
            or "/" in token
            or token in (".", "..")
        )
        if not looks_like_path:
            continue
        resolved = _resolve_argument(token, workspace)
        if resolved is None:
            return {
                "allowed": False,
                "reason": (
                    f"Argumento '{token}' aponta para fora do workspace autorizado "
                    f"ou para uma área protegida."
                ),
                "argv": None,
            }

    return {"allowed": True, "reason": "", "argv": argv}


# ======================================================================
#  Marcador de execução autorizada
# ======================================================================
# O ToolManager é o único componente que activa este marcador, e fá-lo apenas
# depois de a aprovação ter sido concedida. As ferramentas que exigem
# aprovação recusam-se a executar sem ele, pelo que uma chamada directa a
# `ShellTool().execute(...)` não contorna a política.
#
# Limitação declarada: sendo um contextvar de processo, código Python que corra
# dentro do processo do servidor pode, em teoria, activar o marcador. Não é uma
# fronteira contra código arbitrário no processo — é a garantia de que a
# *ferramenta* não executa sem passar pelo gate.

_authorized: contextvars.ContextVar = contextvars.ContextVar(
    "odysseus_v2_authorized_execution", default=None
)


@contextmanager
def authorized_execution(tool_name: str, execution_id: str,
                         fingerprint: str = "") -> Iterator[None]:
    """Marca a execução de *tool_name* como autorizada (uso exclusivo do ToolManager)."""
    token = _authorized.set({
        "tool": tool_name,
        "execution_id": execution_id,
        "fingerprint": fingerprint,
    })
    try:
        yield
    finally:
        _authorized.reset(token)


def current_authorization() -> Optional[dict]:
    """Autorização activa neste contexto, se existir."""
    return _authorized.get()


def is_authorized_for(tool_name: str) -> bool:
    """True apenas se existir autorização activa para esta ferramenta."""
    current = _authorized.get()
    return bool(
        current
        and current.get("tool") == tool_name
        and current.get("execution_id")
    )


def authorization_denied_message(tool_name: str) -> str:
    return (
        f"Execução de '{tool_name}' não autorizada: a ferramenta só corre "
        f"através do ToolManager, depois de aprovação explícita."
    )


# ======================================================================
#  Ligação da aprovação à operação
# ======================================================================

def operation_fingerprint(action_type: str, details: Optional[dict] = None,
                          permissions: Optional[list] = None) -> str:
    """Fingerprint determinístico da operação autorizada.

    Liga uma aprovação ao pedido concreto: ferramenta, parâmetros e
    permissões. Sem isto, uma aprovação concedida para um comando podia ser
    reapresentada para executar outro.
    """
    payload = json.dumps(
        {
            "action": str(action_type),
            "details": details or {},
            "permissions": list(permissions or []),
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ======================================================================
#  Redacção para registo
# ======================================================================

_SECRET_KEY_HINTS = (
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "authorization", "cookie", "credential", "private_key", "session_key",
)

_REDACTED = "<redigido>"


def _looks_secret(key: str) -> bool:
    lowered = str(key).casefold()
    return any(hint in lowered for hint in _SECRET_KEY_HINTS)


def redact_params(params: Any, *, max_string: int = 200) -> Any:
    """Devolve uma cópia segura para registo.

    - chaves que sugerem segredos são substituídas;
    - cadeias longas são truncadas;
    - profundidade limitada, para não serializar objectos arbitrários.
    """
    def _walk(value: Any, depth: int) -> Any:
        if depth > 4:
            return "<...>"
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                if _looks_secret(k):
                    out[k] = _REDACTED
                else:
                    out[k] = _walk(v, depth + 1)
            return out
        if isinstance(value, (list, tuple)):
            return [_walk(v, depth + 1) for v in value[:50]]
        if isinstance(value, str):
            return value if len(value) <= max_string else value[:max_string] + "…[truncado]"
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return str(value)[:max_string]
    return _walk(params if params is not None else {}, 0)


# ======================================================================
#  Execução com captura de output tolerante a ambientes restritos
# ======================================================================
# Em ambientes que proíbem pipes (o sandbox desta auditoria, por exemplo), o
# spawn com PIPE falha com PermissionError/ValueError. Nesse caso o comando é
# reexecutado sem captura (DEVNULL), o que preserva a execução e o código de
# saída, e o resultado declara explicitamente que o output não foi capturado.
# Não se inventa output: o resultado diz que não o tem.

def build_spawn_kwargs(cwd: Optional[Path], capture_output: bool) -> dict:
    from asyncio.subprocess import PIPE  # local: mantém o import leve no topo
    if capture_output:
        return {"stdout": PIPE, "stderr": PIPE, "text": True}
    import asyncio
    return {"stdout": asyncio.subprocess.DEVNULL, "stderr": asyncio.subprocess.DEVNULL}


def is_pipe_unavailable_error(exc: BaseException) -> bool:
    """True quando o ambiente impede a criação de pipes para o subprocesso."""
    if isinstance(exc, PermissionError):
        return True
    if isinstance(exc, ValueError):
        return "text must be False" in str(exc)
    return False


async def terminate_process_tree(proc) -> bool:
    """Termina o processo e, quando possível, os seus descendentes.

    Devolve True se conseguiu terminar a árvore inteira, False se apenas o
    processo directo. Quem chama deve registar qual dos casos ocorreu, para não
    dar como garantida uma limpeza que não aconteceu.
    """
    import asyncio

    if proc is None or proc.returncode is not None:
        return True

    killed_tree = await _terminate_tree(proc)
    if not killed_tree:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except (asyncio.TimeoutError, ProcessLookupError):
        logger.warning(
            "Processo %s não terminou no prazo após kill", getattr(proc, "pid", "?")
        )
    return killed_tree


# ======================================================================
#  Raízes autorizadas e contexto de ferramentas (decidido pelo servidor)
# ======================================================================
# O chamador NÃO escolhe o workspace: o servidor resolve-o a partir de duas
# raízes aprovadas, com finalidade declarada.
#
#   Workspace   -> espaço de trabalho de desenvolvimento
#   Quarantine  -> retenção controlada e preparação de isolamento
#
# Nenhuma destas pastas é, por si só, uma fronteira de segurança. São limites de
# *localização* validados; o isolamento real é outro problema (ver
# isolation_backends).

APPROVED_WORKSPACE_ROOTS: dict[str, str] = {
    "workspace": "Workspace",
    "quarantine": "Quarantine",
}

# Interruptor global do servidor: só *reduz* capacidades, nunca concede.
# (Não substitui a aprovação por operação; serve para desligar interpretadores
# em bloco sem alterar a política de propósitos.)
INTERPRETER_KILL_SWITCH_ENV = "ODYSSEUS_V2_DISABLE_INTERPRETERS"

# Variável histórica da Etapa 4. Deixou de conceder capacidade: um propósito é
# que determina as permissões, e a operação continua a exigir aprovação. É
# mantida apenas para avisar quem a tenha definida.
_DEPRECATED_ALLOW_INTERPRETERS_ENV = "ODYSSEUS_V2_ALLOW_INTERPRETERS"


@dataclass(frozen=True)
class PurposePolicy:
    """Permissões explícitas de um propósito de tarefa.

    Um propósito determina a raiz autorizada e as capacidades permitidas. Não
    concede autoridade: cada operação continua a exigir aprovação humana ligada
    ao seu fingerprint. Nenhum agente ou chamador pode alterar o propósito.
    """
    name: str
    root_key: str            # chave em APPROVED_WORKSPACE_ROOTS
    allow_interpreters: bool  # python / python3 / pytest no ShellTool
    allow_write: bool         # operações de escrita/eliminação em ficheiros
    description: str = ""


# Política central. Restritiva por omissão: `analysis` é o propósito padrão.
PURPOSES: dict[str, PurposePolicy] = {
    "analysis": PurposePolicy(
        name="analysis", root_key="workspace",
        allow_interpreters=False, allow_write=False,
        description="Inspecção e leitura no Workspace: sem interpretadores, sem escrita.",
    ),
    "development": PurposePolicy(
        name="development", root_key="workspace",
        allow_interpreters=True, allow_write=True,
        description="Desenvolvimento no Workspace: python/pytest e escrita, sob aprovação.",
    ),
    "testing": PurposePolicy(
        name="testing", root_key="workspace",
        allow_interpreters=True, allow_write=False,
        description="Correr testes no Workspace: interpretadores permitidos, escrita negada.",
    ),
    "documentation": PurposePolicy(
        name="documentation", root_key="workspace",
        allow_interpreters=False, allow_write=True,
        description="Documentação no Workspace: escrita permitida, sem interpretadores.",
    ),
    "security_validation": PurposePolicy(
        name="security_validation", root_key="quarantine",
        allow_interpreters=False, allow_write=False,
        description=(
            "Validação de segurança em Quarantine: isolada do Workspace e "
            "somente leitura (artefactos entram por via aprovada à parte)."
        ),
    ),
}

DEFAULT_PURPOSE = "analysis"


def resolve_purpose(purpose: Any) -> Optional[PurposePolicy]:
    """Resolve um propósito validado, ou None (fail-closed).

    Correspondência exacta dos tokens canónicos: variantes de caixa, sinónimos
    ou nomes de pastas não são aceites.
    """
    if not isinstance(purpose, str):
        return None
    return PURPOSES.get(purpose)


def purpose_root(purpose: Any) -> Optional[Path]:
    """Raiz autorizada do propósito, ou None se o propósito for inválido."""
    policy = resolve_purpose(purpose)
    if policy is None:
        return None
    return approved_root(policy.root_key)


def interpreters_disabled_globally() -> bool:
    """True se o interruptor global do servidor desliga interpretadores."""
    return os.getenv(INTERPRETER_KILL_SWITCH_ENV, "0").strip().lower() in ("1", "true", "yes")


def purpose_allows_interpreters(purpose: Any) -> bool:
    """Interpretadores: política do propósito **e** interruptor global."""
    if interpreters_disabled_globally():
        return False
    policy = resolve_purpose(purpose)
    return bool(policy and policy.allow_interpreters)


def purpose_allows_write(purpose: Any) -> bool:
    """Escrita/eliminação: depende do propósito.

    Sem propósito no contexto mantém-se o comportamento anterior (compatibilidade
    com o caminho aprovado na Etapa 1); com propósito presente mas inválido, nega
    (fail-closed).
    """
    if purpose is None or purpose == "":
        return True
    policy = resolve_purpose(purpose)
    return bool(policy and policy.allow_write)

# Chaves que o chamador nunca pode impor no contexto das ferramentas: deixariam
# o agente escolher (ou alargar) o próprio perímetro.
CALLER_FORBIDDEN_CONTEXT_KEYS: frozenset[str] = frozenset({
    "workspace_path", "workspace", "workspace_purpose",
    "cwd", "workdir", "allow_interpreters",
    "useragent_knowledge", "useragent_memory",
})


def approved_root(purpose: str = "workspace") -> Optional[Path]:
    """Resolve e valida uma raiz aprovada, ou None se não for aceitável.

    Aceita apenas `workspace` e `quarantine`. Exige que a pasta exista, seja um
    directório real (com symlinks/junctions resolvidos por realpath), esteja
    estritamente dentro da raiz do projecto e não seja área protegida.
    """
    # Correspondência exacta dos tokens canónicos: aceitar variantes de caixa ou
    # o nome da pasta alargaria a superfície de entrada sem benefício.
    key = str(purpose or "").strip()
    name = APPROVED_WORKSPACE_ROOTS.get(key)
    if name is None:
        return None

    try:
        resolved = Path(os.path.realpath(str(V2_PROJECT_ROOT / name)))
        project = Path(os.path.realpath(str(V2_PROJECT_ROOT)))
    except (OSError, ValueError, RuntimeError):
        return None

    if not resolved.is_dir():
        return None
    if resolved == project:
        return None
    if not _is_within(resolved, project):
        return None          # symlink/junction a apontar para fora
    if _is_protected_path(resolved):
        return None
    return resolved


def build_tool_context(purpose: Any = DEFAULT_PURPOSE, *,
                       session_id: Optional[str] = None,
                       user_id: Optional[str] = None,
                       allow_interpreters: Optional[bool] = None) -> dict:
    """Contexto de ferramentas decidido pelo servidor, a partir do propósito.

    As capacidades derivam **apenas** da política central de propósitos (e do
    interruptor global que só reduz). O parâmetro ``allow_interpreters`` existe
    apenas para compatibilidade com código existente; o servidor continua a
    decidir a capacidade final, e o valor do chamador nunca pode elevá-la.
    Devolve ``{}`` quando o propósito é inválido ou a raiz não é aceitável — e
    um contexto vazio faz as ferramentas de execução negarem (fail-closed).
    """
    if os.getenv(_DEPRECATED_ALLOW_INTERPRETERS_ENV) is not None:
        logger.warning(
            "%s está definida mas já não concede capacidades: as permissões "
            "dependem do propósito da tarefa. Usa %s para as desligar.",
            _DEPRECATED_ALLOW_INTERPRETERS_ENV, INTERPRETER_KILL_SWITCH_ENV,
        )

    policy = resolve_purpose(purpose)
    if policy is None:
        logger.warning("Propósito recusado pelo servidor: %r", purpose)
        return {}

    root = approved_root(policy.root_key)
    if root is None:
        logger.warning(
            "Raiz '%s' do propósito '%s' não está disponível", policy.root_key, policy.name
        )
        return {}

    context: dict = {
        "workspace_path": str(root),
        "workspace_purpose": policy.name,
        "workspace_root": policy.root_key,
    }
    if session_id:
        context["session_id"] = str(session_id)
    if user_id:
        context["user_id"] = str(user_id)

    server_allows_interpreters = purpose_allows_interpreters(policy.name)
    if allow_interpreters is not None:
        server_allows_interpreters = bool(allow_interpreters) and server_allows_interpreters
    # Só o servidor coloca esta chave e apenas quando a política do propósito
    # autoriza a capacidade; o chamador não pode elevá-la.
    if server_allows_interpreters:
        context["allow_interpreters"] = True
    return context


def sanitize_caller_context(context: Any) -> dict:
    """Remove do contexto do chamador tudo o que só o servidor pode decidir."""
    if not isinstance(context, dict):
        return {}
    reserved = sorted(set(context) & CALLER_FORBIDDEN_CONTEXT_KEYS)
    if reserved:
        logger.warning(
            "Contexto do chamador continha chaves reservadas (%s); ignoradas.",
            ", ".join(reserved),
        )
    return {k: v for k, v in context.items() if k not in CALLER_FORBIDDEN_CONTEXT_KEYS}


# ======================================================================
#  Isolamento real — sonda de disponibilidade
# ======================================================================
# Uma pasta, um `cwd` ou um ContextVar NÃO são isolamento. Esta sonda responde
# apenas a "existe backend capaz de impor limites de ficheiros/processos?".
# Não inicia serviços; limita-se a verificar o que já está disponível.

def isolation_backends() -> dict:
    """Detecta backends de isolamento disponíveis no ambiente actual."""
    import shutil
    import subprocess

    report: dict = {
        "windows_sandbox": False,
        "docker_cli": False,
        "docker_running": False,
        "wsl_cli": False,
        "wsl_is_security_boundary": False,
    }

    try:
        report["windows_sandbox"] = os.path.isfile(
            r"C:\Program Files\Windows Sandbox\WindowsSandbox.exe"
        )
    except Exception:
        pass

    report["docker_cli"] = shutil.which("docker") is not None
    if report["docker_cli"]:
        try:
            # Sem pipes: em ambientes restritos isto falha com PermissionError.
            done = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
            report["docker_running"] = done.returncode == 0
        except Exception as e:
            logger.info("Sonda docker indisponível: %s", e)

    report["wsl_cli"] = shutil.which("wsl") is not None

    report["effective_isolation_available"] = bool(
        report["windows_sandbox"] or report["docker_running"]
    )
    report["verdict"] = (
        "isolamento real disponível"
        if report["effective_isolation_available"]
        else "sem backend de isolamento: apenas contenção de localização"
    )
    return report


async def _terminate_tree(proc) -> bool:
    """Tenta terminar a árvore de processos do filho (psutil, depois taskkill)."""
    import subprocess

    pid = getattr(proc, "pid", None)
    if not pid:
        return False

    try:
        import psutil  # type: ignore
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except Exception:
                pass
        try:
            parent.kill()
        except Exception:
            pass
        return True
    except ImportError:
        pass
    except Exception as e:
        logger.warning("psutil falhou ao terminar a árvore do PID %s: %s", pid, e)

    if os.name == "nt":
        try:
            # Sem pipes: o ambiente pode proibi-los e não precisamos do texto.
            done = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            return done.returncode == 0
        except Exception as e:
            logger.warning("taskkill falhou para o PID %s: %s", pid, e)
    return False
