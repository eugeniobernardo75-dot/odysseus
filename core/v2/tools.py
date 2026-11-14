"""
Ferramentas Base — Odysseus V2

Coleção de ferramentas padrão do sistema:

    - WebTool:      Fetch de páginas web, extração de conteúdo
    - FileTool:     Operações seguras de ficheiros (read, write, list, search)
    - ShellTool:    Execução de comandos shell (com segurança)
    - PythonTool:   Execução de código Python em sandbox
    - MemoryTool:   Acesso ao sistema de memória (guardar, consultar)
    - SearchTool:   Pesquisa semântica em ficheiros do workspace
    - CodeTool:     Análise de código com AST e métricas
    - GitTool:      Operações git seguras (status, log, diff, branch, etc.)
    - BrowserTool:  Automação web (Playwright/httpx+BS4)
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess  # nosec
import sys
import textwrap
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .tool_registry import BaseTool, ToolDefinition, ToolResult
from .execution_policy import (
    authorization_denied_message,
    build_spawn_kwargs,
    check_command,
    is_authorized_for,
    is_pipe_unavailable_error,
    purpose_allows_write,
    vet_workspace,
)
from .isolation import isolation_flags, isolation_refusal_reason

# Import do BrowserTool feito separadamente em ficheiro próprio
# from .browser_tool import BrowserTool
# (import lazy dentro de register_base_tools para evitar circular imports)

logger = logging.getLogger(__name__)


# ======================================================================
#  Fronteira de acesso a ficheiros
# ======================================================================
# Áreas que as ferramentas de ficheiros do V2 nunca devem alcançar, mesmo
# dentro do projecto e independentemente do workspace:
#   - Obsidian Brain Vault : conhecimento documental persistente
#   - Odysseus_Mind_OS     : arquitectura normativa
#   - data/ e logs/        : estado operacional e segredos
#   - .git                 : histórico de versões
# O acesso ao Brain Vault e ao Mind OS pertence ao UserAgent, não a
# ferramentas que executam pedidos arbitrários.
#
# A contenção é feita com normcase + commonpath (nunca com startswith),
# porque "...\odysseus-outro" partilha o prefixo textual de "...\odysseus".

V2_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

V2_PROTECTED_DIRS: tuple[str, ...] = (
    "Obsidian Brain Vault",
    "Odysseus_Mind_OS",
    "data",
    "logs",
    ".git",
)

V2_PROTECTED_FILES: frozenset[str] = frozenset(
    name.casefold() for name in (".env", ".app_key")
)


def _is_within(path: Path, root: Path) -> bool:
    """True se *path* estiver dentro de *root* (contenção real).

    Usa caminhos reais e comparação insensível a maiúsculas (Windows),
    rejeitando drives diferentes e prefixos meramente textuais.
    """
    try:
        resolved = os.path.realpath(str(path))
        root_resolved = os.path.realpath(str(root))
    except (OSError, ValueError):
        return False
    norm_path = os.path.normcase(resolved)
    norm_root = os.path.normcase(root_resolved)
    if norm_path == norm_root:
        return True
    try:
        return os.path.commonpath([norm_path, norm_root]) == norm_root
    except ValueError:
        return False  # drives diferentes => fora


def _is_protected_path(path: Path) -> bool:
    """True se *path* cair numa área protegida (ver V2_PROTECTED_*)."""
    name = path.name.casefold()
    if name in V2_PROTECTED_FILES or name.startswith(".env."):
        return True
    return any(
        _is_within(path, V2_PROJECT_ROOT / protected)
        for protected in V2_PROTECTED_DIRS
    )


def _resolve_confined(path: str, base: Path) -> Optional[Path]:
    """Resolve *path* contra *base* e valida a fronteira de acesso.

    Regras:
        - caminhos relativos são resolvidos dentro de *base*;
        - caminhos absolutos só são aceites se caírem dentro de *base*;
        - áreas protegidas são sempre negadas;
        - a contenção usa _is_within (normcase + commonpath).

    Devolve o caminho real, ou None se for inválido ou estiver fora.
    """
    if path is None or not str(path).strip():
        return None
    try:
        raw = Path(os.path.expanduser(str(path).strip()))
        root = Path(os.path.realpath(str(base)))
        candidate = raw if raw.is_absolute() else (root / raw)
        resolved = Path(os.path.realpath(str(candidate)))
    except (OSError, ValueError, RuntimeError):
        return None
    if _is_protected_path(resolved):
        return None
    if not _is_within(resolved, root):
        return None
    return resolved


def _resolve_search_target(directory: str, base: Path, confine: bool) -> Optional[Path]:
    """Resolve o diretório de pesquisa do SearchTool.

    confine=True  -> contenção estrita dentro de *base* (usado quando existe
                     workspace explícito no contexto);
    confine=False -> mantém a semântica anterior para caminhos absolutos,
                     negando apenas as áreas protegidas.

    Devolve o caminho real, ou None se for inválido, protegido ou estiver fora.
    """
    if directory is None or not str(directory).strip():
        return None
    try:
        raw = Path(os.path.expanduser(str(directory).strip()))
        root = Path(os.path.realpath(str(base)))
        candidate = raw if raw.is_absolute() else (root / raw)
        resolved = Path(os.path.realpath(str(candidate)))
    except (OSError, ValueError, RuntimeError):
        return None
    if _is_protected_path(resolved):
        return None
    if confine and not _is_within(resolved, root):
        return None
    return resolved


# ======================================================================
#  WebTool
# ======================================================================

class WebTool(BaseTool):
    """
    Ferramenta para fetch de páginas web e extração de conteúdo.

    Capacidades:
        - web.fetch: fetch de URLs com timeout e User-Agent
        - web.extract: extração de texto de HTML
        - web.extract_urls: extração de links de texto
    """

    def _build_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web",
            description="Fetch de páginas web, extração de conteúdo e URLs",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["fetch", "extract", "fetch_and_extract"],
                        "description": "Ação a executar",
                    },
                    "url": {
                        "type": "string",
                        "description": "URL para fetch",
                    },
                    "text": {
                        "type": "string",
                        "description": "Texto para extrair URLs (apenas extract)",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Máximo de caracteres a extrair",
                        "default": 5000,
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout em segundos",
                        "default": 15.0,
                    },
                },
                "required": ["action"],
            },
            capabilities=["web.fetch", "web.extract", "web.extract_urls"],
            permissions=["network"],
            category="web",
            version="1.0.0",
            timeout=30.0,
            tags=["internet", "http", "html", "scraping"],
        )

    async def execute(self, params: dict, context: Optional[dict] = None) -> ToolResult:
        action = params.get("action", "fetch_and_extract")

        if action == "extract":
            text = params.get("text", "")
            if not text:
                return ToolResult(success=False, error="'text' é obrigatório para extract")
            urls = self._extract_urls(text)
            return ToolResult(
                success=True,
                data={"urls": urls, "count": len(urls)},
                metadata={"action": "extract"},
            )

        url = params.get("url", "")
        if not url:
            return ToolResult(success=False, error="'url' é obrigatório para fetch")

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        max_chars = params.get("max_chars", 5000)
        timeout = params.get("timeout", 15.0)

        try:
            import httpx
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=True
            ) as client:
                resp = await client.get(url, headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                })

                if resp.status_code != 200:
                    return ToolResult(
                        success=False,
                        error=f"HTTP {resp.status_code} ao aceder a {url}",
                        metadata={"url": url, "status_code": resp.status_code},
                    )

                html = resp.text
                extracted = self._extract_text(html, max_chars=max_chars)

                result_data = {
                    "url": url,
                    "status_code": resp.status_code,
                    "content_length": len(html),
                    "content_preview": extracted[:1000],
                    "extracted_text": extracted,
                    "content_type": resp.headers.get("content-type", ""),
                }

                if action in ("fetch", "fetch_and_extract"):
                    result_data["urls_found"] = self._extract_urls(extracted)

                return ToolResult(
                    success=True,
                    data=result_data,
                    metadata={
                        "action": action,
                        "url": url,
                        "size_bytes": len(html),
                    },
                )

        except ImportError:
            return ToolResult(
                success=False,
                error="Biblioteca httpx não instalada. Instala: pip install httpx",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Erro ao fazer fetch de {url}: {e}",
                metadata={"url": url},
            )

    def _extract_text(self, html: str, max_chars: int = 5000) -> str:
        """Extração básica de texto de HTML."""
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text

    def _extract_urls(self, text: str) -> list[str]:
        """Extrai URLs de um texto."""
        url_pattern = re.compile(
            r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
            r"(?:/[-\w$.+!*'(),;:@&=?/~#%]*)?"
        )
        return list(set(url_pattern.findall(text)))


# ======================================================================
#  FileTool
# ======================================================================

class FileTool(BaseTool):
    """
    Ferramenta para operações seguras de ficheiros.

    Capacidades:
        - file.read: leitura de ficheiros
        - file.write: escrita de ficheiros
        - file.list: listagem de diretórios
        - file.search: pesquisa de texto em ficheiros
        - file.delete: eliminação de ficheiros
    """

    def _build_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file",
            description="Operações seguras de ficheiros: ler, escrever, listar, pesquisar",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "write", "list", "search", "delete", "exists"],
                        "description": "Ação a executar",
                    },
                    "path": {
                        "type": "string",
                        "description": "Caminho do ficheiro/diretório",
                    },
                    "content": {
                        "type": "string",
                        "description": "Conteúdo a escrever (apenas write)",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern para listar (apenas list)",
                    },
                    "query": {
                        "type": "string",
                        "description": "Texto a pesquisar (apenas search)",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Máximo de resultados",
                        "default": 20,
                    },
                },
                "required": ["action", "path"],
            },
            capabilities=[
                "file.read", "file.write", "file.list",
                "file.search", "file.delete", "file.exists",
            ],
            permissions=["filesystem.read", "filesystem.write"],
            category="file",
            version="1.0.0",
            timeout=30.0,
            requires_approval=False,
            tags=["filesystem", "io"],
        )

    # Ações que alteram o sistema de ficheiros.
    _WRITE_ACTIONS = frozenset({"write", "delete"})

    async def execute(self, params: dict, context: Optional[dict] = None) -> ToolResult:
        action = params["action"]
        path = params["path"]
        workspace_base = self._get_workspace_base(context)

        # ── Capacidade de escrita depende do propósito da tarefa ──
        # Sem propósito no contexto mantém-se o comportamento aprovado na
        # Etapa 1; com propósito presente, a política central decide (e nega por
        # omissão). O agente não pode alterar o propósito: as chaves reservadas
        # do contexto do chamador são descartadas antes de chegar aqui.
        if action in self._WRITE_ACTIONS:
            declared_purpose = (context or {}).get("workspace_purpose")
            if not purpose_allows_write(declared_purpose):
                return ToolResult(
                    success=False,
                    error=(
                        f"O propósito '{declared_purpose}' não permite escrita nem "
                        f"eliminação de ficheiros."
                    ),
                    metadata={"denied": True, "reason": "purpose_read_only"},
                )

        # Resolver caminho (seguro)
        full_path = self._resolve_path(path, workspace_base)
        if full_path is None:
            return ToolResult(
                success=False,
                error=f"Caminho inválido ou fora do workspace: '{path}'",
            )

        try:
            if action == "read":
                return await self._read(full_path, path)
            elif action == "write":
                content = params.get("content", "")
                return await self._write(full_path, path, content)
            elif action == "list":
                pattern = params.get("pattern", "*")
                return await self._list(full_path, path, pattern)
            elif action == "search":
                query = params.get("query", "")
                max_results = params.get("max_results", 20)
                return await self._search(full_path, query, max_results)
            elif action == "delete":
                return await self._delete(full_path, path)
            elif action == "exists":
                return ToolResult(
                    success=True,
                    data={"exists": full_path.exists(), "path": path},
                )
            else:
                return ToolResult(success=False, error=f"Ação desconhecida: {action}")

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Erro na operação '{action}' em '{path}': {e}",
                metadata={"action": action, "path": path},
            )

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _get_workspace_base(self, context: Optional[dict]) -> Optional[Path]:
        """Obtém o diretório base do workspace a partir do contexto."""
        if context and "workspace_path" in context:
            return Path(context["workspace_path"]).resolve()
        return None

    def _resolve_path(self, path: str, workspace_base: Optional[Path]) -> Optional[Path]:
        """
        Resolve e valida um caminho de forma segura.

        - Com workspace: o caminho é resolvido dentro dele e a contenção é
          verificada com normcase + commonpath (não com prefixos textuais);
        - Sem workspace: mantém-se a raiz do projecto como base;
        - Áreas protegidas (Brain Vault, Mind OS, data/, logs/, .git, .env)
          são sempre negadas, com ou sem workspace.
        """
        base = workspace_base if workspace_base is not None else V2_PROJECT_ROOT
        return _resolve_confined(path, base)

    async def _read(self, full_path: Path, original_path: str) -> ToolResult:
        if not full_path.exists():
            return ToolResult(success=False, error=f"Ficheiro não encontrado: '{original_path}'")
        if not full_path.is_file():
            return ToolResult(success=False, error=f"'{original_path}' não é um ficheiro")

        size = full_path.stat().st_size
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        return ToolResult(
            success=True,
            data={
                "path": original_path,
                "content": content,
                "size_bytes": size,
                "lines": content.count("\n") + 1,
            },
            metadata={"path": original_path, "size_bytes": size},
        )

    async def _write(self, full_path: Path, original_path: str, content: str) -> ToolResult:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        return ToolResult(
            success=True,
            data={
                "path": original_path,
                "written_bytes": len(content),
                "lines": content.count("\n") + 1,
            },
            metadata={"path": original_path, "written_bytes": len(content)},
        )

    async def _list(self, full_path: Path, original_path: str, pattern: str) -> ToolResult:
        if not full_path.exists():
            return ToolResult(success=False, error=f"Diretório não encontrado: '{original_path}'")
        if not full_path.is_dir():
            # Se for ficheiro, listar o diretório pai
            full_path = full_path.parent

        files = []
        for fpath in sorted(full_path.glob(pattern)):
            if fpath.is_file() or fpath.is_dir():
                is_dir = fpath.is_dir()
                stat = fpath.stat()
                rel_path = fpath.name
                # Tentar caminho relativo ao original_path
                try:
                    rel_path = str(fpath.relative_to(full_path.parent))
                except ValueError:
                    rel_path = str(fpath)

                info = {
                    "name": fpath.name,
                    "path": rel_path,
                    "type": "directory" if is_dir else "file",
                    "size_bytes": 0 if is_dir else stat.st_size,
                }
                if is_dir:
                    try:
                        info["children"] = len(list(fpath.iterdir()))
                    except Exception:
                        info["children"] = 0

                files.append(info)

        return ToolResult(
            success=True,
            data={
                "path": original_path,
                "files": files,
                "count": len(files),
            },
        )

    async def _search(self, full_path: Path, query: str, max_results: int) -> ToolResult:
        if not full_path.exists():
            return ToolResult(success=False, error=f"Diretório não encontrado: '{full_path}'")

        if not full_path.is_dir():
            search_paths = [full_path]
        else:
            search_paths = [full_path]

        results = []
        for search_path in search_paths:
            for root, _, files in os.walk(search_path):
                for fname in files:
                    if len(results) >= max_results:
                        break
                    if not fname.endswith((".py", ".md", ".txt", ".json", ".yaml", ".yml", ".html", ".css", ".js", ".ts")):
                        continue
                    fpath = Path(root) / fname
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            for i, line in enumerate(f, 1):
                                if query.lower() in line.lower():
                                    results.append({
                                        "file": str(fpath),
                                        "line": i,
                                        "content": line.strip()[:200],
                                    })
                                    if len(results) >= max_results:
                                        break
                    except Exception:
                        continue

        return ToolResult(
            success=True,
            data={
                "query": query,
                "results": results,
                "count": len(results),
                "max_results": max_results,
            },
        )

    async def _delete(self, full_path: Path, original_path: str) -> ToolResult:
        if not full_path.exists():
            return ToolResult(success=False, error=f"Ficheiro não encontrado: '{original_path}'")

        if full_path.is_dir():
            return ToolResult(success=False, error=f"Não é possível apagar diretórios: '{original_path}'")

        size = full_path.stat().st_size
        full_path.unlink()
        return ToolResult(
            success=True,
            data={"path": original_path, "deleted_bytes": size},
            metadata={"path": original_path},
        )


# ======================================================================
#  ShellTool (com segurança)
# ======================================================================

# Lista de comandos bloqueados por segurança
_BLOCKED_COMMANDS = [
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=", ":(){ :|:& };:",  # fork bomb
    "> /dev/sda", "wget -O- | sh", "curl -s http",
    "sudo ", "su ", "chmod 777", "chown ", "passwd",
    "kill -9", "pkill", "systemctl", "reboot", "shutdown",
    "docker rm", "docker rmi", "docker stop",
    "pip uninstall", "pip3 uninstall",
    "npm uninstall", "npx create-",
]


class ShellTool(BaseTool):
    """
    Ferramenta para executar comandos shell com segurança.

    Capacidades:
        - shell.execute: executar comando shell
        - shell.check: verificar se comando é seguro

    Segurança:
        - Lista de comandos bloqueados
        - Timeout configurável
        - Apenas comandos de leitura e desenvolvimento
    """

    def _build_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="shell",
            description="Execução segura de comandos shell (apenas leitura e desenvolvimento)",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Comando shell a executar",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout em segundos",
                        "default": 30.0,
                    },
                    "workdir": {
                        "type": "string",
                        "description": "Diretório de trabalho (opcional)",
                    },
                },
                "required": ["command"],
            },
            capabilities=["shell.execute", "shell.check"],
            permissions=["shell.exec"],
            category="system",
            version="1.0.0",
            timeout=60.0,
            requires_approval=True,  # Comandos shell requerem aprovação
            tags=["shell", "terminal", "command"],
        )

    async def execute(self, params: dict, context: Optional[dict] = None) -> ToolResult:
        command = params.get("command", "")
        workdir = params.get("workdir")

        # ── 1. Autorização ────────────────────────────────────────────
        # Só o ToolManager activa o marcador, e apenas depois de uma aprovação
        # válida e ligada a esta operação. Uma chamada directa não passa aqui.
        if not is_authorized_for(self.name):
            return ToolResult(
                success=False,
                error=authorization_denied_message(self.name),
                metadata={"denied": True, "reason": "not_authorized"},
            )

        if not command:
            return ToolResult(success=False, error="Comando vazio")

        # ── 2. Workspace autorizado ───────────────────────────────────
        workspace = vet_workspace((context or {}).get("workspace_path"))
        if workspace is None:
            return ToolResult(
                success=False,
                error=(
                    "Sem workspace autorizado: o ShellTool exige "
                    "context['workspace_path'] que aponte para um directório "
                    "existente, fora de raízes de filesystem e de áreas protegidas."
                ),
                metadata={"denied": True, "reason": "no_workspace"},
            )

        # ── 3. Deny-list comum + política explícita ───────────────────
        # A deny-list de comandos destrutivos é comum a todas as plataformas.
        # A allow-list é da política (consciente da plataforma): a allow-list
        # POSIX de `_check_safety` não pode decidir aqui, porque negaria as
        # ferramentas de inspecção válidas em Windows.
        blocked_reason = self._check_denylist(command)
        if blocked_reason:
            return ToolResult(
                success=False,
                error=blocked_reason,
                metadata={"blocked": True, "reason": blocked_reason},
            )

        # `allow_interpreters` só existe no contexto quando o *servidor* o
        # autorizou (configuração), nunca por escolha do agente ou do chamador.
        policy = check_command(
            command, workspace,
            allow_interpreters=bool((context or {}).get("allow_interpreters")),
        )
        if not policy["allowed"]:
            return ToolResult(
                success=False,
                error=policy["reason"],
                metadata={"blocked": True, "reason": policy["reason"]},
            )

        # ── 4. Confinamento do directório de trabalho ─────────────────
        cwd = workspace
        if workdir:
            # Contenção real (normcase + commonpath) e negação de áreas
            # protegidas — o mesmo helper que o FileTool usa.
            resolved_cwd = _resolve_confined(workdir, workspace)
            if resolved_cwd is None:
                return ToolResult(
                    success=False,
                    error=(
                        f"workdir '{workdir}' está fora do workspace autorizado "
                        f"ou numa área protegida."
                    ),
                    metadata={"denied": True, "reason": "workdir_out_of_workspace"},
                )
            if not resolved_cwd.is_dir():
                return ToolResult(
                    success=False,
                    error=f"workdir '{workdir}' não é um directório existente.",
                    metadata={"denied": True, "reason": "workdir_missing"},
                )
            cwd = resolved_cwd

        # ── 5. Limites de tempo ───────────────────────────────────────
        try:
            timeout = float(params.get("timeout", 30.0))
        except (TypeError, ValueError):
            return ToolResult(success=False, error="timeout inválido")
        if timeout <= 0:
            return ToolResult(success=False, error="timeout tem de ser positivo")
        timeout = min(timeout, float(self.definition.timeout))

        # ── 6. Isolamento ─────────────────────────────────────────────
        # Sem backend não há isolamento. Se o servidor o exige, recusa-se em vez
        # de executar sem protecção e chamar-lhe isolado.
        refusal = isolation_refusal_reason()
        if refusal:
            return ToolResult(
                success=False,
                error=refusal,
                metadata={"isolation_required": True, **isolation_flags()},
            )

        return await self._spawn(policy["argv"], cwd, timeout, command)

    # ------------------------------------------------------------------
    #  Execução do processo
    # ------------------------------------------------------------------

    async def _spawn_isolated(self, argv: list[str], cwd: Path, timeout: float,
                              command: str) -> Optional[ToolResult]:
        """Executa através do backend de isolamento, quando existe.

        Devolve None quando não há backend: nesse caso o chamador segue o
        caminho directo, e o resultado declara ``isolated: false``.
        """
        from .isolation import isolation_status, run_isolated

        status = isolation_status()
        if not status.isolated:
            return None

        outcome = await run_isolated(argv, cwd, timeout)
        iso = {
            "isolated": True,
            "isolation_backend": status.backend,
            "isolation_detail": status.detail,
            "container": outcome.get("container"),
            "terminated": outcome.get("terminated"),
            "cleanup_unconfirmed": outcome.get("cleanup_unconfirmed"),
        }
        if outcome["error"]:
            return ToolResult(
                success=False,
                error=f"Execução isolada falhou: {outcome['error']}",
                metadata={"command": command[:200], **iso},
            )

        stdout = outcome["stdout"]
        stderr = outcome["stderr"]
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        if stdout is None or stderr is None:
            return ToolResult(
                success=outcome["ok"],
                data={
                    "stdout": None, "stderr": None,
                    "return_code": outcome["return_code"],
                    "command": command[:200],
                    "output_captured": False,
                    **iso,
                },
                error=None if outcome["ok"] else (
                    f"Comando terminou com código {outcome['return_code']} "
                    f"(output não capturado)."
                ),
                metadata={"return_code": outcome["return_code"],
                          "output_captured": False, **iso},
                warnings=["Output não capturado dentro do backend de isolamento."],
            )

        return ToolResult(
            success=outcome["ok"],
            data={
                "stdout": stdout, "stderr": stderr,
                "return_code": outcome["return_code"],
                "command": command[:200],
                "output_captured": True,
                **iso,
            },
            error=stderr if not outcome["ok"] else None,
            metadata={"return_code": outcome["return_code"],
                      "output_captured": True, **iso},
        )

    async def _spawn(self, argv: list[str], cwd: Path, timeout: float,
                     command: str) -> ToolResult:
        """Executa argv (sem shell) com captura de output tolerante.

        Com backend de isolamento disponível, a execução passa por ele; sem
        backend, corre directamente e o resultado declara ``isolated: false``.
        """
        import asyncio

        isolated = await self._spawn_isolated(argv, cwd, timeout, command)
        if isolated is not None:
            return isolated

        capture = True
        for attempt in (1, 2):
            kwargs = build_spawn_kwargs(cwd, capture)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv, cwd=str(cwd), **kwargs
                )
            except Exception as e:
                if attempt == 1 and is_pipe_unavailable_error(e):
                    # Ambiente sem pipes: reexecuta sem captura, preservando a
                    # execução e o código de saída, e declara-o no resultado.
                    logger.warning(
                        "Pipes indisponíveis neste ambiente (%s); '%s' corre sem "
                        "captura de output.", type(e).__name__, argv[0],
                    )
                    capture = False
                    continue
                return ToolResult(
                    success=False,
                    error=f"Não foi possível iniciar o processo: {e}",
                    metadata={
                        "spawn_failed": True,
                        "exception": type(e).__name__,
                        "command": command[:200],
                    },
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                terminated = await self._terminate(proc)
                return ToolResult(
                    success=False,
                    error=(
                        f"Comando excedeu timeout de {timeout}s e foi terminado "
                        f"({'árvore terminada' if terminated else 'apenas o processo directo'})."
                    ),
                    metadata={"timeout": timeout, "terminated": terminated},
                )
            except asyncio.CancelledError:
                await self._terminate(proc)
                raise

            return self._build_result(proc, stdout, stderr, capture, command)

        return ToolResult(
            success=False,
            error="Não foi possível iniciar o processo (tentativas esgotadas).",
            metadata={"spawn_failed": True},
        )

    async def _terminate(self, proc) -> bool:
        """Termina o processo e, quando possível, os descendentes.

        Implementação única em execution_policy, para ser a mesma no ShellTool
        e no PythonTool. Devolve True se terminou a árvore, False se apenas o
        processo directo — e o resultado da execução regista qual foi o caso.
        """
        from .execution_policy import terminate_process_tree
        return await terminate_process_tree(proc)

    def _build_result(self, proc, stdout, stderr, capture: bool,
                      command: str) -> ToolResult:
        """Constrói o resultado, distinguindo os desfechos possíveis."""
        return_code = proc.returncode
        # Todo o resultado declara se correu isolado. Hoje é sempre False:
        # não existe backend de isolamento neste ambiente.
        iso = isolation_flags()

        if return_code is None:
            # Nunca converter ausência de código de saída em sucesso.
            return ToolResult(
                success=False,
                error="O processo terminou sem código de saída (return_code=None).",
                metadata={"return_code": None, "output_captured": capture, **iso},
            )

        success = return_code == 0

        if not capture:
            return ToolResult(
                success=success,
                data={
                    "stdout": None,
                    "stderr": None,
                    "return_code": return_code,
                    "command": command[:200],
                    "output_captured": False,
                    **iso,
                },
                error=None if success else (
                    f"Comando terminou com código de saída {return_code} "
                    f"(output não capturado neste ambiente)."
                ),
                metadata={"return_code": return_code, "output_captured": False, **iso},
                warnings=[
                    "Output não capturado: o ambiente não permite pipes entre "
                    "processos. O código de saída é real.",
                ],
            )

        max_output = 50000
        if len(stdout) > max_output:
            stdout = stdout[:max_output] + f"\n... [truncado, {len(stdout)} caracteres no total]"
        if len(stderr) > max_output:
            stderr = stderr[:max_output] + f"\n... [truncado, {len(stderr)} caracteres no total]"

        return ToolResult(
            success=success,
            data={
                "stdout": stdout,
                "stderr": stderr,
                "return_code": return_code,
                "command": command[:200],
                "output_captured": True,
                **iso,
            },
            error=stderr if not success else None,
            metadata={
                "return_code": return_code,
                "stdout_length": len(stdout),
                "stderr_length": len(stderr),
                "output_captured": True,
                **iso,
            },
        )

    @staticmethod
    def _check_denylist(command: str) -> Optional[str]:
        """Devolve o motivo se o comando tocar a deny-list, senão None.

        A deny-list (comandos destrutivos, comuns a todas as plataformas) é
        aplicada antes da política. A decisão de *permitir* pertence à política
        em execution_policy, que é consciente da plataforma.
        """
        cmd_lower = (command or "").lower().strip()
        for blocked in _BLOCKED_COMMANDS:
            if blocked in cmd_lower:
                return f"Comando bloqueado por segurança: '{blocked}'"
        return None

    def _check_safety(self, command: str) -> dict:
        """Filtro legado: deny-list + allow-list POSIX.

        Mantido com o comportamento original para não quebrar os testes e para
        uso como verificação de conveniência; o caminho de execução usa
        `_check_denylist` + `execution_policy.check_command`.
        """
        cmd_lower = command.lower().strip()

        # Verificar comandos bloqueados
        blocked = self._check_denylist(command)
        if blocked:
            return {"safe": False, "reason": blocked}

        # Permitir comandos comuns de desenvolvimento
        allowed_prefixes = [
            "python", "py", "pip", "npm", "node", "git",
            "ls", "cat", "head", "tail", "echo", "pwd",
            "cd", "mkdir", "touch", "cp", "mv", "grep",
            "find", "wc", "sort", "uniq", "diff", "tree",
            "python3", "pytest", "uvicorn", "fastapi",
            "docker", "docker-compose",
        ]

        first_word = cmd_lower.split()[0] if cmd_lower.split() else ""
        if first_word in allowed_prefixes:
            return {"safe": True, "reason": ""}

        # Comandos de leitura/query são seguros
        readonly_indicators = [
            "cat ", "ls ", "head ", "tail ", "grep ",
            "find ", "wc ", "echo ", "pwd ", "which ",
            "whoami", "date", "pwd", "env",
        ]
        if any(cmd_lower.startswith(ind) for ind in readonly_indicators):
            return {"safe": True, "reason": ""}

        return {
            "safe": False,
            "reason": f"Comando não reconhecido como seguro: '{command[:100]}'",
        }


# ======================================================================
#  PythonTool (execução segura de código)
# ======================================================================

class PythonTool(BaseTool):
    """
    Ferramenta para executar snippets de código Python.

    Capacidades:
        - python.execute: executar código Python
        - python.analyze: analisar código com AST
        - python.format: formatar código

    Segurança:
        - Timeout na execução
        - Bloqueio de imports perigosos
        - Output limitado
    """

    _DANGEROUS_MODULES = {
        "os", "subprocess", "shutil", "socket", "ctypes",
        "multiprocessing", "threading", "signal", "sys",
        "code", "codeop", "codecs", "importlib",
        "pickle", "shelve", "marshal",
    }

    def _build_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="python",
            description="Execução segura de código Python (análise, execução, formatação)",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["execute", "analyze", "format"],
                        "description": "Ação a executar",
                    },
                    "code": {
                        "type": "string",
                        "description": "Código Python",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout em segundos",
                        "default": 10.0,
                    },
                },
                "required": ["action", "code"],
            },
            capabilities=["python.execute", "python.analyze", "python.format"],
            permissions=["code.exec"],
            category="code",
            version="1.0.0",
            timeout=30.0,
            requires_approval=True,  # Execução de código requer aprovação
            tags=["python", "code", "ast"],
        )

    async def execute(self, params: dict, context: Optional[dict] = None) -> ToolResult:
        action = params["action"]
        code = params.get("code", "")

        if not code:
            return ToolResult(success=False, error="Código vazio")

        if action == "analyze":
            return await self._analyze(code)
        elif action == "format":
            return await self._format(code)
        else:
            timeout = params.get("timeout", 10.0)
            return await self._execute_safe(code, timeout, context)

    async def _execute_safe(self, code: str, timeout: float,
                            context: Optional[dict] = None) -> ToolResult:
        """Executa código Python num subprocesso confinado ao workspace.

        Barreiras reais: autorização do ToolManager (após aprovação) e cwd
        dentro do workspace autorizado. A verificação textual de imports e
        chamadas é defesa em profundidade.

        Limitação declarada: executar Python é executar código arbitrário. A
        análise textual não é uma fronteira de segurança; a fronteira é a
        aprovação humana mais o confinamento do directório de trabalho.
        """
        from .execution_policy import terminate_process_tree

        if not is_authorized_for(self.name):
            return ToolResult(
                success=False,
                error=authorization_denied_message(self.name),
                metadata={"denied": True, "reason": "not_authorized"},
            )

        workspace = vet_workspace((context or {}).get("workspace_path"))
        if workspace is None:
            return ToolResult(
                success=False,
                error=(
                    "Sem workspace autorizado: o PythonTool exige "
                    "context['workspace_path'] que aponte para um directório "
                    "existente, fora de raízes de filesystem e de áreas protegidas."
                ),
                metadata={"denied": True, "reason": "no_workspace"},
            )

        # Verificar imports perigosos
        dangerous = self._check_dangerous_imports(code)
        if dangerous:
            return ToolResult(
                success=False,
                error=f"Código contém módulos bloqueados: {', '.join(dangerous)}",
                metadata={"blocked_modules": dangerous},
            )

        # Verificar estruturas perigosas
        if "exec(" in code or "eval(" in code or "compile(" in code or "__import__" in code:
            return ToolResult(
                success=False,
                error="Código contém chamadas bloqueadas (exec, eval, compile)",
            )

        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            return ToolResult(success=False, error="timeout inválido")
        if timeout <= 0:
            return ToolResult(success=False, error="timeout tem de ser positivo")
        timeout = min(timeout, float(self.definition.timeout))

        # Isolamento: sem backend, a execução é recusada quando exigida.
        refusal = isolation_refusal_reason()
        if refusal:
            return ToolResult(
                success=False,
                error=refusal,
                metadata={"isolation_required": True, **isolation_flags()},
            )

        # Executar em subprocesso, com cwd no workspace autorizado
        import asyncio

        iso = isolation_flags()
        capture = True

        # Com backend de isolamento, o código corre dentro dele.
        from .isolation import isolation_status, run_isolated
        if isolation_status().isolated:
            # Dentro do backend, o interpretador é o do contentor: `sys.executable`
            # é um caminho do host (Windows) e não existe lá.
            outcome = await run_isolated(["python", "-c", code], workspace, timeout)
            isolated_flags = {
                "isolated": True,
                "isolation_backend": isolation_status().backend,
                "isolation_detail": isolation_status().detail,
                "container": outcome.get("container"),
                "terminated": outcome.get("terminated"),
                "cleanup_unconfirmed": outcome.get("cleanup_unconfirmed"),
            }
            if outcome["error"]:
                return ToolResult(
                    success=False,
                    error=f"Execução isolada falhou: {outcome['error']}",
                    metadata={**isolated_flags},
                )
            return ToolResult(
                success=outcome["ok"],
                data={
                    "stdout": outcome["stdout"],
                    "stderr": outcome["stderr"],
                    "return_code": outcome["return_code"],
                    "output_captured": outcome["stdout"] is not None,
                    **isolated_flags,
                },
                error=None if outcome["ok"] else "Execução isolada devolveu código não-zero.",
                metadata={"return_code": outcome["return_code"], **isolated_flags},
            )
        for attempt in (1, 2):
            kwargs = build_spawn_kwargs(workspace, capture)
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-c", code, cwd=str(workspace), **kwargs
                )
            except Exception as e:
                if attempt == 1 and is_pipe_unavailable_error(e):
                    logger.warning(
                        "Pipes indisponíveis (%s); PythonTool corre sem captura de output.",
                        type(e).__name__,
                    )
                    capture = False
                    continue
                return ToolResult(
                    success=False,
                    error=f"Não foi possível iniciar o processo: {e}",
                    metadata={"spawn_failed": True, "exception": type(e).__name__},
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                terminated = await terminate_process_tree(proc)
                return ToolResult(
                    success=False,
                    error=(
                        f"Execução excedeu timeout de {timeout}s e foi terminada "
                        f"({'árvore terminada' if terminated else 'apenas o processo directo'})."
                    ),
                    metadata={"timeout": timeout, "terminated": terminated},
                )
            except asyncio.CancelledError:
                await terminate_process_tree(proc)
                raise

            return_code = proc.returncode
            if return_code is None:
                return ToolResult(
                    success=False,
                    error="O processo terminou sem código de saída (return_code=None).",
                    metadata={"return_code": None, "output_captured": capture, **iso},
                )

            success = return_code == 0

            if not capture:
                return ToolResult(
                    success=success,
                    data={"stdout": None, "stderr": None,
                          "return_code": return_code, "output_captured": False, **iso},
                    error=None if success else (
                        f"Execução terminou com código {return_code} "
                        f"(output não capturado neste ambiente)."
                    ),
                    metadata={"return_code": return_code, "output_captured": False, **iso},
                    warnings=[
                        "Output não capturado: o ambiente não permite pipes entre "
                        "processos. O código de saída é real.",
                    ],
                )

            # Truncar output
            max_output = 10000
            if len(stdout) > max_output:
                stdout = stdout[:max_output] + f"\n... [truncado]"
            if len(stderr) > max_output:
                stderr = stderr[:max_output] + f"\n... [truncado]"

            return ToolResult(
                success=success,
                data={
                    "stdout": stdout,
                    "stderr": stderr,
                    "return_code": return_code,
                    "output_captured": True,
                    **iso,
                },
                error=stderr if not success else None,
                metadata={"return_code": return_code, "output_captured": True, **iso},
            )

        return ToolResult(
            success=False,
            error="Não foi possível iniciar o processo (tentativas esgotadas).",
            metadata={"spawn_failed": True},
        )

    async def _analyze(self, code: str) -> ToolResult:
        """Analisa código com AST."""
        try:
            tree = ast.parse(code)

            analysis = {
                "valid_syntax": True,
                "imports": [],
                "classes": [],
                "functions": [],
                "issues": [],
                "metrics": {},
            }

            # Recolher imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        analysis["imports"].append(
                            f"{module}.{alias.name}" if module else alias.name
                        )

            # Recolher classes e funções
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [
                        n.name for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]
                    analysis["classes"].append({
                        "name": node.name,
                        "methods": methods,
                        "n_methods": len(methods),
                        "lineno": node.lineno,
                    })
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    analysis["functions"].append({
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "n_args": len(node.args.args),
                        "lineno": node.lineno,
                    })

            # Métricas
            analysis["metrics"] = {
                "n_lines": len(code.split("\n")),
                "n_imports": len(analysis["imports"]),
                "n_classes": len(analysis["classes"]),
                "n_functions": len(analysis["functions"]),
            }

            return ToolResult(success=True, data=analysis)

        except SyntaxError as e:
            return ToolResult(
                success=False,
                error=f"Erro de sintaxe: {e}",
                data={"valid_syntax": False, "error": str(e)},
            )

    async def _format(self, code: str) -> ToolResult:
        """Tenta formatar código."""
        try:
            import black
            formatted = black.format_str(code, mode=black.Mode())
            return ToolResult(
                success=True,
                data={
                    "formatted": formatted,
                    "original_length": len(code),
                    "formatted_length": len(formatted),
                },
            )
        except ImportError:
            # Fallback: auto-indentação básica
            lines = code.split("\n")
            formatted_lines = []
            indent = 0
            for line in lines:
                stripped = line.strip()
                if stripped.startswith((")", "}", "]")):
                    indent = max(0, indent - 1)
                formatted_lines.append("    " * indent + stripped)
                if stripped.endswith(("{", "(", "[")):
                    indent += 1
            formatted = "\n".join(formatted_lines)
            return ToolResult(
                success=True,
                data={"formatted": formatted, "note": "Usou formatação básica (black não instalado)"},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Erro na formatação: {e}")

    def _check_dangerous_imports(self, code: str) -> list[str]:
        """Verifica se o código importa módulos perigosos."""
        dangerous = []
        for module in self._DANGEROUS_MODULES:
            if f"import {module}" in code or f"from {module}" in code:
                dangerous.append(module)
        return dangerous


# ======================================================================
#  MemoryTool
# ======================================================================

class MemoryTool(BaseTool):
    """
    Ferramenta para aceder ao sistema de memória do Odysseus.

    Capacidades:
        - memory.store: guardar informação na memória permanente
        - memory.recall: recuperar informação da memória
        - memory.list: listar tudo o que está memorizado
        - memory.delete: apagar informação da memória
    """

    def _build_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory",
            description="Aceder ao sistema de memória: guardar, consultar, listar, apagar",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["store", "recall", "list", "delete"],
                        "description": "Ação a executar",
                    },
                    "key": {
                        "type": "string",
                        "description": "Chave da memória",
                    },
                    "value": {
                        "type": "string",
                        "description": "Valor a guardar (apenas store)",
                    },
                    "query": {
                        "type": "string",
                        "description": "Texto a pesquisar nas memórias",
                    },
                },
                "required": ["action"],
            },
            capabilities=["memory.store", "memory.recall", "memory.list", "memory.delete"],
            permissions=["memory.read", "memory.write"],
            category="memory",
            version="1.0.0",
            timeout=10.0,
            requires_approval=False,
            tags=["memory", "knowledge", "persistence"],
        )

    async def execute(self, params: dict, context: Optional[dict] = None) -> ToolResult:
        from .memory import get_memory_manager

        action = params["action"]
        memory = get_memory_manager()

        try:
            if action == "store":
                key = params.get("key", "")
                value = params.get("value", "")
                if not key or not value:
                    return ToolResult(success=False, error="'key' e 'value' são obrigatórios para store")

                memory.request_permanent_storage(key, value)
                return ToolResult(
                    success=True,
                    data={
                        "key": key,
                        "value": value,
                        "status": "pending_approval",
                        "message": "Aguardando aprovação para guardar na memória permanente",
                    },
                    metadata={"action": "store", "key": key},
                )

            elif action == "recall":
                key = params.get("key", "")
                if key:
                    value = memory.recall_permanent(key)
                    if value is not None:
                        return ToolResult(
                            success=True,
                            data={"key": key, "value": value, "found": True},
                        )
                    return ToolResult(
                        success=True,
                        data={"key": key, "found": False},
                    )

                # Sem key, pesquisar por query
                query = params.get("query", "")
                all_memory = memory.recall_all_permanent()
                if query:
                    results = {
                        k: v for k, v in all_memory.items()
                        if query.lower() in k.lower() or query.lower() in str(v).lower()
                    }
                else:
                    results = all_memory

                return ToolResult(
                    success=True,
                    data={"results": results, "count": len(results)},
                )

            elif action == "list":
                all_memory = memory.recall_all_permanent()
                return ToolResult(
                    success=True,
                    data={
                        "entries": [
                            {"key": k, "value_preview": str(v)[:100]}
                            for k, v in all_memory.items()
                        ],
                        "count": len(all_memory),
                    },
                )

            elif action == "delete":
                key = params.get("key", "")
                if not key:
                    return ToolResult(success=False, error="'key' é obrigatório para delete")

                exists = memory.recall_permanent(key) is not None
                if not exists:
                    return ToolResult(
                        success=True,
                        data={"key": key, "deleted": False, "reason": "Não encontrado"},
                    )

                # Memory manager vai pedir aprovação
                return ToolResult(
                    success=True,
                    data={
                        "key": key,
                        "status": "pending_approval",
                        "message": f"Aguardando aprovação para apagar '{key}'",
                    },
                )

        except Exception as e:
            return ToolResult(success=False, error=f"Erro no MemoryTool: {e}")

        return ToolResult(success=False, error=f"Ação desconhecida: {action}")


# ======================================================================
#  SearchTool — Pesquisa semântica no workspace
# ======================================================================

class SearchTool(BaseTool):
    """
    Ferramenta para pesquisa de ficheiros e conteúdo no workspace.

    Capacidades:
        - search.files: pesquisar ficheiros por nome/pattern
        - search.content: pesquisar texto no conteúdo de ficheiros
        - search.code: pesquisar símbolos em código fonte
    """

    def _build_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search",
            description="Pesquisa de ficheiros, conteúdo e símbolos no workspace",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["files", "content", "code"],
                        "description": "Tipo de pesquisa",
                    },
                    "query": {
                        "type": "string",
                        "description": "Texto a pesquisar",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern para filtrar ficheiros (ex: '*.py')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Máximo de resultados",
                        "default": 30,
                    },
                    "directory": {
                        "type": "string",
                        "description": "Diretório base para pesquisa (relativo ao workspace)",
                        "default": ".",
                    },
                },
                "required": ["action", "query"],
            },
            capabilities=["search.files", "search.content", "search.code"],
            permissions=["filesystem.read"],
            category="search",
            version="1.0.0",
            timeout=30.0,
            tags=["search", "grep", "filesystem"],
        )

    async def execute(self, params: dict, context: Optional[dict] = None) -> ToolResult:
        action = params["action"]
        query = params.get("query", "")
        pattern = params.get("pattern", "*")
        max_results = params.get("max_results", 30)
        directory = params.get("directory", ".")

        # Determinar e validar o diretório base da pesquisa.
        # Com workspace explícito a contenção é estrita; sem ele mantém-se a
        # base anterior, mas nunca são alcançadas áreas protegidas.
        base_dir = self._get_base_dir(context) or V2_PROJECT_ROOT
        has_workspace = bool(context and context.get("workspace_path"))
        search_dir = _resolve_search_target(directory, base_dir, confine=has_workspace)
        if search_dir is None:
            return ToolResult(
                success=False,
                error=(
                    "Caminho de pesquisa inválido, fora da área autorizada "
                    f"ou protegido: '{directory}'"
                ),
            )
        if not search_dir.exists():
            return ToolResult(success=False, error=f"Diretório não encontrado: {directory}")

        try:
            if action == "files":
                return await self._search_files(search_dir, query, pattern, max_results)
            elif action == "content":
                return await self._search_content(search_dir, query, pattern, max_results)
            elif action == "code":
                return await self._search_code(search_dir, query, pattern, max_results)
            else:
                return ToolResult(success=False, error=f"Ação desconhecida: {action}")

        except Exception as e:
            return ToolResult(success=False, error=f"Erro na pesquisa: {e}")

    def _get_base_dir(self, context: Optional[dict]) -> Optional[Path]:
        if context and "workspace_path" in context:
            return Path(context["workspace_path"]).resolve()
        project_root = Path(__file__).resolve().parent.parent.parent
        return project_root

    async def _search_files(self, search_dir: Path, query: str, pattern: str, max_results: int) -> ToolResult:
        """Pesquisa ficheiros por nome."""
        results = []
        query_lower = query.lower()

        for fpath in search_dir.rglob(pattern):
            if len(results) >= max_results:
                break
            if fpath.is_file() and query_lower in fpath.name.lower():
                try:
                    rel_path = str(fpath.relative_to(search_dir))
                except ValueError:
                    rel_path = str(fpath)
                results.append({
                    "name": fpath.name,
                    "path": rel_path,
                    "size_bytes": fpath.stat().st_size,
                })

        return ToolResult(
            success=True,
            data={"query": query, "results": results, "count": len(results)},
        )

    async def _search_content(self, search_dir: Path, query: str, pattern: str, max_results: int) -> ToolResult:
        """Pesquisa texto no conteúdo de ficheiros."""
        results = []
        query_lower = query.lower()

        for fpath in search_dir.rglob(pattern):
            if len(results) >= max_results:
                break
            if not fpath.is_file():
                continue
            if fpath.stat().st_size > 500000:  # Ignorar ficheiros > 500KB
                continue

            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if query_lower in line.lower():
                            try:
                                rel_path = str(fpath.relative_to(search_dir))
                            except ValueError:
                                rel_path = str(fpath)
                            results.append({
                                "file": rel_path,
                                "line": i,
                                "content": line.strip()[:200],
                            })
                            if len(results) >= max_results:
                                break
            except Exception:
                continue

        return ToolResult(
            success=True,
            data={"query": query, "results": results, "count": len(results)},
        )

    async def _search_code(self, search_dir: Path, query: str, pattern: str, max_results: int) -> ToolResult:
        """Pesquisa símbolos em código fonte."""
        results = []
        code_pattern = pattern if pattern != "*" else "*.py"

        for fpath in search_dir.rglob(code_pattern):
            if len(results) >= max_results:
                break
            if not fpath.is_file():
                continue

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if query.lower() in node.name.lower():
                            try:
                                rel_path = str(fpath.relative_to(search_dir))
                            except ValueError:
                                rel_path = str(fpath)
                            results.append({
                                "type": "function",
                                "name": node.name,
                                "file": rel_path,
                                "line": node.lineno,
                            })
                            if len(results) >= max_results:
                                break
                    elif isinstance(node, ast.ClassDef):
                        if query.lower() in node.name.lower():
                            try:
                                rel_path = str(fpath.relative_to(search_dir))
                            except ValueError:
                                rel_path = str(fpath)
                            results.append({
                                "type": "class",
                                "name": node.name,
                                "file": rel_path,
                                "line": node.lineno,
                            })
                            if len(results) >= max_results:
                                break
            except (SyntaxError, UnicodeDecodeError):
                continue
            except Exception:
                continue

        return ToolResult(
            success=True,
            data={"query": query, "results": results, "count": len(results)},
        )


# ======================================================================
#  CodeTool — Análise de código com AST
# ======================================================================

class CodeTool(BaseTool):
    """
    Ferramenta para análise de código fonte.

    Capacidades:
        - code.analyze: análise AST completa de ficheiro
        - code.metrics: métricas de código
        - code.validate: validação de sintaxe
    """

    def _build_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="code",
            description="Análise de código fonte com AST, métricas e validação",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["analyze", "metrics", "validate"],
                        "description": "Ação a executar",
                    },
                    "file": {
                        "type": "string",
                        "description": "Caminho do ficheiro a analisar",
                    },
                    "code": {
                        "type": "string",
                        "description": "Código para análise (se não fornecer file)",
                    },
                    "language": {
                        "type": "string",
                        "description": "Linguagem (auto-detetada se omitida)",
                        "default": "python",
                    },
                },
                "required": ["action"],
            },
            capabilities=["code.analyze", "code.metrics", "code.validate"],
            permissions=["filesystem.read"],
            category="code",
            version="1.0.0",
            timeout=15.0,
            tags=["code", "analysis", "ast", "metrics"],
        )

    async def execute(self, params: dict, context: Optional[dict] = None) -> ToolResult:
        action = params["action"]
        file_path = params.get("file", "")
        code = params.get("code", "")
        language = params.get("language", "python")

        # Se file foi fornecido, ler conteúdo
        if file_path and not code:
            file_tool = FileTool()
            read_result = await file_tool.execute(
                {"action": "read", "path": file_path}, context=context
            )
            if not read_result.success:
                return read_result
            code = read_result.data.get("content", "")
            if not code:
                return ToolResult(success=False, error=f"Não foi possível ler '{file_path}'")

        if not code and not file_path:
            return ToolResult(success=False, error="Fornece 'file' ou 'code' para análise")

        if action == "validate":
            return self._validate(code, language)
        elif action == "metrics":
            return self._metrics(code, file_path or "snippet")
        elif action == "analyze":
            result = self._analyze(code, file_path or "snippet")
            return result

        return ToolResult(success=False, error=f"Ação desconhecida: {action}")

    def _validate(self, code: str, language: str) -> ToolResult:
        """Valida sintaxe do código."""
        if language != "python":
            return ToolResult(
                success=True,
                data={"valid": True, "note": f"Validação apenas para Python (linguagem: {language})"},
            )

        try:
            ast.parse(code)
            return ToolResult(
                success=True,
                data={"valid": True, "language": "python"},
            )
        except SyntaxError as e:
            return ToolResult(
                success=False,
                error=f"Erro de sintaxe: {e}",
                data={"valid": False, "error": str(e), "language": "python"},
            )

    def _metrics(self, code: str, filename: str) -> ToolResult:
        """Calcula métricas de código."""
        lines = code.split("\n")
        total_lines = len(lines)
        blank_lines = sum(1 for l in lines if not l.strip())
        comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
        code_lines = total_lines - blank_lines - comment_lines

        metrics = {
            "file": filename,
            "total_lines": total_lines,
            "code_lines": code_lines,
            "blank_lines": blank_lines,
            "comment_lines": comment_lines,
            "size_bytes": len(code),
        }

        # Métricas Python específicas
        try:
            tree = ast.parse(code)
            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            functions = [
                n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            imports = [
                n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
            ]

            metrics.update({
                "classes": len(classes),
                "functions": len(functions),
                "imports": len(imports),
            })
        except SyntaxError:
            pass

        return ToolResult(success=True, data=metrics)

    def _analyze(self, code: str, filename: str) -> ToolResult:
        """Análise AST completa."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ToolResult(
                success=False,
                error=f"Erro de sintaxe: {e}",
                data={"valid_syntax": False, "error": str(e)},
            )

        analysis = {
            "file": filename,
            "valid_syntax": True,
            "imports": [],
            "classes": [],
            "functions": [],
            "issues": [],
            "metrics": {
                "total_lines": len(code.split("\n")),
            },
        }

        for node in ast.walk(tree):
            # Imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    analysis["imports"].append({"module": alias.name, "alias": alias.asname})

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    analysis["imports"].append({
                        "module": f"{module}.{alias.name}" if module else alias.name,
                        "alias": alias.asname,
                    })

            # Classes
            elif isinstance(node, ast.ClassDef):
                methods = [
                    n.name for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                class_info = {
                    "name": node.name,
                    "methods": methods,
                    "n_methods": len(methods),
                    "line": node.lineno,
                }
                # Docstring
                docstring = ast.get_docstring(node)
                if docstring:
                    class_info["docstring"] = docstring[:200]
                analysis["classes"].append(class_info)

            # Funções
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = {
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "n_args": len(node.args.args),
                    "line": node.lineno,
                }
                docstring = ast.get_docstring(node)
                if docstring:
                    func_info["docstring"] = docstring[:200]
                analysis["functions"].append(func_info)

        # Métricas finais
        analysis["metrics"].update({
            "n_imports": len(analysis["imports"]),
            "n_classes": len(analysis["classes"]),
            "n_functions": len(analysis["functions"]),
        })

        return ToolResult(success=True, data=analysis)


# ======================================================================
#  GitTool — Operações Git Seguras
# ======================================================================

class GitTool(BaseTool):
    """
    Ferramenta para operações git seguras.

    Todas as operações são **read-only** — não modificam o repositório.
    Útil para inspecionar o estado do projeto, histórico, branches,
    e alterações sem correr riscos.

    Capacidades:
        - git.status: estado atual do working tree
        - git.log: histórico de commits (com limite configurável)
        - git.diff: diferenças entre working tree, staging, ou commits
        - git.branch: listar branches
        - git.show: detalhes de um commit específico
        - git.stash_list: listar entradas do stash
        - git.remotes: listar remotes configurados
        - git.shortlog: sumário de commits por autor

    Segurança:
        - Comandos git construídos literalmente (sem shell=True)
        - Argumentos passados como lista — sem risk de injection
        - Apenas operações de leitura (status, log, diff, branch)
        - Timeout configurável
        - Output truncado para evitar sobrecarga
    """

    # Comandos git read-only mapeados por ação
    _GIT_COMMANDS = {
        "status":       ["git", "status", "--short", "--branch"],
        "log":          ["git", "log", "--oneline", "--graph", "--decorate", "-30"],
        "diff":         ["git", "diff", "--stat"],
        "diff_staged":  ["git", "diff", "--cached", "--stat"],
        "branch":       ["git", "branch", "-a", "-v"],
        "show":         ["git", "show", "--stat", "--no-patch"],
        "stash_list":   ["git", "stash", "list"],
        "remotes":      ["git", "remote", "-v"],
        "shortlog":     ["git", "shortlog", "-sn", "--all", "-20"],
        "tags":         ["git", "tag", "--sort=-v:refname", "-n"],
    }

    def _build_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="git",
            description=(
                "Operações git seguras e read-only: status, log, diff, "
                "branch, show, stash, remotes, shortlog, tags"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "status", "log", "diff", "diff_staged",
                            "branch", "show", "stash_list",
                            "remotes", "shortlog", "tags",
                        ],
                        "description": "Operação git a executar",
                    },
                    "path": {
                        "type": "string",
                        "description": "Caminho do repositório git (ou subdiretório dentro dele)",
                        "default": ".",
                    },
                    "args": {
                        "type": "string",
                        "description": (
                            "Argumentos adicionais para o comando git "
                            "(ex: para 'log': '--since=2024-01-01'; "
                            "para 'diff': 'HEAD~1'; para 'show': 'abc123'; "
                            "para 'branch': '--merged')"
                        ),
                    },
                    "max_count": {
                        "type": "integer",
                        "description": "Número máximo de commits/resultados (apenas log e shortlog)",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout em segundos",
                        "default": 15.0,
                    },
                },
                "required": ["action"],
            },
            capabilities=[
                "git.status", "git.log", "git.diff",
                "git.diff_staged", "git.branch", "git.show",
                "git.stash_list", "git.remotes", "git.shortlog",
                "git.tags",
            ],
            permissions=["filesystem.read"],
            category="git",
            version="1.0.0",
            timeout=30.0,
            requires_approval=False,
            tags=["git", "vcs", "version-control", "development"],
        )

    async def execute(self, params: dict, context: Optional[dict] = None) -> ToolResult:
        action = params["action"]
        path = params.get("path", ".")
        extra_args = params.get("args", "")
        max_count = params.get("max_count")
        timeout = params.get("timeout", 15.0)

        # ── 1. Resolver caminho do repositório ──
        repo_path = self._resolve_repo_path(path, context)
        if not repo_path:
            return ToolResult(
                success=False,
                error=(
                    f"Caminho inválido ou fora do projeto: '{path}'. "
                    "Fornece um caminho dentro do projeto Odysseus."
                ),
            )

        # ── 2. Verificar se é um repositório git ──
        git_dir = repo_path / ".git"
        if not git_dir.exists():
            # Tentar encontrar o .git num pai
            parent_git = self._find_git_root(repo_path)
            if parent_git:
                repo_path = parent_git
            else:
                return ToolResult(
                    success=False,
                    error=f"'{repo_path}' não está num repositório git",
                )

        # ── 3. Construir comando ──
        cmd = self._build_command(action, extra_args, max_count)
        if cmd is None:
            return ToolResult(
                success=False,
                error=f"Ação desconhecida: '{action}'. "
                       "Usa: status, log, diff, diff_staged, branch, "
                       "show, stash_list, remotes, shortlog, tags",
            )

        # ── 4. Executar ──
        try:
            import asyncio

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(repo_path),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )

            stdout_decoded = stdout.decode("utf-8", errors="replace")
            stderr_decoded = stderr.decode("utf-8", errors="replace")

            success = proc.returncode == 0

            # Truncar output grande
            max_output = 30000
            if len(stdout_decoded) > max_output:
                stdout_decoded = (
                    stdout_decoded[:max_output]
                    + f"\n... [truncado, {len(stdout_decoded)} caracteres no total]"
                )

            if not success:
                return ToolResult(
                    success=False,
                    error=stderr_decoded or f"git {action} falhou (código {proc.returncode})",
                    data={"stderr": stderr_decoded, "return_code": proc.returncode},
                    metadata={"action": action, "repo": str(repo_path)},
                )

            # Parsear output para ações específicas
            parsed = self._parse_output(action, stdout_decoded)

            return ToolResult(
                success=True,
                data={
                    "action": action,
                    "repo": str(repo_path),
                    "raw": stdout_decoded,
                    **parsed,
                },
                metadata={
                    "action": action,
                    "repo": str(repo_path),
                    "lines": stdout_decoded.count("\n"),
                },
            )

        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"Comando git excedeu timeout de {timeout}s",
                metadata={"timeout": timeout},
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="Git não encontrado. Instala o git: https://git-scm.com",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Erro ao executar git {action}: {e}",
                metadata={"action": action},
            )

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _resolve_repo_path(self, path: str, context: Optional[dict]) -> Optional[Path]:
        """Resolve o caminho para o repositório de forma segura.

        Aplica a mesma fronteira do FileTool: contenção real (normcase +
        commonpath) e negação das áreas protegidas (Brain Vault, Mind OS,
        data/, logs/, .git). Com workspace no contexto, a base é o workspace;
        sem ele, a raiz do projecto — que continua a ser o caso de uso normal
        desta ferramenta.
        """
        base = V2_PROJECT_ROOT
        if context and context.get("workspace_path"):
            base = Path(context["workspace_path"]).resolve()
        return _resolve_confined(path, base)

    def _find_git_root(self, start: Path) -> Optional[Path]:
        """Sobe diretórios até encontrar um .git."""
        current = start.resolve()
        project_root = Path(__file__).resolve().parent.parent.parent
        max_parent = project_root.parent

        while current != current.parent:
            if (current / ".git").exists():
                return current
            if current == max_parent:
                break
            current = current.parent

        # Tentar project_root como fallback
        if (project_root / ".git").exists():
            return project_root
        return None

    def _build_command(self, action: str, extra_args: str, max_count: Optional[int]) -> Optional[list[str]]:
        """Constrói a lista de argumentos para o comando git."""
        base = self._GIT_COMMANDS.get(action)
        if base is None:
            return None

        cmd = list(base)  # copiar

        # Aplicar max_count para log e shortlog
        if max_count is not None and action in ("log", "shortlog"):
            # Substituir o -30 ou -20 pelo valor fornecido
            for i, arg in enumerate(cmd):
                if arg.startswith("-") and arg[1:].isdigit():
                    cmd[i] = f"-{max_count}"
                    break

        # Adicionar argumentos extras
        if extra_args:
            # Split seguro — preserva quotes
            import shlex
            try:
                extra = shlex.split(extra_args)
                cmd.extend(extra)
            except ValueError:
                # Se houver erro de parsing, usar split simples
                cmd.extend(extra_args.split())

        return cmd

    def _parse_output(self, action: str, output: str) -> dict:
        """Parseia o output raw do git para dados estruturados."""
        lines = [l for l in output.split("\n") if l.strip()]

        if action == "status":
            return self._parse_status(lines, output)
        elif action == "branch":
            return self._parse_branches(lines)
        elif action == "remotes":
            return self._parse_remotes(lines)
        elif action == "shortlog":
            return self._parse_shortlog(lines)
        elif action == "stash_list":
            return {"stashes": lines, "count": len(lines)}
        elif action == "tags":
            return self._parse_tags(lines)
        elif action in ("log",):
            return {"commits": lines, "count": len(lines)}
        else:
            return {"lines": len(lines), "chars": len(output)}

    def _parse_status(self, lines: list[str], raw: str) -> dict:
        """Parseia git status --short --branch."""
        branch_info = {}
        changes = []
        summary = {
            "staged": 0,
            "unstaged": 0,
            "untracked": 0,
            "conflicts": 0,
        }

        for line in lines:
            if line.startswith("##"):
                # ## main...origin/main [ahead 1, behind 2]
                parts = line.replace("## ", "").split("...")
                branch_info["branch"] = parts[0]
                if len(parts) > 1:
                    ahead_behind = parts[1].strip()
                    branch_info["remote_tracking"] = ahead_behind
                    if "ahead" in ahead_behind:
                        m = re.search(r"ahead (\d+)", ahead_behind)
                        if m:
                            branch_info["ahead"] = int(m.group(1))
                    if "behind" in ahead_behind:
                        m = re.search(r"behind (\d+)", ahead_behind)
                        if m:
                            branch_info["behind"] = int(m.group(1))
            else:
                # Códigos: XY filename
                xy = line[:2]
                file_path = line[3:].strip()

                entry = {"file": file_path, "code": xy}

                if xy == "??":
                    entry["type"] = "untracked"
                    summary["untracked"] += 1
                elif xy.strip() and " " not in xy:
                        # XY estranho (ex: "MM", "AU")
                        entry["type"] = "unknown"
                else:
                    staged = xy[0]
                    unstaged = xy[1]
                    entry["staged"] = staged
                    entry["unstaged"] = unstaged
                    if staged != " " and staged != "?":
                        summary["staged"] += 1
                    if unstaged != " " and unstaged != "?":
                        summary["unstaged"] += 1
                    if staged == "U" or unstaged == "U":
                        summary["conflicts"] += 1

                changes.append(entry)

        result = {
            "branch": branch_info.get("branch", "desconhecido"),
            "branch_info": branch_info,
            "changes": changes,
            "summary": summary,
            "is_clean": summary["staged"] + summary["unstaged"] + summary["untracked"] == 0,
        }

        return result

    def _parse_branches(self, lines: list[str]) -> dict:
        """Parseia git branch -a -v."""
        branches = []
        current = None

        for line in lines:
            is_current = line.startswith("*")
            name_part = line[2:].strip() if is_current else line.strip()

            # Separar nome do commit hash/mensagem
            parts = name_part.split(None, 2)
            if len(parts) >= 2:
                name = parts[0]
                commit_hash = parts[1]
                commit_msg = parts[2] if len(parts) > 2 else ""
            else:
                name = name_part
                commit_hash = ""
                commit_msg = ""

            entry = {
                "name": name,
                "current": is_current,
                "commit": commit_hash,
                "message": commit_msg,
            }
            if is_current:
                current = name
            branches.append(entry)

        return {
            "branches": branches,
            "count": len(branches),
            "current": current,
            "is_detached": current is None and len(branches) > 0,
        }

    def _parse_remotes(self, lines: list[str]) -> dict:
        """Parseia git remote -v."""
        remotes = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                url = parts[1]
                if name not in remotes:
                    remotes[name] = {}
                # Determinar se é fetch ou push
                if len(parts) >= 3:
                    direction = parts[-1].strip("()")
                    remotes[name][direction] = url
                else:
                    remotes[name]["url"] = url

        return {"remotes": remotes, "count": len(remotes)}

    def _parse_shortlog(self, lines: list[str]) -> dict:
        """Parseia git shortlog -sn."""
        authors = []
        for line in lines:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                try:
                    count = int(parts[0])
                    name = parts[1]
                    authors.append({"name": name, "commits": count})
                except ValueError:
                    pass

        return {"authors": authors, "count": len(authors), "total_commits": sum(a["commits"] for a in authors)}

    def _parse_tags(self, lines: list[str]) -> dict:
        """Parseia git tag -n."""
        tags = []
        for line in lines:
            parts = line.strip().split(None, 1)
            if parts:
                tag_name = parts[0]
                message = parts[1] if len(parts) > 1 else ""
                tags.append({"name": tag_name, "message": message})

        return {"tags": tags, "count": len(tags)}


# ======================================================================
#  Registry Helper
# ======================================================================

def register_base_tools(registry, *, only_missing: bool = False) -> list[str]:
    """
    Regista todas as ferramentas base no registry.

    Args:
        registry: ToolRegistry instance.
        only_missing: Regista apenas ferramentas ausentes, preservando
            implementações já configuradas com o mesmo nome.

    Returns:
        Lista de nomes das ferramentas registadas.
    """
    tools = [
        WebTool(),
        FileTool(),
        ShellTool(),
        PythonTool(),
        SearchTool(),
        CodeTool(),
        GitTool(),
    ]

    registered = []
    for tool in tools:
        if only_missing and registry.get_tool(tool.name) is not None:
            continue
        registry.register(tool)
        registered.append(tool.name)

    # BrowserTool — import lazy para evitar que a falta de dependências
    # (playwright, bs4) quebre o registo das outras ferramentas
    try:
        from .browser_tool import BrowserTool
        browser_tool = BrowserTool()
        if not only_missing or registry.get_tool(browser_tool.name) is None:
            registry.register(browser_tool)
            registered.append(browser_tool.name)
            logger.info("BrowserTool registado")
    except Exception as e:
        logger.warning("BrowserTool não disponível: %s", e)

    logger.info("Ferramentas base registadas: %s", ", ".join(registered))
    return registered
