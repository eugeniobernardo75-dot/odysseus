"""
Workspace — Odysseus V2

Sistema de espaços de trabalho isolados por sessão/projeto.

Cada workspace fornece:
    - Diretório base isolado
    - Subdiretórios para: ficheiros, cache, downloads, outputs, temporários
    - Gestão de ciclo de vida (criar, limpar, destruir)
    - Utilitários para operações seguras de ficheiros
    - Integração com o sistema de segurança

Estrutura típica:
    workspace/
    ├── session_<id>/
    │   ├── files/         # Ficheiros do projeto
    │   ├── cache/         # Cache de execuções
    │   ├── downloads/     # Downloads feitos por ferramentas
    │   ├── outputs/       # Outputs de ferramentas
    │   └── temp/          # Ficheiros temporários
    └── shared/            # Recursos partilhados entre sessões
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ======================================================================
#  Configuração
# ======================================================================

@dataclass
class WorkspaceConfig:
    """Configuração do sistema de workspaces."""
    base_dir: str = ""  # Se vazio, usa temp directory
    max_session_dirs: int = 50
    auto_cleanup_days: int = 7
    max_file_size_mb: int = 50
    allowed_extensions: list[str] = field(default_factory=lambda: [
        ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".json",
        ".md", ".txt", ".csv", ".yaml", ".yml", ".toml", ".ini", ".cfg",
        ".env", ".gitignore", ".sql", ".xml", ".svg", ".png", ".jpg",
        ".jpeg", ".gif", ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ])


# ======================================================================
#  Sessão Workspace
# ======================================================================

class SessionWorkspace:
    """
    Espaço de trabalho isolado para uma sessão.

    Cada sessão recebe um diretório próprio com subdiretórios
    organizados para diferentes tipos de ficheiros.
    """

    def __init__(self, session_id: str, base_path: str):
        self.session_id = session_id
        self.base_path = Path(base_path)
        self._dirs: dict[str, Path] = {}
        self._created_at = datetime.now(timezone.utc)
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Cria a estrutura de diretórios da sessão."""
        subdirs = {
            "root": self.base_path,
            "files": self.base_path / "files",
            "cache": self.base_path / "cache",
            "downloads": self.base_path / "downloads",
            "outputs": self.base_path / "outputs",
            "temp": self.base_path / "temp",
        }
        for name, path in subdirs.items():
            path.mkdir(parents=True, exist_ok=True)
            self._dirs[name] = path

    # ------------------------------------------------------------------
    #  Propriedades
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._dirs["root"]

    @property
    def files_dir(self) -> Path:
        return self._dirs["files"]

    @property
    def cache_dir(self) -> Path:
        return self._dirs["cache"]

    @property
    def downloads_dir(self) -> Path:
        return self._dirs["downloads"]

    @property
    def outputs_dir(self) -> Path:
        return self._dirs["outputs"]

    @property
    def temp_dir(self) -> Path:
        return self._dirs["temp"]

    # ------------------------------------------------------------------
    #  Operações com ficheiros (seguras)
    # ------------------------------------------------------------------

    def resolve_path(self, relative_path: str) -> Path:
        """
        Resolve um caminho relativo dentro do workspace.

        Garante que o caminho não escapa do diretório base (path traversal).
        """
        # Normalizar e resolver
        full_path = (self.base_path / relative_path).resolve()

        # Verificar se está dentro do workspace
        if not str(full_path).startswith(str(self.base_path.resolve())):
            raise PermissionError(
                f"Path traversal detetado: '{relative_path}' escapa do workspace"
            )

        return full_path

    def safe_read(self, relative_path: str, binary: bool = False) -> Optional[str | bytes]:
        """
        Lê um ficheiro dentro do workspace de forma segura.

        Args:
            relative_path: Caminho relativo ao workspace.
            binary: Se True, lê em modo binário.

        Returns:
            Conteúdo do ficheiro ou None se não existir.
        """
        try:
            full_path = self.resolve_path(relative_path)
            if not full_path.exists():
                return None
            if not full_path.is_file():
                return None

            mode = "rb" if binary else "r"
            encoding = None if binary else "utf-8"
            with open(full_path, mode, encoding=encoding, errors="ignore") as f:
                return f.read()

        except Exception as e:
            logger.debug("Workspace: erro ao ler '%s': %s", relative_path, e)
            return None

    def safe_write(self, relative_path: str, content: str | bytes,
                   binary: bool = False) -> bool:
        """
        Escreve um ficheiro dentro do workspace de forma segura.

        Args:
            relative_path: Caminho relativo ao workspace.
            content: Conteúdo a escrever.
            binary: Se True, escreve em modo binário.

        Returns:
            True se escreveu com sucesso.
        """
        try:
            full_path = self.resolve_path(relative_path)
            full_path.parent.mkdir(parents=True, exist_ok=True)

            mode = "wb" if binary else "w"
            encoding = None if binary else "utf-8"
            with open(full_path, mode, encoding=encoding) as f:
                f.write(content)

            logger.debug("Workspace: ficheiro escrito '%s' (%d bytes)",
                         relative_path, len(content))
            return True

        except Exception as e:
            logger.error("Workspace: erro ao escrever '%s': %s", relative_path, e)
            return False

    def safe_delete(self, relative_path: str) -> bool:
        """
        Apaga um ficheiro dentro do workspace de forma segura.

        Args:
            relative_path: Caminho relativo ao workspace.

        Returns:
            True se apagou com sucesso.
        """
        try:
            full_path = self.resolve_path(relative_path)
            if full_path.exists() and full_path.is_file():
                full_path.unlink()
                logger.debug("Workspace: ficheiro apagado '%s'", relative_path)
                return True
            return False
        except Exception as e:
            logger.error("Workspace: erro ao apagar '%s': %s", relative_path, e)
            return False

    def list_files(self, subdir: str = "files", pattern: str = "*") -> list[dict]:
        """
        Lista ficheiros num subdiretório do workspace.

        Args:
            subdir: Subdiretório (files, cache, downloads, outputs, temp).
            pattern: Glob pattern para filtrar.

        Returns:
            Lista de metadados dos ficheiros.
        """
        target_dir = self._dirs.get(subdir, self._dirs["files"])
        files = []
        for fpath in target_dir.glob(pattern):
            if fpath.is_file():
                stat = fpath.stat()
                files.append({
                    "name": fpath.name,
                    "path": str(fpath.relative_to(self.base_path)),
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                })
        return sorted(files, key=lambda f: f["name"])

    # ------------------------------------------------------------------
    #  Limpeza
    # ------------------------------------------------------------------

    def clean_temp(self) -> int:
        """Limpa ficheiros temporários. Retorna número de ficheiros removidos."""
        count = 0
        for item in self.temp_dir.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                    count += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    count += 1
            except Exception:
                pass
        return count

    def clean_cache(self) -> int:
        """Limpa cache da sessão. Retorna número de ficheiros removidos."""
        count = 0
        for item in self.cache_dir.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                    count += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    count += 1
            except Exception:
                pass
        return count

    def destroy(self) -> bool:
        """Remove todo o workspace da sessão."""
        try:
            shutil.rmtree(self.base_path)
            logger.info("Workspace da sessão %s destruído", self.session_id)
            return True
        except Exception as e:
            logger.error("Erro ao destruir workspace %s: %s", self.session_id, e)
            return False

    def get_info(self) -> dict:
        """Retorna informação sobre o workspace."""
        total_size = 0
        file_count = 0
        for root, _, files in os.walk(self.base_path):
            for fname in files:
                try:
                    total_size += (Path(root) / fname).stat().st_size
                    file_count += 1
                except Exception:
                    pass

        return {
            "session_id": self.session_id,
            "base_path": str(self.base_path),
            "created_at": self._created_at.isoformat(),
            "file_count": file_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "directories": list(self._dirs.keys()),
        }


# ======================================================================
#  Gestor de Workspaces
# ======================================================================

class WorkspaceManager:
    """
    Gestor central de workspaces.

    Cria, gere e destrói sessões de workspace.
    Mantém o diretório base e controla o número máximo de sessões.
    """

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        self.config = config or WorkspaceConfig()
        self._sessions: dict[str, SessionWorkspace] = {}

        # Determinar diretório base
        if self.config.base_dir:
            self._base_dir = Path(self.config.base_dir)
        else:
            self._base_dir = Path(tempfile.gettempdir()) / "odysseus_workspaces"

        self._base_dir.mkdir(parents=True, exist_ok=True)
        logger.info("WorkspaceManager: diretório base em %s", self._base_dir)

    # ------------------------------------------------------------------
    #  Gestão de sessões
    # ------------------------------------------------------------------

    def create_session(self, session_id: Optional[str] = None) -> SessionWorkspace:
        """
        Cria um novo workspace para uma sessão.

        Args:
            session_id: ID da sessão (UUID se não fornecido).

        Returns:
            SessionWorkspace pronto a usar.
        """
        sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        session_path = self._base_dir / sid

        # Se já existe, não recriar — apenas retornar
        if sid in self._sessions:
            return self._sessions[sid]

        workspace = SessionWorkspace(session_id=sid, base_path=str(session_path))
        self._sessions[sid] = workspace

        # Limpeza automática se exceder o limite
        if len(self._sessions) > self.config.max_session_dirs:
            self._cleanup_oldest()

        logger.info("WorkspaceManager: sessão '%s' criada em %s", sid, session_path)
        return workspace

    def get_session(self, session_id: str) -> Optional[SessionWorkspace]:
        """Obtém o workspace de uma sessão pelo ID."""
        return self._sessions.get(session_id)

    def destroy_session(self, session_id: str) -> bool:
        """Destrói o workspace de uma sessão."""
        workspace = self._sessions.pop(session_id, None)
        if workspace:
            return workspace.destroy()
        return False

    def list_sessions(self) -> list[dict]:
        """Lista todas as sessões ativas."""
        return [
            {
                "session_id": sid,
                "path": str(ws.base_path),
                "created_at": ws._created_at.isoformat(),
            }
            for sid, ws in self._sessions.items()
        ]

    def _cleanup_oldest(self):
        """Remove a sessão mais antiga quando o limite é excedido."""
        if not self._sessions:
            return
        oldest_id = min(
            self._sessions.keys(),
            key=lambda sid: self._sessions[sid]._created_at,
        )
        logger.warning("WorkspaceManager: a remover sessão mais antiga '%s'", oldest_id)
        self.destroy_session(oldest_id)

    def get_stats(self) -> dict:
        """Estatísticas do gestor de workspaces."""
        return {
            "active_sessions": len(self._sessions),
            "base_dir": str(self._base_dir),
            "max_sessions": self.config.max_session_dirs,
            "sessions": self.list_sessions(),
        }

    def cleanup_all(self) -> int:
        """Destrói todos os workspaces. Retorna número de sessões destruídas."""
        count = len(self._sessions)
        for sid in list(self._sessions.keys()):
            self.destroy_session(sid)
        return count


# ======================================================================
#  Singleton
# ======================================================================

_workspace_manager: Optional[WorkspaceManager] = None


def get_workspace_manager(config: Optional[WorkspaceConfig] = None) -> WorkspaceManager:
    """Retorna a instância global do WorkspaceManager."""
    global _workspace_manager
    if _workspace_manager is None:
        _workspace_manager = WorkspaceManager(config=config)
    return _workspace_manager


def reset_workspace_manager() -> None:
    """Reinicia o WorkspaceManager (útil para testes)."""
    global _workspace_manager
    _workspace_manager = None
