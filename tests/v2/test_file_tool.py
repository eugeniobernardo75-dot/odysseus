"""
Testes Unitários — FileTool

Cobertura:
    - Definição: nome, capacidades, parâmetros
    - Operações: read, write, list, search, delete, exists
    - Path safety: path traversal, workspace confinement
    - Edge cases: ficheiro não encontrado, diretório não encontrado

Uso:
    pytest tests/v2/test_file_tool.py -v
"""

from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def file_tool():
    from core.v2.tools import FileTool
    return FileTool()


@pytest.fixture
def temp_dir():
    """Diretório temporário para testes de ficheiros."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ======================================================================
#  Tests: Definição
# ======================================================================

class TestFileToolDefinition:
    def test_name(self, file_tool):
        assert file_tool.name == "file"

    def test_capabilities(self, file_tool):
        caps = file_tool.definition.capabilities
        assert "file.read" in caps
        assert "file.write" in caps
        assert "file.list" in caps
        assert "file.search" in caps
        assert "file.delete" in caps
        assert "file.exists" in caps

    def test_actions(self, file_tool):
        actions = file_tool.definition.parameters["properties"]["action"]["enum"]
        assert "read" in actions
        assert "write" in actions
        assert "list" in actions
        assert "search" in actions
        assert "delete" in actions
        assert "exists" in actions

    def test_required_params(self, file_tool):
        required = file_tool.definition.parameters["required"]
        assert "action" in required
        assert "path" in required

    def test_category(self, file_tool):
        assert file_tool.definition.category == "file"


# ======================================================================
#  Tests: Operações
# ======================================================================

@pytest.mark.asyncio
class TestFileToolOperations:
    """Testes de operações de ficheiro com workspace context."""

    async def test_write_and_read(self, file_tool, temp_dir):
        ctx = {"workspace_path": str(temp_dir)}
        wr = await file_tool.execute({"action": "write", "path": "test.txt", "content": "Hello World\nLine 2"}, context=ctx)
        assert wr.success is True
        rr = await file_tool.execute({"action": "read", "path": "test.txt"}, context=ctx)
        assert rr.success is True
        assert rr.data["content"] == "Hello World\nLine 2"

    async def test_write_creates_dirs(self, file_tool, temp_dir):
        ctx = {"workspace_path": str(temp_dir)}
        r = await file_tool.execute({"action": "write", "path": "sub/nested/f.txt", "content": "t"}, context=ctx)
        assert r.success is True
        assert (temp_dir / "sub" / "nested" / "f.txt").exists()

    async def test_read_not_found(self, file_tool, temp_dir):
        ctx = {"workspace_path": str(temp_dir)}
        r = await file_tool.execute({"action": "read", "path": "nonexistent.txt"}, context=ctx)
        assert r.success is False

    async def test_read_directory(self, file_tool, temp_dir):
        ctx = {"workspace_path": str(temp_dir)}
        r = await file_tool.execute({"action": "read", "path": "."}, context=ctx)
        assert r.success is False

    async def test_list_directory(self, file_tool, temp_dir):
        ctx = {"workspace_path": str(temp_dir)}
        (temp_dir / "a.txt").write_text("a")
        (temp_dir / "b.py").write_text("b")
        (temp_dir / "subdir").mkdir()
        r = await file_tool.execute({"action": "list", "path": "."}, context=ctx)
        assert r.success is True
        names = {f["name"] for f in r.data["files"]}
        assert "a.txt" in names
        assert "subdir" in names

    async def test_list_with_pattern(self, file_tool, temp_dir):
        ctx = {"workspace_path": str(temp_dir)}
        (temp_dir / "s.py").write_text("x")
        (temp_dir / "n.txt").write_text("y")
        r = await file_tool.execute({"action": "list", "path": ".", "pattern": "*.py"}, context=ctx)
        assert r.success is True
        assert r.data["count"] == 1

    async def test_list_nonexistent(self, file_tool):
        r = await file_tool.execute({"action": "list", "path": "/nonexistent"})
        assert r.success is False

    async def test_delete_file(self, file_tool, temp_dir):
        ctx = {"workspace_path": str(temp_dir)}
        (temp_dir / "del.txt").write_text("bye")
        r = await file_tool.execute({"action": "delete", "path": "del.txt"}, context=ctx)
        assert r.success is True
        assert not (temp_dir / "del.txt").exists()

    async def test_delete_nonexistent(self, file_tool, temp_dir):
        ctx = {"workspace_path": str(temp_dir)}
        r = await file_tool.execute({"action": "delete", "path": "ghost.txt"}, context=ctx)
        assert r.success is False

    async def test_delete_directory_refused(self, file_tool, temp_dir):
        ctx = {"workspace_path": str(temp_dir)}
        r = await file_tool.execute({"action": "delete", "path": "."}, context=ctx)
        assert r.success is False

    async def test_exists_true(self, file_tool, temp_dir):
        ctx = {"workspace_path": str(temp_dir)}
        (temp_dir / "ex.txt").write_text("x")
        r = await file_tool.execute({"action": "exists", "path": "ex.txt"}, context=ctx)
        assert r.success is True and r.data["exists"] is True

    async def test_exists_false(self, file_tool, temp_dir):
        ctx = {"workspace_path": str(temp_dir)}
        r = await file_tool.execute({"action": "exists", "path": "missing.txt"}, context=ctx)
        assert r.success is True and r.data["exists"] is False

    async def test_search_content(self, file_tool, temp_dir):
        ctx = {"workspace_path": str(temp_dir)}
        (temp_dir / "hello.py").write_text("print('hello world')")
        r = await file_tool.execute({"action": "search", "path": ".", "query": "hello"}, context=ctx)
        assert r.success is True
        assert r.data["count"] >= 1

    async def test_search_no_results(self, file_tool, temp_dir):
        ctx = {"workspace_path": str(temp_dir)}
        (temp_dir / "data.txt").write_text("nothing")
        r = await file_tool.execute({"action": "search", "path": ".", "query": "zzz"}, context=ctx)
        assert r.success is True and r.data["count"] == 0


# ======================================================================
#  Tests: Path Safety
# ======================================================================
# Estes testes usam ficheiros do próprio repositório e funções puras, para
# não dependerem de escrita em diretórios temporários.

class TestFileToolPathSafety:
    @pytest.mark.asyncio
    async def test_unknown_action(self, file_tool):
        result = await file_tool.execute({
            "action": "unknown_action", "path": "/tmp",
        })
        assert result.success is False

    # ── Contenção de caminhos (funções puras, sem I/O) ──

    def test_containment_accepts_same_and_child(self):
        from core.v2.tools import V2_PROJECT_ROOT, _is_within
        assert _is_within(V2_PROJECT_ROOT, V2_PROJECT_ROOT) is True
        assert _is_within(V2_PROJECT_ROOT / "core" / "v2", V2_PROJECT_ROOT) is True

    def test_containment_rejects_sibling_prefix(self):
        """'...\\odysseus-outro' não pode passar por prefixo textual."""
        from core.v2.tools import V2_PROJECT_ROOT, _is_within
        sibling = V2_PROJECT_ROOT.parent / (V2_PROJECT_ROOT.name + "-outro")
        assert _is_within(sibling, V2_PROJECT_ROOT) is False

    def test_resolve_confined_rejects_escape(self):
        from pathlib import Path
        from core.v2.tools import _resolve_confined
        base = Path(__file__).resolve().parents[1]  # tests/
        assert _resolve_confined("../app.py", base) is None
        assert _resolve_confined("helpers", base) is not None

    def test_protected_detection(self):
        from core.v2.tools import V2_PROJECT_ROOT, _is_protected_path
        assert _is_protected_path(V2_PROJECT_ROOT / "Obsidian Brain Vault" / "x.md") is True
        assert _is_protected_path(V2_PROJECT_ROOT / "Odysseus_Mind_OS" / "x.md") is True
        assert _is_protected_path(V2_PROJECT_ROOT / "data" / "app.db") is True
        assert _is_protected_path(V2_PROJECT_ROOT / "logs" / "app.log") is True
        assert _is_protected_path(V2_PROJECT_ROOT / ".env") is True
        assert _is_protected_path(V2_PROJECT_ROOT / ".git" / "config") is True
        assert _is_protected_path(V2_PROJECT_ROOT / "core" / "v2" / "tools.py") is False

    # ── Áreas protegidas negadas pela ferramenta ──

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rel", [
        "Obsidian Brain Vault/00_SYSTEM/Architecture.md",
        "Odysseus_Mind_OS/Home.md",
        ".env",
        "data/app.db",
        "logs/v2_bridge.log",
        ".git/config",
    ])
    async def test_protected_paths_denied_without_workspace(self, file_tool, rel):
        r = await file_tool.execute({"action": "read", "path": rel})
        assert r.success is False
        assert r.data is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rel", [
        "Obsidian Brain Vault/00_SYSTEM/Architecture.md",
        "Odysseus_Mind_OS/Home.md",
        "data/app.db",
    ])
    async def test_protected_paths_denied_even_with_project_workspace(self, file_tool, rel):
        """A raiz do projecto como workspace não abre as áreas protegidas."""
        from core.v2.tools import V2_PROJECT_ROOT
        ctx = {"workspace_path": str(V2_PROJECT_ROOT)}
        r = await file_tool.execute({"action": "read", "path": rel}, context=ctx)
        assert r.success is False

    @pytest.mark.asyncio
    async def test_write_into_protected_denied(self, file_tool):
        """Escrita em área protegida é negada antes de tocar no disco."""
        r = await file_tool.execute({
            "action": "write",
            "path": "Odysseus_Mind_OS/_probe_should_never_exist.tmp",
            "content": "x",
        })
        assert r.success is False
        assert not (Path(__file__).resolve().parents[2]
                    / "Odysseus_Mind_OS" / "_probe_should_never_exist.tmp").exists()

    # ── Contenção por workspace ──

    @pytest.mark.asyncio
    async def test_workspace_relative_escape_denied(self, file_tool):
        from core.v2.tools import V2_PROJECT_ROOT
        ctx = {"workspace_path": str(V2_PROJECT_ROOT / "core" / "v2")}
        r = await file_tool.execute({"action": "read", "path": "../../app.py"}, context=ctx)
        assert r.success is False

    @pytest.mark.asyncio
    async def test_workspace_absolute_outside_denied(self, file_tool):
        from core.v2.tools import V2_PROJECT_ROOT
        ctx = {"workspace_path": str(V2_PROJECT_ROOT / "core" / "v2")}
        r = await file_tool.execute(
            {"action": "read", "path": str(V2_PROJECT_ROOT / "app.py")}, context=ctx
        )
        assert r.success is False

    @pytest.mark.asyncio
    async def test_workspace_allows_internal_read(self, file_tool):
        from core.v2.tools import V2_PROJECT_ROOT
        ctx = {"workspace_path": str(V2_PROJECT_ROOT / "core" / "v2")}
        r = await file_tool.execute({"action": "read", "path": "tools.py"}, context=ctx)
        assert r.success is True
        assert "V2_PROTECTED_DIRS" in r.data["content"]

    @pytest.mark.asyncio
    async def test_source_read_still_allowed_without_workspace(self, file_tool):
        """A leitura de código do projecto mantém-se (política sem workspace)."""
        r = await file_tool.execute({"action": "read", "path": "core/v2/tools.py"})
        assert r.success is True
        assert "V2_PROJECT_ROOT" in r.data["content"]


# ======================================================================
#  Tests: Capacidade de escrita por propósito (Etapa 5)
# ======================================================================

class TestFileToolPurposeCapabilities:
    """A escrita/eliminação depende do propósito; negada por omissão."""

    def _ctx(self, purpose):
        from core.v2.execution_policy import build_tool_context
        ctx = build_tool_context(purpose)
        assert ctx, purpose
        return ctx

    @pytest.mark.asyncio
    @pytest.mark.parametrize("purpose", ["analysis", "testing", "security_validation"])
    async def test_write_denied_for_read_only_purposes(self, file_tool, purpose):
        result = await file_tool.execute(
            {"action": "write", "path": "probe.txt", "content": "x"},
            context=self._ctx(purpose),
        )
        assert result.success is False
        assert result.metadata.get("reason") == "purpose_read_only"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("purpose", ["analysis", "testing", "security_validation"])
    async def test_delete_denied_for_read_only_purposes(self, file_tool, purpose):
        result = await file_tool.execute(
            {"action": "delete", "path": "probe.txt"}, context=self._ctx(purpose)
        )
        assert result.success is False
        assert result.metadata.get("reason") == "purpose_read_only"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("purpose", ["development", "documentation"])
    async def test_write_not_blocked_by_purpose_when_allowed(self, file_tool, purpose):
        """Com propósito de escrita, o gate do propósito não é o que nega."""
        result = await file_tool.execute(
            {"action": "write", "path": "core/_probe_purpose.tmp", "content": "x"},
            context=self._ctx(purpose),
        )
        assert result.metadata.get("reason") != "purpose_read_only"

    @pytest.mark.asyncio
    async def test_invalid_purpose_denies_write(self, file_tool):
        """Propósito inválido presente no contexto: nega (fail-closed)."""
        result = await file_tool.execute(
            {"action": "write", "path": "probe.txt", "content": "x"},
            context={"workspace_purpose": "root"},
        )
        assert result.success is False
        assert result.metadata.get("reason") == "purpose_read_only"

    @pytest.mark.asyncio
    async def test_read_allowed_under_read_only_purpose(self, file_tool):
        """Ler continua permitido: o propósito restringe escrita, não leitura."""
        ctx = self._ctx("analysis")
        result = await file_tool.execute({"action": "list", "path": "."}, context=ctx)
        assert result.success is True
        assert "files" in result.data

    @pytest.mark.asyncio
    async def test_read_denies_protected_area_under_purpose(self, file_tool):
        """A fronteira de áreas protegidas mantém-se em qualquer propósito."""
        ctx = self._ctx("analysis")
        result = await file_tool.execute(
            {"action": "read", "path": "data/app.db"}, context=ctx
        )
        assert result.success is False


# ======================================================================
#  Tests: Erros
# ======================================================================

class TestFileToolErrors:
    @pytest.mark.asyncio
    async def test_tool_exception(self, file_tool):
        """Exceção na execução é capturada e retorna ToolResult com erro."""
        with patch.object(file_tool, "_read", side_effect=PermissionError("Denied")):
            result = await file_tool.execute({
                "action": "read", "path": "/some/file",
            })
            assert result.success is False
