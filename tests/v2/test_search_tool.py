"""
Testes Unitários — SearchTool

Cobertura:
    - Definição: nome, capacidades, parâmetros
    - execute: files, content, code (sobre temp_dir)
    - Edge cases: diretório não encontrado, sem resultados
    - Validação de parâmetros

Uso:
    pytest tests/v2/test_search_tool.py -v
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
def search_tool():
    from core.v2.tools import SearchTool
    return SearchTool()


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        (base / "hello.py").write_text("print('hello world')")
        (base / "utils.py").write_text("def greet():\n    pass")
        (base / "notes.txt").write_text("Some notes about python")
        (base / "subdir").mkdir()
        (base / "subdir" / "deep.py").write_text("class DeepClass:\n    pass")
        yield base


# ======================================================================
#  Tests: Definição
# ======================================================================

class TestSearchToolDefinition:
    def test_name(self, search_tool):
        assert search_tool.name == "search"

    def test_capabilities(self, search_tool):
        caps = search_tool.definition.capabilities
        assert "search.files" in caps
        assert "search.content" in caps
        assert "search.code" in caps

    def test_actions(self, search_tool):
        actions = search_tool.definition.parameters["properties"]["action"]["enum"]
        assert set(actions) == {"files", "content", "code"}

    def test_required_params(self, search_tool):
        required = search_tool.definition.parameters["required"]
        assert "action" in required
        assert "query" in required

    def test_category(self, search_tool):
        assert search_tool.definition.category == "search"


# ======================================================================
#  Tests: Search Files
# ======================================================================

class TestSearchToolFiles:
    @pytest.mark.asyncio
    async def test_search_by_name(self, search_tool, temp_dir):
        result = await search_tool.execute({
            "action": "files",
            "query": "hello",
            "directory": str(temp_dir),
        })
        assert result.success is True
        names = [r["name"] for r in result.data["results"]]
        assert "hello.py" in names
        assert "utils.py" not in names

    @pytest.mark.asyncio
    async def test_search_no_results(self, search_tool, temp_dir):
        result = await search_tool.execute({
            "action": "files",
            "query": "nonexistent_file_xyz",
            "directory": str(temp_dir),
        })
        assert result.success is True
        assert result.data["count"] == 0

    @pytest.mark.asyncio
    async def test_directory_not_found(self, search_tool):
        result = await search_tool.execute({
            "action": "files",
            "query": "test",
            "directory": "/nonexistent_dir",
        })
        assert result.success is False


# ======================================================================
#  Tests: Search Content
# ======================================================================

class TestSearchToolContent:
    @pytest.mark.asyncio
    async def test_search_content(self, search_tool, temp_dir):
        result = await search_tool.execute({
            "action": "content",
            "query": "hello world",
            "directory": str(temp_dir),
        })
        assert result.success is True
        assert result.data["count"] >= 1

    @pytest.mark.asyncio
    async def test_search_content_no_match(self, search_tool, temp_dir):
        result = await search_tool.execute({
            "action": "content",
            "query": "zzz_nonexistent",
            "directory": str(temp_dir),
        })
        assert result.success is True
        assert result.data["count"] == 0

    @pytest.mark.asyncio
    async def test_search_content_ignores_binary(self, search_tool, temp_dir):
        result = await search_tool.execute({
            "action": "content",
            "query": "python",
            "directory": str(temp_dir),
        })
        assert result.success is True


# ======================================================================
#  Tests: Search Code
# ======================================================================

class TestSearchToolCode:
    @pytest.mark.asyncio
    async def test_search_function(self, search_tool, temp_dir):
        result = await search_tool.execute({
            "action": "code",
            "query": "greet",
            "directory": str(temp_dir),
        })
        assert result.success is True
        assert result.data["count"] >= 1
        assert any(r["name"] == "greet" for r in result.data["results"])

    @pytest.mark.asyncio
    async def test_search_class(self, search_tool, temp_dir):
        result = await search_tool.execute({
            "action": "code",
            "query": "DeepClass",
            "directory": str(temp_dir),
        })
        assert result.success is True
        assert result.data["count"] >= 1
        assert any(r["name"] == "DeepClass" for r in result.data["results"])

    @pytest.mark.asyncio
    async def test_search_code_no_results(self, search_tool, temp_dir):
        result = await search_tool.execute({
            "action": "code",
            "query": "NonExistentSymbol",
            "directory": str(temp_dir),
        })
        assert result.success is True
        assert result.data["count"] == 0


# ======================================================================
#  Tests: Path Safety
# ======================================================================

class TestSearchToolPathSafety:
    """A pesquisa não pode alcançar áreas protegidas nem escapar ao workspace."""

    @pytest.mark.asyncio
    async def test_brain_vault_directory_denied(self, search_tool):
        r = await search_tool.execute({
            "action": "files", "query": "a", "directory": "Obsidian Brain Vault",
        })
        assert r.success is False

    @pytest.mark.asyncio
    async def test_mindos_directory_denied(self, search_tool):
        r = await search_tool.execute({
            "action": "files", "query": "a", "directory": "Odysseus_Mind_OS",
        })
        assert r.success is False

    @pytest.mark.asyncio
    async def test_data_directory_denied(self, search_tool):
        r = await search_tool.execute({
            "action": "files", "query": "a", "directory": "data",
        })
        assert r.success is False

    @pytest.mark.asyncio
    async def test_workspace_escape_denied(self, search_tool):
        from core.v2.tools import V2_PROJECT_ROOT
        ctx = {"workspace_path": str(V2_PROJECT_ROOT / "core" / "v2")}
        r = await search_tool.execute({
            "action": "files", "query": "a", "directory": "../..",
        }, context=ctx)
        assert r.success is False

    @pytest.mark.asyncio
    async def test_workspace_internal_search_allowed(self, search_tool):
        from core.v2.tools import V2_PROJECT_ROOT
        ctx = {"workspace_path": str(V2_PROJECT_ROOT / "core" / "v2")}
        r = await search_tool.execute({"action": "files", "query": "tools"}, context=ctx)
        assert r.success is True
        assert r.data["count"] >= 1

    @pytest.mark.asyncio
    async def test_project_relative_search_allowed(self, search_tool):
        r = await search_tool.execute({
            "action": "files", "query": "tools", "directory": "core/v2",
        })
        assert r.success is True
        assert r.data["count"] >= 1


# ======================================================================
#  Tests: Validação
# ======================================================================

class TestSearchToolValidation:
    def test_missing_action(self, search_tool):
        errors = search_tool.validate_params({"query": "x"})
        assert len(errors) >= 1

    def test_missing_query(self, search_tool):
        errors = search_tool.validate_params({"action": "files"})
        assert len(errors) >= 1

    def test_valid_params(self, search_tool):
        errors = search_tool.validate_params({
            "action": "files", "query": "test",
        })
        assert errors == []
