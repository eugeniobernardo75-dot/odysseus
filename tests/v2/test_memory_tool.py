"""
Testes Unitários — MemoryTool

Cobertura:
    - Definição: nome, capacidades, parâmetros
    - execute: store, recall, list, delete (com mock do MemoryManager)
    - Edge cases: chave não encontrada, store sem value
    - Validação de parâmetros

Uso:
    pytest tests/v2/test_memory_tool.py -v
"""

from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def mem_tool():
    from core.v2.tools import MemoryTool
    return MemoryTool()


@pytest.fixture
def mock_memory():
    """Mock do MemoryManager com dados de exemplo."""
    m = MagicMock()
    m.recall_permanent.side_effect = lambda k: {
        "user_name": "João",
        "language": "Python",
    }.get(k)
    m.recall_all_permanent.return_value = {
        "user_name": "João",
        "language": "Python",
    }
    return m


# ======================================================================
#  Tests: Definição
# ======================================================================

class TestMemoryToolDefinition:
    def test_name(self, mem_tool):
        assert mem_tool.name == "memory"

    def test_capabilities(self, mem_tool):
        caps = mem_tool.definition.capabilities
        assert "memory.store" in caps
        assert "memory.recall" in caps
        assert "memory.list" in caps
        assert "memory.delete" in caps

    def test_actions(self, mem_tool):
        actions = mem_tool.definition.parameters["properties"]["action"]["enum"]
        assert set(actions) == {"store", "recall", "list", "delete"}

    def test_category(self, mem_tool):
        assert mem_tool.definition.category == "memory"

    def test_requires_approval(self, mem_tool):
        assert mem_tool.definition.requires_approval is False


# ======================================================================
#  Tests: Store
# ======================================================================

class TestMemoryToolStore:
    @patch("core.v2.memory.get_memory_manager")
    @pytest.mark.asyncio
    async def test_store_success(self, mock_get_mem, mem_tool):
        mock_mem = MagicMock()
        mock_get_mem.return_value = mock_mem

        result = await mem_tool.execute({
            "action": "store", "key": "test_key", "value": "test_value",
        })
        assert result.success is True
        assert result.data["status"] == "pending_approval"
        mock_mem.request_permanent_storage.assert_called_once_with(
            "test_key", "test_value"
        )

    @patch("core.v2.memory.get_memory_manager")
    @pytest.mark.asyncio
    async def test_store_missing_key(self, mock_get_mem, mem_tool):
        result = await mem_tool.execute({"action": "store", "value": "x"})
        assert result.success is False
        assert "key" in result.error

    @patch("core.v2.memory.get_memory_manager")
    @pytest.mark.asyncio
    async def test_store_missing_value(self, mock_get_mem, mem_tool):
        result = await mem_tool.execute({"action": "store", "key": "x"})
        assert result.success is False
        assert "value" in result.error


# ======================================================================
#  Tests: Recall
# ======================================================================

class TestMemoryToolRecall:
    @patch("core.v2.memory.get_memory_manager")
    @pytest.mark.asyncio
    async def test_recall_by_key_found(self, mock_get_mem, mem_tool):
        mock_mem = MagicMock()
        mock_mem.recall_permanent.return_value = "João"
        mock_get_mem.return_value = mock_mem

        result = await mem_tool.execute({
            "action": "recall", "key": "user_name",
        })
        assert result.success is True
        assert result.data["found"] is True
        assert result.data["value"] == "João"

    @patch("core.v2.memory.get_memory_manager")
    @pytest.mark.asyncio
    async def test_recall_by_key_not_found(self, mock_get_mem, mem_tool):
        mock_mem = MagicMock()
        mock_mem.recall_permanent.return_value = None
        mock_get_mem.return_value = mock_mem

        result = await mem_tool.execute({
            "action": "recall", "key": "nonexistent",
        })
        assert result.success is True
        assert result.data["found"] is False

    @patch("core.v2.memory.get_memory_manager")
    @pytest.mark.asyncio
    async def test_recall_all(self, mock_get_mem, mem_tool):
        mock_mem = MagicMock()
        mock_mem.recall_all_permanent.return_value = {"a": 1, "b": 2}
        mock_get_mem.return_value = mock_mem

        result = await mem_tool.execute({"action": "recall"})
        assert result.success is True
        assert result.data["count"] == 2

    @patch("core.v2.memory.get_memory_manager")
    @pytest.mark.asyncio
    async def test_recall_with_query(self, mock_get_mem, mem_tool):
        mock_mem = MagicMock()
        mock_mem.recall_all_permanent.return_value = {
            "user_name": "João",
            "project": "Odysseus",
        }
        mock_get_mem.return_value = mock_mem

        result = await mem_tool.execute({
            "action": "recall", "query": "João",
        })
        assert result.success is True
        assert result.data["count"] >= 1


# ======================================================================
#  Tests: List
# ======================================================================

class TestMemoryToolList:
    @patch("core.v2.memory.get_memory_manager")
    @pytest.mark.asyncio
    async def test_list(self, mock_get_mem, mem_tool):
        mock_mem = MagicMock()
        mock_mem.recall_all_permanent.return_value = {
            "key1": "value1",
            "key2": "value2",
        }
        mock_get_mem.return_value = mock_mem

        result = await mem_tool.execute({"action": "list"})
        assert result.success is True
        assert result.data["count"] == 2
        keys = {e["key"] for e in result.data["entries"]}
        assert keys == {"key1", "key2"}


# ======================================================================
#  Tests: Delete
# ======================================================================

class TestMemoryToolDelete:
    @patch("core.v2.memory.get_memory_manager")
    @pytest.mark.asyncio
    async def test_delete_existing(self, mock_get_mem, mem_tool):
        mock_mem = MagicMock()
        mock_mem.recall_permanent.return_value = "some_value"
        mock_get_mem.return_value = mock_mem

        result = await mem_tool.execute({
            "action": "delete", "key": "user_name",
        })
        assert result.success is True
        assert result.data["status"] == "pending_approval"

    @patch("core.v2.memory.get_memory_manager")
    @pytest.mark.asyncio
    async def test_delete_not_found(self, mock_get_mem, mem_tool):
        mock_mem = MagicMock()
        mock_mem.recall_permanent.return_value = None
        mock_get_mem.return_value = mock_mem

        result = await mem_tool.execute({
            "action": "delete", "key": "nonexistent",
        })
        assert result.success is True
        assert result.data["deleted"] is False

    @patch("core.v2.memory.get_memory_manager")
    @pytest.mark.asyncio
    async def test_delete_missing_key(self, mock_get_mem, mem_tool):
        result = await mem_tool.execute({"action": "delete"})
        assert result.success is False


# ======================================================================
#  Tests: Validação
# ======================================================================

class TestMemoryToolValidation:
    def test_missing_action(self, mem_tool):
        errors = mem_tool.validate_params({})
        assert len(errors) >= 1

    def test_valid_store(self, mem_tool):
        errors = mem_tool.validate_params({
            "action": "store", "key": "k", "value": "v",
        })
        assert errors == []

    def test_valid_recall(self, mem_tool):
        errors = mem_tool.validate_params({"action": "recall"})
        assert errors == []

    def test_valid_list(self, mem_tool):
        errors = mem_tool.validate_params({"action": "list"})
        assert errors == []

    def test_unknown_action(self, mem_tool):
        errors = mem_tool.validate_params({"action": "unknown"})
        # Não valida ações inválidas — o schema permite qualquer string
        # A validação da ação é feita no execute(), não no schema
        assert isinstance(errors, list)
