"""
Testes Unitários — CodeTool

Cobertura:
    - Definição: nome, capacidades, parâmetros
    - _analyze: classes, funções, imports, métricas
    - _metrics: contagem de linhas
    - _validate: sintaxe válida e inválida
    - execute: analyze, metrics, validate
    - Validação de parâmetros

Uso:
    pytest tests/v2/test_code_tool.py -v
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
def code_tool():
    from core.v2.tools import CodeTool
    return CodeTool()


SAMPLE_CODE = """
import os
import json
from pathlib import Path

class Calculator:
    \"\"\"A simple calculator.\"\"\"

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

def main():
    calc = Calculator()
    print(calc.add(1, 2))
"""


# ======================================================================
#  Tests: Definição
# ======================================================================

class TestCodeToolDefinition:
    def test_name(self, code_tool):
        assert code_tool.name == "code"

    def test_capabilities(self, code_tool):
        caps = code_tool.definition.capabilities
        assert "code.analyze" in caps
        assert "code.metrics" in caps
        assert "code.validate" in caps

    def test_actions(self, code_tool):
        actions = code_tool.definition.parameters["properties"]["action"]["enum"]
        assert set(actions) == {"analyze", "metrics", "validate"}

    def test_category(self, code_tool):
        assert code_tool.definition.category == "code"

    def test_required_params(self, code_tool):
        required = code_tool.definition.parameters["required"]
        assert "action" in required


# ======================================================================
#  Tests: _validate
# ======================================================================

class TestCodeToolValidate:
    @pytest.mark.asyncio
    async def test_validate_valid_syntax(self, code_tool):
        result = await code_tool.execute({
            "action": "validate", "code": "x = 1",
        })
        assert result.success is True
        assert result.data["valid"] is True

    @pytest.mark.asyncio
    async def test_validate_invalid_syntax(self, code_tool):
        result = await code_tool.execute({
            "action": "validate", "code": "def broken(",
        })
        assert result.success is False
        assert result.data["valid"] is False

    @pytest.mark.asyncio
    async def test_validate_non_python(self, code_tool):
        result = await code_tool.execute({
            "action": "validate", "code": "var x = 1",
            "language": "javascript",
        })
        # Validação só para Python — retorna success com nota
        assert result.success is True

    @pytest.mark.asyncio
    async def test_validate_empty(self, code_tool):
        # Código vazio é rejeitado pelo execute (não chega ao validate)
        result = await code_tool.execute({"action": "validate", "code": ""})
        assert result.success is False


# ======================================================================
#  Tests: _metrics
# ======================================================================

class TestCodeToolMetrics:
    @pytest.mark.asyncio
    async def test_metrics_basic(self, code_tool):
        result = await code_tool.execute({
            "action": "metrics", "code": SAMPLE_CODE,
        })
        assert result.success is True
        assert result.data["total_lines"] >= 15
        assert result.data["code_lines"] >= 10
        assert result.data["classes"] >= 1
        assert result.data["functions"] >= 2
        assert result.data["imports"] >= 2

    @pytest.mark.asyncio
    async def test_metrics_empty(self, code_tool):
        # Código vazio é rejeitado pelo execute
        result = await code_tool.execute({"action": "metrics", "code": ""})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_metrics_invalid_syntax(self, code_tool):
        # Metrics funciona mesmo com syntax error
        result = await code_tool.execute({"action": "metrics", "code": "def broken("})
        assert result.success is True


# ======================================================================
#  Tests: _analyze
# ======================================================================

class TestCodeToolAnalyze:
    @pytest.mark.asyncio
    async def test_analyze_imports(self, code_tool):
        result = await code_tool.execute({
            "action": "analyze", "code": SAMPLE_CODE,
        })
        assert result.success is True
        assert len(result.data["imports"]) >= 2
        modules = {imp["module"] for imp in result.data["imports"]}
        assert "os" in modules
        assert "json" in modules

    @pytest.mark.asyncio
    async def test_analyze_classes(self, code_tool):
        result = await code_tool.execute({
            "action": "analyze", "code": SAMPLE_CODE,
        })
        assert result.success is True
        assert len(result.data["classes"]) == 1
        cls = result.data["classes"][0]
        assert cls["name"] == "Calculator"
        assert "add" in cls["methods"]
        assert "subtract" in cls["methods"]
        assert "docstring" in cls

    @pytest.mark.asyncio
    async def test_analyze_functions(self, code_tool):
        result = await code_tool.execute({
            "action": "analyze", "code": SAMPLE_CODE,
        })
        assert result.success is True
        names = {f["name"] for f in result.data["functions"]}
        assert "main" in names

    @pytest.mark.asyncio
    async def test_analyze_syntax_error(self, code_tool):
        result = await code_tool.execute({
            "action": "analyze", "code": "def broken(",
        })
        assert result.success is False
        assert result.data["valid_syntax"] is False

    @pytest.mark.asyncio
    async def test_analyze_metrics_included(self, code_tool):
        result = await code_tool.execute({
            "action": "analyze", "code": SAMPLE_CODE,
        })
        metrics = result.data["metrics"]
        assert metrics["n_imports"] >= 2
        assert metrics["n_classes"] >= 1
        assert metrics["n_functions"] >= 2

    @pytest.mark.asyncio
    async def test_analyze_via_file(self, code_tool, tmp_path):
        f = tmp_path / "test_code.py"
        f.write_text(SAMPLE_CODE)

        mock_ctx = {"workspace_path": str(tmp_path)}
        result = await code_tool.execute({
            "action": "analyze", "file": "test_code.py",
        }, context=mock_ctx)
        assert result.success is True
        assert len(result.data["classes"]) >= 1


# ======================================================================
#  Tests: Execute com parâmetros em falta
# ======================================================================

class TestCodeToolExecute:
    @pytest.mark.asyncio
    async def test_no_code_no_file(self, code_tool):
        result = await code_tool.execute({"action": "analyze"})
        assert result.success is False
        assert "Fornece" in result.error

    @pytest.mark.asyncio
    async def test_unknown_action(self, code_tool):
        result = await code_tool.execute({"action": "unknown"})
        assert result.success is False


# ======================================================================
#  Tests: Validação
# ======================================================================

class TestCodeToolValidation:
    def test_missing_action(self, code_tool):
        errors = code_tool.validate_params({"code": "x=1"})
        assert len(errors) >= 1

    def test_valid(self, code_tool):
        errors = code_tool.validate_params({"action": "analyze"})
        assert errors == []
