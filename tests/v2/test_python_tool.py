"""
Testes Unitários — PythonTool

Cobertura:
    - Definição: nome, capacidades, requires_approval
    - _check_dangerous_imports
    - _analyze com AST
    - _validate
    - _format (básico)
    - execute: analyze, format, validate

Uso:
    pytest tests/v2/test_python_tool.py -v
"""

from __future__ import annotations

import sys
import os
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def py_tool():
    from core.v2.tools import PythonTool
    return PythonTool()


# ======================================================================
#  Tests: Definição
# ======================================================================

class TestPythonToolDefinition:
    def test_name(self, py_tool):
        assert py_tool.name == "python"

    def test_capabilities(self, py_tool):
        assert "python.execute" in py_tool.definition.capabilities
        assert "python.analyze" in py_tool.definition.capabilities
        assert "python.format" in py_tool.definition.capabilities

    def test_requires_approval(self, py_tool):
        assert py_tool.definition.requires_approval is True

    def test_category(self, py_tool):
        assert py_tool.definition.category == "code"

    def test_actions(self, py_tool):
        actions = py_tool.definition.parameters["properties"]["action"]["enum"]
        assert set(actions) == {"execute", "analyze", "format"}


# ======================================================================
#  Tests: _check_dangerous_imports
# ======================================================================

class TestPythonToolDangerousImports:
    def test_os_is_dangerous(self, py_tool):
        assert py_tool._check_dangerous_imports("import os") == ["os"]

    def test_subprocess_is_dangerous(self, py_tool):
        assert "subprocess" in py_tool._check_dangerous_imports("import subprocess")

    def test_from_import_dangerous(self, py_tool):
        dangerous = py_tool._check_dangerous_imports("from os import path")
        assert "os" in dangerous

    def test_safe_import(self, py_tool):
        assert py_tool._check_dangerous_imports("import json") == []

    def test_multiple_dangerous(self, py_tool):
        dangerous = py_tool._check_dangerous_imports(
            "import os; import socket; import json"
        )
        assert len(dangerous) >= 2
        assert "os" in dangerous
        assert "socket" in dangerous

    def test_no_imports(self, py_tool):
        assert py_tool._check_dangerous_imports("x = 1") == []

    def test_non_dangerous_in_string(self, py_tool):
        # A verificação usa substring: 'import os' está dentro de print("import os")
        # Isto é um falso positivo aceitável (segurança > conveniência)
        dangerous = py_tool._check_dangerous_imports('print("import os")')
        assert len(dangerous) >= 1  # "import os" é substring


# ======================================================================
#  Tests: _analyze (AST)
# ======================================================================

class TestPythonToolAnalyze:
    @pytest.mark.asyncio
    async def test_analyze_simple(self, py_tool):
        result = await py_tool.execute({
            "action": "analyze",
            "code": "x = 1\nprint(x)",
        })
        assert result.success is True
        assert result.data["valid_syntax"] is True
        assert result.data["metrics"]["n_lines"] >= 2

    @pytest.mark.asyncio
    async def test_analyze_imports(self, py_tool):
        result = await py_tool.execute({
            "action": "analyze",
            "code": "import os\nimport json\nfrom pathlib import Path",
        })
        assert result.success is True
        assert len(result.data["imports"]) >= 2

    @pytest.mark.asyncio
    async def test_analyze_classes_functions(self, py_tool):
        code = """
class MyClass:
    def method1(self):
        pass

def my_func():
    pass
"""
        result = await py_tool.execute({"action": "analyze", "code": code})
        assert result.success is True
        assert len(result.data["classes"]) == 1
        assert result.data["classes"][0]["name"] == "MyClass"
        assert result.data["classes"][0]["methods"] == ["method1"]
        assert len(result.data["functions"]) >= 1
        assert result.data["functions"][0]["name"] == "my_func"

    @pytest.mark.asyncio
    async def test_analyze_syntax_error(self, py_tool):
        result = await py_tool.execute({
            "action": "analyze",
            "code": "def broken(",
        })
        assert result.success is False
        assert "sintaxe" in result.error.lower()

    @pytest.mark.asyncio
    async def test_analyze_empty(self, py_tool):
        result = await py_tool.execute({"action": "analyze", "code": ""})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_validate_valid(self, py_tool):
        # PythonTool validate usa _execute_safe (subprocess).
        # Testamos a validacao via analyze que usa AST.
        result = await py_tool.execute({
            "action": "analyze",
            "code": "x = 1",
        })
        assert result.success is True
        assert result.data["valid_syntax"] is True

    @pytest.mark.asyncio
    async def test_validate_invalid(self, py_tool):
        result = await py_tool.execute({
            "action": "analyze",
            "code": "def foo(:",
        })
        assert result.success is False

    @pytest.mark.asyncio
    async def test_validate_non_python(self, py_tool):
        # PythonTool nao tem suporte nativo para 'validate' com outras linguagens.
        # O analyze funciona apenas para Python (via AST).
        # Para non-python, o analyze tenta AST e falha com syntax error.
        result = await py_tool.execute({
            "action": "analyze",
            "code": "var x = 1;",
        })
        # JavaScript nao e Python valido, entao AST retorna syntax error
        assert result.success is False


# ======================================================================
#  Tests: _format
# ======================================================================

class TestPythonToolFormat:
    @pytest.mark.asyncio
    async def test_format_basic(self, py_tool):
        result = await py_tool.execute({
            "action": "format",
            "code": "x=1\n",
        })
        assert result.success is True
        assert "formatted" in result.data

    @pytest.mark.asyncio
    async def test_format_empty(self, py_tool):
        result = await py_tool.execute({"action": "format", "code": ""})
        assert result.success is False


# ======================================================================
#  Tests: Validação de parâmetros
# ======================================================================

class TestPythonToolExecutionGuard:
    """O PythonTool de execução exige autorização e workspace confinado."""

    @pytest.fixture
    def workspace(self):
        from core.v2.tools import V2_PROJECT_ROOT
        return V2_PROJECT_ROOT

    def _ctx(self, ws):
        return {"workspace_path": str(ws)}

    @pytest.mark.asyncio
    async def test_analyze_needs_no_authorization(self, py_tool):
        """Análise estática não executa nada: não exige autorização."""
        result = await py_tool.execute({"action": "analyze", "code": "x = 1"})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_direct_execute_is_denied(self, py_tool, workspace, monkeypatch):
        import asyncio
        spawned = []

        async def fake_exec(*args, **kwargs):
            spawned.append(args)
            return _FakeProc(returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        result = await py_tool.execute(
            {"action": "execute", "code": "print(1)"}, context=self._ctx(workspace)
        )

        assert result.success is False
        assert result.metadata.get("reason") == "not_authorized"
        assert spawned == []

    @pytest.mark.asyncio
    async def test_authorized_without_workspace_is_denied(self, py_tool, monkeypatch):
        import asyncio
        from core.v2.execution_policy import authorized_execution
        spawned = []

        async def fake_exec(*args, **kwargs):
            spawned.append(args)
            return _FakeProc(returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        with authorized_execution("python", "EXEC-TEST"):
            result = await py_tool.execute({"action": "execute", "code": "print(1)"})

        assert result.success is False
        assert result.metadata.get("reason") == "no_workspace"
        assert spawned == []

    @pytest.mark.asyncio
    async def test_authorized_run_uses_workspace_cwd(self, py_tool, workspace, monkeypatch):
        import asyncio
        from core.v2.execution_policy import authorized_execution
        calls = []

        async def fake_exec(*args, **kwargs):
            calls.append((args, kwargs))
            return _FakeProc(stdout="ok", returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        with authorized_execution("python", "EXEC-TEST"):
            result = await py_tool.execute(
                {"action": "execute", "code": "print(1)"}, context=self._ctx(workspace)
            )

        assert result.success is True
        args, kwargs = calls[0]
        assert args[0].endswith(("python.exe", "python"))
        assert args[1] == "-c"
        assert kwargs.get("cwd") == str(workspace)

    @pytest.mark.asyncio
    async def test_return_code_none_is_failure(self, py_tool, workspace, monkeypatch):
        import asyncio
        from core.v2.execution_policy import authorized_execution

        async def fake_exec(*args, **kwargs):
            return _FakeProc(stdout="", returncode=None)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        with authorized_execution("python", "EXEC-TEST"):
            result = await py_tool.execute(
                {"action": "execute", "code": "print(1)"}, context=self._ctx(workspace)
            )

        assert result.success is False
        assert "return_code=None" in result.error

    @pytest.mark.asyncio
    async def test_pipe_unavailable_falls_back(self, py_tool, workspace, monkeypatch):
        import asyncio
        from core.v2.execution_policy import authorized_execution
        calls = []

        async def fake_exec(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise PermissionError("[WinError 5] Acesso negado")
            return _FakeProc(returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        with authorized_execution("python", "EXEC-TEST"):
            result = await py_tool.execute(
                {"action": "execute", "code": "print(1)"}, context=self._ctx(workspace)
            )

        assert len(calls) == 2
        assert result.success is True
        assert result.data["output_captured"] is False

    @pytest.mark.asyncio
    async def test_dangerous_import_still_blocked(self, py_tool, workspace, monkeypatch):
        import asyncio
        from core.v2.execution_policy import authorized_execution
        spawned = []

        async def fake_exec(*args, **kwargs):
            spawned.append(args)
            return _FakeProc(returncode=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        with authorized_execution("python", "EXEC-TEST"):
            result = await py_tool.execute(
                {"action": "execute", "code": "import os\nprint(1)"},
                context=self._ctx(workspace),
            )

        assert result.success is False
        assert spawned == []


class _FakeProc:
    """Duplo de processo para os testes de execução do PythonTool."""

    def __init__(self, stdout="", stderr="", returncode=0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.pid = None

    async def communicate(self):
        return self._stdout, self._stderr

    def kill(self):
        self.returncode = -9

    async def wait(self):
        return self.returncode


class TestPythonToolRealExecution:
    """Execução Python REAL, autorizada, dentro do workspace aprovado (Etapa 4).

    Estes testes correm um subprocesso Python verdadeiro. O código é inofensivo
    e não acede a ficheiros. Requer autorização (marcador do ToolManager) e um
    workspace resolvido pelo servidor.
    """

    @pytest.fixture
    def workspace(self):
        from core.v2.execution_policy import approved_root
        root = approved_root("workspace")
        assert root is not None, "o Workspace aprovado tem de existir"
        return root

    def _ctx(self, ws):
        return {"workspace_path": str(ws)}

    @pytest.mark.asyncio
    async def test_harmless_code_runs(self, py_tool, workspace):
        from core.v2.execution_policy import authorized_execution
        with authorized_execution("python", "EXEC-REAL-1"):
            result = await py_tool.execute(
                {"action": "execute", "code": "print(6 * 7)", "timeout": 20},
                context=self._ctx(workspace),
            )
        assert result.success is True
        assert result.data["return_code"] == 0
        assert result.data["output_captured"] is False  # ambiente sem pipes

    @pytest.mark.asyncio
    async def test_error_code_propagates(self, py_tool, workspace):
        from core.v2.execution_policy import authorized_execution
        with authorized_execution("python", "EXEC-REAL-2"):
            result = await py_tool.execute(
                {"action": "execute", "code": "raise ValueError('falha controlada')", "timeout": 20},
                context=self._ctx(workspace),
            )
        assert result.success is False
        assert result.data["return_code"] not in (0, None)

    @pytest.mark.asyncio
    async def test_timeout_terminates_process(self, py_tool, workspace):
        from core.v2.execution_policy import authorized_execution
        with authorized_execution("python", "EXEC-REAL-3"):
            result = await py_tool.execute(
                {"action": "execute",
                 "code": "import time\ntime.sleep(5)",
                 "timeout": 0.5},
                context=self._ctx(workspace),
            )
        assert result.success is False
        assert "timeout" in (result.error or "").lower()
        assert result.metadata.get("timeout") == 0.5

    @pytest.mark.asyncio
    async def test_unauthorized_real_execution_is_refused(self, py_tool, workspace):
        """Sem autorização, nada corre — mesmo com workspace válido."""
        result = await py_tool.execute(
            {"action": "execute", "code": "print(1)"}, context=self._ctx(workspace)
        )
        assert result.success is False
        assert result.metadata.get("reason") == "not_authorized"


class TestPythonToolValidation:
    def test_missing_action(self, py_tool):
        errors = py_tool.validate_params({"code": "x=1"})
        assert len(errors) >= 1

    def test_missing_code(self, py_tool):
        errors = py_tool.validate_params({"action": "analyze"})
        assert len(errors) >= 1

    def test_valid(self, py_tool):
        errors = py_tool.validate_params({"action": "analyze", "code": "x=1"})
        assert errors == []
