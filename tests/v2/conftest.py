"""Isolamento de estado entre testes.

A Fase 3 introduziu persistencia de aprovacoes. Para os testes
verem estado limpo em cada teste, os tres singletons relevantes
sao reiniciados antes e depois de cada teste:

  - safety_manager (persiste aprovacoes em ficheiro)
  - tool_manager (captura safety no __init__)
  - orchestrator (captura safety + tool_manager no __init__)

Sem os tres, o Orchestrator/ToolManager ficam a apontar para
instancias antigas de safety_manager, e os testes que esperam
estado limpo falham de forma dificil de diagnosticar.

Nao apaga ficheiros reais em data/. Apenas redirecciona
ODYSSEUS_V2_APPROVALS_PATH para tmp_path.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_safety_state(tmp_path, monkeypatch):
    from core.v2 import safety as safety_mod
    from core.v2 import tool_manager as tm_mod
    from core.v2 import orchestrator as orch_mod

    isolated_path = tmp_path / "_v2_autouse_approvals.json"
    monkeypatch.setenv("ODYSSEUS_V2_APPROVALS_PATH", str(isolated_path))

    # Antes do teste: resetar os tres singletons, em ordem inversa
    # de dependencia (orchestrator depende de tool_manager depende
    # de safety).
    orch_mod.reset_orchestrator()
    tm_mod.reset_tool_manager()
    safety_mod.reset_safety_manager()

    yield

    # Depois do teste: mesmo reset, para nao contaminar o proximo.
    orch_mod.reset_orchestrator()
    tm_mod.reset_tool_manager()
    safety_mod.reset_safety_manager()
