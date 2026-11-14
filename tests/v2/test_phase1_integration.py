"""Gate 4 — integração real via TestClient (sem auth HTTP)."""
import os
from unittest.mock import patch

# IMPORTANTE: antes de qualquer import da app
os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("ODYSSEUS_V2_REQUIRE_ISOLATION", "0")

import pytest
from fastapi.testclient import TestClient

from tests.helpers.import_state import (
    clear_fake_database_modules,
    clear_module,
    preserve_import_state,
)


@pytest.fixture(scope="module")
def client():
    with patch.dict(os.environ, {"DATABASE_URL": "sqlite:///:memory:"}), \
            preserve_import_state("src.database", "core.database", "src.webhook_manager", "app"):
        clear_fake_database_modules()
        clear_module("src.database")
        from src.database import Webhook  # noqa: F401
        from app import app as fastapi_app
        with TestClient(fastapi_app) as c:
            yield c


def test_v2_agents_lists_custom_and_specialists(client):
    r = client.get("/v2/agents")
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:300]}"
    payload = r.json()
    names = [a.get("name") for a in payload.get("agents", [])]
    # CustomUserAgent do user_config.json
    assert "visas" in names, f"visas ausente; recebido: {names}"
    assert "negocios" in names, f"negocios ausente; recebido: {names}"
    # Especialistas que a Fase 1 activou
    for expected in ("legal", "finance", "documentation", "science"):
        assert expected in names, f"{expected} ausente; recebido: {names}"
    # UserAgent + workers
    assert "useragent" in names
    assert "secretaria" in names


def test_create_agent_requires_approval(client):
    r = client.post(
        "/api/v2/agent-config/agents",
        json={
            "name": "teste_gate4",
            "description": "teste de gate",
            "keywords": ["gate4"],
            "system_prompt": "x",
        },
    )
    assert r.status_code == 202, (
        f"esperava 202 (requires_approval), veio {r.status_code}: "
        f"{r.text[:300]}"
    )
    body = r.json()
    detail = body.get("detail") or body
    assert detail.get("requires_approval") is True, body
    assert detail.get("approval_id"), body


def test_agent_not_created_without_approval(client):
    r = client.get("/api/v2/agent-config/agents")
    assert r.status_code == 200
    names = [a.get("name") for a in r.json().get("agents", [])]
    assert "teste_gate4" not in names, (
        "BYPASS: agente criado sem aprovação! Gate falhou."
    )
