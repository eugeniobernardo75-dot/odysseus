"""Testes Fase 2 — create_agent via chat, com gate de aprovação."""
import json
import pytest

from core.v2.safety import SafetyManager
from core.v2.user_agent_config import UserAgent, UserConfig, reset_user_agent
from core.v2.agents import AgentFactory


@pytest.fixture
def clean_safety(monkeypatch):
    """SafetyManager limpo (sem estado de outros testes)."""
    import core.v2.safety as safety_mod
    sm = SafetyManager()
    monkeypatch.setattr(safety_mod, "_safety_instance", sm)
    return sm


@pytest.fixture
def user_agent(tmp_path, monkeypatch, clean_safety):
    cfg_file = tmp_path / "user_config.json"
    cfg_file.write_text(json.dumps({
        "name": "Test", "language": "pt-PT", "style": "conversacional",
        "tools": {}, "preferences": {}, "persona": {}, "model": {},
        "custom_agents": [],
    }), encoding="utf-8")
    import core.v2.user_agent_config as uac
    monkeypatch.setattr(uac, "CONFIG_FILE", str(cfg_file))
    reset_user_agent()
    return UserAgent()


@pytest.mark.asyncio
async def test_create_agent_proposes_approval(user_agent, clean_safety):
    r = await user_agent._handle_create_agent_request(
        "cria um agente especialista em direito fiscal", None,
    )
    assert r is not None
    assert "APPROVAL-" in r
    assert "aprovação" in r.lower() or "aprovacao" in r.lower()
    assert "direito_fiscal" in r


@pytest.mark.asyncio
async def test_create_agent_commits_after_approval(user_agent, clean_safety):
    r1 = await user_agent._handle_create_agent_request(
        "cria um agente especialista em direito fiscal", None,
    )
    import re
    m = re.search(r"APPROVAL-\d+", r1)
    assert m, f"sem APPROVAL na resposta: {r1}"
    approval_id = m.group(0)
    clean_safety.approve(approval_id)

    r2 = await user_agent._handle_create_agent_request(
        "cria um agente especialista em direito fiscal", approval_id,
    )
    assert "criado" in r2.lower()
    assert "direito_fiscal" in r2
    names = [a["name"] for a in user_agent.user_config.custom_agents]
    assert "direito_fiscal" in names


@pytest.mark.asyncio
async def test_create_agent_without_spec_returns_error(user_agent, clean_safety):
    r = await user_agent._handle_create_agent_request(
        "quero criar um agente", None,
    )
    assert r is not None
    assert "não consegui" in r.lower() or "nao consegui" in r.lower()


@pytest.mark.asyncio
async def test_create_agent_duplicate_returns_error(user_agent, clean_safety):
    r1 = await user_agent._handle_create_agent_request(
        "cria um agente especialista em direito fiscal", None,
    )
    import re
    aid = re.search(r"APPROVAL-\d+", r1).group(0)
    clean_safety.approve(aid)
    await user_agent._handle_create_agent_request(
        "cria um agente especialista em direito fiscal", aid,
    )
    # Tentar criar de novo — deve detectar duplicado em _propose_agent_spec
    r2 = await user_agent._handle_create_agent_request(
        "cria um agente especialista em direito fiscal", None,
    )
    assert r2 is not None
    assert "não consegui" in r2.lower() or "nao consegui" in r2.lower()


@pytest.mark.asyncio
async def test_non_create_message_returns_none(user_agent, clean_safety):
    r = await user_agent._handle_create_agent_request("olá, tudo bem?", None)
    assert r is None


@pytest.mark.asyncio
async def test_process_routes_to_create_agent(user_agent, clean_safety):
    r = await user_agent.process("cria um agente especialista em culinária")
    assert "APPROVAL-" in r.message
    assert "culin" in r.message.lower()
