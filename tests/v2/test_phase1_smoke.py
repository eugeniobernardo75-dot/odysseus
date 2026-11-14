"""Smoke test Fase 1 — deve passar antes de avançar."""
import json

import pytest

from core.v2.agents import AgentFactory
from core.v2.user_agent_config import UserConfig, get_user_agent, reset_user_agent


@pytest.fixture
def temp_config(tmp_path, monkeypatch):
    cfg_file = tmp_path / "user_config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "name": "Test",
                "language": "pt-PT",
                "style": "conversacional",
                "tools": {},
                "preferences": {},
                "persona": {},
                "model": {},
                "custom_agents": [
                    {
                        "name": "visas",
                        "description": "Vistos",
                        "keywords": ["visto"],
                        "system_prompt": "x",
                        "enabled": True,
                    },
                    {
                        "name": "negocios",
                        "description": "Negócios",
                        "keywords": ["negócio"],
                        "system_prompt": "x",
                        "enabled": True,
                    },
                    {
                        "name": "off_agent",
                        "description": "Desactivado",
                        "keywords": ["off"],
                        "system_prompt": "x",
                        "enabled": False,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    import core.v2.user_agent_config as uac

    monkeypatch.setattr(uac, "CONFIG_FILE", str(cfg_file))
    reset_user_agent()
    return cfg_file


def test_reload_registers_enabled_only(temp_config):
    f = AgentFactory()
    reg = f.reload_custom_agents()
    assert "visas" in reg and "negocios" in reg
    assert "off_agent" not in reg
    assert f.get_agent("visas") is not None
    assert f.get_agent("off_agent") is None


def test_reload_idempotent(temp_config):
    f = AgentFactory()
    a = f.reload_custom_agents()
    b = f.reload_custom_agents()
    assert sorted(a) == sorted(b)


def test_reload_hot_reload(temp_config):
    f = AgentFactory()
    f.reload_custom_agents()
    data = json.loads(temp_config.read_text(encoding="utf-8"))
    data["custom_agents"].append(
        {
            "name": "novo",
            "description": "n",
            "keywords": ["novo"],
            "system_prompt": "x",
            "enabled": True,
        }
    )
    temp_config.write_text(json.dumps(data), encoding="utf-8")
    reset_user_agent()
    reg = f.reload_custom_agents()
    assert "novo" in reg
    assert f.get_agent("novo") is not None


def test_select_agent_finds_custom(temp_config):
    f = AgentFactory()
    f.reload_custom_agents()
    get_user_agent().reload_config()
    chosen = f.select_agent("preciso de um visto para o Brasil")
    assert chosen is not None
    assert chosen.name == "visas"


def test_atomic_save(tmp_path):
    cfg = UserConfig(name="T", custom_agents=[{"name": "x"}])
    t = tmp_path / "c.json"
    assert cfg.save(str(t)) is True
    assert json.loads(t.read_text(encoding="utf-8"))["name"] == "T"


def test_optional_agents_registers_custom(temp_config):
    f = AgentFactory()
    f.register_optional_agents()
    assert f.get_agent("useragent") is not None
    assert f.get_agent("visas") is not None
    assert f.get_agent("negocios") is not None
