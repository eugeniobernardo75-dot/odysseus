"""Regression tests for Ollama local-context and streaming fixes."""

from pathlib import Path

from core.v2.llm_client import LLMClient, LLMConfig
from src import model_context


def test_ollama_11434_uses_native_chat_endpoint():
    client = LLMClient(LLMConfig(base_url="http://localhost:11434/v1", model="qwen3:8b", timeout=60.0))
    client._get_chat_url = lambda: "http://localhost:11434/v1/chat/completions"
    client._get_provider = lambda: "openai"

    assert client._get_request_target() == (
        "http://localhost:11434/api/chat",
        "ollama",
    )


def test_model_context_uses_ollama_api_show_for_real_value(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"model_info": {"qwen3.context_length": 40960}, "context_length": 40960}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(model_context.httpx, "post", fake_post)
    model_context._OLLAMA_CONTEXT_CACHE.clear()

    assert model_context._fetch_ollama_context_length("qwen3:8b", "http://localhost:11434/v1") == 40960
    assert captured["url"] == "http://localhost:11434/api/show"
    assert captured["json"] == {"model": "qwen3:8b"}
    assert captured["timeout"] == 3.0
    assert model_context._query_context_length("http://localhost:11434/v1", "qwen3:8b") == (40960, True)


def test_llm_client_static_guardrails_are_present():
    source = Path(__file__).resolve().parents[2] / "core" / "v2" / "llm_client.py"
    text = source.read_text(encoding="utf-8")

    assert "if parsed.port == 11434" in text
    assert text.count("max(self.config.timeout, 180.0)") == 4
    assert text.count('payload["think"] = False') == 2
