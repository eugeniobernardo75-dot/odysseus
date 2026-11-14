"""
Testes Unitários — BrowserTool

Cobertura:
    - Definição: nome, 8 capacidades, parâmetros, timeout
    - Engine detection: _is_playwright_available, _is_bs4_available
    - Ações interativas sem Playwright — retornam erro
    - Ações read-only sem Playwright — extract/html/links retornam erro
    - Navigate sem URL
    - _parse_html com BS4 e fallback regex
    - Validação de parâmetros

Uso:
    pytest tests/v2/test_browser_tool.py -v
"""

from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def browser_tool():
    from core.v2.browser_tool import BrowserTool
    return BrowserTool()


# ======================================================================
#  Tests: Definição
# ======================================================================

class TestBrowserToolDefinition:
    def test_name(self, browser_tool):
        assert browser_tool.name == "browser"

    def test_capabilities(self, browser_tool):
        caps = browser_tool.definition.capabilities
        expected = {
            "browser.navigate", "browser.screenshot", "browser.click",
            "browser.type", "browser.extract", "browser.evaluate",
            "browser.html", "browser.links",
        }
        assert set(caps) == expected

    def test_actions(self, browser_tool):
        actions = browser_tool.definition.parameters["properties"]["action"]["enum"]
        expected = [
            "navigate", "screenshot", "click", "type",
            "extract", "evaluate", "html", "links",
        ]
        assert list(actions) == expected

    def test_params_schema(self, browser_tool):
        props = browser_tool.definition.parameters["properties"]
        assert "url" in props
        assert "selector" in props
        assert "text" in props
        assert "script" in props
        assert "max_chars" in props
        assert "timeout" in props
        assert "wait_until" in props
        assert "full_page" in props
        assert "max_links" in props
        assert "filter_domain" in props

    def test_category(self, browser_tool):
        assert browser_tool.definition.category == "browser"

    def test_timeout(self, browser_tool):
        assert browser_tool.definition.timeout == 60.0

    def test_requires_approval(self, browser_tool):
        assert browser_tool.definition.requires_approval is False

    def test_interactive_actions_set(self, browser_tool):
        assert browser_tool._INTERACTIVE_ACTIONS == {"screenshot", "click", "type", "evaluate"}


# ======================================================================
#  Tests: Engine detection
# ======================================================================

class TestBrowserToolEngineDetection:
    def test_playwright_not_available(self):
        """Playwright não está instalado no ambiente de teste."""
        from core.v2.browser_tool import _is_playwright_available
        assert _is_playwright_available() is False

    def test_bs4_available(self):
        """BeautifulSoup está instalado."""
        from core.v2.browser_tool import _is_bs4_available
        assert _is_bs4_available() is True

    def test_playwright_available_when_importable(self):
        """Simular Playwright instalado via mock de sys.modules."""
        import importlib
        # Adicionar o módulo 'playwright' fake antes de recarregar
        fake_pw = MagicMock()
        fake_pw.__file__ = "<mock>"
        old = sys.modules.get("playwright")
        sys.modules["playwright"] = fake_pw
        try:
            importlib.reload(sys.modules["core.v2.browser_tool"])
            from core.v2.browser_tool import _is_playwright_available
            assert _is_playwright_available() is True
        finally:
            if old:
                sys.modules["playwright"] = old
            else:
                del sys.modules["playwright"]
            importlib.reload(sys.modules["core.v2.browser_tool"])


# ======================================================================
#  Tests: Ações interativas sem Playwright
# ======================================================================

class TestBrowserToolInteractiveWithoutPW:
    @pytest.mark.asyncio
    async def test_screenshot_requires_playwright(self, browser_tool):
        result = await browser_tool.execute({"action": "screenshot"})
        assert result.success is False
        assert "Playwright" in result.error

    @pytest.mark.asyncio
    async def test_click_requires_playwright(self, browser_tool):
        result = await browser_tool.execute({
            "action": "click", "selector": "body",
        })
        assert result.success is False
        assert "Playwright" in result.error

    @pytest.mark.asyncio
    async def test_type_requires_playwright(self, browser_tool):
        result = await browser_tool.execute({
            "action": "type", "selector": "input", "text": "hello",
        })
        assert result.success is False
        assert "Playwright" in result.error

    @pytest.mark.asyncio
    async def test_evaluate_requires_playwright(self, browser_tool):
        result = await browser_tool.execute({
            "action": "evaluate", "script": "1+1",
        })
        assert result.success is False
        assert "Playwright" in result.error


# ======================================================================
#  Tests: Ações read-only sem Playwright
# ======================================================================

class TestBrowserToolReadOnlyWithoutPW:
    @pytest.mark.asyncio
    async def test_extract_requires_playwright(self, browser_tool):
        result = await browser_tool.execute({"action": "extract"})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_html_requires_playwright(self, browser_tool):
        result = await browser_tool.execute({"action": "html"})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_links_requires_playwright(self, browser_tool):
        result = await browser_tool.execute({"action": "links"})
        assert result.success is False


# ======================================================================
#  Tests: Navigate sem Playwright
# ======================================================================

class TestBrowserToolNavigate:
    @pytest.mark.asyncio
    async def test_navigate_missing_url(self, browser_tool):
        result = await browser_tool.execute({"action": "navigate"})
        assert result.success is False
        assert "url" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unknown_action(self, browser_tool):
        result = await browser_tool.execute({"action": "fly"})
        assert result.success is False
        assert "desconhecida" in result.error.lower()


# ======================================================================
#  Tests: _parse_html (com BeautifulSoup)
# ======================================================================

class TestBrowserToolParseHTML:
    def test_parse_simple_html(self, browser_tool):
        html = "<html><head><title>Test Page</title></head><body><p>Hello</p></body></html>"
        text, title, links = browser_tool._parse_html(html)
        assert "Test Page" in title
        assert "Hello" in text
        assert links == []

    def test_parse_with_links(self, browser_tool):
        html = """
        <html><body>
            <a href="https://example.com">Example</a>
            <a href="https://test.com">Test</a>
        </body></html>
        """
        text, title, links = browser_tool._parse_html(html)
        assert len(links) == 2
        urls = {l["href"] for l in links}
        assert "https://example.com" in urls
        assert "https://test.com" in urls

    def test_parse_removes_scripts(self, browser_tool):
        html = "<html><body><script>alert('xss')</script><p>Content</p></body></html>"
        text, title, links = browser_tool._parse_html(html)
        assert "alert" not in text
        assert "Content" in text

    def test_parse_empty_html(self, browser_tool):
        text, title, links = browser_tool._parse_html("")
        assert text == ""
        assert title == ""

    def test_parse_max_chars(self, browser_tool):
        html = "<body>" + "a" * 200 + "</body>"
        text, title, links = browser_tool._parse_html(html, max_chars=50)
        assert len(text) <= 53

    def test_parse_duplicate_links(self, browser_tool):
        html = """
        <body>
            <a href="https://same.com">Link 1</a>
            <a href="https://same.com">Link 2</a>
        </body>
        """
        text, title, links = browser_tool._parse_html(html)
        assert len(links) == 1  # deduplicado

    def test_parse_no_http_links(self, browser_tool):
        html = """
        <body>
            <a href="/relative">Relative</a>
            <a href="#anchor">Anchor</a>
            <a href="mailto:user@test.com">Email</a>
        </body>
        """
        text, title, links = browser_tool._parse_html(html)
        relative_links = [l for l in links if not l["href"].startswith("http")]
        assert len(relative_links) == 0


# ======================================================================
#  Tests: Validação de parâmetros
# ======================================================================

class TestBrowserToolValidation:
    def test_missing_action(self, browser_tool):
        errors = browser_tool.validate_params({})
        assert len(errors) >= 1

    def test_valid_params(self, browser_tool):
        errors = browser_tool.validate_params({"action": "navigate"})
        assert errors == []

    def test_invalid_url_type(self, browser_tool):
        errors = browser_tool.validate_params({
            "action": "navigate", "url": 123,
        })
        type_errors = [e for e in errors if "deve ser string" in e]
        assert len(type_errors) >= 1
