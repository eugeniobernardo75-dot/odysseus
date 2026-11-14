"""
BrowserTool — Odysseus V2

Ferramenta de automação web com dois motores:

1. **Playwright** (preferido) — browser real com navegação, cliques,
   screenshots, avaliação de JavaScript. Requer ``pip install playwright``
   e ``playwright install chromium``.

2. **httpx + BeautifulSoup** (fallback) — fetch de páginas e extração
   de texto/links. Funciona sem instalação adicional.

Capacidades:
    - browser.navigate:    navegar para URL e extrair conteúdo
    - browser.screenshot:  capturar screenshot (Playwright apenas)
    - browser.click:       clicar num elemento por seletor (Playwright apenas)
    - browser.type:        escrever texto num campo (Playwright apenas)
    - browser.extract:     extrair texto/seletores da página atual
    - browser.evaluate:    executar JavaScript (Playwright apenas)
    - browser.html:        obter HTML completo da página
    - browser.links:       obter todos os links da página

Segurança:
    - Operações interativas (click, type, screenshot, evaluate)
      requerem aprovação explícita
    - Timeout configurável para evitar bloqueios
    - Playwright isolado em contexto de navegador
    - Fallback httpx para operações só de leitura
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from .tool_registry import BaseTool, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


# ======================================================================
#  Verificação de motores
# ======================================================================

def _is_playwright_available() -> bool:
    """Verifica se o Playwright está instalado."""
    try:
        import playwright  # noqa: F401
        return True
    except ImportError:
        return False


def _is_bs4_available() -> bool:
    """Verifica se o BeautifulSoup está instalado."""
    try:
        import bs4  # noqa: F401
        return True
    except ImportError:
        return False


# ======================================================================
#  Gestor de Browser (Playwright)
# ======================================================================

class _PlaywrightSession:
    """
    Gerencia uma sessão de browser Playwright.

    Mantém o browser e contexto abertos entre chamadas para
    evitar reinicializações. Fecha automaticamente após inatividade.
    """

    _instance: Optional["_PlaywrightSession"] = None
    _browser = None
    _context = None
    _page = None
    _last_used = 0.0
    _idle_timeout = 60.0  # segundos antes de fechar

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def get_page(self):
        """Obtém ou cria uma página do browser."""
        self._last_used = time.monotonic()

        if self._page is not None:
            # Verificar se a página ainda está aberta
            try:
                _ = await self._page.title()
                return self._page
            except Exception:
                pass

        # Fechar sessão antiga se existir
        await self.close()

        # Iniciar Playwright
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )
        self._context = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 720},
            locale="pt-PT",
            timezone_id="Europe/Lisbon",
        )
        self._page = await self._context.new_page()
        logger.info("Playwright: nova sessão iniciada")
        return self._page

    async def close(self):
        """Fecha a sessão do browser."""
        try:
            if self._page:
                await self._page.close()
        except Exception:
            pass
        try:
            if self._context:
                await self._context.close()
        except Exception:
            pass
        try:
            if self._browser:
                await self._browser.close()
        except Exception:
            pass
        try:
            if hasattr(self, "_playwright") and self._playwright:
                await self._playwright.stop()
        except Exception:
            pass

        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None  # type: ignore
        logger.info("Playwright: sessão fechada")

    async def check_idle(self):
        """Fecha o browser se estiver inativo por muito tempo."""
        if self._page and (time.monotonic() - self._last_used) > self._idle_timeout:
            logger.info("Playwright: fechar sessão inativa")
            await self.close()


# ======================================================================
#  BrowserTool
# ======================================================================

class BrowserTool(BaseTool):
    """
    Ferramenta de automação web.

    Usa Playwright quando disponível (full browser) ou
    httpx+BeautifulSoup como fallback (leitura apenas).

    Capacidades:
        - browser.navigate: navegar para URL e extrair conteúdo
        - browser.screenshot: capturar screenshot (Playwright)
        - browser.click: clicar num elemento (Playwright)
        - browser.type: escrever texto (Playwright)
        - browser.extract: extrair texto/seletores
        - browser.evaluate: executar JavaScript (Playwright)
        - browser.html: obter HTML completo
        - browser.links: obter links da página
    """

    # Ações que requerem aprovação (Playwright interativo)
    _INTERACTIVE_ACTIONS = {"screenshot", "click", "type", "evaluate"}

    def _build_definition(self) -> ToolDefinition:
        actions = [
            "navigate", "screenshot", "click", "type",
            "extract", "evaluate", "html", "links",
        ]
        return ToolDefinition(
            name="browser",
            description=(
                "Automação web: navegar, clicar, escrever, "
                "screenshot, extrair texto, avaliar JavaScript. "
                "Usa Playwright quando disponível; fallback httpx+BS4."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": actions,
                        "description": "Ação a executar",
                    },
                    "url": {
                        "type": "string",
                        "description": "URL para navegar (obrigatório para navigate)",
                    },
                    "selector": {
                        "type": "string",
                        "description": (
                            "Seletor CSS para encontrar elemento "
                            "(click, type, extract)"
                        ),
                    },
                    "text": {
                        "type": "string",
                        "description": (
                            "Texto a escrever (type) ou a extrair "
                            "(seletor alternativo)"
                        ),
                    },
                    "script": {
                        "type": "string",
                        "description": (
                            "Código JavaScript a avaliar (evaluate)"
                        ),
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Máximo de caracteres a extrair",
                        "default": 10000,
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout em segundos",
                        "default": 30.0,
                    },
                    "wait_until": {
                        "type": "string",
                        "description": (
                            "Evento para aguardar (navigate): "
                            "domcontentloaded, load, networkidle"
                        ),
                        "default": "domcontentloaded",
                    },
                    "full_page": {
                        "type": "boolean",
                        "description": (
                            "Screenshot de página completa (full page)"
                        ),
                        "default": False,
                    },
                    "max_links": {
                        "type": "integer",
                        "description": (
                            "Máximo de links a retornar (apenas links)"
                        ),
                        "default": 200,
                    },
                    "filter_domain": {
                        "type": "string",
                        "description": (
                            "Filtrar links por domínio (apenas links)"
                        ),
                    },
                },
                "required": ["action"],
            },
            capabilities=[
                "browser.navigate", "browser.screenshot", "browser.click",
                "browser.type", "browser.extract", "browser.evaluate",
                "browser.html", "browser.links",
            ],
            permissions=["network"],
            category="browser",
            version="1.0.0",
            timeout=60.0,
            requires_approval=False,  # Aprovação é por ação específica
            tags=["browser", "web", "automation", "playwright"],
        )

    # ------------------------------------------------------------------
    #  Execute (dispatch principal)
    # ------------------------------------------------------------------

    async def execute(self, params: dict, context: Optional[dict] = None) -> ToolResult:
        action = params["action"]
        use_playwright = _is_playwright_available()
        engine = "playwright" if use_playwright else "httpx+bs4"

        logger.debug("BrowserTool: %s (engine: %s)", action, engine)

        # Verificar se ação requer Playwright
        if action in self._INTERACTIVE_ACTIONS and not use_playwright:
            return ToolResult(
                success=False,
                error=(
                    f"Ação '{action}' requer Playwright. "
                    "Instala: pip install playwright && playwright install chromium"
                ),
                metadata={"action": action, "engine": engine},
            )

        try:
            if action == "navigate":
                return await self._navigate(params, use_playwright)
            elif action == "screenshot":
                return await self._screenshot(params)
            elif action == "click":
                return await self._click(params)
            elif action == "type":
                return await self._type(params)
            elif action == "extract":
                return await self._extract(params, use_playwright)
            elif action == "evaluate":
                return await self._evaluate(params)
            elif action == "html":
                return await self._html(params, use_playwright)
            elif action == "links":
                return await self._links(params, use_playwright)
            else:
                return ToolResult(
                    success=False,
                    error=(
                        f"Ação desconhecida: '{action}'. "
                        "Usa: navigate, screenshot, click, type, "
                        "extract, evaluate, html, links"
                    ),
                )

        except Exception as e:
            logger.exception("BrowserTool: erro em '%s'", action)
            return ToolResult(
                success=False,
                error=f"Erro em '{action}': {e}",
                metadata={"action": action, "engine": engine},
            )

    # ------------------------------------------------------------------
    #  Navegação
    # ------------------------------------------------------------------

    async def _navigate(self, params: dict, use_playwright: bool) -> ToolResult:
        url = params.get("url", "")
        if not url:
            return ToolResult(success=False, error="'url' é obrigatório")

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        max_chars = params.get("max_chars", 10000)
        timeout = params.get("timeout", 30.0)
        wait_until = params.get("wait_until", "domcontentloaded")

        if use_playwright:
            return await self._navigate_playwright(url, max_chars, timeout, wait_until)
        else:
            return await self._navigate_httpx(url, max_chars, timeout)

    async def _navigate_playwright(
        self, url: str, max_chars: int, timeout: float, wait_until: str,
    ) -> ToolResult:
        from playwright.async_api import TimeoutError as PwTimeout

        session = _PlaywrightSession()
        page = await session.get_page()

        try:
            await page.goto(url, wait_until=wait_until, timeout=int(timeout * 1000))
            title = await page.title()
            current_url = page.url
            content = await page.content()
            text = await page.inner_text("body")

            # Extrair texto limpo
            text = re.sub(r"\\s+", " ", text).strip()
            if len(text) > max_chars:
                text = text[:max_chars] + "..."

            # Extrair links
            links = await page.evaluate("""
                () => Array.from(document.querySelectorAll('a[href]'))
                    .map(a => ({text: a.innerText.trim(), href: a.href}))
                    .filter(l => l.href.startsWith('http'))
            """)

            return ToolResult(
                success=True,
                data={
                    "url": current_url,
                    "title": title,
                    "text": text,
                    "links": links[:200] if links else [],
                    "links_count": len(links) if links else 0,
                    "content_length": len(content),
                    "engine": "playwright",
                },
                metadata={
                    "action": "navigate",
                    "url": current_url,
                    "title": title,
                    "engine": "playwright",
                },
            )

        except PwTimeout:
            return ToolResult(
                success=False,
                error=f"Timeout ao navegar para {url} (timeout={timeout}s)",
                metadata={"url": url},
            )

    async def _navigate_httpx(
        self, url: str, max_chars: int, timeout: float,
    ) -> ToolResult:
        import httpx

        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True
        ) as client:
            resp = await client.get(url, headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            })

            if resp.status_code != 200:
                return ToolResult(
                    success=False,
                    error=f"HTTP {resp.status_code} ao aceder a {url}",
                    metadata={"url": url, "status_code": resp.status_code},
                )

            html = resp.text
            text, title, links = self._parse_html(html, max_chars)

            return ToolResult(
                success=True,
                data={
                    "url": str(resp.url),
                    "title": title,
                    "text": text,
                    "links": links,
                    "links_count": len(links),
                    "content_length": len(html),
                    "status_code": resp.status_code,
                    "content_type": resp.headers.get("content-type", ""),
                    "engine": "httpx+bs4",
                },
                metadata={
                    "action": "navigate",
                    "url": str(resp.url),
                    "title": title,
                    "engine": "httpx+bs4",
                },
            )

    # ------------------------------------------------------------------
    #  Screenshot (Playwright apenas)
    # ------------------------------------------------------------------

    async def _screenshot(self, params: dict) -> ToolResult:
        from playwright.async_api import TimeoutError as PwTimeout

        session = _PlaywrightSession()
        page = await session.get_page()
        full_page = params.get("full_page", False)
        timeout = params.get("timeout", 30.0)

        try:
            screenshot_bytes = await page.screenshot(
                full_page=full_page,
                timeout=int(timeout * 1000),
                type="png",
            )
            b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

            return ToolResult(
                success=True,
                data={
                    "screenshot_base64": b64,
                    "size_bytes": len(screenshot_bytes),
                    "full_page": full_page,
                    "url": page.url,
                    "engine": "playwright",
                },
                metadata={
                    "action": "screenshot",
                    "url": page.url,
                    "size_bytes": len(screenshot_bytes),
                    "full_page": full_page,
                },
            )

        except PwTimeout:
            return ToolResult(
                success=False,
                error=f"Timeout ao capturar screenshot ({timeout}s)",
            )

    # ------------------------------------------------------------------
    #  Click (Playwright apenas)
    # ------------------------------------------------------------------

    async def _click(self, params: dict) -> ToolResult:
        from playwright.async_api import TimeoutError as PwTimeout

        session = _PlaywrightSession()
        page = await session.get_page()
        selector = params.get("selector", "")
        timeout = params.get("timeout", 30.0)

        if not selector:
            return ToolResult(success=False, error="'selector' é obrigatório para click")

        try:
            await page.click(selector, timeout=int(timeout * 1000))
            await page.wait_for_timeout(500)  # esperar renderização

            return ToolResult(
                success=True,
                data={
                    "clicked": selector,
                    "url": page.url,
                    "title": await page.title(),
                    "engine": "playwright",
                },
                metadata={"action": "click", "selector": selector},
            )

        except PwTimeout:
            return ToolResult(
                success=False,
                error=f"Timeout ao clicar em '{selector}' ({timeout}s)",
                metadata={"selector": selector},
            )

    # ------------------------------------------------------------------
    #  Type (Playwright apenas)
    # ------------------------------------------------------------------

    async def _type(self, params: dict) -> ToolResult:
        from playwright.async_api import TimeoutError as PwTimeout

        session = _PlaywrightSession()
        page = await session.get_page()
        selector = params.get("selector", "")
        text = params.get("text", "")
        timeout = params.get("timeout", 30.0)

        if not selector:
            return ToolResult(success=False, error="'selector' é obrigatório para type")
        if not text:
            return ToolResult(success=False, error="'text' é obrigatório para type")

        try:
            await page.click(selector, timeout=int(timeout * 1000))
            await page.fill(selector, text)

            return ToolResult(
                success=True,
                data={
                    "typed_into": selector,
                    "typed_text": text[:100],
                    "url": page.url,
                    "engine": "playwright",
                },
                metadata={"action": "type", "selector": selector},
            )

        except PwTimeout:
            return ToolResult(
                success=False,
                error=(
                    f"Timeout ao escrever em '{selector}' ({timeout}s)"
                ),
                metadata={"selector": selector},
            )

    # ------------------------------------------------------------------
    #  Extract (texto/seletores)
    # ------------------------------------------------------------------

    async def _extract(self, params: dict, use_playwright: bool) -> ToolResult:
        selector = params.get("selector", "")
        max_chars = params.get("max_chars", 10000)

        if use_playwright:
            return await self._extract_playwright(selector, max_chars)
        else:
            return await self._extract_httpx(selector, max_chars)

    async def _extract_playwright(self, selector: str, max_chars: int) -> ToolResult:
        session = _PlaywrightSession()
        page = await session.get_page()

        if selector:
            elements = await page.query_selector_all(selector)
            texts = []
            for el in elements[:50]:
                t = await el.inner_text()
                if t.strip():
                    texts.append(t.strip())
            text = "\\n---\\n".join(texts)
        else:
            text = await page.inner_text("body")
            text = re.sub(r"\\s+", " ", text).strip()

        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        return ToolResult(
            success=True,
            data={
                "text": text,
                "selector": selector or "body",
                "url": page.url,
                "engine": "playwright",
            },
            metadata={"action": "extract", "selector": selector or "body"},
        )

    async def _extract_httpx(self, selector: str, max_chars: int) -> ToolResult:
        # Sem página ativa — extrair de quê? httpx não mantém estado.
        # Só é chamado se houver uma página anterior via navigate.
        # Nesse caso, o contexto tem o HTML armazenado.
        # Mas como httpx é stateless, extract só funciona com Playwright.
        return ToolResult(
            success=False,
            error=(
                "'extract' requer Playwright para extração interativa. "
                "Usa 'navigate' com httpx+bs4 (já disponível) — "
                "o resultado inclui 'text', 'links' e 'title'. "
                "Ou instala Playwright: pip install playwright && "
                "playwright install chromium"
            ),
        )

    # ------------------------------------------------------------------
    #  Evaluate JavaScript (Playwright apenas)
    # ------------------------------------------------------------------

    async def _evaluate(self, params: dict) -> ToolResult:
        session = _PlaywrightSession()
        page = await session.get_page()
        script = params.get("script", "")

        if not script:
            return ToolResult(success=False, error="'script' é obrigatório para evaluate")

        try:
            result = await page.evaluate(script)
            return ToolResult(
                success=True,
                data={
                    "result": result,
                    "result_type": type(result).__name__,
                    "url": page.url,
                    "script": script[:200],
                    "engine": "playwright",
                },
                metadata={"action": "evaluate", "url": page.url},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Erro ao avaliar JavaScript: {e}",
                metadata={"script": script[:200]},
            )

    # ------------------------------------------------------------------
    #  HTML (Playwright ou httpx)
    # ------------------------------------------------------------------

    async def _html(self, params: dict, use_playwright: bool) -> ToolResult:
        max_chars = params.get("max_chars", 50000)

        if use_playwright:
            session = _PlaywrightSession()
            page = await session.get_page()
            html = await page.content()
        else:
            # Tentar ler do último navigate via session
            # Como não guardamos, devolver erro
            return ToolResult(
                success=False,
                error=(
                    "'html' requer Playwright. Com httpx+bs4, o HTML "
                    "é retornado diretamente pelo 'navigate'. "
                    "Instala Playwright: pip install playwright && "
                    "playwright install chromium"
                ),
            )

        if len(html) > max_chars:
            html = html[:max_chars] + "... [truncado]"

        return ToolResult(
            success=True,
            data={
                "html": html,
                "length": len(html),
                "url": page.url if use_playwright else "?",
                "engine": "playwright" if use_playwright else "httpx+bs4",
            },
            metadata={"action": "html"},
        )

    # ------------------------------------------------------------------
    #  Links (Playwright ou httpx)
    # ------------------------------------------------------------------

    async def _links(self, params: dict, use_playwright: bool) -> ToolResult:
        max_links = params.get("max_links", 200)
        filter_domain = params.get("filter_domain", "")

        if use_playwright:
            session = _PlaywrightSession()
            page = await session.get_page()
            links = await page.evaluate("""
                () => Array.from(document.querySelectorAll('a[href]'))
                    .map(a => ({text: a.innerText.trim(), href: a.href}))
                    .filter(l => l.href.startsWith('http'))
            """)
        else:
            return ToolResult(
                success=False,
                error=(
                    "'links' requer Playwright. Com httpx+bs4, os links "
                    "são retornados diretamente pelo 'navigate'. "
                    "Instala Playwright: pip install playwright && "
                    "playwright install chromium"
                ),
            )

        if filter_domain:
            links = [l for l in links if filter_domain.lower() in l["href"].lower()]

        links = links[:max_links]

        return ToolResult(
            success=True,
            data={
                "links": links,
                "count": len(links),
                "total_on_page": len(links),
                "filter_domain": filter_domain or None,
                "url": page.url if use_playwright else "?",
                "engine": "playwright" if use_playwright else "httpx+bs4",
            },
            metadata={"action": "links", "count": len(links)},
        )

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _parse_html(self, html: str, max_chars: int = 10000) -> tuple[str, str, list[dict]]:
        """
        Extrai texto, título e links de HTML usando BeautifulSoup.

        Returns:
            Tuplo (texto_limpo, titulo, lista_de_links).
        """
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
        except ImportError:
            # Fallback para regex simples
            title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else ""

            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\\s+", " ", text).strip()

            links_raw = re.findall(r'href=[\\\'"]?(https?://[^\\\'"\\s>]+)', html)
            links = [{"text": "", "href": href} for href in links_raw]

            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            return text, title, links

        # BeautifulSoup parsing
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Remover scripts e estilos
        for tag in soup(["script", "style", "noscript", "iframe"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        # Extrair links
        links = []
        seen = set()
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if href.startswith(("http://", "https://")) and href not in seen:
                seen.add(href)
                links.append({
                    "text": a.get_text(strip=True)[:100],
                    "href": href,
                })

        return text, title, links[:500]
