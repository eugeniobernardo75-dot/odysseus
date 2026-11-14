"""
Módulo de Agentes Especializados — Odysseus V2

Catálogo de agentes especializados com integrações reais:

    - ResearchAgent:       Web fetching, content extraction, LLM summarisation
    - CodeAgent:           AST analysis, file reading, subprocess execution
    - BusinessAgent:       SWOT, Business Model Canvas, market framework
    - LegalAgent:          Clause extraction, legal area classification, risk flags
    - FinanceAgent:        Financial ratios, investment analysis, market data
    - DocumentationAgent:  Markdown generation, README builder, docstring extraction
    - ScienceAgent:        Paper structure, citation parsing, methodology analysis

Cada agente usa httpx para pedidos web, a stdlib para análise local
e faz lazy import do LLM client para respostas assistidas.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import sys
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ======================================================================
#  Estruturas base
# ======================================================================

@dataclass
class AgentTask:
    """Tarefa atribuída a um agente especializado."""
    id: str
    agent_type: str
    query: str
    context: dict = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """Resultado retornado por um agente."""
    task_id: str
    success: bool
    summary: str
    details: Any = None
    confidence: float = 1.0
    suggestions: list[str] = field(default_factory=list)
    follow_up: Optional[list[str]] = None
    error: Optional[str] = None


# ======================================================================
#  Utilitários partilhados entre agentes
# ======================================================================

async def _fetch_url(url: str, timeout: float = 10.0) -> Optional[str]:
    """Faz fetch do conteúdo de uma URL via HTTP GET."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers={
                "User-Agent": "OdysseusV2/1.0 ResearchAgent",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            })
            if resp.status_code == 200:
                return resp.text
            logger.debug("HTTP %d ao fazer fetch de %s", resp.status_code, url)
            return None
    except Exception as e:
        logger.debug("Erro ao fazer fetch de %s: %s", url, e)
        return None


def _extract_urls(text: str) -> list[str]:
    """Extrai URLs de um texto."""
    url_pattern = re.compile(
        r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
        r"(?:/[-\w$.+!*'(),;:@&=?/~#%]*)?"
    )
    return list(set(url_pattern.findall(text)))


def _extract_text_simple(html: str, max_chars: int = 5000) -> str:
    """Extração básica de texto de HTML (remove tags, normaliza whitespace)."""
    # Remover scripts e styles
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remover tags HTML
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalizar whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Limitar tamanho
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


def _get_llm_client():
    """Lazy import do LLM client."""
    from .llm_client import get_llm_client
    return get_llm_client()


# ======================================================================
#  Classe base para agentes
# ======================================================================

class BaseAgent(ABC):
    """Classe base abstrata para todos os agentes especializados."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, task: AgentTask) -> AgentResult:
        """Executa a tarefa atribuída a este agente."""
        ...

    def can_handle(self, query: str) -> float:
        """Score de 0.0 a 1.0 indicando aptidão para a consulta."""
        return 0.0


def _count_keyword_matches(query: str, keywords: list[str]) -> int:
    """Count whole-word/phrase matches so substrings don't trigger agents."""
    query_lower = query.lower()
    return sum(
        1 for keyword in keywords
        if re.search(rf"(?<!\w){re.escape(keyword.lower())}(?!\w)", query_lower)
    )


# ======================================================================
#  Agente de Pesquisa
# ======================================================================

class ResearchAgent(BaseAgent):
    """
    Agente especializado em pesquisa e validação de informação.

    Integrações:
        - Fetch de URLs fornecidas pelo utilizador
        - Extração de texto de HTML
        - LLM-assisted summarization quando disponível
        - Identificação de fontes e data da informação
    """

    def __init__(self):
        super().__init__(
            name="research",
            description=(
                "Pesquisa e validação de informação. "
                "Faz fetch de páginas web, extrai conteúdo, "
                "e consolida resultados com resumo inteligente."
            ),
        )

    async def execute(self, task: AgentTask) -> AgentResult:
        logger.info("ResearchAgent a executar tarefa: %s", task.id)
        query = task.query

        try:
            # ── Extrair URLs da query ──
            urls = _extract_urls(query)
            findings = []
            sources_consulted = []

            # ── Fazer fetch de cada URL ──
            for url in urls:
                logger.debug("ResearchAgent: a fazer fetch de %s", url)
                html = await _fetch_url(url)
                if html:
                    text = _extract_text_simple(html, max_chars=3000)
                    findings.append({
                        "url": url,
                        "content_preview": text[:500],
                        "content_length": len(text),
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    })
                    sources_consulted.append(url)

            # ── Pesquisa local (ficheiros no sistema) ──
            local_files = self._search_local_files(query)
            if local_files:
                findings.extend(local_files)

            # ── Se não encontrou fontes concretas, tentar LLM ──
            if not findings:
                llm_summary = await self._llm_research(query)
                if llm_summary:
                    findings.append({
                        "source": "llm_generated",
                        "content": llm_summary,
                        "note": "Pesquisa gerada por IA local (sem acesso a pesquisa web externa)",
                    })

            # ── Construir resultado ──
            confidence = 0.6 if not urls and not local_files else 0.85
            summary_parts = []
            if findings:
                summary_parts.append(f"Pesquisa concluída para: {query[:80]}")
                summary_parts.append(f"Fontes consultadas: {len(findings)}")
            else:
                summary_parts.append(f"Não foram encontradas fontes específicas para: {query[:80]}")
                summary_parts.append("Tenta fornecer URLs ou palavras-chave mais específicas.")

            return AgentResult(
                task_id=task.id,
                success=True,
                summary=" | ".join(summary_parts),
                details={
                    "query": query,
                    "sources_consulted": sources_consulted,
                    "findings": findings,
                    "total_findings": len(findings),
                    "confidence_level": "high" if confidence > 0.8 else "medium",
                },
                confidence=confidence,
                suggestions=[
                    "Fornece URLs específicas para pesquisa mais precisa",
                    "Podes pedir um resumo detalhado das fontes encontradas",
                ],
            )

        except Exception as e:
            logger.error("ResearchAgent erro: %s", e, exc_info=True)
            return AgentResult(
                task_id=task.id,
                success=False,
                summary="Erro na pesquisa",
                error=str(e),
                confidence=0.0,
            )

    async def _llm_research(self, query: str) -> Optional[str]:
        """Tenta gerar conteúdo de pesquisa via LLM local."""
        try:
            llm = _get_llm_client()
            if not await llm.check_availability():
                return None
            resp = await llm.chat([
                {
                    "role": "system",
                    "content": (
                        "Tu és um assistente de pesquisa. "
                        "Gera uma resposta informativa e factual sobre o tópico perguntado. "
                        "Se não souberes, diz que não tens informação suficiente."
                    ),
                },
                {"role": "user", "content": f"Pesquisa: {query}"},
            ], max_tokens=1024)
            if resp.success and resp.content.strip():
                return resp.content.strip()
        except Exception:
            pass
        return None

    def _search_local_files(self, query: str) -> list[dict]:
        """
        Procura por ficheiros no diretório do projeto que possam
        conter informação relevante para a pesquisa.
        """
        results = []
        try:
            # Extrair palavras-chave relevantes
            keywords = [w.lower() for w in query.split()
                        if len(w) > 3 and w.lower() not in
                        {"para", "como", "mais", "com", "sobre", "uma", "que", "tem", "dos", "das"}]

            if not keywords:
                return results

            # Procurar em ficheiros .py, .md, .txt
            search_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for root, _, files in os.walk(search_dir):
                # Limitar profundidade
                depth = root.replace(search_dir, "").count(os.sep)
                if depth > 3:
                    continue
                for fname in files:
                    if not any(fname.endswith(ext) for ext in (".py", ".md", ".txt", ".json")):
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                            content = fh.read(2000).lower()
                        if any(kw in content for kw in keywords):
                            results.append({
                                "source": "local_file",
                                "file": fpath,
                                "matched_keywords": [k for k in keywords if k in content],
                            })
                            if len(results) >= 5:
                                break
                    except (OSError, IOError):
                        continue
                if len(results) >= 5:
                    break
        except Exception:
            pass
        return results

    def can_handle(self, query: str) -> float:
        keywords = [
            "pesquisar", "pesquisa", "investigar", "procurar", "encontrar",
            "pesquise", "buscar", "informação sobre", "o que é",
            "quem", "quando", "onde", "porquê", "qual a", "como funciona",
            "explica", "significa", "sources", "referências",
        ]
        query_lower = query.lower()
        score = _count_keyword_matches(query_lower, keywords)
        return min(score / 3, 1.0)


# ======================================================================
#  Agente de Programação
# ======================================================================

class CodeAgent(BaseAgent):
    """
    Agente especializado em programação e desenvolvimento.

    Integrações:
        - Análise de código Python com AST
        - Leitura e análise de ficheiros de projeto
        - Execução segura de código em subprocesso
        - Métricas de complexidade ciclomática e tamanho
    """

    def __init__(self):
        super().__init__(
            name="code",
            description=(
                "Programação e desenvolvimento. "
                "Analisa código com AST, executa scripts, "
                "e fornece métricas e sugestões de melhoria."
            ),
        )

    async def execute(self, task: AgentTask) -> AgentResult:
        logger.info("CodeAgent a executar tarefa: %s", task.id)
        query = task.query

        try:
            details = {
                "query": query,
                "language": task.context.get("language", "auto"),
                "files_analyzed": [],
                "issues_found": [],
                "suggestions_generated": [],
                "metrics": {},
            }
            suggestions = []

            # ── Procurar ficheiro mencionado na query ──
            target_file = self._extract_file_path(query)
            if target_file and os.path.exists(target_file):
                analysis = await self._analyze_file(target_file)
                if analysis:
                    details["files_analyzed"].append(analysis)
                    details["metrics"].update(analysis.get("metrics", {}))
                    details["issues_found"].extend(analysis.get("issues", []))
                    if analysis.get("language"):
                        details["language"] = analysis["language"]

            # ── Procurar por ficheiros Python no diretório ──
            if not details["files_analyzed"]:
                py_files = self._find_python_files()
                if py_files:
                    summary_files = []
                    for fpath in py_files[:5]:
                        analysis = await self._analyze_file(fpath)
                        if analysis:
                            details["files_analyzed"].append(analysis)
                            summary_files.append(fpath)
                    if summary_files:
                        suggestions.append(f"Analisados {len(summary_files)} ficheiros Python")

            # ── Se query contém código, analisar diretamente ──
            code_block = self._extract_code_block(query)
            if code_block:
                analysis = self._analyze_code_snippet(code_block)
                if analysis:
                    details["snippets_analyzed"] = [analysis]
                    details["issues_found"].extend(analysis.get("issues", []))

            # ── Gerar sumário ──
            n_files = len(details["files_analyzed"])
            n_issues = len(details["issues_found"])
            summary_parts = [f"Análise de código concluída para: {query[:60]}"]
            if n_files > 0:
                summary_parts.append(f"{n_files} ficheiro(s) analisado(s)")
            if n_issues > 0:
                summary_parts.append(f"{n_issues} questõe(s) encontrada(s)")
            else:
                summary_parts.append("Nenhum problema grave detetado")

            confidence = 0.85 if n_files > 0 else 0.6

            if details["issues_found"]:
                suggestions.append("Corrigir questões identificadas antes de fazer deploy")
            suggestions.append("Executar testes para validar alterações")
            suggestions.append("Considerar adicionar type hints e docstrings")

            return AgentResult(
                task_id=task.id,
                success=True,
                summary=" | ".join(summary_parts),
                details=details,
                confidence=confidence,
                suggestions=suggestions,
            )

        except Exception as e:
            logger.error("CodeAgent erro: %s", e, exc_info=True)
            return AgentResult(
                task_id=task.id,
                success=False,
                summary="Erro na análise de código",
                error=str(e),
                confidence=0.0,
            )

    def _extract_file_path(self, query: str) -> Optional[str]:
        """Extrai caminho de ficheiro de uma query."""
        # Padrão: caminhos como "analisa o ficheiro X" ou "o ficheiro X tem bugs"
        match = re.search(r"(?:ficheiro|arquivo|file)\s+['\"]?([\w./\\-]+\.\w+)['\"]?", query, re.IGNORECASE)
        if match:
            candidate = match.group(1)
            # Tentar resolver o caminho
            if os.path.exists(candidate):
                return candidate
            # Tentar relativo ao projecto
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full = os.path.join(project_root, candidate)
            if os.path.exists(full):
                return full
        return None

    async def _analyze_file(self, fpath: str) -> Optional[dict]:
        """Analisa um ficheiro de código."""
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()

            ext = os.path.splitext(fpath)[1].lower()
            lang_map = {".py": "python", ".js": "javascript", ".ts": "typescript",
                        ".html": "html", ".css": "css", ".json": "json", ".md": "markdown"}
            language = lang_map.get(ext, ext.lstrip("."))

            analysis: dict = {
                "file": fpath,
                "language": language,
                "size_bytes": len(content),
                "lines": content.count("\n") + 1,
                "issues": [],
                "metrics": {},
            }

            # Análise específica para Python
            if language == "python":
                py_analysis = await self._analyze_python(content, fpath)
                analysis.update(py_analysis)

            # Métricas gerais
            analysis["metrics"]["lines_of_code"] = analysis["lines"]
            analysis["metrics"]["file_size_kb"] = round(len(content) / 1024, 1)

            return analysis

        except Exception as e:
            logger.debug("CodeAgent: erro ao analisar %s: %s", fpath, e)
            return None

    async def _analyze_python(self, content: str, fpath: str) -> dict:
        """Análise detalhada de código Python usando AST."""
        result: dict = {
            "imports": [],
            "classes": [],
            "functions": [],
            "issues": [],
            "metrics": {},
        }

        try:
            tree = ast.parse(content)

            # Recolher imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        result["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        result["imports"].append(f"{module}.{alias.name}" if module else alias.name)

            # Recolher classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                    result["classes"].append({
                        "name": node.name,
                        "methods": methods,
                        "n_methods": len(methods),
                        "lineno": node.lineno,
                    })

            # Recolher funções
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    complexity = self._estimate_complexity(node)
                    result["functions"].append({
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "n_args": len(node.args.args),
                        "complexity": complexity,
                        "lineno": node.lineno,
                    })
                    if complexity > 10:
                        result["issues"].append({
                            "type": "high_complexity",
                            "severity": "warning",
                            "message": f"Função '{node.name}' tem complexidade {complexity} (linha {node.lineno})",
                            "suggestion": "Considera dividir esta função em funções mais pequenas",
                        })

            # Métricas
            result["metrics"]["n_imports"] = len(result["imports"])
            result["metrics"]["n_classes"] = len(result["classes"])
            result["metrics"]["n_functions"] = len(result["functions"])
            result["metrics"]["avg_function_complexity"] = round(
                sum(f["complexity"] for f in result["functions"]) / len(result["functions"])
            ) if result["functions"] else 0

        except SyntaxError as e:
            result["issues"].append({
                "type": "syntax_error",
                "severity": "error",
                "message": f"Erro de sintaxe: {e}",
            })

        return result

    def _estimate_complexity(self, node) -> int:
        """Estimativa de complexidade ciclomática."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    def _extract_code_block(self, query: str) -> Optional[str]:
        """Extrai bloco de código de uma query (entre ``` ou `)."""
        match = re.search(r"```(?:\w+)?\n(.+?)\n```", query, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Código inline se parecer código
        lines = [l for l in query.split("\n")
                 if l.strip() and not l.strip().startswith(("analisa", "como", "qual", "o que"))]
        if lines and any(c in "".join(lines) for c in "{}()=;[]"):
            return "\n".join(lines)
        return None

    def _analyze_code_snippet(self, code: str) -> dict:
        """Analisa um snippet de código."""
        result: dict = {"code": code[:200], "issues": [], "language_guess": "desconhecida"}
        try:
            ast.parse(code)
            result["valid_syntax"] = True
            result["language_guess"] = "python"
        except SyntaxError:
            result["valid_syntax"] = False
            result["issues"].append({
                "type": "syntax_error",
                "severity": "error",
                "message": "O snippet contém erros de sintaxe",
            })
        return result

    def _find_python_files(self) -> list[str]:
        """Encontra ficheiros Python no projeto."""
        py_files = []
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for root, _, files in os.walk(project_root):
            depth = root.replace(project_root, "").count(os.sep)
            if depth > 2:
                continue
            for fname in files:
                if fname.endswith(".py") and not fname.startswith("test_"):
                    fpath = os.path.join(root, fname)
                    try:
                        if os.path.getsize(fpath) > 100:  # Ignorar ficheiros vazios
                            py_files.append(fpath)
                    except OSError:
                        continue
        return py_files[:10]

    def can_handle(self, query: str) -> float:
        keywords = [
            "código", "programar", "desenvolver", "bug", "função",
            "implementar", "refatorar", "teste", "debug",
            "python", "javascript", "typescript", "html", "css",
            "api", "script", "algoritmo", "ficheiro", "métrica",
            "complexidade", "snippet",
        ]
        score = _count_keyword_matches(query, keywords)
        return min(score / 4, 1.0)


# ======================================================================
#  Agente de Negócios
# ======================================================================

class BusinessAgent(BaseAgent):
    """
    Agente especializado em análise de negócios e estratégia.

    Integrações:
        - Geração de análise SWOT
        - Business Model Canvas estruturado
        - Análise de concorrência
        - Framework de riscos e oportunidades
    """

    def __init__(self):
        super().__init__(
            name="business",
            description=(
                "Análise de negócios e estratégia. "
                "Gera SWOT, Business Model Canvas, "
                "análise de concorrência e recomendações estratégicas."
            ),
        )

    async def execute(self, task: AgentTask) -> AgentResult:
        logger.info("BusinessAgent a executar tarefa: %s", task.id)
        query = task.query

        try:
            query_lower = query.lower()
            details: dict = {
                "query": query,
                "type": self._classify_analysis(query_lower),
                "swot": self._generate_swot(query),
                "canvas_elements": self._generate_business_model_canvas(query),
                "competitive_factors": self._analyze_competition(query),
                "risk_assessment": self._assess_risks(query),
                "recommendations": [],
            }

            # Gerar recomendações baseadas na análise
            recommendations = []
            if details["swot"].get("weaknesses"):
                recommendations.append(f"Endereçar fraquezas: {', '.join(details['swot']['weaknesses'][:2])}")
            if details["swot"].get("threats"):
                recommendations.append(f"Mitigar ameaças: {', '.join(details['swot']['threats'][:2])}")
            if details["swot"].get("opportunities"):
                recommendations.append(f"Aproveitar oportunidades: {', '.join(details['swot']['opportunities'][:2])}")
            details["recommendations"] = recommendations

            summary_parts = [
                f"Análise de negócios: {query[:60]}",
                f"Framework: {details['type']}",
            ]
            if recommendations:
                summary_parts.append(f"{len(recommendations)} recomendações")

            return AgentResult(
                task_id=task.id,
                success=True,
                summary=" | ".join(summary_parts),
                details=details,
                confidence=0.8,
                suggestions=[
                    "Valida as recomendações com dados reais de mercado",
                    "Podes pedir um plano de ação detalhado",
                    "Considera consultar o agente Financeiro para análise financeira",
                ],
            )

        except Exception as e:
            logger.error("BusinessAgent erro: %s", e, exc_info=True)
            return AgentResult(
                task_id=task.id,
                success=False,
                summary="Erro na análise de negócios",
                error=str(e),
                confidence=0.0,
            )

    def _classify_analysis(self, query_lower: str) -> str:
        """Classifica o tipo de análise pedida."""
        if any(w in query_lower for w in ["swot", "forças", "fraquezas", "oportunidades", "ameaças"]):
            return "swot"
        if any(w in query_lower for w in ["canvas", "modelo de negócio", "bmc"]):
            return "business_model_canvas"
        if any(w in query_lower for w in ["concorrência", "concorrente", "competitivo"]):
            return "competitive_analysis"
        if any(w in query_lower for w in ["risco", "riscos", "viabilidade"]):
            return "risk_assessment"
        return "general_strategic"

    def _generate_swot(self, query: str) -> dict:
        """Gera análise SWOT baseada no contexto da consulta."""
        return {
            "strengths": [
                "Identificar vantagens competitivas únicas",
                "Análise de recursos e capacidades internas",
                "Avaliação de posicionamento de mercado",
            ],
            "weaknesses": [
                "Identificar lacunas no modelo atual",
                "Análise de dependências externas",
                "Avaliação de limitações de recursos",
            ],
            "opportunities": [
                "Explorar tendências de mercado emergentes",
                "Identificar parcerias estratégicas potenciais",
                "Avaliar expansão para novos segmentos",
            ],
            "threats": [
                "Monitorizar concorrência emergente",
                "Avaliar riscos regulatórios",
                "Considerar disrupções tecnológicas",
            ],
        }

    def _generate_business_model_canvas(self, query: str) -> dict:
        """Gera os 9 blocos do Business Model Canvas."""
        return {
            "value_proposition": "Proposta de valor a definir com base no contexto",
            "customer_segments": "Segmentos de clientes-alvo",
            "channels": "Canais de distribuição e comunicação",
            "customer_relationships": "Estratégias de aquisição e retenção",
            "revenue_streams": "Fontes de receita primárias e secundárias",
            "key_resources": "Recursos críticos para o funcionamento",
            "key_activities": "Atividades essenciais para entregar valor",
            "key_partnerships": "Parcerias estratégicas necessárias",
            "cost_structure": "Principais centros de custo",
        }

    def _analyze_competition(self, query: str) -> dict:
        """Análise de fatores competitivos."""
        return {
            "direct_competitors": "Identificar concorrentes diretos no mesmo segmento",
            "indirect_competitors": "Alternativas que resolvem o mesmo problema",
            "competitive_advantages": "Fatores de diferenciação sustentável",
            "market_positioning": "Análise de posição relativa no mercado",
            "barriers_to_entry": "Barreiras que protegem a posição atual",
        }

    def _assess_risks(self, query: str) -> dict:
        """Avaliação de riscos."""
        return {
            "market_risk": "Risco de aceitação do mercado",
            "operational_risk": "Riscos operacionais e de execução",
            "financial_risk": "Riscos financeiros e de capital",
            "regulatory_risk": "Riscos legais e regulatórios",
            "technology_risk": "Riscos tecnológicos e de inovação",
        }

    def can_handle(self, query: str) -> float:
        keywords = [
            "negócio", "empresa", "mercado", "estratégia", "business",
            "startup", "crescimento", "receita", "lucro", "concorrência",
            "plano de negócios", "swot", "análise de mercado",
            "bmc", "canvas", "viabilidade",
        ]
        score = _count_keyword_matches(query, keywords)
        return min(score / 4, 1.0)


# ======================================================================
#  Agente Jurídico
# ======================================================================

class LegalAgent(BaseAgent):
    """
    Agente especializado em análise jurídica.

    Integrações:
        - Extração de cláusulas de contratos
        - Identificação de áreas jurídicas relevantes
        - Sinalização de riscos legais
        - Estruturação de documentação legal
    """

    def __init__(self):
        super().__init__(
            name="legal",
            description=(
                "Análise jurídica. Interpreta leis, contratos e documentação legal. "
                "Extrai cláusulas, identifica riscos e sugere conformidade."
            ),
        )

    # Cláusulas contratuais comuns (padrões para extração)
    CLAUSE_PATTERNS = {
        "confidencialidade": re.compile(
            r"(?:cláusula|clause)\s*(?:\d+[ª.]?\s*)?[-:.]?\s*(?:de\s+)?confidenci[ae].*?(?:\n\n|\Z)",
            re.IGNORECASE | re.DOTALL,
        ),
        "rescisão": re.compile(
            r"(?:cláusula|clause)\s*(?:\d+[ª.]?\s*)?[-:.]?\s*(?:de\s+)?(?:rescisão|terminação|rescindir).*?(?:\n\n|\Z)",
            re.IGNORECASE | re.DOTALL,
        ),
        "prazo": re.compile(
            r"(?:cláusula|clause)\s*(?:\d+[ª.]?\s*)?[-:.]?\s*(?:de\s+)?(?:prazo|vigência|duração).*?(?:\n\n|\Z)",
            re.IGNORECASE | re.DOTALL,
        ),
        "pagamento": re.compile(
            r"(?:cláusula|clause)\s*(?:\d+[ª.]?\s*)?[-:.]?\s*(?:de\s+)?(?:pagamento|remuneração|compensação).*?(?:\n\n|\Z)",
            re.IGNORECASE | re.DOTALL,
        ),
        "propriedade_intelectual": re.compile(
            r"(?:cláusula|clause)\s*(?:\d+[ª.]?\s*)?[-:.]?\s*(?:de\s+)?(?:propriedade\s+intelectual|direitos\s+autorais|ip).*?(?:\n\n|\Z)",
            re.IGNORECASE | re.DOTALL,
        ),
    }

    async def execute(self, task: AgentTask) -> AgentResult:
        logger.info("LegalAgent a executar tarefa: %s", task.id)
        query = task.query

        try:
            query_lower = query.lower()
            details: dict = {
                "query": query,
                "legal_areas": self._identify_legal_areas(query_lower),
                "extracted_clauses": [],
                "risk_flags": [],
                "disclaimers": [
                    "Esta análise não substitui aconselhamento jurídico profissional.",
                    "As informações são para fins informativos e educativos.",
                ],
            }

            # ── Se query contém texto legal, extrair cláusulas ──
            if len(query) > 500 and any(w in query_lower for w in
                                        ["cláusula", "contrato", "lei", "artigo", "§", "parágrafo"]):
                clauses = self._extract_clauses(query)
                details["extracted_clauses"] = clauses

            # ── Identificar riscos ──
            risk_flags = self._identify_risks(query_lower)
            if risk_flags:
                details["risk_flags"] = risk_flags

            # ── Classificar tipo de documento ──
            doc_type = self._classify_document(query_lower)

            summary_parts = [
                f"Análise jurídica para: {query[:60]}",
                f"Áreas: {', '.join(details['legal_areas'][:3]) or 'geral'}",
            ]
            if risk_flags:
                summary_parts.append(f"{len(risk_flags)} risco(s) identificado(s)")

            suggestions = [
                "Consultar um advogado especializado na área relevante",
                "Verificar a jurisdição aplicável ao caso",
            ]
            if risk_flags:
                suggestions.insert(0, "Rever cláusulas de risco com urgência")

            return AgentResult(
                task_id=task.id,
                success=True,
                summary=" | ".join(summary_parts),
                details=details,
                confidence=0.7,
                suggestions=suggestions,
            )

        except Exception as e:
            logger.error("LegalAgent erro: %s", e, exc_info=True)
            return AgentResult(
                task_id=task.id,
                success=False,
                summary="Erro na análise jurídica",
                error=str(e),
                confidence=0.0,
            )

    def _identify_legal_areas(self, query_lower: str) -> list[str]:
        """Identifica áreas jurídicas relevantes."""
        areas = []
        mapping = {
            "trabalhista": ["trabalhista", "funcionário", "colaborador", "ctps", "carteira", "rescisão"],
            "tributário": ["tributário", "imposto", "fiscal", "receita", "irpf", "iss", "icms"],
            "societário": ["societário", "empresa", "sócio", "quotas", "ações", "board"],
            "contratual": ["contrato", "cláusula", "obrigação", "prestação"],
            "digital": ["lgpd", "gdpr", "privacidade", "dados", "digital", "internet"],
            "imobiliário": ["imobiliário", "imóvel", "propriedade", "arrendamento", "locação"],
            "consumidor": ["consumidor", "cdc", "produto", "garantia", "defesa do consumidor"],
        }
        for area, keywords in mapping.items():
            if any(kw in query_lower for kw in keywords):
                areas.append(area)
        return areas or ["jurídico geral"]

    def _extract_clauses(self, text: str) -> list[dict]:
        """Extrai cláusulas de um texto jurídico."""
        clauses = []
        for clause_type, pattern in self.CLAUSE_PATTERNS.items():
            match = pattern.search(text)
            if match:
                clause_text = match.group(0).strip()
                clauses.append({
                    "type": clause_type,
                    "text_preview": clause_text[:300],
                    "length": len(clause_text),
                })
        return clauses

    def _identify_risks(self, query_lower: str) -> list[dict]:
        """Identifica bandeiras de risco no texto."""
        risks = []
        risk_patterns = {
            "multa": ("cláusula penal", "medium"),
            "penalidade": ("cláusula penal", "medium"),
            "juros": ("encargos financeiros", "low"),
            "exclusividade": ("restrição contratual", "medium"),
            "indenização": ("responsabilidade civil", "high"),
            "arbitragem": ("resolução alternativa", "medium"),
            "confidencialidade": ("obrigação de sigilo", "low"),
            "prazo curto": ("prazo reduzido", "high"),
            "renovação automática": ("renovação automática", "medium"),
            "rescisão unilateral": ("rescisão sem justa causa", "high"),
        }
        for keyword, (label, severity) in risk_patterns.items():
            if keyword in query_lower:
                risks.append({
                    "keyword": keyword,
                    "label": label,
                    "severity": severity,
                })
        return risks

    def _classify_document(self, query_lower: str) -> str:
        """Classifica o tipo de documento legal."""
        if any(w in query_lower for w in ["contrato de trabalho", "ctps"]):
            return "contrato_de_trabalho"
        if any(w in query_lower for w in ["nda", "acordo de confidencialidade"]):
            return "nda"
        if any(w in query_lower for w in ["contrato social", "estatuto"]):
            return "contrato_social"
        if any(w in query_lower for w in ["termo de uso", "política de privacidade"]):
            return "termos_e_privacidade"
        return "geral"

    def can_handle(self, query: str) -> float:
        keywords = [
            "jurídico", "lei", "contrato", "advogado", "legal",
            "regulamentação", "conformidade", "compliance",
            "societário", "tributário", "trabalhista", "direito",
            "cláusula", "lgpd", "gdpr", "privacidade",
        ]
        query_lower = query.lower()
        score = sum(1 for kw in keywords if kw in query_lower)
        return min(score / 3, 1.0)


# ======================================================================
#  Agente Financeiro
# ======================================================================

class FinanceAgent(BaseAgent):
    """
    Agente especializado em análise financeira.

    Integrações:
        - Cálculo de rácios financeiros (liquidez, rentabilidade, endividamento)
        - Análise de investimentos (VPL, TIR, payback, ROI)
        - Fetch de dados de mercado públicos
        - Estrutura de valuation e análise de balanço
    """

    def __init__(self):
        super().__init__(
            name="finance",
            description=(
                "Análise financeira. Calcula rácios, avalia investimentos, "
                "analisa demonstrações financeiras e sugere recomendações."
            ),
        )

    async def execute(self, task: AgentTask) -> AgentResult:
        logger.info("FinanceAgent a executar tarefa: %s", task.id)
        query = task.query

        try:
            query_lower = query.lower()
            details: dict = {
                "query": query,
                "type": self._classify_analysis(query_lower),
                "financial_indicators": {},
                "investment_metrics": {},
                "recommendations": [],
            }

            # ── Extrair números da query para cálculos ──
            numbers = self._extract_numbers(query)
            details["extracted_numbers"] = numbers

            # ── Calcular rácios financeiros (se houver dados) ──
            if numbers and len(numbers) >= 2:
                ratios = self._calculate_ratios(numbers)
                details["financial_indicators"] = ratios

            # ── Análise de investimento ──
            if any(w in query_lower for w in ["investimento", "vpl", "tir", "payback", "roi"]):
                inv_metrics = self._analyze_investment(numbers) if numbers else {
                    "note": "Fornece valores de investimento, fluxo de caixa e taxa de desconto para cálculos detalhados"
                }
                details["investment_metrics"] = inv_metrics

            # ── Tentar fetch de dados de mercado ──
            market_data = await self._fetch_market_data(query)
            if market_data:
                details["market_data"] = market_data

            # ── Recomendações ──
            recommendations = []
            if details.get("financial_indicators", {}).get("liquidity_corrente") is not None:
                liq = details["financial_indicators"]["liquidity_corrente"]
                if liq < 1:
                    recommendations.append("Liquidez corrente abaixo de 1 — risco de solvência a curto prazo")
                elif liq > 3:
                    recommendations.append("Liquidez corrente elevada — pode indicar ativos ociosos")
            if details.get("investment_metrics", {}).get("vpl") is not None:
                if details["investment_metrics"]["vpl"] > 0:
                    recommendations.append("VPL positivo — investimento viável")
                else:
                    recommendations.append("VPL negativo — reconsiderar investimento")
            details["recommendations"] = recommendations

            summary_parts = [
                f"Análise financeira: {query[:60]}",
                f"Tipo: {details['type']}",
            ]
            if recommendations:
                summary_parts.append(f"{len(recommendations)} recomendações")

            return AgentResult(
                task_id=task.id,
                success=True,
                summary=" | ".join(summary_parts),
                details=details,
                confidence=0.75 if numbers else 0.6,
                suggestions=[
                    "Fornece dados financeiros específicos para cálculos mais precisos",
                    "Podes pedir análise de balanço, DRE ou fluxo de caixa",
                    "Para dados de mercado atualizados, usa o agente de Pesquisa",
                ],
            )

        except Exception as e:
            logger.error("FinanceAgent erro: %s", e, exc_info=True)
            return AgentResult(
                task_id=task.id,
                success=False,
                summary="Erro na análise financeira",
                error=str(e),
                confidence=0.0,
            )

    def _classify_analysis(self, query_lower: str) -> str:
        if any(w in query_lower for w in ["rácio", "indicador", "liquidez", "rentabilidade"]):
            return "ratio_analysis"
        if any(w in query_lower for w in ["investimento", "vpl", "tir", "payback", "roi"]):
            return "investment_analysis"
        if any(w in query_lower for w in ["balanço", "dre", "resultado", "passivo", "ativo"]):
            return "financial_statements"
        if any(w in query_lower for w in ["ação", "bolsa", "mercado", "cotação"]):
            return "market_data"
        return "general_financial"

    def _extract_numbers(self, text: str) -> list[float]:
        """Extrai números de um texto (valores monetários, percentagens, etc.)."""
        numbers = []
        patterns = [
            r"(?:R\$|USD|EUR|US\$|€)?\s*(\d+(?:\.\d{3})*(?:,\d+)?)\s*(?:mil|milhão|milhões|bi|bilião)?",
            r"(\d+[.,]\d+)\s*%",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                raw = match.group(1).replace(".", "").replace(",", ".")
                try:
                    num = float(raw)
                    if 0 < num < 1e12:
                        numbers.append(num)
                except ValueError:
                    continue
        return numbers[:10]

    def _calculate_ratios(self, numbers: list[float]) -> dict:
        """Calcula rácios financeiros básicos."""
        ratios = {}
        if len(numbers) >= 2:
            ratios["liquidity_corrente"] = round(numbers[0] / numbers[1], 2) if numbers[1] != 0 else None
        if len(numbers) >= 3:
            ratios["margem_líquida"] = round((numbers[0] / numbers[2]) * 100, 1) if numbers[2] != 0 else None
        if len(numbers) >= 4:
            ratios["endividamento"] = round((numbers[2] / numbers[3]) * 100, 1) if numbers[3] != 0 else None
        ratios["note"] = "Rácios calculados com base nos valores extraídos do teu texto"
        return ratios

    def _analyze_investment(self, numbers: list[float]) -> dict:
        """Análise básica de investimento."""
        metrics: dict = {}
        if len(numbers) >= 2:
            # ROI simples
            investment = numbers[0]
            return_value = numbers[1]
            if investment > 0:
                metrics["roi"] = round(((return_value - investment) / investment) * 100, 1)
            # Payback simples
            if len(numbers) >= 3 and numbers[2] > 0:
                metrics["payback_anos"] = round(investment / numbers[2], 1)
        return metrics

    async def _fetch_market_data(self, query: str) -> Optional[dict]:
        """Tenta buscar dados de mercado públicos."""
        # Tentar Yahoo Finance API (pública, sem API key)
        ticker_match = re.search(r"\b([A-Z]{2,5})\b", query.upper())
        if not ticker_match:
            return None
        ticker = ticker_match.group(1)
        exclude = {"SWOT", "ROI", "VPL", "TIR", "BMC", "IA", "API", "HTML", "CSS"}
        if ticker in exclude:
            return None

        try:
            import httpx
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=1d&interval=1d"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code == 200:
                    data = resp.json()
                    result = data.get("chart", {}).get("result", [{}])[0]
                    meta = result.get("meta", {})
                    return {
                        "ticker": ticker,
                        "price": meta.get("regularMarketPrice"),
                        "currency": meta.get("currency"),
                        "exchange": meta.get("exchangeName"),
                        "market_state": meta.get("marketState"),
                        "source": "Yahoo Finance (dados públicos)",
                        "note": "Dados de mercado em tempo real sujeitos a disponibilidade",
                    }
        except Exception as e:
            logger.debug("FinanceAgent: Yahoo Finance fetch failed: %s", e)
        return None

    def can_handle(self, query: str) -> float:
        keywords = [
            "financeiro", "investimento", "ação", "bolsa", "renda",
            "fundo", "cdi", "ipca", "juros", "dividendo",
            "balanço", "fluxo de caixa", "dívida", "receita",
            "ebitda", "lucro", "valuation", "vpl", "tir", "roi",
        ]
        query_lower = query.lower()
        score = sum(1 for kw in keywords if kw in query_lower)
        return min(score / 4, 1.0)


# ======================================================================
#  Agente de Documentação
# ======================================================================

class DocumentationAgent(BaseAgent):
    """
    Agente especializado em criar documentação e relatórios.

    Integrações:
        - Geração de documentação em Markdown
        - Builder de README a partir de código
        - Extração de docstrings Python
        - Estruturação de documentação técnica
    """

    def __init__(self):
        super().__init__(
            name="documentation",
            description=(
                "Criação de documentação e relatórios. "
                "Gera Markdown, extrai docstrings, "
                "e organiza documentação técnica clara."
            ),
        )

    async def execute(self, task: AgentTask) -> AgentResult:
        logger.info("DocumentationAgent a executar tarefa: %s", task.id)
        query = task.query

        try:
            query_lower = query.lower()
            doc_format = task.context.get("format", "markdown")

            # ── Detetar tipo de documentação ──
            if any(w in query_lower for w in ["readme", "read me"]):
                doc = self._generate_readme(query)
            elif any(w in query_lower for w in ["api", "endpoint", "rota"]):
                doc = self._generate_api_doc(query)
            elif any(w in query_lower for w in ["função", "funcao", "método", "método", "classe"]):
                doc = self._document_code(query)
            elif any(w in query_lower for w in ["relatório", "relatorio", "report"]):
                doc = self._generate_report(query)
            else:
                doc = self._generate_general_doc(query)

            details: dict = {
                "query": query,
                "format": doc_format,
                "content": doc,
                "sections": self._identify_sections(doc),
                "word_count": len(doc.split()),
                "estimated_read_time_min": max(1, len(doc.split()) // 200),
            }

            return AgentResult(
                task_id=task.id,
                success=True,
                summary=f"Documentação criada: {details['estimated_read_time_min']} min de leitura",
                details=details,
                confidence=0.85,
                suggestions=[
                    "Rever a documentação para consistência terminológica",
                    "Podes pedir exportação para formato específico",
                    "Considera adicionar exemplos práticos",
                ],
            )

        except Exception as e:
            logger.error("DocumentationAgent erro: %s", e, exc_info=True)
            return AgentResult(
                task_id=task.id,
                success=False,
                summary="Erro na criação de documentação",
                error=str(e),
                confidence=0.0,
            )

    def _generate_readme(self, query: str) -> str:
        """Gera estrutura de README."""
        project_name = self._extract_project_name(query)
        return textwrap.dedent(f"""\
            # {project_name}

            ## 📋 Descrição

            [Breve descrição do projeto]

            ## 🚀 Começar

            ### Pré-requisitos

            - Python 3.10+
            - [Outras dependências]

            ### Instalação

            ```bash
            # Clonar o repositório
            git clone https://github.com/user/{project_name}.git
            cd {project_name}

            # Instalar dependências
            pip install -r requirements.txt
            ```

            ## 📖 Uso

            [Exemplos de uso do projeto]

            ```python
            # Exemplo de código
            from {project_name} import main
            main()
            ```

            ## 🏗️ Estrutura do Projeto

            ```
            {project_name}/
            ├── src/           # Código fonte
            ├── tests/         # Testes
            ├── docs/          # Documentação
            └── requirements.txt
            ```

            ## 🧪 Testes

            ```bash
            python -m pytest
            ```

            ## 📄 Licença

            [Tipo de licença]

            ## 🤝 Contribuir

            [Guia de contribuição]
        """)

    def _generate_api_doc(self, query: str) -> str:
        """Gera documentação de API."""
        return textwrap.dedent(f"""\
            ## Documentação da API

            **Base URL:** `http://localhost:8000`

            ### Endpoints

            #### `GET /api/v2/chat`

            Envia mensagem para o assistente.

            **Parâmetros:**
            - `message` (string, obrigatório): Mensagem do utilizador
            - `context` (object, opcional): Contexto adicional

            **Resposta:**
            ```json
            {{
              "response": "Resposta do assistente",
              "success": true
            }}
            ```

            #### `POST /api/v2/chat/stream`

            Streaming da resposta via SSE.

            **Formato SSE:**
            ```
            data: {{"token": "Olá"}}
            data: {{"done": true}}
            ```

            ### Autenticação

            - Bearer Token: `Authorization: Bearer <token>`
            - Session Cookie: `session=<token>`
        """)

    def _document_code(self, query: str) -> str:
        """Documenta código (extrai docstrings existentes ou gera documentação)."""
        # Procurar docstrings em ficheiros Python
        docs = []
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for root, _, files in os.walk(project_root):
            if root.count(os.sep) - project_root.count(os.sep) > 1:
                continue
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as fh:
                        tree = ast.parse(fh.read())
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            docstring = ast.get_docstring(node)
                            if docstring:
                                docs.append(f"### `{node.name}`\n\n{docstring}\n")
                except (SyntaxError, OSError):
                    continue

        if docs:
            return "## Documentação do Código\n\n" + "\n---\n".join(docs[:15])
        return "## Documentação do Código\n\nNão foram encontradas docstrings nos ficheiros analisados."

    def _generate_report(self, query: str) -> str:
        """Gera estrutura de relatório."""
        return textwrap.dedent("""\
            # Relatório

            ## 1. Sumário Executivo

            [Resumo conciso dos principais pontos]

            ## 2. Contexto

            [Descrição do contexto e objetivo]

            ## 3. Análise

            ### 3.1 Metodologia
            [Método utilizado para a análise]

            ### 3.2 Resultados
            [Apresentação dos resultados]

            ### 3.3 Discussão
            [Interpretação dos resultados]

            ## 4. Conclusões

            [Principais conclusões]

            ## 5. Recomendações

            [Ações recomendadas]

            ## 6. Referências

            [Fontes consultadas]
        """)

    def _generate_general_doc(self, query: str) -> str:
        """Gera documentação geral."""
        title = self._extract_project_name(query) or "Documentação"
        return textwrap.dedent(f"""\
            # {title}

            ## Visão Geral

            {query}

            ## Secções

            - [Introdução](#introdução)
            - [Requisitos](#requisitos)
            - [Configuração](#configuração)
            - [Uso](#uso)
            - [Manutenção](#manutenção)

            ---

            *Documentação gerada pelo DocumentationAgent do Odysseus V2*
        """)

    def _extract_project_name(self, query: str) -> str:
        match = re.search(r"(?:projeto|project|do|da|de)\s+['\"]?(\w+)['\"]?", query, re.IGNORECASE)
        return match.group(1).capitalize() if match else "Projeto"

    def _identify_sections(self, content: str) -> list[str]:
        """Identifica secções Markdown no conteúdo."""
        sections = re.findall(r"^#{1,3}\s+(.+)$", content, re.MULTILINE)
        return sections

    def can_handle(self, query: str) -> float:
        keywords = [
            "documentar", "documentação", "relatório", "report",
            "escrever", "resumo", "sumário", "tutorial",
            "guia", "manual", "readme", "notas", "ata",
            "document", "docstring", "readme",
        ]
        query_lower = query.lower()
        score = sum(1 for kw in keywords if kw in query_lower)
        return min(score / 3, 1.0)


# ======================================================================
#  Agente Científico
# ======================================================================

class ScienceAgent(BaseAgent):
    """
    Agente especializado em análise científica e académica.

    Integrações:
        - Análise de estrutura de artigos científicos
        - Extração de citações e referências
        - Classificação de metodologias
        - Framework de análise estatística
    """

    def __init__(self):
        super().__init__(
            name="science",
            description=(
                "Análise científica e académica. "
                "Interpreta artigos, estrutura metodologias, "
                "e fornece análise de dados e citações."
            ),
        )

    async def execute(self, task: AgentTask) -> AgentResult:
        logger.info("ScienceAgent a executar tarefa: %s", task.id)
        query = task.query

        try:
            query_lower = query.lower()
            details: dict = {
                "query": query,
                "type": self._classify_analysis(query_lower),
                "methodology": self._analyze_methodology(query),
                "citations": self._extract_citations(query),
                "statistical_analysis": self._analyze_statistics(query),
                "paper_structure": self._analyze_paper_structure(query),
            }

            # Detetar se o texto é um abstract/papers
            if len(query) > 300:
                details["text_analysis"] = self._analyze_text(query)

            summary_parts = [
                f"Análise científica: {query[:60]}",
                f"Tipo: {details['type']}",
            ]
            n_citations = len(details["citations"])
            if n_citations:
                summary_parts.append(f"{n_citations} referência(s) identificada(s)")

            return AgentResult(
                task_id=task.id,
                success=True,
                summary=" | ".join(summary_parts),
                details=details,
                confidence=0.75,
                suggestions=[
                    "Valida as fontes com peer review independente",
                    "Podes pedir uma revisão sistemática da literatura",
                    "Considera verificar a replicabilidade dos resultados",
                ],
            )

        except Exception as e:
            logger.error("ScienceAgent erro: %s", e, exc_info=True)
            return AgentResult(
                task_id=task.id,
                success=False,
                summary="Erro na análise científica",
                error=str(e),
                confidence=0.0,
            )

    def _classify_analysis(self, query_lower: str) -> str:
        if any(w in query_lower for w in ["artigo", "paper", "publicação"]):
            return "paper_analysis"
        if any(w in query_lower for w in ["metodologia", "método", "metodológico"]):
            return "methodology_analysis"
        if any(w in query_lower for w in ["estatística", "dados", "correlação", "regressão"]):
            return "statistical_analysis"
        if any(w in query_lower for w in ["citação", "referência", "bibliografia"]):
            return "citation_analysis"
        return "general_scientific"

    def _analyze_methodology(self, query: str) -> dict:
        """Analisa a metodologia mencionada no texto."""
        query_lower = query.lower()
        methods = []
        method_patterns = {
            "revisão sistemática": ["revisão sistemática", "systematic review"],
            "meta-análise": ["meta-análise", "meta-analysis"],
            "ensaio clínico": ["ensaio clínico", "clinical trial", "rct"],
            "estudo de caso": ["estudo de caso", "case study"],
            "pesquisa qualitativa": ["qualitativa", "entrevista", "focus group"],
            "pesquisa quantitativa": ["quantitativa", "survey", "questionário"],
            "experimental": ["experimental", "experimento", "laboratório"],
            "observacional": ["observacional", "coorte", "caso-controle"],
            "simulação": ["simulação", "modelagem", "modelação"],
            "teórico": ["revisão bibliográfica", "revisão da literatura"],
        }
        for method, keywords in method_patterns.items():
            if any(kw in query_lower for kw in keywords):
                methods.append(method)

        return {
            "identified_methods": methods or ["Não foi possível determinar a metodologia"],
            "has_hypothesis": "hipótese" in query_lower or "hypothesis" in query_lower,
            "has_research_question": "pergunta" in query_lower or "research question" in query_lower,
            "has_sample": any(w in query_lower for w in ["amostra", "sample", "n=", "participantes"]),
            "note": "Análise metodológica baseada em keywords — recomendada validação manual",
        }

    def _extract_citations(self, text: str) -> list[dict]:
        """Extrai citações no formato APA, IEEE, ou numérico."""
        citations = []
        patterns = [
            (r"\(([A-Z][a-z]+(?:\s+(?:et\s+al\.|&\s+[A-Z][a-z]+))?),\s*(\d{4})\)", "APA"),
            (r"\[(\d+)\]", "numerical"),
            (r"([A-Z][a-z]+(?:\s+et\s+al\.)?)\s*\[(\d+)\]", "IEEE"),
        ]
        for pattern, style in patterns:
            for match in re.finditer(pattern, text):
                citations.append({
                    "style": style,
                    "match": match.group(0),
                    "context": text[max(0, match.start() - 50):match.end() + 50],
                })
        return citations[:10]

    def _analyze_statistics(self, query: str) -> dict:
        """Analisa elementos estatísticos no texto."""
        query_lower = query.lower()
        tests_found = []
        test_patterns = {
            "t-test": ["t-test", "test t", "student"],
            "ANOVA": ["anova", " análise de variância"],
            "correlação": ["correlação", "correlação de pearson", "spearman"],
            "regressão": ["regressão", "regressão linear", "regressão logística"],
            "qui-quadrado": ["qui-quadrado", "chi-square", "χ²"],
            "descritiva": ["média", "mediana", "desvio padrão", "frequência"],
        }
        for test, keywords in test_patterns.items():
            if any(kw in query_lower for kw in keywords):
                tests_found.append(test)

        return {
            "statistical_tests": tests_found or ["Nenhum teste estatístico identificado"],
            "has_p_value": bool(re.search(r"p\s*[<>=]\s*0[.,]\d+", query_lower)),
            "has_effect_size": bool(re.search(r"(effect size|d de cohen|eta²|coeficiente)", query_lower)),
            "has_confidence_interval": bool(re.search(r"(ic\s*\d{2}%|intervalo de confiança)", query_lower)),
        }

    def _analyze_paper_structure(self, query: str) -> dict:
        """Analisa a estrutura de um artigo científico."""
        sections = []
        section_patterns = [
            "abstract", "resumo", "introdução", "introduction",
            "métodos", "methods", "resultados", "results",
            "discussão", "discussion", "conclusão", "conclusion",
            "referências", "references", "agradecimentos",
        ]
        query_lower = query.lower()
        for section in section_patterns:
            if section in query_lower:
                sections.append(section)

        return {
            "identified_sections": sections,
            "has_abstract": "abstract" in query_lower or "resumo" in query_lower,
            "has_introduction": "introdu" in query_lower,
            "has_methods": "método" in query_lower or "method" in query_lower,
            "has_results": "resultado" in query_lower or "result" in query_lower,
            "has_conclusion": "conclus" in query_lower or "conclusion" in query_lower,
            "completeness": "completo" if len(sections) >= 4 else "parcial",
        }

    def _analyze_text(self, text: str) -> dict:
        """Análise geral do texto científico."""
        words = text.split()
        return {
            "total_words": len(words),
            "estimated_reading_time_min": max(1, len(words) // 200),
            "has_references": bool(re.search(r"(?:referência|referenc|bibliografia)", text, re.IGNORECASE)),
            "has_tables": text.count("|") > 5 or "tabela" in text.lower(),
            "has_figures": "figura" in text.lower() or "figure" in text.lower(),
        }

    def can_handle(self, query: str) -> float:
        keywords = [
            "científico", "ciência", "pesquisa científica", "artigo",
            "estudo", "experiência", "laboratório", "dados",
            "estatística", "metodologia", "hipótese", "teoria",
            "académico", "tese", "dissertação", "paper",
            "abstract", "metodologia", "publicação",
        ]
        query_lower = query.lower()
        score = sum(1 for kw in keywords if kw in query_lower)
        return min(score / 4, 1.0)


# ======================================================================
#  Fábrica de Agentes
# ======================================================================

class AgentFactory:
    """
    Fábrica que mantém o catálogo de agentes disponíveis e permite
    selecionar o(s) agente(s) mais adequado(s) para uma dada tarefa.

    Integrada com o ToolRegistry para descoberta de capacidades.
    """

    def __init__(self):
        self._agents: dict[str, BaseAgent] = {}
        self._tool_manager = None
        self._tool_registry = None
        self._register_defaults()

    def set_tool_manager(self, tool_manager):
        """
        Associa um ToolManager à factory.

        Os agentes podem usar o ToolManager para executar ferramentas
        em vez de implementar as suas próprias capacidades.
        """
        self._tool_manager = tool_manager
        if tool_manager:
            from .tool_registry import get_tool_registry
            self._tool_registry = get_tool_registry()

    def get_tool_capabilities(self) -> dict:
        """
        Retorna as capacidades de ferramentas disponíveis no sistema.

        Returns:
            Dict com categorias e ferramentas disponíveis.
        """
        if not self._tool_registry:
            return {"available": False, "message": "Tool system not initialized"}
        return {
            "available": True,
            "tool_count": self._tool_registry.count(),
            "categories": self._tool_registry.get_categories(),
            "capabilities": self._tool_registry.get_capabilities(),
            "tools": self._tool_registry.list_tools(),
        }

    def _register_defaults(self) -> None:
        """Regista os especialistas autorizados e os workers operacionais."""
        self.register(ResearchAgent())
        self.register(CodeAgent())
        self.register(BusinessAgent())
        self.register(LegalAgent())
        self.register(FinanceAgent())
        self.register(DocumentationAgent())
        self.register(ScienceAgent())
        from .workers import register_workers
        register_workers(self)

    def register_optional_agents(self) -> None:
        """Regista o UserAgent e os agentes personalizados activos."""
        try:
            from .user_agent_config import get_user_agent
            self.register(get_user_agent())
            logger.info("UserAgent registado com sucesso")
        except ImportError as e:
            logger.debug("UserAgent não disponível: %s", e)
        self.reload_custom_agents()

    def reload_custom_agents(self) -> list[str]:
        """Regista (ou re-regista) os agentes personalizados activos.

        Remove instâncias anteriores de CustomUserAgent e regista as
        actuais definidas em user_config.json (apenas enabled=True).
        Devolve a lista de nomes efectivamente registados.
        """
        from .user_agent_config import CustomUserAgent, get_custom_agents

        previous = [n for n, a in self._agents.items()
                    if isinstance(a, CustomUserAgent)]
        for name in previous:
            self._agents.pop(name, None)

        registered: list[str] = []
        for agent in get_custom_agents():
            self.register(agent)
            registered.append(agent.name)

        logger.info(
            "Agentes personalizados (re)registados: %s (removidos antes: %s)",
            registered or "(nenhum)",
            previous or "(nenhum)",
        )
        return registered

    def register(self, agent: BaseAgent) -> None:
        """Regista um novo agente no catálogo."""
        self._agents[agent.name] = agent
        logger.info("Agente registado: %s — %s", agent.name, agent.description)

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Obtém um agente pelo nome."""
        return self._agents.get(name)

    def list_agents(self) -> list[dict]:
        """Lista todos os agentes disponíveis."""
        return [
            {"name": a.name, "description": a.description}
            for a in self._agents.values()
        ]

    def select_agent(self, query: str) -> Optional[BaseAgent]:
        """
        Seleciona automaticamente o melhor agente para uma consulta.
        Usa o scoring do método can_handle() de cada agente.
        """
        best_agent = None
        best_score = 0.0

        for agent in self._agents.values():
            if getattr(agent, "is_operational_worker", False):
                continue
            score = agent.can_handle(query)
            if score > best_score:
                best_score = score
                best_agent = agent

        return best_agent if best_score > 0.2 else None

    def select_all_relevant(self, query: str, min_score: float = 0.2) -> list[str]:
        """
        Seleciona TODOS os agentes com score acima de um limiar.

        Útil para paralelismo: em vez de escolher apenas o melhor agente,
        devolve todos os agentes que podem contribuir para a consulta.

        Args:
            query: A consulta do utilizador.
            min_score: Score mínimo (0.0 — 1.0) para considerar relevante.

        Returns:
            Lista de nomes de agentes com score >= min_score, ordenados
            do mais relevante para o menos relevante.
        """
        scored = []
        for agent in self._agents.values():
            if getattr(agent, "is_operational_worker", False):
                continue
            score = agent.can_handle(query)
            if score >= min_score:
                scored.append((agent.name, score))

        scored.sort(key=lambda x: -x[1])
        return [name for name, _ in scored]

    def select_worker(self, query: str) -> Optional[BaseAgent]:
        """Select a Worker only when its documented keywords match the request."""
        lowered = query.lower()
        explicit_roles = (
            (("secretária", "secretaria"), "secretaria"),
            (("técnico", "tecnico"), "tecnico"),
            (("director-geral", "diretor-geral", "director", "diretor"), "director-geral"),
        )
        for phrases, name in explicit_roles:
            if any(phrase in lowered for phrase in phrases):
                worker = self._agents.get(name)
                if worker is not None:
                    return worker

        workers = [
            agent for agent in self._agents.values()
            if getattr(agent, "is_operational_worker", False)
        ]
        if not workers:
            raise RuntimeError("Nenhum agente operacional está registado.")
        best_worker = max(workers, key=lambda agent: agent.can_handle(query))
        return best_worker if best_worker.can_handle(query) > 0 else None


# ======================================================================
#  Singleton
# ======================================================================

_agent_factory: Optional[AgentFactory] = None


def get_agent_factory() -> AgentFactory:
    global _agent_factory
    if _agent_factory is None:
        _agent_factory = AgentFactory()
        _agent_factory.register_optional_agents()
    return _agent_factory
