"""
OpenCode Agent — Odysseus V2

Agente especializado que integra o OpenCode (ferramenta de coding assistida
por IA, originalmente em https://github.com/opencode-ai/opencode) como um
agente de programação dentro do ecossistema Odysseus V2.

O OpenCode funciona como um terminal-based coding assistant que pode:
    - Gerar e modificar código automaticamente
    - Executar comandos de shell para build/test
    - Analisar projetos e sugerir refactorings
    - Operar em loop agente-ferramenta para tarefas complexas

Este agente adapta o padrão agent-loop / tool-execution do OpenCode
para funcionar como um agente especializado dentro do Odysseus V2,
podendo ser orquestrado pelo OdysseusOrchestrator.

Uso:
    from core.v2.opencode_agent import OpenCodeAgent
    agent = OpenCodeAgent()
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .agents import BaseAgent, AgentTask, AgentResult, _get_llm_client

logger = logging.getLogger(__name__)


# ======================================================================
#  Constantes e Configuração
# ======================================================================

# Caminho padrão para o binário OpenCode (pode ser alterado via env var)
OPENCODE_BIN = os.getenv("OPENCODE_BIN", "opencode")

# Project root (diretório do projeto Odysseus)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class OpenCodeConfig:
    """Configuração para execução do OpenCode.

    Pode ser personalizada por tarefa ou globalmente.
    """
    bin_path: str = OPENCODE_BIN
    working_dir: str = str(PROJECT_ROOT)
    timeout_seconds: float = 120.0
    max_iterations: int = 5
    model: str = ""  # modelo a usar (ex: "gpt-4", "claude-3", "llama3")
    temperature: float = 0.7
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "OpenCodeConfig":
        """Carrega configuração de variáveis de ambiente."""
        return cls(
            bin_path=os.getenv("OPENCODE_BIN", OPENCODE_BIN),
            working_dir=os.getenv("OPENCODE_WORKING_DIR", str(PROJECT_ROOT)),
            timeout_seconds=float(os.getenv("OPENCODE_TIMEOUT", "120")),
            max_iterations=int(os.getenv("OPENCODE_MAX_ITERATIONS", "5")),
            model=os.getenv("OPENCODE_MODEL", ""),
            temperature=float(os.getenv("OPENCODE_TEMPERATURE", "0.7")),
        )


# ======================================================================
#  OpenCode Agent
# ======================================================================

class OpenCodeAgent(BaseAgent):
    """
    Agente especializado em coding assistido via OpenCode.

    Adapta o loop agente-ferramenta do OpenCode como um agente
    do Odysseus V2. Permite:

        - Gerar/modificar código automaticamente
        - Executar análises de projeto
        - Sugerir refactorings e correções
        - Executar testes e builds

    Integrações:
        - OpenCode CLI (terminal-based coding assistant)
        - LLM via _get_llm_client() para fallback e análise
        - Sistema de ficheiros do projeto Odysseus
        - Execução segura de comandos em subprocesso
    """

    def __init__(self):
        super().__init__(
            name="opencode",
            description=(
                "Coding assistido via OpenCode. Gera e modifica código, "
                "executa análises de projeto, sugere refactorings, "
                "e automatiza tarefas de desenvolvimento."
            ),
        )
        self.config = OpenCodeConfig.from_env()

    # ------------------------------------------------------------------
    #  Execução principal
    # ------------------------------------------------------------------

    async def execute(self, task: AgentTask) -> AgentResult:
        logger.info("OpenCodeAgent a executar tarefa: %s", task.id)
        query = task.query
        context = task.context or {}

        # Permitir override de configuração via context
        config = OpenCodeConfig(
            bin_path=context.get("bin_path", self.config.bin_path),
            working_dir=context.get("working_dir", self.config.working_dir),
            timeout_seconds=context.get("timeout", self.config.timeout_seconds),
            max_iterations=context.get("max_iterations", self.config.max_iterations),
            model=context.get("model", self.config.model),
            temperature=context.get("temperature", self.config.temperature),
        )

        try:
            details: dict = {
                "query": query,
                "config": {
                    "bin_path": config.bin_path,
                    "working_dir": config.working_dir,
                    "timeout": config.timeout_seconds,
                    "max_iterations": config.max_iterations,
                    "model": config.model or "default",
                },
                "executions": [],
                "files_created": [],
                "files_modified": [],
                "commands_run": [],
                "issues": [],
                "output": "",
            }
            suggestions = []

            # ── Estratégia: múltiplas abordagens ──
            # 1. Tentar usar OpenCode CLI diretamente
            opencode_output = await self._run_opencode(query, config)
            if opencode_output is not None:
                details["executions"].append({
                    "method": "opencode_cli",
                    "success": True,
                    "output_preview": opencode_output[:500],
                })
                details["output"] = opencode_output
            else:
                # 2. Fallback: usar LLM integrado para resposta
                logger.info("OpenCode CLI não disponível — a usar LLM fallback")
                llm_output = await self._llm_coding_assist(query)
                if llm_output:
                    details["executions"].append({
                        "method": "llm_fallback",
                        "success": True,
                        "output_preview": llm_output[:500],
                    })
                    details["output"] = llm_output
                else:
                    details["executions"].append({
                        "method": "llm_fallback",
                        "success": False,
                        "error": "LLM não disponível",
                    })

            # ── Extrair informações do output ──
            if details["output"]:
                # Detetar ficheiros criados/modificados
                created = self._detect_created_files(details["output"])
                details["files_created"] = created

                modified = self._detect_modified_files(details["output"])
                details["files_modified"] = modified

                # Detetar comandos executados
                commands = self._detect_commands(details["output"])
                details["commands_run"] = commands

                # Detetar issues/erros
                issues = self._detect_issues(details["output"])
                details["issues"] = issues

            # ── Construir sumário ──
            summary_parts = [
                f"OpenCode: {query[:80]}",
            ]
            if details["files_created"]:
                summary_parts.append(f"{len(details['files_created'])} ficheiro(s) criado(s)")
            if details["files_modified"]:
                summary_parts.append(f"{len(details['files_modified'])} ficheiro(s) modificado(s)")
            if details["issues"]:
                summary_parts.append(f"{len(details['issues'])} issue(s) detetada(s)")
            else:
                summary_parts.append("Sem issues detetadas")

            # Sugestões inteligentes
            if details["issues"]:
                suggestions.append("Rever issues detetadas antes de continuar")
            if details["files_created"] or details["files_modified"]:
                suggestions.append("Executar testes para validar alterações")
            suggestions.append("Usar `opencode` diretamente para interação mais rica")

            n_executions = len([e for e in details["executions"] if e.get("success")])
            confidence = 0.5 if n_executions == 0 else (0.7 if "llm_fallback" in str(details["executions"]) else 0.9)

            return AgentResult(
                task_id=task.id,
                success=n_executions > 0,
                summary=" | ".join(summary_parts),
                details=details,
                confidence=confidence,
                suggestions=suggestions,
                follow_up=[
                    "Podes pedir para executar o código gerado",
                    "Queres que analise o projeto com mais profundidade?",
                    "Posso refatorar ficheiros específicos se indicares quais",
                ],
            )

        except Exception as e:
            logger.error("OpenCodeAgent erro: %s", e, exc_info=True)
            return AgentResult(
                task_id=task.id,
                success=False,
                summary="Erro na execução do OpenCode",
                error=str(e),
                confidence=0.0,
            )

    # ------------------------------------------------------------------
    #  Execução OpenCode CLI
    # ------------------------------------------------------------------

    async def _run_opencode(self, query: str, config: OpenCodeConfig) -> Optional[str]:
        """
        Tenta executar o OpenCode CLI com a query fornecida.

        O OpenCode funciona como um assistente de terminal que aceita
        prompts diretamente. Tentamos várias formas de invocação:
        1. opencode --prompt "<query>"
        2. opencode -m "<query>"
        3. echo "<query>" | opencode

        Returns:
            str com o output, ou None se o OpenCode não estiver disponível.
        """
        # Verificar se o binário existe
        if not self._check_opencode_available(config.bin_path):
            logger.info("OpenCode CLI não encontrado em: %s", config.bin_path)
            return None

        # Estratégias de invocação
        strategies = [
            # Estratégia 1: --prompt (modo mais comum)
            [config.bin_path, "--prompt", query],
            # Estratégia 2: -m (short flag para mensagem)
            [config.bin_path, "-m", query],
            # Estratégia 3: stdin pipe
            [config.bin_path],
        ]

        for cmd in strategies:
            try:
                logger.debug("OpenCodeAgent: a tentar execução: %s", cmd)

                if cmd == strategies[-1]:
                    # stdin: escrever query no stdin do processo
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=config.working_dir,
                    )
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(input=query.encode("utf-8")),
                        timeout=config.timeout_seconds,
                    )
                else:
                    # Argumentos: passar query como argumento
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=config.working_dir,
                    )
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=config.timeout_seconds,
                    )

                if proc.returncode == 0:
                    output = stdout.decode("utf-8", errors="replace").strip()
                    if output:
                        return output
                else:
                    stderr_text = stderr.decode("utf-8", errors="replace").strip()
                    logger.debug(
                        "OpenCode CLI (estratégia %d) falhou: %s",
                        strategies.index(cmd) + 1,
                        stderr_text[:200],
                    )

            except asyncio.TimeoutError:
                logger.debug(
                    "OpenCode CLI (estratégia %d) timeout após %ds",
                    strategies.index(cmd) + 1,
                    config.timeout_seconds,
                )
            except FileNotFoundError:
                logger.debug("OpenCode CLI não encontrado (FileNotFoundError)")
                return None
            except Exception as e:
                logger.debug(
                    "OpenCode CLI (estratégia %d) erro: %s",
                    strategies.index(cmd) + 1,
                    e,
                )

        return None

    # ------------------------------------------------------------------
    #  Fallback: LLM Coding Assist
    # ------------------------------------------------------------------

    async def _llm_coding_assist(self, query: str) -> Optional[str]:
        """
        Fallback para quando o OpenCode CLI não está disponível.

        Usa o LLM configurado (via _get_llm_client) para responder
        a perguntas de programação com contexto de código.
        """
        try:
            llm = _get_llm_client()
            if not await llm.check_availability():
                return None

            # Construir prompt de sistema para coding assist
            system_content = (
                "Tu és um assistente de programação especializado (modo OpenCode). "
                "O teu objetivo é ajudar o utilizador com tarefas de desenvolvimento:\n\n"
                "- Gerar código limpo e bem documentado\n"
                "- Analisar e sugerir melhorias em código existente\n"
                "- Explicar conceitos de programação\n"
                "- Debuggar problemas\n"
                "- Sugerir arquiteturas e patterns\n\n"
                "Quando gerares código, inclui o nome do ficheiro sugerido "
                "como comentário no início (ex: `# file: exemplo.py`).\n"
                "Responde em português de Portugal.\n"
                "Sê prático e focado em soluções executáveis."
            )

            resp = await llm.chat([
                {"role": "system", "content": system_content},
                {"role": "user", "content": query},
            ], max_tokens=2048)

            if resp.success and resp.content.strip():
                return resp.content.strip()
        except Exception as e:
            logger.debug("OpenCodeAgent LLM fallback erro: %s", e)

        return None

    # ------------------------------------------------------------------
    #  Utilitários
    # ------------------------------------------------------------------

    def _check_opencode_available(self, bin_path: str) -> bool:
        """Verifica se o binário OpenCode está disponível no sistema."""
        try:
            if os.path.isfile(bin_path) and os.access(bin_path, os.X_OK):
                return True
            # Verificar no PATH
            import shutil
            return shutil.which(bin_path) is not None
        except Exception:
            return False

    def _detect_created_files(self, output: str) -> list[str]:
        """Deteta menções de ficheiros criados no output."""
        files = []
        # Padrão: "Created file: ..." ou "# file: ..." ou "created ..."
        patterns = [
            r"(?:created|create|criado?)\s+(?:file|ficheiro)?\s*:?\s*[`'\"]?([\w./\\-]+\.[a-z]+)[`'\"]?",
            r"#\s*file:\s*([\w./\\-]+\.[a-z]+)",
            r"//\s*file:\s*([\w./\\-]+\.[a-z]+)",
            r"/\*\*\s*file:\s*([\w./\\-]+\.[a-z]+)\s*\*\*/",
            r"<!--\s*file:\s*([\w./\\-]+\.[a-z]+)\s*-->",
        ]
        for pattern in patterns:
            files.extend(m.group(1) for m in re.finditer(pattern, output, re.IGNORECASE))
        return list(set(files))

    def _detect_modified_files(self, output: str) -> list[str]:
        """Deteta menções de ficheiros modificados no output."""
        files = []
        pattern = r"(?:modified|modifi(?:cado|ed)|updated?|atualizado?)\s+(?:file|ficheiro)?\s*:?\s*[`'\"]?([\w./\\-]+\.[a-z]+)[`'\"]?"
        files.extend(m.group(1) for m in re.finditer(pattern, output, re.IGNORECASE))
        return list(set(files))

    def _detect_commands(self, output: str) -> list[str]:
        """Deteta comandos executados (blocos de código bash/cmd)."""
        commands = []
        pattern = r"```(?:bash|sh|shell|cmd|powershell)\s*\n(.+?)\n```"
        for m in re.finditer(pattern, output, re.DOTALL):
            cmd = m.group(1).strip()
            if cmd and len(cmd) < 500:
                commands.append(cmd)
        return commands

    def _detect_issues(self, output: str) -> list[dict]:
        """Deteta issues/erros/advertências no output."""
        issues = []
        patterns = {
            "error": re.compile(r"(?:erro|error|fail(?:ed|ure)?|exception)\s*:?\s*(.+)$", re.IGNORECASE | re.MULTILINE),
            "warning": re.compile(r"(?:warning|aviso|atenção|atenção)\s*:?\s*(.+)$", re.IGNORECASE | re.MULTILINE),
            "suggestion": re.compile(r"(?:sugest[ãa]o|suggestion|tip|dica)\s*:?\s*(.+)$", re.IGNORECASE | re.MULTILINE),
        }
        for severity, pattern in patterns.items():
            for m in pattern.finditer(output):
                msg = m.group(1).strip()
                if msg and len(msg) > 5:
                    issues.append({
                        "severity": severity,
                        "message": msg[:200],
                    })
        return issues[:10]

    # ------------------------------------------------------------------
    #  Deteção de relevância
    # ------------------------------------------------------------------

    def can_handle(self, query: str) -> float:
        """Score de aptidão para queries de programação/OpenCode."""
        keywords = [
            # Programação geral
            "código", "programar", "desenvolver", "implementar",
            "refatorar", "debug", "bug", "teste", "build",
            "compilar", "deploy", "commit", "pull request",
            # OpenCode específico
            "opencode", "coding assistant", "agent loop",
            "tool execution", "terminal coding",
            # Ações OpenCode
            "gera código", "modifica", "cria ficheiro",
            "executa comando", "shell",
            # Linguagens
            "python", "javascript", "typescript", "go", "rust",
            "java", "c++", "html", "css",
        ]
        query_lower = query.lower()
        score = sum(1 for kw in keywords if kw in query_lower)
        # Se mencionar "opencode" diretamente, score alto
        if "opencode" in query_lower:
            score += 3
        return min(score / 4, 1.0)


# ======================================================================
#  Singleton
# ======================================================================

_opencode_agent_instance: Optional[OpenCodeAgent] = None


def get_opencode_agent(config: Optional[OpenCodeConfig] = None) -> OpenCodeAgent:
    """Retorna a instância global do OpenCodeAgent."""
    global _opencode_agent_instance
    if _opencode_agent_instance is None:
        _opencode_agent_instance = OpenCodeAgent()
        if config is not None:
            _opencode_agent_instance.config = config
    elif config is not None:
        _opencode_agent_instance = OpenCodeAgent()
        _opencode_agent_instance.config = config
    return _opencode_agent_instance


def reset_opencode_agent() -> None:
    """Reinicia o OpenCodeAgent (útil para testes)."""
    global _opencode_agent_instance
    _opencode_agent_instance = None
