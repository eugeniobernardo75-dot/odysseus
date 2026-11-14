"""
Tool Chaining — Odysseus V2

Sistema para encadear execuções de ferramentas automaticamente,
onde o output de cada passo alimenta o input do passo seguinte.

Fluxo típico::

    web.fetch → code.analyze → memory.store

Cada passo pode referenciar resultados de passos anteriores
através de variáveis no formato ``{prev.path}`` ou ``{steps.N.path}``.

Exemplo::

    chain = ToolChain(
        name="web_fetch_analyze_store",
        description="Fetch de URL → análise → guardar na memória",
        steps=[
            ChainStep(
                tool="web",
                params={"action": "fetch", "url": "{input.url}"},
                output_mapping={"extracted_text": "data.extracted_text"},
            ),
            ChainStep(
                tool="code",
                params={
                    "action": "analyze",
                    "code": "{prev.data.extracted_text}",
                },
                output_mapping={"analysis": "data"},
            ),
            ChainStep(
                tool="memory",
                params={
                    "action": "store",
                    "key": "fetched:{input.url}",
                    "value": "Análise: {steps.1.data.metrics}",
                },
            ),
        ],
    )
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .tool_registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


# ======================================================================
#  Constantes
# ======================================================================

# Padrão para detetar variáveis: {alguma.coisa.aqui}
_VARIABLE_PATTERN = re.compile(r"\{([^}]+)\}")


# ======================================================================
#  Estruturas
# ======================================================================

@dataclass
class ChainStep:
    """
    Um passo dentro de uma ToolChain.

    Attributes:
        tool: Nome da ferramenta a executar (ex: "web", "code").
        params: Dicionário de parâmetros. Valores podem conter
            variáveis ``{input.xxx}``, ``{prev.xxx}`` ou ``{steps.N.xxx}``.
        output_mapping: Mapa de ``{nome}: {caminho.no.resultado}``
            para extrair campos específicos do resultado deste passo.
            Opcional — se vazio, o ``data`` completo fica disponível
            como ``steps.N.data``.
        description: Descrição legível do passo (opcional).
        skip_on_failure: Se True, continua a chain mesmo que este passo falhe.
        timeout: Timeout específico para este passo (sobrepõe o da ferramenta).
    """
    tool: str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    skip_on_failure: bool = False
    timeout: Optional[float] = None


@dataclass
class ChainDefinition:
    """
    Definição de uma ToolChain (versão registável, sem estado).

    Attributes:
        name: Nome único da chain.
        description: Descrição legível.
        steps: Sequência de ChainStep a executar.
        tags: Tags para categorização.
        category: Categoria funcional.
        version: Versão semântica.
    """
    name: str
    description: str
    steps: list[ChainStep]
    tags: list[str] = field(default_factory=list)
    category: str = "chain"
    version: str = "1.0.0"


@dataclass
class ChainResult:
    """
    Resultado completo da execução de uma ToolChain.

    Attributes:
        success: True se TODOS os passos成功ram (ou skip_on_failure=True).
        chain_name: Nome da chain executada.
        steps: Resultados individuais de cada passo (na ordem).
        total_time_ms: Tempo total da chain.
        error: Mensagem de erro do primeiro passo que falhou (se aplicável).
        failed_step: Índice do passo que falhou (-1 se nenhum).
        execution_id: ID único desta execução.
    """
    success: bool
    chain_name: str
    steps: list[dict]
    total_time_ms: float
    error: Optional[str] = None
    failed_step: int = -1
    execution_id: str = ""


@dataclass
class StepExecution:
    """Registo interno da execução de um passo."""
    index: int
    tool: str
    params: dict
    resolved_params: dict
    result: Optional[ToolResult] = None
    success: bool = False
    error: Optional[str] = None
    duration_ms: float = 0.0
    skipped: bool = False


# ======================================================================
#  Chain Executor
# ======================================================================

class ChainExecutor:
    """
    Executa ToolChains, resolvendo variáveis entre passos.

    O executor:
        1. Recebe uma ChainDefinition + input inicial
        2. Para cada passo, resolve variáveis nos parâmetros
        3. Executa a ferramenta via ToolManager
        4. Recolhe métricas de cada passo (duração, estado)
        5. Retorna ChainResult completo com resultados por passo

    Variáveis suportadas:
        - ``{input.xxx}``:  valor do input inicial fornecido pelo utilizador
        - ``{prev.data.xxx}``:  campo ``xxx`` do ``data`` do passo anterior
        - ``{prev.xxx}``:  campo ``xxx`` do resultado do passo anterior
        - ``{steps.N.data.xxx}``:  campo ``xxx`` do ``data`` do passo N
        - ``{steps.N.xxx}``:  campo ``xxx`` do resultado do passo N
        - ``{prev.metadata.xxx}``:  campo ``xxx`` do metadata do passo anterior
    """

    def __init__(self, tool_manager: Any, registry: Optional[ToolRegistry] = None):
        """
        Args:
            tool_manager: Instância do ToolManager para executar ferramentas.
            registry: ToolRegistry opcional (para validação extra).
        """
        self._tool_manager = tool_manager
        self._registry = registry

    # ------------------------------------------------------------------
    #  Execução
    # ------------------------------------------------------------------

    async def execute(
        self,
        chain: ChainDefinition,
        input_data: dict[str, Any],
        context: Optional[dict] = None,
    ) -> ChainResult:
        """
        Executa uma chain completa.

        Args:
            chain: Definição da chain a executar.
            input_data: Input inicial do utilizador.
            context: Contexto de execução (passado ao ToolManager).

        Returns:
            ChainResult com resultados de todos os passos.
        """
        execution_id = f"CHAIN-{uuid.uuid4().hex[:8].upper()}"
        context = context or {}
        steps_output: list[StepExecution] = []
        start = datetime.now(timezone.utc)

        logger.info(
            "Chain '%s' iniciada [%s] com %d passos",
            chain.name, execution_id, len(chain.steps),
        )

        for index, step in enumerate(chain.steps):
            step_start = datetime.now(timezone.utc)
            step_exec = StepExecution(
                index=index,
                tool=step.tool,
                params=step.params,
                resolved_params={},
            )

            try:
                # ── Resolver variáveis nos parâmetros ──
                resolved = self._resolve_params(
                    step.params,
                    input_data=input_data,
                    steps=steps_output,
                )
                step_exec.resolved_params = resolved

                logger.debug(
                    "Chain '%s' passo %d: %s → %s",
                    chain.name, index, step.tool,
                    json.dumps(resolved, default=str)[:200],
                )

                # ── Executar ferramenta ──
                timeout_context = {}
                if step.timeout:
                    timeout_context["_timeout"] = step.timeout

                result = await self._tool_manager.execute(
                    step.tool,
                    resolved,
                    context={**context, **timeout_context},
                )

                step_exec.result = result
                step_exec.success = result.success

                if not result.success and not step.skip_on_failure:
                    step_exec.error = result.error
                    duration = (datetime.now(timezone.utc) - step_start).total_seconds() * 1000
                    step_exec.duration_ms = round(duration, 1)
                    steps_output.append(step_exec)

                    logger.error(
                        "Chain '%s' falhou no passo %d (%s): %s",
                        chain.name, index, step.tool, result.error,
                    )

                    return self._build_result(
                        chain=chain,
                        steps=steps_output + [self._build_missing_step(i, chain.steps) for i in range(index + 1, len(chain.steps))],
                        input_data=input_data,
                        start=start,
                        success=False,
                        error=f"Passo {index} ({step.tool}): {result.error}",
                        failed_step=index,
                        execution_id=execution_id,
                    )

            except Exception as e:
                step_exec.error = str(e)
                step_exec.success = False
                duration = (datetime.now(timezone.utc) - step_start).total_seconds() * 1000
                step_exec.duration_ms = round(duration, 1)
                steps_output.append(step_exec)

                if not step.skip_on_failure:
                    logger.exception("Chain '%s' exceção no passo %d", chain.name, index)
                    return self._build_result(
                        chain=chain,
                        steps=steps_output,
                        input_data=input_data,
                        start=start,
                        success=False,
                        error=f"Exceção no passo {index} ({step.tool}): {e}",
                        failed_step=index,
                        execution_id=execution_id,
                    )

            # Duração do passo
            duration = (datetime.now(timezone.utc) - step_start).total_seconds() * 1000
            step_exec.duration_ms = round(duration, 1)
            steps_output.append(step_exec)

            logger.info(
                "Chain '%s' passo %d OK (%s, %.0fms)",
                chain.name, index, step.tool, duration,
            )

        # ── Chain completa ──
        logger.info("Chain '%s' concluída com sucesso [%s]", chain.name, execution_id)

        return self._build_result(
            chain=chain,
            steps=steps_output,
            input_data=input_data,
            start=start,
            success=True,
            execution_id=execution_id,
        )

    # ------------------------------------------------------------------
    #  Resolução de Variáveis
    # ------------------------------------------------------------------

    def _resolve_params(
        self,
        params: dict,
        input_data: dict,
        steps: list[StepExecution],
    ) -> dict:
        """
        Resolve todas as variáveis nos parâmetros.

        Procura padrões ``{caminho}`` em todos os valores string
        e substitui pelo valor correspondente.
        """
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str):
                resolved[key] = self._resolve_string(value, input_data, steps)
            elif isinstance(value, dict):
                resolved[key] = {
                    k: self._resolve_string(v, input_data, steps) if isinstance(v, str) else v
                    for k, v in value.items()
                }
            elif isinstance(value, list):
                resolved[key] = [
                    self._resolve_string(v, input_data, steps) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                resolved[key] = value
        return resolved

    def _resolve_string(self, text: str, input_data: dict, steps: list[StepExecution]) -> str:
        """Resolve variáveis numa string."""
        if "{" not in text:
            return text

        def replacer(match):
            path = match.group(1)
            resolved = self._resolve_variable(path, input_data, steps)
            if resolved is None:
                logger.warning("Chain: variável '%s' não resolvida em '%s'", path, text[:100])
                return f"<{path}:nao_resolvido>"
            if not isinstance(resolved, str):
                return json.dumps(resolved, ensure_ascii=False, default=str)
            return str(resolved)

        return _VARIABLE_PATTERN.sub(replacer, text)

    def _resolve_variable(self, path: str, input_data: dict, steps: list[StepExecution]) -> Any:
        """
        Resolve uma única variável pelo caminho.

        Formatos:
            input.xxx        → input_data["xxx"]
            prev.data.xxx    → steps[-1].result.data["xxx"]
            prev.xxx         → steps[-1].result.xxx (se não for data)
            steps.N.data.xxx → steps[N].result.data["xxx"]
            steps.N.xxx      → steps[N].result.xxx
        """
        parts = path.strip().split(".")
        if not parts:
            return None

        source = parts[0]

        if source == "input":
            # input.xxx.yyy
            return self._deep_get(input_data, parts[1:])

        if source == "prev":
            if not steps:
                return None
            last = steps[-1]
            if not last.result:
                return None
            # prev.data.xxx → last.result.data["xxx"]
            # prev.metadata.xxx → last.result.metadata["xxx"]
            # prev.xxx → atributo do resultado
            return self._deep_get(last.result, parts[1:]) if len(parts) > 1 else last.result

        if source == "steps":
            if len(parts) < 2:
                return None
            try:
                idx = int(parts[1])
            except ValueError:
                return None
            if idx >= len(steps):
                return None
            step = steps[idx]
            if not step.result:
                return None
            # steps.N.data.xxx ou steps.N.metadata.xxx ou steps.N.xxx
            return self._deep_get(step.result, parts[2:]) if len(parts) > 2 else step.result

        return None

    def _deep_get(self, obj: Any, keys: list[str]) -> Any:
        """Navega num objeto/dict com uma lista de chaves."""
        current = obj
        for key in keys:
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(key)
            elif hasattr(current, key):
                current = getattr(current, key)
            elif hasattr(current, "__dict__"):
                current = getattr(current, key, None)
            else:
                try:
                    current = current[key]
                except (TypeError, KeyError, IndexError):
                    return None
        return current

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _build_result(
        self,
        chain: ChainDefinition,
        steps: list[StepExecution],
        input_data: dict,
        start: datetime,
        success: bool,
        error: Optional[str] = None,
        failed_step: int = -1,
        execution_id: str = "",
    ) -> ChainResult:
        """Constrói ChainResult a partir da execução."""
        total_ms = round(
            (datetime.now(timezone.utc) - start).total_seconds() * 1000, 1
        )

        steps_data = []
        for s in steps:
            entry: dict[str, Any] = {
                "index": s.index,
                "tool": s.tool,
                "success": s.success,
                "duration_ms": s.duration_ms,
                "skipped": s.skipped,
            }
            if s.result:
                entry["result"] = {
                    "success": s.result.success,
                    "data": s.result.data,
                    "execution_time": s.result.execution_time,
                }
                if s.result.error:
                    entry["result"]["error"] = s.result.error
            if s.error:
                entry["error"] = s.error
            if s.resolved_params:
                entry["resolved_params"] = s.resolved_params
            steps_data.append(entry)

        return ChainResult(
            success=success,
            chain_name=chain.name,
            steps=steps_data,
            total_time_ms=total_ms,
            error=error,
            failed_step=failed_step,
            execution_id=execution_id,
        )

    def _build_missing_step(
        self, start_from: int, all_steps: list[ChainStep],
    ) -> StepExecution:
        """Cria StepExecution para passos que não foram executados."""
        return StepExecution(
            index=start_from,
            tool=all_steps[start_from].tool if start_from < len(all_steps) else "?",
            params={},
            resolved_params={},
            skipped=True,
            success=False,
            error="Chain abortada em passo anterior",
        )


# ======================================================================
#  Chain Registry
# ======================================================================

class ChainRegistry:
    """
    Registo de chains predefinidas.

    Permite:
        - register / unregister chains
        - list chains disponíveis
        - obter chain por nome
    """

    def __init__(self):
        self._chains: dict[str, ChainDefinition] = {}

    def register(self, chain: ChainDefinition) -> bool:
        """Regista uma chain. Substitui se já existir."""
        exists = chain.name in self._chains
        self._chains[chain.name] = chain
        logger.info(
            "ChainRegistry: %s '%s' (%d passos)",
            "substituiu" if exists else "registou",
            chain.name,
            len(chain.steps),
        )
        return not exists

    def unregister(self, name: str) -> bool:
        """Remove uma chain."""
        if name not in self._chains:
            return False
        del self._chains[name]
        logger.info("ChainRegistry: removida '%s'", name)
        return True

    def get(self, name: str) -> Optional[ChainDefinition]:
        """Obtém chain pelo nome."""
        return self._chains.get(name)

    def list(self, tag: Optional[str] = None) -> list[dict]:
        """
        Lista chains registadas.

        Args:
            tag: Filtrar por tag (opcional).

        Returns:
            Lista de metadados das chains.
        """
        chains = []
        for name, chain in self._chains.items():
            if tag and tag not in chain.tags:
                continue
            chains.append({
                "name": name,
                "description": chain.description,
                "n_steps": len(chain.steps),
                "tags": chain.tags,
                "category": chain.category,
                "version": chain.version,
            })
        return sorted(chains, key=lambda c: c["name"])

    def count(self) -> int:
        return len(self._chains)


# ======================================================================
#  Singleton
# ======================================================================

_chain_registry: Optional[ChainRegistry] = None


def get_chain_registry() -> ChainRegistry:
    """Retorna a instância global do ChainRegistry."""
    global _chain_registry
    if _chain_registry is None:
        _chain_registry = ChainRegistry()
    return _chain_registry


def reset_chain_registry() -> None:
    """Reinicia o registry (útil para testes)."""
    global _chain_registry
    _chain_registry = None


# ======================================================================
#  Chains Predefinidas
# ======================================================================

def register_default_chains(registry: Optional[ChainRegistry] = None) -> list[str]:
    """
    Regista as chains padrão do sistema.

    Chains predefinidas:
        1. ``web_fetch_analyze_store`` — fetch → code.analyze → memory.store
        2. ``web_fetch_store`` — fetch → memory.store (análise simples)
        3. ``file_analyze_report`` — file.read → code.analyze → memory.store
        4. ``web_search_analyze`` — web.fetch → search.content (extrair URLs → pesquisar)

    Args:
        registry: ChainRegistry (usa singleton se omitido).

    Returns:
        Lista de nomes das chains registadas.
    """
    reg = registry or get_chain_registry()
    chains: list[ChainDefinition] = [
        ChainDefinition(
            name="web_fetch_analyze_store",
            description=(
                "Fetch de uma página web, analisa o texto com AST "
                "e guarda a análise na memória permanente"
            ),
            category="web",
            tags=["web", "analysis", "memory", "automation"],
            steps=[
                ChainStep(
                    tool="web",
                    params={
                        "action": "fetch",
                        "url": "{input.url}",
                        "max_chars": "{input.max_chars}",
                    },
                    description="Fazer fetch da página web",
                ),
                ChainStep(
                    tool="code",
                    params={
                        "action": "analyze",
                        "code": "{prev.data.extracted_text}",
                    },
                    description="Analisar o texto extraído com AST",
                    skip_on_failure=True,
                ),
                ChainStep(
                    tool="memory",
                    params={
                        "action": "store",
                        "key": "web:{input.url}",
                        "value": (
                            "Fetch de {input.url} | "
                            "Análise: {steps.1.data.metrics}"
                        ),
                    },
                    description="Guardar resultado na memória",
                ),
            ],
        ),
        ChainDefinition(
            name="web_fetch_store",
            description=(
                "Fetch de página web e armazenamento direto na memória"
            ),
            category="web",
            tags=["web", "memory", "automation"],
            steps=[
                ChainStep(
                    tool="web",
                    params={
                        "action": "fetch_and_extract",
                        "url": "{input.url}",
                        "max_chars": 10000,
                    },
                    description="Fazer fetch da página web",
                ),
                ChainStep(
                    tool="memory",
                    params={
                        "action": "store",
                        "key": "web:{input.url}",
                        "value": "{prev.data.content_preview}",
                    },
                    description="Guardar conteúdo na memória",
                ),
            ],
        ),
        ChainDefinition(
            name="file_analyze_report",
            description=(
                "Lê um ficheiro, analisa o código com AST "
                "e guarda as métricas na memória"
            ),
            category="code",
            tags=["file", "analysis", "memory", "automation"],
            steps=[
                ChainStep(
                    tool="file",
                    params={
                        "action": "read",
                        "path": "{input.path}",
                    },
                    description="Ler ficheiro do workspace",
                ),
                ChainStep(
                    tool="code",
                    params={
                        "action": "metrics",
                        "code": "{prev.data.content}",
                        "file": "{input.path}",
                    },
                    description="Calcular métricas do código",
                ),
                ChainStep(
                    tool="memory",
                    params={
                        "action": "store",
                        "key": "file:analysis:{input.path}",
                        "value": "Métricas: {prev.data}",
                    },
                    description="Guardar métricas na memória",
                ),
            ],
        ),
        ChainDefinition(
            name="web_search_analyze",
            description=(
                "Fetch de uma página, extrai URLs, "
                "e pesquisa cada URL no workspace"
            ),
            category="web",
            tags=["web", "search", "automation"],
            steps=[
                ChainStep(
                    tool="web",
                    params={
                        "action": "fetch_and_extract",
                        "url": "{input.url}",
                        "max_chars": 3000,
                    },
                    description="Fetch da página e extração de URLs",
                ),
                ChainStep(
                    tool="search",
                    params={
                        "action": "content",
                        "query": "{input.search_query}",
                        "directory": "{input.directory}",
                    },
                    description="Pesquisar pelo texto da página no workspace",
                ),
            ],
        ),
    ]

    registered = []
    for chain in chains:
        reg.register(chain)
        registered.append(chain.name)

    logger.info("Chains predefinidas registadas: %s", ", ".join(registered))
    return registered


# ======================================================================
#  Chain Runner (conveniência)
# ======================================================================

class ChainRunner:
    """
    Runner de conveniência que combina ChainRegistry + ChainExecutor.

    Uso típico::

        runner = ChainRunner(tool_manager)
        result = await runner.run("web_fetch_analyze_store", {"url": "https://..."})
    """

    def __init__(self, tool_manager: Any, registry: Optional[ToolRegistry] = None):
        self.executor = ChainExecutor(tool_manager, registry=registry)
        self.registry = get_chain_registry()

    async def run(
        self,
        chain_name: str,
        input_data: dict[str, Any],
        context: Optional[dict] = None,
    ) -> ChainResult:
        """
        Executa uma chain pelo nome.

        Args:
            chain_name: Nome da chain registada.
            input_data: Input inicial.
            context: Contexto de execução.

        Returns:
            ChainResult.
        """
        chain = self.registry.get(chain_name)
        if not chain:
            return ChainResult(
                success=False,
                chain_name=chain_name,
                steps=[],
                total_time_ms=0.0,
                error=f"Chain '{chain_name}' não encontrada",
            )
        return await self.executor.execute(chain, input_data, context=context)
