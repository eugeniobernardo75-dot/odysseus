"""
PlannerAgent — Odysseus V2

Agente especializado em planeamento de tarefas complexas.

Recebe um objetivo do utilizador e produz um plano estruturado com:
    - Passos (steps) com descrições claras
    - Agentes necessários para cada passo
    - Ferramentas necessárias
    - Dependências entre passos
    - Grupos de execução paralela
    - Estimativa de complexidade
    - Critérios de validação

O Orchestrator usa o PlannerAgent em vez de criar planos manualmente,
permitindo que o LLM decida a melhor estratégia para cada tarefa.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .agents import BaseAgent, AgentTask, AgentResult

logger = logging.getLogger(__name__)


# ======================================================================
#  Prompt do PlannerAgent
# ======================================================================

PLANNER_SYSTEM_PROMPT = """# Prompt do PlannerAgent — Odysseus V2

## Função

Tu és o agente especializado em planeamento do sistema Odysseus V2.
A tua única função é analisar pedidos do utilizador e produzir planos de execução estruturados.

NÃO executas tarefas. NÃO pesquisas informação. NÃO escreves código.
Apenas PLANEias.

## Formato de saída

Deves responder APENAS com um JSON válido neste formato:

```json
{
  "objective": "Resumo do objetivo principal",
  "complexity": "low|medium|high",
  "estimated_steps": 3,
  "steps": [
    {
      "step": "unique_step_name",
      "description": "Descrição clara do que fazer neste passo",
      "type": "agent|tool|llm|reasoning",
      "agent": "research|code|business|legal|finance|documentation|science|null",
      "tool": "web|file|shell|python|memory|search|code|null",
      "tool_params": {"command": "git status"},
      "dependencies": [],
      "parallel_group": "group_name_or_null",
      "validation_criteria": "Critério para saber se este passo está completo"
    }
  ],
  "parallel_groups": [
    { "name": "group_name", "steps": ["step1", "step2"], "description": "Porque correm em paralelo" }
  ],
  "estimated_tools": ["web", "file"],
  "risk_assessment": "Riscos identificados neste plano",
  "fallback_strategy": "O que fazer se algum passo falhar"
}
```

## Regras de planeamento

1. PROBLEMAS SIMPLES (perguntas factuais, conversa):
   - 1-2 passos apenas
   - Usar "llm" ou "reasoning" como tipo
   - complexity = "low"

2. PROBLEMAS MÉDIOS (pesquisa, análise de código):
   - 2-4 passos
   - Podem incluir agentes ou ferramentas
   - Identificar paralelismo quando possível
   - complexity = "medium"

3. PROBLEMAS COMPLEXOS (múltiplos agentes, ferramentas, dependências):
   - 4-8 passos
   - Usar parallel_groups para passos independentes
   - Incluir agentes especializados e ferramentas
   - Incluir fallback_strategy
   - complexity = "high"

4. Paralelismo:
   - Passos sem dependências entre si DEVEM estar no mesmo parallel_group
   - Passos com dependências DEVEM ser sequenciais (parallel_group = null)

5. Seleção de agentes:
   - research: pesquisa web, extração de conteúdo
   - code: análise de código, métricas, debugging
   - business: SWOT, canvas, estratégia
   - legal: análise jurídica, contratos
   - finance: análise financeira, rácios
   - documentation: relatórios, markdown
   - science: análise científica

6. Parâmetros de ferramentas:
   - Usa exclusivamente os campos definidos no JSON Schema da ferramenta indicada.
   - Para `shell`, o campo obrigatório é `command`; não uses `action`.

6. Seleção de ferramentas:
   - web: fetch de URLs, extração de conteúdo
   - file: ler/escrever/listar ficheiros
   - shell: executar comandos (requer aprovação)
   - python: executar código (requer aprovação)
   - memory: guardar/consultar memória
   - search: pesquisar no workspace
   - code: análise AST

7. validation_criteria:
   - Deve ser específico e verificável
   - Exemplo: "O código foi analisado sem erros de sintaxe"
   - Exemplo: "A pesquisa devolveu pelo menos 2 fontes"
"""


# ======================================================================
#  Estruturas do plano
# ======================================================================

@dataclass
class PlanStep:
    """Um passo individual de um plano de execução."""
    step: str
    description: str
    type: str = "reasoning"  # agent, tool, llm, reasoning
    agent: Optional[str] = None
    tool: Optional[str] = None
    tool_params: dict = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    parallel_group: Optional[str] = None
    validation_criteria: str = ""
    status: str = "pending"  # pending, running, completed, failed, skipped
    result: Any = None


@dataclass
class Plan:
    """Plano de execução completo gerado pelo PlannerAgent."""
    id: str
    query: str
    objective: str = ""
    complexity: str = "medium"
    steps: list[PlanStep] = field(default_factory=list)
    parallel_groups: list[dict] = field(default_factory=list)
    estimated_tools: list[str] = field(default_factory=list)
    risk_assessment: str = ""
    fallback_strategy: str = ""
    delegated_worker: Optional[str] = None
    created_at: str = ""
    status: str = "pending"  # pending, approved, running, completed, failed

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Converte o plano para dicionário serializável."""
        return {
            "id": self.id,
            "query": self.query[:100],
            "objective": self.objective,
            "complexity": self.complexity,
            "steps": [
                {
                    "step": s.step,
                    "description": s.description,
                    "type": s.type,
                    "agent": s.agent,
                    "tool": s.tool,
                    "dependencies": s.dependencies,
                    "parallel_group": s.parallel_group,
                    "validation_criteria": s.validation_criteria,
                    "status": s.status,
                }
                for s in self.steps
            ],
            "parallel_groups": self.parallel_groups,
            "estimated_tools": self.estimated_tools,
            "risk_assessment": self.risk_assessment,
            "fallback_strategy": self.fallback_strategy,
            "delegated_worker": self.delegated_worker,
            "created_at": self.created_at,
            "status": self.status,
            "n_steps": len(self.steps),
        }


# ======================================================================
#  PlannerAgent
# ======================================================================

class PlannerAgent(BaseAgent):
    """
    Agente especializado em planeamento.

    Usa o LLM para gerar planos estruturados a partir de objetivos,
    substituindo o planeamento manual do Orchestrator.

    Capacidades:
        - generate_plan: recebe objetivo, devolve plano estruturado
        - revise_plan: recebe feedback, melhora o plano
        - estimate_complexity: classifica complexidade do objetivo
    """

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__(
            name="planner",
            description=(
                "Planeamento de tarefas complexas. "
                "Recebe objetivos e produz planos estruturados "
                "com passos, agentes, ferramentas e dependências."
            ),
        )
        # O PlannerAgent deve usar o mesmo cliente injectado no Orchestrator.
        # Cair silenciosamente no singleton global podia seleccionar outro
        # endpoint/modelo e transformar um erro de LLM num plano de fallback.
        self._llm_client = llm_client

    def set_llm_client(self, client: Any) -> None:
        """Actualiza o cliente usado exclusivamente pelo planeador."""
        self._llm_client = client

    async def execute(self, task: AgentTask) -> AgentResult:
        logger.info("PlannerAgent a executar tarefa: %s", task.id)
        query = task.query
        context = task.context or {}

        try:
            # Se context já contém um plano, estamos a revisar
            if "existing_plan" in context:
                return await self._revise_plan(query, context)
            if "feedback" in context:
                return await self._revise_plan(query, context)

            return await self._generate_plan(query, context)

        except Exception as e:
            logger.error("PlannerAgent erro: %s", e, exc_info=True)
            return AgentResult(
                task_id=task.id,
                success=False,
                summary="Erro no planeamento",
                error=str(e),
                confidence=0.0,
            )

    # ------------------------------------------------------------------
    #  Gerar plano via LLM
    # ------------------------------------------------------------------

    async def _generate_plan(self, query: str, context: dict) -> AgentResult:
        """Gera um plano estruturado usando o LLM."""
        llm = self._get_llm()

        # Obter agentes e ferramentas disponíveis para contexto
        available_agents = self._get_available_agents()
        available_tools = self._get_available_tools()

        # Informação adicional do contexto
        memory_context = context.get("memory_context", "")
        tool_capabilities = context.get("tool_capabilities", {})

        messages = [
            {
                "role": "system",
                "content": PLANNER_SYSTEM_PROMPT + (
                    "\nREGRA DE CONTRATO: um comando git status deve ser representado como "
                    "type=tool, tool=shell, tool_params={command: git status}. "
                    "git não é type/tool e não inventes chamadas.\n"
                    f"\n\n## Agentes disponíveis\n{available_agents}\n\n"
                    f"## Ferramentas disponíveis\n{available_tools}\n\n"
                    + (f"## Contexto adicional\n{memory_context}\n\n" if memory_context else "")
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Gera um plano de execução para:\n\n{query}\n\n"
                    f"Responde APENAS com o JSON do plano, sem texto adicional."
                ),
            },
        ]

        try:
            resp = await llm.chat(messages, max_tokens=2048, temperature=0.3)
            if not resp.success or not resp.content.strip():
                return self._fallback_plan(query, "LLM não disponível para gerar plano")

            # Extrair JSON da resposta
            plan_dict = self._extract_plan_json(resp.content)
            if not plan_dict:
                logger.warning("PlannerAgent: resposta LLM não contém JSON válido")
                return self._fallback_plan(query, "Resposta LLM inválida")

            # Validar campos obrigatórios
            validation = self._validate_plan_dict(plan_dict)
            if validation:
                logger.warning("PlannerAgent: plano inválido: %s", validation)
                # Uma única correcção pelo próprio planeador. Não executa nem
                # normaliza chamadas: apenas pede ao mesmo LLM o contrato exato.
                correction = (
                    "O JSON anterior foi rejeitado: " + ", ".join(validation) +
                    ". Responde novamente APENAS com JSON válido. Para git status "
                    "usa type=tool, tool=shell e tool_params={command: git status}; "
                    "não uses type=git."
                )
                retry_messages = messages + [{"role": "user", "content": correction}]
                retry = await llm.chat(retry_messages, max_tokens=2048, temperature=0.0)
                retry_dict = self._extract_plan_json(retry.content) if retry.success else None
                retry_validation = self._validate_plan_dict(retry_dict) if retry_dict else ["Resposta de correcção inválida"]
                if retry_validation:
                    return self._fallback_plan(query, f"Plano inválido: {validation}")
                plan_dict = retry_dict
                resp = retry

            # Construir objeto Plan
            plan = self._build_plan(query, plan_dict)
            steps_text = "\n".join(
                f"  {i+1}. {s.description} [{s.type}]"
                for i, s in enumerate(plan.steps)
            )

            return AgentResult(
                task_id=f"PLAN-{uuid.uuid4().hex[:8].upper()}",
                success=True,
                summary=f"Plano criado: {len(plan.steps)} passo(s), complexidade {plan.complexity}",
                details={
                    "plan": plan.to_dict(),
                    "plan_object": plan,
                    "raw_llm_response": resp.content[:500],
                },
                confidence=0.85,
                suggestions=[
                    "O plano pode ser ajustado manualmente se necessário",
                    "Passos independentes serão executados em paralelo",
                ],
            )

        except Exception as e:
            logger.error("PlannerAgent: erro ao gerar plano: %s", e)
            return self._fallback_plan(query, str(e))

    # ------------------------------------------------------------------
    #  Revisar plano com feedback
    # ------------------------------------------------------------------

    async def _revise_plan(self, query: str, context: dict) -> AgentResult:
        """Revisa um plano existente com base em feedback."""
        existing_plan = context.get("existing_plan", {})
        feedback = context.get("feedback", "")

        if isinstance(existing_plan, Plan):
            existing_plan = existing_plan.to_dict()

        llm = self._get_llm()
        messages = [
            {
                "role": "system",
                "content": PLANNER_SYSTEM_PROMPT + (
                    "\nREGRA DE CONTRATO: um comando git status deve ser representado como "
                    "type=tool, tool=shell, tool_params={command: git status}. "
                    "git não é type/tool e não inventes chamadas.\n"
                    "\n\nEstás a revisar um plano existente com base em feedback do utilizador."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Plano original:\n{json.dumps(existing_plan, indent=2, default=str)}\n\n"
                    f"Feedback do utilizador:\n{feedback}\n\n"
                    f"Gera uma versão revista do plano. Responde APENAS com o JSON."
                ),
            },
        ]

        try:
            resp = await llm.chat(messages, max_tokens=2048, temperature=0.3)
            if not resp.success:
                return AgentResult(
                    task_id=f"PLAN-{uuid.uuid4().hex[:8].upper()}",
                    success=False,
                    summary="Erro ao revisar plano",
                    error="LLM não disponível",
                    confidence=0.0,
                )

            plan_dict = self._extract_plan_json(resp.content)
            if plan_dict:
                plan = self._build_plan(query, plan_dict)
                return AgentResult(
                    task_id=f"PLAN-{uuid.uuid4().hex[:8].upper()}",
                    success=True,
                    summary=f"Plano revisto: {len(plan.steps)} passo(s)",
                    details={"plan": plan.to_dict(), "plan_object": plan},
                    confidence=0.8,
                )

            return self._fallback_plan(query, "Não foi possível revisar o plano")

        except Exception as e:
            return self._fallback_plan(query, str(e))

    # ------------------------------------------------------------------
    #  Fallback (quando LLM não está disponível)
    # ------------------------------------------------------------------

    def _fallback_plan(self, query: str, reason: str) -> AgentResult:
        """Gera um plano simples de fallback quando o LLM não está disponível."""
        plan = Plan(
            id=f"PLAN-{uuid.uuid4().hex[:8].upper()}",
            query=query,
            objective=f"Processar: {query[:80]}",
            complexity="medium",
            fallback_strategy=f"Fallback ativado: {reason}",
        )

        # Detetar agentes relevantes por keywords
        query_lower = query.lower()
        agents_map = {
            "research": ["pesquisa", "pesquisar", "informação", "o que é", "quem", "url", "http"],
            "code": ["código", "programar", "python", "função", "bug", "ficheiro"],
            "business": ["negócio", "swot", "empresa", "mercado", "estratégia"],
            "legal": ["jurídico", "lei", "contrato", "advogado", "legal"],
            "finance": ["financeiro", "investimento", "ação", "bolsa", "receita"],
            "documentation": ["documentar", "relatório", "readme", "documentação"],
            "science": ["científico", "artigo", "estudo", "metodologia"],
        }

        selected_agents = []
        for agent_name, keywords in agents_map.items():
            if any(kw in query_lower for kw in keywords):
                selected_agents.append(agent_name)

        if selected_agents:
            # Criar passos para agentes encontrados
            for agent_name in selected_agents:
                step = PlanStep(
                    step=f"execute_{agent_name}",
                    description=f"Executar análise com agente {agent_name}",
                    type="agent",
                    agent=agent_name,
                    parallel_group="agents" if len(selected_agents) > 1 else None,
                    validation_criteria=f"Agente {agent_name} concluiu a análise",
                )
                plan.steps.append(step)

            if len(selected_agents) > 1:
                plan.parallel_groups.append({
                    "name": "agents",
                    "steps": [s.step for s in plan.steps],
                    "description": "Agentes independentes em paralelo",
                })

        # Passo de consolidação
        plan.steps.append(PlanStep(
            step="consolidate",
            description="Consolidar resultados e gerar resposta",
            type="llm",
            dependencies=[s.step for s in plan.steps],
            validation_criteria="Resposta coerente gerada",
        ))

        return AgentResult(
            task_id=f"PLAN-{uuid.uuid4().hex[:8].upper()}",
            success=True,
            summary=f"Plano de fallback: {len(plan.steps)} passo(s)",
            details={
                "plan": plan.to_dict(),
                "plan_object": plan,
                "fallback_reason": reason,
            },
            confidence=0.5,
            suggestions=["O LLM está offline — plano gerado por regras"],
        )

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _get_llm(self):
        """Obtém o cliente LLM do Orchestrator, sem trocar de singleton."""
        if self._llm_client is not None:
            return self._llm_client
        from .llm_client import get_llm_client
        return get_llm_client()

    def _get_available_agents(self) -> str:
        """Lista agentes disponíveis para contexto do LLM."""
        try:
            from .agents import get_agent_factory
            factory = get_agent_factory()
            agents = factory.list_agents()
            return "\n".join(f"  - {a['name']}: {a['description']}" for a in agents)
        except Exception:
            return "  (não foi possível obter lista de agentes)"

    def _get_available_tools(self) -> str:
        """Lista ferramentas disponíveis para contexto do LLM."""
        try:
            from .tool_registry import get_tool_registry
            registry = get_tool_registry()
            tools = registry.list_tools()
            return "\n".join(
                f"  - {t['name']}: {t['description']} "
                f"(categoria: {t['category']}; parâmetros: "
                f"{json.dumps(t.get('parameters', {}), ensure_ascii=False, separators=(',', ':'))})"
                for t in tools
            )
        except Exception:
            return "  (ferramentas não disponíveis)"

    def _extract_plan_json(self, text: str) -> Optional[dict]:
        """Extrai um dicionário JSON da resposta do LLM."""
        import re

        # Tentar extrair bloco JSON ```json ... ```
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Tentar encontrar JSON diretamente
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        return None

    def _validate_plan_dict(self, plan: dict) -> list[str]:
        """Valida o plano: estruturais, deps, ciclos, paralelismo."""
        errors = []

        if "steps" not in plan or not plan["steps"]:
            return ["Plano não contém steps"]

        allowed_types = {"agent", "tool", "llm", "reasoning"}
        steps = plan["steps"]

        # ── Estruturais + recolher ids ──
        ids: list[str] = []
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                errors.append(f"Step {i}: não é um objecto")
                continue
            if "step" not in step or not str(step.get("step") or "").strip():
                errors.append(f"Step {i}: falta campo 'step' ou vazio")
                continue
            sid = str(step["step"]).strip()
            ids.append(sid)

            if "description" not in step or not str(step.get("description") or "").strip():
                errors.append(f"Step {i} ({sid}): falta campo 'description'")

            step_type = step.get("type", "reasoning")
            if step_type not in allowed_types:
                errors.append(f"Step {i} ({sid}): tipo inválido '{step_type}'")

            if step_type == "tool":
                if not step.get("tool"):
                    errors.append(f"Step {i} ({sid}): passo tool sem campo 'tool'")
                elif not isinstance(step.get("tool_params", {}), dict):
                    errors.append(f"Step {i} ({sid}): tool_params deve ser objecto")

        # ── Ids duplicados ──
        seen = set()
        dupes = set()
        for sid in ids:
            if sid in seen:
                dupes.add(sid)
            seen.add(sid)
        if dupes:
            errors.append(f"Ids duplicados: {sorted(dupes)}")

        # ── Dependências inexistentes + auto-dependência ──
        known = set(ids)
        for step in steps:
            if not isinstance(step, dict):
                continue
            sid = str(step.get("step") or "").strip()
            if not sid:
                continue
            deps = step.get("dependencies") or []
            if not isinstance(deps, list):
                errors.append(f"Step '{sid}': dependencies deve ser lista")
                continue
            for dep in deps:
                dep_s = str(dep).strip()
                if dep_s == sid:
                    errors.append(f"Step '{sid}': auto-dependência")
                elif dep_s not in known:
                    errors.append(f"Step '{sid}': dependência inexistente '{dep_s}'")

        # ── Ciclos (Kahn) ──
        graph: dict[str, set[str]] = {sid: set() for sid in ids}
        for step in steps:
            if not isinstance(step, dict):
                continue
            sid = str(step.get("step") or "").strip()
            if sid not in graph:
                continue
            for dep in (step.get("dependencies") or []):
                dep_s = str(dep).strip()
                if dep_s in graph:
                    graph[sid].add(dep_s)

        in_degree = {sid: len(deps) for sid, deps in graph.items()}
        queue = [sid for sid, d in in_degree.items() if d == 0]
        visited = 0
        while queue:
            cur = queue.pop()
            visited += 1
            for sid, deps in graph.items():
                if cur in deps:
                    in_degree[sid] -= 1
                    if in_degree[sid] == 0:
                        queue.append(sid)
        if visited != len(graph):
            cycle_nodes = [sid for sid, d in in_degree.items() if d > 0]
            errors.append(f"Ciclo de dependências entre: {sorted(cycle_nodes)}")

        # ── parallel_group coerente ──
        groups: dict[str, list[dict]] = {}
        for step in steps:
            if not isinstance(step, dict):
                continue
            grp = step.get("parallel_group")
            if not grp:
                continue
            groups.setdefault(str(grp), []).append(step)

        for grp_name, grp_steps in groups.items():
            grp_ids = {str(s.get("step") or "") for s in grp_steps}
            for s in grp_steps:
                sid = str(s.get("step") or "")
                internal_deps = {
                    str(d) for d in (s.get("dependencies") or [])
                    if str(d) in grp_ids
                }
                if internal_deps:
                    errors.append(
                        f"parallel_group '{grp_name}': step '{sid}' "
                        f"tem dependências dentro do grupo ({sorted(internal_deps)})"
                    )

        return errors

    def _build_plan(self, query: str, plan_dict: dict) -> Plan:
        """Constrói objeto Plan a partir de dicionário."""
        steps = []
        for s in plan_dict.get("steps", []):
            step = PlanStep(
                step=s.get("step", f"step_{len(steps)+1}"),
                description=s.get("description", ""),
                type=s.get("type", "reasoning"),
                agent=s.get("agent"),
                tool=s.get("tool"),
                tool_params=s.get("tool_params", {}),
                dependencies=s.get("dependencies", []),
                parallel_group=s.get("parallel_group"),
                validation_criteria=s.get("validation_criteria", ""),
            )
            steps.append(step)

        return Plan(
            id=f"PLAN-{uuid.uuid4().hex[:8].upper()}",
            query=query,
            objective=plan_dict.get("objective", query[:80]),
            complexity=plan_dict.get("complexity", "medium"),
            steps=steps,
            parallel_groups=plan_dict.get("parallel_groups", []),
            estimated_tools=plan_dict.get("estimated_tools", []),
            risk_assessment=plan_dict.get("risk_assessment", ""),
            fallback_strategy=plan_dict.get("fallback_strategy", ""),
        )

    def can_handle(self, query: str) -> float:
        """Score baseado em keywords de planeamento."""
        keywords = [
            "plano", "planeamento", "planear", "planeie", "plan",
            "estratégia", "passos", "etapas", "sequência",
            "organizar", "estruturar", "dividir",
            "como fazer", "como criar", "como implementar",
            "preciso de", "quero fazer", "quero criar",
            "quero construir", "quero desenvolver",
        ]
        query_lower = query.lower()
        score = sum(1 for kw in keywords if kw in query_lower)
        return min(score / 3, 1.0)
