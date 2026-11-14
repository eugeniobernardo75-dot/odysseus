"""Operational agents with explicit, least-privilege tool scopes."""

from __future__ import annotations

import logging

from .agents import AgentResult, AgentTask, BaseAgent, _count_keyword_matches, _get_llm_client

logger = logging.getLogger(__name__)


class _OperationalWorker(BaseAgent):
    is_operational_worker = True
    allowed_tools: frozenset[str] = frozenset()
    keywords: tuple[str, ...] = ()
    role_prompt = ""

    def __init__(self) -> None:
        super().__init__(self.name, self.description)

    def can_handle(self, query: str) -> float:
        matches = _count_keyword_matches(query, list(self.keywords))
        return min(matches / 3, 1.0)

    async def execute(self, task: AgentTask) -> AgentResult:
        """Analyse a delegated task using only context selected by UserAgent."""
        reference_data = (task.context or {}).get("useragent_reference_data", "")
        messages = [
            {
                "role": "system",
                "content": (
                    f"{self.role_prompt}\n\n"
                    "Não acedas directamente ao Brain Vault, MIND OS ou memória global. "
                    "Usa apenas o contexto explicitamente fornecido pelo UserAgent. "
                    "Não declares operações como executadas; a execução de ferramentas "
                    "é feita pelo servidor e sujeita à allowlist e aprovação."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Tarefa: {task.query}\n\n"
                    f"Contexto autorizado do UserAgent:\n{reference_data}"
                ),
            },
        ]
        try:
            response = await _get_llm_client().chat(messages, max_tokens=700)
        except Exception as exc:
            logger.error("Worker %s falhou ao analisar a tarefa: %s", self.name, exc)
            return AgentResult(
                task_id=task.id,
                success=False,
                summary=f"Falha na análise da tarefa pelo agente {self.name}.",
                error=str(exc),
            )

        if not response.success or not response.content.strip():
            error = response.error or "O modelo não devolveu uma análise."
            return AgentResult(
                task_id=task.id,
                success=False,
                summary=f"Não foi possível analisar a tarefa pelo agente {self.name}.",
                error=error,
            )

        return AgentResult(
            task_id=task.id,
            success=True,
            summary=response.content.strip(),
            details={"worker": self.name},
        )


class SecretariaWorker(_OperationalWorker):
    """Tarefas administrativas e documentais suportadas."""

    name = "secretaria"
    description = "Trata tarefas administrativas e documentais dentro do âmbito autorizado."
    keywords = (
        "secretaria", "secretária", "carta", "memorando", "ata", "agenda",
        "reunião", "administrativo", "documento", "formatação", "email",
        "correio", "calendário", "calendario", "agendar",
    )
    role_prompt = (
        "És a Secretária operacional. Organiza comunicações e documentos "
        "suportados pelas capacidades disponíveis; não assumes acesso a email "
        "ou calendário, que não estão confirmados no catálogo V2."
    )
    allowed_tools = frozenset(
        {"web", "file.read", "file.list", "file.search", "file.exists"}
    )


class TecnicoWorker(_OperationalWorker):
    """Technical diagnosis and authorized workspace operations."""

    name = "tecnico"
    description = "Diagnostica problemas técnicos e executa operações autorizadas no workspace."
    keywords = (
        "tecnico", "técnico", "código", "codigo", "ficheiro", "arquivo",
        "debug", "erro", "teste", "script", "configuração", "configuracao",
        "corrigir", "desenvolvimento", "executar", "executa", "comando",
        "shell", "terminal", "git", "pytest", "python",
    )
    role_prompt = (
        "És o Técnico operacional. Diagnostica, testa e corrige tarefas técnicas "
        "dentro do workspace aprovado. Operações de escrita, shell e Python "
        "continuam sujeitas aos controlos do servidor."
    )
    allowed_tools = frozenset(
        {
            "file.read", "file.write", "file.list", "file.search", "file.exists",
            "shell", "python", "search", "code", "git",
        }
    )


class DirectorWorker(_OperationalWorker):
    """Operational planning, prioritization, and coordination."""

    name = "director-geral"
    description = "Coordena prioridades, planeamento e integração de resultados."
    keywords = (
        "director", "diretor", "director-geral", "diretor-geral",
        "prioridade", "prioridades", "coordena", "coordenar", "plano",
        "planeamento", "estratégia", "estrategia", "alternativas",
    )
    role_prompt = (
        "És o Director-Geral operacional. Coordena prioridades, avalia "
        "alternativas e integra resultados dentro da autoridade aplicável."
    )
    allowed_tools = frozenset({"web"})


def register_workers(factory) -> None:
    """Register the three concrete operational agents."""
    factory.register(SecretariaWorker())
    factory.register(TecnicoWorker())
    factory.register(DirectorWorker())
