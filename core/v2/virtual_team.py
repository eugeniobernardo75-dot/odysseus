"""
Equipa Virtual — Odysseus V2
=============================

Os três modelos locais formam a equipa virtual oficial do Odysseus V2.
Não são concorrentes — são especialistas. O Orchestrator decide
automaticamente qual utilizar (modo automático).

    - Secretário IA        → llama3.2:3b  (atendimento, agenda, emails, CRM, ...)
    - Técnico IA           → qwen3:8b     (código, debug, infraestrutura, ...)
    - Diretor Estratégico  → qwen3:8b     (planeamento, negócios, estratégia, ...)

Em modo MANUAL, o utilizador escolhe diretamente um modelo (Gemma, Llama ou
Qwen) e o Orchestrator respeita essa escolha — nunca redireciona para outro.
"""

from __future__ import annotations

from typing import Optional

# ======================================================================
#  Definição da equipa virtual
# ======================================================================

VIRTUAL_TEAM = {
    "secretario": {
        "name": "Secretário IA",
        "model": "llama3.2:3b",
        "role": "Atendimento",
        "responsibilities": [
            "Atendimento", "Agenda", "WhatsApp", "Emails",
            "CRM", "Documentação", "Organização",
        ],
        "keywords": [
            "atendimento", "agenda", "marcar", "whatsapp", "email", "e-mail",
            "crm", "documentação", "documento", "organiza", "lembrete",
            "reunião", "reuniao", "cliente", "mensagem", "contato", "contacto",
        ],
        "description": (
            "Secretário IA — trata de atendimento, agenda, WhatsApp, emails, "
            "CRM, documentação e organização do dia-a-dia."
        ),
    },
    "tecnico": {
        "name": "Técnico IA",
        "model": "qwen3:8b",
        "role": "Técnico",
        "responsibilities": [
            "Código", "Debug", "Infraestrutura", "Servidores",
            "Segurança", "Logs", "Automação", "Backups",
        ],
        "keywords": [
            "código", "codigo", "programar", "debug", "bug", "infraestrutura",
            "servidor", "servidores", "segurança", "seguranca", "log", "logs",
            "automação", "automacao", "backup", "backups", "python", "docker",
            "rede", "firewall", "deploy", "erro", "exceção", "exception",
        ],
        "description": (
            "Técnico IA — especialista em código, debug, infraestrutura, "
            "servidores, segurança, logs, automação e backups."
        ),
    },
    "diretor": {
        "name": "Diretor Estratégico IA",
        "model": "qwen3:8b",
        "role": "Estratégia",
        "responsibilities": [
            "Planeamento", "Negócios", "Estratégia", "Análise",
            "Relatórios", "Decisões", "Academia", "Holding",
        ],
        "keywords": [
            "planeamento", "planejamento", "negócio", "negocio", "negócios",
            "negocios", "estratégia", "estrategia", "análise", "analise",
            "relatório", "relatorio", "decisão", "decisao", "academia",
            "holding", "investimento", "orçamento", "orcamento", "mercado",
            "plano", "crescimento", "receita", "lucro", "estudo",
        ],
        "description": (
            "Diretor Estratégico IA — planeamento, negócios, estratégia, "
            "análise, relatórios, decisões, academia e holding."
        ),
    },
}

# Modelo padrão quando não há correspondência clara
DEFAULT_TEAM_MODEL = "qwen3:8b"


# ======================================================================
#  Seleção
# ======================================================================

def select_specialist(query: str) -> dict:
    """
    Escolhe o especialista mais adequado para uma consulta.

    Usa correspondência de palavras-chave (substring, case-insensitive).
    Devolve o dict da equipa virtual correspondente. Em caso de empate,
    devolve o primeiro com maior score.

    Args:
        query: Texto da consulta do utilizador.

    Returns:
        Dict do especialista (name, model, keywords, ...).
    """
    if not query:
        return VIRTUAL_TEAM["tecnico"]

    q = query.lower()
    best = None
    best_score = 0
    for specialist in VIRTUAL_TEAM.values():
        score = sum(1 for kw in specialist["keywords"] if kw in q)
        if score > best_score:
            best_score = score
            best = specialist

    if best is None or best_score == 0:
        # Sem correspondência clara — usa o modelo padrão (técnico).
        return VIRTUAL_TEAM["tecnico"]
    return best


def select_specialist_model(query: str) -> str:
    """Devolve o modelo (nome) do especialista escolhido para a consulta."""
    return select_specialist(query)["model"]


def get_virtual_team() -> list[dict]:
    """Devolve a lista de especialistas da equipa virtual (para a API/UI)."""
    return [dict(v) for v in VIRTUAL_TEAM.values()]


def get_team_by_model(model: Optional[str]) -> Optional[dict]:
    """Devolve o especialista associado a um modelo, se existir."""
    if not model:
        return None
    for specialist in VIRTUAL_TEAM.values():
        if specialist["model"] == model:
            return specialist
    return None


def is_virtual_model(model: Optional[str]) -> bool:
    """True se o modelo faz parte da equipa virtual oficial."""
    return any(
        specialist["model"] == model for specialist in VIRTUAL_TEAM.values()
    )
