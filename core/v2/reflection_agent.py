"""
ReflectionAgent — Odysseus V2

Agente especializado em autoavaliação e melhoria de respostas.

Após cada resposta gerada (pelo LLM ou por agentes), o ReflectionAgent:
    - Analisa a resposta para erros factuais, lógicos ou de compreensão
    - Verifica se a resposta responde completamente à pergunta original
    - Identifica ambiguidades ou informação em falta
    - Sugere melhorias específicas
    - Gera uma versão melhorada (se necessário)

Integra-se no pipeline do Orchestrator como etapa obrigatória
após a geração da resposta, antes de enviar ao utilizador.

Fluxo:
    Pergunta → Agentes/LLM → Resposta → ReflectionAgent → Resposta Final
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
#  Prompt do ReflectionAgent
# ======================================================================

REFLECTION_SYSTEM_PROMPT = """# Prompt do ReflectionAgent — Odysseus V2

## Função

Tu és o agente especializado em autoavaliação do sistema Odysseus V2.
A tua função é analisar respostas geradas pelo sistema e garantir que são
corretas, completas e úteis antes de serem enviadas ao utilizador.

NÃO és o utilizador. NÃO és o assistente original.
És um crítico construtivo que ajuda a melhorar a qualidade do sistema.

## O que avalias

Para cada resposta, deves verificar:

1. CORREÇÃO FACTUAL
   - A resposta contém factos verificáveis?
   - Há contradições internas?
   - As fontes são referenciadas corretamente?

2. COMPLETUDE
   - A resposta responde à pergunta original?
   - Falta alguma parte importante?
   - O utilizador consegue agir com esta resposta?

3. CLAREZA
   - A linguagem é clara e apropriada?
   - Há ambiguidades?
   - A estrutura é lógica?

4. SEGURANÇA
   - A resposta sugere ações inseguras?
   - Há instruções que podem ser perigosas?
   - Informação sensível foi exposta?

5. UTILIDADE
   - A resposta é prática e acionável?
   - Inclui exemplos ou alternativas?
   - O tom é apropriado?

## Formato de saída

Deves responder APENAS com um JSON válido neste formato:

```json
{
  "quality_score": 8.5,
  "scale": "1-10",
  "assessment": {
    "factual_correctness": {
      "score": 9,
      "issues": ["Ponto 1 que pode estar incorreto..."],
      "strengths": ["A resposta cita fontes confiáveis"]
    },
    "completeness": {
      "score": 7,
      "issues": ["Não aborda o aspeto X da pergunta"],
      "strengths": ["Cobre os pontos principais"]
    },
    "clarity": {
      "score": 8,
      "issues": ["O parágrafo 3 podia ser mais claro"],
      "strengths": ["Boa estrutura geral"]
    },
    "safety": {
      "score": 10,
      "issues": [],
      "strengths": ["Nenhum risco identificado"]
    },
    "usefulness": {
      "score": 8,
      "issues": ["Podia incluir um exemplo concreto"],
      "strengths": ["Resposta prática e direta"]
    }
  },
  "critical_issues": [],
  "improvements": [
    "Sugestão específica de melhoria 1",
    "Sugestão específica de melhoria 2"
  ],
  "requires_revision": false,
  "revised_response": "Versão melhorada da resposta (se requires_revision for true)",
  "summary": "Resumo da avaliação em 1-2 frases"
}
```

## Regras

1. Se quality_score >= 8.0 e sem critical_issues → requires_revision = false
2. Se quality_score < 8.0 ou existir critical_issues → requires_revision = true
3. critical_issues são apenas problemas graves (factos errados, segurança, omissão crítica)
4. improvements são sugestões construtivas e acionáveis
5. revised_response só é preenchido se requires_revision = true
6. Sê honesto mas construtivo — o objetivo é melhorar, não criticar
"""


# ======================================================================
#  Estruturas de avaliação
# ======================================================================

@dataclass
class ReflectionScore:
    """Pontuação de um aspeto específico da avaliação."""
    score: float = 7.0
    issues: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)


@dataclass
class ReflectionResult:
    """Resultado completo da reflexão sobre uma resposta."""
    quality_score: float = 7.0
    scale: str = "1-10"
    assessment: dict = field(default_factory=dict)
    critical_issues: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    requires_revision: bool = False
    revised_response: str = ""
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "quality_score": self.quality_score,
            "scale": self.scale,
            "assessment": self.assessment,
            "critical_issues": self.critical_issues,
            "improvements": self.improvements,
            "requires_revision": self.requires_revision,
            "revised_response": self.revised_response[:500] if self.revised_response else "",
            "summary": self.summary,
        }


# ======================================================================
#  ReflectionAgent
# ======================================================================

class ReflectionAgent(BaseAgent):
    """
    Agente especializado em autoavaliação de respostas.

    Analisa respostas geradas pelo sistema para garantir qualidade,
    correção e completude antes da entrega ao utilizador.

    Capacidades:
        - reflect: avaliar uma resposta gerada
        - improve: gerar versão melhorada
        - validate: verificar factos e segurança
    """

    def __init__(self):
        super().__init__(
            name="reflection",
            description=(
                "Autoavaliação e melhoria de respostas. "
                "Analisa qualidade, correção factual, completude, "
                "segurança e utilidade antes da entrega ao utilizador."
            ),
        )

    async def execute(self, task: AgentTask) -> AgentResult:
        logger.info("ReflectionAgent a avaliar: %s", task.id)
        query = task.query
        context = task.context or {}

        # O que estamos a avaliar
        # A query do AgentTask pode conter a pergunta original + resposta
        # Ou podemos receber no context

        original_question = context.get("original_question", query)
        response_to_evaluate = context.get("response_to_evaluate", query)

        try:
            # Se a resposta for pequena ou óbvia, não precisa de reflexão
            if len(response_to_evaluate) < 50:
                return self._quick_pass(original_question, response_to_evaluate)

            # Se LLM não estiver disponível, usar reflexão baseada em regras
            llm = self._get_llm()
            llm_available = await self._check_llm(llm)

            if not llm_available:
                return self._rule_based_reflection(original_question, response_to_evaluate)

            # Reflexão via LLM
            return await self._llm_reflection(original_question, response_to_evaluate, llm)

        except Exception as e:
            logger.error("ReflectionAgent erro: %s", e, exc_info=True)
            return AgentResult(
                task_id=task.id,
                success=False,
                summary="Erro na reflexão",
                error=str(e),
                confidence=0.0,
            )

    # ------------------------------------------------------------------
    #  Reflexão via LLM
    # ------------------------------------------------------------------

    async def _llm_reflection(self, question: str, response: str,
                               llm) -> AgentResult:
        """Avalia a resposta usando o LLM."""
        messages = [
            {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"## Pergunta original\n{question}\n\n"
                    f"## Resposta gerada\n{response}\n\n"
                    f"Avalia a resposta de acordo com os critérios definidos. "
                    f"Responde APENAS com o JSON da avaliação."
                ),
            },
        ]

        try:
            resp = await llm.chat(messages, max_tokens=2048, temperature=0.2)
            if not resp.success or not resp.content.strip():
                return self._rule_based_reflection(question, response)

            # Extrair JSON
            reflection_dict = self._extract_reflection_json(resp.content)
            if not reflection_dict:
                logger.warning("ReflectionAgent: resposta LLM não contém JSON válido")
                return self._rule_based_reflection(question, response)

            # Construir resultado
            result = self._build_reflection_result(reflection_dict)

            # Se requer revisão e temos LLM disponível, gerar resposta melhorada
            if result.requires_revision and not result.revised_response:
                revised = await self._generate_improved(question, response, llm)
                if revised:
                    result.revised_response = revised

            details = {
                "original_question": question[:100],
                "original_response": response[:500],
                "reflection": result.to_dict(),
            }

            if result.revised_response:
                details["revised_response"] = result.revised_response[:500]

            return AgentResult(
                task_id=f"REFL-{uuid.uuid4().hex[:8].upper()}",
                success=True,
                summary=(
                    f"Reflexão: score {result.quality_score}/10"
                    + (" (requer revisão)" if result.requires_revision else " ✓")
                ),
                details=details,
                confidence=result.quality_score / 10 if result.quality_score > 0 else 0.5,
                suggestions=result.improvements,
            )

        except Exception as e:
            logger.error("ReflectionAgent: erro no LLM: %s", e)
            return self._rule_based_reflection(question, response)

    # ------------------------------------------------------------------
    #  Reflexão baseada em regras (fallback)
    # ------------------------------------------------------------------

    def _rule_based_reflection(self, question: str, response: str) -> AgentResult:
        """Avaliação baseada em regras quando o LLM não está disponível."""
        issues = []
        improvements = []
        question_lower = question.lower()
        response_lower = response.lower()

        # Verificar se a resposta é demasiado genérica
        generic_phrases = [
            "não sei", "não tenho informação", "não posso responder",
            "como assistente de ia", "como modelo de linguagem",
        ]
        for phrase in generic_phrases:
            if phrase in response_lower:
                issues.append(f"Resposta contém frase genérica: '{phrase}'")
                improvements.append("Tenta fornecer informação específica mesmo com limitações")

        # Verificar se a resposta responde à pergunta
        question_words = set(question_lower.split())
        response_words = set(response_lower.split())
        overlap = len(question_words & response_words)
        if overlap < 2 and len(question_words) > 5:
            issues.append("Pouca sobreposição entre pergunta e resposta")
            improvements.append("Garantir que a resposta aborda diretamente a pergunta")

        # Verificar comprimento mínimo
        if len(response.split()) < 10:
            issues.append("Resposta muito curta")
            improvements.append("Expandir a resposta com mais detalhe")

        # Verificar segurança
        dangerous_terms = ["rm -rf", "sudo", "chmod 777", "format", "mkfs"]
        for term in dangerous_terms:
            if term in response_lower:
                issues.append(f"Resposta contém termo potencialmente perigoso: '{term}'")
                improvements.append("Adicionar aviso de segurança antes de comandos destrutivos")

        quality_score = 8.0
        if issues:
            quality_score = max(5.0, 8.0 - len(issues) * 1.0)

        details = {
            "original_question": question[:100],
            "original_response": response[:500],
            "reflection": {
                "quality_score": quality_score,
                "scale": "1-10",
                "critical_issues": issues[:3],
                "improvements": improvements[:3],
                "requires_revision": len(issues) > 1,
                "summary": (
                    f"Avaliação por regras: {quality_score}/10"
                    + (f", {len(issues)} questão(õe)s encontrada(s)" if issues else ", sem problemas detetados")
                ),
                "note": "Avaliação baseada em regras (LLM não disponível)",
            },
        }

        return AgentResult(
            task_id=f"REFL-{uuid.uuid4().hex[:8].upper()}",
            success=True,
            summary=(
                f"Reflexão (regras): {quality_score}/10"
                + (" ⚠️" if issues else " ✓")
            ),
            details=details,
            confidence=0.6,
            suggestions=improvements or ["Resposta parece adequada — sem sugestões automáticas"],
        )

    # ------------------------------------------------------------------
    #  Gerar resposta melhorada
    # ------------------------------------------------------------------

    async def _generate_improved(self, question: str, response: str,
                                  llm) -> Optional[str]:
        """Gera uma versão melhorada da resposta."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Tu és um assistente que melhora respostas. "
                        "Recebes uma pergunta e uma resposta, e deves "
                        "gerar uma versão melhorada que seja mais clara, "
                        "completa e útil. Mantém o tom original."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"## Pergunta\n{question}\n\n"
                        f"## Resposta atual\n{response}\n\n"
                        f"Gera uma versão melhorada desta resposta."
                    ),
                },
            ]

            resp = await llm.chat(messages, max_tokens=2048, temperature=0.3)
            if resp.success and resp.content.strip():
                improved = resp.content.strip()
                # Não devolver se for igual à original
                if improved != response.strip():
                    return improved

        except Exception as e:
            logger.debug("ReflectionAgent: erro ao gerar melhoria: %s", e)

        return None

    # ------------------------------------------------------------------
    #  Pass rápido (respostas curtas)
    # ------------------------------------------------------------------

    def _quick_pass(self, question: str, response: str) -> AgentResult:
        """Para respostas muito curtas que não precisam de reflexão."""
        return AgentResult(
            task_id=f"REFL-{uuid.uuid4().hex[:8].upper()}",
            success=True,
            summary="Reflexão: resposta curta, pass rápido ✓",
            details={
                "original_question": question[:100],
                "original_response": response,
                "reflection": {
                    "quality_score": 8.0,
                    "scale": "1-10",
                    "note": "Resposta curta — reflexão automática aprovada",
                    "requires_revision": False,
                },
            },
            confidence=0.9,
        )

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _get_llm(self):
        from .llm_client import get_llm_client
        return get_llm_client()

    async def _check_llm(self, llm) -> bool:
        try:
            return await llm.check_availability()
        except Exception:
            return False

    def _extract_reflection_json(self, text: str) -> Optional[dict]:
        """Extrai JSON de avaliação da resposta do LLM."""
        import re

        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        return None

    def _build_reflection_result(self, data: dict) -> ReflectionResult:
        """Constrói ReflectionResult a partir de dicionário."""
        return ReflectionResult(
            quality_score=data.get("quality_score", 7.0),
            scale=data.get("scale", "1-10"),
            assessment=data.get("assessment", {}),
            critical_issues=data.get("critical_issues", []),
            improvements=data.get("improvements", []),
            requires_revision=data.get("requires_revision", False),
            revised_response=data.get("revised_response", ""),
            summary=data.get("summary", ""),
        )

    def can_handle(self, query: str) -> float:
        """Score para deteção de pedidos de revisão/avaliação."""
        keywords = [
            "revisão", "revisar", "avalia", "avaliar",
            "melhorar", "refinar", "corrigir",
            "qualidade", "verificar", "confirmar",
            "review", "reflect", "improve",
        ]
        query_lower = query.lower()
        score = sum(1 for kw in keywords if kw in query_lower)
        return min(score / 3, 1.0)
