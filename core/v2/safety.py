"""
Módulo de Segurança — Odysseus V2

Camada de validação que previne ações perigosas, destrutivas ou não autorizadas.

Regras:
    - Nunca apagar ficheiros sem confirmação explícita.
    - Nunca modificar código sem autorização.
    - Nunca instalar software automaticamente.
    - Nunca executar comandos perigosos (rm -rf, dd, formatação, etc.).
    - Nunca alterar configurações críticas do sistema.
    - Todas as operações de alto risco necessitam de confirmação em duas etapas.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Lista de padrões de comandos perigosos (nunca executar sem aprovação) ──
DANGEROUS_COMMAND_PATTERNS: list[re.Pattern] = [
    re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
    re.compile(r"\brm\s+-r\b", re.IGNORECASE),
    re.compile(r"\brm\s+--recursive\b", re.IGNORECASE),
    re.compile(r"\bdd\b\s+if=", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bfdisk\b", re.IGNORECASE),
    re.compile(r"\bmkswap\b", re.IGNORECASE),
    re.compile(r"\bformat\b", re.IGNORECASE),
    re.compile(r">\s*/dev/", re.IGNORECASE),
    re.compile(r"\bchmod\s+777\b", re.IGNORECASE),
    re.compile(r"\bchown\b", re.IGNORECASE),
    re.compile(r"sudo\s+", re.IGNORECASE),
    re.compile(r"\bpasswd\b", re.IGNORECASE),
    re.compile(r"\bwget\b.*\||\bcurl\b.*\|", re.IGNORECASE),
    re.compile(r"\bshutdown\b|\breboot\b|\bpoweroff\b|\binit\b", re.IGNORECASE),
    re.compile(r"\bdrop\s+table\b", re.IGNORECASE),
    re.compile(r"\bdrop\s+database\b", re.IGNORECASE),
    re.compile(r"\bDELETE\s+FROM\b", re.IGNORECASE),
    re.compile(r"\bDROP\s+USER\b", re.IGNORECASE),
]

# ── Ficheiros/diretórios críticos que nunca devem ser modificados ──
CRITICAL_PATHS: list[Path] = [
    Path("/etc"),
    Path("/boot"),
    Path("/sys"),
    Path("/proc"),
    Path(os.path.expanduser("~/.ssh")),
    Path(os.path.expanduser("~/..")),
]

# ======================================================================
#  Utilitários de Segurança
# ======================================================================

def classify_command(command: str) -> str:
    """
    Classifica um comando por nível de risco.

    Retorna: "safe", "low", "medium", "high", "critical"
    """
    command_lower = command.lower()

    # Verificar padrões perigosos
    for pattern in DANGEROUS_COMMAND_PATTERNS:
        if pattern.search(command_lower):
            return "critical"

    # Verificar palavras-chave de alto risco
    high_risk_keywords = [
        "delete", "apagar", "remover", "wipe", "limpar",
        "format", "formatar", "mkfs", "fdisk",
        "install", "instalar", "pip install", "npm install -g",
        "chmod", "chown", "sudo", "su ",
        "drop", "truncate", "alter",
    ]
    for keyword in high_risk_keywords:
        if keyword in command_lower:
            return "high"

    # Verificar palavras-chave de risco médio
    medium_risk_keywords = [
        "mv", "move", "mover", "rm ", "remove",
        "cp ", "copy", "copiar",
        "modify", "modificar", "edit", "editar",
        "config", "configure",
        "write", "escrever", "overwrite",
    ]
    for keyword in medium_risk_keywords:
        if keyword in command_lower:
            return "medium"

    # Comandos de leitura/informação
    safe_keywords = ["ls", "cat", "head", "tail", "grep", "find",
                     "pwd", "echo", "which", "whoami", "date",
                     "ler", "mostrar", "listar", "pesquisar"]
    for keyword in safe_keywords:
        if command.startswith(keyword):
            return "safe"

    return "low"


def validate_file_operation(path: str, operation: str) -> tuple[bool, str]:
    """
    Valida se uma operação num ficheiro é segura.

    Retorna: (permitido: bool, mensagem: str)
    """
    try:
        resolved = Path(path).resolve()
    except Exception:
        return False, f"Caminho inválido: {path}"

    # Verificar se o caminho está dentro de uma área crítica
    for critical in CRITICAL_PATHS:
        try:
            resolved.relative_to(critical)
            return False, f"Operação não permitida em área crítica do sistema: {critical}"
        except ValueError:
            pass

    # Verificar extensões críticas
    dangerous_extensions = {".pem", ".key", ".env", ".secret", ".password"}
    if resolved.suffix.lower() in dangerous_extensions:
        risk = classify_command(operation)
        if risk in ("high", "critical"):
            return False, f"Operação não permitida em ficheiros sensíveis: {resolved.name}"

    return True, "Operação permitida"


def requires_approval(action: str) -> tuple[bool, str, str]:
    """
    Determina se uma ação necessita de aprovação do utilizador.

    Retorna: (precisa_aprovacao: bool, risco: str, motivo: str)
    """
    risk = classify_command(action)

    if risk == "critical":
        return True, risk, "Ação crítica — requer confirmação explícita em duas etapas."
    elif risk == "high":
        return True, risk, "Ação de alto risco — requer confirmação do utilizador."
    elif risk == "medium":
        return True, risk, "Ação de risco médio — recomenda-se confirmação."
    else:
        return False, risk, "Ação segura — pode ser executada."


# ======================================================================
#  Gestor de Segurança
# ======================================================================

class SafetyManager:
    """
    Camada de segurança centralizada.

    Todas as ações potencialmente perigosas passam por este gestor,
    que decide se podem ser executadas automaticamente ou se precisam
    de aprovação humana.
    """

    def __init__(self, approvals_path: Optional[str] = None):
        self._pending_approvals: dict[str, dict] = {}
        self._approval_counter = 0
        self._approvals_path = (
            approvals_path
            or os.getenv("ODYSSEUS_V2_APPROVALS_PATH")
            or str(Path("data") / "v2_pending_approvals.json")
        )
        self._load_approvals()

    def _load_approvals(self) -> None:
        """Carrega aprovações persistidas; marca expiradas sem apagar."""
        if not os.path.exists(self._approvals_path):
            return
        try:
            with open(self._approvals_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            approvals = data.get("approvals") or {}
            counter = int(data.get("counter") or 0)
            now = datetime.now(timezone.utc)
            loaded = 0
            for aid, entry in approvals.items():
                if not isinstance(entry, dict):
                    continue
                if entry.get("status") == "pending":
                    expires_at = entry.get("expires_at")
                    if expires_at:
                        try:
                            exp = datetime.fromisoformat(expires_at)
                            if exp.tzinfo is None:
                                exp = exp.replace(tzinfo=timezone.utc)
                            if exp < now:
                                entry["status"] = "expired"
                        except (ValueError, TypeError):
                            pass
                self._pending_approvals[aid] = entry
                loaded += 1
            self._approval_counter = counter
            logger.info(
                "SafetyManager: %d aprovações carregadas de %s (counter=%d)",
                loaded, self._approvals_path, counter,
            )
        except Exception as e:
            logger.warning(
                "SafetyManager: falha a carregar %s: %s. "
                "A continuar em modo memória.",
                self._approvals_path, e,
            )

    def _save_approvals(self) -> None:
        """Persiste aprovações com escrita atómica."""
        try:
            from core.atomic_io import atomic_write_json
            Path(self._approvals_path).parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self._approvals_path, {
                "counter": self._approval_counter,
                "approvals": self._pending_approvals,
            }, indent=2)
        except Exception as e:
            logger.error(
                "SafetyManager: falha a gravar %s: %s",
                self._approvals_path, e,
            )

    @staticmethod
    def _is_not_expired(entry: dict) -> bool:
        expires_at = entry.get("expires_at")
        if not expires_at:
            return True
        try:
            exp = datetime.fromisoformat(expires_at)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            return exp >= datetime.now(timezone.utc)
        except (ValueError, TypeError):
            return True

    def check_action(self, action_type: str, description: str,
                     details: Optional[dict] = None,
                     existing_approval_id: Optional[str] = None,
                     permissions: Optional[list[str]] = None,
                     fingerprint_material: Optional[dict] = None) -> dict:
        """
        Verifica se uma ação pode ser executada.

        Se fornecido um existing_approval_id, verifica se essa aprovação
        já foi concedida — se sim, permite a ação automaticamente.

        Se forem declaradas `permissions`, a ação passa a exigir aprovação
        explícita. Isto impede o caso em que o nome genérico da ação (por
        exemplo "tool_execution") se classifica como risco baixo e a
        ferramenta executa sem qualquer aprovação, apesar de declarar
        permissões como "shell.exec".

        Retorna:
            {
                "allowed": bool,
                "requires_approval": bool,
                "approval_id": str | None,
                "risk": str,
                "message": str,
            }
        """
        # Fingerprint da operação concreta (ferramenta, parâmetros, permissões).
        # Liga a aprovação ao pedido que a originou: sem isto, uma aprovação
        # concedida para um comando podia ser reapresentada para executar outro.
        #
        # `fingerprint_material` permite hashar os parâmetros reais enquanto
        # `details` (que fica visível na listagem de aprovações) é a versão
        # redigida — assim o hash liga a operação sem expor segredos.
        from .execution_policy import operation_fingerprint
        material = fingerprint_material if fingerprint_material is not None else details
        fingerprint = operation_fingerprint(action_type, material, permissions)

        # Se temos uma aprovação existente, verificar o seu estado e a operação
        if existing_approval_id and existing_approval_id in self._pending_approvals:
            existing = self._pending_approvals[existing_approval_id]
            if existing["status"] == "approved":
                if existing.get("fingerprint") == fingerprint:
                    return {
                        "allowed": True,
                        "requires_approval": False,
                        "approval_id": existing_approval_id,
                        "risk": existing["risk"],
                        "message": "Ação aprovada pelo utilizador.",
                    }
                logger.warning(
                    "Aprovação %s reapresentada para uma operação diferente "
                    "(fingerprint divergente) — negada.",
                    existing_approval_id,
                )
                return {
                    "allowed": False,
                    "requires_approval": False,
                    "approval_id": existing_approval_id,
                    "risk": existing["risk"],
                    "message": (
                        "A aprovação apresentada não corresponde a esta operação "
                        "(ferramenta, parâmetros ou permissões diferentes). "
                        "É necessário um novo pedido de aprovação."
                    ),
                }
            elif existing["status"] == "denied":
                return {
                    "allowed": False,
                    "requires_approval": False,
                    "approval_id": existing_approval_id,
                    "risk": existing["risk"],
                    "message": "Ação negada pelo utilizador.",
                }

        needs_approval, risk, reason = requires_approval(action_type)

        # Permissões declaradas => aprovação obrigatória. A classificação por
        # texto não vê a ferramenta nem os parâmetros, pelo que a decisão não
        # pode depender apenas dela.
        if permissions:
            declared = ", ".join(str(p) for p in permissions)
            if not needs_approval:
                needs_approval = True
                if risk in ("safe", "low", "medium"):
                    risk = "high"
                reason = (
                    f"Permissões declaradas ({declared}) — requer aprovação "
                    f"explícita do utilizador."
                )
            else:
                reason = f"{reason} Permissões: {declared}."

        result = {
            "allowed": not needs_approval,
            "requires_approval": needs_approval,
            "approval_id": None,
            "risk": risk,
            "message": reason,
        }

        if needs_approval:
            self._approval_counter += 1
            approval_id = f"APPROVAL-{self._approval_counter:04d}"
            expires_at = (
                datetime.now(timezone.utc) + timedelta(hours=24)
            ).isoformat()
            self._pending_approvals[approval_id] = {
                "action_type": action_type,
                "description": description,
                "details": details or {},
                "risk": risk,
                "status": "pending",
                "source_message": description,
                "fingerprint": fingerprint,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "expires_at": expires_at,
            }
            self._save_approvals()
            result["approval_id"] = approval_id

        return result

    def approve(self, approval_id: str) -> tuple[bool, str]:
        """
        Aprova uma ação pendente.

        Retorna: (sucesso: bool, mensagem: str)
        """
        if approval_id not in self._pending_approvals:
            return False, f"ID de aprovação '{approval_id}' não encontrado."

        pending = self._pending_approvals[approval_id]
        if pending["status"] != "pending":
            return False, f"A aprovação '{approval_id}' já foi {pending['status']}."

        if not self._is_not_expired(pending):
            pending["status"] = "expired"
            pending["resolved_at"] = datetime.now(timezone.utc).isoformat()
            self._save_approvals()
            return False, f"A aprovação '{approval_id}' expirou."

        pending["status"] = "approved"
        pending["resolved_at"] = datetime.now(timezone.utc).isoformat()
        self._save_approvals()
        logger.info("Aprovação concedida: %s — %s", approval_id, pending["description"])
        return True, f"Aprovação '{approval_id}' concedida com sucesso."

    def deny(self, approval_id: str) -> tuple[bool, str]:
        """
        Nega uma ação pendente.

        Retorna: (sucesso: bool, mensagem: str)
        """
        if approval_id not in self._pending_approvals:
            return False, f"ID de aprovação '{approval_id}' não encontrado."

        pending = self._pending_approvals[approval_id]
        if pending["status"] != "pending":
            return False, f"A aprovação '{approval_id}' já foi {pending['status']}."

        if not self._is_not_expired(pending):
            pending["status"] = "expired"
            pending["resolved_at"] = datetime.now(timezone.utc).isoformat()
            self._save_approvals()
            return False, f"A aprovação '{approval_id}' expirou."

        pending["status"] = "denied"
        pending["resolved_at"] = datetime.now(timezone.utc).isoformat()
        self._save_approvals()
        logger.info("Aprovação negada: %s — %s", approval_id, pending["description"])
        return True, f"Aprovação '{approval_id}' negada."

    def get_approval(self, approval_id: str) -> Optional[dict]:
        """Retorna os detalhes de uma aprovação específica."""
        entry = self._pending_approvals.get(approval_id)
        if entry is None:
            return None
        return {**entry, "approval_id": approval_id}

    def get_pending(self) -> list[dict]:
        """Retorna todas as aprovações pendentes."""
        return [
            {**data, "approval_id": aid}
            for aid, data in self._pending_approvals.items()
            if data["status"] == "pending"
            and self._is_not_expired(data)
        ]

    def get_all(self) -> list[dict]:
        """Retorna TODAS as aprovações (para auditoria)."""
        return [
            {**data, "approval_id": aid}
            for aid, data in self._pending_approvals.items()
        ]

    def clear_pending(self) -> None:
        """Limpa todas as aprovações pendentes."""
        self._pending_approvals.clear()
        self._save_approvals()

    def count_pending(self) -> int:
        """Conta aprovações pendentes."""
        return sum(
            1 for approval in self._pending_approvals.values()
            if approval["status"] == "pending" and self._is_not_expired(approval)
        )

    # ------------------------------------------------------------------
    #  Sanitização de entrada
    # ------------------------------------------------------------------

    @staticmethod
    def sanitize_input(text: str, max_length: int = 4000) -> str:
        """
        Sanitiza texto de entrada do utilizador:
        - Limita tamanho
        - Remove caracteres de controlo (exceto quebras de linha)
        """
        if len(text) > max_length:
            text = text[:max_length]
        # Remover caracteres de controlo (manter \n, \r, \t)
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return cleaned.strip()


# Singleton global
_safety_instance: Optional[SafetyManager] = None


def get_safety_manager() -> SafetyManager:
    global _safety_instance
    if _safety_instance is None:
        _safety_instance = SafetyManager()
    return _safety_instance


def reset_safety_manager() -> None:
    """Reinicia o singleton (útil para testes).

    Não apaga o ficheiro persistido — apenas descarta a instância
    em memória. Chamadas subsequentes a `get_safety_manager()`
    criam uma instância nova que recarrega de `_approvals_path`
    (ou do env var `ODYSSEUS_V2_APPROVALS_PATH` se estiver
    definido).
    """
    global _safety_instance
    _safety_instance = None
