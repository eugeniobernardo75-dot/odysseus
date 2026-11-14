"""
Módulo de Memória — Odysseus V2

Gerencia dois tipos de memória:
    1. Temporária (volátil): contexto da conversa atual, descartada ao fim da sessão.
    2. Permanente (persistente): informação explícita que o utilizador autorizou guardar.

Desde a introdução do MemoryBackend (memory_backend.py), a persistência é
delegada a um backend concreto (DictMemoryBackend, ChromaMemoryBackend, etc.)
que o MemoryManager recebe por Dependency Injection.

Se não for fornecido nenhum backend, o MemoryManager usa o DictMemoryBackend
como fallback (idêntico ao comportamento anterior com ficheiro JSON).

Regras:
    - Nada é memorizado permanentemente sem aprovação explícita do utilizador.
    - A memória permanente é armazenada através do backend configurado.
    - Toda a memória pode ser auditada, alterada ou apagada pelo utilizador.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from .memory_backend import (
    MemoryBackend,
    MemoryRecord,
    DictMemoryBackend,
    ChromaMemoryBackend,
)

logger = logging.getLogger(__name__)

# ── Caminho para armazenamento persistente (fallback) ──
DATA_DIR = Path(os.getenv("ODYSSEUS_DATA_DIR", "data"))
MEMORY_FILE = DATA_DIR / "memory_store.json"


# ======================================================================
#  Estruturas de dados
# ======================================================================

@dataclass
class MemoryEntry:
    """Uma unidade de memória permanente."""
    key: str
    value: Any
    category: str = "general"           # "fact", "preference", "skill", "context"
    created_at: str = ""
    updated_at: str = ""
    source: str = "conversation"        # como foi obtida

    def __post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


@dataclass
class ConversationMemory:
    """Memória temporária da conversa atual."""
    messages: list[dict] = field(default_factory=list)
    current_task: Optional[str] = None
    task_stack: list = field(default_factory=list)
    pending_approvals: list = field(default_factory=list)
    agent_results: dict = field(default_factory=dict)
    _data: dict = field(default_factory=dict)  # armazenamento chave-valor arbitrário


# ======================================================================
#  Gestor de Memória
# ======================================================================

MEMORY_LEARNING_CATEGORIES = (
    "preference", "fact", "decision", "style",
    "name", "company", "project", "goal",
)


class MemoryManager:
    """
    Gerencia memória temporária (volátil) e permanente (persistente).

    Uso:
        mm = MemoryManager()
        mm.store_temporary("user_name", "João")
        name = mm.recall_temporary("user_name")

        # Memória permanente requer aprovação
        mm.request_permanent_storage("user_email", "joao@email.com")
        # ... após aprovação do utilizador:
        mm.confirm_permanent_storage("user_email")
    """

    def __init__(self, storage_path: Path = MEMORY_FILE,
                 backend: Optional[MemoryBackend] = None):
        """
        Args:
            storage_path: Caminho para ficheiro JSON (usado apenas se backend=None).
            backend:      Implementação concreta de MemoryBackend.
                          Se None, cria um DictMemoryBackend (compatível com o
                          comportamento anterior baseado em ficheiro JSON).
        """
        self.storage_path = storage_path
        self._lock = Lock()

        # ── Backend de persistência ──
        self._backend: MemoryBackend = backend or DictMemoryBackend()

        # ── Memória temporária (volátil) ──
        self.temp: ConversationMemory = ConversationMemory()

        # ── Memória permanente (persistente) — mantida para retrocompatibilidade ──
        self._permanent: dict[str, MemoryEntry] = {}
        self._pending: dict[str, MemoryEntry] = {}       # aguardando aprovação
        self._load_permanent()

    # ------------------------------------------------------------------
    #  Memória temporária
    # ------------------------------------------------------------------

    def store_temporary(self, key: str, value: Any) -> None:
        """Guarda um valor na memória temporária (apenas para esta sessão)."""
        # Usar dicionário interno para evitar conflitos com campos do dataclass
        self.temp._data[key] = value

    def recall_temporary(self, key: str, default: Any = None) -> Any:
        """Recupera um valor da memória temporária."""
        return self.temp._data.get(key, default)

    def add_message(self, role: str, content: str) -> None:
        """Adiciona uma mensagem ao histórico da conversa."""
        self.temp.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_conversation_context(self, max_messages: int = 20) -> list[dict]:
        """Retorna as últimas N mensagens para contexto."""
        return self.temp.messages[-max_messages:]

    def clear_temporary(self) -> None:
        """Limpa toda a memória temporária (nova sessão)."""
        self.temp = ConversationMemory()

    # ------------------------------------------------------------------
    #  Memória permanente (requer aprovação)
    # ------------------------------------------------------------------

    def request_permanent_storage(self, key: str, value: Any,
                                   category: str = "general",
                                   source: str = "conversation") -> None:
        """
        Solicita armazenamento permanente.
        A memória só é guardada quando o utilizador aprovar com confirm_permanent_storage().
        """
        entry = MemoryEntry(
            key=key,
            value=value,
            category=category,
            source=source,
        )
        self._pending[key] = entry

    def confirm_permanent_storage(self, key: str) -> bool:
        """Confirma o armazenamento permanente de uma entrada pendente."""
        if key not in self._pending:
            logger.warning("Nenhuma entrada pendente com a chave '%s'", key)
            return False
        entry = self._pending.pop(key)
        entry.updated_at = datetime.now(timezone.utc).isoformat()

        # Guarda no backend e mantém compatibilidade com _permanent
        self._permanent[key] = entry
        self._save_permanent()

        # Também persiste no backend se for diferente do dict nativo
        if not isinstance(self._backend, DictMemoryBackend):
            import asyncio
            try:
                record = MemoryRecord(
                    key=key,
                    text=str(entry.value),
                    value=entry.value,
                    category=entry.category,
                    source=entry.source,
                )
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._backend.store(record))
                else:
                    loop.run_until_complete(self._backend.store(record))
            except Exception as e:
                logger.warning("Erro ao guardar no backend externo: %s", e)

        logger.info("Memória permanente guardada: %s = %s", key, entry.value)
        return True

    def deny_permanent_storage(self, key: str) -> bool:
        """Recusa o armazenamento permanente e remove a entrada pendente."""
        if key not in self._pending:
            return False
        del self._pending[key]
        return True

    def get_pending_approvals(self) -> list[dict]:
        """Retorna lista de entradas pendentes de aprovação."""
        return [asdict(e) for e in self._pending.values()]

    def remember(self, key: str, value: Any, category: str = "fact",
                 source: str = "learning") -> bool:
        """Guarda diretamente na memória permanente (aprendizagem).

        O UserAgent usa este método para aprender preferências, factos,
        decisões, estilo, nomes, empresas, projetos e objetivos — sem
        passar pela fila de aprovação (o utilizador autorizou a
        aprendizagem permanente do agente pessoal).
        """
        if category not in MEMORY_LEARNING_CATEGORIES:
            category = "fact"
        existing = self._permanent.get(key)
        if existing is not None:
            existing.value = value
            existing.category = category
            existing.updated_at = datetime.now(timezone.utc).isoformat()
        else:
            self._permanent[key] = MemoryEntry(
                key=key, value=value, category=category, source=source,
            )
        self._save_permanent()
        if not isinstance(self._backend, DictMemoryBackend):
            import asyncio
            try:
                record = MemoryRecord(
                    key=key, text=str(value), value=value,
                    category=category, source=source,
                )
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._backend.store(record))
                else:
                    loop.run_until_complete(self._backend.store(record))
            except Exception as e:
                logger.warning("Erro ao guardar no backend externo: %s", e)
        logger.info("Memória aprendida [%s]: %s = %s", category, key, value)
        return True

    def recall_permanent(self, key: str) -> Optional[Any]:
        """Recupera um valor da memória permanente.

        Tenta primeiro o backend externo; se não encontrar, usa o dict interno.
        """
        # Tenta backend externo primeiro
        if not isinstance(self._backend, DictMemoryBackend):
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    fut = asyncio.run_coroutine_threadsafe(
                        self._backend.recall(key), loop
                    )
                    record = fut.result(timeout=5.0)
                else:
                    record = loop.run_until_complete(self._backend.recall(key))
                if record is not None:
                    return record.value if record.value is not None else record.text
            except Exception as e:
                logger.warning("Erro ao ler do backend externo: %s", e)

        # Fallback: dict interno
        entry = self._permanent.get(key)
        return entry.value if entry else None

    def recall_all_permanent(self) -> dict[str, Any]:
        """Retorna toda a memória permanente."""
        return {k: v.value for k, v in self._permanent.items()}

    def recall_profile(self) -> dict:
        """Perfil do utilizador: memória permanente agrupada por categoria
        (preferences, facts, decisions, style, names, companies, projects,
        goals). Usado pelo UserAgent para responder com contexto pessoal."""
        profile: dict = {}
        for key, entry in self._permanent.items():
            profile.setdefault(entry.category, {})[key] = entry.value
        return profile

    def update_permanent(self, key: str, value: Any,
                         category: Optional[str] = None) -> bool:
        """
        Atualiza um valor existente na memória permanente.
        Retorna False se a chave não existir.
        """
        if key not in self._permanent:
            return False
        entry = self._permanent[key]
        entry.value = value
        entry.updated_at = datetime.now(timezone.utc).isoformat()
        if category:
            entry.category = category
        self._save_permanent()

        # Sincroniza com backend externo
        if not isinstance(self._backend, DictMemoryBackend):
            import asyncio
            try:
                record = MemoryRecord(
                    key=key,
                    text=str(value),
                    value=value,
                    category=category or entry.category,
                    source=entry.source,
                )
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._backend.store(record))
                else:
                    loop.run_until_complete(self._backend.store(record))
            except Exception as e:
                logger.warning("Erro ao sincronizar com backend externo: %s", e)

        return True

    def delete_permanent(self, key: str) -> bool:
        """Apaga uma entrada da memória permanente."""
        if key not in self._permanent:
            return False
        del self._permanent[key]
        self._save_permanent()

        # Sincroniza com backend externo
        if not isinstance(self._backend, DictMemoryBackend):
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._backend.delete(key))
                else:
                    loop.run_until_complete(self._backend.delete(key))
            except Exception as e:
                logger.warning("Erro ao apagar no backend externo: %s", e)

        return True

    def clear_all_permanent(self) -> None:
        """Apaga TODA a memória permanente (requer confirmação do utilizador)."""
        self._permanent.clear()
        self._pending.clear()
        self._save_permanent()

    # ------------------------------------------------------------------
    #  Propriedade pública do backend
    # ------------------------------------------------------------------

    @property
    def backend(self) -> MemoryBackend:
        """Retorna o backend de memória atualmente configurado."""
        return self._backend

    # ------------------------------------------------------------------
    #  Persistência
    # ------------------------------------------------------------------

    def _load_permanent(self) -> None:
        """Carrega memória permanente do disco (só para retrocompatibilidade)."""
        try:
            if self.storage_path.exists():
                with self._lock, open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for key, entry_data in data.items():
                    self._permanent[key] = MemoryEntry(**entry_data)
                logger.info("Memória permanente carregada: %d entradas (ficheiro JSON)", len(self._permanent))
            else:
                logger.info("Ficheiro de memória permanente não encontrado.")
                self.storage_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_permanent()
        except Exception as e:
            logger.error("Erro ao carregar memória permanente: %s", e)

    def _save_permanent(self) -> None:
        """Guarda memória permanente no disco (só para retrocompatibilidade).

        Quando um backend externo está configurado, este método continua a
        guardar no ficheiro JSON como cache local, mas a fonte de verdade
        é o backend.
        """
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {k: asdict(v) for k, v in self._permanent.items()}
            with self._lock, open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Erro ao guardar memória permanente: %s", e)

    # ------------------------------------------------------------------
    #  Auditoria
    # ------------------------------------------------------------------

    def audit_log(self) -> list[dict]:
        """Retorna um relatório completo da memória para auditoria."""
        return [
            {
                "key": k,
                "value": v.value,
                "category": v.category,
                "created_at": v.created_at,
                "updated_at": v.updated_at,
                "source": v.source,
            }
            for k, v in self._permanent.items()
        ]


# ------------------------------------------------------------------
#  Singleton (para uso global na aplicação)
# ------------------------------------------------------------------

_memory_instance: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Retorna a instância global do gestor de memória."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = MemoryManager()
    return _memory_instance
