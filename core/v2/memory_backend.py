"""
Módulo de Backend de Memória — Odysseus V2

Fornece uma interface abstrata (MemoryBackend) para armazenamento persistente,
libertando o MemoryManager do conhecimento sobre onde/como os dados são guardados.

Implementações incluídas:
    - DictMemoryBackend  : em memória (volátil, para testes / fallback)
    - ChromaMemoryBackend: adaptador para o ChromaDB do Odysseus principal

Arquitetura:
    MemoryManager
        │  depende de uma interface, não de uma implementação
        ▼
    MemoryBackend (ABC)
        │
        ├── DictMemoryBackend   (dict nativo, volátil)
        └── ChromaMemoryBackend (ChromaDB via singleton do Odysseus)
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ======================================================================
#  Estruturas de dados neutras
# ======================================================================

@dataclass
class MemoryRecord:
    """Registo de memória independente do backend de armazenamento."""
    id: str = ""
    key: str = ""
    text: str = ""
    value: Any = None
    category: str = "fact"
    source: str = "user"
    owner: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = int(time.time())


# ======================================================================
#  Interface abstrata
# ======================================================================

class MemoryBackend(ABC):
    """Interface que qualquer backend de memória persistente deve implementar.

    O MemoryManager do V2 depende desta interface, não de uma implementação
    concreta. Assim, podes trocar de backend (ChromaDB, PostgreSQL, Redis, ...)
    sem alterar o núcleo do Orchestrator.
    """

    @abstractmethod
    async def store(self, record: MemoryRecord) -> str:
        """Guarda um registo de memória. Retorna o ID do registo."""
        ...

    @abstractmethod
    async def recall(self, key: str) -> Optional[MemoryRecord]:
        """Recupera um registo pela chave (key). Retorna None se não existir."""
        ...

    @abstractmethod
    async def search(self, query: str, top_k: int = 5,
                     category: Optional[str] = None) -> list[MemoryRecord]:
        """Pesquisa memórias por semelhança semântica ao texto 'query'."""
        ...

    @abstractmethod
    async def list_all(self, category: Optional[str] = None,
                       limit: int = 100) -> list[MemoryRecord]:
        """Lista todos os registos, opcionalmente filtrados por categoria."""
        ...

    @abstractmethod
    async def delete(self, record_id: str) -> bool:
        """Apaga um registo pelo ID. Retorna False se não existir."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Número total de registos armazenados."""
        ...

    @property
    @abstractmethod
    def healthy(self) -> bool:
        """Indica se o backend está operacional.

        Deve ser seguro chamar esta propriedade sem inicialização prévia.
        """
        ...

    @abstractmethod
    async def health_check(self) -> dict:
        """Retorna diagnóstico de saúde do backend."""
        ...


# ======================================================================
#  Implementação 1: DictMemoryBackend (em memória)
# ======================================================================

class DictMemoryBackend(MemoryBackend):
    """Backend de memória em dicionário nativo (volátil).

    Útil para:
        - Testes unitários (rápido, sem dependências externas)
        - Fallback quando o ChromaDB não está disponível
        - Desenvolvimento local sem ChromaDB

    Nota: os dados são voláteis — perdidos ao reiniciar o processo.
    """

    def __init__(self):
        self._store: dict[str, MemoryRecord] = {}
        self._by_key: dict[str, str] = {}          # key → id
        self._healthy = True

    # ── Implementação MemoryBackend ──────────────────────────────

    async def store(self, record: MemoryRecord) -> str:
        record.id = record.id or str(uuid.uuid4())
        record.timestamp = record.timestamp or int(time.time())
        self._store[record.id] = record

        if record.key:
            self._by_key[record.key] = record.id

        logger.debug("DictMemoryBackend.store: %s (%s)", record.id, record.key)
        return record.id

    async def recall(self, key: str) -> Optional[MemoryRecord]:
        # Tenta key primeiro, depois id como fallback
        record_id = self._by_key.get(key)
        if record_id:
            return self._store.get(record_id)
        return self._store.get(key)

    async def search(self, query: str, top_k: int = 5,
                     category: Optional[str] = None) -> list[MemoryRecord]:
        # Pesquisa textual simples (case-insensitive)
        query_lower = query.lower()
        results: list[tuple[float, MemoryRecord]] = []

        for record in self._store.values():
            if category and record.category != category:
                continue

            text = (record.text or str(record.value or "")).lower()
            if query_lower in text:
                # Score: proporção de palavras da query que aparecem no texto
                words = set(query_lower.split())
                if words:
                    text_words = set(text.split())
                    score = len(words & text_words) / len(words)
                else:
                    score = 1.0
                results.append((score, record))

        results.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in results[:top_k]]

    async def list_all(self, category: Optional[str] = None,
                       limit: int = 100) -> list[MemoryRecord]:
        records = list(self._store.values())
        if category:
            records = [r for r in records if r.category == category]
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records[:limit]

    async def delete(self, record_id: str) -> bool:
        # Tenta key primeiro, depois id
        stored_id = self._by_key.pop(record_id, None)
        if stored_id:
            record = self._store.pop(stored_id, None)
            if record is not None:
                logger.debug("DictMemoryBackend.delete (by key): %s -> %s", record_id, stored_id)
                return True

        record = self._store.pop(record_id, None)
        if record is None:
            return False

        if record.key:
            self._by_key.pop(record.key, None)
        logger.debug("DictMemoryBackend.delete (by id): %s", record_id)
        return True

    async def count(self) -> int:
        return len(self._store)

    @property
    def healthy(self) -> bool:
        return self._healthy

    async def health_check(self) -> dict:
        return {
            "healthy": self._healthy,
            "type": "dict",
            "entries": len(self._store),
        }


# ======================================================================
#  Implementação 2: ChromaMemoryBackend
# ======================================================================

class ChromaMemoryBackend(MemoryBackend):
    """Backend de memória que utiliza o ChromaDB do Odysseus principal.

    Este adaptador usa os singletons globais `_memory_manager` e
    `_memory_vector` do `src.ai_interaction` para armazenar e pesquisar
    memórias. O V2 nunca depende diretamente do ChromaDB — apenas através
    desta interface.

    Se o ChromaDB não estiver disponível, faz fallback para o JSON file
    do MemoryManager nativo do Odysseus.
    """

    def __init__(self, memory_manager=None, memory_vector=None):
        """
        Args:
            memory_manager: Instância de src.memory.MemoryManager (ou None,
                            tenta obter via src.ai_interaction._memory_manager)
            memory_vector:  Instância de src.memory_vector.MemoryVectorStore
                            (ou None, tenta obter via src.ai_interaction)
        """
        self._memory_manager = memory_manager
        self._memory_vector = memory_vector
        self._resolved = False
        self._healthy = True
        self._fallback: Optional[DictMemoryBackend] = None

    # ── Inicialização lazy (resolve singletons do Odysseus) ──────

    def _resolve_dependencies(self) -> None:
        """Tenta resolver os singletons do Odysseus se não foram injetados."""
        if self._resolved:
            return
        self._resolved = True

        if self._memory_manager is None:
            try:
                from src.ai_interaction import _memory_manager as mm
                if mm is not None:
                    self._memory_manager = mm
                    logger.info("ChromaMemoryBackend: resolved _memory_manager")
            except ImportError:
                logger.warning("ChromaMemoryBackend: src.ai_interaction not available")

        if self._memory_vector is None:
            try:
                from src.ai_interaction import _memory_vector as mv
                if mv is not None:
                    self._memory_vector = mv
                    logger.info("ChromaMemoryBackend: resolved _memory_vector")
            except ImportError:
                logger.warning("ChromaMemoryBackend: src.memory_vector not available")

        if self._memory_manager is None:
            logger.warning("ChromaMemoryBackend: no MemoryManager available, "
                           "using fallback DictMemoryBackend")
            self._fallback = DictMemoryBackend()
            self._healthy = False

    # ── Conversão de formatos ────────────────────────────────────

    def _odysseus_to_record(self, entry: dict) -> MemoryRecord:
        """Converte uma entrada do MemoryManager do Odysseus para MemoryRecord."""
        return MemoryRecord(
            id=entry.get("id", ""),
            key=entry.get("key", entry.get("id", "")),
            text=entry.get("text", ""),
            value=entry.get("value", entry.get("text", "")),
            category=entry.get("category", "fact"),
            source=entry.get("source", "user"),
            owner=entry.get("owner"),
            session_id=entry.get("session_id"),
            timestamp=entry.get("timestamp", 0),
            metadata=entry.get("metadata", {}),
        )

    def _record_to_odysseus(self, record: MemoryRecord) -> dict:
        """Converte MemoryRecord para dicionário usado pelo Odysseus MemoryManager."""
        return {
            "id": record.id,
            "text": record.text or str(record.value or ""),
            "timestamp": record.timestamp or int(time.time()),
            "source": record.source,
            "category": record.category,
            "owner": record.owner,
            "session_id": record.session_id,
            "uses": 0,
            "metadata": dict(record.metadata),
        }

    # ── Implementação MemoryBackend ──────────────────────────────

    async def store(self, record: MemoryRecord) -> str:
        self._resolve_dependencies()

        if self._fallback:
            return await self._fallback.store(record)

        if self._memory_manager is None:
            raise RuntimeError("ChromaMemoryBackend: no MemoryManager available")

        # Guarda no MemoryManager (JSON file)
        entry = self._memory_manager.add_entry(
            record.text or str(record.value or ""),
            source=record.source,
            category=record.category,
            owner=record.owner,
        )
        entry["id"] = record.id or entry["id"]
        if record.session_id:
            entry["session_id"] = record.session_id

        memories = self._memory_manager.load_all()
        # Verifica duplicados pela key
        entry["key"] = record.key
        if record.key:
            existing = [m for m in memories if m.get("key") == record.key]
            if existing:
                existing[0].update(entry)
                self._memory_manager.save(memories)
                return existing[0]["id"]

        memories.append(entry)
        self._memory_manager.save(memories)

        # Guarda também no vector store (ChromaDB) se disponível
        if self._vector_available():
            try:
                self._memory_vector.add(entry["id"], entry["text"])
            except Exception as e:
                logger.warning("ChromaMemoryBackend: vector add failed: %s", e)

        logger.debug("ChromaMemoryBackend.store: %s", entry["id"])
        return entry["id"]

    async def recall(self, key: str) -> Optional[MemoryRecord]:
        self._resolve_dependencies()

        if self._fallback:
            return await self._fallback.recall(key)

        if self._memory_manager is None:
            return None

        # Procura por key ou id no JSON file
        memories = self._memory_manager.load_all()
        for m in memories:
            if m.get("key") == key or m.get("id") == key:
                return self._odysseus_to_record(m)

        # Se não encontrou, tenta recall permanente (para compatibilidade)
        if hasattr(self._memory_manager, "recall_permanent"):
            try:
                value = self._memory_manager.recall_permanent(key)
                if value is not None:
                    return MemoryRecord(key=key, text=str(value), value=value)
            except Exception:
                pass

        return None

    async def search(self, query: str, top_k: int = 5,
                     category: Optional[str] = None) -> list[MemoryRecord]:
        self._resolve_dependencies()

        if self._fallback:
            return await self._fallback.search(query, top_k, category)

        if self._memory_manager is None:
            return []

        memories = self._memory_manager.load_all()

        # Se o vector store está disponível, usa pesquisa semântica
        if self._vector_available():
            try:
                results = self._memory_vector.search(query, k=top_k)
                if results:
                    hits = []
                    for r in results:
                        memory_id = r.get("memory_id") if isinstance(r, dict) else None
                        if not memory_id:
                            continue
                        for m in memories:
                            if m.get("id") == memory_id:
                                if category and m.get("category") != category:
                                    continue
                                hits.append(self._odysseus_to_record(m))
                                break
                        if len(hits) >= top_k:
                            break
                    if hits:
                        return hits
            except Exception as e:
                logger.warning("ChromaMemoryBackend: vector search failed: %s", e)

        # Fallback: pesquisa textual
        query_lower = query.lower()
        results = []
        for m in memories:
            if category and m.get("category") != category:
                continue
            text = (m.get("text", "") or "").lower()
            if query_lower in text or any(
                word in text for word in query_lower.split()
            ):
                results.append(self._odysseus_to_record(m))

        # Ordena por relevância (proporção de palavras correspondentes)
        def _score(rec: MemoryRecord) -> float:
            words = set(query_lower.split())
            text_words = set((rec.text or "").lower().split())
            if not words:
                return 0.0
            return len(words & text_words) / len(words)

        results.sort(key=_score, reverse=True)
        return results[:top_k]

    async def list_all(self, category: Optional[str] = None,
                       limit: int = 100) -> list[MemoryRecord]:
        self._resolve_dependencies()

        if self._fallback:
            return await self._fallback.list_all(category, limit)

        if self._memory_manager is None:
            return []

        memories = self._memory_manager.load_all()
        if category:
            memories = [m for m in memories if m.get("category") == category]
        memories.sort(key=lambda m: m.get("timestamp", 0), reverse=True)
        return [self._odysseus_to_record(m) for m in memories[:limit]]

    async def delete(self, record_id: str) -> bool:
        self._resolve_dependencies()

        if self._fallback:
            return await self._fallback.delete(record_id)

        if self._memory_manager is None:
            return False

        memories = self._memory_manager.load_all()
        original_len = len(memories)
        found = []

        for m in memories:
            if m.get("id") == record_id or m.get("key") == record_id:
                found.append(m)

        if not found:
            return False

        # Remove do vector store
        if self._vector_available():
            for m in found:
                try:
                    self._memory_vector.remove(m["id"])
                except Exception as e:
                    logger.warning("ChromaMemoryBackend: vector remove failed: %s", e)

        # Remove do JSON file
        memories = [m for m in memories if m.get("id") not in {f.get("id") for f in found}]
        self._memory_manager.save(memories)

        logger.debug("ChromaMemoryBackend.delete: %s (removed %d)", record_id, len(found))
        return len(memories) < original_len

    async def count(self) -> int:
        self._resolve_dependencies()

        if self._fallback:
            return await self._fallback.count()

        if self._memory_manager is None:
            return 0

        return len(self._memory_manager.load_all())

    @property
    def healthy(self) -> bool:
        self._resolve_dependencies()
        return self._healthy

    async def health_check(self) -> dict:
        self._resolve_dependencies()

        if self._fallback:
            fb_health = await self._fallback.health_check()
            # Nota: não usar **fb_health porque sobrescreveria "healthy": False
            return {
                "healthy": False,
                "type": "chroma_fallback",
                "fallback_healthy": fb_health.get("healthy", True),
                "entries": fb_health.get("entries", 0),
            }

        return {
            "healthy": self._healthy,
            "type": "chroma",
            "memory_manager": self._memory_manager is not None,
            "memory_vector": self._memory_vector is not None,
            "vector_available": self._vector_available(),
            "entries": len(self._memory_manager.load_all()) if self._memory_manager else 0,
        }

    def _vector_available(self) -> bool:
        return bool(
            self._memory_vector
            and getattr(self._memory_vector, "healthy", True)
        )
