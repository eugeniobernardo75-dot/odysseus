"""
Cache Semântico — Odysseus V2

Cache de respostas do LLM baseado em embeddings semânticos com similaridade
de cosseno. Evita chamadas repetidas ao LLM para perguntas semanticamente
equivalentes, poupando tempo e recursos.

Arquitetura:
    1. Quando uma pergunta chega, o seu embedding é calculado via Ollama
       (POST /api/embeddings) usando o mesmo modelo configurado no LLMClient.
    2. O embedding é comparado (cosine similarity) com todos os embeddings
       em cache.
    3. Se a similaridade máxima > threshold, a resposta em cache é devolvida.
    4. Se não, a pergunta é enviada ao LLM e a resposta é guardada em cache.

Características:
    - Sem dependências externas (cosine similarity em Python puro)
    - Embeddings via Ollama API nativa (POST /api/embeddings)
    - LRU eviction quando o cache atinge o tamanho máximo
    - TTL (time-to-live) por entrada
    - Persistência em ficheiro JSON (opcional)
    - Per-model isolation (cache separado por modelo/base_url)
    - Threshold configurável
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ======================================================================
#  Configuração do Cache
# ======================================================================

@dataclass
class CacheConfig:
    """Configuração do cache semântico."""
    enabled: bool = True
    threshold: float = 0.85          # Similaridade mínima para considerar hit (0.0 — 1.0)
    max_entries: int = 500           # Número máximo de entradas (LRU eviction)
    ttl_seconds: float = 3600.0      # Tempo de vida de cada entrada (1h)
    persist_path: str = ""           # Caminho para ficheiro de persistência (vazio = sem persistência)
    model_fingerprint: str = ""      # Identificador do modelo (muda quando model/base_url muda)

    @property
    def cache_key(self) -> str:
        """Chave única para o cache atual (baseada no modelo + server)."""
        return self.model_fingerprint or "default"


# ======================================================================
#  Entrada do Cache
# ======================================================================

@dataclass
class CacheEntry:
    """Uma entrada no cache semântico."""
    query: str
    response: str
    embedding: list[float]
    model: str
    created_at: float
    accessed_at: float
    access_count: int = 1
    ttl: float = 3600.0

    @property
    def is_expired(self) -> bool:
        """Verifica se a entrada expirou."""
        return (time.time() - self.created_at) > self.ttl

    def to_dict(self) -> dict:
        """Serializa para dict (para persistência)."""
        return {
            "query": self.query,
            "response": self.response,
            "embedding": self.embedding[:8],  # Só amostra (não guarda o embedding completo)
            "model": self.model,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "ttl": self.ttl,
        }


# ======================================================================
#  Cache Semântico
# ======================================================================

class SemanticCache:
    """
    Cache de respostas LLM baseado em similaridade semântica.

    Guarda perguntas anteriores e suas respostas como embeddings.
    Quando uma nova pergunta chega, calcula o embedding e compara
    com todos os embeddings em cache usando cosine similarity.
    Se a similaridade for superior ao threshold, devolve a resposta
    em cache em vez de chamar o LLM.

    Uso:
        cache = SemanticCache(llm_client)
        cache.configure(threshold=0.88)

        # Verificar cache
        entry = await cache.lookup("Qual é a capital de França?")
        if entry:
            return entry.response

        # Chamar LLM e guardar
        response = await llm_client.chat(messages)
        await cache.store("Qual é a capital de França?", response.content, model)
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        config: Optional[CacheConfig] = None,
    ):
        self._llm_client = llm_client
        self.config = config or CacheConfig()
        self._entries: list[CacheEntry] = []       # Cache em memória
        self._lock: Any = None                      # asyncio.Lock (lazy)
        self._stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
            "expirations": 0,
        }

        # Carregar cache persistido se existir
        if self.config.persist_path and os.path.exists(self.config.persist_path):
            self._load_persisted()

    # ------------------------------------------------------------------
    #  Lock
    # ------------------------------------------------------------------

    async def _get_lock(self):
        if self._lock is None:
            import asyncio
            self._lock = asyncio.Lock()
        return self._lock

    # ------------------------------------------------------------------
    #  Configuração
    # ------------------------------------------------------------------

    def set_llm_client(self, client: Any) -> None:
        """Define o cliente LLM (necessário para gerar embeddings)."""
        self._llm_client = client

    def configure(self, **kwargs) -> dict:
        """
        Atualiza a configuração do cache em tempo real.

        Args:
            enabled: bool — Ativar/desativar cache
            threshold: float — Similaridade mínima (0.0 — 1.0)
            max_entries: int — Máximo de entradas
            ttl_seconds: float — Tempo de vida (segundos)
            persist_path: str — Caminho para persistência
        """
        changed = []
        valid_fields = {"enabled", "threshold", "max_entries", "ttl_seconds", "persist_path"}

        for key, value in kwargs.items():
            if key not in valid_fields:
                continue

            current = getattr(self.config, key, None)

            if key == "enabled":
                value = bool(value)
            elif key == "threshold":
                value = max(0.0, min(1.0, float(value)))
            elif key == "max_entries":
                value = max(1, int(value))
            elif key == "ttl_seconds":
                value = max(1.0, float(value))
            elif key == "persist_path":
                value = str(value) if value else ""

            if current == value:
                continue

            setattr(self.config, key, value)
            changed.append(key)

        if changed:
            logger.info("Cache config atualizada: %s", ", ".join(changed))

        return {
            "success": True,
            "changed": changed,
            "config": {
                "enabled": self.config.enabled,
                "threshold": self.config.threshold,
                "max_entries": self.config.max_entries,
                "ttl_seconds": self.config.ttl_seconds,
                "persist_path": self.config.persist_path,
                "model_fingerprint": self.config.model_fingerprint,
                "entries": len(self._entries),
                "stats": dict(self._stats),
            },
        }

    def invalidate(self) -> int:
        """
        Invalida todo o cache (útil quando o modelo/base_url muda).

        Returns:
            Número de entradas removidas.
        """
        count = len(self._entries)
        self._entries.clear()
        self._stats["evictions"] += count
        logger.info("Cache invalidado: %d entradas removidas.", count)
        return count

    # ------------------------------------------------------------------
    #  Embeddings via Ollama
    # ------------------------------------------------------------------

    async def _compute_embedding(self, text: str) -> Optional[list[float]]:
        """
        Calcula o embedding de um texto usando o LLM (Ollama API nativa).

        Chama POST /api/embeddings no servidor Ollama com o modelo
        configurado. Se o cliente LLM não estiver disponível ou a
        chamada falhar, retorna None e o cache é desativado para esta query.

        Args:
            text: Texto a嵌入 (embedding).

        Returns:
            Lista de floats (embedding) ou None em caso de erro.
        """
        from .llm_client import useragent_ollama_request_active
        if useragent_ollama_request_active():
            logger.info(
                "Embedding do cache semântico omitido durante um pedido do "
                "UserAgent; evitar uma chamada Ollama implícita."
            )
            return None

        if not self._llm_client:
            return None

        try:
            client = await self._llm_client._get_client()

            model = self._llm_client.config.model

            payload = {
                "model": model,
                "prompt": text,
            }

            logger.debug("A calcular embedding para: %.60s... (modelo: %s)", text, model)

            resp = await client.post("/api/embeddings", json=payload, timeout=10.0)

            if resp.status_code != 200:
                logger.debug("Embedding API retornou status %d", resp.status_code)
                return None

            data = resp.json()
            embedding = data.get("embedding")

            if not embedding or not isinstance(embedding, list):
                logger.debug("Resposta de embedding inválida: %s", type(embedding))
                return None

            return embedding

        except Exception as e:
            logger.debug("Erro ao calcular embedding: %s", e)
            return None

    # ------------------------------------------------------------------
    #  Cosine Similarity (Python puro, sem dependências)
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """
        Calcula a similaridade de cosseno entre dois vetores.

        Fórmula: cos(θ) = (A · B) / (||A|| * ||B||)

        Args:
            a: Primeiro vetor.
            b: Segundo vetor.

        Returns:
            Similaridade entre 0.0 e 1.0. 0.0 se os vetores forem inválidos.
        """
        if not a or not b:
            return 0.0

        # Verificar se têm o mesmo tamanho (podem ser de modelos diferentes)
        if len(a) != len(b):
            # Truncar ao menor tamanho
            min_len = min(len(a), len(b))
            a = a[:min_len]
            b = b[:min_len]

        dot_product = 0.0
        norm_a = 0.0
        norm_b = 0.0

        for i in range(len(a)):
            dot_product += a[i] * b[i]
            norm_a += a[i] * a[i]
            norm_b += b[i] * b[i]

        norm_a = math.sqrt(norm_a)
        norm_b = math.sqrt(norm_b)

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    # ------------------------------------------------------------------
    #  Lookup
    # ------------------------------------------------------------------

    async def lookup(
        self, query: str,
    ) -> tuple[Optional[CacheEntry], Optional[list[float]]]:
        """
        Procura uma resposta em cache para uma query.

        Calcula o embedding da query e compara com todas as entradas
        em cache usando cosine similarity. Devolve a entrada com
        maior similaridade se esta for >= threshold.

        Devolve TAMBÉM o embedding calculado para que o caller
        o possa reutilizar em store() — evitando computar o embedding
        duas vezes em caso de cache MISS.

        Args:
            query: Texto da pergunta.

        Returns:
            Tuplo (CacheEntry | None, embedding | None).
            - Se cache HIT: (CacheEntry, None) — embedding não é necessário
            - Se cache MISS: (None, embedding) — embedding pode ser passado a store()
            - Se erro: (None, None)
        """
        if not self.config.enabled or not query:
            self._stats["misses"] += 1
            return (None, None)

        if not self._entries:
            self._stats["misses"] += 1
            return (None, None)

        # Calcular embedding da query
        query_embedding = await self._compute_embedding(query)
        if query_embedding is None:
            # Embedding indisponível — cache desativado para esta query
            self._stats["misses"] += 1
            return (None, None)

        best_entry = None
        best_similarity = 0.0

        async with await self._get_lock():
            # Limpar entradas expiradas primeiro
            now = time.time()
            active_entries = []
            for entry in self._entries:
                if (now - entry.created_at) > entry.ttl:
                    self._stats["expirations"] += 1
                    continue
                active_entries.append(entry)
            self._entries = active_entries

            # Procurar entrada mais similar
            for entry in self._entries:
                similarity = self._cosine_similarity(query_embedding, entry.embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = entry

            # Verificar threshold
            if best_entry and best_similarity >= self.config.threshold:
                # Atualizar metadata
                best_entry.accessed_at = now
                best_entry.access_count += 1
                self._stats["hits"] += 1
                logger.info(
                    "Cache HIT: similaridade=%.4f (threshold=%.2f) | query: %.60s",
                    best_similarity, self.config.threshold, query,
                )
                return (best_entry, None)  # embedding descartado no HIT

        self._stats["misses"] += 1
        logger.debug(
            "Cache MISS: melhor similaridade=%.4f (threshold=%.2f) | query: %.60s",
            best_similarity, self.config.threshold, query,
        )
        return (None, query_embedding)  # embedding devolvido para store() reutilizar

    # ------------------------------------------------------------------
    #  Store
    # ------------------------------------------------------------------

    async def store(
        self, query: str, response: str, model: str = "",
        embedding: Optional[list[float]] = None,
    ) -> bool:
        """
        Guarda uma query e a sua resposta em cache.

        Se o embedding já tiver sido calculado (ex: por lookup()),
        podes passá-lo como argumento para evitar computá-lo novamente.

        Args:
            query: Pergunta original.
            response: Resposta do LLM.
            model: Modelo usado (para fingerprint).
            embedding: Embedding pré-calculado (opcional, evita duplicação).

        Returns:
            True se guardado com sucesso, False caso contrário.
        """
        if not self.config.enabled or not query or not response:
            return False

        # Reutilizar embedding se fornecido, ou calcular
        if embedding is not None:
            query_embedding = embedding
        else:
            query_embedding = await self._compute_embedding(query)
        if query_embedding is None:
            return False

        entry = CacheEntry(
            query=query,
            response=response,
            embedding=query_embedding,
            model=model or "",
            created_at=time.time(),
            accessed_at=time.time(),
            ttl=self.config.ttl_seconds,
        )

        async with await self._get_lock():
            # LRU eviction: se excedeu o limite, remover as menos acedidas
            while len(self._entries) >= self.config.max_entries:
                # Remover a entrada com menor access_count (LRU aproximado)
                oldest = min(self._entries, key=lambda e: (e.access_count, e.accessed_at))
                self._entries.remove(oldest)
                self._stats["evictions"] += 1
                logger.debug("Cache LRU eviction: %.60s (access_count=%d)", oldest.query, oldest.access_count)

            self._entries.append(entry)
            self._stats["stores"] += 1

        # Persistir se configurado (fire-and-forget)
        if self.config.persist_path:
            self._persist_async()

        logger.debug("Cache STORE: %d entradas | query: %.60s", len(self._entries), query)
        return True

    # ------------------------------------------------------------------
    #  Estatísticas
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Retorna estatísticas detalhadas do cache."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0

        return {
            "enabled": self.config.enabled,
            "threshold": self.config.threshold,
            "max_entries": self.config.max_entries,
            "ttl_seconds": self.config.ttl_seconds,
            "entries": len(self._entries),
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "stores": self._stats["stores"],
            "evictions": self._stats["evictions"],
            "expirations": self._stats["expirations"],
            "total_requests": total,
            "hit_rate": round(hit_rate, 4),
            "model_fingerprint": self.config.model_fingerprint,
        }

    def get_entries_snapshot(self, limit: int = 20) -> list[dict]:
        """Retorna uma amostra das entradas em cache (para debug/UI)."""
        snapshot = []
        for entry in sorted(self._entries, key=lambda e: e.accessed_at, reverse=True)[:limit]:
            snapshot.append({
                "query": entry.query[:120],
                "response_preview": entry.response[:200],
                "model": entry.model,
                "access_count": entry.access_count,
                "created_ago_seconds": round(time.time() - entry.created_at, 1),
                "expired": entry.is_expired,
            })
        return snapshot

    # ------------------------------------------------------------------
    #  Clear
    # ------------------------------------------------------------------

    def clear(self) -> int:
        """Remove todas as entradas do cache."""
        count = len(self._entries)
        self._entries.clear()
        self._stats = {k: 0 for k in self._stats}
        logger.info("Cache limpo: %d entradas removidas.", count)
        return count

    # ------------------------------------------------------------------
    #  Persistência
    # ------------------------------------------------------------------

    def _persist_async(self) -> None:
        """Guarda o cache em ficheiro (fire-and-forget, síncrono curto)."""
        if not self.config.persist_path:
            return
        try:
            data = {
                "version": 1,
                "config": {
                    "threshold": self.config.threshold,
                    "max_entries": self.config.max_entries,
                    "ttl_seconds": self.config.ttl_seconds,
                },
                "entries": [e.to_dict() for e in self._entries[-50:]],  # Só últimas 50
            }
            os.makedirs(os.path.dirname(self.config.persist_path) or ".", exist_ok=True)
            with open(self.config.persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug("Falha ao persistir cache: %s", e)

    def _load_persisted(self) -> None:
        """Carrega cache de ficheiro persistido."""
        try:
            with open(self.config.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            entries = data.get("entries", [])
            if not entries:
                return

            # Nota: embeddings não são persistidos (apenas amostra),
            # por isso o cache precisa de ser repovoado. Mostramos
            # apenas as queries em log.
            query_samples = [e.get("query", "")[:60] for e in entries[:5]]
            logger.info(
                "Cache persistido carregado: %d entradas (apenas metadados — "
                "embeddings incompletos). Amostra: %s",
                len(entries), ", ".join(query_samples),
            )
        except Exception as e:
            logger.debug("Falha ao carregar cache persistido: %s", e)


# ======================================================================
#  Singleton
# ======================================================================

_cache_instance: Optional[SemanticCache] = None


def get_cache(llm_client: Optional[Any] = None) -> SemanticCache:
    """Retorna a instância global do cache semântico."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache(llm_client=llm_client)
    else:
        if llm_client is not None:
            _cache_instance.set_llm_client(llm_client)
    return _cache_instance


def reset_cache() -> None:
    """Reinicia o cache (útil para testes)."""
    global _cache_instance
    _cache_instance = None
