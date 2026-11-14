"""
Teste de integração — Orchestrator V2 ⇄ ChromaDB

Liga o OdysseusOrchestrator (V2) a um backend de memória persistente
baseado em ChromaDB e verifica o fluxo completo:

    conversa → memória → recall

Dois modos (fixture parametrizada ``chroma_stack``):

- ``real_chromadb``: ChromaDB real in-process (``chromadb.EphemeralClient``)
  com um embedder determinístico baseado em hashing de tokens — não precisa
  de servidor nem de downloads de modelos.
- ``mock``: ``ChromaMemoryBackend`` com um vector store falso em memória
  (mesma interface de ``MemoryVectorStore``) — sem dependência do chromadb.

O backend (``ChromaMemoryBackend``) continua a usar o ``MemoryManager`` real
do Odysseus (``src.memory``) para persistência JSON em ambos os modos;
apenas a camada vetorial (ChromaDB) é real ou simulada.

Uso:
    pytest tests/v2/test_orchestrator_chromadb_integration.py -v
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ======================================================================
#  Embedder determinístico (sem downloads / sem rede)
# ======================================================================

def _hash_embedder(dimension: int = 128):
    """Cria um embedder determinístico baseado em hashing de tokens.

    Textos com tokens em comum produzem vetores com similaridade de
    cosseno > 0, o que permite testar a pesquisa semântica do ChromaDB
    sem depender de um modelo real (FastEmbed/HTTP).
    """
    class HashEmbedder:
        model = "test-hash"
        url = "test://hash"

        def __init__(self, dim: int):
            # Atributo de instância: o corpo de uma classe aninhada NÃO
            # captura variáveis do escopo da função envolvente.
            self.dimension = dim

        def get_sentence_embedding_dimension(self) -> int:
            return self.dimension

        def encode(self, texts, normalize_embeddings: bool = True):
            vecs = []
            for text in texts:
                vec = [0.0] * self.dimension
                for token in re.findall(r"\w+", str(text).lower()):
                    idx = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.dimension
                    vec[idx] += 1.0
                norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                if normalize_embeddings:
                    vec = [x / norm for x in vec]
                vecs.append(vec)
            return vecs

    return HashEmbedder(dimension)


# ======================================================================
#  Vector store falso (modo mock)
# ======================================================================

class FakeVectorStore:
    """Substituto de ``MemoryVectorStore`` em memória (mesma interface).

    Interface usada pelo ``ChromaMemoryBackend``:
    ``healthy``, ``add(id, text)``, ``remove(id)``, ``count()`` e
    ``search(query, k) -> [{"memory_id": ..., "score": ...}]``.
    """

    def __init__(self):
        self._rows: dict[str, str] = {}
        self._embedder = _hash_embedder()

    @property
    def healthy(self) -> bool:
        return True

    def add(self, memory_id: str, text: str) -> None:
        self._rows[memory_id] = text

    def remove(self, memory_id: str) -> None:
        self._rows.pop(memory_id, None)

    def count(self) -> int:
        return len(self._rows)

    def search(self, query: str, k: int = 8):
        qvec = self._embedder.encode([query])[0]
        scored = []
        for mid, text in self._rows.items():
            tvec = self._embedder.encode([text])[0]
            sim = sum(a * b for a, b in zip(qvec, tvec))
            if sim > 0:
                scored.append({"memory_id": mid, "score": round(sim, 4)})
        scored.sort(key=lambda r: -r["score"])
        return scored[:k]


# ======================================================================
#  Helpers de espera (para o store assíncrono disparado pelo confirm)
# ======================================================================

async def _wait_until_async(factory, timeout: float = 5.0, interval: float = 0.05) -> bool:
    """Espera até ``await factory()`` devolver verdade (com timeout)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await factory():
            return True
        await asyncio.sleep(interval)
    return False


async def _wait_until_sync(pred, timeout: float = 5.0, interval: float = 0.05) -> bool:
    """Espera até ``pred()`` (síncrono) devolver verdade (com timeout)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        await asyncio.sleep(interval)
    return False


async def _backend_count_at_least(backend, n: int) -> bool:
    """True quando o backend tiver pelo menos ``n`` registos."""
    return await backend.count() >= n


# ======================================================================
#  Cliente ChromaDB real (EphemeralClient ou servidor HTTP)
# ======================================================================

def _port_open(host: str, port: int, timeout: float = 1.5) -> bool:
    """True se for possível abrir uma conexão TCP a host:port."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _make_real_chroma_client():
    """Devolve um cliente ChromaDB real utilizável em testes, ou None.

    Ordem de tentativas:
        1. ``chromadb.EphemeralClient()`` — ChromaDB in-process (pacote
           completo instalado).
        2. ``chromadb.HttpClient`` para um servidor real (thin client
           ``chromadb-client`` ou servidor dedicado) — o endpoint pode ser
           configurado via ``CHROMADB_HOST`` / ``CHROMADB_PORT``.

    Retorna None quando nenhuma opção está disponível (o teste é
    então pulado com um motivo claro).
    """
    chromadb = pytest.importorskip("chromadb")

    try:
        client = chromadb.EphemeralClient()
        # Validar que o cliente de facto aceita criação de coleções.
        # Nome único — evita corridas entre processos (ex: pytest-xdist).
        import uuid as _uuid
        probe = client.get_or_create_collection(f"__probe_{_uuid.uuid4().hex[:8]}__")
        client.delete_collection(probe.name)
        return client
    except Exception:
        pass

    host = os.getenv("CHROMADB_HOST", "localhost")
    port = int(os.getenv("CHROMADB_PORT", "8100"))
    if not _port_open(host, port):
        return None

    try:
        client = chromadb.HttpClient(host=host, port=port)
        client.heartbeat()
        return client
    except Exception:
        return None


# ======================================================================
#  Fixture principal — stack de integração
# ======================================================================

@pytest.fixture(params=["real_chromadb", "mock"])
def chroma_stack(request, tmp_path, monkeypatch):
    """Constrói a stack completa: ChromaMemoryBackend + MemoryManager + Orchestrator.

    Retorna um dicionário com:
        mode          : "real_chromadb" | "mock"
        orchestrator  : OdysseusOrchestrator (V2)
        memory        : core.v2.memory.MemoryManager (com o backend Chroma)
        backend       : ChromaMemoryBackend
        vector_store  : MemoryVectorStore real (ou FakeVectorStore)
        storage_path  : Path do ficheiro JSON persistente do V2
        src_memory    : src.memory.MemoryManager (JSON do Odysseus)
    """
    mode = request.param
    data_dir = tmp_path / "memory_data"
    data_dir.mkdir()

    from src.memory import MemoryManager as SrcMemoryManager
    src_mm = SrcMemoryManager(data_dir=str(data_dir))

    client = None
    unique_collection = None

    if mode == "real_chromadb":
        client = _make_real_chroma_client()
        if client is None:
            pytest.skip(
                "ChromaDB real indisponível: sem chromadb completo "
                "(EphemeralClient) e sem servidor acessível em "
                "CHROMADB_HOST:CHROMADB_PORT"
            )

        from src.embedding_lanes import LANE_CUSTOM, LANE_FASTEMBED
        import uuid as _uuid

        # Nome de coleção único por teste — não polui dados reais do servidor.
        unique_collection = f"odysseus_memories_test_{_uuid.uuid4().hex[:8]}"
        from src.memory_vector import MemoryVectorStore
        monkeypatch.setattr(MemoryVectorStore, "COLLECTION_NAME", unique_collection)

        embedder = _hash_embedder()

        def _fake_fastembed():
            return embedder

        def _no_custom():
            raise RuntimeError("custom embedding lane disabled in tests")

        # Apontar o singleton ChromaDB e as lanes de embedding para o
        # cliente real + embedder determinístico.
        monkeypatch.setattr("src.chroma_client.get_chroma_client", lambda: client)
        monkeypatch.setattr("src.embedding_lanes._build_fastembed_client", _fake_fastembed)
        monkeypatch.setattr("src.embedding_lanes._build_custom_client", _no_custom)

        vector_store = MemoryVectorStore(data_dir=str(data_dir))
        assert vector_store.healthy, "MemoryVectorStore não ficou saudável com o ChromaDB real"

        def _teardown():
            """Remove as coleções criadas pelo teste (servidor real)."""
            if client is None or unique_collection is None:
                return
            for suffix in ("", f"_{LANE_CUSTOM}", f"_{LANE_FASTEMBED}"):
                try:
                    client.delete_collection(unique_collection + suffix)
                except Exception:
                    pass

        request.addfinalizer(_teardown)
    else:
        vector_store = FakeVectorStore()

    from core.v2.memory_backend import ChromaMemoryBackend
    backend = ChromaMemoryBackend(memory_manager=src_mm, memory_vector=vector_store)

    from core.v2.memory import MemoryManager
    storage_path = tmp_path / "memory_store.json"
    memory = MemoryManager(storage_path=storage_path, backend=backend)

    from core.v2.orchestrator import OdysseusOrchestrator
    orchestrator = OdysseusOrchestrator(memory=memory, llm_client=None)

    return {
        "mode": mode,
        "orchestrator": orchestrator,
        "memory": memory,
        "backend": backend,
        "vector_store": vector_store,
        "storage_path": storage_path,
        "src_memory": src_mm,
    }


# ======================================================================
#  Teste principal — fluxo completo: conversa → memória → recall
# ======================================================================

class TestFluxoCompleto:
    """Verifica o fluxo de memória do UserAgent com backend ChromaDB."""

    @pytest.mark.asyncio
    async def test_conversa_memoria_recall(self, chroma_stack, monkeypatch):
        orch = chroma_stack["orchestrator"]
        memory = chroma_stack["memory"]
        backend = chroma_stack["backend"]
        vector_store = chroma_stack["vector_store"]
        storage_path = chroma_stack["storage_path"]

        from core.v2 import llm_client, orchestrator as orchestrator_module
        from core.v2 import user_agent_config

        monkeypatch.setattr(
            orchestrator_module, "get_orchestrator", lambda **_kwargs: orch
        )
        monkeypatch.setattr(llm_client, "get_llm_client", lambda: object())
        monkeypatch.setattr(
            user_agent_config, "_get_memory_manager", lambda: memory
        )
        agent = user_agent_config.UserAgent()
        agent._retrieve_relevant_knowledge = lambda _query: []
        agent._retrieve_relevant_memory = lambda _query: ""

        # ── 1. CONVERSA ─────────────────────────────────────────────
        # O UserAgent propõe o armazenamento, sujeito a aprovação humana.
        resp = await agent.process("guarda o meu nome é João")
        assert "Posso guardar" in resp.message

        pending = memory.get_pending_approvals()
        assert any(e["key"] == "nome" and "joão" in str(e["value"]).lower() for e in pending)

        # ── 2. MEMÓRIA ──────────────────────────────────────────────
        # Aprovação → confirm_permanent_storage persiste no backend
        # (JSON + vector store / ChromaDB).
        assert memory.confirm_permanent_storage("nome") is True

        # O confirm dispara um store assíncrono no backend — esperar.
        ok = await _wait_until_async(lambda: _backend_count_at_least(backend, 1))
        assert ok, "backend não recebeu o registo de memória"
        assert await backend.count() == 1

        # O registo está recuperável pelo backend (recall por key).
        record = await backend.recall("nome")
        assert record is not None
        assert "joão" in str(record.value).lower()

        # O vector store também recebeu o texto (índice semântico).
        ok = await _wait_until_sync(lambda: vector_store.count() >= 1)
        assert ok, "vector store não recebeu o texto da memória"

        # ── 3. RECALL — mesma sessão ────────────────────────────────
        # A leitura da memória é tratada pelo UserAgent, não pelo Orchestrator.
        resp_recall = await agent.process("lembra-te do meu nome")
        assert "joão" in resp_recall.message.lower()

        # ── 4. RECALL — nova sessão (novo orquestrador, mesmo backend) ──
        # Simula o reinício da aplicação: um MemoryManager + Orchestrator
        # novos partilham o mesmo armazenamento persistente.
        from core.v2.memory import MemoryManager as MemoryManager2
        memory2 = MemoryManager2(storage_path=storage_path, backend=backend)

        from core.v2.orchestrator import OdysseusOrchestrator as Orch2
        orch2 = Orch2(memory=memory2, llm_client=None)

        monkeypatch.setattr(
            orchestrator_module, "get_orchestrator", lambda **_kwargs: orch2
        )
        monkeypatch.setattr(
            user_agent_config, "_get_memory_manager", lambda: memory2
        )
        agent2 = user_agent_config.UserAgent()
        agent2._retrieve_relevant_knowledge = lambda _query: []
        resp2 = await agent2.process("lembra-te do meu nome")
        assert "joão" in resp2.message.lower()

        # Recall direto pelo backend na "nova sessão".
        record2 = await backend.recall("nome")
        assert record2 is not None
        assert "joão" in str(record2.value).lower()

        # ── 5. RECALL — pesquisa semântica (vector store) ───────────
        # Guarda uma memória com texto rico e pesquisa por semelhança.
        from core.v2.memory_backend import MemoryRecord
        await backend.store(MemoryRecord(
            key="interesse",
            text="O utilizador gosta de programar em Python",
            category="preference",
        ))

        hits = await backend.search("gosta de programar em Python")
        assert any(h.key == "interesse" for h in hits)

    @pytest.mark.asyncio
    async def test_backend_health_and_search(self, chroma_stack):
        """Health check e pesquisa semântica direta no backend."""
        backend = chroma_stack["backend"]
        vector_store = chroma_stack["vector_store"]

        health = await backend.health_check()
        assert health["healthy"] is True

        # Sem registos, a pesquisa devolve vazio (sem erros).
        assert await backend.search("nada para encontrar") == []

        # Com registos, a pesquisa semântica devolve correspondências.
        from core.v2.memory_backend import MemoryRecord
        await backend.store(MemoryRecord(
            key="projeto",
            text="O utilizador está a trabalhar no projeto Odysseus",
            category="fact",
        ))
        hits = await backend.search("projeto Odysseus")
        assert any(h.key == "projeto" for h in hits)

        # list_all / delete também funcionam no fluxo persistente.
        all_records = await backend.list_all()
        assert any(r.key == "projeto" for r in all_records)
        assert await backend.delete("projeto") is True
        assert all(r.key != "projeto" for r in await backend.list_all())


# ======================================================================
#  Teste do API recall_permanent (contexto síncrono)
# ======================================================================

class TestRecallPermanentAPI:
    """recall_permanent do MemoryManager V2 em contexto sem loop a correr."""

    def test_request_confirm_recall(self, chroma_stack):
        memory = chroma_stack["memory"]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            memory.request_permanent_storage("cidade", "Lisboa")
            assert memory.confirm_permanent_storage("cidade") is True
            assert memory.recall_permanent("cidade") == "Lisboa"
        finally:
            loop.close()

    def test_recall_permanent_after_reload(self, chroma_stack):
        """Uma nova instância de MemoryManager recupera a memória persistida."""
        memory = chroma_stack["memory"]
        storage_path = chroma_stack["storage_path"]
        backend = chroma_stack["backend"]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            memory.request_permanent_storage("idioma", "Português")
            assert memory.confirm_permanent_storage("idioma") is True
            assert memory.recall_permanent("idioma") == "Português"

            # Nova instância partilhando o mesmo backend + ficheiro JSON.
            from core.v2.memory import MemoryManager
            memory2 = MemoryManager(storage_path=storage_path, backend=backend)
            assert memory2.recall_permanent("idioma") == "Português"
        finally:
            loop.close()
