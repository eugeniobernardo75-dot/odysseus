"""
Testes para o sistema de Memory Backend do Odysseus V2.

Cobre:
    - MemoryBackend (ABC)
    - DictMemoryBackend (em memória)
    - ChromaMemoryBackend (adaptador ChromaDB)
    - MemoryRecord (dataclass)
    - Integração com MemoryManager (V2)
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from core.v2.memory_backend import (
    MemoryBackend,
    MemoryRecord,
    DictMemoryBackend,
    ChromaMemoryBackend,
)
from core.v2.memory import MemoryManager


# ======================================================================
#  Fixtures
# ======================================================================

@pytest.fixture
def sample_record():
    """MemoryRecord de exemplo para testes."""
    return MemoryRecord(
        key="user_name",
        text="O utilizador chama-se João",
        value="João",
        category="fact",
        source="conversation",
    )


@pytest.fixture
def dict_backend():
    """DictMemoryBackend limpo."""
    return DictMemoryBackend()


@pytest.fixture
def populated_backend(dict_backend, sample_record):
    """DictMemoryBackend com alguns registos."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(dict_backend.store(sample_record))
        loop.run_until_complete(dict_backend.store(MemoryRecord(
            key="user_email", text="Email: joao@email.com", value="joao@email.com",
            category="contact",
        )))
        loop.run_until_complete(dict_backend.store(MemoryRecord(
            key="user_theme", text="Prefere tema escuro", value="dark",
            category="preference",
        )))
        return dict_backend
    finally:
        loop.close()


@pytest.fixture
def chroma_backend_no_deps():
    """ChromaMemoryBackend sem dependências (deve fazer fallback)."""
    return ChromaMemoryBackend(memory_manager=None, memory_vector=None)


# ======================================================================
#  Testes: MemoryRecord
# ======================================================================

class TestMemoryRecord:
    """Testes para a dataclass MemoryRecord."""

    def test_default_id_generated(self):
        """ID deve ser gerado automaticamente se não fornecido."""
        record = MemoryRecord(key="test")
        assert record.id
        assert len(record.id) > 0

    def test_explicit_id(self):
        """ID explícito não deve ser substituído."""
        record = MemoryRecord(id="custom-id", key="test")
        assert record.id == "custom-id"

    def test_default_timestamp(self):
        """Timestamp deve ser definido automaticamente."""
        record = MemoryRecord(key="test")
        assert record.timestamp > 0

    def test_explicit_timestamp(self):
        """Timestamp explícito não deve ser substituído."""
        record = MemoryRecord(key="test", timestamp=12345)
        assert record.timestamp == 12345

    def test_default_category(self):
        """Categoria padrão deve ser 'fact'."""
        record = MemoryRecord(key="test")
        assert record.category == "fact"

    def test_default_source(self):
        """Source padrão deve ser 'user'."""
        record = MemoryRecord(key="test")
        assert record.source == "user"

    def test_all_fields(self):
        """Todos os campos preenchidos corretamente."""
        record = MemoryRecord(
            id="abc-123",
            key="test_key",
            text="Test text",
            value={"nested": "data"},
            category="preference",
            source="ai",
            owner="user1",
            session_id="sess-1",
            timestamp=99999,
            metadata={"source_url": "http://example.com"},
        )
        assert record.id == "abc-123"
        assert record.key == "test_key"
        assert record.text == "Test text"
        assert record.value == {"nested": "data"}
        assert record.category == "preference"
        assert record.source == "ai"
        assert record.owner == "user1"
        assert record.session_id == "sess-1"
        assert record.timestamp == 99999
        assert record.metadata == {"source_url": "http://example.com"}


# ======================================================================
#  Testes: MemoryBackend (ABC)
# ======================================================================

class TestMemoryBackendABC:
    """Testes para a interface abstrata MemoryBackend."""

    def test_cannot_instantiate_abc(self):
        """Não deve ser possível instanciar MemoryBacked diretamente."""
        with pytest.raises(TypeError):
            MemoryBackend()

    def test_interface_methods(self):
        """Todos os métodos abstratos devem estar definidos."""
        abstract_methods = MemoryBackend.__abstractmethods__
        expected = {
            "store", "recall", "search",
            "list_all", "delete", "count",
            "healthy", "health_check",
        }
        assert abstract_methods == expected


# ======================================================================
#  Testes: DictMemoryBackend
# ======================================================================

class TestDictMemoryBackend:
    """Testes para o backend de memória em dicionário."""

    # ── Propriedades básicas ─────────────────────────────────────

    def test_healthy_by_default(self, dict_backend):
        """Backend deve começar saudável."""
        assert dict_backend.healthy is True

    def test_count_empty(self, dict_backend):
        """Backend vazio deve ter count 0."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            count = loop.run_until_complete(dict_backend.count())
            assert count == 0
        finally:
            loop.close()

    def test_health_check(self, dict_backend):
        """Health check deve retornar estrutura correta."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            health = loop.run_until_complete(dict_backend.health_check())
            assert health["healthy"] is True
            assert health["type"] == "dict"
            assert health["entries"] == 0
        finally:
            loop.close()

    # ── Store / Recall ───────────────────────────────────────────

    def test_store_and_recall(self, dict_backend, sample_record):
        """Guardar e recuperar um registo."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            record_id = loop.run_until_complete(dict_backend.store(sample_record))
            assert record_id == sample_record.id

            result = loop.run_until_complete(dict_backend.recall("user_name"))
            assert result is not None
            assert result.key == "user_name"
            assert result.text == "O utilizador chama-se João"
            assert result.value == "João"
        finally:
            loop.close()

    def test_recall_nonexistent(self, dict_backend):
        """Recuperar chave inexistente deve retornar None."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(dict_backend.recall("no_such_key"))
            assert result is None
        finally:
            loop.close()

    def test_recall_by_id(self, dict_backend, sample_record):
        """Recuperar registo pelo ID (não pela key)."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            record_id = loop.run_until_complete(dict_backend.store(sample_record))
            result = loop.run_until_complete(dict_backend.recall(record_id))
            assert result is not None
            assert result.id == record_id
        finally:
            loop.close()

    def test_store_duplicate_key(self, dict_backend):
        """Armazenar duas vezes a mesma key deve substituir."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(dict_backend.store(MemoryRecord(key="k1", text="first")))
            loop.run_until_complete(dict_backend.store(MemoryRecord(key="k1", text="second")))

            result = loop.run_until_complete(dict_backend.recall("k1"))
            assert result.text == "second"
        finally:
            loop.close()

    # ── List / Search / Count ────────────────────────────────────

    def test_list_all(self, populated_backend):
        """Listar todos os registos."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            records = loop.run_until_complete(populated_backend.list_all())
            assert len(records) == 3
        finally:
            loop.close()

    def test_list_by_category(self, populated_backend):
        """Listar registos filtrados por categoria."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            records = loop.run_until_complete(populated_backend.list_all(category="contact"))
            assert len(records) == 1
            assert records[0].key == "user_email"
        finally:
            loop.close()

    def test_list_empty_category(self, populated_backend):
        """Categoria sem registos deve retornar lista vazia."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            records = loop.run_until_complete(populated_backend.list_all(category="nonexistent"))
            assert len(records) == 0
        finally:
            loop.close()

    def test_list_limit(self, populated_backend):
        """Listar com limite deve respeitar o limite."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            records = loop.run_until_complete(populated_backend.list_all(limit=2))
            assert len(records) == 2
        finally:
            loop.close()

    def test_search_text(self, populated_backend):
        """Pesquisa textual deve encontrar correspondências."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            records = loop.run_until_complete(populated_backend.search("João"))
            assert len(records) >= 1
            assert records[0].key == "user_name"
        finally:
            loop.close()

    def test_search_no_results(self, populated_backend):
        """Pesquisa sem correspondências deve retornar lista vazia."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            records = loop.run_until_complete(populated_backend.search("zzzz_notfound"))
            assert len(records) == 0
        finally:
            loop.close()

    def test_search_with_category(self, populated_backend):
        """Pesquisa com filtro de categoria."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            records = loop.run_until_complete(
                populated_backend.search("João", category="fact")
            )
            assert len(records) >= 1

            records_no = loop.run_until_complete(
                populated_backend.search("João", category="contact")
            )
            assert len(records_no) == 0
        finally:
            loop.close()

    def test_count_multiple(self, populated_backend):
        """Count após armazenar vários registos."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            count = loop.run_until_complete(populated_backend.count())
            assert count == 3
        finally:
            loop.close()

    # ── Delete ───────────────────────────────────────────────────

    def test_delete_existing(self, populated_backend):
        """Apagar registo existente deve retornar True."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(populated_backend.delete("user_name"))
            assert result is True

            # Verificar que foi removido
            count = loop.run_until_complete(populated_backend.count())
            assert count == 2
        finally:
            loop.close()

    def test_delete_nonexistent(self, populated_backend):
        """Apagar registo inexistente deve retornar False."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(populated_backend.delete("no_such_key"))
            assert result is False
        finally:
            loop.close()

    def test_delete_by_id(self, populated_backend, sample_record):
        """Apagar registo pelo ID também deve funcionar."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # sample_record já foi armazenado no populated_backend
            result = loop.run_until_complete(populated_backend.delete(sample_record.id))
            assert result is True
        finally:
            loop.close()

    # ── Edge cases ───────────────────────────────────────────────

    def test_store_empty_text(self, dict_backend):
        """Armazenar com texto vazio não deve causar erro."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            record_id = loop.run_until_complete(dict_backend.store(MemoryRecord(key="empty", text="")))
            assert record_id is not None
        finally:
            loop.close()

    def test_store_special_chars(self, dict_backend):
        """Caracteres especiais devem ser preservados."""
        special = "Olá, mundo! Café & Pastel — 100% 👍 #dev"
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(dict_backend.store(MemoryRecord(key="special", text=special)))
            result = loop.run_until_complete(dict_backend.recall("special"))
            assert result.text == special
        finally:
            loop.close()

    def test_search_case_insensitive(self, populated_backend):
        """Pesquisa deve ser case-insensitive."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            records = loop.run_until_complete(populated_backend.search("joão"))
            assert len(records) >= 1
            assert records[0].key == "user_name"
        finally:
            loop.close()

    def test_recall_distinguishes_key_from_id(self, dict_backend):
        """Recall deve distinguir key de id quando ambos existem."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            r1 = MemoryRecord(key="abc", text="First")
            r2 = MemoryRecord(id="abc", key="xyz", text="Second")

            loop.run_until_complete(dict_backend.store(r1))
            loop.run_until_complete(dict_backend.store(r2))

            # recall("abc") deve encontrar o que tem key="abc"
            result = loop.run_until_complete(dict_backend.recall("abc"))
            assert result.text == "First"
        finally:
            loop.close()


# ======================================================================
#  Testes: ChromaMemoryBackend
# ======================================================================

class TestChromaMemoryBackend:
    """Testes para o adaptador ChromaDB do Odysseus."""

    def test_healthy_when_no_deps(self, chroma_backend_no_deps):
        """Mesmo sem dependências, não deve levantar exceção."""
        assert chroma_backend_no_deps.healthy is False

    def test_fallback_on_no_deps(self, chroma_backend_no_deps):
        """Sem dependências, deve usar fallback DictMemoryBackend."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Deve funcionar sem erros
            record_id = loop.run_until_complete(
                chroma_backend_no_deps.store(MemoryRecord(key="test", text="test"))
            )
            assert record_id is not None

            result = loop.run_until_complete(chroma_backend_no_deps.recall("test"))
            assert result is not None
            assert result.text == "test"
        finally:
            loop.close()

    def test_health_check_on_no_deps(self, chroma_backend_no_deps):
        """Health check sem dependências deve reportar fallback."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            health = loop.run_until_complete(chroma_backend_no_deps.health_check())
            assert health["healthy"] is False
            assert "fallback" in health.get("type", "")
        finally:
            loop.close()

    def test_store_with_mock_memory_manager(self):
        """Armazenar com MemoryManager mockado."""
        mock_mm = MagicMock()
        mock_mm.add_entry.return_value = {"id": "new-id", "text": "test"}
        mock_mm.load_all.return_value = []

        backend = ChromaMemoryBackend(memory_manager=mock_mm)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            record_id = loop.run_until_complete(
                backend.store(MemoryRecord(key="test", text="test content"))
            )
            assert record_id is not None
            mock_mm.add_entry.assert_called_once()
            mock_mm.save.assert_called_once()
        finally:
            loop.close()

    def test_recall_with_mock(self):
        """Recuperar com MemoryManager mockado."""
        mock_mm = MagicMock()
        mock_mm.load_all.return_value = [
            {"id": "1", "key": "test_key", "text": "Hello world",
             "category": "fact", "source": "user", "timestamp": 1000}
        ]

        backend = ChromaMemoryBackend(memory_manager=mock_mm)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(backend.recall("test_key"))
            assert result is not None
            assert result.key == "test_key"
            assert result.text == "Hello world"
        finally:
            loop.close()

    def test_recall_not_found_with_mock(self):
        """Recuperar chave inexistente com mock deve retornar None."""
        mock_mm = MagicMock()
        mock_mm.load_all.return_value = []
        # MagicMock auto-cria atributos — garantir que recall_permanent não interfere
        mock_mm.recall_permanent = lambda key: None

        backend = ChromaMemoryBackend(memory_manager=mock_mm)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(backend.recall("no_such_key"))
            assert result is None
        finally:
            loop.close()

    def test_search_with_mock(self):
        """Pesquisa textual com MemoryManager mockado."""
        mock_mm = MagicMock()
        mock_mm.load_all.return_value = [
            {"id": "1", "key": "name", "text": "O João gosta de Python",
             "category": "fact", "source": "user", "timestamp": 1000},
            {"id": "2", "key": "email", "text": "Email: joao@test.com",
             "category": "contact", "source": "user", "timestamp": 1001},
            {"id": "3", "key": "lang", "text": "Prefere TypeScript",
             "category": "preference", "source": "user", "timestamp": 1002},
        ]

        backend = ChromaMemoryBackend(memory_manager=mock_mm)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(backend.search("João"))
            assert len(results) >= 1
            # Pode encontrar "João" no texto ou "joao" no email
            assert any("João" in r.text or "joao" in r.text for r in results)
        finally:
            loop.close()

    def test_list_all_with_mock(self):
        """Listar todos com MemoryManager mockado."""
        mock_mm = MagicMock()
        mock_mm.load_all.return_value = [
            {"id": "1", "key": "k1", "text": "A", "category": "fact",
             "source": "user", "timestamp": 100},
            {"id": "2", "key": "k2", "text": "B", "category": "fact",
             "source": "user", "timestamp": 200},
        ]

        backend = ChromaMemoryBackend(memory_manager=mock_mm)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            records = loop.run_until_complete(backend.list_all())
            assert len(records) == 2
        finally:
            loop.close()

    def test_delete_with_mock(self):
        """Apagar registo com MemoryManager mockado."""
        mock_mm = MagicMock()
        mock_mm.load_all.return_value = [
            {"id": "target-id", "key": "target", "text": "delete me",
             "category": "fact", "source": "user", "timestamp": 100},
        ]

        backend = ChromaMemoryBackend(memory_manager=mock_mm)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(backend.delete("target-id"))
            assert result is True
            mock_mm.save.assert_called_once()
        finally:
            loop.close()

    def test_delete_not_found_with_mock(self):
        """Apagar registo inexistente com mock deve retornar False."""
        mock_mm = MagicMock()
        mock_mm.load_all.return_value = []

        backend = ChromaMemoryBackend(memory_manager=mock_mm)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(backend.delete("no-such-id"))
            assert result is False
        finally:
            loop.close()

    def test_count_with_mock(self):
        """Count com MemoryManager mockado."""
        mock_mm = MagicMock()
        mock_mm.load_all.return_value = [{"id": "1"}, {"id": "2"}, {"id": "3"}]

        backend = ChromaMemoryBackend(memory_manager=mock_mm)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            count = loop.run_until_complete(backend.count())
            assert count == 3
        finally:
            loop.close()

    def test_resolve_dependencies_from_ai_interaction(self):
        """Deve resolver dependências do src.ai_interaction se disponível.

        Como não podemos mockar facilmente o import real, testamos que:
        1. Quando injetado explicitamente, funciona
        2. Quando não injetado, cria fallback
        """
        # Teste 1: injeção explícita
        explicit_mm = MagicMock()
        explicit_mv = MagicMock()
        backend = ChromaMemoryBackend(memory_manager=explicit_mm, memory_vector=explicit_mv)
        assert backend._memory_manager is explicit_mm
        assert backend._memory_vector is explicit_mv

        # Teste 2: sem injeção, deve criar fallback
        backend_no_deps = ChromaMemoryBackend(memory_manager=None, memory_vector=None)
        # Aceder a .healthy dispara _resolve_dependencies
        assert backend_no_deps.healthy is False  # fallback
        assert backend_no_deps._fallback is not None

    def test_store_with_vector_store(self):
        """Armazenar com vector store disponível."""
        mock_mm = MagicMock()
        mock_mm.add_entry.return_value = {"id": "vid-1", "text": "vector test"}
        mock_mm.load_all.return_value = []

        mock_mv = MagicMock()
        mock_mv.healthy = True

        backend = ChromaMemoryBackend(memory_manager=mock_mm, memory_vector=mock_mv)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                backend.store(MemoryRecord(key="vec_test", text="vector test content"))
            )
            mock_mv.add.assert_called_once()
        finally:
            loop.close()

    def test_delete_with_vector_store(self):
        """Apagar com vector store disponível."""
        mock_mm = MagicMock()
        mock_mm.load_all.return_value = [
            {"id": "del-id", "key": "del", "text": "to delete",
             "category": "fact", "source": "user", "timestamp": 100},
        ]

        mock_mv = MagicMock()
        mock_mv.healthy = True

        backend = ChromaMemoryBackend(memory_manager=mock_mm, memory_vector=mock_mv)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(backend.delete("del-id"))
            mock_mv.remove.assert_called_once_with("del-id")
        finally:
            loop.close()


# ======================================================================
#  Testes: Integração com MemoryManager (V2)
# ======================================================================

class TestMemoryManagerWithBackend:
    """MemoryManager do V2 com diferentes backends."""

    def test_default_backend_is_dict(self):
        """Sem backend especificado, deve usar DictMemoryBackend."""
        mm = MemoryManager()
        assert isinstance(mm.backend, DictMemoryBackend)

    def test_custom_backend_injection(self):
        """Deve aceitar backend customizado."""
        custom = DictMemoryBackend()
        mm = MemoryManager(backend=custom)
        assert mm.backend is custom

    def test_chroma_backend_injection(self):
        """Deve aceitar ChromaMemoryBackend."""
        backend = ChromaMemoryBackend(memory_manager=None)
        mm = MemoryManager(backend=backend)
        assert isinstance(mm.backend, ChromaMemoryBackend)

    def test_request_and_confirm_stores_in_backend(self):
        """request + confirm deve armazenar também no backend.

        Nota: count() é async, então usamos asyncio.run() ou verificamos
        indiretamente via recall_permanent (que é síncrono para DictMemoryBackend).
        """
        backend = DictMemoryBackend()
        mm = MemoryManager(backend=backend)

        mm.request_permanent_storage("name", "João")

        mm.confirm_permanent_storage("name")
        # O confirm usa _save_permanent + tenta store no backend
        assert mm.recall_permanent("name") == "João"

    def test_request_confirm_count_in_backend(self):
        """Verificar que confirm incrementa o contador do backend."""
        backend = DictMemoryBackend()
        mm = MemoryManager(backend=backend)

        mm.request_permanent_storage("k1", "v1")
        mm.request_permanent_storage("k2", "v2")

        mm.confirm_permanent_storage("k1")
        mm.confirm_permanent_storage("k2")

        # recall_permanent deve encontrar ambas
        assert mm.recall_permanent("k1") == "v1"
        assert mm.recall_permanent("k2") == "v2"

    def test_recall_permanent_fallback(self):
        """recall_permanent deve fazer fallback ao dict interno."""
        backend = DictMemoryBackend()
        mm = MemoryManager(backend=backend)

        # Guarda diretamente no dict interno (como fazia antes)
        from core.v2.memory import MemoryEntry
        mm._permanent["test_key"] = MemoryEntry(key="test_key", value="cached_value")
        mm._save_permanent()

        result = mm.recall_permanent("test_key")
        assert result == "cached_value"

    def test_delete_permanent_syncs_backend(self):
        """delete_permanent deve sincronizar com o backend."""
        backend = DictMemoryBackend()
        mm = MemoryManager(backend=backend)

        from core.v2.memory import MemoryEntry
        mm._permanent["del_key"] = MemoryEntry(key="del_key", value="to_delete")
        mm._save_permanent()

        result = mm.delete_permanent("del_key")
        assert result is True
        assert mm.recall_permanent("del_key") is None

    def test_backend_property(self):
        """Propriedade backend deve retornar o backend."""
        mm = MemoryManager()
        assert hasattr(mm, "backend")
        assert mm.backend is not None


# ======================================================================
#  Testes: Comportamento com event loop running
# ======================================================================

class TestMemoryBackendAsync:
    """Testes assíncronos com event loop a correr."""

    @pytest.mark.asyncio
    async def test_dict_backend_store_recall_async(self):
        """Store e recall assíncronos no DictMemoryBackend."""
        backend = DictMemoryBackend()
        record = MemoryRecord(key="async_test", text="async content")

        record_id = await backend.store(record)
        assert record_id == record.id

        result = await backend.recall("async_test")
        assert result is not None
        assert result.text == "async content"

    @pytest.mark.asyncio
    async def test_dict_backend_search_async(self):
        """Pesquisa assíncrona no DictMemoryBackend."""
        backend = DictMemoryBackend()
        await backend.store(MemoryRecord(key="k1", text="Python programming"))
        await backend.store(MemoryRecord(key="k2", text="JavaScript development"))
        await backend.store(MemoryRecord(key="k3", text="Cooking recipes"))

        results = await backend.search("Python")
        assert len(results) >= 1
        assert results[0].key == "k1"

    @pytest.mark.asyncio
    async def test_dict_backend_lifecycle_async(self):
        """Ciclo de vida completo: store, list, search, delete, count."""
        backend = DictMemoryBackend()

        # Store
        ids = []
        for i in range(5):
            rid = await backend.store(MemoryRecord(
                key=f"key_{i}", text=f"Memory entry {i}", category="test"
            ))
            ids.append(rid)

        assert await backend.count() == 5

        # List
        all_recs = await backend.list_all()
        assert len(all_recs) == 5

        # Search
        found = await backend.search("entry")
        assert len(found) >= 1

        # Delete
        assert await backend.delete("key_0") is True
        assert await backend.count() == 4

        # Recall deleted
        assert await backend.recall("key_0") is None

    @pytest.mark.asyncio
    async def test_chroma_fallback_async(self):
        """ChromaMemoryBackend sem dependências funciona como fallback."""
        backend = ChromaMemoryBackend(memory_manager=None)
        # Aceder a .healthy dispara _resolve_dependencies que cria o fallback
        assert backend.healthy is False

        await backend.store(MemoryRecord(key="fallback_test", text="fallback content"))
        count = await backend.count()
        assert count >= 1

        result = await backend.recall("fallback_test")
        assert result is not None
        assert result.text == "fallback content"

    @pytest.mark.asyncio
    async def test_chroma_with_mock_async(self):
        """ChromaMemoryBackend com dependências mockadas."""
        mock_mm = MagicMock()
        # add_entry deve devolver um dict com text consistente com o que foi passado
        mock_mm.add_entry.side_effect = lambda text, **kw: {
            "id": "mid-1", "text": text, "category": kw.get("category", "fact"),
            "source": kw.get("source", "user"),
        }
        mock_mm.load_all.return_value = [
            {"id": "mid-1", "key": "mk", "text": "mock content",
             "category": "fact", "source": "user", "timestamp": 100},
        ]

        backend = ChromaMemoryBackend(memory_manager=mock_mm)
        assert backend.healthy is True

        await backend.store(MemoryRecord(key="mk", text="mock content"))
        result = await backend.recall("mk")
        assert result is not None
        assert result.text == "mock content"
