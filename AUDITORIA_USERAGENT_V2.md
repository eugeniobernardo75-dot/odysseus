# Auditoria do UserAgent V2 — evidências reais (26/10/2026)

## 1. Âmbito e limites

Trabalho limitado ao UserAgent do Odysseus V2 (`core/v2/`) e aos componentes
estritamente necessários ao seu funcionamento (`llm_client`, `cache`,
`execution_policy`, `orchestrator`). Não foi alterado nenhum outro agente,
Worker ou Specialist, nenhum documento normativo ou arquitectónico do MIND OS
ou do Brain Vault, e nenhuma protecção foi desactivada. Nenhum serviço foi
iniciado e nenhuma operação destrutiva foi executada.

## 2. Estado do repositório (antes e depois)

`git status --short` é idêntico no início e no fim desta sessão:

- Modificados (pré-existentes, preservados): `.env.example`, `README.md`,
  `app.py`, `routes/chat_routes.py`, `src/llm_core.py`, `static/js/chat.js`,
  `tests/test_document_diff_discard_on_update_js.py`,
  `tests/test_llm_core_ollama.py`.
- Não rastreados: `core/v2/`, `tests/v2/`, `Workspace/`, `Quarantine/`,
  `Obsidian Brain Vault/`, `Odysseus_Mind_OS/`, `_audit/`, `static/v2/`,
  `v2_server.py`, `src/v2_bridge_integration.py`, `run_v2.bat`,
  `install-bridge-autostart.ps1`, `typings/`, relatórios `.txt`.

Nada foi descartado, movido ou staged. `git diff --stat` continua em 8
ficheiros e 700 inserções — nenhuma alteração anterior foi perdida.

As modificações em `src/llm_core.py` e `tests/test_llm_core_ollama.py`
(`think=False` em modelos com reasoning) são anteriores a este trabalho; foram
preservadas e ficam fora do âmbito do UserAgent.

## 3. Ambiente — causa real do bloqueio de escrita

Durante as primeiras sessões, a escrita em
`C:\Users\Eugenio Bernardo\Documents\odysseus` falhava com `ENOENT` na
criação de ficheiros e `EBADF` na escrita, enquanto `Desktop`, `Downloads`,
home e `%TEMP%` gravavam normalmente. O `git status` nunca reflectiu qualquer
alteração, confirmando que nada chegou a ser gravado.

Diagnóstico efectuado nesta sessão:

| Verificação | Resultado |
| --- | --- |
| ACL da pasta e do projecto | utilizador com `(F)` herdado; sem *deny* de escrita |
| Integridade do processo | `Nivel Obrigatorio Medio`; grupo `CodexSandboxUsers` não pertence ao token |
| `Get-MpPreference` | `EnableControlledFolderAccess = 1` (Controlled Folder Access activo) |
| Escrita em `Documents`, `Pictures`, `Favorites` | falha |
| Escrita em `AppData\Local`, `Public`, `Desktop`, home | funciona |

As três pastas que falham são exactamente as pastas protegidas por predefinição
pelo Controlled Folder Access do Windows Defender, e a lista de aplicações
permitidas estava vazia. Tratava-se, portanto, de uma protecção do sistema
operativo e não do projecto.

Resolução, com aprovação explícita do utilizador via prompt UAC:

    Add-MpPreference -ControlledFolderAccessAllowedApplications
        "...\AppData\Local\Python\pythoncore-3.14-64\python.exe"
    Add-MpPreference -ControlledFolderAccessAllowedApplications
        "...\AppData\Local\Programs\@codebufffreebuff-desktop\Freebuff.exe"

O Controlled Folder Access **continua activo** (`EnableControlledFolderAccess
= 1`) e continua a proteger todas as restantes aplicações. Nenhuma protecção foi
desligada.

Após a alteração: criação de ficheiros e escrita em `odysseus` funcionam.

## 4. V1 — comportamento analisado e simulado

Cadeia executável confirmada no código:

- `routes/chat_routes.py` importa `stream_agent_loop` de `src.agent_loop`
  (linha 18) e invoca `chat_handler` (linhas 360, 418, 712, 1611);
- `src/chat_handler.py` importa `stream_llm_with_fallback` (linha 20);
- `src/agent_loop.py` usa `stream_llm_with_fallback` (linhas 2413, 2993, 3015).

Factos observados:

- O pedido é classificado antes de responder (`_classify_agent_request`), com
  detecção de sinal baixo (saudação) e de domínios técnicos;
- as ferramentas correm num ciclo com limite de rondas e o output é truncado
  antes de regressar ao modelo;
- a chamada ao modelo passa por `stream_llm_with_fallback`, com fallback para o
  endpoint secundário quando o primário falha (ex.: 503);
- o payload Ollama nativo desactiva `think` em modelos com reasoning.

Simulação com mocks e transporte falso, sem rede real:

    pytest tests/test_agent_loop.py tests/test_llm_core_ollama.py \
           tests/test_llm_core_fallback.py tests/test_llm_core_ollama_thinking.py
    → 95 passed

Transportado para o V2: classificação de intenção, execução directa antes de
delegar, fallback controlado quando o modelo não responde e histórico limitado
ao pedido. Deliberadamente não transportado: chamada automática ao modelo em
todos os turnos e substituição silenciosa do modelo por outro instalado.

## 5. UserAgent V2 — comportamento implementado

- `process` / `process_stream`: sanitiza a entrada, trata pedidos de memória,
  recupera conhecimento do Brain Vault e do MIND OS e entrega ao Orchestrator
  com conhecimento, memória e histórico do pedido. Metadados devolvidos:
  `coordinated_by`, `resolution_route`, `knowledge_sources`;
- `execute`: resposta directa — modelo apenas com autorização expressa, caso
  contrário template — e nunca sonda o servidor de modelos;
- rotas observadas: `direct` (conversa), `planned` (tarefa), `delegated`
  (com Worker) e `approval_required`;
- delegação justificada: `AgentFactory.select_worker` devolve `None` quando
  nenhum Worker documentado corresponde às palavras-chave, e
  `Orchestrator._bind_delegated_worker` recusa worker sem
  `is_operational_worker`, deixando as ferramentas bloqueadas. A identidade do
  Worker é decidida pelo servidor, nunca pela saída do modelo;
- ficheiros: as ferramentas ficam confinadas às raízes aprovadas e o servidor
  constrói o contexto (`build_tool_context`), descartando as chaves impostas
  pelo chamador (`workspace_path`, `cwd`, `allow_interpreters`,
  `useragent_knowledge`, `useragent_memory`).

## 6. Ollama — bloqueio automático e indirecto

Autorização por pedido (`useragent_ollama_authorization`) derivada do texto do
utilizador (`explicit_model_use_requested`), com negação explícita a vencer.

Ponto de bloqueio aplicado em `core/v2/llm_client.py`, `_ollama_request_denied()`:

| Linha | Método | Comportamento |
| --- | --- | --- |
| 593 | `check_availability` | devolve `False`, sem rede |
| 744 | `chat` | devolve erro de controlo, sem rede |
| 1009 | `chat_stream` | devolve "bloqueada" + `__DONE__` |
| 1221 | `list_models` | levanta `OllamaAuthorizationError` |

Adicionalmente: pedido do UserAgent não sonda `/api/tags` nem substitui o modelo
(uma tentativa, sem retry); `cache.SemanticCache._compute_embedding` não pede
embeddings durante um pedido do UserAgent; `resume_approved_plan` re-deriva o
consentimento do pedido guardado no plano, pelo que aprovar uma ferramenta não
autoriza o modelo.

Evidência: `pytest tests/v2/test_useragent_ollama_policy.py` → **13 passed**,
cubrindo chamada directa, streaming, orquestrador indirecto, cache, retoma
aprovada e `execute()`.

## 7. core/v2/execution_policy.py

Estado actual, verificado por inspecção:

- `approved_root()` exige pasta existente, directório real (realpath),
  estritamente dentro da raiz do projecto e fora de áreas protegidas;
- **não contém `mkdir`** — não cria directorias.

A criação automática descrita no relatório anterior não está no código e não era
necessária: a falha que motivou a investigação era do Controlled Folder Access
(secção 3), não de raízes em falta. Este relatório substitui o anterior, que
estava incorrecto nesse ponto e classificava o problema como um erro de
teardown do pytest.

## 8. tests/v2/test_orchestrator_adapter.py — investigação e desfecho

O teste **não foi substituído nem alterado**.

A falha observada era `FileNotFoundError` na linha 649, na primeira escrita da
própria fixture do teste (`test_file.write_bytes(b"test")` dentro de
`Workspace/`), antes de qualquer código da aplicação. A mesma escrita falhava
fora do pytest e falhava também na raiz do projecto, em `Pictures` e em
`Favorites`, e nunca em pastas fora da protecção do CFA — ou seja, era ambiente,
não lógica.

Depois de resolvido o acesso de escrita:

    pytest "tests/v2/test_orchestrator_adapter.py::TestToolManagerEventCallbacks::test_tool_manager_fires_tool_output_on_success"
    → 1 passed

## 9. Aprovações, áreas reservadas e limites do Workspace

Passam na suite: `test_execution_policy.py`, `test_bridge_approval_flow.py`,
`test_tool_manager.py`, `test_shell_tool.py`,
`test_orchestrator_tool_integration.py`, `test_v2_server_bridge.py`.
O contexto de ferramentas é decidido pelo servidor; propósito inválido devolve
`{}` (fail-closed); Brain Vault, MIND OS, `.git`, `data` e `logs` continuam
protegidos; a aprovação humana continua a ser verificada pelo `SafetyManager`.

## 10. Isolamento Docker

- Daemon Docker 29.8.2 em execução; `docker_cli=true`, `docker_running=true`;
- `ODYSSEUS_V2_ISOLATION_IMAGE` não está definida.

`run_isolated(['python','-c','print(1)'], Workspace)` devolve:

    {"ok": false, "isolated": false, "return_code": null,
     "error": "isolamento indisponível: Docker em execução mas sem imagem
               definida (ODYSSEUS_V2_ISOLATION_IMAGE); a execução isolada
               está indisponível."}

Nada é executado fora do backend e nenhuma protecção foi desactivada.
`REQUIRE_ISOLATION` não está activa, pelo que o isolamento não é barreira
obrigatória nesta configuração — limitação registada, não contornada.

## 11. Brain Vault — caminho oficial e proveniência

**Caminho oficial actual (decisão confirmada pelo utilizador):**

    C:\Users\Eugenio Bernardo\Documents\odysseus\Obsidian Brain Vault\00_SYSTEM\00_Index\ODYSSEUS_USERAGENT_CONTEXT_AND_ARCHITECTURE_MAP.md

Correspondência relativa usada pelo código: `00_SYSTEM/00_Index/ODYSSEUS_USERAGENT_CONTEXT_AND_ARCHITECTURE_MAP.md`.

**Divergência face à instrução original.** A missão inicial mandava testar
`C:\Users\Eugenio Bernardo\Documents\odysseus\Obsidian Brain Vault\00_SYSTEM\Index`.
Essa pasta existiu na primeira fase desta sessão e a leitura foi testada com
sucesso nesse caminho; às 04:33 foi renomeada por um actor externo — não por
esta sessão — para `00_SYSTEM\00_Index`, com o conteúdo inalterado (mtime
25/10 23:24). Depois de confirmada a decisão do utilizador, o teste e este
relatório passaram a usar explicitamente `00_SYSTEM/00_Index` como caminho
oficial. **Não existe fallback silencioso** para `00_SYSTEM\Index`: o teste
falha se o código devolver o caminho legado quando esse ficheiro não existe em
disco.

Proveniência medida no caminho oficial:

    source_type = Brain Vault
    path        = 00_SYSTEM/00_Index/ODYSSEUS_USERAGENT_CONTEXT_AND_ARCHITECTURE_MAP.md
    authority   = architectural
    status      = proposed
    content     = 3000 caracteres lidos (comparados com o ficheiro em disco)

O teste `test_brain_vault_index_file_is_readable_with_provenance` valida
exactamente: caminho devolvido igual ao oficial, `source_type == "Brain Vault"`,
conteúdo presente e igual ao início do ficheiro em disco, `matched_keywords`,
`authority` e `status`. Faz `pytest.skip` apenas se o ficheiro oficial não
existir no host, para não depender da máquina.

Aviso: `BRAIN_VAULT_ROOT/00_Index` (top-level) continua inexistente; o código
degrada para leitura directa em vez de falhar.

## 12. Testes executados e resultados reais

Resultados **históricos** (26/10, antes da correcção do caminho oficial):

| Comando | Resultado |
| --- | --- |
| `pytest tests/v2` (base, com escrita desbloqueada) | 760 passed, 8 skipped, 0 failed |
| `pytest tests/v2/test_useragent_ollama_policy.py` | 13 passed |
| teste isolado `...TestToolManagerEventCallbacks::test_tool_manager_fires_tool_output_on_success` | 1 passed |
| `pytest tests/test_agent_loop.py tests/test_llm_core_ollama.py tests/test_llm_core_fallback.py tests/test_llm_core_ollama_thinking.py` (V1 com mocks) | 95 passed |
| `pytest tests/v2` (com o teste novo apontado a `00_SYSTEM/Index`) | 761 passed, 8 skipped, 0 failed |
| `pytest tests/v2` (após renomeação externa; teste novo em `pytest.skip`) | 760 passed, 9 skipped, 0 failed |

Resultados **actuais** (depois de o teste apontar ao caminho oficial
`00_SYSTEM/00_Index`):

| Comando | Resultado |
| --- | --- |
| `pytest tests/v2/test_useragent_coordination.py::test_brain_vault_index_file_is_readable_with_provenance` | **1 passed** |
| `pytest tests/v2 -rs` | **761 passed, 8 skipped, 0 failed** (exit 0) |

Os 8 `skip` actuais são: 7 em `tests/v2/test_isolation.py` (Docker em execução
mas sem `ODYSSEUS_V2_ISOLATION_IMAGE`, portanto a fronteira não pode ser
demonstrada) e 1 em `tests/v2/test_orchestrator_adapter.py:784` (teste de
integração que requer servidor em execução). Nenhum skip refere o Brain Vault.

Todos os comandos correram com `python -m pytest -p no:cacheprovider` e um
`--basetemp` dedicado, porque o directório por omissão em `%TEMP%`
(`pytest-of-Eugenio Bernardo`) contém entradas antigas de 21/09 que não são
removíveis (`Permission denied`) e fazem o `pytest` terminar com erro de
teardown depois dos testes.

Nenhum teste foi enfraquecido, ignorado ou substituído.

## 13. Alterações efectuadas

1. `tests/v2/test_useragent_coordination.py` — acrescentado
   `test_brain_vault_index_file_is_readable_with_provenance` (append, sem
   alterar código existente) e depois actualizado para o caminho oficial
   `00_SYSTEM/00_Index`, validando caminho devolvido, `source_type`, conteúdo e
   metadados de proveniência e recusando o caminho legado.
2. `AUDITORIA_USERAGENT_V2.md` — escrito de raiz com as evidências da sessão e
   depois corrigido (secções 11 a 15) para reflectir o caminho oficial.
3. Ambiente (com aprovação UAC): `python.exe` e `Freebuff.exe` passaram a
   constar das aplicações permitidas do Controlled Folder Access; a protecção
   mantém-se activa (`EnableControlledFolderAccess = 1`).
4. Commit local selectivo com os dois artefactos desta tarefa
   (`AUDITORIA_USERAGENT_V2.md` e `tests/v2/test_useragent_coordination.py`);
   nenhum ficheiro alheio foi incluído e as alterações preexistentes
   continuam por estagiar.

## 14. Limitações que permanecem

- Sem `ODYSSEUS_V2_ISOLATION_IMAGE` não é possível demonstrar uma execução real
  dentro de um contentor; o que está demonstrado é a recusa fail-closed.
- `REQUIRE_ISOLATION` não está activa, portanto o isolamento não é obrigatório
  nesta configuração (7 dos 8 skips actuais vêm daqui).
- O índice oficial `00_Index` (top-level) também não existe no vault; o código
  degrada para leitura directa em vez de falhar.
- O `pytest` só fica limpo com `--basetemp` dedicado, devido a entradas antigas
  em `%TEMP%`.
- O commit local inclui apenas os dois artefactos autorizados; `core/v2/` e os
  restantes ficheiros de `tests/v2/` continuam não rastreados, pelo que o
  teste commitado depende de módulos que não fazem parte do commit.
- A ferramenta de edição do Freebuff continua sem permissão de escrita em
  `Documents` (só o processo `python.exe` foi autorizado no CFA); os ficheiros
  foram gravados através do Python.
- A instrução original indicava `00_SYSTEM/Index`; o caminho oficial é agora
  `00_SYSTEM/00_Index`. A divergência está documentada na secção 11 e nenhum
  ficheiro do Brain Vault foi renomeado ou movido por esta sessão.

## 15. Conclusão

Os critérios comportamentais do UserAgent V2 estão suportados por execução:
bloqueio do Ollama em todos os caminhos conhecidos (rede, modelos, embeddings e
retoma de planos), execução directa antes de delegar, delegação apenas com
Worker identificado pelo servidor, leitura com proveniência do caminho oficial
`00_SYSTEM/00_Index`, e aprovais, áreas reservadas e limites do Workspace
intactos.

Estado actual: **761 passed, 8 skipped, 0 failed** na suite `tests/v2`, com o
teste de proveniência a passar no caminho oficial e sem qualquer `skip` do
Brain Vault. Os resultados anteriores (761/8 com `00_SYSTEM/Index` e 760/9 em
`pytest.skip`) ficam registados na secção 12 como histórico, não como estado
vigente.

Nenhuma limitação foi contornada com testes mais fracos.
