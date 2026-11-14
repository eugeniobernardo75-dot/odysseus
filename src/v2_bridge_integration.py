"""
src/v2_bridge_integration.py
=============================
Integração definitiva do Core V2 (RC1) com a aplicação principal.

Responsabilidades:
  1. Registar o modelo "Odysseus V2" como um ModelEndpoint na base de dados da
     app principal, apontando para o bridge V2 (v2_server.py). Isto faz o modelo
     aparecer no picker da interface Odysseus, sem configuração manual.
  2. Garantir que o bridge V2 está a correr (healthcheck + autostart em subprocesso)
     quando a app principal arranca, para que o utilizador possa simplesmente
     abrir o Odysseus, selecionar "Odysseus V2" e conversar.

O bridge V2 (v2_server.py) expõe /v1/chat/completions OpenAI-compatível e, desde
o RC1, usa por omissão o OrchestratorAdapter (Orchestrator V2 com memória,
agentes, ferramentas, skills, aprendizagem e streaming). A app principal trata
"Odysseus V2" como qualquer outro endpoint LLM — sem duplicação de código.

Configuração (env):
    ODYSSEUS_V2_BRIDGE_URL       — URL base do bridge (default http://127.0.0.1:8001/v1)
    ODYSSEUS_V2_BRIDGE_AUTOSTART — "0"/"false" desativa o arranque automático do bridge
                                    (default: ativo)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Modelo virtual exposto pelo bridge para o Orchestrator V2
from core.v2.constants import (
    V2_BRIDGE_PORT,
    V2_ENDPOINT_ID,
    V2_ENDPOINT_NAME,
    V2_MODEL_ID,
)


def get_v2_bridge_url() -> str:
    """URL base do bridge V2 (OpenAI-compatível)."""
    return os.getenv(
        "ODYSSEUS_V2_BRIDGE_URL",
        f"http://127.0.0.1:{V2_BRIDGE_PORT}/v1",
    ).strip().rstrip("/")


def get_v2_health_url() -> str:
    """URL de health do bridge V2, derivado da base URL."""
    url = get_v2_bridge_url()
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    return origin.rstrip("/") + "/health"


def is_v2_bridge_reachable(timeout: float = 1.0) -> bool:
    """True quando o bridge V2 responde em /health."""
    try:
        import httpx
        r = httpx.get(get_v2_health_url(), timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _bridge_process_environment() -> dict[str, str]:
    """Copia o ambiente da app e recupera o token User no Windows, se preciso."""
    env = os.environ.copy()
    token_name = "ODYSSEUS_V2_APPROVAL_TOKEN"
    if env.get(token_name, "").strip():
        return env

    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as user_env:
                token, _ = winreg.QueryValueEx(user_env, token_name)
        except FileNotFoundError:
            token = ""
        except OSError as exc:
            logger.warning(
                "Não foi possível ler a configuração User do token de aprovação V2 (%s).",
                type(exc).__name__,
            )
            token = ""

        if isinstance(token, str) and token.strip():
            env[token_name] = token.strip()
        else:
            logger.warning(
                "Token de aprovação V2 não encontrado no processo nem na configuração User; "
                "as rotas de aprovação do bridge permanecerão fechadas."
            )

    return env


def ensure_v2_model_endpoint(db=None) -> str:
    """
    Regista (idempotente) o endpoint "Odysseus V2" na tabela ModelEndpoint.

    A linha é criada apenas se ainda não existir; se já existir, apenas garante
    que o modelo odysseus-v2 está pinned e visível. NUNCA re-ativa um endpoint
    que o utilizador desativou na interface, nem altera a sua base_url
    (configuração do utilizador é respeitada).

    Args:
        db: Sessão SQLAlchemy opcional (para testes). Se None, usa SessionLocal.

    Returns:
        O id do endpoint garantido.
    """
    from core.database import ModelEndpoint, SessionLocal

    own_session = db is None
    session = db or SessionLocal()
    try:
        base_url = get_v2_bridge_url()
        ep = (
            session.query(ModelEndpoint)
            .filter(ModelEndpoint.base_url == base_url)
            .first()
        )
        # Fallback por id: evita colisão de PK se o utilizador já tiver um
        # endpoint com id "odysseus-v2" mas base_url diferente. Nesse caso NÃO
        # alteramos a URL do utilizador — apenas avisamos para a discrepância
        # ficar visível (o modelo pinned aponta para a URL existente).
        if ep is None:
            ep = (
                session.query(ModelEndpoint)
                .filter(ModelEndpoint.id == V2_ENDPOINT_ID)
                .first()
            )
            if ep is not None and ep.base_url != base_url:
                logger.warning(
                    "Endpoint id='odysseus-v2' já existe com base_url diferente "
                    "(%s vs %s). A configurar o modelo sobre a URL existente — "
                    "verifica se é o bridge V2 pretendido.", ep.base_url, base_url,
                )
        if ep is None:
            ep = ModelEndpoint(
                id=V2_ENDPOINT_ID,
                name=V2_ENDPOINT_NAME,
                base_url=base_url,
                is_enabled=True,
                model_type="llm",
                endpoint_kind="local",
                model_refresh_mode="manual",
            )
            session.add(ep)
            logger.info("Registado modelo 'Odysseus V2' → %s", base_url)

        # Garantir que o modelo virtual aparece no picker (pinned + cache).
        # Nota: is_enabled só é True na criação — nunca re-ativamos por cima
        # de uma escolha explícita do utilizador.
        pinned = json.loads(ep.pinned_models) if ep.pinned_models else []
        if V2_MODEL_ID not in pinned:
            pinned.append(V2_MODEL_ID)
            ep.pinned_models = json.dumps(pinned)
        cached = json.loads(ep.cached_models) if ep.cached_models else []
        if V2_MODEL_ID not in cached:
            cached.append(V2_MODEL_ID)
            ep.cached_models = json.dumps(cached)
        session.commit()
        return ep.id
    finally:
        if own_session:
            session.close()


def ensure_v2_bridge_running(force: bool = False) -> bool:
    """
    Garante que o bridge V2 está a correr; arranca-o se necessário.

    Se o bridge já responde em /health, não faz nada. Caso contrário, lança
    `v2_server.py` como subprocesso desanexado e AGUARDA até o /health responder
    (janela limitada), para que o arranque da app não reporte sucesso antes de
    o bridge estar realmente utilizável e para não lançar processos duplicados.

    A menos que o autostart esteja desativado via ODYSSEUS_V2_BRIDGE_AUTOSTART=0.

    Returns:
        True se o bridge foi (ou já estava) garantido e saudável; False se o
        autostart está desativado, o arranque falhou, ou o /health não
        respondeu dentro da janela de espera após o spawn.
    """
    if is_v2_bridge_reachable():
        return True

    autostart = os.getenv("ODYSSEUS_V2_BRIDGE_AUTOSTART", "1").strip().lower()
    if not force and autostart in ("0", "false", "no", "off", ""):
        logger.info(
            "Bridge V2 offline e autostart desativado (ODYSSEUS_V2_BRIDGE_AUTOSTART=%s). "
            "O modelo 'Odysseus V2' só funcionará com o bridge a correr "
            "(python v2_server.py).", autostart,
        )
        return False

    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v2_server.py")
    if not os.path.exists(script):
        logger.error("v2_server.py não encontrado em %s — não é possível arrancar o bridge.", script)
        return False

    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "v2_bridge.log")

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0)

    log_fh = None
    try:
        log_fh = open(log_path, "a", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, script],
            cwd=os.path.dirname(script),
            env=_bridge_process_environment(),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=(os.name != "nt"),
            creationflags=creationflags,
        )
        logger.info("Bridge V2 iniciado (pid=%s). Log: %s", proc.pid, log_path)
    except Exception as e:
        logger.error("Falha ao arrancar o bridge V2: %s", e)
        return False
    finally:
        if log_fh is not None:
            log_fh.close()

    # Aguardar até o bridge ficar saudável (janela limitada — evita reportar
    # sucesso prematuro e evitar spawns duplicados em arranques concorrentes).
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if is_v2_bridge_reachable():
            return True
        time.sleep(0.5)

    logger.warning(
        "Bridge V2 iniciado mas /health não respondeu após 15s. "
        "O bridge pode ainda estar a inicializar o LLM/orquestrador; "
        "verifica logs/v2_bridge.log."
    )
    # Processo lançado, mas não confirmado saudável — reporta como não garantido
    # para o estado do arranque ser honesto (a app continua a funcionar; o
    # endpoint fica registado na mesma).
    return False


async def start_v2_bridge_integration() -> dict:
    """
    Ponto de entrada chamado no arranque da app principal (_startup_event).

    Regista o endpoint "Odysseus V2" (para o picker) e garante que o bridge
    está a correr. Nunca levanta exceções — falhas degradam para logs para não
    bloquear o arranque da app.

    Returns:
        dict com estado: {"endpoint_id": str, "bridge_running": bool}
    """
    import asyncio

    state = {"endpoint_id": None, "bridge_running": False}
    try:
        state["endpoint_id"] = await asyncio.to_thread(ensure_v2_model_endpoint)
    except Exception as e:
        logger.warning("Falha ao registar o endpoint 'Odysseus V2': %s", e)

    try:
        state["bridge_running"] = await asyncio.to_thread(ensure_v2_bridge_running)
    except Exception as e:
        logger.warning("Falha ao garantir o bridge V2: %s", e)

    return state
