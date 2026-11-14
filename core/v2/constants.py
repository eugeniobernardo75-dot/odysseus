"""
Constantes globais do Odysseus V2
==================================

Fonte única de verdade para valores partilhados entre o bridge V2
(v2_server.py), a integração na app principal (src/v2_bridge_integration.py)
e os testes. Nunca repetir portas ou ids em literais espalhados pelo código.

Uso:
    from core.v2.constants import V2_BRIDGE_PORT, V2_MODEL_ID
"""

from __future__ import annotations

import os

# ── Porta do bridge V2 ─────────────────────────────────────────────────────
# A única porta do bridge V2. Todas as referências (servidor, integração,
# testes) devem usar esta constante — nunca literais 8001/8002 avulsos.
V2_BRIDGE_PORT = int(os.getenv("V2_BRIDGE_PORT", "8001"))
# Loopback por omissão: o bridge não tem autenticação própria, pelo que o bind
# em todas as interfaces (0.0.0.0) o tornava alcançável na rede local sem
# qualquer controlo. Expor exige decisão explícita via V2_BRIDGE_HOST.
V2_BRIDGE_HOST = os.getenv("V2_BRIDGE_HOST", "127.0.0.1")

# ── Modelo virtual exposto pelo bridge ─────────────────────────────────────
V2_MODEL_ID = "odysseus-v2"
V2_ENDPOINT_ID = "odysseus-v2"
V2_ENDPOINT_NAME = "Odysseus V2"

# ── Porta da app principal (dashboard / UI) ────────────────────────────────
APP_PORT = int(os.getenv("APP_PORT", "7000"))
