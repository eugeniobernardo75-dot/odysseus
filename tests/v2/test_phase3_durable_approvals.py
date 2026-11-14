"""Testes Fase 3 — persistência de aprovações."""
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.v2.safety import SafetyManager


def _make_approval(sm: SafetyManager) -> str:
    r = sm.check_action(
        action_type="create_agent",
        description="test",
        details={"spec": {"name": "test"}},
        permissions=["agents.create"],
    )
    assert r["requires_approval"] is True
    return r["approval_id"]


def test_approval_persists_across_restart(tmp_path):
    path = str(tmp_path / "approvals.json")
    sm1 = SafetyManager(approvals_path=path)
    aid = _make_approval(sm1)

    sm2 = SafetyManager(approvals_path=path)
    entry = sm2.get_approval(aid)
    assert entry is not None
    assert entry["status"] == "pending"


def test_counter_continues_after_restart(tmp_path):
    path = str(tmp_path / "approvals.json")
    sm1 = SafetyManager(approvals_path=path)
    aid1 = _make_approval(sm1)

    sm2 = SafetyManager(approvals_path=path)
    aid2 = _make_approval(sm2)

    assert aid1 != aid2
    n1 = int(aid1.split("-")[1])
    n2 = int(aid2.split("-")[1])
    assert n2 > n1, f"contador regrediu: {aid1} → {aid2}"


def test_denied_persists_across_restart(tmp_path):
    path = str(tmp_path / "approvals.json")
    sm1 = SafetyManager(approvals_path=path)
    aid = _make_approval(sm1)
    ok, _ = sm1.deny(aid)
    assert ok is True

    sm2 = SafetyManager(approvals_path=path)
    assert sm2.get_approval(aid)["status"] == "denied"


def test_approved_persists_across_restart(tmp_path):
    path = str(tmp_path / "approvals.json")
    sm1 = SafetyManager(approvals_path=path)
    aid = _make_approval(sm1)
    ok, _ = sm1.approve(aid)
    assert ok is True

    sm2 = SafetyManager(approvals_path=path)
    assert sm2.get_approval(aid)["status"] == "approved"


def test_expired_marked_on_load(tmp_path):
    path = str(tmp_path / "approvals.json")
    sm1 = SafetyManager(approvals_path=path)
    aid = _make_approval(sm1)

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    data["approvals"][aid]["expires_at"] = past
    Path(path).write_text(json.dumps(data), encoding="utf-8")

    sm2 = SafetyManager(approvals_path=path)
    assert sm2.get_approval(aid)["status"] == "expired"
    pendings = [p["approval_id"] for p in sm2.get_pending()]
    assert aid not in pendings


def test_missing_file_starts_empty(tmp_path):
    path = str(tmp_path / "never_created.json")
    sm = SafetyManager(approvals_path=path)
    assert sm.get_pending() == []
    # e ainda permite criar novas
    aid = _make_approval(sm)
    assert aid.startswith("APPROVAL-")


def test_atomic_write_creates_parent_dir(tmp_path):
    nested = tmp_path / "nested" / "deeper" / "approvals.json"
    sm = SafetyManager(approvals_path=str(nested))
    aid = _make_approval(sm)
    assert nested.exists(), "atomic_write não criou a pasta pai"
    data = json.loads(nested.read_text(encoding="utf-8"))
    assert aid in data["approvals"]


def test_legacy_approval_without_expires_at_accepted(tmp_path):
    """Aprovação persistida por versão anterior sem `expires_at`."""
    path = str(tmp_path / "approvals.json")
    legacy = {
        "counter": 1,
        "approvals": {
            "APPROVAL-0001": {
                "action_type": "x",
                "description": "legacy",
                "details": {},
                "risk": "high",
                "status": "pending",
                "fingerprint": "abc",
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        },
    }
    Path(path).write_text(json.dumps(legacy), encoding="utf-8")
    sm = SafetyManager(approvals_path=path)
    assert sm.get_approval("APPROVAL-0001") is not None
    assert "APPROVAL-0001" in [p["approval_id"] for p in sm.get_pending()]
