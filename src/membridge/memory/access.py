"""Consent for private memories: a PIN only the user knows, and grants that are short, bound to one client, and counted.

Why this shape (see docs/threat-model.md):
  * The PIN is typed by the user into a terminal prompt or the dashboard. An agent that can run shell commands can run
    `membridge grant-sensitive`, but it cannot answer a hidden prompt and it never sees the PIN, so a grant cannot be
    minted by the thing that wants the data. The PIN never travels through a chat or an MCP tool.
  * A grant names one client, expires in minutes, and allows a fixed number of sensitive reads. It is a *consent
    control*: the client name is declared by the connecting process, so this does not stop malware running as the same
    OS user. That boundary is the encrypted-at-rest vault (`membridge lock`).
  * Wrong PINs are rate-limited.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Any

from ..store.sqlite import Store

SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**14, 8, 1
MIN_PIN = 6
MAX_MINUTES = 120
MAX_USES = 200
LOCKOUT_AFTER = 5
LOCKOUT_SECONDS = 900


class AccessError(ValueError):
    """Safe to show to the person at the terminal; never contains the PIN."""


def _hash(pin: str, salt: bytes) -> str:
    return hashlib.scrypt(pin.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32).hex()


def _kv_get(store: Store, key: str) -> str | None:
    row = store.conn.execute("SELECT v FROM settings_kv WHERE k=?", (key,)).fetchone()
    return None if row is None else str(row[0])


def _kv_set(store: Store, key: str, value: str) -> None:
    store.conn.execute("INSERT OR REPLACE INTO settings_kv(k, v) VALUES (?,?)", (key, value))


def has_pin(store: Store) -> bool:
    return _kv_get(store, "pin_hash") is not None


def set_pin(store: Store, new_pin: str, old_pin: str | None = None) -> None:
    if len(new_pin) < MIN_PIN:
        raise AccessError(f"the PIN must be at least {MIN_PIN} characters")
    if has_pin(store) and not verify_pin(store, old_pin or ""):
        raise AccessError("the current PIN is wrong")
    salt = secrets.token_bytes(16)
    _kv_set(store, "pin_hash", f"{salt.hex()}${_hash(new_pin, salt)}")
    _kv_set(store, "pin_fail", "0|0")
    store.audit_sync("user", "pin_set", None)


def verify_pin(store: Store, pin: str) -> bool:
    stored = _kv_get(store, "pin_hash")
    if stored is None:
        return False
    fails, since = (int(x) for x in (_kv_get(store, "pin_fail") or "0|0").split("|"))
    now = int(time.time())
    if fails >= LOCKOUT_AFTER and now - since < LOCKOUT_SECONDS:
        raise AccessError(f"too many wrong PINs; try again in {LOCKOUT_SECONDS - (now - since)} seconds")
    salt_hex, want = stored.split("$")
    ok = hmac.compare_digest(_hash(pin, bytes.fromhex(salt_hex)), want)
    if ok:
        _kv_set(store, "pin_fail", "0|0")
    else:
        fails = fails + 1 if now - since < LOCKOUT_SECONDS or fails == 0 else 1
        _kv_set(store, "pin_fail", f"{fails}|{now}")
        store.audit_sync("user", "pin_wrong", None, fails=fails)
    store.conn.commit()
    return ok


def create_grant(store: Store, pin: str, client: str, minutes: int = 10, uses: int = 25) -> dict[str, Any]:
    if not has_pin(store):
        raise AccessError("no PIN is set yet: run `membridge set-pin` first")
    if not client or client == "*" or len(client) > 80:
        raise AccessError("a grant names exactly one client")
    if not 1 <= minutes <= MAX_MINUTES or not 1 <= uses <= MAX_USES:
        raise AccessError(f"minutes must be 1-{MAX_MINUTES} and uses 1-{MAX_USES}")
    if not verify_pin(store, pin):
        raise AccessError("wrong PIN")
    now = time.time()
    cur = store.conn.execute(
        "INSERT INTO sensitive_grants(client, created_at, expires_at, uses_left) VALUES (?,?,?,?)", (client, now, now + 60 * minutes, uses)
    )
    store.audit_sync("user", "sensitive_grant", client, minutes=minutes, uses=uses)
    return {"grant_id": cur.lastrowid, "client": client, "expires_in_minutes": minutes, "uses": uses}


def consume_grant(store: Store, client: str) -> bool:
    """One sensitive read for `client`, if it holds a live grant. Atomic: two requests cannot spend the same use."""
    now = time.time()
    cur = store.conn.execute(
        "UPDATE sensitive_grants SET uses_left = uses_left - 1 WHERE id = ("
        "SELECT id FROM sensitive_grants WHERE client=? AND revoked=0 AND expires_at>? AND uses_left>0 "
        "ORDER BY expires_at LIMIT 1)",
        (client, now),
    )
    store.conn.commit()
    return cur.rowcount == 1


def revoke(store: Store, client: str | None = None) -> int:
    cur = store.conn.execute(
        "UPDATE sensitive_grants SET revoked=1 WHERE revoked=0 AND (? IS NULL OR client=?)", (client, client)
    )
    store.audit_sync("user", "sensitive_revoke", client)
    return cur.rowcount


def list_grants(store: Store) -> list[dict[str, Any]]:
    now = time.time()
    return [
        {**dict(r), "active": bool(not r["revoked"] and r["expires_at"] > now and r["uses_left"] > 0)}
        for r in store.conn.execute("SELECT id, client, created_at, expires_at, uses_left, revoked FROM sensitive_grants ORDER BY id DESC LIMIT 50")
    ]
