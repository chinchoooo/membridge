"""Passphrase encryption for backups, exports and the database vault.

Format (MBENC1): magic | scrypt salt(16) | nonce prefix(8) | chunk size (u32) | AES-256-GCM chunks. Each chunk's nonce is
prefix || counter, and its associated data records the counter and whether it is the last chunk, so reordering, dropping or
truncating chunks fails authentication. The passphrase is never stored; the key is derived with scrypt (n=2^15, r=8, p=1).
"""

from __future__ import annotations

import getpass
import os
import struct
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

MAGIC = b"MBENC1\n"
CHUNK = 1 << 20
_HDR = len(MAGIC) + 16 + 8 + 4


class CryptoError(Exception):
    """Wrong passphrase, tampered file, or not an MBENC1 file."""


def _key(passphrase: str, salt: bytes) -> bytes:
    return Scrypt(salt=salt, length=32, n=2**15, r=8, p=1).derive(passphrase.encode("utf-8"))


def _aad(counter: int, last: bool) -> bytes:
    return struct.pack(">IB", counter, 1 if last else 0)


def encrypt_file(src: Path, dst: Path, passphrase: str, *, chunk_size: int = CHUNK) -> Path:
    if not passphrase:
        raise CryptoError("empty passphrase")
    salt, prefix = os.urandom(16), os.urandom(8)
    aes = AESGCM(_key(passphrase, salt))
    with src.open("rb") as fi, dst.open("wb") as fo:
        fo.write(MAGIC + salt + prefix + struct.pack(">I", chunk_size))
        cur = fi.read(chunk_size)
        n = 0
        while True:
            nxt = fi.read(chunk_size)
            last = not nxt
            fo.write(aes.encrypt(prefix + struct.pack(">I", n), cur, _aad(n, last)))
            if last:
                break
            cur, n = nxt, n + 1
    return dst


def decrypt_file(src: Path, dst: Path, passphrase: str) -> Path:
    with src.open("rb") as fi:
        head = fi.read(_HDR)
        if len(head) < _HDR or not head.startswith(MAGIC):
            raise CryptoError("not an encrypted MemBridge file")
        salt, prefix = head[len(MAGIC) : len(MAGIC) + 16], head[len(MAGIC) + 16 : len(MAGIC) + 24]
        (chunk_size,) = struct.unpack(">I", head[-4:])
        aes = AESGCM(_key(passphrase, salt))
        tmp = dst.with_suffix(dst.suffix + ".part")
        try:
            with tmp.open("wb") as fo:
                cur = fi.read(chunk_size + 16)
                n = 0
                while True:
                    nxt = fi.read(chunk_size + 16)
                    last = not nxt
                    try:
                        fo.write(aes.decrypt(prefix + struct.pack(">I", n), cur, _aad(n, last)))
                    except InvalidTag as e:
                        raise CryptoError("wrong passphrase, or the file is corrupted/truncated") from e
                    if last:
                        break
                    cur, n = nxt, n + 1
            tmp.replace(dst)
        finally:
            tmp.unlink(missing_ok=True)
    return dst


def get_passphrase(env_name: str | None = None, *, confirm: bool = False) -> str:
    """From the named environment variable (the *name* is configuration, the value is never stored) or a hidden prompt."""
    if env_name:
        v = os.environ.get(env_name)
        if not v:
            raise CryptoError(f"environment variable {env_name} is not set")
        return v
    p = getpass.getpass("Passphrase: ")
    if confirm and getpass.getpass("Repeat passphrase: ") != p:
        raise CryptoError("passphrases do not match")
    return p


def lock_database(db_path: Path, passphrase: str) -> Path:
    """Encrypt the database file at rest and remove the plaintext (only after the encrypted copy verifies).

    Refuses while any other process has the database open: deleting a file under a running server loses its writes
    (POSIX) or fails half way (Windows)."""
    import sqlite3
    import tempfile

    from .store.openlock import held_by_anyone

    if held_by_anyone(db_path):
        raise CryptoError(
            "the database is open in another process (the dashboard, an MCP server, or another command). "
            "Stop it first, then lock; nothing was changed."
        )
    conn = sqlite3.connect(db_path, timeout=2)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.OperationalError as e:
        raise CryptoError("the database is busy (another process is writing to it); nothing was changed") from e
    finally:
        conn.close()
    enc = db_path.with_suffix(db_path.suffix + ".enc")
    encrypt_file(db_path, enc, passphrase)
    with tempfile.TemporaryDirectory() as td:  # verify the round trip before destroying the plaintext
        probe = decrypt_file(enc, Path(td) / "probe.db", passphrase)
        if probe.stat().st_size != db_path.stat().st_size:
            enc.unlink(missing_ok=True)
            raise CryptoError("verification failed; plaintext kept")
    try:
        for suffix in ("", "-wal", "-shm"):
            Path(str(db_path) + suffix).unlink(missing_ok=True)
    except PermissionError as e:
        enc.unlink(missing_ok=True)  # keep exactly one authoritative copy: the plaintext that is still in use
        raise CryptoError("could not remove the plaintext (a process still has it open); the encrypted copy was discarded") from e
    return enc


def plaintext_artifacts(data_dir: Path) -> list[Path]:
    """Files beside the database that hold readable memories: exports, backups, LLM dry-run payloads, import manifests,
    and pre-restore copies. Locking the database does nothing for these."""
    found: list[Path] = []
    for sub, pattern in (("exports", "*"), ("backups", "*.db"), ("llm-dryrun", "*"), ("imports", "*")):
        d = data_dir / sub
        if d.is_dir():
            found += [p for p in d.glob(pattern) if p.is_file() and p.suffix != ".enc"]
    found += [p for p in data_dir.glob("*.pre-restore-*.db") if p.is_file()]
    return sorted(found)


def protect_artifacts(data_dir: Path, passphrase: str) -> dict[str, int]:
    """Encrypt plaintext backups (kept, as .enc) and delete the re-creatable rest (exports, dry-run payloads, manifests)."""
    encrypted = deleted = 0
    for p in plaintext_artifacts(data_dir):
        if p.parent.name == "backups" or ".pre-restore-" in p.name:
            encrypt_file(p, p.with_name(p.name + ".enc"), passphrase)
            p.unlink()
            encrypted += 1
        else:
            p.unlink()
            deleted += 1
    return {"encrypted": encrypted, "deleted": deleted}


def unlock_database(db_path: Path, passphrase: str) -> Path:
    enc = db_path.with_suffix(db_path.suffix + ".enc")
    if db_path.exists():
        raise CryptoError(f"{db_path} already exists (unlocked); refusing to overwrite it")
    decrypt_file(enc, db_path, passphrase)
    enc.unlink()
    return db_path
