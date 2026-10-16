from __future__ import annotations

import os
from pathlib import Path

import pytest

from membridge.crypto import CryptoError, decrypt_file, encrypt_file, get_passphrase, lock_database, unlock_database
from membridge.ops import backup, restore
from membridge.store.sqlite import Store


def _blob(p: Path, n: int) -> bytes:
    data = os.urandom(n)
    p.write_bytes(data)
    return data


@pytest.mark.parametrize("size", [0, 1, 999, 1000, 1001, 5000])
def test_round_trip_across_chunk_boundaries(tmp_path: Path, size: int) -> None:
    src = tmp_path / "a.bin"
    data = _blob(src, size)
    enc = encrypt_file(src, tmp_path / "a.enc", "correct horse", chunk_size=1000)
    assert enc.read_bytes()[:7] == b"MBENC1\n" and (size < 16 or data not in enc.read_bytes())
    assert decrypt_file(enc, tmp_path / "b.bin", "correct horse").read_bytes() == data


def test_wrong_passphrase_tampering_truncation_and_reordering_all_fail(tmp_path: Path) -> None:
    src = tmp_path / "a.bin"
    _blob(src, 4500)
    enc = encrypt_file(src, tmp_path / "a.enc", "pw", chunk_size=1000)
    raw = enc.read_bytes()
    with pytest.raises(CryptoError, match="wrong passphrase"):
        decrypt_file(enc, tmp_path / "x", "nope")
    flipped = bytearray(raw)
    flipped[60] ^= 1
    (tmp_path / "t.enc").write_bytes(bytes(flipped))
    with pytest.raises(CryptoError):
        decrypt_file(tmp_path / "t.enc", tmp_path / "x", "pw")
    (tmp_path / "trunc.enc").write_bytes(raw[: 31 + 2 * 1016])  # whole chunks dropped from the end: still must fail
    with pytest.raises(CryptoError):
        decrypt_file(tmp_path / "trunc.enc", tmp_path / "x", "pw")
    hdr, body = raw[:31], raw[31:]
    chunks = [body[i : i + 1016] for i in range(0, len(body), 1016)]
    (tmp_path / "swap.enc").write_bytes(hdr + chunks[1] + chunks[0] + b"".join(chunks[2:]))
    with pytest.raises(CryptoError):
        decrypt_file(tmp_path / "swap.enc", tmp_path / "x", "pw")
    (tmp_path / "plain.txt").write_text("hello")
    with pytest.raises(CryptoError, match="not an encrypted"):
        decrypt_file(tmp_path / "plain.txt", tmp_path / "x", "pw")
    assert not (tmp_path / "x").exists() and not list(tmp_path.glob("*.part"))  # no half-written plaintext left behind


def test_each_encryption_is_unique_and_passphrase_comes_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "a.bin"
    _blob(src, 100)
    a = encrypt_file(src, tmp_path / "1.enc", "pw").read_bytes()
    b = encrypt_file(src, tmp_path / "2.enc", "pw").read_bytes()
    assert a != b  # fresh salt and nonce prefix every time
    monkeypatch.setenv("MB_PW", "from-env")
    assert get_passphrase("MB_PW") == "from-env"
    with pytest.raises(CryptoError, match="not set"):
        get_passphrase("MB_UNSET")
    with pytest.raises(CryptoError, match="empty"):
        encrypt_file(src, tmp_path / "3.enc", "")


def test_encrypted_backup_restores_with_the_right_passphrase_only(store: Store, tmp_path: Path, export_file: Path) -> None:
    from membridge.importers.service import run_import

    run_import(store, export_file)
    plain = backup(store, tmp_path / "bk")
    enc = encrypt_file(Path(plain["path"]), tmp_path / "bk.db.enc", "secret")
    Path(plain["path"]).unlink()
    with pytest.raises(CryptoError):
        decrypt_file(enc, tmp_path / "bad.db", "wrong")
    restored = decrypt_file(enc, tmp_path / "ok.db", "secret")
    res = restore(tmp_path / "new" / "membridge.db", restored) if (tmp_path / "new").mkdir() is None else {}
    assert res["stats"]["conversations"] == 6 and res["stats"]["messages"] == 16


def test_vault_lock_and_unlock_round_trip(tmp_path: Path, export_file: Path) -> None:
    from membridge.importers.service import run_import

    db = tmp_path / "membridge.db"
    s = Store(db)
    run_import(s, export_file)
    s.close()
    size = db.stat().st_size
    enc = lock_database(db, "vault-pass")
    assert enc.exists() and not db.exists() and not Path(str(db) + "-wal").exists()
    with pytest.raises(CryptoError):
        unlock_database(db, "wrong")
    assert enc.exists() and not db.exists()  # a failed unlock destroys nothing
    unlock_database(db, "vault-pass")
    assert db.exists() and not enc.exists() and db.stat().st_size == size
    s2 = Store(db)
    assert s2.stats()["messages"] == 16
    s2.close()
    with pytest.raises(CryptoError, match="already exists"):
        unlock_database(db, "p")  # the plaintext is present: never overwrite it
