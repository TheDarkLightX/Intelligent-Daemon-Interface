from __future__ import annotations

import sys
import types

import pytest

from idi.ian.network.node import NodeIdentity


def _install_fake_keyring(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], str]:
    store: dict[tuple[str, str], str] = {}

    fake = types.ModuleType("keyring")

    def set_password(service: str, account: str, secret: str) -> None:
        store[(service, account)] = secret

    def get_password(service: str, account: str) -> str | None:
        return store.get((service, account))

    fake.set_password = set_password  # type: ignore[attr-defined]
    fake.get_password = get_password  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "keyring", fake)
    return store


def test_identity_ref_keyring_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_keyring(monkeypatch)

    identity = NodeIdentity.generate()
    ref = "keyring://idi.ian/test_node_identity"

    identity.save_to_ref(ref)
    loaded = NodeIdentity.load_from_ref(ref)

    assert loaded.node_id == identity.node_id


def test_identity_ref_keyring_missing_entry_is_filenotfound(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_keyring(monkeypatch)

    with pytest.raises(FileNotFoundError):
        NodeIdentity.load_from_ref("keyring://idi.ian/does_not_exist")

