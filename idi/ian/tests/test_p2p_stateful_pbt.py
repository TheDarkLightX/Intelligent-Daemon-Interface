from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from idi.ian.network.node import NodeCapabilities, NodeIdentity, NodeInfo
from idi.ian.network.p2p_manager import P2PConfig, P2PManager, PeerSession, PeerState, TokenBucketRateLimiter
from idi.ian.network.protocol import HandshakeResponse, MessageType
from idi.ian.tests.corpus_utils import write_json_corpus_case

try:
    from hypothesis import HealthCheck, settings
    from hypothesis import strategies as st
    from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule

    HAS_HYPOTHESIS = True
except ImportError:
    class _HealthCheckStub:
        too_slow = object()

    def settings(*_args: Any, **_kwargs: Any):
        def _decorator(fn: Any) -> Any:
            return fn

        return _decorator

    class _StrategiesStub:
        @staticmethod
        def text(**_kwargs: Any) -> Any:
            return None

        @staticmethod
        def booleans() -> Any:
            return None

    st = _StrategiesStub()  # type: ignore

    class RuleBasedStateMachine:  # type: ignore
        class TestCase:  # pragma: no cover
            pass

    def initialize(*_args: Any, **_kwargs: Any):
        def _decorator(fn: Any) -> Any:
            return fn

        return _decorator

    def invariant(*_args: Any, **_kwargs: Any):
        def _decorator(fn: Any) -> Any:
            return fn

        return _decorator

    def precondition(*_args: Any, **_kwargs: Any):
        def _decorator(fn: Any) -> Any:
            return fn

        return _decorator

    def rule(*_args: Any, **_kwargs: Any):
        def _decorator(fn: Any) -> Any:
            return fn

        return _decorator

    HealthCheck = _HealthCheckStub()  # type: ignore
    HAS_HYPOTHESIS = False

if not HAS_HYPOTHESIS:
    def test_hypothesis_dependency_present() -> None:
        pytest.importorskip("hypothesis")

_REFERENCE_NOW_S = 1704067200.0
_REFERENCE_NOW_MS = int(_REFERENCE_NOW_S * 1000)
_CORPUS_DIR = Path(__file__).parent / "corpus" / "p2p_stateful"


@dataclass
class Action:
    name: str
    payload: dict[str, Any]


def _make_manager() -> P2PManager:
    identity = NodeIdentity(private_key=b"\x01" * 32)
    config = P2PConfig(
        max_messages_per_second=1.0,
        rate_limit_burst=2,
    )
    mgr = P2PManager(identity=identity, config=config)
    mgr.register_default_handlers()
    return mgr


def _make_peer_identity() -> NodeIdentity:
    return NodeIdentity(private_key=b"\x02" * 32)


def _make_session(node_id: str) -> PeerSession:
    session = PeerSession(
        node_id=node_id,
        address="127.0.0.1",
        port=9001,
        state=PeerState.CONNECTED,
        connected_at=_REFERENCE_NOW_S,
        rate_limiter=TokenBucketRateLimiter(rate=1.0, burst=2),
    )
    return session


def _message_dict(
    *,
    msg_type: MessageType,
    sender_id: str,
    nonce: str,
    timestamp_ms: int,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "type": msg_type.value,
        "sender_id": sender_id,
        "timestamp": timestamp_ms,
        "nonce": nonce,
        "signature": None,
    }
    if extra:
        base.update(extra)
    return base


async def _run_handshake_response_and_drain_tasks(
    mgr: P2PManager,
    session: PeerSession,
    response: dict[str, Any],
) -> bool:
    tasks: list[asyncio.Task] = []

    original_create_task = asyncio.create_task

    def _create_task_wrapper(coro: Any) -> asyncio.Task:
        t = original_create_task(coro)
        tasks.append(t)
        return t

    with patch("idi.ian.network.p2p_manager.asyncio.create_task", _create_task_wrapper):
        ok = await mgr._handle_handshake_response(session, response)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return ok


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestP2PManagerStateful:
    """Stateful security properties for P2PManager.

    Invariants:
    - Replay is rejected (same sender_id:nonce)
    - Handshake auth failure triggers disconnect
    - Replay cache remains bounded
    """

    @settings(
        max_examples=50,
        deadline=None,
        derandomize=True,
        suppress_health_check=[HealthCheck.too_slow],
    )
    class Machine(RuleBasedStateMachine):
        def __init__(self) -> None:
            super().__init__()
            self.mgr = _make_manager()
            self.peer = _make_peer_identity()
            self.session_id = "peer_0"
            self.session: Optional[PeerSession] = None
            self.actions: List[Action] = []
            self._timestamp_ms = _REFERENCE_NOW_MS
            self._nonce_timestamps: dict[str, int] = {}

        def _persist_failure(self, *, reason: str) -> None:
            payload = {
                "reason": reason,
                "session_id": self.session_id,
                "peer_node_id": self.peer.node_id,
                "actions": [{"name": a.name, "payload": a.payload} for a in self.actions],
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:16]
            filename = f"case_{digest}.json"
            write_json_corpus_case(corpus_dir=str(_CORPUS_DIR), filename=filename, payload=payload)

        @initialize()
        def init(self) -> None:
            self.session = _make_session(self.session_id)
            self.mgr._peers[self.session_id] = self.session
            self.actions.append(Action(name="connect", payload={"peer": self.session_id}))

        @rule(nonce=st.text(min_size=1, max_size=24))
        def send_message(self, nonce: str) -> None:
            if self.session is None:
                return

            self._timestamp_ms += 1
            self._nonce_timestamps[nonce] = self._timestamp_ms
            msg = _message_dict(
                msg_type=MessageType.PING,
                sender_id=self.peer.node_id,
                nonce=nonce,
                timestamp_ms=self._timestamp_ms,
            )
            data = json.dumps(msg).encode("utf-8")

            with patch("idi.ian.network.p2p_manager.time.time", lambda: _REFERENCE_NOW_S):
                asyncio.run(self.mgr._handle_message(self.session, data))

            self.actions.append(Action(name="send_message", payload={"nonce": nonce}))

        @rule(nonce=st.text(min_size=1, max_size=24))
        def replay_same_message(self, nonce: str) -> None:
            if self.session is None:
                return

            timestamp_ms = self._nonce_timestamps.get(nonce, self._timestamp_ms)
            msg = _message_dict(
                msg_type=MessageType.PING,
                sender_id=self.peer.node_id,
                nonce=nonce,
                timestamp_ms=timestamp_ms,
            )
            data = json.dumps(msg).encode("utf-8")

            with patch("idi.ian.network.p2p_manager.time.time", lambda: _REFERENCE_NOW_S):
                asyncio.run(self.mgr._handle_message(self.session, data))
                size_after_first = len(self.mgr._peer_nonce_cache.get(self.peer.node_id, {}))
                asyncio.run(self.mgr._handle_message(self.session, data))
                size_after_second = len(self.mgr._peer_nonce_cache.get(self.peer.node_id, {}))

            self.actions.append(Action(name="replay", payload={"nonce": nonce}))

            if size_after_second != size_after_first:
                self._persist_failure(reason="replay_cache_size_changed")
                raise AssertionError("Replay changed replay cache size")

        @rule(make_invalid=st.booleans())
        def handshake_response(self, make_invalid: bool) -> None:
            if self.session is None:
                return

            self.session.pending_challenge = b"A" * 32
            # Provide deterministic X25519 kx keys (required for handshake completion).
            # These are fixed for reproducibility in the state machine.
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

            our_kx_priv = X25519PrivateKey.from_private_bytes(b"\x11" * 32)
            our_kx_pub = our_kx_priv.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            self.session.kx_private_key = our_kx_priv.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
            self.session.kx_public_key = our_kx_pub

            # Peer kx key for response
            peer_kx_priv = X25519PrivateKey.from_private_bytes(b"\x22" * 32)
            peer_kx_pub = peer_kx_priv.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )

            # Build a real HandshakeResponse message (protocol-level auth).
            import base64

            their_nonce = b"B" * 32
            hs = HandshakeResponse(
                sender_id=self.peer.node_id,
                timestamp=_REFERENCE_NOW_MS,
                nonce="hs_nonce",
                challenge_nonce=self.session.pending_challenge.hex(),
                response_nonce=their_nonce.hex(),
                kx_public_key=base64.b64encode(peer_kx_pub).decode("utf-8"),
                public_key=base64.b64encode(self.peer.public_key).decode("utf-8"),
            )
            self.peer.sign_message(hs)
            if make_invalid and hs.signature is not None:
                hs.signature = hs.signature[:-1] + bytes([(hs.signature[-1] ^ 0xFF)])

            response = hs.to_dict()

            with patch("idi.ian.network.p2p_manager.time.time", lambda: _REFERENCE_NOW_S):
                ok = asyncio.run(_run_handshake_response_and_drain_tasks(self.mgr, self.session, response))

            self.actions.append(Action(name="handshake_response", payload={"invalid": make_invalid}))

            if make_invalid:
                if ok:
                    self._persist_failure(reason="invalid_handshake_accepted")
                    raise AssertionError("Invalid handshake response was accepted")
                if self.session_id in self.mgr._peers:
                    self._persist_failure(reason="invalid_handshake_not_disconnected")
                    raise AssertionError("Invalid handshake did not disconnect peer")
            else:
                if not ok:
                    self._persist_failure(reason="valid_handshake_rejected")
                    raise AssertionError("Valid handshake response was rejected")

        @invariant()
        def replay_cache_bounded(self) -> None:
            for cache in self.mgr._peer_nonce_cache.values():
                if len(cache) > self.mgr._max_nonce_cache_per_peer:
                    self._persist_failure(reason="replay_cache_exceeded_bound")
                    raise AssertionError("Replay cache exceeded bound")

    TestCase = Machine.TestCase


if HAS_HYPOTHESIS:
    TestCase = TestP2PManagerStateful.Machine.TestCase
