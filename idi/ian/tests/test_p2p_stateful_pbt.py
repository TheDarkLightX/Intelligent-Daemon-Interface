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
from idi.ian.network.protocol import MessageType
from idi.ian.tests.corpus_utils import write_json_corpus_case

try:
    from hypothesis import HealthCheck, settings
    from hypothesis import strategies as st
    from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


_REFERENCE_NOW_S = 1704067200.0
_REFERENCE_NOW_MS = int(_REFERENCE_NOW_S * 1000)
_CORPUS_DIR = Path(__file__).parent / "corpus" / "p2p_stateful"


@dataclass
class Action:
    name: str
    payload: dict[str, Any]


def _make_manager() -> P2PManager:
    identity = NodeIdentity()
    config = P2PConfig(
        max_messages_per_second=1.0,
        rate_limit_burst=2,
    )
    mgr = P2PManager(identity=identity, config=config)
    mgr.register_default_handlers()
    return mgr


def _make_peer_identity() -> NodeIdentity:
    return NodeIdentity()


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


def _message_dict(*, msg_type: MessageType, sender_id: str, nonce: str, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    base: dict[str, Any] = {
        "type": msg_type.value,
        "sender_id": sender_id,
        "timestamp": _REFERENCE_NOW_MS,
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

    class Machine(RuleBasedStateMachine):
        def __init__(self) -> None:
            super().__init__()
            self.mgr = _make_manager()
            self.peer = _make_peer_identity()
            self.session_id = "peer_0"
            self.session: Optional[PeerSession] = None
            self.actions: List[Action] = []

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

            msg = _message_dict(
                msg_type=MessageType.PING,
                sender_id=self.peer.node_id,
                nonce=nonce,
            )
            data = json.dumps(msg).encode("utf-8")

            with patch("idi.ian.network.p2p_manager.time.time", lambda: _REFERENCE_NOW_S):
                asyncio.run(self.mgr._handle_message(self.session, data))

            self.actions.append(Action(name="send_message", payload={"nonce": nonce}))

        @rule(nonce=st.text(min_size=1, max_size=24))
        def replay_same_message(self, nonce: str) -> None:
            if self.session is None:
                return

            msg = _message_dict(
                msg_type=MessageType.PING,
                sender_id=self.peer.node_id,
                nonce=nonce,
            )
            data = json.dumps(msg).encode("utf-8")

            with patch("idi.ian.network.p2p_manager.time.time", lambda: _REFERENCE_NOW_S):
                asyncio.run(self.mgr._handle_message(self.session, data))
                size_after_first = len(self.mgr._seen_messages)
                asyncio.run(self.mgr._handle_message(self.session, data))
                size_after_second = len(self.mgr._seen_messages)

            self.actions.append(Action(name="replay", payload={"nonce": nonce}))

            if size_after_second != size_after_first:
                self._persist_failure(reason="replay_cache_size_changed")
                raise AssertionError("Replay changed replay cache size")

        @rule(make_invalid=st.booleans())
        def handshake_response(self, make_invalid: bool) -> None:
            if self.session is None:
                return

            self.session.pending_challenge = b"A" * 32

            peer_info = NodeInfo(
                node_id=self.peer.node_id,
                public_key=self.peer.public_key,
                addresses=["tcp://127.0.0.1:9001"],
                capabilities=NodeCapabilities(),
                timestamp=_REFERENCE_NOW_MS,
                signature=None,
            )
            self.session.info = peer_info

            their_nonce = b"B" * 32
            msg_to_sign = hashlib.sha256(self.session.pending_challenge + their_nonce + self.peer.node_id.encode()).digest()
            sig = self.peer.sign(msg_to_sign)

            if make_invalid:
                sig = sig[:-1] + bytes([(sig[-1] ^ 0xFF)])

            response = {
                "sender_id": self.peer.node_id,
                "response_nonce": their_nonce.hex(),
                "signature": sig.hex(),
            }

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
            if len(self.mgr._seen_messages) > self.mgr._max_seen_messages:
                self._persist_failure(reason="replay_cache_exceeded_bound")
                raise AssertionError("Replay cache exceeded bound")

    TestCase = Machine.TestCase
