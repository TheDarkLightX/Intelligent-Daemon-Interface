from __future__ import annotations

from unittest.mock import patch

import pytest

from idi.ian.models import AgentPack, Contribution, GoalID, GoalSpec
from idi.ian.security import InputValidator, ProofOfWork, RateLimiter, SecurityLimits, SybilResistance, TokenBucket

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

    class _StrategyStub:
        def map(self, _fn):
            return None

    def given(*args, **kwargs):  # type: ignore
        def decorator(fn):
            return pytest.mark.skip(reason="hypothesis not installed")(fn)

        return decorator

    def settings(*args, **kwargs):  # type: ignore
        def decorator(fn):
            return fn

        return decorator

    class _StStub:
        @staticmethod
        def integers(**kwargs):
            return _StrategyStub()

        @staticmethod
        def binary(**kwargs):
            return None

        @staticmethod
        def text(**kwargs):
            return None

        @staticmethod
        def dictionaries(**kwargs):
            return None

        @staticmethod
        def lists(*args, **kwargs):
            return None

        @staticmethod
        def tuples(*args, **kwargs):
            return None

        @staticmethod
        def sampled_from(values):
            return None

        @staticmethod
        def booleans():
            return None

        @staticmethod
        def builds(fn, *args, **kwargs):
            return None

    st = _StStub()  # type: ignore


_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(value=st.text(alphabet=_ALPHABET, min_size=1, max_size=64))
@settings(max_examples=200, deadline=None, derandomize=True)
def test_goal_id_accepts_valid_charset(value: str) -> None:
    goal_id = GoalID(value)
    assert str(goal_id) == value


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(prefix=st.text(alphabet=_ALPHABET, min_size=1, max_size=63), bad=st.sampled_from(list("/- .:")))
@settings(max_examples=100, deadline=None, derandomize=True)
def test_goal_id_rejects_invalid_chars(prefix: str, bad: str) -> None:
    with pytest.raises(ValueError):
        GoalID(prefix + bad)


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(value=st.text(alphabet=_ALPHABET, min_size=65, max_size=80))
@settings(max_examples=50, deadline=None, derandomize=True)
def test_goal_id_rejects_too_long(value: str) -> None:
    with pytest.raises(ValueError):
        GoalID(value)


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(
    capacity=st.integers(min_value=1, max_value=50),
    refill_rate=st.integers(min_value=0, max_value=200).map(float),
    initial_tokens=st.integers(min_value=0, max_value=50),
    steps=st.integers(min_value=1, max_value=50),
    dt_ms=st.integers(min_value=0, max_value=250),
    tokens_to_consume=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=150, deadline=None, derandomize=True)
def test_token_bucket_tokens_bounded(
    capacity: int,
    refill_rate: float,
    initial_tokens: int,
    steps: int,
    dt_ms: int,
    tokens_to_consume: int,
) -> None:
    initial = float(min(initial_tokens, capacity))
    now_s = [0.0]

    def fake_monotonic() -> float:
        return now_s[0]

    with patch("idi.ian.security.time.monotonic", fake_monotonic):
        bucket = TokenBucket(capacity=capacity, tokens=initial, refill_rate=refill_rate)
        # Fix: Explicitly set last_refill to match patched time (0.0)
        # Otherwise, last_refill gets real system time before patch is active
        bucket.last_refill = 0.0
        for i in range(steps):
            now_s[0] = float(i * dt_ms) / 1000.0
            if i % 2 == 0:
                bucket.refill()
            else:
                bucket.try_consume(tokens_to_consume)
            assert 0.0 <= bucket.tokens <= float(capacity)


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(
    challenge=st.binary(min_size=32, max_size=32),
    nonce=st.integers(min_value=0, max_value=10_000),
    low=st.integers(min_value=0, max_value=24),
    high=st.integers(min_value=0, max_value=24),
)
@settings(max_examples=200, deadline=None, derandomize=True)
def test_pow_verify_monotonic_in_difficulty(
    challenge: bytes,
    nonce: int,
    low: int,
    high: int,
) -> None:
    lo = min(low, high)
    hi = max(low, high)

    hi_ok = ProofOfWork(challenge=challenge, nonce=nonce, difficulty=hi).verify()
    lo_ok = ProofOfWork(challenge=challenge, nonce=nonce, difficulty=lo).verify()

    if hi_ok:
        assert lo_ok


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(
    version=st.text(alphabet=_ALPHABET, min_size=1, max_size=10),
    parameters=st.binary(min_size=1, max_size=64),
    metadata=st.dictionaries(
        keys=st.text(alphabet=_ALPHABET, min_size=1, max_size=5),
        values=st.text(alphabet=_ALPHABET, min_size=0, max_size=10),
        min_size=0,
        max_size=3,
    ),
    contributor_id=st.text(alphabet=_ALPHABET, min_size=1, max_size=12),
)
@settings(max_examples=200, deadline=None, derandomize=True)
def test_input_validator_accepts_constructed_valid_contribution(
    version: str,
    parameters: bytes,
    metadata: dict[str, str],
    contributor_id: str,
) -> None:
    limits = SecurityLimits(
        MAX_PACK_VERSION_LEN=10,
        MAX_PACK_PARAMETERS_SIZE=64,
        MAX_PACK_METADATA_KEYS=3,
        MAX_PACK_METADATA_KEY_LEN=5,
        MAX_PACK_METADATA_VALUE_SIZE=10,
        MAX_PACK_METADATA_SIZE=100,
        MAX_CONTRIBUTOR_ID_LEN=12,
        MAX_GOAL_ID_LEN=64,
        MAX_PROOFS_COUNT=2,
        MAX_PROOF_SIZE=8,
        MAX_TOTAL_PROOFS_SIZE=16,
    )

    validator = InputValidator(limits)
    pack = AgentPack(version=version, parameters=parameters, metadata=metadata)
    contrib = Contribution(
        goal_id=GoalID("VALID_GOAL"),
        agent_pack=pack,
        proofs={},
        contributor_id=contributor_id,
        seed=0,
    )

    assert validator.validate_agent_pack(pack).valid
    assert validator.validate_contribution(contrib).valid


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(proof_size=st.integers(min_value=9, max_value=64))
@settings(max_examples=50, deadline=None, derandomize=True)
def test_input_validator_rejects_oversized_proof(proof_size: int) -> None:
    limits = SecurityLimits(
        MAX_PROOFS_COUNT=2,
        MAX_PROOF_SIZE=8,
        MAX_TOTAL_PROOFS_SIZE=16,
    )
    validator = InputValidator(limits)

    pack = AgentPack(version="1", parameters=b"x")
    contrib = Contribution(
        goal_id=GoalID("VALID_GOAL"),
        agent_pack=pack,
        proofs={"p": b"x" * proof_size},
        contributor_id="c",
        seed=0,
    )

    result = validator.validate_contribution(contrib)
    assert not result.valid
    assert result.field is not None
    assert result.field.startswith("proofs.")


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(drift_s=st.integers(min_value=0, max_value=299))
@settings(max_examples=100, deadline=None, derandomize=True)
def test_sybil_challenge_stable_within_ttl(drift_s: int) -> None:
    sybil = SybilResistance(enabled=True)
    now_s = [0.0]

    with patch("idi.ian.security.time.time", lambda: now_s[0]), patch(
        "idi.ian.security.secrets.token_bytes", return_value=b"\x01" * 32
    ):
        c1 = sybil.get_challenge("contributor")
        now_s[0] = float(drift_s)
        c2 = sybil.get_challenge("contributor")

    assert c1 == c2


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(drift_s=st.integers(min_value=301, max_value=2000))
@settings(max_examples=50, deadline=None, derandomize=True)
def test_sybil_challenge_rotates_after_ttl(drift_s: int) -> None:
    sybil = SybilResistance(enabled=True)
    now_s = [0.0]

    def token_bytes(_n: int) -> bytes:
        return b"\x01" * 32 if now_s[0] == 0.0 else b"\x02" * 32

    with patch("idi.ian.security.time.time", lambda: now_s[0]), patch(
        "idi.ian.security.secrets.token_bytes", token_bytes
    ):
        c1 = sybil.get_challenge("contributor")
        now_s[0] = float(drift_s)
        c2 = sybil.get_challenge("contributor")

    assert c1 != c2


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(
    version=st.text(alphabet=_ALPHABET, min_size=1, max_size=10),
    parameters=st.binary(min_size=1, max_size=32),
    items=st.lists(
        st.tuples(
            st.text(alphabet=_ALPHABET, min_size=1, max_size=5),
            st.text(alphabet=_ALPHABET, min_size=0, max_size=10),
        ),
        min_size=0,
        max_size=5,
        unique_by=lambda kv: kv[0],
    ),
)
@settings(max_examples=200, deadline=None, derandomize=True)
def test_agent_pack_hash_invariant_under_metadata_order(
    version: str,
    parameters: bytes,
    items: list[tuple[str, str]],
) -> None:
    metadata_a: dict[str, str] = {}
    for k, v in items:
        metadata_a[k] = v

    metadata_b: dict[str, str] = {}
    for k, v in reversed(items):
        metadata_b[k] = v

    pack_a = AgentPack(version=version, parameters=parameters, metadata=metadata_a)
    pack_b = AgentPack(version=version, parameters=parameters, metadata=metadata_b)

    assert pack_a.pack_hash == pack_b.pack_hash


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(extra_keys=st.integers(min_value=2, max_value=10))
@settings(max_examples=50, deadline=None, derandomize=True)
def test_input_validator_rejects_too_many_metadata_keys(extra_keys: int) -> None:
    limits = SecurityLimits(MAX_PACK_METADATA_KEYS=1)
    validator = InputValidator(limits)

    metadata = {f"k{i}": "v" for i in range(extra_keys)}
    pack = AgentPack(version="1", parameters=b"x", metadata=metadata)

    result = validator.validate_agent_pack(pack)
    assert not result.valid
    assert result.field == "metadata"


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(capacity=st.integers(min_value=1, max_value=20), extra=st.integers(min_value=1, max_value=40))
@settings(max_examples=100, deadline=None, derandomize=True)
def test_rate_limiter_capacity_exhaustion_blocks_forever_when_no_refill(
    capacity: int,
    extra: int,
) -> None:
    limiter = RateLimiter(capacity=capacity, refill_rate=0.0)

    for _ in range(capacity):
        allowed, wait = limiter.check("c")
        assert allowed
        assert wait == 0.0

    for _ in range(extra):
        allowed, wait = limiter.check("c")
        assert not allowed
        assert wait == float("inf")


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(max_buckets=st.integers(min_value=1, max_value=20), extra=st.integers(min_value=1, max_value=40))
@settings(max_examples=100, deadline=None, derandomize=True)
def test_rate_limiter_never_exceeds_max_buckets(
    max_buckets: int,
    extra: int,
) -> None:
    limiter = RateLimiter(capacity=1, refill_rate=0.0, max_buckets=max_buckets)

    for i in range(max_buckets + extra):
        _allowed, _wait = limiter.check(f"c{i}")
        assert len(limiter._buckets) <= max_buckets


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(
    goal_id=st.text(alphabet=_ALPHABET, min_size=1, max_size=10),
    name=st.text(min_size=0, max_size=20),
    description=st.text(min_size=0, max_size=30),
)
@settings(max_examples=200, deadline=None, derandomize=True)
def test_input_validator_accepts_goal_spec_within_limits(
    goal_id: str,
    name: str,
    description: str,
) -> None:
    limits = SecurityLimits(
        MAX_GOAL_ID_LEN=10,
        MAX_GOAL_NAME_LEN=20,
        MAX_GOAL_DESCRIPTION_LEN=30,
    )
    validator = InputValidator(limits)
    spec = GoalSpec(goal_id=GoalID(goal_id), name=name, description=description)

    assert validator.validate_goal_spec(spec).valid


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(goal_id=st.text(alphabet=_ALPHABET, min_size=6, max_size=64))
@settings(max_examples=100, deadline=None, derandomize=True)
def test_input_validator_rejects_goal_spec_goal_id_too_long(goal_id: str) -> None:
    limits = SecurityLimits(MAX_GOAL_ID_LEN=5)
    validator = InputValidator(limits)
    spec = GoalSpec(goal_id=GoalID(goal_id), name="n", description="")

    result = validator.validate_goal_spec(spec)
    assert not result.valid
    assert result.field == "goal_id"


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(name=st.text(min_size=6, max_size=256))
@settings(max_examples=100, deadline=None, derandomize=True)
def test_input_validator_rejects_goal_spec_name_too_long(name: str) -> None:
    limits = SecurityLimits(MAX_GOAL_NAME_LEN=5)
    validator = InputValidator(limits)
    spec = GoalSpec(goal_id=GoalID("VALID_GOAL"), name=name, description="")

    result = validator.validate_goal_spec(spec)
    assert not result.valid
    assert result.field == "name"


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(description=st.text(min_size=6, max_size=512))
@settings(max_examples=100, deadline=None, derandomize=True)
def test_input_validator_rejects_goal_spec_description_too_long(description: str) -> None:
    limits = SecurityLimits(MAX_GOAL_DESCRIPTION_LEN=5)
    validator = InputValidator(limits)
    spec = GoalSpec(goal_id=GoalID("VALID_GOAL"), name="n", description=description)

    result = validator.validate_goal_spec(spec)
    assert not result.valid
    assert result.field == "description"


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(max_challenges=st.integers(min_value=1, max_value=10), extra=st.integers(min_value=1, max_value=20))
@settings(max_examples=100, deadline=None, derandomize=True)
def test_sybil_get_challenge_enforces_max_challenges_by_eviction(
    max_challenges: int,
    extra: int,
) -> None:
    total = max_challenges + extra
    now_s = [0.0]
    sybil = SybilResistance(enabled=True, difficulty=0, max_challenges=max_challenges)

    with patch("idi.ian.security.time.time", lambda: now_s[0]), patch(
        "idi.ian.security.secrets.token_bytes", return_value=b"\x01" * 32
    ):
        for i in range(total):
            now_s[0] = float(i)
            sybil.get_challenge(f"c{i}")

    assert len(sybil._pending_challenges) == max_challenges
    assert set(sybil._pending_challenges.keys()) == {
        f"c{i}" for i in range(total - max_challenges, total)
    }


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(
    now_s=st.integers(min_value=0, max_value=600).map(float),
    challenge_matches=st.booleans(),
    proof_difficulty=st.integers(min_value=0, max_value=10),
    proof_valid=st.booleans(),
)
@settings(max_examples=200, deadline=None, derandomize=True)
def test_sybil_verify_pow_matches_contract_when_enabled(
    now_s: float,
    challenge_matches: bool,
    proof_difficulty: int,
    proof_valid: bool,
) -> None:
    contributor_id = "c"
    required_difficulty = 8
    sybil = SybilResistance(enabled=True, difficulty=required_difficulty)

    with patch("idi.ian.security.secrets.token_bytes", return_value=b"\x11" * 32), patch(
        "idi.ian.security.time.time", return_value=0.0
    ):
        challenge = sybil.get_challenge(contributor_id)

    proof = ProofOfWork(
        challenge=challenge if challenge_matches else b"\x22" * 32,
        nonce=0,
        difficulty=proof_difficulty,
    )

    with patch("idi.ian.security.time.time", return_value=now_s), patch(
        "idi.ian.security.ProofOfWork.verify", return_value=proof_valid
    ):
        ok = sybil.verify_pow(contributor_id, proof)

    expected = (
        now_s < 300.0
        and challenge_matches
        and proof_difficulty >= required_difficulty
        and proof_valid
    )
    assert ok == expected

    if ok or now_s >= 300.0:
        assert contributor_id not in sybil._pending_challenges


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(key_len=st.integers(min_value=6, max_value=64))
@settings(max_examples=50, deadline=None, derandomize=True)
def test_input_validator_rejects_metadata_key_too_long(key_len: int) -> None:
    limits = SecurityLimits(MAX_PACK_METADATA_KEY_LEN=5, MAX_PACK_METADATA_KEYS=10)
    validator = InputValidator(limits)

    key = "k" * key_len
    pack = AgentPack(version="1", parameters=b"x", metadata={key: "v"})

    result = validator.validate_agent_pack(pack)
    assert not result.valid
    assert result.field is not None
    assert result.field.startswith("metadata.")


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(value_len=st.integers(min_value=11, max_value=256))
@settings(max_examples=50, deadline=None, derandomize=True)
def test_input_validator_rejects_metadata_value_too_large(value_len: int) -> None:
    limits = SecurityLimits(
        MAX_PACK_METADATA_VALUE_SIZE=10,
        MAX_PACK_METADATA_KEYS=10,
        MAX_PACK_METADATA_KEY_LEN=10,
        MAX_PACK_METADATA_SIZE=10_000,
    )
    validator = InputValidator(limits)

    value = "v" * value_len
    pack = AgentPack(version="1", parameters=b"x", metadata={"k": value})

    result = validator.validate_agent_pack(pack)
    assert not result.valid
    assert result.field is not None
    assert result.field.startswith("metadata.")


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(total_value_len=st.integers(min_value=20, max_value=256))
@settings(max_examples=50, deadline=None, derandomize=True)
def test_input_validator_rejects_total_metadata_size(total_value_len: int) -> None:
    limits = SecurityLimits(
        MAX_PACK_METADATA_SIZE=10,
        MAX_PACK_METADATA_KEYS=10,
        MAX_PACK_METADATA_KEY_LEN=10,
        MAX_PACK_METADATA_VALUE_SIZE=10_000,
    )
    validator = InputValidator(limits)

    pack = AgentPack(
        version="1",
        parameters=b"x",
        metadata={"a": "v" * total_value_len},
    )

    result = validator.validate_agent_pack(pack)
    assert not result.valid
    assert result.field == "metadata"
