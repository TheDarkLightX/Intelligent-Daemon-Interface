"""Pure preflight decisions for the IAN imperative bridge.

These checks do not establish signature validity, Merkle membership, consensus
or finality. Cryptographic and governance evidence is supplied by the caller.
"""

from dataclasses import dataclass
from math import isfinite

U64_MAX = (1 << 64) - 1


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str = ""


def u64(value):
    return type(value) is int and 0 <= value <= U64_MAX


def hash32(value):
    return isinstance(value, bytes) and len(value) == 32


def registration(registered, name, spec_hash):
    if registered:
        return Decision(False, "Goal is already registered")
    if not isinstance(name, str) or not name.strip():
        return Decision(False, "Goal name is empty")
    if not hash32(spec_hash) or not any(spec_hash):
        return Decision(False, "Invalid goal specification hash")
    return Decision(True)


def log_commit(
    registered, log_root, log_size, previous_size, leaderboard_root, leaderboard_size
):
    if not registered:
        return Decision(False, "Goal is not registered")
    if not hash32(log_root) or not hash32(leaderboard_root):
        return Decision(False, "Roots must be 32 bytes")
    if not all(u64(x) for x in (log_size, previous_size, leaderboard_size)):
        return Decision(False, "Sizes must be unsigned 64-bit integers")
    if log_size < previous_size:
        return Decision(False, "Log size cannot decrease")
    if leaderboard_size > log_size:
        return Decision(False, "Leaderboard cannot exceed the log")
    return Decision(True)


def upgrade(
    registered,
    pack_hash,
    log_root,
    log_index,
    score,
    now,
    last,
    cooldown,
    count,
    governance_required,
    governance_approved,
):
    if not registered:
        return Decision(False, "Goal is not registered")
    if not hash32(pack_hash) or not any(pack_hash) or not hash32(log_root):
        return Decision(False, "Invalid policy or log hash")
    if (
        not all(u64(x) for x in (log_index, now, last, cooldown, count))
        or count == U64_MAX
    ):
        return Decision(False, "Invalid or exhausted 64-bit counter")
    try:
        finite_score = not isinstance(score, bool) and isfinite(score)
    except (TypeError, ValueError, OverflowError):
        finite_score = False
    if not isinstance(score, (int, float)) or not finite_score:
        return Decision(False, "Score must be finite")
    if now < last or now - last < cooldown:
        return Decision(False, "Upgrade cooldown not elapsed")
    if governance_required and not governance_approved:
        return Decision(False, "Verified governance approval is required")
    return Decision(True)


def half_slash(bond):
    """Conserve every unit: reward and burn round down; treasury gets remainder."""
    if not u64(bond):
        raise ValueError("Bond must be an unsigned 64-bit integer")
    slash = bond // 2
    reward = slash // 4
    burn = slash // 4
    treasury = slash - reward - burn
    return (bond - slash, reward, burn, treasury)


def deposit(bond, amount, maximum):
    if not all(u64(x) for x in (bond, amount, maximum)):
        raise ValueError("Amounts must be unsigned 64-bit integers")
    return (
        bond + amount
        if amount > 0 and bond <= maximum and amount <= maximum - bond
        else bond
    )
