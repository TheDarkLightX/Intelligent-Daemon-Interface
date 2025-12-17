"""
Invertible Bloom Lookup Table (IBLT) for bandwidth-optimal set reconciliation.

Security Controls:
- MAX_IBLT_CELLS prevents memory exhaustion DoS
- MAX_DECODE_ITERATIONS prevents infinite loop DoS
- SHA-256 hashing (not MD5/SHA1) for collision resistance
- secrets.token_bytes for unpredictable hash seeds
- Input validation on all external data

Based on: Goodrich & Mitzenmacher (2011), Bitcoin Erlay (BIP 330)

Author: DarkLightX
"""

from __future__ import annotations

from collections import deque
import hashlib
import hmac
import secrets
from dataclasses import dataclass, field

# Security: Bounded constants to prevent DoS
MAX_IBLT_CELLS = 10000
MAX_DECODE_ITERATIONS = MAX_IBLT_CELLS * 2
MIN_IBLT_CELLS = 10
DEFAULT_NUM_HASHES = 3
HASH_SIZE = 32  # SHA-256 output size


@dataclass
class IBLTCell:
    """
    Single cell in an IBLT.

    Invariants:
        - count can be negative (after subtraction)
        - key_sum and hash_sum are always HASH_SIZE bytes
    """
    count: int = 0
    key_sum: bytes = field(default_factory=lambda: b'\x00' * HASH_SIZE)
    hash_sum: bytes = field(default_factory=lambda: b'\x00' * HASH_SIZE)

    def is_pure(self) -> bool:
        """
        Check if cell is "pure" (contains exactly one element).

        A cell is pure if:
            - count == 1 or count == -1
            - hash_sum == SHA256(key_sum)
        """
        if abs(self.count) != 1:
            return False
        expected_hash = hashlib.sha256(self.key_sum).digest()
        return self.hash_sum == expected_hash

    def is_empty(self) -> bool:
        """Check if cell is empty (no elements)."""
        return (
            self.count == 0 and
            self.key_sum == b'\x00' * HASH_SIZE and
            self.hash_sum == b'\x00' * HASH_SIZE
        )


@dataclass
class IBLTConfig:
    """
    Configuration for IBLT operations.

    Security: All values are bounded to prevent DoS.
    """
    num_cells: int = 1000
    num_hashes: int = DEFAULT_NUM_HASHES
    # Security: Random seed for hash functions (unpredictable)
    hash_seed: bytes = field(default_factory=lambda: secrets.token_bytes(32))

    def __post_init__(self) -> None:
        """Validate configuration bounds."""
        # Security: Enforce bounds
        if not MIN_IBLT_CELLS <= self.num_cells <= MAX_IBLT_CELLS:
            raise ValueError(
                f"num_cells must be in [{MIN_IBLT_CELLS}, {MAX_IBLT_CELLS}], "
                f"got {self.num_cells}"
            )
        if not 1 <= self.num_hashes <= 10:
            raise ValueError(f"num_hashes must be in [1, 10], got {self.num_hashes}")
        if len(self.hash_seed) != 32:
            raise ValueError("hash_seed must be 32 bytes")


class IBLT:
    """
    Invertible Bloom Lookup Table for set reconciliation.

    Security features:
        - Bounded cell count to prevent memory exhaustion
        - SHA-256 hashing for collision resistance
        - Random hash seed for unpredictability
        - Input validation on all operations

    Usage:
        iblt_a = IBLT(config)
        for entry in entries_a:
            iblt_a.insert(entry)

        iblt_b = IBLT(config)  # Same config!
        for entry in entries_b:
            iblt_b.insert(entry)

        diff = iblt_a.subtract(iblt_b)
        only_a, only_b, success = diff.decode()
    """

    def __init__(self, config: IBLTConfig) -> None:
        """
        Initialize IBLT with given configuration.

        Preconditions:
            - config passes validation
        """
        self._config = config
        self._cells: list[IBLTCell] = [
            IBLTCell() for _ in range(config.num_cells)
        ]
        self._size = 0  # Track number of insertions for debugging

    @property
    def config(self) -> IBLTConfig:
        return self._config

    @property
    def num_cells(self) -> int:
        return self._config.num_cells

    def _hash_to_indices(self, key: bytes) -> list[int]:
        """
        Map key to cell indices using keyed hashing.

        Security: Uses SHA-256 with seed for unpredictability.
        """
        indices = []
        for i in range(self._config.num_hashes):
            # Security: Include hash index to get independent hashes
            h = hashlib.sha256(
                self._config.hash_seed +
                i.to_bytes(4, 'big') +
                key
            ).digest()
            # Convert to index
            idx = int.from_bytes(h[:8], 'big') % self._config.num_cells
            indices.append(idx)
        return indices

    def _key_hash(self, key: bytes) -> bytes:
        """Compute hash of key for verification."""
        return hashlib.sha256(key).digest()

    @staticmethod
    def _xor_bytes(a: bytes, b: bytes) -> bytes:
        """XOR two byte strings of equal length."""
        if len(a) != len(b):
            raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")
        return bytes(x ^ y for x, y in zip(a, b, strict=False))

    def insert(self, key: bytes) -> None:
        """
        Insert a key into the IBLT.

        Preconditions:
            - len(key) == HASH_SIZE (32 bytes)

        Security: Validates key size to prevent malformed input.
        """
        # Security: Validate input
        if len(key) != HASH_SIZE:
            raise ValueError(f"Key must be {HASH_SIZE} bytes, got {len(key)}")

        key_hash = self._key_hash(key)
        indices = self._hash_to_indices(key)

        for idx in indices:
            cell = self._cells[idx]
            cell.count += 1
            cell.key_sum = self._xor_bytes(cell.key_sum, key)
            cell.hash_sum = self._xor_bytes(cell.hash_sum, key_hash)

        self._size += 1

    def remove(self, key: bytes) -> None:
        """
        Remove a key from the IBLT.

        Note: Removing a non-existent key corrupts the IBLT.
        """
        if len(key) != HASH_SIZE:
            raise ValueError(f"Key must be {HASH_SIZE} bytes, got {len(key)}")

        key_hash = self._key_hash(key)
        indices = self._hash_to_indices(key)

        for idx in indices:
            cell = self._cells[idx]
            cell.count -= 1
            cell.key_sum = self._xor_bytes(cell.key_sum, key)
            cell.hash_sum = self._xor_bytes(cell.hash_sum, key_hash)

        self._size -= 1

    def subtract(self, other: IBLT) -> IBLT:
        """
        Compute difference IBLT: (self - other).

        After subtraction:
            - Positive count cells contain keys only in self
            - Negative count cells contain keys only in other

        Preconditions:
            - self and other have same configuration

        Security: Validates configuration match.
        """
        # Security: Ensure compatible configurations
        if self._config.num_cells != other._config.num_cells:
            raise ValueError("IBLT num_cells mismatch")
        if self._config.num_hashes != other._config.num_hashes:
            raise ValueError("IBLT num_hashes mismatch")
        if self._config.hash_seed != other._config.hash_seed:
            raise ValueError("IBLT hash_seed mismatch (required for compatible hashing)")

        result = IBLT(self._config)

        for i in range(self._config.num_cells):
            result._cells[i] = IBLTCell(
                count=self._cells[i].count - other._cells[i].count,
                key_sum=self._xor_bytes(self._cells[i].key_sum, other._cells[i].key_sum),
                hash_sum=self._xor_bytes(self._cells[i].hash_sum, other._cells[i].hash_sum),
            )

        return result

    def _apply_peel(
        self,
        work_cells: list[IBLTCell],
        *,
        key: bytes,
        count_sign: int,
        pure_queue: deque[int],
        in_queue: list[bool],
    ) -> None:
        """Apply a single peeling step for `key` across all affected cells."""
        key_hash = self._key_hash(key)
        for idx in self._hash_to_indices(key):
            work_cell = work_cells[idx]
            work_cell.count -= count_sign
            work_cell.key_sum = self._xor_bytes(work_cell.key_sum, key)
            work_cell.hash_sum = self._xor_bytes(work_cell.hash_sum, key_hash)

            if in_queue[idx]:
                continue
            if work_cell.is_pure():
                pure_queue.append(idx)
                in_queue[idx] = True

    def decode(self) -> tuple[set[bytes], set[bytes], bool]:
        """
        Decode IBLT to recover set differences.

        Returns:
            (keys_only_in_self, keys_only_in_other, success)

        Security:
            - Bounded iterations to prevent DoS
            - Returns partial results on failure

        Algorithm:
            1. Find pure cells (count = ±1, hash matches)
            2. Extract key from pure cell
            3. Remove key from all cells it maps to
            4. Repeat until no pure cells or max iterations
            5. Success if all cells empty
        """
        # Work on a copy to preserve original
        work_cells = [IBLTCell(c.count, c.key_sum, c.hash_sum) for c in self._cells]

        only_in_self: set[bytes] = set()
        only_in_other: set[bytes] = set()
        output_for_sign = {1: only_in_self, -1: only_in_other}

        # O(n) peeling decode: maintain a queue of pure cells.
        pure_queue: deque[int] = deque()
        in_queue = [False] * len(work_cells)
        for idx, cell in enumerate(work_cells):
            if cell.is_pure():
                pure_queue.append(idx)
                in_queue[idx] = True

        iterations = 0
        while pure_queue and iterations < MAX_DECODE_ITERATIONS:
            idx = pure_queue.popleft()
            in_queue[idx] = False
            iterations += 1

            cell = work_cells[idx]
            if not cell.is_pure():
                continue

            key = cell.key_sum
            count_sign = cell.count  # (+1 or -1)

            output_for_sign[count_sign].add(key)
            self._apply_peel(
                work_cells,
                key=key,
                count_sign=count_sign,
                pure_queue=pure_queue,
                in_queue=in_queue,
            )

        # Check if fully decoded
        success = all(c.is_empty() for c in work_cells)

        return only_in_self, only_in_other, success

    def serialize(self, auth_key: bytes | None = None) -> bytes:
        """
        Serialize IBLT for network transmission.

        Format: config_hash(32) + num_cells(4) + num_hashes(4) + cells(...) + [hmac(32)]

        Security: If auth_key is provided, appends HMAC-SHA256 for authentication.
        This prevents tampering by malicious peers.

        Args:
            auth_key: Optional 32-byte key for HMAC authentication
        """
        # Config identifier (for validation on receive)
        config_id = hashlib.sha256(
            self._config.hash_seed +
            self._config.num_cells.to_bytes(4, 'big') +
            self._config.num_hashes.to_bytes(4, 'big')
        ).digest()

        data = bytearray()
        data.extend(config_id)
        data.extend(self._config.num_cells.to_bytes(4, 'big'))
        data.extend(self._config.num_hashes.to_bytes(4, 'big'))

        for cell in self._cells:
            data.extend(cell.count.to_bytes(8, 'big', signed=True))
            data.extend(cell.key_sum)
            data.extend(cell.hash_sum)

        # Security: Add HMAC if auth_key provided
        if auth_key is not None:
            if len(auth_key) != 32:
                raise ValueError("auth_key must be 32 bytes")
            mac = hmac.new(auth_key, bytes(data), hashlib.sha256).digest()
            data.extend(mac)

        return bytes(data)

    @classmethod
    def deserialize(
        cls,
        data: bytes,
        config: IBLTConfig,
        auth_key: bytes | None = None,
    ) -> IBLT:
        """
        Deserialize IBLT from network data.

        Security:
            - Validates config_id matches
            - Validates data length
            - Bounded cell count
            - HMAC verification if auth_key provided

        Args:
            data: Serialized IBLT bytes
            config: Expected IBLT configuration
            auth_key: Optional 32-byte key for HMAC verification

        Raises:
            ValueError: On validation failure or HMAC mismatch
        """
        # Security: Minimum length check
        header_size = 32 + 4 + 4  # config_id + num_cells + num_hashes
        hmac_size = 32 if auth_key else 0
        if len(data) < header_size + hmac_size:
            raise ValueError(f"Data too short: {len(data)} < {header_size + hmac_size}")

        # Security: Verify HMAC before any other processing
        if auth_key is not None:
            if len(auth_key) != 32:
                raise ValueError("auth_key must be 32 bytes")
            received_mac = data[-32:]
            data_without_mac = data[:-32]
            expected_mac = hmac.new(auth_key, data_without_mac, hashlib.sha256).digest()
            if not hmac.compare_digest(received_mac, expected_mac):
                raise ValueError("HMAC verification failed: data may be tampered")
            data = data_without_mac  # Continue with authenticated data

        if len(data) < header_size:
            raise ValueError(f"Data too short: {len(data)} < {header_size}")

        # Extract and validate config
        config_id = data[:32]
        num_cells = int.from_bytes(data[32:36], 'big')
        num_hashes = int.from_bytes(data[36:40], 'big')

        # Security: Validate config matches
        expected_config_id = hashlib.sha256(
            config.hash_seed +
            config.num_cells.to_bytes(4, 'big') +
            config.num_hashes.to_bytes(4, 'big')
        ).digest()

        if config_id != expected_config_id:
            raise ValueError("Config mismatch: IBLT was created with different config")

        if num_cells != config.num_cells or num_hashes != config.num_hashes:
            raise ValueError("Config parameters mismatch")

        # Security: Validate data length
        cell_size = 8 + HASH_SIZE + HASH_SIZE  # count + key_sum + hash_sum
        expected_len = header_size + num_cells * cell_size
        if len(data) != expected_len:
            raise ValueError(f"Data length mismatch: {len(data)} != {expected_len}")

        # Create IBLT and populate cells
        iblt = cls(config)
        offset = header_size

        for i in range(num_cells):
            count = int.from_bytes(data[offset:offset+8], 'big', signed=True)
            offset += 8
            key_sum = data[offset:offset+HASH_SIZE]
            offset += HASH_SIZE
            hash_sum = data[offset:offset+HASH_SIZE]
            offset += HASH_SIZE

            iblt._cells[i] = IBLTCell(count, key_sum, hash_sum)

        return iblt


def estimate_iblt_size(expected_diff: int, overhead: float = 1.5) -> int:
    """
    Estimate number of IBLT cells needed for expected difference.

    Args:
        expected_diff: Expected number of differing elements
        overhead: Multiplier for reliability (1.5 = 50% overhead)

    Returns:
        Number of cells (bounded by MAX_IBLT_CELLS)

    Security: Result is always bounded.
    """
    cells = int(expected_diff * overhead * DEFAULT_NUM_HASHES)
    return max(MIN_IBLT_CELLS, min(cells, MAX_IBLT_CELLS))
