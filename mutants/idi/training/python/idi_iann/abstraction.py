"""State abstraction utilities (e.g., tile coding)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from .config import TileCoderConfig


@dataclass
class TileCoder:
    """Multi-tiling cartesian product encoder for compact Q-tables."""

    config: TileCoderConfig

    def __post_init__(self) -> None:
        self._dimensions = len(self.config.tile_sizes)
        self._offsets = [
            tuple((offset + shift) % size for offset, size in zip(self.config.offsets, self.config.tile_sizes))
            for shift in range(self.config.num_tilings)
        ]

    def encode(self, state: Sequence[int]) -> Tuple[int, ...]:
        if len(state) != self._dimensions:
            raise ValueError(f"TileCoder expected {self._dimensions} dims, got {len(state)}")
        tiles = []
        for offsets in self._offsets:
            code = 0
            stride = 1
            for value, size, offset in zip(state, self.config.tile_sizes, offsets):
                bucket = (value + offset) % size
                code += bucket * stride
                stride *= size
            tiles.append(code)
        return tuple(tiles)

