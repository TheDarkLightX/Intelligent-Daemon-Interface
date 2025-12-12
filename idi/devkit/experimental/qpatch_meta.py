from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class QPatchMeta:
    """Metadata for experimental QAgent patches.

    This is kept in a dedicated module to avoid circular imports between
    patch definitions and KRR planning utilities.
    """

    name: str
    description: str
    version: str
    tags: Tuple[str, ...] = ()
