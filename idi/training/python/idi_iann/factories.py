"""Factory functions for creating agents and environments from configs."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .config import TrainingConfig
from .env import SyntheticMarketEnv
from .crypto_env import CryptoMarket, MarketParams
from .policy import LookupPolicy
from .trainer import QTrainer
from .strategies import ActionStrategy, EpsilonGreedyStrategy


@runtime_checkable
class Environment(Protocol):
    """Protocol for environment interface."""

    ACTIONS: tuple[str, ...]

    def reset(self) -> object:
        """Reset environment to initial state."""
        ...

    def step(self, action: str) -> tuple[object, float] | tuple[object, float, dict]:
        """Execute action and return (obs, reward) or (obs, reward, info)."""
        ...


def create_environment(
    config: TrainingConfig,
    use_crypto: bool = False,
    seed: int = 0,
    market_params: MarketParams | None = None,
) -> Environment:
    """Create an environment from config.

    Args:
        config: Training configuration
        use_crypto: If True, use CryptoMarket; otherwise SyntheticMarketEnv
        seed: Random seed
        market_params: Optional market parameters for crypto env

    Returns:
        Environment instance
    """
    if use_crypto:
        mp = market_params or MarketParams()
        mp.seed = seed
        return CryptoMarket(mp)
    return SyntheticMarketEnv(config.quantizer, config.rewards, seed=seed)


def create_policy() -> LookupPolicy:
    """Create a new lookup policy.

    Returns:
        Empty LookupPolicy instance
    """
    return LookupPolicy()


def create_trainer(
    config: TrainingConfig,
    strategy: ActionStrategy | None = None,
    use_crypto_env: bool = False,
    seed: int | None = 0,
    market_params: MarketParams | None = None,
) -> QTrainer:
    """Create a QTrainer from config.

    Args:
        config: Training configuration
        strategy: Optional exploration strategy (defaults to EpsilonGreedyStrategy)
        use_crypto_env: If True, use CryptoMarket
        seed: Random seed (None for random)
        market_params: Optional market parameters for crypto env

    Returns:
        Configured QTrainer instance
    """
    def env_factory(cfg: TrainingConfig, use_crypto: bool, s: int) -> Environment:
        return create_environment(cfg, use_crypto, s, market_params)

    return QTrainer(
        config=config,
        strategy=strategy or EpsilonGreedyStrategy(),
        use_crypto_env=use_crypto_env,
        seed=seed,
        env_factory=env_factory,
    )

