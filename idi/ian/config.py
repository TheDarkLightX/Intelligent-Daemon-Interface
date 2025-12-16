"""
IAN Configuration Module - Centralized configuration management.

Provides:
1. Hierarchical configuration with defaults
2. Environment variable overrides (IAN_* prefix)
3. Config file loading (TOML/JSON)
4. Validation on startup
5. Type-safe access

Configuration Hierarchy (highest to lowest priority):
1. Environment variables
2. Config file
3. Default values

Example:
    # Load config
    config = IANConfig.load()
    
    # Access values
    print(config.coordinator.leaderboard_capacity)
    print(config.security.rate_limit_tokens)
    
    # Override with environment
    # IAN_COORDINATOR_LEADERBOARD_CAPACITY=100
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Sections
# =============================================================================

@dataclass
class CoordinatorConfig:
    """Coordinator configuration."""
    leaderboard_capacity: int = 100
    use_pareto: bool = False
    expected_contributions: int = 100_000
    bloom_fp_rate: float = 0.01
    
    def __post_init__(self):
        if self.leaderboard_capacity <= 0:
            raise ValueError("leaderboard_capacity must be positive")
        if not (0 < self.bloom_fp_rate < 1):
            raise ValueError("bloom_fp_rate must be in (0, 1)")


@dataclass
class SecurityConfig:
    """Security configuration."""
    # Input limits
    max_pack_version_len: int = 64
    max_pack_parameters_size: int = 10 * 1024 * 1024  # 10 MB
    max_pack_metadata_size: int = 64 * 1024  # 64 KB
    max_contributor_id_len: int = 128
    max_proofs_count: int = 16
    max_proof_size: int = 1 * 1024 * 1024  # 1 MB
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_tokens: int = 10
    rate_limit_refill_per_second: float = 0.1
    
    # Proof of work
    pow_enabled: bool = False
    pow_difficulty: int = 20
    
    # Timing
    min_process_time_ms: float = 100.0


@dataclass  
class TauConfig:
    """Tau Net integration configuration."""
    enabled: bool = True
    endpoint: str = "http://localhost:8080"
    commit_interval_seconds: int = 300
    commit_threshold_contributions: int = 100
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    upgrade_cooldown_hours: int = 24
    governance_quorum: int = 3


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    timeout_seconds: int = 60
    max_memory_mb: int = 512
    max_episodes: int = 100
    max_steps_per_episode: int = 1000
    sandbox_enabled: bool = True
    subprocess_enabled: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"  # "json" or "text"
    file: Optional[str] = None
    max_size_mb: int = 100
    backup_count: int = 5
    include_timestamps: bool = True


@dataclass
class MetricsConfig:
    """Metrics configuration."""
    enabled: bool = True
    port: int = 9090
    path: str = "/metrics"
    namespace: str = "ian"


@dataclass
class StorageConfig:
    """Storage configuration."""
    data_dir: str = "./data/ian"
    checkpoint_interval: int = 1000
    max_checkpoints: int = 10


# =============================================================================
# Main Configuration
# =============================================================================

@dataclass
class IANConfig:
    """
    Main IAN configuration.
    
    Combines all configuration sections into a single object.
    """
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    tau: TauConfig = field(default_factory=TauConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    @classmethod
    def load(
        cls,
        config_file: Optional[Union[str, Path]] = None,
        env_prefix: str = "IAN",
    ) -> "IANConfig":
        """
        Load configuration with hierarchy: env vars > config file > defaults.
        
        Args:
            config_file: Path to config file (TOML or JSON)
            env_prefix: Prefix for environment variables
            
        Returns:
            Loaded and validated configuration
        """
        # Start with defaults
        config_dict: Dict[str, Any] = {}
        
        # Load from file if provided
        if config_file:
            config_dict = cls._load_file(Path(config_file))
        
        # Override with environment variables
        config_dict = cls._apply_env_overrides(config_dict, env_prefix)
        
        # Build config object
        return cls._from_dict(config_dict)
    
    @classmethod
    def _load_file(cls, path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return {}
        
        content = path.read_text()
        
        if path.suffix == ".json":
            return json.loads(content)
        elif path.suffix == ".toml":
            try:
                import tomllib
                return tomllib.loads(content)
            except ImportError:
                try:
                    import toml
                    return toml.loads(content)
                except ImportError:
                    logger.warning("TOML parsing not available, install tomllib or toml")
                    return {}
        elif path.suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError:
                logger.warning("YAML parsing not available, install PyYAML")
                return {}

            parsed = yaml.safe_load(content)
            if isinstance(parsed, dict):
                return parsed
            logger.warning("YAML config must be a mapping at top level")
            return {}
        else:
            logger.warning(f"Unknown config file format: {path.suffix}")
            return {}
    
    @classmethod
    def _apply_env_overrides(cls, config: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        for key, value in os.environ.items():
            if not key.startswith(f"{prefix}_"):
                continue
            
            # Parse key: IAN_COORDINATOR_LEADERBOARD_CAPACITY -> coordinator.leaderboard_capacity
            parts = key[len(prefix) + 1:].lower().split("_")
            
            if len(parts) < 2:
                continue
            
            section = parts[0]
            field_name = "_".join(parts[1:])
            
            # Ensure section exists
            if section not in config:
                config[section] = {}
            
            # Type conversion
            config[section][field_name] = cls._parse_env_value(value)
        
        return config
    
    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # String
        return value
    
    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> "IANConfig":
        """Build config object from dictionary."""
        return cls(
            coordinator=CoordinatorConfig(**config_dict.get("coordinator", {})),
            security=SecurityConfig(**config_dict.get("security", {})),
            tau=TauConfig(**config_dict.get("tau", {})),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
            logging=LoggingConfig(**config_dict.get("logging", {})),
            metrics=MetricsConfig(**config_dict.get("metrics", {})),
            storage=StorageConfig(**config_dict.get("storage", {})),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == ".json":
            path.write_text(self.to_json())
        elif path.suffix == ".toml":
            try:
                import toml
                path.write_text(toml.dumps(self.to_dict()))
            except ImportError:
                # Fallback to JSON
                path.with_suffix(".json").write_text(self.to_json())
                logger.warning("TOML not available, saved as JSON")
        else:
            path.write_text(self.to_json())
    
    def validate(self) -> None:
        """Validate configuration."""
        # Coordinator validation happens in __post_init__
        
        # Security validation
        if self.security.rate_limit_tokens <= 0:
            raise ValueError("rate_limit_tokens must be positive")
        
        if self.security.pow_difficulty < 0:
            raise ValueError("pow_difficulty must be non-negative")
        
        # Tau validation
        if self.tau.commit_interval_seconds <= 0:
            raise ValueError("commit_interval_seconds must be positive")
        
        # Evaluation validation
        if self.evaluation.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        
        # Logging validation
        if self.logging.level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"Invalid logging level: {self.logging.level}")
        
        if self.logging.format not in ("json", "text"):
            raise ValueError(f"Invalid logging format: {self.logging.format}")


# =============================================================================
# Global Config Instance
# =============================================================================

_global_config: Optional[IANConfig] = None


def get_config() -> IANConfig:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = IANConfig.load()
    return _global_config


def set_config(config: IANConfig) -> None:
    """Set global configuration instance."""
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset global configuration to None (forces reload)."""
    global _global_config
    _global_config = None
