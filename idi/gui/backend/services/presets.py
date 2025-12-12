"""Preset management service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PresetMeta:
    """Preset metadata."""
    id: str
    name: str
    description: str
    icon: str
    tags: tuple[str, ...]
    goal_spec_path: Path
    
    @property
    def difficulty(self) -> str:
        if "beginner" in self.tags:
            return "beginner"
        if "advanced" in self.tags:
            return "advanced"
        return "intermediate"


# Built-in preset definitions
BUILTIN_PRESETS: Dict[str, dict] = {
    "conservative_qagent": {
        "name": "Conservative Trader",
        "description": "Low-risk agent with stable returns and cautious exploration. Perfect for beginners.",
        "icon": "shield",
        "tags": ("beginner", "low-risk", "stable"),
    },
    "research_qagent": {
        "name": "Research Mode",
        "description": "Exploratory agent for testing new strategies across multiple environments.",
        "icon": "flask",
        "tags": ("intermediate", "experimental", "multi-env"),
    },
    "quick_test": {
        "name": "Quick Test",
        "description": "Fast iteration with minimal budget. Great for testing ideas.",
        "icon": "zap",
        "tags": ("beginner", "fast", "testing"),
    },
    "production": {
        "name": "Production Ready",
        "description": "Balanced configuration with proven stability. Ready for deployment.",
        "icon": "check-circle",
        "tags": ("advanced", "production", "stable"),
    },
}


class PresetService:
    """Service for managing presets."""
    
    def __init__(self) -> None:
        self._cache: Dict[str, PresetMeta] = {}
        self._examples_path = Path(__file__).parent.parent.parent.parent.parent / "examples" / "synth"
        self._discover()
    
    def _discover(self) -> None:
        """Discover available presets."""
        # Load from examples directory
        if self._examples_path.exists():
            for file in self._examples_path.glob("*_goal.json"):
                preset_id = file.stem.replace("_goal", "")
                builtin = BUILTIN_PRESETS.get(preset_id, {})
                
                self._cache[preset_id] = PresetMeta(
                    id=preset_id,
                    name=builtin.get("name", preset_id.replace("_", " ").title()),
                    description=builtin.get("description", "User-defined preset"),
                    icon=builtin.get("icon", "file"),
                    tags=tuple(builtin.get("tags", ())),
                    goal_spec_path=file,
                )
        
        # Add built-in presets that don't have files yet
        for preset_id, meta in BUILTIN_PRESETS.items():
            if preset_id not in self._cache:
                # Create in-memory preset
                self._cache[preset_id] = PresetMeta(
                    id=preset_id,
                    name=meta["name"],
                    description=meta["description"],
                    icon=meta["icon"],
                    tags=tuple(meta["tags"]),
                    goal_spec_path=Path("/dev/null"),  # No file
                )
    
    def list_all(self) -> List[PresetMeta]:
        """List all presets."""
        return list(self._cache.values())
    
    def get(self, preset_id: str) -> Optional[PresetMeta]:
        """Get preset by ID."""
        return self._cache.get(preset_id)
    
    def load_goal_spec(self, preset_id: str) -> Dict[str, Any]:
        """Load goal spec JSON for a preset."""
        preset = self._cache.get(preset_id)
        if not preset:
            return self._default_goal_spec()
        
        if preset.goal_spec_path.exists():
            return json.loads(preset.goal_spec_path.read_text())
        
        # Generate default based on preset type
        return self._generate_goal_spec(preset_id)
    
    def _default_goal_spec(self) -> Dict[str, Any]:
        """Default goal spec."""
        return {
            "agent_family": "qagent",
            "profiles": ["conservative"],
            "packs": {"include": ["qagent_base"], "extra": []},
            "objectives": [
                {"id": "avg_reward", "direction": "maximize"},
                {"id": "risk_stability", "direction": "maximize"},
            ],
            "training": {
                "envs": [{"id": "default", "weight": 1.0}],
                "curriculum": {"enabled": False},
                "budget": {
                    "max_agents": 8,
                    "max_generations": 3,
                    "max_episodes_per_agent": 64,
                    "wallclock_hours": 0.5,
                },
            },
            "eval_mode": "synthetic",
            "outputs": {"num_final_agents": 3, "bundle_format": "wire_v1"},
        }
    
    def _generate_goal_spec(self, preset_id: str) -> Dict[str, Any]:
        """Generate goal spec based on preset ID."""
        base = self._default_goal_spec()
        
        if preset_id == "quick_test":
            base["training"]["budget"]["max_agents"] = 4
            base["training"]["budget"]["max_generations"] = 2
            base["training"]["budget"]["max_episodes_per_agent"] = 32
            base["training"]["budget"]["wallclock_hours"] = 0.1
        elif preset_id == "production":
            base["profiles"] = ["conservative"]
            base["packs"]["include"] = ["qagent_base", "risk_conservative"]
            base["training"]["budget"]["max_agents"] = 16
            base["training"]["budget"]["max_generations"] = 4
            base["training"]["budget"]["max_episodes_per_agent"] = 128
            base["training"]["budget"]["wallclock_hours"] = 1.0
            base["eval_mode"] = "real"
        
        return base
