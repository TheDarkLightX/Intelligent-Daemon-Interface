"""Macro control service.

Macros are high-level controls that map to multiple underlying goal spec fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class MacroDefinition:
    """Definition of a macro control."""
    id: str
    label: str
    description: str
    default: float
    effects: List[str]
    apply_fn: Callable[[float, Dict[str, Any]], Dict[str, Any]]


def _apply_risk_appetite(value: float, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Apply risk appetite macro.
    
    value: 0.0 (ultra-conservative) to 1.0 (aggressive)
    """
    spec = spec.copy()
    
    # Adjust learning rate cap
    # Conservative: 0.01-0.05, Aggressive: 0.05-0.2
    base_lr = 0.01 + value * 0.19
    
    # Adjust exploration
    # Conservative: 0.1-0.3, Aggressive: 0.4-0.8
    eps_start = 0.1 + value * 0.7
    
    # Adjust profiles
    if value < 0.3:
        profiles = ["conservative"]
        packs = ["qagent_base", "risk_conservative"]
    elif value < 0.7:
        profiles = ["conservative"]
        packs = ["qagent_base"]
    else:
        profiles = ["conservative", "experimental"]
        packs = ["qagent_base", "risk_moderate"]
    
    spec["profiles"] = profiles
    spec["packs"] = {"include": packs, "extra": spec.get("packs", {}).get("extra", [])}
    
    # Store derived values for patch generation
    spec.setdefault("patch_params", {})
    spec["patch_params"]["learning_rate"] = round(base_lr, 4)
    spec["patch_params"]["epsilon_start"] = round(eps_start, 2)
    
    return spec


def _apply_exploration_intensity(value: float, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Apply exploration intensity macro.
    
    value: 0.0 (exploit) to 1.0 (explore heavily)
    """
    spec = spec.copy()
    
    # Epsilon schedule
    eps_start = 0.1 + value * 0.7  # 0.1 to 0.8
    eps_end = 0.01 + value * 0.14  # 0.01 to 0.15
    
    # Beam width (more exploration = wider beam)
    budget = spec.get("training", {}).get("budget", {})
    base_agents = budget.get("max_agents", 8)
    adjusted_agents = int(base_agents * (0.5 + value))
    
    spec.setdefault("training", {}).setdefault("budget", {})
    spec["training"]["budget"]["max_agents"] = max(4, adjusted_agents)
    
    spec.setdefault("patch_params", {})
    spec["patch_params"]["epsilon_start"] = round(eps_start, 2)
    spec["patch_params"]["epsilon_end"] = round(eps_end, 3)
    
    return spec


def _apply_training_time(value: float, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Apply training time macro.
    
    value: 0.0 (quick) to 1.0 (thorough)
    """
    spec = spec.copy()
    
    # Map to wallclock hours: 0.1h to 4h
    hours = 0.1 + value * 3.9
    
    # Map to episodes: 32 to 512
    episodes = int(32 + value * 480)
    
    # Map to generations: 2 to 8
    generations = int(2 + value * 6)
    
    spec.setdefault("training", {}).setdefault("budget", {})
    spec["training"]["budget"]["wallclock_hours"] = round(hours, 2)
    spec["training"]["budget"]["max_episodes_per_agent"] = episodes
    spec["training"]["budget"]["max_generations"] = generations
    
    return spec


def _apply_conservatism(value: float, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Apply conservatism macro.
    
    value: 0.0 (experimental) to 1.0 (ultra-safe)
    """
    spec = spec.copy()
    
    # Discount factor: experimental allows lower, conservative requires higher
    discount = 0.7 + value * 0.25  # 0.7 to 0.95
    
    # State size: conservative uses smaller state space
    if value > 0.7:
        bins = 8
    elif value > 0.4:
        bins = 10
    else:
        bins = 16
    
    spec.setdefault("patch_params", {})
    spec["patch_params"]["discount_factor"] = round(discount, 3)
    spec["patch_params"]["num_price_bins"] = bins
    spec["patch_params"]["num_inventory_bins"] = bins
    
    return spec


def _apply_stability_reward(value: float, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Apply stability vs reward emphasis macro.
    
    value: 0.0 (max reward) to 1.0 (max stability)
    """
    spec = spec.copy()
    
    objectives = []
    
    # Always include reward
    objectives.append({"id": "avg_reward", "direction": "maximize"})
    
    # Add stability with weight based on value
    if value > 0.3:
        objectives.append({"id": "risk_stability", "direction": "maximize"})
    
    if value > 0.6:
        objectives.append({"id": "min_reward", "direction": "maximize"})
    
    if value < 0.3:
        # Low stability emphasis - allow complexity minimization
        objectives.append({"id": "complexity", "direction": "minimize"})
    
    spec["objectives"] = objectives
    
    return spec


# Define all macros
MACROS: Dict[str, MacroDefinition] = {
    "risk_appetite": MacroDefinition(
        id="risk_appetite",
        label="Risk Appetite",
        description="How much risk the agent should take. Higher = more aggressive strategies.",
        default=0.3,
        effects=["Learning rate", "Exploration bounds", "Risk packs"],
        apply_fn=_apply_risk_appetite,
    ),
    "exploration_intensity": MacroDefinition(
        id="exploration_intensity",
        label="Exploration Intensity",
        description="How much the agent explores vs exploits known strategies.",
        default=0.4,
        effects=["Epsilon schedule", "Search beam width"],
        apply_fn=_apply_exploration_intensity,
    ),
    "training_time": MacroDefinition(
        id="training_time",
        label="Training Time",
        description="How long to train. Longer = more thorough but slower.",
        default=0.3,
        effects=["Episodes", "Generations", "Wallclock limit"],
        apply_fn=_apply_training_time,
    ),
    "conservatism": MacroDefinition(
        id="conservatism",
        label="Conservatism",
        description="How cautious the agent configuration should be.",
        default=0.6,
        effects=["Discount factor", "State space size"],
        apply_fn=_apply_conservatism,
    ),
    "stability_reward": MacroDefinition(
        id="stability_reward",
        label="Stability vs Reward",
        description="Balance between maximizing returns and minimizing volatility.",
        default=0.5,
        effects=["Optimization objectives"],
        apply_fn=_apply_stability_reward,
    ),
}


class MacroService:
    """Service for applying macro controls."""
    
    def list_all(self) -> List[MacroDefinition]:
        """List all available macros."""
        return list(MACROS.values())
    
    def apply_all(
        self, 
        macro_values: Dict[str, float], 
        base_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply all macro values to a goal spec."""
        result = base_spec.copy()
        
        for macro_id, value in macro_values.items():
            macro = MACROS.get(macro_id)
            if macro:
                result = macro.apply_fn(value, result)
        
        return result
    
    def preview_changes(
        self,
        macro_values: Dict[str, float],
        base_spec: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Preview what changes macros would make."""
        changes = []
        
        for macro_id, value in macro_values.items():
            macro = MACROS.get(macro_id)
            if not macro:
                continue
            
            # Get before/after
            before = base_spec.copy()
            after = macro.apply_fn(value, before)
            
            # Find differences
            for effect in macro.effects:
                changes.append({
                    "macro": macro_id,
                    "effect": effect,
                    "value": value,
                    "description": f"{macro.label} at {value:.0%} affects {effect}",
                })
        
        return changes
