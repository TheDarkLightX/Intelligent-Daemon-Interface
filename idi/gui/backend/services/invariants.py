"""Invariant checking service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class InvariantMeta:
    """Invariant metadata."""
    id: str
    label: str
    description: str
    formula: str
    safe_threshold: float


INVARIANTS: Dict[str, InvariantMeta] = {
    "I1": InvariantMeta(
        id="I1",
        label="State Size Bound",
        description="Q-table cannot exceed 2048 states to ensure tractable learning",
        formula="price_bins × inventory_bins ≤ 2048",
        safe_threshold=2048.0,
    ),
    "I2": InvariantMeta(
        id="I2",
        label="Discount Factor Bound",
        description="Discount must be at least 0.5 for learning stability",
        formula="discount_factor ≥ 0.5",
        safe_threshold=0.5,
    ),
    "I3": InvariantMeta(
        id="I3",
        label="Learning Rate Bound",
        description="Learning rate must not exceed 0.5 to prevent divergence",
        formula="learning_rate ≤ 0.5",
        safe_threshold=0.5,
    ),
    "I4": InvariantMeta(
        id="I4",
        label="Exploration Decay Bound",
        description="Exploration decay must be positive for convergence",
        formula="epsilon_decay_steps > 0",
        safe_threshold=1.0,
    ),
    "I5": InvariantMeta(
        id="I5",
        label="Budget Sanity",
        description="Training budget must allow meaningful exploration",
        formula="max_agents × max_episodes ≥ 64",
        safe_threshold=64.0,
    ),
}


class InvariantService:
    """Service for checking invariants."""
    
    def check_all(self, goal_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check all invariants for a goal spec."""
        results = []
        
        # Extract relevant values from goal spec
        training = goal_spec.get("training", {})
        budget = training.get("budget", {})
        
        # Default patch params (would come from patch definition in real use)
        patch_params = goal_spec.get("patch_params", {
            "num_price_bins": 10,
            "num_inventory_bins": 10,
            "learning_rate": 0.1,
            "discount_factor": 0.99,
            "epsilon_decay_steps": 1000,
        })
        
        # I1: State size
        state_size = patch_params.get("num_price_bins", 10) * patch_params.get("num_inventory_bins", 10)
        i1_ok = state_size <= 2048
        results.append({
            "id": "I1",
            "label": INVARIANTS["I1"].label,
            "ok": i1_ok,
            "message": f"State size {state_size} {'≤' if i1_ok else '>'} 2048",
            "value": float(state_size),
            "threshold": 2048.0,
        })
        
        # I2: Discount
        discount = patch_params.get("discount_factor", 0.99)
        i2_ok = discount >= 0.5
        results.append({
            "id": "I2",
            "label": INVARIANTS["I2"].label,
            "ok": i2_ok,
            "message": f"Discount {discount:.2f} {'≥' if i2_ok else '<'} 0.5",
            "value": discount,
            "threshold": 0.5,
        })
        
        # I3: Learning rate
        lr = patch_params.get("learning_rate", 0.1)
        i3_ok = lr <= 0.5
        results.append({
            "id": "I3",
            "label": INVARIANTS["I3"].label,
            "ok": i3_ok,
            "message": f"Learning rate {lr:.3f} {'≤' if i3_ok else '>'} 0.5",
            "value": lr,
            "threshold": 0.5,
        })
        
        # I4: Epsilon decay
        decay = patch_params.get("epsilon_decay_steps", 1000)
        i4_ok = decay > 0
        results.append({
            "id": "I4",
            "label": INVARIANTS["I4"].label,
            "ok": i4_ok,
            "message": f"Decay steps {decay} {'>' if i4_ok else '≤'} 0",
            "value": float(decay),
            "threshold": 1.0,
        })
        
        # I5: Budget sanity
        max_agents = budget.get("max_agents", 8)
        max_episodes = budget.get("max_episodes_per_agent", 64)
        budget_product = max_agents * max_episodes
        i5_ok = budget_product >= 64
        results.append({
            "id": "I5",
            "label": INVARIANTS["I5"].label,
            "ok": i5_ok,
            "message": f"Budget {max_agents}×{max_episodes}={budget_product} {'≥' if i5_ok else '<'} 64",
            "value": float(budget_product),
            "threshold": 64.0,
        })
        
        return results
    
    def get_descriptions(self) -> List[Dict[str, str]]:
        """Get human-readable descriptions of all invariants."""
        return [
            {
                "id": inv.id,
                "label": inv.label,
                "description": inv.description,
                "formula": inv.formula,
            }
            for inv in INVARIANTS.values()
        ]
