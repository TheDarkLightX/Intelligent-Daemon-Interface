"""Wizard controller - manages wizard state machine and data collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal
from enum import Enum

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec


class WizardStep(Enum):
    """Wizard step enumeration."""
    STRATEGY = "strategy"
    INPUTS = "inputs"
    LAYERS = "layers"
    SAFETY = "safety"
    REVIEW = "review"


@dataclass
class WizardData:
    """Data collected through wizard steps."""
    name: str = ""
    strategy: Literal["momentum", "mean_reversion", "regime_aware", "custom"] = "momentum"
    selected_inputs: Dict[str, bool] = field(default_factory=dict)
    num_layers: int = 1
    include_safety: bool = True
    include_communication: bool = False
    num_steps: int = 10


class WizardController:
    """Manages wizard state machine and data collection."""
    
    STEPS = [
        WizardStep.STRATEGY,
        WizardStep.INPUTS,
        WizardStep.LAYERS,
        WizardStep.SAFETY,
        WizardStep.REVIEW,
    ]
    
    def __init__(self):
        """Initialize wizard controller."""
        self.current_step_idx = 0
        self.data = WizardData()
        self._validation_errors: Dict[str, str] = {}
    
    @property
    def current_step(self) -> WizardStep:
        """Get current wizard step."""
        return self.STEPS[self.current_step_idx]
    
    @property
    def is_first_step(self) -> bool:
        """Check if on first step."""
        return self.current_step_idx == 0
    
    @property
    def is_last_step(self) -> bool:
        """Check if on last step."""
        return self.current_step_idx == len(self.STEPS) - 1
    
    @property
    def step_number(self) -> int:
        """Get current step number (1-indexed)."""
        return self.current_step_idx + 1
    
    @property
    def total_steps(self) -> int:
        """Get total number of steps."""
        return len(self.STEPS)
    
    def next(self, step_data: Dict[str, Any]) -> bool:
        """Move to next step with validation.
        
        Args:
            step_data: Data collected from current step
            
        Returns:
            True if step is valid and advanced, False otherwise
        """
        if not self._validate_step(self.current_step, step_data):
            return False
        
        # Update data
        self._update_data(step_data)
        
        # Advance step
        if not self.is_last_step:
            self.current_step_idx += 1
            self._validation_errors.clear()
        
        return True
    
    def prev(self) -> bool:
        """Move to previous step.
        
        Returns:
            True if moved back, False if already at first step
        """
        if self.is_first_step:
            return False
        
        self.current_step_idx -= 1
        self._validation_errors.clear()
        return True
    
    def _validate_step(self, step: WizardStep, data: Dict[str, Any]) -> bool:
        """Validate step data.
        
        Args:
            step: Current step
            data: Step data to validate
            
        Returns:
            True if valid, False otherwise
        """
        self._validation_errors.clear()
        
        if step == WizardStep.STRATEGY:
            if "strategy" not in data:
                self._validation_errors["strategy"] = "Strategy is required"
                return False
            if data["strategy"] not in ("momentum", "mean_reversion", "regime_aware", "custom"):
                self._validation_errors["strategy"] = "Invalid strategy"
                return False
        
        elif step == WizardStep.INPUTS:
            if "name" not in data or not data["name"].strip():
                self._validation_errors["name"] = "Agent name is required"
                return False
            if "selected_inputs" not in data or not data["selected_inputs"]:
                self._validation_errors["inputs"] = "At least one input must be selected"
                return False
        
        elif step == WizardStep.LAYERS:
            if "num_layers" not in data:
                self._validation_errors["layers"] = "Number of layers is required"
                return False
            if not isinstance(data["num_layers"], int) or data["num_layers"] < 1:
                self._validation_errors["layers"] = "Number of layers must be >= 1"
                return False
        
        return True
    
    def _update_data(self, step_data: Dict[str, Any]) -> None:
        """Update wizard data from step data."""
        if "name" in step_data:
            self.data.name = step_data["name"]
        if "strategy" in step_data:
            self.data.strategy = step_data["strategy"]
        if "selected_inputs" in step_data:
            self.data.selected_inputs = step_data["selected_inputs"]
        if "num_layers" in step_data:
            self.data.num_layers = step_data["num_layers"]
        if "include_safety" in step_data:
            self.data.include_safety = step_data["include_safety"]
        if "include_communication" in step_data:
            self.data.include_communication = step_data["include_communication"]
        if "num_steps" in step_data:
            self.data.num_steps = step_data["num_steps"]
    
    def get_validation_errors(self) -> Dict[str, str]:
        """Get current validation errors."""
        return self._validation_errors.copy()
    
    def generate_schema(self) -> AgentSchema:
        """Generate AgentSchema from collected wizard data.
        
        Returns:
            Complete AgentSchema ready for spec generation
        """
        # Build streams from selected inputs
        streams = []
        
        # Standard inputs based on strategy
        input_options = {
            "q_buy": StreamConfig(name="q_buy", stream_type="sbf"),
            "q_sell": StreamConfig(name="q_sell", stream_type="sbf"),
            "price_up": StreamConfig(name="price_up", stream_type="sbf"),
            "price_down": StreamConfig(name="price_down", stream_type="sbf"),
            "trend": StreamConfig(name="trend", stream_type="bv", width=4),
            "volume": StreamConfig(name="volume", stream_type="bv", width=4),
            "regime": StreamConfig(name="regime", stream_type="bv", width=5),
            "risk_budget_ok": StreamConfig(name="risk_budget_ok", stream_type="sbf"),
        }
        
        # Add selected inputs
        for input_name, selected in self.data.selected_inputs.items():
            if selected and input_name in input_options:
                streams.append(input_options[input_name])
        
        # Add required outputs
        streams.append(StreamConfig(name="position", stream_type="sbf", is_input=False))
        streams.append(StreamConfig(name="buy_signal", stream_type="sbf", is_input=False))
        streams.append(StreamConfig(name="sell_signal", stream_type="sbf", is_input=False))
        
        # Build logic blocks
        logic_blocks = []
        
        # FSM for position
        buy_input = "q_buy" if "q_buy" in self.data.selected_inputs else "price_up"
        sell_input = "q_sell" if "q_sell" in self.data.selected_inputs else "price_down"
        
        logic_blocks.append(
            LogicBlock(
                pattern="fsm",
                inputs=(buy_input, sell_input),
                output="position",
            )
        )
        
        # Buy/sell signals
        logic_blocks.append(
            LogicBlock(
                pattern="passthrough",
                inputs=(buy_input,),
                output="buy_signal",
            )
        )
        
        logic_blocks.append(
            LogicBlock(
                pattern="passthrough",
                inputs=(sell_input,),
                output="sell_signal",
            )
        )
        
        return AgentSchema(
            name=self.data.name or "unnamed_agent",
            strategy=self.data.strategy,
            streams=tuple(streams),
            logic_blocks=tuple(logic_blocks),
            num_steps=self.data.num_steps,
            include_mirrors=True,
        )
    
    def generate_spec(self) -> str:
        """Generate Tau spec from wizard data.
        
        Returns:
            Complete Tau spec as string
        """
        schema = self.generate_schema()
        return generate_tau_spec(schema)

