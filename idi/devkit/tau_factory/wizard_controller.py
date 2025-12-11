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
    strategy: Literal["momentum", "mean_reversion", "regime_aware", "custom", "ensemble"] = "momentum"
    selected_inputs: Dict[str, bool] = field(default_factory=dict)
    num_layers: int = 1
    include_safety: bool = True
    include_communication: bool = False
    num_steps: int = 10
    # Ensemble pattern options
    ensemble_pattern: Optional[str] = None  # "majority", "unanimous", "custom"
    ensemble_threshold: Optional[int] = None  # For majority: N-of-M
    ensemble_total: Optional[int] = None  # For majority: total agents
    custom_expression: Optional[str] = None  # For custom pattern


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
            if data["strategy"] not in ("momentum", "mean_reversion", "regime_aware", "custom", "ensemble"):
                self._validation_errors["strategy"] = "Invalid strategy"
                return False
            # Validate ensemble-specific data if ensemble strategy
            if data.get("strategy") == "ensemble":
                if "ensemble_pattern" not in data or not data["ensemble_pattern"]:
                    self._validation_errors["ensemble_pattern"] = "Ensemble pattern is required"
                    return False
                if data["ensemble_pattern"] == "majority":
                    if "ensemble_threshold" not in data or data["ensemble_threshold"] < 1:
                        self._validation_errors["ensemble_threshold"] = "Majority threshold must be >= 1"
                        return False
                    if "ensemble_total" not in data or data["ensemble_total"] < data["ensemble_threshold"]:
                        self._validation_errors["ensemble_total"] = "Total must be >= threshold"
                        return False
                elif data["ensemble_pattern"] == "custom":
                    if "custom_expression" not in data or not data["custom_expression"].strip():
                        self._validation_errors["custom_expression"] = "Custom expression is required"
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
        if "ensemble_pattern" in step_data:
            self.data.ensemble_pattern = step_data["ensemble_pattern"]
        if "ensemble_threshold" in step_data:
            self.data.ensemble_threshold = step_data["ensemble_threshold"]
        if "ensemble_total" in step_data:
            self.data.ensemble_total = step_data["ensemble_total"]
        if "custom_expression" in step_data:
            self.data.custom_expression = step_data["custom_expression"]
    
    def get_validation_errors(self) -> Dict[str, str]:
        """Get current validation errors."""
        return self._validation_errors.copy()
    
    def generate_schema(self) -> AgentSchema:
        """Generate AgentSchema from collected wizard data.
        
        Returns:
            Complete AgentSchema ready for spec generation
            
        Complexity: O(n) where n = number of selected inputs
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
        
        # Use set for O(1) membership testing instead of repeated list comprehensions
        stream_names: set[str] = {s.name for s in streams}
        
        # Determine buy/sell inputs for FSM
        buy_input = "q_buy" if self.data.selected_inputs.get("q_buy", False) else "price_up"
        sell_input = "q_sell" if self.data.selected_inputs.get("q_sell", False) else "price_down"
        
        # Ensure fallback inputs are added if not already selected (O(1) lookup)
        if buy_input not in stream_names and buy_input in input_options:
            streams.append(input_options[buy_input])
            stream_names.add(buy_input)
        if sell_input not in stream_names and sell_input in input_options:
            streams.append(input_options[sell_input])
            stream_names.add(sell_input)
        
        # Add required outputs
        streams.append(StreamConfig(name="position", stream_type="sbf", is_input=False))
        streams.append(StreamConfig(name="buy_signal", stream_type="sbf", is_input=False))
        streams.append(StreamConfig(name="sell_signal", stream_type="sbf", is_input=False))
        
        # Build logic blocks
        logic_blocks = []
        
        # Handle ensemble strategy
        if self.data.strategy == "ensemble":
            # Get agent inputs (at least 2 required for ensemble)
            agent_inputs = []
            for input_name, selected in self.data.selected_inputs.items():
                if selected and input_name in input_options:
                    agent_inputs.append(input_name)
            
            # Ensure at least 2 agents for ensemble (use stream_names set for O(1) lookup)
            if len(agent_inputs) < 2:
                # Add fallback agents using set-based membership check
                for fallback_agent in ("agent1", "agent2", "agent3"):
                    if len(agent_inputs) >= 3:
                        break
                    if fallback_agent not in stream_names:
                        streams.append(StreamConfig(name=fallback_agent, stream_type="sbf"))
                        stream_names.add(fallback_agent)
                        agent_inputs.append(fallback_agent)
            
            # Limit to total if specified
            if self.data.ensemble_total:
                agent_inputs = agent_inputs[:self.data.ensemble_total]
            
            # Add ensemble output
            ensemble_output_name = f"{self.data.ensemble_pattern}_vote"
            streams.append(StreamConfig(name=ensemble_output_name, stream_type="sbf", is_input=False))
            
            # Create ensemble logic block
            if self.data.ensemble_pattern == "majority":
                logic_blocks.append(
                    LogicBlock(
                        pattern="majority",
                        inputs=tuple(agent_inputs),
                        output=ensemble_output_name,
                        params={
                            "threshold": self.data.ensemble_threshold or (len(agent_inputs) // 2 + 1),
                            "total": self.data.ensemble_total or len(agent_inputs),
                        }
                    )
                )
            elif self.data.ensemble_pattern == "unanimous":
                logic_blocks.append(
                    LogicBlock(
                        pattern="unanimous",
                        inputs=tuple(agent_inputs),
                        output=ensemble_output_name,
                    )
                )
            elif self.data.ensemble_pattern == "custom":
                logic_blocks.append(
                    LogicBlock(
                        pattern="custom",
                        inputs=tuple(agent_inputs),
                        output=ensemble_output_name,
                        params={"expression": self.data.custom_expression or ""}
                    )
                )
            
            # Add position FSM using ensemble output
            streams.append(StreamConfig(name="position", stream_type="sbf", is_input=False))
            logic_blocks.append(
                LogicBlock(
                    pattern="fsm",
                    inputs=(ensemble_output_name, sell_input),
                    output="position",
                )
            )
        else:
            # Standard strategy logic
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

