"""Wizard GUI - child-friendly tkinter interface for agent generation."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path
from typing import Optional

from idi.devkit.tau_factory.wizard_controller import WizardController, WizardStep


class StepFrame(ttk.Frame):
    """Base class for wizard step frames."""
    
    def __init__(self, parent, controller, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.controller = controller
        self.setup_ui()
    
    def setup_ui(self):
        """Override to set up step-specific UI."""
        pass
    
    def get_data(self) -> dict:
        """Override to return step data."""
        return {}
    
    def validate(self) -> bool:
        """Override to validate step data."""
        return True


class StrategyStepFrame(StepFrame):
    """Step 1: Strategy selection."""
    
    def setup_ui(self):
        """Set up strategy selection UI."""
        ttk.Label(
            self,
            text="Pick Your Agent's Strategy",
            font=("Arial", 16, "bold"),
        ).pack(pady=20)
        
        ttk.Label(
            self,
            text="How should your agent trade?",
            font=("Arial", 12),
        ).pack(pady=10)
        
        self.strategy_var = tk.StringVar(value="momentum")
        
        strategies = [
            ("momentum", "Follow the Trend 📈\nBuy when prices go up"),
            ("mean_reversion", "Buy Low, Sell High 📊\nBuy dips, sell spikes"),
            ("regime_aware", "Smart Adaptation 🧠\nChange strategy with market"),
        ]
        
        for value, description in strategies:
            frame = ttk.Frame(self)
            frame.pack(fill=tk.X, padx=40, pady=10)
            
            radio = ttk.Radiobutton(
                frame,
                variable=self.strategy_var,
                value=value,
                text=description,
                font=("Arial", 11),
            )
            radio.pack(anchor=tk.W)
    
    def get_data(self) -> dict:
        """Get strategy selection."""
        return {"strategy": self.strategy_var.get()}


class InputsStepFrame(StepFrame):
    """Step 2: Input selection."""
    
    def setup_ui(self):
        """Set up input selection UI."""
        ttk.Label(
            self,
            text="What Should Your Agent Watch?",
            font=("Arial", 16, "bold"),
        ).pack(pady=20)
        
        ttk.Label(
            self,
            text="Select the market signals your agent needs:",
            font=("Arial", 12),
        ).pack(pady=10)
        
        # Agent name
        name_frame = ttk.Frame(self)
        name_frame.pack(fill=tk.X, padx=40, pady=10)
        ttk.Label(name_frame, text="Agent Name:", font=("Arial", 11)).pack(side=tk.LEFT)
        self.name_var = tk.StringVar()
        name_entry = ttk.Entry(name_frame, textvariable=self.name_var, width=30, font=("Arial", 11))
        name_entry.pack(side=tk.LEFT, padx=10)
        
        # Input checkboxes
        self.input_vars = {}
        inputs_frame = ttk.LabelFrame(self, text="Market Signals", padding=20)
        inputs_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=10)
        
        input_options = {
            "q_buy": "Q-Learning Buy Signal",
            "q_sell": "Q-Learning Sell Signal",
            "price_up": "Price Going Up",
            "price_down": "Price Going Down",
            "trend": "Price Trend",
            "volume": "Trading Volume",
            "regime": "Market Regime",
            "risk_budget_ok": "Risk Budget OK",
        }
        
        for key, label in input_options.items():
            var = tk.BooleanVar(value=(key in ["q_buy", "q_sell", "price_up", "price_down"]))
            self.input_vars[key] = var
            
            cb = ttk.Checkbutton(
                inputs_frame,
                text=label,
                variable=var,
                font=("Arial", 10),
            )
            cb.pack(anchor=tk.W, pady=5)
    
    def get_data(self) -> dict:
        """Get input selection."""
        return {
            "name": self.name_var.get(),
            "selected_inputs": {k: v.get() for k, v in self.input_vars.items()},
        }


class LayersStepFrame(StepFrame):
    """Step 3: Layer configuration."""
    
    def setup_ui(self):
        """Set up layer configuration UI."""
        ttk.Label(
            self,
            text="How Many Layers?",
            font=("Arial", 16, "bold"),
        ).pack(pady=20)
        
        ttk.Label(
            self,
            text="Layers help your agent make better decisions:",
            font=("Arial", 12),
        ).pack(pady=10)
        
        layers_frame = ttk.Frame(self)
        layers_frame.pack(pady=20)
        
        self.layers_var = tk.IntVar(value=1)
        ttk.Label(layers_frame, text="Number of Layers:", font=("Arial", 11)).pack()
        
        layers_spin = ttk.Spinbox(
            layers_frame,
            from_=1,
            to=5,
            textvariable=self.layers_var,
            width=10,
            font=("Arial", 11),
        )
        layers_spin.pack(pady=10)
    
    def get_data(self) -> dict:
        """Get layer configuration."""
        return {"num_layers": self.layers_var.get()}


class SafetyStepFrame(StepFrame):
    """Step 4: Safety options."""
    
    def setup_ui(self):
        """Set up safety options UI."""
        ttk.Label(
            self,
            text="Safety Settings",
            font=("Arial", 16, "bold"),
        ).pack(pady=20)
        
        safety_frame = ttk.LabelFrame(self, text="Options", padding=20)
        safety_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=10)
        
        self.safety_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            safety_frame,
            text="Include Risk Management",
            variable=self.safety_var,
            font=("Arial", 11),
        ).pack(anchor=tk.W, pady=5)
        
        self.comm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            safety_frame,
            text="Enable Communication Outputs",
            variable=self.comm_var,
            font=("Arial", 11),
        ).pack(anchor=tk.W, pady=5)
    
    def get_data(self) -> dict:
        """Get safety configuration."""
        return {
            "include_safety": self.safety_var.get(),
            "include_communication": self.comm_var.get(),
        }


class ReviewStepFrame(StepFrame):
    """Step 5: Review and generate."""
    
    def setup_ui(self):
        """Set up review UI."""
        ttk.Label(
            self,
            text="Review Your Agent",
            font=("Arial", 16, "bold"),
        ).pack(pady=20)
        
        # Preview generated spec
        preview_label = ttk.Label(
            self,
            text="Generated Tau Spec Preview:",
            font=("Arial", 11, "bold"),
        )
        preview_label.pack(anchor=tk.W, padx=40, pady=10)
        
        self.preview_text = scrolledtext.ScrolledText(
            self,
            width=70,
            height=20,
            font=("Courier", 9),
            wrap=tk.NONE,
        )
        self.preview_text.pack(fill=tk.BOTH, expand=True, padx=40, pady=10)
        self.preview_text.config(state=tk.DISABLED)
        
        self.update_preview()
    
    def update_preview(self):
        """Update preview with generated spec."""
        spec = self.controller.generate_spec()
        self.preview_text.config(state=tk.NORMAL)
        self.preview_text.delete(1.0, tk.END)
        self.preview_text.insert(1.0, spec)
        self.preview_text.config(state=tk.DISABLED)
    
    def get_data(self) -> dict:
        """No data to collect in review step."""
        return {}


class WizardGUI:
    """Main wizard GUI application."""
    
    STEP_FRAMES = {
        WizardStep.STRATEGY: StrategyStepFrame,
        WizardStep.INPUTS: InputsStepFrame,
        WizardStep.LAYERS: LayersStepFrame,
        WizardStep.SAFETY: SafetyStepFrame,
        WizardStep.REVIEW: ReviewStepFrame,
    }
    
    def __init__(self, root):
        """Initialize wizard GUI."""
        self.root = root
        self.root.title("Tau Agent Factory - Create Your Agent")
        self.root.geometry("800x700")
        
        self.controller = WizardController()
        self.current_frame: Optional[StepFrame] = None
        
        self.setup_ui()
        self.show_step()
    
    def setup_ui(self):
        """Set up main UI structure."""
        # Header
        header = ttk.Frame(self.root)
        header.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(
            header,
            text="Tau Agent Factory",
            font=("Arial", 18, "bold"),
        ).pack()
        
        # Progress bar
        self.progress_label = ttk.Label(
            header,
            text=f"Step {self.controller.step_number} of {self.controller.total_steps}",
            font=("Arial", 11),
        )
        self.progress_label.pack(pady=5)
        
        # Main content area
        self.content_frame = ttk.Frame(self.root)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Navigation buttons
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.back_btn = ttk.Button(
            nav_frame,
            text="← Back",
            command=self.on_back,
            width=15,
        )
        self.back_btn.pack(side=tk.LEFT, padx=5)
        
        self.next_btn = ttk.Button(
            nav_frame,
            text="Next →",
            command=self.on_next,
            width=15,
        )
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        self.finish_btn = ttk.Button(
            nav_frame,
            text="Finish & Save",
            command=self.on_finish,
            width=15,
        )
        self.finish_btn.pack(side=tk.LEFT, padx=5)
        
        self.update_navigation()
    
    def show_step(self):
        """Show current wizard step."""
        # Remove current frame
        if self.current_frame:
            self.current_frame.destroy()
        
        # Create new frame
        step = self.controller.current_step
        frame_class = self.STEP_FRAMES[step]
        self.current_frame = frame_class(self.content_frame, self.controller)
        self.current_frame.pack(fill=tk.BOTH, expand=True)
        
        # Update progress
        self.progress_label.config(
            text=f"Step {self.controller.step_number} of {self.controller.total_steps}: {step.value.title()}"
        )
        
        # Update navigation buttons
        self.update_navigation()
        
        # Update review preview if on review step
        if step == WizardStep.REVIEW and isinstance(self.current_frame, ReviewStepFrame):
            self.current_frame.update_preview()
    
    def update_navigation(self):
        """Update navigation button states."""
        self.back_btn.config(state=tk.NORMAL if not self.controller.is_first_step else tk.DISABLED)
        
        if self.controller.is_last_step:
            self.next_btn.pack_forget()
            self.finish_btn.pack(side=tk.LEFT, padx=5)
        else:
            self.finish_btn.pack_forget()
            self.next_btn.pack(side=tk.LEFT, padx=5)
    
    def on_next(self):
        """Handle Next button click."""
        if not self.current_frame:
            return
        
        data = self.current_frame.get_data()
        if self.controller.next(data):
            self.show_step()
        else:
            errors = self.controller.get_validation_errors()
            if errors:
                error_msg = "\n".join(f"- {field}: {msg}" for field, msg in errors.items())
                messagebox.showerror("Validation Error", f"Please fix these errors:\n\n{error_msg}")
    
    def on_back(self):
        """Handle Back button click."""
        if self.controller.prev():
            self.show_step()
    
    def on_finish(self):
        """Handle Finish button click."""
        # Generate spec
        spec = self.controller.generate_spec()
        
        # Ask for save location
        from tkinter import filedialog
        save_path = filedialog.asksaveasfilename(
            defaultextension=".tau",
            filetypes=[("Tau files", "*.tau"), ("All files", "*.*")],
            initialfile=f"{self.controller.data.name}.tau",
        )
        
        if save_path:
            Path(save_path).write_text(spec)
            messagebox.showinfo("Success", f"Agent spec saved to:\n{save_path}")
            self.root.quit()


def main():
    """Run wizard GUI."""
    root = tk.Tk()
    app = WizardGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

