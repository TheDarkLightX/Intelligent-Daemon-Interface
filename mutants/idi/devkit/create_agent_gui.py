#!/usr/bin/env python3
"""Agent Development GUI - Python version using tkinter.

Interactive GUI for creating and managing Tau Language intelligent agents.
Leverages Python's rapid prototyping and tkinter's simplicity.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import json
import subprocess
import sys


class AgentDevGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("IDI Agent Development GUI")
        self.root.geometry("600x500")
        
        # Variables
        self.agent_name = tk.StringVar()
        self.strategy = tk.StringVar(value="momentum")
        self.output_dir = tk.StringVar(value=str(Path("../practice").resolve()))
        
        self.setup_ui()
    
    def setup_ui(self):
        # Header
        header = ttk.Label(
            self.root,
            text="IDI Agent Development",
            font=("Arial", 16, "bold")
        )
        header.pack(pady=10)
        
        # Agent name
        frame_name = ttk.Frame(self.root)
        frame_name.pack(fill=tk.X, padx=20, pady=5)
        ttk.Label(frame_name, text="Agent Name:").pack(side=tk.LEFT)
        ttk.Entry(frame_name, textvariable=self.agent_name, width=30).pack(side=tk.LEFT, padx=10)
        
        # Strategy selection
        frame_strategy = ttk.Frame(self.root)
        frame_strategy.pack(fill=tk.X, padx=20, pady=5)
        ttk.Label(frame_strategy, text="Strategy:").pack(side=tk.LEFT)
        strategy_combo = ttk.Combobox(
            frame_strategy,
            textvariable=self.strategy,
            values=["momentum", "mean_reversion", "regime_aware"],
            state="readonly",
            width=27
        )
        strategy_combo.pack(side=tk.LEFT, padx=10)
        
        # Output directory
        frame_output = ttk.Frame(self.root)
        frame_output.pack(fill=tk.X, padx=20, pady=5)
        ttk.Label(frame_output, text="Output Dir:").pack(side=tk.LEFT)
        ttk.Entry(frame_output, textvariable=self.output_dir, width=25).pack(side=tk.LEFT, padx=10)
        ttk.Button(
            frame_output,
            text="Browse",
            command=self.browse_output_dir
        ).pack(side=tk.LEFT)
        
        # Strategy description
        self.desc_label = ttk.Label(
            self.root,
            text="Momentum following strategy",
            font=("Arial", 10, "italic"),
            foreground="gray"
        )
        self.desc_label.pack(pady=5)
        strategy_combo.bind("<<ComboboxSelected>>", self.on_strategy_change)
        
        # Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=20)
        
        ttk.Button(
            button_frame,
            text="Create Agent",
            command=self.create_agent,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="List Templates",
            command=self.list_templates,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        # Status area
        self.status_text = tk.Text(self.root, height=10, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        scrollbar = ttk.Scrollbar(self.status_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.status_text.yview)
        
        self.log("Ready. Enter agent name and click 'Create Agent'.")
    
    def on_strategy_change(self, event=None):
        descriptions = {
            "momentum": "Momentum following strategy",
            "mean_reversion": "Mean reversion strategy",
            "regime_aware": "Regime-aware adaptive strategy",
        }
        self.desc_label.config(text=descriptions.get(self.strategy.get(), ""))
    
    def browse_output_dir(self):
        directory = filedialog.askdirectory(initialdir=self.output_dir.get())
        if directory:
            self.output_dir.set(directory)
    
    def log(self, message):
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.root.update()
    
    def list_templates(self):
        try:
            # Call CLI to list templates
            result = subprocess.run(
                [sys.executable, "create_agent.py", "--list-templates"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            self.log(result.stdout)
            if result.stderr:
                self.log(f"Error: {result.stderr}")
        except Exception as e:
            self.log(f"Error listing templates: {e}")
    
    def create_agent(self):
        name = self.agent_name.get().strip()
        if not name:
            messagebox.showerror("Error", "Agent name is required")
            return
        
        strategy = self.strategy.get()
        output_dir = Path(self.output_dir.get())
        
        self.log(f"Creating agent '{name}' with strategy '{strategy}'...")
        self.log(f"Output directory: {output_dir}")
        
        try:
            # Call CLI to create agent
            result = subprocess.run(
                [
                    sys.executable,
                    "create_agent.py",
                    "--name", name,
                    "--strategy", strategy,
                    "--out", str(output_dir),
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            
            if result.returncode == 0:
                self.log(result.stdout)
                messagebox.showinfo("Success", f"Agent '{name}' created successfully!")
            else:
                self.log(f"Error: {result.stderr}")
                messagebox.showerror("Error", f"Failed to create agent:\n{result.stderr}")
        except Exception as e:
            self.log(f"Error: {e}")
            messagebox.showerror("Error", f"Failed to create agent: {e}")


def main():
    root = tk.Tk()
    app = AgentDevGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

