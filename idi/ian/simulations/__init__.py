"""
IAN Simulations - Simulation scripts for the Intelligent Augmentation Network.

Available simulations:
- trading_agent_demo: End-to-end trading agent competition simulation
"""

from .trading_agent_demo import run_demo, DemoConfig

__all__ = ["run_demo", "DemoConfig"]
