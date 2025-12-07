#!/usr/bin/env python3
"""Agent Development CLI - Python version.

Rapid prototyping tool for creating new Tau Language intelligent agents.
Leverages Python's rich ecosystem for quick iteration.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional
import shutil


# Agent templates
TEMPLATES = {
    "momentum": {
        "description": "Momentum following strategy",
        "tau_spec": """# {name} Agent (IDI Q-Learning)
#
# Strategy: {description}
#
# Inputs from IDI training stack:
#   i0 : q_buy (Q-table buy signal)
#   i1 : q_sell (Q-table sell signal)
#   i2 : risk_budget_ok (Risk management gate)
#   i3 : price_up (Price increased)
#   i4 : price_down (Price decreased)
#   i5 : q_regime (Market regime: bull/bear/chop/panic)
#
# Outputs:
#   o0 : position (1=holding, 0=no position)
#   o1 : buy_signal (Entry trigger)
#   o2 : sell_signal (Exit trigger)

# === INPUT STREAMS ===
i0:sbf = in file("inputs/q_buy.in").
i1:sbf = in file("inputs/q_sell.in").
i2:sbf = in file("inputs/risk_budget_ok.in").
i3:sbf = in file("inputs/price_up.in").
i4:sbf = in file("inputs/price_down.in").
i5:bv[5] = in file("inputs/q_regime.in").

# === INPUT MIRRORS (Interpreter requirement) ===
i0:sbf = out file("outputs/i0_mirror.out").
i1:sbf = out file("outputs/i1_mirror.out").
i2:sbf = out file("outputs/i2_mirror.out").
i3:sbf = out file("outputs/i3_mirror.out").
i4:sbf = out file("outputs/i4_mirror.out").
i5:bv[5] = out file("outputs/i5_mirror.out").

# === OUTPUT STREAMS ===
o0:sbf = out file("outputs/position.out").
o1:sbf = out file("outputs/buy_signal.out").
o2:sbf = out file("outputs/sell_signal.out").

r
(
    # Buy signal: Q-table says buy, risk OK, not already holding
    (o1[t] = i0[t] & i2[t] & o0[t-1]') &&

    # Sell signal: Q-table says sell, or risk budget blocked
    (o2[t] = (i1[t] | i2[t]') & o0[t-1]) &&

    # Position: set on buy, cleared on sell
    (o0[t] = o1[t] | (o0[t-1] & o2[t]'))
)
""",
        "training_config": {
            "episodes": 256,
            "episode_length": 128,
            "discount": 0.95,
            "learning_rate": 0.15,
            "exploration_decay": 0.996,
            "quantizer": {
                "price_buckets": 8,
                "volume_buckets": 4,
                "trend_buckets": 8,
                "scarcity_buckets": 4,
                "mood_buckets": 4,
            },
            "rewards": {
                "pnl": 1.0,
                "scarcity_alignment": 0.3,
                "ethics_bonus": 0.2,
                "communication_clarity": 0.1,
            },
        },
    },
    "mean_reversion": {
        "description": "Mean reversion strategy",
        "tau_spec": """# {name} Agent (IDI Q-Learning)
#
# Strategy: {description}
#
# Inputs from IDI training stack:
#   i0 : q_buy (Q-table buy signal)
#   i1 : q_sell (Q-table sell signal)
#   i2 : risk_budget_ok (Risk management gate)
#   i3 : price_up (Price increased)
#   i4 : price_down (Price decreased)
#   i5 : q_regime (Market regime: bull/bear/chop/panic)
#
# Outputs:
#   o0 : position (1=holding, 0=no position)
#   o1 : buy_signal (Entry trigger - buy on dips)
#   o2 : sell_signal (Exit trigger - sell on spikes)

# === INPUT STREAMS ===
i0:sbf = in file("inputs/q_buy.in").
i1:sbf = in file("inputs/q_sell.in").
i2:sbf = in file("inputs/risk_budget_ok.in").
i3:sbf = in file("inputs/price_up.in").
i4:sbf = in file("inputs/price_down.in").
i5:bv[5] = in file("inputs/q_regime.in").

# === INPUT MIRRORS ===
i0:sbf = out file("outputs/i0_mirror.out").
i1:sbf = out file("outputs/i1_mirror.out").
i2:sbf = out file("outputs/i2_mirror.out").
i3:sbf = out file("outputs/i3_mirror.out").
i4:sbf = out file("outputs/i4_mirror.out").
i5:bv[5] = out file("outputs/i5_mirror.out").

# === OUTPUT STREAMS ===
o0:sbf = out file("outputs/position.out").
o1:sbf = out file("outputs/buy_signal.out").
o2:sbf = out file("outputs/sell_signal.out").

r
(
    # Buy signal: Q-table says buy, risk OK, price down (mean reversion entry)
    (o1[t] = i0[t] & i2[t] & i4[t] & o0[t-1]') &&

    # Sell signal: Q-table says sell, or price up (mean reversion exit)
    (o2[t] = ((i1[t] | i3[t]) | i2[t]') & o0[t-1]) &&

    # Position: set on buy, cleared on sell
    (o0[t] = o1[t] | (o0[t-1] & o2[t]'))
)
""",
        "training_config": {
            "episodes": 256,
            "episode_length": 128,
            "discount": 0.92,
            "learning_rate": 0.2,
            "exploration_decay": 0.995,
            "quantizer": {
                "price_buckets": 8,
                "volume_buckets": 4,
                "trend_buckets": 4,
                "scarcity_buckets": 4,
                "mood_buckets": 4,
            },
            "rewards": {
                "pnl": 1.0,
                "scarcity_alignment": 0.5,
                "ethics_bonus": 0.3,
                "communication_clarity": 0.1,
            },
        },
    },
    "regime_aware": {
        "description": "Regime-aware adaptive strategy",
        "tau_spec": """# {name} Agent (IDI Q-Learning)
#
# Strategy: {description}
#
# Inputs from IDI training stack:
#   i0 : q_buy (Q-table buy signal)
#   i1 : q_sell (Q-table sell signal)
#   i2 : risk_budget_ok (Risk management gate)
#   i3 : price_up (Price increased)
#   i4 : price_down (Price decreased)
#   i5 : q_regime (Market regime: bull/bear/chop/panic)

# === INPUT STREAMS ===
i0:sbf = in file("inputs/q_buy.in").
i1:sbf = in file("inputs/q_sell.in").
i2:sbf = in file("inputs/risk_budget_ok.in").
i3:sbf = in file("inputs/price_up.in").
i4:sbf = in file("inputs/price_down.in").
i5:bv[5] = in file("inputs/q_regime.in").

# === INPUT MIRRORS ===
i0:sbf = out file("outputs/i0_mirror.out").
i1:sbf = out file("outputs/i1_mirror.out").
i2:sbf = out file("outputs/i2_mirror.out").
i3:sbf = out file("outputs/i3_mirror.out").
i4:sbf = out file("outputs/i4_mirror.out").
i5:bv[5] = out file("outputs/i5_mirror.out").

# === OUTPUT STREAMS ===
o0:sbf = out file("outputs/position.out").
o1:sbf = out file("outputs/buy_signal.out").
o2:sbf = out file("outputs/sell_signal.out").
o3:bv[5] = out file("outputs/regime_tracked.out").

r
(
    # Regime tracking
    (o3[t] = i5[t]) &&

    # Buy signal: Q-table says buy, risk OK, not in panic regime
    (o1[t] = i0[t] & i2[t] & (i5[t] != {31}:bv[5]) & o0[t-1]') &&

    # Sell signal: Q-table says sell, or risk budget blocked, or panic regime
    (o2[t] = ((i1[t] | i2[t]') | (i5[t] = {31}:bv[5])) & o0[t-1]) &&

    # Position: set on buy, cleared on sell
    (o0[t] = o1[t] | (o0[t-1] & o2[t]'))
)
""",
        "training_config": {
            "episodes": 512,
            "episode_length": 256,
            "discount": 0.95,
            "learning_rate": 0.12,
            "exploration_decay": 0.998,
            "quantizer": {
                "price_buckets": 8,
                "volume_buckets": 4,
                "trend_buckets": 8,
                "scarcity_buckets": 4,
                "mood_buckets": 4,
            },
            "rewards": {
                "pnl": 1.0,
                "scarcity_alignment": 0.4,
                "ethics_bonus": 0.3,
                "communication_clarity": 0.2,
            },
        },
    },
}


def create_agent_directory(
    name: str,
    strategy: str,
    output_dir: Path,
    template: Optional[Dict] = None,
) -> Path:
    """Create agent directory structure with all necessary files."""
    agent_dir = output_dir / name
    agent_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (agent_dir / "inputs").mkdir(exist_ok=True)
    (agent_dir / "outputs").mkdir(exist_ok=True)
    (agent_dir / "tests").mkdir(exist_ok=True)
    
    if template is None:
        template = TEMPLATES.get(strategy, TEMPLATES["momentum"])
    
    # Write Tau spec
    spec_content = template["tau_spec"].format(name=name, description=template["description"])
    (agent_dir / f"{name}.tau").write_text(spec_content)
    
    # Write training script
    training_script = f"""#!/usr/bin/env python3
\"\"\"Train Q-table for {name} agent and export traces.\"\"\"

import sys
import os
from pathlib import Path

# Add training directory to path
idi_root = Path(__file__).resolve().parents[2]
training_python = idi_root / "training" / "python"
sys.path.insert(0, str(training_python))
os.environ["PYTHONPATH"] = str(training_python)

from idi_iann.config import TrainingConfig, QuantizerConfig, RewardWeights, EmoteConfig, CommunicationConfig
from idi_iann.factories import create_trainer

def main():
    config = TrainingConfig(**{json.dumps(template["training_config"], indent=8)})
    
    print("Training {name} agent...")
    trainer = create_trainer(config, seed=42, use_crypto_env=True)
    policy, trace = trainer.run()
    
    stats = trainer.stats()
    print(f"\\nTraining complete!")
    print(f"  Mean reward: {{stats['mean_reward']:.4f}}")
    print(f"  Policy states: {{len(policy._table)}}")
    
    # Export traces
    inputs_dir = Path(__file__).parent / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    trace.export(inputs_dir)
    print(f"\\nTraces exported to: {{inputs_dir}}")

if __name__ == "__main__":
    main()
"""
    (agent_dir / "train_agent.py").write_text(training_script)
    (agent_dir / "train_agent.py").chmod(0o755)
    
    # Write run script
    run_script = f"""#!/bin/bash
# Run {name} agent with Tau

set -e

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
TAU_BIN="${{TAU_BIN:-/path/to/tau}}"

cd "$SCRIPT_DIR"

if [ ! -f "$TAU_BIN" ]; then
    echo "Error: Tau binary not found. Set TAU_BIN environment variable."
    exit 1
fi

mkdir -p outputs
"$TAU_BIN" {name}.tau
"""
    (agent_dir / "run_agent.sh").write_text(run_script)
    (agent_dir / "run_agent.sh").chmod(0o755)
    
    # Write README
    readme = f"""# {name}

**Strategy**: {template["description"]}

## Quick Start

1. Train Q-table:
   ```bash
   python3 train_agent.py
   ```

2. Run agent:
   ```bash
   ./run_agent.sh
   ```

## Configuration

Training config: See `train_agent.py`

Tau spec: `{name}.tau`

## Development Notes

- Created with IDI Agent Development CLI
- Strategy: {strategy}
"""
    (agent_dir / "README.md").write_text(readme)
    
    return agent_dir


def main():
    parser = argparse.ArgumentParser(
        description="Create a new Tau Language intelligent agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create momentum agent in practice folder
  python create_agent.py --name momentum_test --strategy momentum --out ../practice/

  # Create custom agent with template
  python create_agent.py --name my_agent --strategy momentum --out ../practice/ --custom-template custom.json
        """,
    )
    
    parser.add_argument(
        "--name",
        required=True,
        help="Agent name (e.g., 'momentum_agent')",
    )
    parser.add_argument(
        "--strategy",
        choices=list(TEMPLATES.keys()),
        default="momentum",
        help="Strategy template to use",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("../practice"),
        help="Output directory (default: ../practice)",
    )
    parser.add_argument(
        "--custom-template",
        type=Path,
        help="Path to custom template JSON file",
    )
    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="List available templates and exit",
    )
    
    args = parser.parse_args()
    
    if args.list_templates:
        print("Available templates:")
        for name, template in TEMPLATES.items():
            print(f"  {name:15} - {template['description']}")
        return
    
    # Load custom template if provided
    template = None
    if args.custom_template:
        if not args.custom_template.exists():
            print(f"Error: Template file not found: {args.custom_template}")
            return
        template = json.loads(args.custom_template.read_text())
    
    # Create agent
    agent_dir = create_agent_directory(
        args.name,
        args.strategy,
        args.out,
        template,
    )
    
    print(f"✅ Created agent '{args.name}' at {agent_dir}")
    print(f"\nNext steps:")
    print(f"  1. cd {agent_dir}")
    print(f"  2. python3 train_agent.py")
    print(f"  3. ./run_agent.sh")


if __name__ == "__main__":
    main()

