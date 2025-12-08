"""Tau spec generator from training artifacts and configs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from idi.training.python.idi_iann.config import TrainingConfig
from idi.contracts.streams import get_contract


class TauSpecGenerator:
    """Generates Tau-language specs from training artifacts."""

    def __init__(self, config: TrainingConfig):
        """Initialize generator with training config."""
        self.config = config

    def generate_v38_spec(
        self,
        output_path: Path,
        *,
        include_tile_coding: bool = False,
        include_communication: bool = True,
        contract_name: str = "v38",
    ) -> None:
        """Generate V38 Minimal Core agent spec.

        Args:
            output_path: Path to write spec file
            include_tile_coding: Whether to include tile coding inputs
            include_communication: Whether to include communication Q-table inputs
        """
        contract = get_contract(contract_name)

        spec_lines = [
            "# IDI-ready agent spec (V38 Minimal Core)",
            "# Generated from training config",
            "",
            "# === INPUT STREAMS (IDI-ready) ===",
        ]

        for idx, stream in enumerate(s for s in contract if s.role == "input"):
            spec_lines.append(f'i{idx}:{stream.tau_type} = in file("{stream.filename}").')

        start_mirror_idx = len([s for s in contract if s.role == "input"])

        spec_lines.extend([
            "",
            "# === INPUT MIRRORS (Interpreter requirement) ===",
        ])

        for offset, stream in enumerate(s for s in contract if s.role == "input"):
            mirror_idx = start_mirror_idx + offset
            spec_lines.append(f"o_i{mirror_idx}:{stream.tau_type} = out file(\"outputs/i{offset}_mirror.out\").")

        spec_lines.extend([
            "",
            "# === OUTPUT STREAMS ===",
            'o0:sbf = out file("outputs/state.out").',
            'o1:sbf = out file("outputs/holding.out").',
            'o2:sbf = out file("outputs/buy_signal.out").',
            'o3:sbf = out file("outputs/sell_signal.out").',
            'o6:sbf = out file("outputs/timer_b0.out").',
            'o7:sbf = out file("outputs/timer_b1.out").',
            'o9:sbf = out file("outputs/nonce.out").',
            'o10:sbf = out file("outputs/entry_price.out").',
            'o11:sbf = out file("outputs/profit.out").',
            'o13:sbf = out file("outputs/has_burned.out").',
            "",
            "# === RECURRENCE RELATIONS ===",
            "r (",
            "    # State machine",
            "    (o0[t] = (o0[t-1]' & (i0[t]' & i1[t] & i2[t] & o1[t-1]') &",
            "              o0[t-1]' & i1[t] & (o7[t-1] & o6[t-1])' & o9[t-1]' & i4[t]' & i7[t]) |",
            "             (o0[t-1] & (i0[t] & o1[t-1])' &",
            "              (o7[t-1] & o6[t-1])' & i1[t] & i4[t]' & i7[t])) &&",
            "",
            "    # Trading signals",
            "    (o2[t] = i5[t] & i7[t] & o0[t] & o0[t-1]' & o1[t-1]') &&",
            "    (o3[t] = i6[t] & o0[t-1] & o0[t]' & o1[t-1]) &&",
            "    (o1[t] = o2[t] | (o3[t]' & o1[t-1])) &&",
            "",
            "    # Timer",
            "    (o6[t] = o0[t] & o6[t-1]') &&",
            "    (o7[t] = o0[t] & ((o7[t-1] & o6[t-1]') | (o7[t-1]' & o6[t-1]))) &&",
            "",
            "    # Nonce",
            "    (o9[t] = o2[t] | (o0[t-1] & o3[t]' & o9[t-1])) &&",
            "",
            "    # Economic logic",
            "    (o10[t] = (o2[t] & i0[t]) | (o0[t-1] & o2[t]' & o3[t]' & o10[t-1])) &&",
            "    (o11[t] = o3[t] & i0[t] & o10[t-1]' & i3[t]) &&",
            "",
            "    # Burn tracking",
            "    (o13[t] = o13[t-1] | o11[t])",
        ])

        # Append input mirror bindings
        mirror_bindings = []
        for offset in range(start_mirror_idx):
            mirror_bindings.append(f"(o_i{start_mirror_idx + offset}[t] = i{offset}[t])")
        if mirror_bindings:
            spec_lines.append(" &&")
            spec_lines.append("    " + " &&\n    ".join(mirror_bindings))

        spec_lines.append(")")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(spec_lines), encoding="utf-8")

    def generate_layered_spec(
        self,
        output_path: Path,
        *,
        num_layers: int = 3,
    ) -> None:
        """Generate layered strategy agent spec.

        Args:
            output_path: Path to write spec file
            num_layers: Number of Q-layers (default: 3)
        """
        spec_lines = [
            "# Layered Strategy Agent",
            "# Generated from training config",
            "",
            "# === INPUT STREAMS ===",
            'i0:sbf = in file("inputs/price_up.in").',
            'i1:sbf = in file("inputs/price_down.in").',
        ]

        for i in range(num_layers):
            spec_lines.append(f'i{2+i}:sbf = in file("inputs/weight_layer{i}.in").')

        spec_lines.extend([
            "",
            "# === OUTPUT STREAMS ===",
            'o0:sbf = out file("outputs/buy_signal.out").',
            'o1:sbf = out file("outputs/sell_signal.out").',
            'o2:sbf = out file("outputs/position.out").',
            "",
            "# === RECURRENCE RELATIONS ===",
            "r (",
            "    # Layer votes and action selection",
            "    (o0[t] = o2[t-1]' & (",
        ])

        layer_conditions = []
        for i in range(num_layers):
            layer_conditions.append(f"(i{2+i}[t] & i0[t])")
        spec_lines.append(" | ".join(layer_conditions) + ")) &&")

        spec_lines.extend([
            "    (o1[t] = o2[t-1] & (",
        ])

        layer_conditions = []
        for i in range(num_layers):
            layer_conditions.append(f"(i{2+i}[t] & i1[t])")
        spec_lines.append(" | ".join(layer_conditions) + ")) &&")

        spec_lines.extend([
            "    (o2[t] = o0[t] | (o2[t-1] & o1[t]'))",
            ")",
        ])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(spec_lines), encoding="utf-8")


def generate_spec_from_config(
    config_path: Path,
    output_path: Path,
    spec_type: str = "v38",
    contract_name: str = "v38",
) -> None:
    """Generate Tau spec from config file.

    Args:
        config_path: Path to training config JSON
        output_path: Path to write spec file
        spec_type: Type of spec to generate ("v38" or "layered")
    """
    import json
    from idi.training.python.idi_iann.config import TrainingConfig

    config_dict = json.loads(config_path.read_text())
    config = TrainingConfig(**config_dict)
    generator = TauSpecGenerator(config)

    if spec_type == "v38":
        generator.generate_v38_spec(output_path, contract_name=contract_name)
    elif spec_type == "layered":
        generator.generate_layered_spec(output_path)
    else:
        raise ValueError(f"Unknown spec type: {spec_type}")
