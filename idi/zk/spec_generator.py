"""Tau spec generator from training artifacts and configs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from idi.training.python.idi_iann.config import TrainingConfig


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
    ) -> None:
        """Generate V38 Minimal Core agent spec.

        Args:
            output_path: Path to write spec file
            include_tile_coding: Whether to include tile coding inputs
            include_communication: Whether to include communication Q-table inputs
        """
        spec_lines = [
            "# IDI-ready agent spec (V38 Minimal Core)",
            "# Generated from training config",
            "",
            "# === INPUT STREAMS (IDI-ready) ===",
            'i0:sbf = in file("inputs/price.in").',
            'i1:sbf = in file("inputs/volume.in").',
            'i2:sbf = in file("inputs/trend.in").',
            'i3:sbf = in file("inputs/profit_guard.in").',
            'i4:sbf = in file("inputs/failure_echo.in").',
            'i5:sbf = in file("inputs/q_buy.in").',
            'i6:sbf = in file("inputs/q_sell.in").',
            'i7:sbf = in file("inputs/risk_budget_ok.in").',
            'i8:bv[5] = in file("inputs/q_regime.in").',
        ]

        if include_communication:
            spec_lines.extend([
                'i9:sbf = in file("inputs/q_emote_positive.in").',
                'iA:sbf = in file("inputs/q_emote_alert.in").',
            ])

        spec_lines.extend([
            "",
            "# === INPUT MIRRORS (Interpreter requirement) ===",
            'o_i0:sbf = out file("outputs/i0_mirror.out").',
            'o_i1:sbf = out file("outputs/i1_mirror.out").',
            'o_i2:sbf = out file("outputs/i2_mirror.out").',
            'o_i3:sbf = out file("outputs/i3_mirror.out").',
            'o_i4:sbf = out file("outputs/i4_mirror.out").',
            'o_i5:sbf = out file("outputs/i5_mirror.out").',
            'o_i6:sbf = out file("outputs/i6_mirror.out").',
            'o_i7:sbf = out file("outputs/i7_mirror.out").',
            'o_i8:bv[5] = out file("outputs/i8_mirror.out").',
        ])

        if include_communication:
            spec_lines.extend([
                'o_i9:sbf = out file("outputs/i9_mirror.out").',
                'o_iA:sbf = out file("outputs/iA_mirror.out").',
            ])

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
            "    (o13[t] = o13[t-1] | o11[t]) &&",
            "",
            "    # Input mirrors",
            "    (o_i0[t] = i0[t]) &&",
            "    (o_i1[t] = i1[t]) &&",
            "    (o_i2[t] = i2[t]) &&",
            "    (o_i3[t] = i3[t]) &&",
            "    (o_i4[t] = i4[t]) &&",
            "    (o_i5[t] = i5[t]) &&",
            "    (o_i6[t] = i6[t]) &&",
            "    (o_i7[t] = i7[t]) &&",
            "    (o_i8[t] = i8[t])",
        ])

        if include_communication:
            spec_lines.extend([
                " &&",
                "    (o_i9[t] = i9[t]) &&",
                "    (o_iA[t] = iA[t])",
            ])

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
        generator.generate_v38_spec(output_path)
    elif spec_type == "layered":
        generator.generate_layered_spec(output_path)
    else:
        raise ValueError(f"Unknown spec type: {spec_type}")

