from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from idi.gui.backend.services.invariants import InvariantService
from idi.devkit.experimental.mpb_vm import (
    BinExpr,
    BinOp,
    IDI_REG_LAYOUT,
    IntLiteral,
    MpbVm,
    Var,
    compile_expr_to_mpb,
)


SCALE = 1_000_000


def _build_i1_i5_expr() -> BinExpr:
    i1 = BinExpr(BinOp.LE, Var("state_size"), Var("max_state_size"))
    i2 = BinExpr(BinOp.GE, Var("discount_fp"), Var("min_discount_fp"))
    i3 = BinExpr(BinOp.LE, Var("learning_rate_fp"), Var("max_learning_rate_fp"))
    i4 = BinExpr(BinOp.GT, Var("epsilon_decay_steps"), IntLiteral(0))
    i5 = BinExpr(BinOp.GE, Var("budget_total"), Var("min_budget_total"))

    left = BinExpr(BinOp.AND, i1, i2)
    left = BinExpr(BinOp.AND, left, i3)
    left = BinExpr(BinOp.AND, left, i4)
    return BinExpr(BinOp.AND, left, i5)


def _build_registers_from_goal_spec(goal_spec: dict) -> list[int]:
    registers = [0] * 32

    training = goal_spec.get("training", {})
    budget = training.get("budget", {})

    patch_params = goal_spec.get(
        "patch_params",
        {
            "num_price_bins": 10,
            "num_inventory_bins": 10,
            "learning_rate": 0.1,
            "discount_factor": 0.99,
            "epsilon_decay_steps": 1000,
        },
    )

    price_bins = patch_params.get("num_price_bins", 10)
    inventory_bins = patch_params.get("num_inventory_bins", 10)
    state_size = int(price_bins * inventory_bins)

    discount = float(patch_params.get("discount_factor", 0.99))
    discount_fp = int(round(discount * SCALE))

    lr = float(patch_params.get("learning_rate", 0.1))
    lr_fp = int(round(lr * SCALE))

    decay = int(patch_params.get("epsilon_decay_steps", 1000))

    max_agents = int(budget.get("max_agents", 8))
    max_episodes = int(budget.get("max_episodes_per_agent", 64))
    budget_total = int(max_agents * max_episodes)

    regs = IDI_REG_LAYOUT
    registers[regs["state_size"]] = state_size
    registers[regs["discount_fp"]] = discount_fp
    registers[regs["learning_rate_fp"]] = lr_fp
    registers[regs["epsilon_decay_steps"]] = decay
    registers[regs["budget_total"]] = budget_total

    registers[regs["max_state_size"]] = 2048
    registers[regs["min_discount_fp"]] = int(0.5 * SCALE)
    registers[regs["max_learning_rate_fp"]] = int(0.5 * SCALE)
    registers[regs["min_budget_total"]] = 64

    return registers


def _load_conservative_goal_spec() -> dict:
    # This test file lives at:
    #   idi/gui/backend/services/test_invariants_mpb_consistency.py
    # The project root (containing examples/) is four levels up from this file:
    #   .../Intelligent-Daemon-Interface
    path = (
        Path(__file__)
        .resolve()
        .parents[4]
        / "examples"
        / "synth"
        / "conservative_qagent_goal.json"
    )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_invariants_conservative_consistent_with_mpb() -> None:
    goal_spec = _load_conservative_goal_spec()

    service = InvariantService()
    py_results = service.check_all(goal_spec)
    py_ok = all(r["ok"] for r in py_results)

    expr = _build_i1_i5_expr()
    bytecode = compile_expr_to_mpb(expr, IDI_REG_LAYOUT)
    registers = _build_registers_from_goal_spec(goal_spec)
    vm = MpbVm(registers=registers)
    mpb_ok = vm.execute(bytecode) != 0

    assert py_ok is True
    assert mpb_ok is True
    assert py_ok == mpb_ok


def test_invariants_violation_consistent_with_mpb() -> None:
    goal_spec = _load_conservative_goal_spec()
    bad = deepcopy(goal_spec)

    bad_patch = {
        "num_price_bins": 1000,
        "num_inventory_bins": 1000,
        "learning_rate": 0.9,
        "discount_factor": 0.2,
        "epsilon_decay_steps": 1000,
    }
    bad["patch_params"] = bad_patch

    training = bad.setdefault("training", {})
    budget = training.setdefault("budget", {})
    budget["max_agents"] = 1
    budget["max_episodes_per_agent"] = 1

    service = InvariantService()
    py_results = service.check_all(bad)
    py_ok = all(r["ok"] for r in py_results)

    expr = _build_i1_i5_expr()
    bytecode = compile_expr_to_mpb(expr, IDI_REG_LAYOUT)
    registers = _build_registers_from_goal_spec(bad)
    vm = MpbVm(registers=registers)
    mpb_ok = vm.execute(bytecode) != 0

    assert py_ok is False
    assert mpb_ok is False
    assert py_ok == mpb_ok
