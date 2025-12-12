from __future__ import annotations

from idi.devkit.experimental.mpb_vm import (
    BinExpr,
    BinOp,
    IDI_REG_LAYOUT,
    IntLiteral,
    MpbVm,
    MpbVmError,
    Opcode,
    UnExpr,
    UnOp,
    Var,
    VmStatus,
    compile_expr_to_mpb,
)


def _make_registers(size: int = 32) -> list[int]:
    return [0] * size


def test_push_and_halt_constant_true() -> None:
    value = 1
    bytecode = bytes([Opcode.PUSH]) + int(value).to_bytes(8, "little", signed=True) + bytes([Opcode.HALT])
    vm = MpbVm(registers=_make_registers())
    result = vm.execute(bytecode)
    assert result == 1


def test_risk_threshold_policy() -> None:
    layout = {"risk": 0, "max_risk": 1}
    expr = BinExpr(BinOp.LE, Var("risk"), Var("max_risk"))
    bytecode = compile_expr_to_mpb(expr, layout)

    registers = _make_registers()
    registers[0] = 50
    registers[1] = 100
    vm = MpbVm(registers=registers)
    assert vm.execute(bytecode) == 1

    registers[0] = 150
    vm = MpbVm(registers=registers)
    assert vm.execute(bytecode) == 0


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


def test_i1_i5_invariants_all_ok() -> None:
    expr = _build_i1_i5_expr()
    bytecode = compile_expr_to_mpb(expr, IDI_REG_LAYOUT)

    registers = _make_registers()
    registers[IDI_REG_LAYOUT["state_size"]] = 100
    registers[IDI_REG_LAYOUT["max_state_size"]] = 2048
    registers[IDI_REG_LAYOUT["discount_fp"]] = 990_000
    registers[IDI_REG_LAYOUT["min_discount_fp"]] = 500_000
    registers[IDI_REG_LAYOUT["learning_rate_fp"]] = 100_000
    registers[IDI_REG_LAYOUT["max_learning_rate_fp"]] = 500_000
    registers[IDI_REG_LAYOUT["epsilon_decay_steps"]] = 1000
    registers[IDI_REG_LAYOUT["budget_total"]] = 512
    registers[IDI_REG_LAYOUT["min_budget_total"]] = 64

    vm = MpbVm(registers=registers)
    result = vm.execute(bytecode)
    assert result == 1


def test_i1_i5_invariants_violation() -> None:
    expr = _build_i1_i5_expr()
    bytecode = compile_expr_to_mpb(expr, IDI_REG_LAYOUT)

    registers = _make_registers()
    registers[IDI_REG_LAYOUT["state_size"]] = 8192
    registers[IDI_REG_LAYOUT["max_state_size"]] = 2048
    registers[IDI_REG_LAYOUT["discount_fp"]] = 300_000
    registers[IDI_REG_LAYOUT["min_discount_fp"]] = 500_000
    registers[IDI_REG_LAYOUT["learning_rate_fp"]] = 100_000
    registers[IDI_REG_LAYOUT["max_learning_rate_fp"]] = 500_000
    registers[IDI_REG_LAYOUT["epsilon_decay_steps"]] = 1000
    registers[IDI_REG_LAYOUT["budget_total"]] = 512
    registers[IDI_REG_LAYOUT["min_budget_total"]] = 64

    vm = MpbVm(registers=registers)
    result = vm.execute(bytecode)
    assert result == 0


def test_division_by_zero_error() -> None:
    bytecode = bytearray()
    bytecode.append(Opcode.PUSH)
    bytecode.extend(int(1).to_bytes(8, "little", signed=True))
    bytecode.append(Opcode.PUSH)
    bytecode.extend(int(0).to_bytes(8, "little", signed=True))
    bytecode.append(Opcode.DIV)
    bytecode.append(Opcode.HALT)

    vm = MpbVm(registers=_make_registers())
    try:
        vm.execute(bytes(bytecode))
    except MpbVmError as exc:
        assert exc.status is VmStatus.DIVISION_BY_ZERO
    else:
        assert False


def test_invalid_opcode_error() -> None:
    bytecode = bytes([0x99, Opcode.HALT])
    vm = MpbVm(registers=_make_registers())
    try:
        vm.execute(bytecode)
    except MpbVmError as exc:
        assert exc.status is VmStatus.INVALID_OPCODE
    else:
        assert False


def test_out_of_fuel() -> None:
    bytecode = bytearray()
    for _ in range(10):
        bytecode.append(Opcode.PUSH)
        bytecode.extend(int(1).to_bytes(8, "little", signed=True))
    bytecode.append(Opcode.HALT)

    vm = MpbVm(registers=_make_registers(), fuel=5)
    try:
        vm.execute(bytes(bytecode))
    except MpbVmError as exc:
        assert exc.status is VmStatus.OUT_OF_FUEL
    else:
        assert False


def test_unary_ops() -> None:
    expr = UnExpr(UnOp.NOT, IntLiteral(0))
    bytecode = compile_expr_to_mpb(expr, {})
    vm = MpbVm(registers=_make_registers())
    assert vm.execute(bytecode) == 1

    expr = UnExpr(UnOp.BIT_NOT, IntLiteral(0))
    bytecode = compile_expr_to_mpb(expr, {})
    vm = MpbVm(registers=_make_registers())
    assert vm.execute(bytecode) == -1
