from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from typing import Dict, List, Union


MAX_STACK = 64
MAX_REGISTERS = 32
DEFAULT_FUEL = 10_000


class VmStatus(Enum):
    RUNNING = auto()
    HALTED = auto()
    OUT_OF_FUEL = auto()
    STACK_OVERFLOW = auto()
    STACK_UNDERFLOW = auto()
    DIVISION_BY_ZERO = auto()
    INVALID_OPCODE = auto()
    UNEXPECTED_END = auto()


class MpbVmError(Exception):
    def __init__(self, status: VmStatus, message: str | None = None) -> None:
        self.status = status
        super().__init__(message or status.name)


class Opcode(IntEnum):
    PUSH = 0x01
    POP = 0x02
    DUP = 0x03
    SWAP = 0x04
    LOAD_REG = 0x10
    ADD = 0x20
    SUB = 0x21
    MUL = 0x22
    DIV = 0x23
    MOD = 0x24
    NEG = 0x25
    ABS = 0x26
    EQ = 0x30
    NE = 0x31
    LT = 0x32
    LE = 0x33
    GT = 0x34
    GE = 0x35
    AND = 0x40
    OR = 0x41
    NOT = 0x42
    BIT_AND = 0x50
    BIT_OR = 0x51
    BIT_XOR = 0x52
    BIT_NOT = 0x53
    SHL = 0x54
    SHR = 0x55
    HALT = 0xFF


Value = int


@dataclass
class MpbVm:
    registers: List[int]
    fuel: int = DEFAULT_FUEL

    def execute(self, bytecode: bytes) -> int:
        if len(self.registers) > MAX_REGISTERS:
            raise ValueError("too many registers")

        stack: List[Value] = []
        ip = 0
        fuel = self.fuel
        n = len(bytecode)

        while True:
            if fuel == 0:
                raise MpbVmError(VmStatus.OUT_OF_FUEL)
            fuel -= 1

            if ip >= n:
                if stack:
                    return stack.pop()
                return 0

            opcode = bytecode[ip]
            ip += 1

            try:
                op = Opcode(opcode)
            except ValueError as exc:
                raise MpbVmError(VmStatus.INVALID_OPCODE, f"invalid opcode {opcode:#x}") from exc

            if op is Opcode.PUSH:
                if ip + 8 > n:
                    raise MpbVmError(VmStatus.UNEXPECTED_END)
                value = int.from_bytes(bytecode[ip : ip + 8], "little", signed=True)
                ip += 8
                self._push(stack, value)

            elif op is Opcode.POP:
                self._pop(stack)

            elif op is Opcode.DUP:
                value = self._peek(stack)
                self._push(stack, value)

            elif op is Opcode.SWAP:
                if len(stack) < 2:
                    raise MpbVmError(VmStatus.STACK_UNDERFLOW)
                stack[-1], stack[-2] = stack[-2], stack[-1]

            elif op is Opcode.LOAD_REG:
                if ip >= n:
                    raise MpbVmError(VmStatus.UNEXPECTED_END)
                reg_index = bytecode[ip]
                ip += 1
                if reg_index >= len(self.registers):
                    raise MpbVmError(VmStatus.INVALID_OPCODE, "register index out of range")
                self._push(stack, int(self.registers[reg_index]))

            elif op in (Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD):
                self._exec_arith(stack, op)

            elif op in (Opcode.EQ, Opcode.NE, Opcode.LT, Opcode.LE, Opcode.GT, Opcode.GE):
                self._exec_cmp(stack, op)

            elif op in (Opcode.AND, Opcode.OR):
                self._exec_bool_bin(stack, op)

            elif op is Opcode.NOT:
                self._exec_not(stack)

            elif op in (Opcode.BIT_AND, Opcode.BIT_OR, Opcode.BIT_XOR, Opcode.SHL, Opcode.SHR):
                self._exec_bit_bin(stack, op)

            elif op is Opcode.BIT_NOT:
                self._exec_bit_not(stack)

            elif op is Opcode.NEG:
                if not stack:
                    raise MpbVmError(VmStatus.STACK_UNDERFLOW)
                value = stack.pop()
                stack.append(self._sat_neg(value))

            elif op is Opcode.ABS:
                if not stack:
                    raise MpbVmError(VmStatus.STACK_UNDERFLOW)
                value = stack.pop()
                if value == -2**63:
                    stack.append(value)
                else:
                    stack.append(abs(value))

            elif op is Opcode.HALT:
                if stack:
                    return stack.pop()
                return 0

            else:
                raise MpbVmError(VmStatus.INVALID_OPCODE, f"unsupported opcode {opcode:#x}")

    def _push(self, stack: List[Value], value: Value) -> None:
        if len(stack) >= MAX_STACK:
            raise MpbVmError(VmStatus.STACK_OVERFLOW)
        stack.append(int(value))

    def _pop(self, stack: List[Value]) -> Value:
        if not stack:
            raise MpbVmError(VmStatus.STACK_UNDERFLOW)
        return stack.pop()

    def _peek(self, stack: List[Value]) -> Value:
        if not stack:
            raise MpbVmError(VmStatus.STACK_UNDERFLOW)
        return stack[-1]

    def _pop2(self, stack: List[Value]) -> tuple[Value, Value]:
        if len(stack) < 2:
            raise MpbVmError(VmStatus.STACK_UNDERFLOW)
        b = stack.pop()
        a = stack.pop()
        return a, b

    def _sat_add(self, a: int, b: int) -> int:
        result = a + b
        if (a > 0 and b > 0 and result < 0) or result > 2**63 - 1:
            return 2**63 - 1
        if (a < 0 and b < 0 and result > 0) or result < -2**63:
            return -2**63
        return result

    def _sat_sub(self, a: int, b: int) -> int:
        return self._sat_add(a, -b)

    def _sat_mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        result = a * b
        if result > 2**63 - 1:
            return 2**63 - 1
        if result < -2**63:
            return -2**63
        return result

    def _sat_neg(self, a: int) -> int:
        if a == -2**63:
            return a
        return -a

    def _exec_arith(self, stack: List[Value], op: Opcode) -> None:
        a, b = self._pop2(stack)
        if op is Opcode.ADD:
            res = self._sat_add(a, b)
        elif op is Opcode.SUB:
            res = self._sat_sub(a, b)
        elif op is Opcode.MUL:
            res = self._sat_mul(a, b)
        elif op is Opcode.DIV:
            if b == 0:
                raise MpbVmError(VmStatus.DIVISION_BY_ZERO)
            res = int(a // b)
        elif op is Opcode.MOD:
            if b == 0:
                raise MpbVmError(VmStatus.DIVISION_BY_ZERO)
            res = int(a % b)
        else:
            raise MpbVmError(VmStatus.INVALID_OPCODE)
        stack.append(res)

    def _exec_cmp(self, stack: List[Value], op: Opcode) -> None:
        a, b = self._pop2(stack)
        if op is Opcode.EQ:
            res = 1 if a == b else 0
        elif op is Opcode.NE:
            res = 1 if a != b else 0
        elif op is Opcode.LT:
            res = 1 if a < b else 0
        elif op is Opcode.LE:
            res = 1 if a <= b else 0
        elif op is Opcode.GT:
            res = 1 if a > b else 0
        elif op is Opcode.GE:
            res = 1 if a >= b else 0
        else:
            raise MpbVmError(VmStatus.INVALID_OPCODE)
        stack.append(res)

    def _exec_bool_bin(self, stack: List[Value], op: Opcode) -> None:
        a, b = self._pop2(stack)
        a_bool = a != 0
        b_bool = b != 0
        if op is Opcode.AND:
            res = 1 if (a_bool and b_bool) else 0
        elif op is Opcode.OR:
            res = 1 if (a_bool or b_bool) else 0
        else:
            raise MpbVmError(VmStatus.INVALID_OPCODE)
        stack.append(res)

    def _exec_not(self, stack: List[Value]) -> None:
        a = self._pop(stack)
        res = 0 if a != 0 else 1
        stack.append(res)

    def _exec_bit_bin(self, stack: List[Value], op: Opcode) -> None:
        a, b = self._pop2(stack)
        if op is Opcode.BIT_AND:
            res = a & b
        elif op is Opcode.BIT_OR:
            res = a | b
        elif op is Opcode.BIT_XOR:
            res = a ^ b
        elif op is Opcode.SHL:
            shift = b if b >= 0 else 0
            if shift > 63:
                shift = 63
            res = a << shift
        elif op is Opcode.SHR:
            shift = b if b >= 0 else 0
            if shift > 63:
                shift = 63
            res = a >> shift
        else:
            raise MpbVmError(VmStatus.INVALID_OPCODE)
        stack.append(res)

    def _exec_bit_not(self, stack: List[Value]) -> None:
        a = self._pop(stack)
        stack.append(~a)


IDI_REG_LAYOUT: Dict[str, int] = {
    "state_size": 0,
    "discount_fp": 1,
    "learning_rate_fp": 2,
    "epsilon_decay_steps": 3,
    "budget_total": 4,
    "max_state_size": 5,
    "min_discount_fp": 6,
    "max_learning_rate_fp": 7,
    "min_budget_total": 8,
}


class BinOp(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    AND = auto()
    OR = auto()
    BIT_AND = auto()
    BIT_OR = auto()
    BIT_XOR = auto()
    SHL = auto()
    SHR = auto()


class UnOp(Enum):
    NEG = auto()
    NOT = auto()
    BIT_NOT = auto()


class Expr:
    pass


class IntLiteral(Expr):
    def __init__(self, value: int) -> None:
        self.value = int(value)


class Var(Expr):
    def __init__(self, name: str) -> None:
        self.name = name


class Reg(Expr):
    def __init__(self, index: int) -> None:
        self.index = index


class BinExpr(Expr):
    def __init__(self, op: BinOp, left: Expr, right: Expr) -> None:
        self.op = op
        self.left = left
        self.right = right


class UnExpr(Expr):
    def __init__(self, op: UnOp, expr: Expr) -> None:
        self.op = op
        self.expr = expr


def _resolve_vars(expr: Expr, layout: Dict[str, int]) -> Expr:
    if isinstance(expr, IntLiteral):
        return expr
    if isinstance(expr, Var):
        if expr.name not in layout:
            raise KeyError(f"unknown variable: {expr.name}")
        return Reg(layout[expr.name])
    if isinstance(expr, Reg):
        return expr
    if isinstance(expr, BinExpr):
        return BinExpr(expr.op, _resolve_vars(expr.left, layout), _resolve_vars(expr.right, layout))
    if isinstance(expr, UnExpr):
        return UnExpr(expr.op, _resolve_vars(expr.expr, layout))
    raise TypeError(f"unsupported expr: {expr!r}")


def _emit_rpn(expr: Expr, out: List[tuple[str, object]]) -> None:
    if isinstance(expr, IntLiteral):
        out.append(("PUSH", expr.value))
    elif isinstance(expr, Reg):
        out.append(("LOAD_REG", expr.index))
    elif isinstance(expr, BinExpr):
        _emit_rpn(expr.left, out)
        _emit_rpn(expr.right, out)
        out.append(("BINOP", expr.op))
    elif isinstance(expr, UnExpr):
        _emit_rpn(expr.expr, out)
        out.append(("UNOP", expr.op))
    else:
        raise TypeError(f"unsupported expr in rpn: {expr!r}")


def _binop_to_opcode(op: BinOp) -> Opcode:
    mapping: Dict[BinOp, Opcode] = {
        BinOp.ADD: Opcode.ADD,
        BinOp.SUB: Opcode.SUB,
        BinOp.MUL: Opcode.MUL,
        BinOp.DIV: Opcode.DIV,
        BinOp.MOD: Opcode.MOD,
        BinOp.EQ: Opcode.EQ,
        BinOp.NE: Opcode.NE,
        BinOp.LT: Opcode.LT,
        BinOp.LE: Opcode.LE,
        BinOp.GT: Opcode.GT,
        BinOp.GE: Opcode.GE,
        BinOp.AND: Opcode.AND,
        BinOp.OR: Opcode.OR,
        BinOp.BIT_AND: Opcode.BIT_AND,
        BinOp.BIT_OR: Opcode.BIT_OR,
        BinOp.BIT_XOR: Opcode.BIT_XOR,
        BinOp.SHL: Opcode.SHL,
        BinOp.SHR: Opcode.SHR,
    }
    return mapping[op]


def _unop_to_opcode(op: UnOp) -> Opcode:
    mapping: Dict[UnOp, Opcode] = {
        UnOp.NEG: Opcode.NEG,
        UnOp.NOT: Opcode.NOT,
        UnOp.BIT_NOT: Opcode.BIT_NOT,
    }
    return mapping[op]


def _encode_i64_le(value: int) -> bytes:
    return int(value).to_bytes(8, byteorder="little", signed=True)


def compile_expr_to_mpb(expr: Expr, layout: Dict[str, int]) -> bytes:
    resolved = _resolve_vars(expr, layout)
    rpn: List[tuple[str, object]] = []
    _emit_rpn(resolved, rpn)

    buf = bytearray()
    for kind, arg in rpn:
        if kind == "PUSH":
            buf.append(Opcode.PUSH)
            buf.extend(_encode_i64_le(int(arg)))
        elif kind == "LOAD_REG":
            buf.append(Opcode.LOAD_REG)
            buf.append(int(arg))
        elif kind == "BINOP":
            buf.append(_binop_to_opcode(arg))
        elif kind == "UNOP":
            buf.append(_unop_to_opcode(arg))
        else:
            raise ValueError(f"unknown rpn element kind: {kind}")
    buf.append(Opcode.HALT)
    return bytes(buf)
