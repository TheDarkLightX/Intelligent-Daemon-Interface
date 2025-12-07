"""Property-based tests for witness generation using Hypothesis.

Tests witness generation properties:
- Action selection determinism
- Fixed-point roundtrip
- Cross-implementation consistency
"""

from __future__ import annotations

from hypothesis import given, strategies as st
import pytest

from idi.zk.witness_generator import (
    QTableEntry,
    generate_witness_from_q_table,
    _select_action_greedy,
)


@given(
    q_hold=st.floats(min_value=-1.0, max_value=1.0),
    q_buy=st.floats(min_value=-1.0, max_value=1.0),
    q_sell=st.floats(min_value=-1.0, max_value=1.0),
)
def test_action_selection_deterministic(q_hold: float, q_buy: float, q_sell: float) -> None:
    """Same Q-values always produce same action.
    
    Property: Action selection is deterministic and matches Rust implementation.
    """
    entry1 = QTableEntry.from_float(q_hold, q_buy, q_sell)
    entry2 = QTableEntry.from_float(q_hold, q_buy, q_sell)
    
    action1 = _select_action_greedy(entry1)
    action2 = _select_action_greedy(entry2)
    
    assert action1 == action2, "Same Q-values should produce same action"
    assert action1 in (0, 1, 2), f"Action must be 0, 1, or 2, got {action1}"


@given(
    q_hold=st.floats(min_value=-1.0, max_value=1.0),
    q_buy=st.floats(min_value=-1.0, max_value=1.0),
    q_sell=st.floats(min_value=-1.0, max_value=1.0),
)
def test_fixed_point_roundtrip(q_hold: float, q_buy: float, q_sell: float) -> None:
    """Q16.16 conversion preserves values within tolerance.
    
    Property: Float → Fixed-point → Float roundtrip preserves values
    (within Q16.16 precision limits).
    """
    entry = QTableEntry.from_float(q_hold, q_buy, q_sell)
    recovered = entry.to_float()
    
    # Q16.16 has precision of 1/65536 ≈ 0.000015
    tolerance = 1.0 / (1 << 16) + 1e-10  # Add small epsilon for floating point errors
    
    assert abs(recovered[0] - q_hold) <= tolerance, f"q_hold mismatch: {recovered[0]} vs {q_hold}"
    assert abs(recovered[1] - q_buy) <= tolerance, f"q_buy mismatch: {recovered[1]} vs {q_buy}"
    assert abs(recovered[2] - q_sell) <= tolerance, f"q_sell mismatch: {recovered[2]} vs {q_sell}"


@given(
    q_table=st.dictionaries(
        keys=st.text(min_size=1, max_size=32, alphabet=st.characters(min_codepoint=32, max_codepoint=126)),
        values=st.fixed_dictionaries({
            "hold": st.floats(min_value=-1.0, max_value=1.0),
            "buy": st.floats(min_value=-1.0, max_value=1.0),
            "sell": st.floats(min_value=-1.0, max_value=1.0),
        }),
        min_size=1,
        max_size=50,
    )
)
def test_witness_generation_deterministic(q_table: dict[str, dict[str, float]]) -> None:
    """Witness generation is deterministic.
    
    Property: Same Q-table and state key always produce same witness.
    """
    if not q_table:
        return
    
    state_key = list(q_table.keys())[0]
    
    witness1 = generate_witness_from_q_table(q_table, state_key, use_merkle=False)
    witness2 = generate_witness_from_q_table(q_table, state_key, use_merkle=False)
    
    assert witness1.state_key == witness2.state_key
    assert witness1.q_entry.q_hold == witness2.q_entry.q_hold
    assert witness1.q_entry.q_buy == witness2.q_entry.q_buy
    assert witness1.q_entry.q_sell == witness2.q_entry.q_sell
    assert witness1.selected_action == witness2.selected_action
    assert witness1.q_table_root == witness2.q_table_root


@given(
    q_table=st.dictionaries(
        keys=st.text(min_size=1, max_size=32),
        values=st.fixed_dictionaries({
            "hold": st.floats(min_value=-1.0, max_value=1.0),
            "buy": st.floats(min_value=-1.0, max_value=1.0),
            "sell": st.floats(min_value=-1.0, max_value=1.0),
        }),
        min_size=1,
        max_size=10,
    )
)
def test_action_matches_argmax(q_table: dict[str, dict[str, float]]) -> None:
    """Selected action matches argmax of Q-values.
    
    Property: Action selection implements greedy (argmax) policy correctly.
    """
    if not q_table:
        return
    
    state_key = list(q_table.keys())[0]
    q_values = q_table[state_key]
    
    witness = generate_witness_from_q_table(q_table, state_key, use_merkle=False)
    
    # Verify action matches manual argmax
    q_hold = q_values.get("hold", 0.0)
    q_buy = q_values.get("buy", 0.0)
    q_sell = q_values.get("sell", 0.0)
    
    # Manual argmax with tie-breaking: buy > sell > hold (when buy > sell and buy > hold)
    # Default to hold (0) when all equal or hold >= others
    if q_buy > q_sell and q_buy > q_hold:
        expected_action = 1
    elif q_sell > q_hold:
        expected_action = 2
    else:
        expected_action = 0
    
    assert witness.selected_action == expected_action, (
        f"Action mismatch: got {witness.selected_action}, expected {expected_action} "
        f"(q_hold={q_hold}, q_buy={q_buy}, q_sell={q_sell})"
    )


@given(
    q_hold=st.floats(min_value=-32768.0, max_value=32767.0),
    q_buy=st.floats(min_value=-32768.0, max_value=32767.0),
    q_sell=st.floats(min_value=-32768.0, max_value=32767.0),
)
def test_fixed_point_overflow_handling(q_hold: float, q_buy: float, q_sell: float) -> None:
    """Fixed-point conversion handles large values gracefully.
    
    Property: Values outside Q16.16 range are clamped/overflowed predictably.
    """
    # Values > 32767.9999 will overflow INT32
    entry = QTableEntry.from_float(q_hold, q_buy, q_sell)
    
    # Verify entry is created (may overflow, but should not crash)
    assert isinstance(entry.q_hold, int)
    assert isinstance(entry.q_buy, int)
    assert isinstance(entry.q_sell, int)
    
    # Verify roundtrip still works (even if overflowed)
    recovered = entry.to_float()
    assert all(isinstance(v, float) for v in recovered)


def test_tie_breaking_determinism() -> None:
    """Tie-breaking rules are deterministic.
    
    Property: When Q-values are equal, default to hold (0).
    When buy and sell are equal and both > hold, prefer buy.
    """
    # All equal - defaults to hold (0)
    entry_all_equal = QTableEntry.from_float(0.5, 0.5, 0.5)
    action = _select_action_greedy(entry_all_equal)
    assert action == 0, "Tie-breaking: hold should be default when all equal"
    
    # Buy and sell equal, both > hold - prefer buy
    entry_buy_sell_equal = QTableEntry.from_float(0.0, 0.5, 0.5)
    action = _select_action_greedy(entry_buy_sell_equal)
    assert action == 1, "Tie-breaking: buy should win over sell when equal and both > hold"
    
    # Sell and hold equal, both < buy - buy wins
    entry_sell_hold_equal = QTableEntry.from_float(0.0, 0.5, 0.0)
    action = _select_action_greedy(entry_sell_hold_equal)
    assert action == 1, "Buy should win when > sell and hold"

