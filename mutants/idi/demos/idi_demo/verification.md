# Verification Checklist

## AoT summary
1. **Premise:** `q_buy` signals arrive only when the lookup prover asserts a buy and `risk_budget_ok = 1`.
2. **Reasoning:** Tau clause `o2[t] = i5[t] & i7[t] & o0[t] & o0[t-1]' & o1[t-1]'` ensures a buy occurs only on state entry and only if the timer/nonce conditions hold.
3. **Reasoning:** `o3[t] = i6[t] & o0[t-1] & o0[t]' & o1[t-1]` couples sells to IDI `q_sell` requests while still respecting exclusivity (`o1[t-1]`).
4. **Conclusion:** Whenever `risk_budget_ok = 0` (tick 3), the state machine refuses to enter (`o0[t]` stays low) even though `q_buy` briefly asserted—confirming daemon-driven budget discipline.

## Truth table spot check

| Tick | q_buy | risk_budget_ok | prev_state (o0[t-1]) | Entry allowed? |
|------|-------|----------------|----------------------|----------------|
| 1 | 1 | 1 | 0 | ✅ `o2[1]=1`, position opens. |
| 3 | 0 | 0 | 1 | ❌ timer/nonce hold, no entry despite bullish trend. |
| 5 | 1 | 1 | 0 | ✅ second entry after sell sequence. |

Emotive outputs mirror the same reasoning:
- `o1D` (positive) only high while in executing state (`o0[t]=1`) and `q_emote_positive=1`.
- `o1F` latches for ≥1 tick after an alert, proving persistence logic works (ticks 3–4).

