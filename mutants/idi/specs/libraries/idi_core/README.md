# IDI Core Stream Contract

This library documents the canonical stream layout that every Intelligent Daemon Interface (IDI)–ready Tau specification should follow.

## Input streams
| Name | Type | Description |
|------|------|-------------|
| `i0`-`i4` | `sbf` | Legacy price/volume/trend/profit guard/failure echo. |
| `i5` | `sbf` | `q_buy` – set when the zk-proved lookup table wants to enter. |
| `i6` | `sbf` | `q_sell` – set when the table wants to exit. |
| `i7` | `sbf` | `risk_budget_ok` – per-tick allow signal for sizing/cooldowns. |
| `i8` | `bv[5]` | `q_regime` – coarse regime identifier (mirrored for audit). |
| `i9` | `sbf` | `q_emote_positive` – art/communication cue (1 = positive). |
| `iA` | `sbf` | `q_emote_alert` – cue for anxious/critical moods. |

Offline trainers must output deterministic `.in` files for each stream inside the spec-local `inputs/` folder.

## Output streams
| Name | Purpose |
|------|---------|
| `o0`-`o13` | Unchanged minimal-core trading logic (state, timer, nonce, burns). |
| `o14`-`o18` | Mirrors for legacy inputs. |
| `o19`-`o1C` | Mirrors for `q_buy`, `q_sell`, `risk_budget_ok`, and `q_regime`. |
| `o1D`-`o1F` | Emotive outputs derived from cues plus persistence helper. |

Mirrors satisfy the Tau binary’s symmetric stream requirement and allow trace replay in demos/diagnostics.

## Helper predicates (see `idi_core.tau`)
- `idi_buy_en(t)` – true when `i5[t] & i7[t]` (buy request + risk budget).
- `idi_sell_en(t)` – equals `i6[t]`.
- `idi_mood_pos(t)` – equals `o0[t] & i9[t]` (emit mood only while executing).
- `idi_mood_alert(t)` – equals `o0[t] & iA[t]`.
- `idi_mood_memory(t)` – standard linger logic so emojis stay on-screen for ≥1 tick.

Specs can `cat specification/libraries/idi_core/idi_core.tau` alongside their main definition or simply follow the same clauses inline (as V38 now does). The goal is to keep cyclomatic complexity low and let the daemon/zk coprocessor own the hard decisions while Tau enforces invariants.***

