# Nova-IAN FFI Bindings

Rust FFI bindings for Nova IVC (Incrementally Verifiable Computation) 
used in IAN's zkML training proof system.

## Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- Python 3.9+
- maturin (`pip install maturin`)

## Build

```bash
cd idi/ian/zkml/nova_ffi
maturin build --release
pip install target/wheels/nova_ian-*.whl
```

## Development Build

```bash
maturin develop
```

## Usage

```python
from nova_ian import RustNovaProver, RustNovaVerifier, RustNovaProof

# Create prover
prover = RustNovaProver(model_dimension=1000, curve=0)  # 0 = Pasta

# Execute training steps
for batch_hash, gradient_hash, weights_hash in training_data:
    epoch, new_state, new_weights = prover.step(
        epoch=prover.current_epoch,
        prev_state=prover.current_state_hash,
        batch_hash=batch_hash,
        weights_hash=weights_hash,
        gradient_hash=gradient_hash,
    )

# Finalize proof
proof = prover.finalize()

# Verify
verifier = RustNovaVerifier(expected_model_dimension=1000, expected_curve=0)
is_valid = verifier.verify(proof)
```

## Integration with Python nova.py

The Rust backend can be passed to `NovaVerifier` and `NovaProver`:

```python
from idi.ian.zkml import NovaVerifier
import nova_ian

rust_backend = nova_ian.RustNovaVerifier(1000, 0)
verifier = NovaVerifier(
    expected_model_dimension=1000,
    rust_backend=rust_backend,
)
```

## Architecture

```
Python (nova.py)          Rust (nova_ian)
┌─────────────────┐       ┌─────────────────┐
│ NovaProver      │──────▶│ RustNovaProver  │
│ NovaVerifier    │──────▶│ RustNovaVerifier│
│ NovaProof       │◀─────▶│ RustNovaProof   │
└─────────────────┘       └─────────────────┘
        │                         │
        ▼                         ▼
   Orchestration            Cryptography
   Session mgmt             SNARK proofs
   Commitments              Field arithmetic
```

## License

MIT (same as microsoft/Nova)
