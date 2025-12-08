# Tau Testnet ZK Integration Bridge

This module provides a clean integration layer between IDI ZK proof infrastructure and the [Tau Testnet](https://github.com/IDNI/tau-testnet) blockchain, following SOLID principles and TDD methodology.

## Overview

The bridge module enables:
- **ZK proof verification** in transaction validation pipeline
- **ZK proof propagation** via P2P gossip protocol
- **Proof-verified state transitions** for balance updates
- **Block extension** with ZK proof commitments

## Architecture

```
idi/taunet_bridge/
├── protocols.py          # Core protocols and data models
├── adapter.py            # Bridge to IDI ZK infrastructure
├── validation.py         # Validation pipeline step
├── block_extension.py    # Block structure extension
├── gossip.py             # P2P proof propagation
├── state_integration.py  # State transition integration
└── config.py             # Configuration management
```

## Usage

### Basic Integration

```python
from idi.taunet_bridge import TauNetZkAdapter, ZkConfig, ZkValidationStep
from idi.taunet_bridge.validation import ValidationContext

# Configure ZK verification
config = ZkConfig(enabled=True, proof_system="risc0")  # Use "risc0" for production
adapter = TauNetZkAdapter(config)

# Create validation step
zk_step = ZkValidationStep(adapter, required=False)

# Validate transaction
ctx = ValidationContext(tx_hash="...", payload={"zk_proof": proof_bundle})
zk_step.run(ctx)  # Raises InvalidZkProofError if verification fails
```

### Integration with Tau Testnet sendtx

Add ZK validation to `commands/sendtx.py`:

```python
from idi.taunet_bridge import TauNetZkAdapter, ZkConfig, ZkValidationStep
from idi.taunet_bridge.validation import ValidationContext

# In queue_transaction() after BLS verification:
if zk_config.enabled and payload.get("zk_proof"):
    zk_step = ZkValidationStep(get_zk_verifier())
    ctx = ValidationContext(tx_hash=tx_message_id, payload=payload)
    zk_step.run(ctx)
```

### P2P Proof Propagation

```python
from idi.taunet_bridge.gossip import ZkGossipProtocol, TAU_PROTOCOL_ZK_PROOFS

# Initialize gossip protocol
zk_gossip = ZkGossipProtocol(adapter, network_service.gossip)

# Broadcast proof
await zk_gossip.broadcast_proof(proof_bundle)

# Handle incoming proof
proof = await zk_gossip.handle_proof(data)
```

## Configuration

```python
from idi.taunet_bridge.config import ZkConfig

config = ZkConfig(
    enabled=True,              # Enable ZK verification
    proof_system="risc0",      # "stub" (testing) or "risc0" (production)
    require_proofs=False,      # Require proofs for all txs
    merkle_threshold=100,      # Use Merkle for tables > 100 entries
)
```

## Testing

Run all bridge tests:

```bash
pytest idi/taunet_bridge/tests/ -v
```

## License

This integration respects the [IDNI AG license](https://github.com/IDNI/tau-testnet/blob/main/LICENSE.md) of Tau Testnet, creating a separate bridge module without modifying core Tau Testnet files unnecessarily.

