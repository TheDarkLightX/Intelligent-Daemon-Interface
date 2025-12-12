# Tau Testnet ZK Integration Guide

This guide explains how to enable and use the IDI ZK proving integration with Tau Testnet.

## Overview

- ZK verification is optional and feature-flagged.
- All ZK logic lives in `idi/taunet_bridge/`; Tau Testnet changes are minimal hooks.
- Proof systems supported: `stub` (hash-only) and `risc0` (zkVM).

## Enabling ZK

Set environment variables before running Tau Testnet:

```bash
export ZK_ENABLED=1            # enable ZK verification
export ZK_REQUIRE_PROOFS=0     # 1 to require proofs on all txs
export ZK_PROOF_SYSTEM=stub    # or risc0
```

## Components

- `idi/taunet_bridge/integration_config.py`: maps env → `ZkConfig`, provides `get_zk_verifier()`.
- `commands/sendtx.py`: optional ZK validation after BLS checks.
- `commands/createblock.py`: optional `BlockZkExtension` placeholder for commitments.
- `network/service.py`: optional ZK proof gossip (publish/subscribe).
- `chain_state.py`: optional ZK-verified balance updates.

## Transaction Flow with ZK

1. Client submits `sendtx` JSON payload, optionally including `zk_proof`.
2. `sendtx` validates signature/sequence, then runs ZK validation if enabled.
3. On success, transaction is queued in mempool.
4. Blocks may include ZK commitments (future extension).
5. Network may propagate proofs via gossip (if enabled).

## Minimal Payload Example

```json
{
  "from": "<bls_pubkey_hex_96>",
  "sender_pubkey": "<bls_pubkey_hex_96>",
  "sequence_number": 0,
  "expiration_time": 9999999999,
  "operations": {},
  "fee_limit": 0,
  "signature": "00",
  "zk_proof": {"stub": true}
}
```

## Hooks Summary

- `sendtx`: runs `ZkValidationStep` (feature-flagged).
- `createblock`: attaches optional `BlockZkExtension` (backward compatible).
- `network/service`: registers ZK gossip protocol when enabled.
- `chain_state`: `update_balances_after_transfer_verified` available for verified transitions.

## Testing

```bash
# Run bridge integration tests
python3 -m pytest idi/taunet_bridge/tests/test_sendtx_integration.py -v

# End-to-end (stubbed) integration tests
python3 -m pytest idi/taunet_bridge/tests/test_e2e_tau_integration.py -v
```

## License

Tau Testnet is under IDNI AG license (allows modification). ZK integration code remains in `idi/taunet_bridge/` and is permissively licensed; include IDNI AG notice when distributing Tau Testnet with these hooks.

