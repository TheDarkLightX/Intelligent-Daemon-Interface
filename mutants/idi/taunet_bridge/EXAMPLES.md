# ZK Integration Examples (Tau Testnet Bridge)

This file provides example payloads and commands to exercise the ZK bridge on our side. These do NOT modify Tau Testnet; maintainers can choose to apply the minimal hooks described in `INTEGRATION_GUIDE.md`.

## Environment Flags

```bash
# Enable ZK (stub)
export ZK_ENABLED=1
export ZK_REQUIRE_PROOFS=0    # set to 1 to require proofs on every tx
export ZK_PROOF_SYSTEM=stub   # or risc0
```

## Minimal Transaction Payload (JSON)

```json
{
  "from": "<bls_pubkey_hex_96>",
  "sender_pubkey": "<bls_pubkey_hex_96>",
  "sequence_number": 0,
  "expiration_time": 9999999999,
  "operations": {},
  "fee_limit": 0,
  "signature": "00",
  "zk_proof": { "stub": true }
}
```

Save as `tx.json` and send to Tau Testnet `sendtx` (if maintainers enable the hook).

## Using the Bridge in Isolation (Stubbed)

Run bridge integration tests (no Tau deps):
```bash
python3 -m pytest idi/taunet_bridge/tests/test_sendtx_integration.py -v
python3 -m pytest idi/taunet_bridge/tests/test_e2e_tau_integration.py -v
```

## Risc0 Proof Workflow (our stack)

1. Generate witness: `idi/zk/witness_generator.py`
2. Prove (stub or risc0): `idi/zk/proof_manager.py` or Risc0 host
3. Verify: `idi/zk/proof_manager.verify_proof`

These produce a `receipt` and `proof` that can be attached to a transaction as `zk_proof`.

## Patch (for Tau maintainers)

See `INTEGRATION_GUIDE.md` for the minimal code hook locations. We keep these changes separate; maintainers can choose to apply them. Our bridge remains self-contained.

