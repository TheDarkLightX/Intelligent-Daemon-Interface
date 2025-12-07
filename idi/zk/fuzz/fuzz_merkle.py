#!/usr/bin/env python3
"""Fuzzing target for Merkle tree construction and verification.

Uses Atheris (Google's Python fuzzer) to find crashes and bugs in Merkle tree code.
Run with: python -m idi.zk.fuzz.fuzz_merkle
"""

import sys
import atheris

# Import after atheris to ensure proper initialization
with atheris.instrument_imports():
    from idi.zk.merkle_tree import MerkleTreeBuilder


def fuzz_merkle_build(data: bytes) -> None:
    """Fuzz MerkleTreeBuilder with arbitrary byte sequences.
    
    Parses data into key-value pairs and calls build(), verifying no crashes.
    """
    builder = MerkleTreeBuilder()
    
    # Parse data into key-value pairs
    # Format: [key_len:1][key:key_len][data_len:2][data:data_len]...
    idx = 0
    max_entries = 100  # Limit to prevent excessive memory usage
    
    try:
        while idx < len(data) and len(builder.leaves) < max_entries:
            if idx + 1 > len(data):
                break
            
            # Read key length (1 byte, max 32)
            key_len = min(data[idx] % 33, len(data) - idx - 1)
            idx += 1
            
            if key_len == 0 or idx + key_len > len(data):
                break
            
            # Read key
            key = data[idx:idx + key_len].decode('utf-8', errors='ignore')
            if not key:  # Skip empty keys
                idx += key_len
                continue
            idx += key_len
            
            # Read data length (2 bytes, max 128)
            if idx + 2 > len(data):
                break
            data_len = min(int.from_bytes(data[idx:idx+2], 'little') % 129, len(data) - idx - 2)
            idx += 2
            
            if data_len == 0 or idx + data_len > len(data):
                break
            
            # Read data
            leaf_data = data[idx:idx + data_len]
            idx += data_len
            
            # Add leaf
            builder.add_leaf(key, leaf_data)
        
        # Build tree (should not crash)
        root_hash, proofs = builder.build()
        
        # Verify all proofs (should not crash)
        for key, leaf_data in [(k, d) for k, d in zip(
            [k for k, _ in builder.leaves],
            [d for _, d in builder.leaves]
        )]:
            if key in proofs:
                proof_path = proofs[key]
                # Verification may fail, but should not crash
                builder.verify_proof(key, leaf_data, proof_path, root_hash)
    
    except (ValueError, KeyError, IndexError, UnicodeDecodeError):
        # Expected exceptions for malformed input
        pass
    except Exception as e:
        # Unexpected exceptions indicate bugs
        raise


def main() -> None:
    """Main fuzzing entry point."""
    atheris.Setup(sys.argv, fuzz_merkle_build)
    atheris.Fuzz()


if __name__ == "__main__":
    main()

