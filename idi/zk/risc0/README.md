# Risc0 ZK Proof Infrastructure

## Overview

This directory contains the Risc0 zkVM guest programs and host programs for generating zero-knowledge proofs of Q-table lookups and manifest verification.

## Structure

```
risc0/
├── Cargo.toml              # Workspace configuration
├── methods/                 # Guest programs (zkVM)
│   ├── Cargo.toml          # Methods package
│   ├── idi-manifest/       # Manifest verification guest
│   │   ├── Cargo.toml
│   │   └── src/main.rs
│   └── idi-qtable/         # Q-table action selection guest
│       ├── Cargo.toml
│       └── src/main.rs
└── host/                    # Host program (proof generation)
    ├── Cargo.toml
    └── src/main.rs
```

## Building

### Prerequisites

Install the Risc0 toolchain:

```bash
curl -L https://risc0.github.io/gh-release/dev/risc0-rust-toolchain.toml -o risc0-rust-toolchain.toml
rustup toolchain install --path risc0-rust-toolchain.toml
```

### Build Guest Programs

```bash
cd idi/zk/risc0
cargo build --release -p idi_risc0_methods
```

This compiles both guest programs and embeds them into the methods package.

### Build Host Program

```bash
cargo build --release -p idi_risc0_host
```

## Usage

### Generate Manifest Proof

```bash
cargo run --release -p idi_risc0_host -- \
    --manifest artifacts/my_agent/artifact_manifest.json \
    --streams artifacts/my_agent/streams \
    --proof artifacts/my_agent/proof_risc0/proof.bin \
    --receipt artifacts/my_agent/proof_risc0/receipt.json
```

### Verify Proof

The receipt JSON contains:
- `digest_hex`: SHA256 hash of manifest + streams
- `method_id`: Risc0 method ID for verification
- `proof`: Path to proof binary

## Guest Programs

### idi-manifest

Verifies that manifest and stream files match their claimed hashes.

**Input**: `Vec<FileBlob>` (manifest + stream files)
**Output**: SHA256 digest of all files

### idi-qtable

Verifies Q-table action selection (argmax) matches Q-values.

**Input**: `QTableProofInput` (state, Q-entry, Merkle proof, selected action)
**Output**: SHA256 commitment to verified data

## Troubleshooting

### Compilation Errors

- **Duplicate lang item**: Ensure guest programs use `#![no_main]` and `risc0_zkvm::guest::entry!`
- **Missing dependencies**: Run `cargo build` from workspace root
- **Toolchain issues**: Verify Risc0 toolchain is installed correctly

### Runtime Errors

- **File not found**: Ensure all paths are absolute or relative to current directory
- **Proof verification fails**: Check that manifest and streams match expected hashes
- **Guest execution fails**: Verify input format matches expected structs

## Integration

See `idi/zk/INTEGRATION_GUIDE.md` for integration with training pipeline and Tau execution.

