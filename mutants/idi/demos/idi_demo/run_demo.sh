#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TAU_BIN="${TAU_BIN:-$REPO_ROOT/tau_daemon_alpha/bin/tau}"
SPEC_FILE="$REPO_ROOT/idi/specs/V38_Minimal_Core/agent4_testnet_v38.tau"

if [ ! -x "$TAU_BIN" ]; then
    echo "❌ Tau binary not found at $TAU_BIN"
    exit 1
fi

WORK_DIR="$(mktemp -d)"
cleanup() { rm -rf "$WORK_DIR"; }
trap cleanup EXIT

cp "$SPEC_FILE" "$WORK_DIR/agent4_testnet_v38.tau"
mkdir -p "$WORK_DIR/inputs" "$WORK_DIR/outputs"
cp "$SCRIPT_DIR"/inputs/*.in "$WORK_DIR/inputs/"

pushd "$WORK_DIR" >/dev/null
"$TAU_BIN" < agent4_testnet_v38.tau > /dev/null
popd >/dev/null

rm -rf "$SCRIPT_DIR/outputs"
cp -r "$WORK_DIR/outputs" "$SCRIPT_DIR/outputs"

printf "\n\e[1;92mIDI demo completed.\e[0m Outputs copied to %s/outputs\n" "$SCRIPT_DIR"

