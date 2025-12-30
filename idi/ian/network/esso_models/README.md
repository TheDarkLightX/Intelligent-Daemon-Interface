# ESSO models for `idi.ian.network`

This folder contains ESSO-IR v1 models derived from Python Enum-driven state machines in `idi/ian/network/`.

## Models

- `slot_state_fsm.yaml`
  - Source: `idi/ian/network/slotbatcher.py:SlotState` (via `valid_transitions`)

## How to work with these models

Run from the repo root (`Intelligent-Daemon-Interface/`):

NOTE: The legacy `python3 -m internal.tools.evolver ...` entrypoint is not part of this repo anymore.

These files are **ESSO-IR v1** models, so you can use the ESSO CLI directly:

- Setup:
  - `export PYTHONPATH=external/ESSO`
- Validate:
  - `python3 -m ESSO validate idi/ian/network/esso_models/slot_state_fsm.yaml`
- Evolve:
  - `python3 -m ESSO evolve idi/ian/network/esso_models/slot_state_fsm.yaml --output verification/_slot_state_run`
- Verify a candidate against the reference:
  - `python3 -m ESSO verify verification/_slot_state_run/best.yaml --reference idi/ian/network/esso_models/slot_state_fsm.yaml`

Separately, the supported in-tree Foundry workflow is REQ → verified kernels via:

- `internal/esso/requirements/*.req.yaml`
- `internal/esso/scripts/esso_runner.py` (see `internal/esso/ESSO_WORKFLOW.md`)

If you need to regenerate Enum-derived ESSO-IR models like `slot_state_fsm.yaml`,
you’ll need to do that out-of-tree (or add an extractor) and update the YAML here.

