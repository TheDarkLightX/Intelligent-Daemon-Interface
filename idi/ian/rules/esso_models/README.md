# ESSO models for IAN rules

This folder contains ESSO-IR v1 models derived from `idi/ian/rules/ian_state_machine.tau`.

Run from the repo root (`Intelligent-Daemon-Interface/`):

NOTE: The legacy `python3 -m internal.tools.evolver ...` entrypoint is not part of this repo anymore.

## Working with these ESSO-IR YAMLs (validate / evolve / verify)

These files are already **ESSO-IR v1** models, so you can use the ESSO CLI directly:

- Setup:
  - `export PYTHONPATH=external/ESSO`
- Validate a model:
  - `python3 -m ESSO validate idi/ian/rules/esso_models/ian_state_machine_full.yaml`
- Evolve (search for smaller/faster candidates):
  - `python3 -m ESSO evolve idi/ian/rules/esso_models/ian_state_machine_control.yaml --generations 25 --population 20 --output verification/_ian_esso_run`
- Verify a candidate against a reference (refinement/no-extra behaviors):
  - `python3 -m ESSO verify verification/_ian_esso_run/best.yaml --reference idi/ian/rules/esso_models/ian_state_machine_control.yaml`

## REQ → verified kernels workflow (Foundry)

Separately, the supported in-tree Foundry workflow (REQ → verified kernels) is driven by:

- REQ specs in `internal/esso/requirements/*.req.yaml`
- Runner scripts in `internal/esso/scripts/` (see `internal/esso/ESSO_WORKFLOW.md`)

Common commands:

- Validate all REQs:
  - `python3 internal/esso/scripts/esso_runner.py validate`
- Verify (SMT; add `--full` for z3+cvc5):
  - `python3 internal/esso/scripts/esso_runner.py verify --full`
- Compile REQs → Python kernels (writes into `idi/ian/network/kernels/`):
  - `python3 internal/esso/scripts/esso_runner.py compile`

If you need to *regenerate* Tau-derived ESSO-IR models, you’ll need to do that out-of-tree (or add
an extractor) and update the YAMLs in this folder.

Notes:
- The `*_control.yaml` model intentionally observes only the control-plane outputs
  (`registered`, `log_committed`, `upgraded`, `upgrade_count`) to enable aggressive
  state-space reduction (e.g., dropping unobserved data vars).
- The auto-minimized result checked into this folder is:
  - `idi/ian/rules/esso_models/ian_state_machine_control_min.yaml`
