# IDI Devkit

Utilities that streamline creation of Intelligent Daemon Interface lookup tables.

## Builder CLI
```
python -m idi.devkit.builder \
  --config idi/devkit/configs/sample_config.json \
  --out build/artifacts/sample \
  --install-inputs idi/specs/V38_Minimal_Core/inputs \
  --meta owner=research --meta note="sample run"
```

Steps performed:
1. Loads the JSON config and instantiates `TrainingConfig`.
2. Runs the Python `QTrainer` to produce new `q_*` stream traces.
3. Writes streams to `<out>/streams/` and copies them into the Tau spec inputs (if `--install-inputs` provided).
4. Generates `artifact_manifest.json` with SHA-256 fingerprints of every stream and config file, ready for zk proving.
5. Emits `policy_manifest.json` summarizing the trained lookup table.

## Layer plans

Use the batch builder to run multiple configs in one go:
```
python -m idi.devkit.build_layers --plan idi/devkit/configs/layer_plan.json
```
The sample plan builds macro/micro/emotive layers (see `configs/regime_macro.json`, `regime_micro.json`, `emote_balanced.json`) and installs the macro layer directly into `idi/specs/V38_Minimal_Core/inputs/`.

## Config schema

See `configs/sample_config.json`. Every field is optional; unspecified values fall back to the defaults used by `TrainingConfig`.

## Testing

Run the existing virtualenv from `idi/training/python/.venv`:
```
idi/training/python/.venv/bin/python -m pytest idi/devkit/tests
```

