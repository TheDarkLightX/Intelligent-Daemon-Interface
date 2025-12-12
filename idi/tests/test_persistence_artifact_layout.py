import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def _import_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_artifact_dir_strips_suffixes() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    scaled_mod = _import_module_from_path(
        "tau_q_agents_phase9_scaled_scaled_q_system",
        repo_root / "tau_q_agents/phase9_scaled/scaled_q_system.py",
    )
    full_mod = _import_module_from_path(
        "tau_q_agents_phase10_training_full_training_pipeline",
        repo_root / "tau_q_agents/phase10_training/full_training_pipeline.py",
    )

    assert full_mod.MegaQTable._artifact_dir(Path("model.tar.gz")).name == "model"

    ts = scaled_mod.TrainingSystem.__new__(scaled_mod.TrainingSystem)
    assert ts._artifact_dir(Path("model.tar.gz")).name == "model"


def test_megaqtable_persistence_directory_layout(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    full_mod = _import_module_from_path(
        "tau_q_agents_phase10_training_full_training_pipeline_2",
        repo_root / "tau_q_agents/phase10_training/full_training_pipeline.py",
    )

    base_path = tmp_path / "qtable.tar.gz"

    qt = full_mod.MegaQTable(n_states=10, n_actions=4)
    qt.q[1] = np.zeros(4)
    qt.visits[1] = np.ones(4)
    qt.episode_rewards = [1.0]
    qt.save(base_path)

    artifact_dir = tmp_path / "qtable"
    assert artifact_dir.is_dir()
    assert (artifact_dir / "metadata.json").is_file()
    assert (artifact_dir / "arrays.npz").is_file()

    # Ensure we did not create misleading suffix-based files
    assert not (tmp_path / "qtable.tar.meta.json").exists()
    assert not (tmp_path / "qtable.tar.arrays.npz").exists()

    loaded = full_mod.MegaQTable.load(base_path)
    assert loaded.n_states == 10
    assert loaded.n_actions == 4
    assert 1 in loaded.q


def test_megaqtable_legacy_suffix_layout_still_loads(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    full_mod = _import_module_from_path(
        "tau_q_agents_phase10_training_full_training_pipeline_3",
        repo_root / "tau_q_agents/phase10_training/full_training_pipeline.py",
    )

    base_path = tmp_path / "legacy.pkl"
    meta_path = base_path.with_suffix(".meta.json")
    arrays_path = base_path.with_suffix(".arrays.npz")

    np.savez_compressed(
        arrays_path,
        q_states=np.array([1], dtype=np.int64),
        q_values=np.array([[0.0, 0.0, 0.0, 0.0]]),
        visit_values=np.array([[1.0, 1.0, 1.0, 1.0]]),
        episode_rewards=np.array([1.0]),
        episode_trades=np.array([0]),
        episode_wins=np.array([0]),
        priorities=np.array([]),
    )

    arrays_digest = hashlib.sha256(arrays_path.read_bytes()).hexdigest()

    meta_path.write_text(
        json.dumps(
            {
                "version": 2,
                "format": "json+npz",
                "n_states": 10,
                "n_actions": 4,
                "lr": 0.1,
                "gamma": 0.95,
                "epsilon": 0.2,
                "total_updates": 0,
                "unique_states": 0,
                "replay_buffer": [],
                "arrays_sha256": arrays_digest,
            }
        )
    )

    loaded = full_mod.MegaQTable.load(base_path)
    assert loaded.n_states == 10
    assert loaded.n_actions == 4
    assert 1 in loaded.q
