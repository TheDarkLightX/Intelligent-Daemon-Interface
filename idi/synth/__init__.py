from __future__ import annotations

"""Stable synth namespace (initial scaffold).

This module re-exports selected experimental APIs under a more stable path.
The goal is to allow callers to depend on `idi.synth` while the underlying
implementations can evolve.
"""

from idi.devkit.experimental.agent_patch import (
    AgentPatch,
    AgentPatchMeta,
    agent_patch_from_dict,
    agent_patch_to_dict,
    diff_agent_patches,
    load_agent_patch,
    save_agent_patch,
    validate_agent_patch,
)
from idi.devkit.experimental.qagent_agentpatch_adapter import (
    AGENT_TYPE_QTABLE,
    agent_patch_to_qagent_patch,
    qagent_patch_to_agent_patch,
)
from idi.devkit.experimental.qagent_synth import (
    QAgentSynthConfig,
    QAgentSynthesizer,
    load_qagent_patch_preset,
)
from idi.devkit.experimental.agent_synth import (
    AgentSynthConfig,
    AgentSynthesizer,
)
from idi.devkit.experimental.auto_qagent import (
    AutoQAgentGoalSpec,
    SynthTimeoutError,
    load_goal_spec,
    run_auto_qagent_synth,
    run_auto_qagent_synth_agentpatches,
)
from idi.devkit.experimental.synth_logging import (
    SynthLogger,
    SynthRunConfig,
    SynthRunLog,
    SynthRunStats,
    generate_run_id,
)

__all__ = [
    # AgentPatch engine
    "AgentPatchMeta",
    "AgentPatch",
    "agent_patch_from_dict",
    "agent_patch_to_dict",
    "load_agent_patch",
    "save_agent_patch",
    "diff_agent_patches",
    "validate_agent_patch",
    # QAgent adapter
    "AGENT_TYPE_QTABLE",
    "qagent_patch_to_agent_patch",
    "agent_patch_to_qagent_patch",
    # Generic synth
    "AgentSynthConfig",
    "AgentSynthesizer",
    # QAgent synth
    "QAgentSynthConfig",
    "QAgentSynthesizer",
    "load_qagent_patch_preset",
    # Auto-QAgent
    "AutoQAgentGoalSpec",
    "SynthTimeoutError",
    "load_goal_spec",
    "run_auto_qagent_synth",
    "run_auto_qagent_synth_agentpatches",
    # Logging
    "SynthLogger",
    "SynthRunConfig",
    "SynthRunLog",
    "SynthRunStats",
    "generate_run_id",
]
