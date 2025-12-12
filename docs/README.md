# IDI Documentation Index

This index helps you navigate the Intelligent Daemon Interface (IDI) documentation like a small wiki. Use it as the main entry point when exploring the project.

---

## 1. Getting Started

- **Project Overview & Quick Start**  
  See the root [`README.md`](../README.md) for:
  - What IDI is and who it is for.
  - Local quick start for creating and training an agent.
  - Tau Testnet integration quick start.

- **Tau Testnet Integration**  
  [`idi/taunet_bridge/README.md`](../idi/taunet_bridge/README.md) – how to plug IDI's proof infrastructure into Tau Testnet.

---

## 2. Concepts & Architecture

- **IDI Architecture**  
  [`IDI_IAN_ARCHITECTURE.md`](IDI_IAN_ARCHITECTURE.md) – high-level system architecture and component relationships.

- **Modular Synth & Auto-QAgent**  
  [`MODULAR_SYNTH_AND_AUTO_QAGENT.md`](MODULAR_SYNTH_AND_AUTO_QAGENT.md) – modular synth backend and Auto-QAgent automation layer.

- **idi.synth API**  
  [`IDI_SYNTH_API.md`](IDI_SYNTH_API.md) – stable API surface for synth and Auto-QAgent.

- **Parameterization & Visuals**  
  [`PARAMETERIZATION_AND_SYNTH_VISUALS.md`](PARAMETERIZATION_AND_SYNTH_VISUALS.md) – visual explanation of parameterization and the synth search space.

- **Threat Model**  
  [`THREAT_MODEL.md`](THREAT_MODEL.md) – security assumptions and adversarial model.

- **Formal Aggregation Plan**  
  [`FORMAL_AGGREGATION_PLAN.md`](FORMAL_AGGREGATION_PLAN.md) – formalization and aggregation strategy for agent behavior.

- **Verification Summary**  
  [`VERIFICATION_SUMMARY.md`](VERIFICATION_SUMMARY.md) – overview of verification layers and guarantees.

- **Simulation Results**  
  [`SIMULATION_RESULTS.md`](SIMULATION_RESULTS.md) – selected experimental and simulation outcomes.

- **TauNet Deflationary Ecosystem Deep Dive**  
  [`TauNet_Deflationary_Ecosystem_DeepDive.md`](TauNet_Deflationary_Ecosystem_DeepDive.md) – background economics and ecosystem analysis.

---

## 3. Agent Factory & Patterns

These live primarily under `idi/devkit/tau_factory/` but are central to how agents are authored.

- **Tau Agent Factory Guide**  
  [`tau_factory/README.md`](../idi/devkit/tau_factory/README.md) – how to use the agent factory and wizard.

- **Pattern Landscape**  
  [`PATTERN_LANDSCAPE.md`](../idi/devkit/tau_factory/PATTERN_LANDSCAPE.md) – catalogue of available agent patterns.

- **Implementation Status**  
  [`IMPLEMENTATION_STATUS.md`](../idi/devkit/tau_factory/IMPLEMENTATION_STATUS.md) – current pattern implementation status.

- **Ensemble & DAO Patterns**  
  [`ENSEMBLE_PATTERNS.md`](../idi/devkit/tau_factory/ENSEMBLE_PATTERNS.md) – voting and ensemble constructions.

- **Hierarchical FSMs**  
  [`HIERARCHICAL_FSM_DESIGN.md`](../idi/devkit/tau_factory/HIERARCHICAL_FSM_DESIGN.md) – supervisor/worker, orthogonal regions, composition.

- **Bitvector Patterns**  
  [`BITVECTOR_PATTERNS.md`](../idi/devkit/tau_factory/BITVECTOR_PATTERNS.md) – weighted voting and time-lock patterns.

- **Hex Pattern**  
  [`HEX_SUMMARY.md`](../idi/devkit/tau_factory/HEX_SUMMARY.md) – hex-staking and related constructs.

- **High-Priority Patterns**  
  [`HIGH_PRIORITY_PATTERNS_COMPLETE.md`](../idi/devkit/tau_factory/HIGH_PRIORITY_PATTERNS_COMPLETE.md) – summary of critical patterns.

- **Limitations**  
  [`LIMITATIONS_AND_WHY.md`](../idi/devkit/tau_factory/LIMITATIONS_AND_WHY.md) – what cannot be expressed and why.

---

## 4. Training, Evaluation & Lookup

- **Lookup Q-Table Brief**  
  [`lookup_q_table_brief.md`](lookup_q_table_brief.md) – notes on Q-table lookup and related design.

- **Simulation & Evaluation**  
  For training configs and evaluation, see tests and configs under:
  - `idi/training/python/`
  - `idi/training/python/tests/`

---

## 5. Verification & ZK Integration

For the current repository:

- **Zero-Knowledge & Verification Overview**  
  [`VERIFICATION_SUMMARY.md`](VERIFICATION_SUMMARY.md) – high-level description of verification and ZK layers.

- **ZK Integration (Risc0)**  
  See the `zk/` directory and root `README.md` sections on ZK integration for details on Risc0 usage.

---

## 6. How to Use This Index

- Start here when exploring the project.
- Use section headings (Concepts, Patterns, Verification, etc.) as a mental map.
- When you add new documentation, link it from this index so the documentation stays discoverable and "wiki-like".
