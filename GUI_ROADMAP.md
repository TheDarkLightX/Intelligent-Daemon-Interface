# IDI/IAN GUI Improvement Roadmap

Great to hear the system is working! To take this from a "Functional MVP" to a "Production-Grade Platform," here are the recommended next steps:

## 1. 🕸️ Interactive Agent Visualization (High Impact)
**Why:** Currently, the agent is just code. Users should *see* the logic.
**Feature:** Implement a node-link diagram (using libraries like `reactflow`) to visualize the Agent's FSM, logic blocks, and input streams dynamically as they are configured in the Wizard.
**Status:** ⬜ Proposed

## 2. 💾 Persistence & Project Management
**Why:** Wizard state is lost on restart. Agents are just generated text.
**Feature:** 
- Save/Load functionality for Wizard drafts.
- A "My Agents" dashboard to view previously generated agents in `idi/practice/`.
- File system integration to read/write actual agent files.
**Status:** ⬜ Proposed

**Feature:** Add a "Deploy to Daemon" button that sends the spec to the `tau_daemon_alpha` for execution.
**Status:** ⬜ Proposed

## Recommendation
I recommend starting with **#1 (Interactive Visualization)** or **#2 (Persistence)** as they provide the most immediate value to the user experience.
