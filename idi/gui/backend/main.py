"""IDI Synth Studio - Backend API Server.

FastAPI server providing REST endpoints for the GUI frontend.
Integrates with the Auto-QAgent synthesis pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# IDI imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from idi.devkit.experimental.auto_qagent import (
    AutoQAgentGoalSpec,
    load_goal_spec,
    run_auto_qagent_synth,
)
from idi.devkit.experimental.sape_q_patch import QAgentPatch
from idi.gui.backend.services.presets import PresetService
from idi.gui.backend.services.invariants import InvariantService
from idi.gui.backend.services.macros import MacroService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for active runs
active_runs: Dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("IDI Synth Studio backend starting...")
    yield
    logger.info("IDI Synth Studio backend shutting down...")


app = FastAPI(
    title="IDI Synth Studio",
    description="Backend API for the IDI parameterization GUI",
    version="0.1.0",
    lifespan=lifespan,
)


def _parse_cors_origins(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


# CORS for Tauri frontend
_cors_origins_env = os.environ.get("IDI_GUI_CORS_ORIGINS")
if _cors_origins_env is None:
    cors_origins = [
        "tauri://localhost",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
else:
    cors_origins = _parse_cors_origins(_cors_origins_env)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
preset_service = PresetService()
invariant_service = InvariantService()
macro_service = MacroService()


# ============================================================================
# Pydantic Models
# ============================================================================


class MacroValue(BaseModel):
    """A macro control value."""
    id: str
    value: float = Field(ge=0.0, le=1.0)


class GoalSpecRequest(BaseModel):
    """Request to create/update a goal spec."""
    agent_family: str = "qagent"
    profiles: List[str] = ["conservative"]
    packs_include: List[str] = ["qagent_base"]
    packs_extra: List[str] = []
    objectives: List[Dict[str, str]] = []
    training_envs: List[Dict[str, Any]] = []
    budget_max_agents: int = 8
    budget_max_generations: int = 3
    budget_max_episodes: int = 64
    budget_wallclock_hours: float = 0.5
    eval_mode: str = "synthetic"
    num_final_agents: int = 3


class RunRequest(BaseModel):
    """Request to start a synthesis run."""
    goal_spec: Dict[str, Any]
    mode: str = "preview"  # preview or full


class PresetResponse(BaseModel):
    """Preset metadata response."""
    id: str
    name: str
    description: str
    icon: str
    tags: List[str]
    difficulty: str


class InvariantStatus(BaseModel):
    """Invariant check result."""
    id: str
    label: str
    ok: bool
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None


class MacroDefinition(BaseModel):
    """Macro control definition."""
    id: str
    label: str
    description: str
    default: float
    effects: List[str]


class CandidateResult(BaseModel):
    """Synthesis candidate result."""
    id: str
    metrics: Dict[str, float]
    params: Dict[str, Any]
    invariants_ok: bool


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "IDI Synth Studio", "version": "0.1.0"}


# ----------------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------------


@app.get("/api/presets", response_model=List[PresetResponse])
async def list_presets():
    """List all available presets."""
    presets = preset_service.list_all()
    return [
        PresetResponse(
            id=p.id,
            name=p.name,
            description=p.description,
            icon=p.icon,
            tags=list(p.tags),
            difficulty=p.difficulty,
        )
        for p in presets
    ]


@app.get("/api/presets/{preset_id}")
async def get_preset(preset_id: str):
    """Get a specific preset with its goal spec."""
    preset = preset_service.get(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")
    
    goal_spec = preset_service.load_goal_spec(preset_id)
    return {
        "preset": PresetResponse(
            id=preset.id,
            name=preset.name,
            description=preset.description,
            icon=preset.icon,
            tags=list(preset.tags),
            difficulty=preset.difficulty,
        ),
        "goal_spec": goal_spec,
    }


# ----------------------------------------------------------------------------
# Macros
# ----------------------------------------------------------------------------


@app.get("/api/macros", response_model=List[MacroDefinition])
async def list_macros():
    """List all available macro controls."""
    macros = macro_service.list_all()
    return [
        MacroDefinition(
            id=m.id,
            label=m.label,
            description=m.description,
            default=m.default,
            effects=m.effects,
        )
        for m in macros
    ]


@app.post("/api/macros/apply")
async def apply_macros(macros: List[MacroValue], base_spec: Dict[str, Any]):
    """Apply macro values to a goal spec."""
    result = macro_service.apply_all(
        {m.id: m.value for m in macros},
        base_spec,
    )
    return {"goal_spec": result}


@app.post("/api/macros/preview")
async def preview_macros(macros: List[MacroValue], base_spec: Dict[str, Any]):
    """Preview what changes macros would make."""
    changes = macro_service.preview_changes(
        {m.id: m.value for m in macros},
        base_spec,
    )
    return {"changes": changes}


# ----------------------------------------------------------------------------
# Invariants
# ----------------------------------------------------------------------------


@app.post("/api/invariants/check", response_model=List[InvariantStatus])
async def check_invariants(goal_spec: Dict[str, Any]):
    """Check all invariants for a goal spec."""
    results = invariant_service.check_all(goal_spec)
    return [
        InvariantStatus(
            id=r["id"],
            label=r["label"],
            ok=r["ok"],
            message=r["message"],
            value=r.get("value"),
            threshold=r.get("threshold"),
        )
        for r in results
    ]


@app.get("/api/invariants/descriptions")
async def get_invariant_descriptions():
    """Get human-readable descriptions of all invariants."""
    return invariant_service.get_descriptions()


# ----------------------------------------------------------------------------
# Synthesis Runs
# ----------------------------------------------------------------------------


@app.post("/api/runs/start")
async def start_run(request: RunRequest):
    """Start a new synthesis run."""
    run_id = f"run_{int(time.time() * 1000)}"
    
    # Initialize run state
    active_runs[run_id] = {
        "id": run_id,
        "status": "starting",
        "progress": 0,
        "candidates": [],
        "logs": [],
        "started_at": time.time(),
        "goal_spec": request.goal_spec,
        "mode": request.mode,
    }
    
    # Start async task
    asyncio.create_task(_execute_run(run_id))
    
    return {"run_id": run_id, "status": "started"}


@app.get("/api/runs/{run_id}")
async def get_run_status(run_id: str):
    """Get the status of a run."""
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    return active_runs[run_id]


@app.post("/api/runs/{run_id}/stop")
async def stop_run(run_id: str):
    """Stop a running synthesis."""
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    active_runs[run_id]["status"] = "stopping"
    return {"status": "stopping"}


async def _execute_run(run_id: str):
    """Execute synthesis run in background."""
    run = active_runs[run_id]
    
    try:
        run["status"] = "running"
        run["logs"].append({"time": time.time(), "message": "Starting synthesis..."})
        
        # Parse goal spec
        goal_spec = AutoQAgentGoalSpec.from_dict(run["goal_spec"])
        
        # Run synthesis (this is blocking, should be in thread pool)
        def run_synth():
            return run_auto_qagent_synth(goal_spec)
        
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, run_synth)
        
        # Process results
        candidates = []
        for patch, score in results:
            candidates.append({
                "id": patch.identifier,
                "metrics": {
                    "reward": score[0] if len(score) > 0 else 0,
                    "risk": score[1] if len(score) > 1 else 0,
                    "complexity": score[2] if len(score) > 2 else 0,
                },
                "params": {
                    "num_price_bins": patch.num_price_bins,
                    "num_inventory_bins": patch.num_inventory_bins,
                    "learning_rate": patch.learning_rate,
                    "discount_factor": patch.discount_factor,
                    "epsilon_start": patch.epsilon_start,
                    "epsilon_end": patch.epsilon_end,
                    "epsilon_decay_steps": patch.epsilon_decay_steps,
                },
            })
        
        run["candidates"] = candidates
        run["status"] = "completed"
        run["progress"] = 100
        run["completed_at"] = time.time()
        run["logs"].append({"time": time.time(), "message": f"Completed with {len(candidates)} candidates"})
        
    except Exception as e:
        run["status"] = "failed"
        run["error"] = str(e)
        run["logs"].append({"time": time.time(), "message": f"Error: {e}"})
        logger.exception(f"Run {run_id} failed")


# ----------------------------------------------------------------------------
# WebSocket for real-time updates
# ----------------------------------------------------------------------------


@app.websocket("/ws/runs/{run_id}")
async def run_websocket(websocket: WebSocket, run_id: str):
    """WebSocket for real-time run updates."""
    await websocket.accept()
    
    try:
        last_update = 0
        while True:
            if run_id not in active_runs:
                await websocket.send_json({"error": "Run not found"})
                break
            
            run = active_runs[run_id]
            current_time = time.time()
            
            # Send update
            await websocket.send_json({
                "status": run["status"],
                "progress": run["progress"],
                "candidates_count": len(run["candidates"]),
                "logs": run["logs"][-10:],  # Last 10 logs
            })
            
            if run["status"] in ("completed", "failed", "stopped"):
                break
            
            await asyncio.sleep(0.5)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for run {run_id}")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
