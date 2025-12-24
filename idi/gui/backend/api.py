import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import asdict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# Import internal tools
# Note: These imports depend on the PYTHONPATH being set correctly to the project root
from idi.devkit.tau_factory.wizard_controller import WizardController, WizardStep
from idi.training.python.idi_iann.config import TrainingConfig
from idi.training.python.idi_iann.trainer import QTrainer, run_training_in_thread
from idi.gui.backend.agent_manager import AgentManager, AgentMetadata
from idi.gui.backend.settings_manager import SettingsManager
from idi.ian.network.decentralized_node import DecentralizedNode, DecentralizedNodeConfig
from idi.ian.network.node import NodeIdentity
from idi.ian.models import GoalSpec, GoalID, AgentPack, Contribution, Metrics, EvaluationLimits, Thresholds
from idi.ian.idi_integration import create_idi_coordinator
from idi.ian.hooks import (
    CoordinatorHooks,
    ContributionAcceptedEvent,
    ContributionRejectedEvent,
    LeaderboardUpdatedEvent,
)
import time
import hashlib
import json

router = APIRouter()
logger = logging.getLogger("idi_api")
_agent_manager = AgentManager()
_settings_manager = SettingsManager()

_ALLOW_INCOMPLETE_WIZARD = os.environ.get("IDI_GUI_ALLOW_INCOMPLETE_WIZARD", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}

# Global IAN node instance
_node: Optional[DecentralizedNode] = None
_node_task: Optional[asyncio.Task] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None


# =============================================================================
# WebSocket Event Hub for Real-Time GUI Updates
# =============================================================================

class WebSocketEventHub:
    """Manages WebSocket connections for real-time GUI events."""
    
    def __init__(self):
        self.connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.connections.append(websocket)
        logger.info(f"Event WS connected. Total: {len(self.connections)}")
    
    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self.connections:
                self.connections.remove(websocket)
        logger.info(f"Event WS disconnected. Total: {len(self.connections)}")
    
    async def broadcast(self, event_type: str, data: Dict[str, Any]) -> int:
        """Broadcast event to all connected clients."""
        message = {
            "type": event_type,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        sent = 0
        dead_connections = []
        
        for ws in self.connections:
            try:
                await ws.send_json(message)
                sent += 1
            except Exception as e:
                logger.debug(f"Failed to send to WS: {e}")
                dead_connections.append(ws)
        
        # Cleanup dead connections
        for ws in dead_connections:
            await self.disconnect(ws)
        
        return sent


# Global event hub instance
_event_hub = WebSocketEventHub()


class WebSocketCoordinatorHooks:
    """
    Adapter that bridges sync coordinator hooks to async WebSocket broadcasts.
    
    Uses asyncio.run_coroutine_threadsafe to safely bridge from sync context.
    """
    
    def __init__(self, event_hub: WebSocketEventHub, loop: Optional[asyncio.AbstractEventLoop] = None):
        self._hub = event_hub
        self._loop = loop
    
    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the event loop (called after startup)."""
        self._loop = loop
    
    def on_contribution_accepted(self, event: ContributionAcceptedEvent) -> None:
        if not self._loop:
            logger.warning("Event loop not set for WebSocket hooks")
            return
        
        data = {
            "goal_id": event.goal_id,
            "pack_hash": event.pack_hash.hex(),
            "contributor_id": event.contributor_id,
            "score": event.score,
            "log_index": event.log_index,
            "leaderboard_position": event.leaderboard_position,
            "is_new_leader": event.is_new_leader,
            "metrics": {
                "reward": event.metrics.reward,
                "risk": event.metrics.risk,
                "complexity": event.metrics.complexity,
            },
        }
        asyncio.run_coroutine_threadsafe(
            self._hub.broadcast("contribution_accepted", data),
            self._loop,
        )
    
    def on_contribution_rejected(self, event: ContributionRejectedEvent) -> None:
        if not self._loop:
            return
        
        data = {
            "goal_id": event.goal_id,
            "pack_hash": event.pack_hash.hex(),
            "contributor_id": event.contributor_id,
            "rejection_reason": event.rejection_reason.name,
            "reason_detail": event.reason_detail,
        }
        asyncio.run_coroutine_threadsafe(
            self._hub.broadcast("contribution_rejected", data),
            self._loop,
        )
    
    def on_leaderboard_updated(self, event: LeaderboardUpdatedEvent) -> None:
        if not self._loop:
            return
        
        data = {
            "goal_id": event.goal_id,
            "entries": [m.to_dict() for m in event.entries],
            "active_policy_hash": event.active_policy_hash.hex() if event.active_policy_hash else None,
        }
        asyncio.run_coroutine_threadsafe(
            self._hub.broadcast("leaderboard_updated", data),
            self._loop,
        )


# Global hooks instance (created at module load, loop set at startup)
_ws_hooks = WebSocketCoordinatorHooks(_event_hub)

def _create_default_goal_spec() -> GoalSpec:
    """Create default goal spec for local GUI training."""
    return GoalSpec(
        goal_id=GoalID("IDI_LOCAL_TRAINING"),
        name="Local Training Goal",
        description="Default goal for GUI-based agent training",
        eval_limits=EvaluationLimits(
            # Keep these aligned with the default GUI training config so the
            # leaderboard metrics look sensible in local/dev mode.
            max_episodes=128,
            max_steps_per_episode=64,
            timeout_seconds=60.0,
        ),
        thresholds=Thresholds(
            min_reward=0.0,
            max_risk=1.0,
        ),
        ranking_weights={
            "reward": 1.0,
            "risk": -0.5,
            "complexity": -0.01,
        },
    )

async def _submit_contribution(contribution: Contribution):
    """Submit contribution to IAN node."""
    if _node:
        logger.info(f"Submitting contribution: goal_id={contribution.goal_id}, contributor={contribution.contributor_id}")
        logger.info(f"Contribution pack_hash: {contribution.pack_hash.hex()[:16] if contribution.pack_hash else 'None'}...")
        success, reason = await _node.submit_contribution(contribution)
        if success:
            logger.info(f"Contribution accepted: {reason}")
        else:
            logger.warning(f"Contribution rejected: {reason}")

# Local Pack Templates (not distributed via IAN network)
_packs = [
    {
        "id": "starter",
        "name": "Starter Pack",
        "description": "Balanced configuration for beginners. Low risk, moderate reward.",
        "config": {"episodes": 1000, "batch_size": 32, "learning_rate": 0.001},
        "price": 0
    },
    {
        "id": "scalper",
        "name": "HFT Scalper",
        "description": "High-frequency trading setup for volatile markets. High risk.",
        "config": {"episodes": 5000, "batch_size": 64, "learning_rate": 0.005},
        "price": 100
    },
    {
        "id": "trend",
        "name": "Trend Follower",
        "description": "Captures long-term trends with reduced noise.",
        "config": {"episodes": 2000, "batch_size": 128, "learning_rate": 0.0005},
        "price": 50
    }
]

# --- Models ---

class WizardState(BaseModel):

    current_step_idx: int
    data: Dict[str, Any]
    validation_errors: Dict[str, str]

class StepUpdate(BaseModel):
    step_data: Dict[str, Any]

from idi.training.python.idi_iann.crypto_env import MarketParams

class TrainingRequest(BaseModel):
    config: Dict[str, Any]
    use_crypto: bool = False
    sim_config: Dict[str, Any] = {}

# --- Wizard Endpoints ---

# We need a way to persist wizard state across requests. 
# For a single-user local app, a global variable is acceptable.
_wizard_controller = WizardController()

@router.get("/wizard/state", response_model=WizardState)
async def get_wizard_state():
    return WizardState(
        current_step_idx=_wizard_controller.current_step_idx,
        data=_wizard_controller.data.__dict__,
        validation_errors=_wizard_controller.get_validation_errors()
    )

@router.post("/wizard/next", response_model=WizardState)
async def wizard_next(update: StepUpdate):
    success = _wizard_controller.next(update.step_data)
    if not success:
        # Even if failed, we return the state so UI can show errors
        pass
    return await get_wizard_state()

@router.post("/wizard/prev", response_model=WizardState)
async def wizard_prev():
    boolean = _wizard_controller.prev()
    return await get_wizard_state()

@router.get("/wizard/spec")
async def get_wizard_spec():
    if not _wizard_controller.is_last_step and not _ALLOW_INCOMPLETE_WIZARD:
        raise HTTPException(
            status_code=400,
            detail=(
                "Wizard is not complete. Finish all steps before generating a spec, "
                "or set IDI_GUI_ALLOW_INCOMPLETE_WIZARD=1 to override."
            ),
        )
    try:
         # Assuming the wizard has enough data
         return {"spec": _wizard_controller.generate_spec()}
    except Exception as e:
         raise HTTPException(status_code=400, detail=str(e))

@router.post("/wizard/reset")
async def wizard_reset():
    global _wizard_controller
    _wizard_controller = WizardController()
    return await get_wizard_state()


# --- Trainer Endpoints ---

class TrainingManager:
    def __init__(self):
        self.trainer: Optional[QTrainer] = None
        self.thread = None
        self.active_websockets: List[WebSocket] = []
        self.latest_stats: Dict[str, Any] = {}
        self.is_running: bool = False

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_websockets.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_websockets.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_websockets:
            try:
                await connection.send_json(message)
            except Exception:
                pass # Handle disconnects gracefully

    def progress_callback(self, episode, total, reward, history=None):
        # This runs in a thread, so we need to be careful with asyncio
        # For simplicity in this demo, we'll just store stats and let the 
        # main loop or a periodic task poll/broadcast, or use run_coroutine_threadsafe
        # However, broadcasting directly might require an event loop reference.
        self.latest_stats = {
            "episode": episode,
            "total_episodes": total,
            "last_reward": reward,
            "status": "running",
            "history": history
        }
        # In a real app we'd push this to the websocket loop

_training_manager = TrainingManager()

@router.post("/trainer/start")
async def start_training(req: TrainingRequest):
    if _training_manager.is_running:
        raise HTTPException(status_code=400, detail="Training already running")
    
    try:
        # Load default config and apply minimal overrides from request.
        # (TrainingConfig is frozen; build a new instance with supported fields.)
        default_config = TrainingConfig()
        user_cfg = req.config or {}
        config = TrainingConfig(
            episodes=int(user_cfg.get("episodes", default_config.episodes)),
            episode_length=int(user_cfg.get("episode_length", default_config.episode_length)),
            discount=float(user_cfg.get("discount", default_config.discount)),
            learning_rate=float(user_cfg.get("learning_rate", default_config.learning_rate)),
            exploration_decay=float(user_cfg.get("exploration_decay", default_config.exploration_decay)),
            quantizer=default_config.quantizer,
            rewards=default_config.rewards,
            emote=default_config.emote,
            layers=default_config.layers,
            tile_coder=default_config.tile_coder,
            communication=default_config.communication,
            fractal=default_config.fractal,
            multi_layer=default_config.multi_layer,
            episodic=default_config.episodic,
        )
        
        market_params = None
        if req.use_crypto and req.sim_config:
            # Safely create MarketParams from dict
            valid_keys = MarketParams.__annotations__.keys()
            filtered_config = {k: v for k, v in req.sim_config.items() if k in valid_keys}
            market_params = MarketParams(**filtered_config)

        # Start training in thread
        _training_manager.is_running = True
        _training_manager.trainer = None
        _training_manager.thread = None
        _training_manager.latest_stats = { # Reset stats
            "episode": 0,
            "total_episodes": config.episodes,
            "last_reward": 0.0,
            "status": "starting",
        }
        
        trainer_ref: Dict[str, Optional[QTrainer]] = {"trainer": None}

        def on_complete(policy, trace, stats):
            _training_manager.is_running = False
            cancelled = bool(trainer_ref["trainer"] and trainer_ref["trainer"].is_cancelled)
            _training_manager.latest_stats["status"] = "cancelled" if cancelled else "completed"
            _training_manager.latest_stats["final_stats"] = stats
            
            # Submit to IAN network as real Contribution
            try:
                # Serialize policy to JSON (safe alternative to pickle)
                # Extract Q-table if available
                policy_data = {}
                if hasattr(policy, 'to_entries'):
                    policy_data = policy.to_entries()
                elif hasattr(policy, '__dict__'):
                    # Fallback: serialize basic attributes
                    policy_data = {k: v for k, v in policy.__dict__.items() 
                                   if isinstance(v, (int, float, str, list, dict))}
                
                parameters = json.dumps(policy_data).encode('utf-8')
                
                agent_pack = AgentPack(
                    version="1.0",
                    parameters=parameters,
                    metadata={
                        "episodes": stats.get("episodes", 0),
                        "mean_reward": stats.get("mean_reward", 0.0),
                        "config": asdict(config),
                        "serialization": "json",  # Mark serialization format
                        "cancelled": cancelled,
                        "use_crypto_env": bool(req.use_crypto),
                        "market_params": asdict(market_params) if market_params is not None else None,
                    },
                )
                
                if _node:
                    contribution = Contribution(
                        goal_id=_node._goal_spec.goal_id,
                        agent_pack=agent_pack,
                        contributor_id="local_gui_user",
                        # In IAN ordering, Contribution.seed is treated as a deterministic
                        # ordering timestamp when non-zero; use ms resolution here.
                        seed=int(time.time() * 1000),
                    )
                    
                    # Submit from thread to main event loop
                    logger.info(f"Event loop available: {_event_loop is not None}")
                    if _event_loop:
                        asyncio.run_coroutine_threadsafe(_submit_contribution(contribution), _event_loop)
                        logger.info(f"Submitted contribution to IAN network")
                    else:
                        logger.error("Event loop not available for submission")
                else:
                    logger.warning("IAN node not initialized, contribution not submitted")
                    
            except Exception as e:
                logger.error(f"Failed to submit contribution: {e}")
            
        thread, trainer = run_training_in_thread(
            config,
            use_crypto_env=req.use_crypto,
            progress_callback=_training_manager.progress_callback,
            on_complete=on_complete,
            market_params=market_params
        )

        trainer_ref["trainer"] = trainer
        _training_manager.trainer = trainer
        _training_manager.thread = thread
        
        return {"status": "started"}
    except Exception as e:
        _training_manager.is_running = False
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trainer/stop")
async def stop_training():
    if _training_manager.trainer is not None:
        _training_manager.trainer.cancel()
    _training_manager.is_running = False
    _training_manager.latest_stats["status"] = "cancelled"
    return {"status": "stopping"}

@router.websocket("/ws/trainer")
async def websocket_trainer(websocket: WebSocket):
    await _training_manager.connect(websocket)
    try:
        while True:
            # Poll for updates (simple implementation)
            if _training_manager.latest_stats:
                await websocket.send_json(_training_manager.latest_stats)
            await asyncio.sleep(0.5) 
            # Check for client disconnect
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
            except asyncio.TimeoutError:
                pass
    except (WebSocketDisconnect, Exception):
        _training_manager.disconnect(websocket)


@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """WebSocket endpoint for real-time contribution/leaderboard events."""
    await _event_hub.connect(websocket)
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "data": {"message": "Real-time events connected"},
            "timestamp": int(time.time() * 1000),
        })
        
        # Keep connection alive, listen for client messages (e.g., ping)
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Echo pings
                if msg == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "keepalive"})
    except (WebSocketDisconnect, Exception):
        await _event_hub.disconnect(websocket)


# --- Agent Management Endpoints ---

class SaveAgentRequest(BaseModel):
    name: str

@router.get("/agents", response_model=List[AgentMetadata])
async def list_agents():
    return _agent_manager.list_agents()

@router.post("/agents/save")
async def save_agent(req: SaveAgentRequest):
    if not _wizard_controller.is_last_step and not _ALLOW_INCOMPLETE_WIZARD:
        raise HTTPException(
            status_code=400,
            detail=(
                "Wizard is not complete. Finish all steps before saving, "
                "or set IDI_GUI_ALLOW_INCOMPLETE_WIZARD=1 to override."
            ),
        )
    
    try:
        spec = _wizard_controller.generate_spec()
        # Collect full wizard data
        data = _wizard_controller.data.__dict__
        # Ensure name matches request or wizard data
        name = req.name or data.get("name") or "unnamed"
        
        path = _agent_manager.save_agent(name, spec, data)
        return {"status": "saved", "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/{name}/load")
async def load_agent(name: str):
    data = _agent_manager.get_agent_data(name)
    if not data:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Load into wizard controller
    global _wizard_controller
    _wizard_controller = WizardController() 
    # Monkey-patch or use internal update for now
    _wizard_controller._update_data(data)
    # Move to last step so user can review/generate
    _wizard_controller.current_step_idx = len(_wizard_controller.STEPS) - 1
    
    return {"status": "loaded"}

@router.delete("/agents/{name}")
async def delete_agent(name: str):
    success = _agent_manager.delete_agent(name)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"status": "deleted"}

@router.get("/wizard/export")
async def export_spec():
    try:
        spec = _wizard_controller.generate_spec()
        name = _wizard_controller.data.name or "agent"
        return Response(
            content=spec,
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={name}.tau"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/settings")
async def get_settings():
    return _settings_manager.get_settings()

@router.post("/settings")
async def update_settings(settings: Dict[str, Any]):
    return _settings_manager.update_settings(settings)

# --- Leaderboard & Packs Endpoints ---

@router.get("/leaderboard")
async def get_leaderboard():
    if not _node:
        return []
    
    leaderboard = _node._base_coordinator.get_leaderboard()
    return [m.to_dict() for m in leaderboard]

@router.get("/debug/coordinator")
async def debug_coordinator():
    if not _node:
        return {"error": "Node not initialized"}
    
    stats = _node._base_coordinator.get_stats()
    
    # Add consensus/mempool info
    mempool = _node._consensus._mempool
    consensus_info = {
        "mempool_size": len(mempool._by_hash),
        "consensus_running": _node._consensus._running,
        "processing_lock_locked": _node._consensus._processing_lock.locked(),
        "contributions_since_check": _node._consensus._contributions_since_check,
    }
    
    return {
        "coordinator_stats": stats,
        "consensus": consensus_info,
    }

@router.get("/packs")
async def get_packs():
    return _packs

@router.post("/packs/{pack_id}/install")
async def install_pack(pack_id: str):
    pack = next((p for p in _packs if p["id"] == pack_id), None)
    if not pack:
        raise HTTPException(status_code=404, detail="Pack not found")
    
    # In a real app, this might download content. Here we just return success.
    # Optionally we could pre-load the wizard with this config.
    return {"status": "installed", "pack": pack}
