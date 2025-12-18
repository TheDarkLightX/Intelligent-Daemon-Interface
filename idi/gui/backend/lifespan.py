"""
FastAPI lifespan management for IAN DecentralizedNode.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage DecentralizedNode lifecycle."""
    from idi.gui.backend import api
    
    # Startup
    logger.info("Initializing IAN DecentralizedNode...")
    
    goal_spec = api._create_default_goal_spec()
    identity = api.NodeIdentity.generate()
    
    # Configure for local-only mode (no external network)
    config = api.DecentralizedNodeConfig(
        listen_port=9000,
        seed_addresses=[],  # No external peers
        enable_p2p=False,  # Standalone GUI mode; no network sockets required
        accept_contributions=True,
        serve_evaluations=False,
        commit_to_tau=False,  # Disable Tau commits for local GUI
        enable_health_server=False,
    )
    
    coordinator = api.create_idi_coordinator(
        goal_spec=goal_spec,
        leaderboard_capacity=100,
        harness_type="backtest",  # Real deterministic evaluation (standalone mode)
    )
    
    api._node = api.DecentralizedNode(
        goal_spec=goal_spec,
        identity=identity,
        coordinator=coordinator,
        config=config,
    )
    
    # Start node and wait for initialization to complete
    # This ensures consensus coordinator is running before accepting requests
    await api._node.start()
    
    # Store event loop reference for thread-safe async calls
    api._event_loop = asyncio.get_running_loop()
    logger.info(f"Stored event loop reference: {api._event_loop}")
    
    logger.info(f"IAN node started with goal: {goal_spec.goal_id}")

    
    yield
    
    # Shutdown
    logger.info("Shutting down IAN DecentralizedNode...")
    if api._node:
        await api._node.stop()
    logger.info("IAN node stopped")
