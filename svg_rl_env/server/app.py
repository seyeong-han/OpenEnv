# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Svg Rl Env Environment.

This module creates an HTTP server that exposes the SvgRlEnvironment
over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv_core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError("openenv_core is required for the web interface. Install dependencies with '\n    uv sync\n'") from e

from .svg_rl_env_environment import SvgRlEnvironment
from ..models import SvgRlAction, SvgRlObservation

# Create the environment instance
env = SvgRlEnvironment()

# Create the app with web interface and README integration
app = create_app(
    env,
    SvgRlAction,
    SvgRlObservation,
    env_name="svg_rl_env",
)

# ========================================
# Real-Time Monitoring Integration
# ========================================

import socketio
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .training_manager import TrainingManager
from .events import TrainingStatus

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create Socket.IO server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',  # In production, restrict to specific origins
    logger=False,
    engineio_logger=False
)

# Wrap with ASGI app
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Create training manager
training_manager = TrainingManager()

# Set event callback to broadcast via WebSocket
async def broadcast_event(event_payload):
    """Broadcast event to all connected clients."""
    await sio.emit('training_event', event_payload)

training_manager.set_event_callback(broadcast_event)


# WebSocket event handlers
@sio.event
async def connect(sid, environ):
    """Handle client connection."""
    print(f"Client connected: {sid}")
    # Send current status on connect
    status = training_manager.get_status()
    await sio.emit('status', status, to=sid)


@sio.event
async def disconnect(sid):
    """Handle client disconnection."""
    print(f"Client disconnected: {sid}")


# REST API endpoints for training control
@app.post("/api/training/start")
async def start_training(config: dict = None):
    """
    Start training with optional configuration.
    
    Request body:
    {
        "episodes": 5,
        "max_env_steps": 4,
        "learning_rate": 2e-6
    }
    """
    if training_manager.status == TrainingStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Training is already running")
    
    # Default configuration
    config = config or {}
    episodes = config.get("episodes", 5)
    max_env_steps = config.get("max_env_steps", 4)
    learning_rate = config.get("learning_rate", 2e-6)
    
    # Import trainer (lazy import to avoid loading model at startup)
    try:
        from ..qwen_svg_rl import QwenSvgPolicy, SvgRlQwenTrainer, QwenPolicyConfig
        
        # Initialize policy
        policy_config = QwenPolicyConfig()
        policy = QwenSvgPolicy(policy_config)
        
        # Use the DIRECT environment instance (not HTTP client to avoid self-loop)
        # The 'env' variable is already created at the top of this file
        
        # Create trainer with direct environment
        trainer = SvgRlQwenTrainer(
            policy=policy,
            env_client=env,  # Use direct environment, not HTTP client
            learning_rate=learning_rate,
            episodes=episodes,
            max_env_steps=max_env_steps,
        )
        
        # Start training
        await training_manager.start_training(
            trainer=trainer,
            episodes=episodes,
            max_env_steps=max_env_steps,
            learning_rate=learning_rate,
        )
        
        return {"status": "started", "config": config}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


@app.post("/api/training/stop")
async def stop_training():
    """Stop training."""
    try:
        await training_manager.stop_training()
        return {"status": "stopped"}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/training/pause")
async def pause_training():
    """Pause training."""
    try:
        await training_manager.pause_training()
        return {"status": "paused"}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/training/resume")
async def resume_training():
    """Resume training."""
    try:
        await training_manager.resume_training()
        return {"status": "resumed"}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/training/status")
async def get_training_status():
    """Get current training status."""
    return training_manager.get_status()


# Update app reference to use socket_app wrapper
app = socket_app


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m svg_rl_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn svg_rl_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
