"""
Training manager for controlling and monitoring RL training.

Provides async event-driven architecture to monitor training progress
without blocking the training loop.
"""

import asyncio
import logging
import traceback
from typing import Optional, Callable, Awaitable, Dict, Any
from datetime import datetime

from .events import (
    TrainingStatus,
    EventType,
    StepEvent,
    EpisodeEvent,
    StatusEvent,
    ErrorEvent,
    serialize_event
)

logger = logging.getLogger(__name__)


class TrainingManager:
    """
    Manages training lifecycle and emits real-time events.
    
    This class wraps the training loop and provides:
    - Start/stop/pause/resume controls
    - Non-blocking event emission via asyncio queue
    - Training state management
    - Error handling and recovery
    """
    
    def __init__(self):
        """Initialize training manager."""
        self.status = TrainingStatus.IDLE
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.training_task: Optional[asyncio.Task] = None
        self.pause_event = asyncio.Event()
        self.stop_event = asyncio.Event()
        
        # Training state
        self.current_episode: Optional[int] = None
        self.current_step: Optional[int] = None
        self.best_reward: float = float('-inf')
        
        # Training configuration (set when starting)
        self.config: Dict[str, Any] = {}
        
        # Event callback (called when events are emitted)
        self.event_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    
    def set_event_callback(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """Set callback for event emission (e.g., WebSocket broadcast)."""
        self.event_callback = callback
    
    async def emit_event(self, event_type: EventType, event_data: Any):
        """
        Emit an event to the event queue.
        
        Args:
            event_type: Type of event
            event_data: Event data object
        """
        payload = serialize_event(event_type, event_data)
        payload["timestamp"] = datetime.utcnow().isoformat()
        
        # Put in queue (non-blocking)
        await self.event_queue.put(payload)
        
        # Call callback if registered (e.g., WebSocket broadcast)
        if self.event_callback:
            try:
                await self.event_callback(payload)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
    
    async def start_training(
        self,
        trainer,
        episodes: int = 5,
        max_env_steps: int = 4,
        learning_rate: float = 2e-6,
        **kwargs
    ):
        """
        Start training in background task.
        
        Args:
            trainer: SvgRlQwenTrainer instance
            episodes: Number of training episodes
            max_env_steps: Max steps per episode
            learning_rate: Learning rate
            **kwargs: Additional training parameters
        """
        if self.status == TrainingStatus.RUNNING:
            raise RuntimeError("Training is already running")
        
        # Store configuration
        self.config = {
            "episodes": episodes,
            "max_env_steps": max_env_steps,
            "learning_rate": learning_rate,
            **kwargs
        }
        
        # Reset state
        self.stop_event.clear()
        self.pause_event.set()  # Not paused initially
        self.current_episode = 0
        self.current_step = 0
        self.best_reward = float('-inf')
        
        # Update status
        self.status = TrainingStatus.RUNNING
        await self.emit_event(
            EventType.STATUS_CHANGED,
            StatusEvent(
                status=TrainingStatus.RUNNING,
                message="Training started",
                episodes=episodes,
                max_env_steps=max_env_steps,
                learning_rate=learning_rate,
                current_episode=0,
                current_step=0,
            )
        )
        
        # Start training task
        self.training_task = asyncio.create_task(
            self._training_loop(trainer, episodes, max_env_steps)
        )
    
    async def _training_loop(self, trainer, episodes: int, max_env_steps: int):
        """
        Internal training loop that runs in background task.
        
        Args:
            trainer: SvgRlQwenTrainer instance
            episodes: Number of episodes
            max_env_steps: Max steps per episode
        """
        import sys
        print(f"[TrainingManager] Starting training loop: {episodes} episodes, {max_env_steps} steps/episode", flush=True)
        sys.stdout.flush()
        
        try:
            # Training loop
            for episode_idx in range(1, episodes + 1):
                # Check if stopped
                if self.stop_event.is_set():
                    logger.info("Training stopped by user")
                    break
                
                # Check if paused (will block here until resumed)
                await self.pause_event.wait()
                
                self.current_episode = episode_idx
                self.current_step = 0
                
                # Reset environment
                print(f"[TrainingManager] Starting episode {episode_idx}/{episodes}", flush=True)
                logger.info(f"Starting episode {episode_idx}/{episodes}")
                result = trainer.env_client.reset()
                print(f"[TrainingManager] Environment reset complete", flush=True)
                
                # Handle both direct environment and HTTP client
                if hasattr(result, 'observation'):
                    # HTTP client returns StepResult with .observation
                    obs = result.observation
                else:
                    # Direct environment returns observation directly
                    obs = result
                print(f"[TrainingManager] Got observation, target_image present: {obs.target_image is not None}", flush=True)
                
                # Emit episode start event
                await self.emit_event(
                    EventType.EPISODE_STARTED,
                    EpisodeEvent(
                        episode_idx=episode_idx,
                        total_reward=0.0,
                        total_steps=0,
                        best_reward=self.best_reward,
                        status="started",
                        target_image=obs.target_image,
                    )
                )
                
                # Episode loop
                done = False
                step = 0
                total_reward = 0.0
                episode_rewards = []
                
                while not done and step < max_env_steps:
                    # Check pause/stop
                    if self.stop_event.is_set():
                        break
                    await self.pause_event.wait()
                    
                    step += 1
                    self.current_step = step
                    print(f"[TrainingManager] Episode {episode_idx} Step {step}: Generating SVG...", flush=True)
                    
                    # Generate action
                    svg_code, logprob_mean, debug = trainer.policy.generate_svg(obs)
                    print(f"[TrainingManager] Generated SVG (length={len(svg_code)})", flush=True)
                    
                    # Step environment
                    from ..models import SvgRlAction
                    action = SvgRlAction(svg_code=svg_code, is_complete=True)
                    result = trainer.env_client.step(action)
                    
                    # Handle both direct environment and HTTP client
                    if hasattr(result, 'observation'):
                        # HTTP client returns StepResult with .observation
                        obs = result.observation
                    else:
                        # Direct environment returns observation directly
                        obs = result
                    
                    # Get reward
                    reward = obs.reward if obs.reward is not None else 0.0
                    total_reward += reward
                    episode_rewards.append(reward)
                    
                    # Compute advantage
                    advantage = reward - trainer.baseline
                    
                    # Update model (if training)
                    loss = -advantage * logprob_mean
                    loss.backward()
                    
                    # Gradient clipping
                    import torch.nn as nn
                    nn.utils.clip_grad_norm_(
                        trainer.policy.model.parameters(),
                        max_norm=trainer.grad_clip
                    )
                    
                    trainer.optimizer.step()
                    trainer.optimizer.zero_grad()
                    
                    # Update baseline
                    trainer.baseline = (
                        trainer.baseline_momentum * trainer.baseline +
                        (1.0 - trainer.baseline_momentum) * reward
                    )
                    
                    # Emit step event
                    await self.emit_event(
                        EventType.STEP_COMPLETED,
                        StepEvent(
                            episode_idx=episode_idx,
                            step=step,
                            reward=reward,
                            advantage=advantage,
                            baseline=trainer.baseline,
                            pixel_similarity=obs.pixel_similarity,
                            structural_similarity=obs.structural_similarity,
                            perceptual_distance=obs.perceptual_distance,
                            edge_similarity=obs.edge_similarity,
                            color_histogram_distance=obs.color_histogram_distance,
                            svg_code=svg_code,
                            svg_length=len(svg_code),
                            svg_complexity=obs.svg_complexity,
                            svg_valid=obs.svg_valid,
                            rendered_image=obs.rendered_image,
                            target_image=obs.target_image,
                            done=obs.done,
                            total_reward=total_reward,
                        )
                    )
                    
                    # Check if done
                    done = obs.done
                    
                    # Log
                    logger.info(
                        f"Episode {episode_idx} Step {step} | "
                        f"Reward {reward:.4f} | Advantage {advantage:.4f} | "
                        f"SVG length {len(svg_code)}"
                    )
                
                # Episode complete
                if total_reward > self.best_reward:
                    self.best_reward = total_reward
                    # Save checkpoint
                    trainer.save_checkpoint("best")
                    logger.info(f"New best reward: {self.best_reward:.4f}")
                
                # Save periodic checkpoint
                if episode_idx % trainer.save_every == 0:
                    trainer.save_checkpoint(f"episode_{episode_idx}")
                
                # Emit episode complete event
                await self.emit_event(
                    EventType.EPISODE_COMPLETED,
                    EpisodeEvent(
                        episode_idx=episode_idx,
                        total_reward=total_reward,
                        total_steps=step,
                        best_reward=self.best_reward,
                        status="completed",
                        avg_reward=total_reward / step if step > 0 else 0.0,
                        max_step_reward=max(episode_rewards) if episode_rewards else 0.0,
                        min_step_reward=min(episode_rewards) if episode_rewards else 0.0,
                    )
                )
                
                logger.info(
                    f"Episode {episode_idx} finished | "
                    f"Total reward: {total_reward:.4f} | Steps: {step}"
                )
            
            # Training complete
            self.status = TrainingStatus.IDLE
            await self.emit_event(
                EventType.TRAINING_COMPLETED,
                StatusEvent(
                    status=TrainingStatus.IDLE,
                    message="Training completed successfully",
                    current_episode=episodes,
                    current_step=0,
                )
            )
            logger.info("Training completed successfully")
            
        except Exception as e:
            # Training error
            error_tb = traceback.format_exc()
            print(f"[TrainingManager] TRAINING ERROR: {e}", flush=True)
            print(f"[TrainingManager] Traceback:\n{error_tb}", flush=True)
            logger.error(f"Training error: {e}")
            logger.error(error_tb)
            
            self.status = TrainingStatus.ERROR
            await self.emit_event(
                EventType.TRAINING_ERROR,
                ErrorEvent(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    episode_idx=self.current_episode,
                    step=self.current_step,
                    traceback=error_tb,
                )
            )
    
    async def pause_training(self):
        """Pause training."""
        if self.status != TrainingStatus.RUNNING:
            raise RuntimeError("Training is not running")
        
        self.pause_event.clear()
        self.status = TrainingStatus.PAUSED
        
        await self.emit_event(
            EventType.STATUS_CHANGED,
            StatusEvent(
                status=TrainingStatus.PAUSED,
                message="Training paused",
                current_episode=self.current_episode,
                current_step=self.current_step,
            )
        )
        logger.info("Training paused")
    
    async def resume_training(self):
        """Resume training."""
        if self.status != TrainingStatus.PAUSED:
            raise RuntimeError("Training is not paused")
        
        self.pause_event.set()
        self.status = TrainingStatus.RUNNING
        
        await self.emit_event(
            EventType.STATUS_CHANGED,
            StatusEvent(
                status=TrainingStatus.RUNNING,
                message="Training resumed",
                current_episode=self.current_episode,
                current_step=self.current_step,
            )
        )
        logger.info("Training resumed")
    
    async def stop_training(self):
        """Stop training."""
        if self.status not in (TrainingStatus.RUNNING, TrainingStatus.PAUSED):
            raise RuntimeError("Training is not running or paused")
        
        self.stop_event.set()
        self.pause_event.set()  # Resume if paused to allow stop
        
        # Wait for training task to finish
        if self.training_task:
            await self.training_task
        
        self.status = TrainingStatus.STOPPED
        await self.emit_event(
            EventType.STATUS_CHANGED,
            StatusEvent(
                status=TrainingStatus.STOPPED,
                message="Training stopped",
                current_episode=self.current_episode,
                current_step=self.current_step,
            )
        )
        logger.info("Training stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            "status": self.status.value,
            "current_episode": self.current_episode,
            "current_step": self.current_step,
            "best_reward": self.best_reward if self.best_reward != float('-inf') else None,
            "config": self.config,
        }
