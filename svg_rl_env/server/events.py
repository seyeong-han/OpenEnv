"""
Event data models for real-time training monitoring.

These models define the structure of events emitted during training
and broadcast to connected WebSocket clients.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Any
from enum import Enum


class TrainingStatus(str, Enum):
    """Training status enum."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class EventType(str, Enum):
    """Event type enum."""
    TRAINING_STARTED = "training_started"
    TRAINING_STOPPED = "training_stopped"
    TRAINING_PAUSED = "training_paused"
    TRAINING_RESUMED = "training_resumed"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_ERROR = "training_error"
    EPISODE_STARTED = "episode_started"
    EPISODE_COMPLETED = "episode_completed"
    STEP_COMPLETED = "step_completed"
    STATUS_CHANGED = "status_changed"


@dataclass
class StepEvent:
    """Event emitted after each training step."""
    episode_idx: int
    step: int
    reward: float
    advantage: float
    baseline: float
    
    # Visual metrics from observation
    pixel_similarity: float
    structural_similarity: float
    perceptual_distance: float
    edge_similarity: float
    color_histogram_distance: float
    
    # SVG metadata
    svg_code: str
    svg_length: int
    svg_complexity: int
    svg_valid: bool
    
    # Images (base64 encoded PNG)
    rendered_image: Optional[str] = None
    target_image: Optional[str] = None
    
    # Episode state
    done: bool = False
    total_reward: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class EpisodeEvent:
    """Event emitted when an episode starts or completes."""
    episode_idx: int
    total_reward: float
    total_steps: int
    best_reward: float
    status: str  # 'started' or 'completed'
    
    # Target image (only on episode start)
    target_image: Optional[str] = None
    
    # Episode statistics (only on episode complete)
    avg_reward: float = 0.0
    max_step_reward: float = 0.0
    min_step_reward: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class StatusEvent:
    """Event emitted when training status changes."""
    status: TrainingStatus
    message: str
    
    # Training configuration
    episodes: Optional[int] = None
    max_env_steps: Optional[int] = None
    learning_rate: Optional[float] = None
    
    # Current progress
    current_episode: Optional[int] = None
    current_step: Optional[int] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert enum to string
        if isinstance(data.get('status'), TrainingStatus):
            data['status'] = data['status'].value
        return data


@dataclass
class ErrorEvent:
    """Event emitted when an error occurs during training."""
    error_type: str
    error_message: str
    episode_idx: Optional[int] = None
    step: Optional[int] = None
    traceback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def serialize_event(event_type: EventType, event_data: Any) -> Dict[str, Any]:
    """
    Serialize an event for WebSocket transmission.
    
    Args:
        event_type: Type of event
        event_data: Event data object (StepEvent, EpisodeEvent, etc.)
    
    Returns:
        Dictionary ready for JSON serialization
    """
    payload = {
        "type": event_type.value if isinstance(event_type, EventType) else event_type,
        "timestamp": None,  # Will be set by server
    }
    
    if hasattr(event_data, 'to_dict'):
        payload["data"] = event_data.to_dict()
    elif isinstance(event_data, dict):
        payload["data"] = event_data
    else:
        payload["data"] = {}
    
    return payload
