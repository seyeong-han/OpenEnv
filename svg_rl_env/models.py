# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the SVG-RL Environment.

SVG Rendering-Aware Reinforcement Learning Environment based on:
"Rendering-Aware Reinforcement Learning for Vector Graphics Generation"

This environment allows agents to learn SVG generation through visual feedback.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict

from openenv_core.env_server.types import Action, Observation


@dataclass(kw_only=True)
class SvgRlAction(Action):
    """
    Action for SVG-RL environment - generate or modify SVG code.
    
    The agent can either provide:
    - Full SVG code (complete)
    - Incremental SVG command (partial update)
    """

    svg_code: str  # SVG code to render
    is_complete: bool = False  # Whether this completes the SVG


@dataclass(kw_only=True)
class SvgRlObservation(Observation):
    """
    Observation from SVG-RL environment - rendering feedback.
    
    Contains visual feedback about how well the rendered SVG
    matches the target image, following the RLRF approach.
    """

    # Visual similarity metrics
    pixel_similarity: float  # MSE-based similarity (0-1)
    structural_similarity: float  # SSIM score (0-1)
    perceptual_distance: float  # Perceptual distance metric
    
    # SVG statistics
    svg_complexity: int  # Number of SVG elements
    svg_valid: bool  # Whether SVG is valid
    
    # Rendered images (base64 encoded PNG)
    rendered_image: Optional[str] = None  # Current rendered image
    target_image: Optional[str] = None  # Target image (for reference)
    
    # Additional metrics
    edge_similarity: float = 0.0  # Edge detection similarity
    color_histogram_distance: float = 0.0  # Color distribution difference
    
    # Episode info
    step_number: int = 0
    metadata: Dict = field(default_factory=dict)

