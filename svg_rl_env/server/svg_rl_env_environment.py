# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SVG-RL Environment Implementation.

Rendering-Aware Reinforcement Learning for Vector Graphics Generation.
Based on the paper: "RLRF - Rendering Feedback for SVG Generation"

This environment enables agents to learn SVG generation through visual feedback
by comparing rendered outputs with target images.
"""

import base64
import io
import logging
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

import cairosvg
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from openenv_core.env_server.interfaces import Environment
from openenv_core.env_server.types import State

from ..models import SvgRlAction, SvgRlObservation

logger = logging.getLogger(__name__)


class SvgRlEnvironment(Environment):
    """
    SVG Rendering-Aware RL Environment.

    This environment allows agents to learn vector graphics generation
    through visual feedback. The agent generates SVG code, which is
    rendered and compared against a target image using multiple metrics.

    Features:
    - Multi-metric visual feedback (MSE, SSIM, perceptual distance)
    - Support for both complete and incremental SVG generation
    - Configurable target images
    - Detailed rendering statistics

    Example:
        >>> env = SvgRlEnvironment()
        >>> obs = env.reset()
        >>> print(f"Target set, similarity: {obs.pixel_similarity}")
        >>>
        >>> # Agent generates SVG
        >>> svg_code = '<svg><circle cx="50" cy="50" r="40" fill="red"/></svg>'
        >>> obs = env.step(SvgRlAction(svg_code=svg_code, is_complete=True))
        >>> print(f"Reward: {obs.reward}, SSIM: {obs.structural_similarity}")
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        target_image_path: Optional[str] = None,
        max_steps: int = 50,
        reward_weights: Optional[dict] = None,
    ):
        """
        Initialize the SVG-RL environment.

        Args:
            image_size: Size of rendered images (width, height)
            target_image_path: Path to target image file. If None, generates simple targets
            max_steps: Maximum steps per episode
            reward_weights: Weights for different reward components
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.image_size = image_size
        self.max_steps = max_steps
        
        # Reward weights for different metrics
        self.reward_weights = reward_weights or {
            "pixel_similarity": 0.3,
            "structural_similarity": 0.4,
            "perceptual_distance": 0.2,
            "edge_similarity": 0.1,
        }
        
        # Episode state
        self.current_svg = ""
        self.target_image_path = target_image_path
        self.target_image: Optional[np.ndarray] = None
        self.target_image_b64: Optional[str] = None

    def reset(self) -> SvgRlObservation:
        """
        Reset the environment with a new target image.

        Returns:
            SvgRlObservation with initial state
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self.current_svg = ""
        
        # Load or generate target image
        self._load_target_image()
        
        # Initial observation (blank SVG)
        return SvgRlObservation(
            pixel_similarity=0.0,
            structural_similarity=0.0,
            perceptual_distance=1.0,
            svg_complexity=0,
            svg_valid=True,
            target_image=self.target_image_b64,
            step_number=0,
            done=False,
            reward=0.0,
            metadata={"episode_id": self._state.episode_id},
        )

    def step(self, action: SvgRlAction) -> SvgRlObservation:  # type: ignore[override]
        """
        Execute a step by rendering SVG and computing visual feedback.

        Args:
            action: SvgRlAction containing SVG code

        Returns:
            SvgRlObservation with rendering feedback and reward
        """
        self._state.step_count += 1
        
        # Update SVG (complete or incremental)
        if action.is_complete:
            self.current_svg = action.svg_code
        else:
            self.current_svg += action.svg_code
        
        # Render and compute metrics
        try:
            rendered_image = self._render_svg(self.current_svg)
            rendered_b64 = self._image_to_base64(rendered_image)
            
            metrics = self._compute_visual_metrics(rendered_image, self.target_image)
            reward = self._compute_reward(metrics)
            svg_valid = True
            
        except Exception as e:
            logger.warning(f"SVG rendering failed: {e}")
            # Invalid SVG gets negative reward
            metrics = {
                "pixel_similarity": 0.0,
                "structural_similarity": 0.0,
                "perceptual_distance": 1.0,
                "edge_similarity": 0.0,
                "color_histogram_distance": 1.0,
            }
            reward = -1.0
            svg_valid = False
            rendered_b64 = None
        
        # Check episode termination
        done = self._state.step_count >= self.max_steps or action.is_complete
        
        # Count SVG elements (simple heuristic)
        svg_complexity = self.current_svg.count("<") - self.current_svg.count("</")
        
        return SvgRlObservation(
            pixel_similarity=metrics["pixel_similarity"],
            structural_similarity=metrics["structural_similarity"],
            perceptual_distance=metrics["perceptual_distance"],
            edge_similarity=metrics.get("edge_similarity", 0.0),
            color_histogram_distance=metrics.get("color_histogram_distance", 0.0),
            svg_complexity=svg_complexity,
            svg_valid=svg_valid,
            rendered_image=rendered_b64,
            target_image=self.target_image_b64,
            step_number=self._state.step_count,
            done=done,
            reward=reward,
            metadata={
                "svg_length": len(self.current_svg),
                "is_complete": action.is_complete,
            },
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    # ============= Helper Methods =============

    def _load_target_image(self):
        """Load or generate a target image for the episode."""
        if self.target_image_path and Path(self.target_image_path).exists():
            # Load from file
            img = Image.open(self.target_image_path).convert("RGB")
            img = img.resize(self.image_size)
            self.target_image = np.array(img)
        else:
            # Generate simple geometric target
            self.target_image = self._generate_simple_target()
        
        self.target_image_b64 = self._image_to_base64(self.target_image)

    def _generate_simple_target(self) -> np.ndarray:
        """Generate a simple geometric shape as target."""
        img = np.ones((*self.image_size, 3), dtype=np.uint8) * 255
        
        # Random simple shape (circle, rectangle, or triangle)
        shape_type = np.random.choice(["circle", "rectangle", "triangle"])
        
        center = (self.image_size[0] // 2, self.image_size[1] // 2)
        color = tuple(np.random.randint(0, 256, 3).tolist())
        
        if shape_type == "circle":
            radius = np.random.randint(30, min(self.image_size) // 3)
            cv2.circle(img, center, radius, color, -1)
        elif shape_type == "rectangle":
            size = np.random.randint(50, min(self.image_size) // 2)
            pt1 = (center[0] - size // 2, center[1] - size // 2)
            pt2 = (center[0] + size // 2, center[1] + size // 2)
            cv2.rectangle(img, pt1, pt2, color, -1)
        else:  # triangle
            pts = np.array([
                [center[0], center[1] - 60],
                [center[0] - 50, center[1] + 40],
                [center[0] + 50, center[1] + 40],
            ], np.int32)
            cv2.fillPoly(img, [pts], color)
        
        return img

    def _render_svg(self, svg_code: str) -> np.ndarray:
        """
        Render SVG code to numpy array image.

        Args:
            svg_code: SVG code string

        Returns:
            Rendered image as numpy array (H, W, 3)
        """
        # Wrap in SVG tag if not already wrapped
        if not svg_code.strip().startswith("<svg"):
            svg_code = f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.image_size[0]}" height="{self.image_size[1]}">{svg_code}</svg>'
        
        # Render to PNG bytes
        png_bytes = cairosvg.svg2png(
            bytestring=svg_code.encode("utf-8"),
            output_width=self.image_size[0],
            output_height=self.image_size[1],
        )
        
        # Convert to numpy array
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        return np.array(img)

    def _compute_visual_metrics(
        self, rendered: np.ndarray, target: np.ndarray
    ) -> dict:
        """
        Compute multiple visual similarity metrics.

        Args:
            rendered: Rendered image
            target: Target image

        Returns:
            Dictionary of metrics
        """
        # Ensure same size
        if rendered.shape != target.shape:
            rendered = cv2.resize(rendered, (target.shape[1], target.shape[0]))
        
        # 1. Pixel-level MSE similarity (0 = identical, 1 = max difference)
        mse = np.mean((rendered.astype(float) - target.astype(float)) ** 2)
        pixel_similarity = 1.0 - min(mse / (255 ** 2), 1.0)
        
        # 2. Structural Similarity Index (SSIM)
        gray_rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY)
        gray_target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
        ssim_score = ssim(gray_target, gray_rendered)
        
        # 3. Perceptual distance (simplified - using LAB color space)
        lab_rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2LAB)
        lab_target = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)
        perceptual_dist = np.mean(np.abs(lab_rendered - lab_target)) / 255.0
        
        # 4. Edge similarity
        edges_rendered = cv2.Canny(gray_rendered, 50, 150)
        edges_target = cv2.Canny(gray_target, 50, 150)
        edge_sim = 1.0 - np.mean(np.abs(edges_rendered - edges_target)) / 255.0
        
        # 5. Color histogram distance
        hist_rendered = cv2.calcHist([rendered], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_target = cv2.calcHist([target], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_rendered = cv2.normalize(hist_rendered, hist_rendered).flatten()
        hist_target = cv2.normalize(hist_target, hist_target).flatten()
        color_dist = cv2.compareHist(hist_rendered, hist_target, cv2.HISTCMP_CHISQR)
        color_dist = min(color_dist / 100.0, 1.0)  # Normalize
        
        return {
            "pixel_similarity": float(pixel_similarity),
            "structural_similarity": float((ssim_score + 1) / 2),  # Convert from [-1,1] to [0,1]
            "perceptual_distance": float(perceptual_dist),
            "edge_similarity": float(edge_sim),
            "color_histogram_distance": float(color_dist),
        }

    def _compute_reward(self, metrics: dict) -> float:
        """
        Compute reward from visual metrics using weighted combination.

        Args:
            metrics: Dictionary of computed metrics

        Returns:
            Scalar reward value
        """
        reward = 0.0
        
        # Weighted combination of metrics
        reward += self.reward_weights.get("pixel_similarity", 0) * metrics["pixel_similarity"]
        reward += self.reward_weights.get("structural_similarity", 0) * metrics["structural_similarity"]
        reward += self.reward_weights.get("perceptual_distance", 0) * (1.0 - metrics["perceptual_distance"])
        reward += self.reward_weights.get("edge_similarity", 0) * metrics.get("edge_similarity", 0)
        
        # Bonus for near-perfect match
        if metrics["structural_similarity"] > 0.95:
            reward += 1.0
        
        return float(reward)

    def _image_to_base64(self, img: np.ndarray) -> str:
        """Convert numpy image to base64 string."""
        pil_img = Image.fromarray(img.astype(np.uint8))
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
