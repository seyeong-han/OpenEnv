# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SVG-RL Environment HTTP Client.

This module provides the client for connecting to the SVG Rendering-Aware
Reinforcement Learning Environment server over HTTP.
"""

from typing import Dict

from openenv_core.client_types import StepResult
from openenv_core.env_server.types import State
from openenv_core.http_env_client import HTTPEnvClient

from .models import SvgRlAction, SvgRlObservation


class SvgRlEnv(HTTPEnvClient[SvgRlAction, SvgRlObservation]):
    """
    HTTP client for the SVG-RL Environment.

    This client connects to a SVG Rendering-Aware RL server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = SvgRlEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(f"SSIM: {result.observation.structural_similarity}")
        >>>
        >>> # Generate SVG
        >>> svg = '<circle cx="128" cy="128" r="50" fill="red"/>'
        >>> result = client.step(SvgRlAction(svg_code=svg, is_complete=True))
        >>> print(f"Reward: {result.reward}")
        >>> print(f"Similarity: {result.observation.pixel_similarity}")

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = SvgRlEnv.from_docker_image("svg-rl-env:latest")
        >>> result = client.reset()
        >>> svg = '<rect x="50" y="50" width="156" height="156" fill="blue"/>'
        >>> result = client.step(SvgRlAction(svg_code=svg, is_complete=True))
    """

    def _step_payload(self, action: SvgRlAction) -> Dict:
        """
        Convert SvgRlAction to JSON payload for step request.

        Args:
            action: SvgRlAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "svg_code": action.svg_code,
            "is_complete": action.is_complete,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SvgRlObservation]:
        """
        Parse server response into StepResult[SvgRlObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with SvgRlObservation
        """
        obs_data = payload.get("observation", {})
        observation = SvgRlObservation(
            pixel_similarity=obs_data.get("pixel_similarity", 0.0),
            structural_similarity=obs_data.get("structural_similarity", 0.0),
            perceptual_distance=obs_data.get("perceptual_distance", 1.0),
            svg_complexity=obs_data.get("svg_complexity", 0),
            svg_valid=obs_data.get("svg_valid", True),
            rendered_image=obs_data.get("rendered_image"),
            target_image=obs_data.get("target_image"),
            edge_similarity=obs_data.get("edge_similarity", 0.0),
            color_histogram_distance=obs_data.get("color_histogram_distance", 0.0),
            step_number=obs_data.get("step_number", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
