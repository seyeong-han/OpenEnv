"""
Qwen2.5-VL based reinforcement learning loop for the SVG-RL environment.

This module implements a lightweight policy gradient trainer and inference
helper that follow the "Rendering-Aware Reinforcement Learning for Vector
Graphics Generation" paper (arXiv: 2505.20793).

Key ideas from the paper that are reflected here:
- The agent receives rendered feedback (SSIM, MSE-style pixel similarity,
  perceptual distance, edge similarity) to optimize SVG generation.
- A vision-language model (Qwen2.5-VL) proposes SVG code conditioned on the
  target image and current metrics.
- Reinforcement learning updates the policy to maximize the rendering-based
  reward.

The code below keeps the training loop intentionally simple (REINFORCE-style
with an EMA baseline) so it is easy to extend with more advanced algorithms
like PPO if desired.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch import nn
from transformers import AutoProcessor, AutoTokenizer, GenerationConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from svg_rl_env.server.svg_rl_env_environment import SvgRlEnvironment
from svg_rl_env.models import SvgRlAction, SvgRlObservation

logger = logging.getLogger(__name__)


# ---------------------------
# Policy and prompt utilities
# ---------------------------


def _decode_base64_image(b64_image: Optional[str]) -> Optional[Image.Image]:
    """Convert a base64 PNG string to a PIL image."""
    if not b64_image:
        return None
    try:
        binary = base64.b64decode(b64_image)
        return Image.open(io.BytesIO(binary)).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to decode base64 target image: %s", exc)
        return None


@dataclass
class QwenPolicyConfig:
    """Configuration for the Qwen2.5-VL policy."""

    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    device: Optional[str] = None
    dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32"
    max_new_tokens: int = 196
    temperature: float = 0.7
    top_p: float = 0.9
    guidance_system_prompt: str = (
        "You are an SVG designer. Generate only valid <svg> markup that matches "
        "the target image as closely as possible. The SVG should render correctly "
        "when passed to a browser without extra text."
    )


class QwenSvgPolicy:
    """
    Thin wrapper around Qwen2.5-VL for SVG generation.

    The policy turns an SvgRlObservation into a chat prompt, generates SVG code,
    and exposes log probabilities for policy gradient updates.
    """

    def __init__(self, config: Optional[QwenPolicyConfig] = None):
        self.config = config or QwenPolicyConfig()

        dtype_map = {
            "auto": None,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.dtype, None)

        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        # Keep tokenizer reference for compatibility
        self.tokenizer = self.processor
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            device_map=self.config.device or "auto",
            trust_remote_code=True,
        )

        # Get eos_token_id from the wrapped tokenizer
        eos_token_id = getattr(self.processor, 'eos_token_id', None)
        if eos_token_id is None and hasattr(self.processor, 'tokenizer'):
            eos_token_id = self.processor.tokenizer.eos_token_id
        
        self.generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=eos_token_id if eos_token_id else 151643,  # Default Qwen2.5 eos_token_id
        )

    # Public API -------------------------------------------------------------

    def build_messages(self, observation: SvgRlObservation) -> List[Dict]:
        """
        Construct a Qwen chat prompt that conditions on the rendering metrics
        and the target image.
        """
        user_text = (
            "You are optimizing SVG code via reinforcement learning.\n"
            f"Metrics so far -> SSIM: {observation.structural_similarity:.4f}, "
            f"Pixel similarity: {observation.pixel_similarity:.4f}, "
            f"Perceptual distance: {observation.perceptual_distance:.4f}, "
            f"Edge similarity: {observation.edge_similarity:.4f}.\n"
            "Generate improved SVG that matches the target image. "
            "Return ONLY SVG markup (no commentary)."
        )

        content: List[Dict[str, str]] = [{"type": "text", "text": user_text}]
        target_img = _decode_base64_image(observation.target_image)
        if target_img:
            content.append({"type": "image", "image": target_img})

        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.config.guidance_system_prompt}
                ],
            },
            {"role": "user", "content": content},
        ]

    def generate_svg(
        self, observation: SvgRlObservation
    ) -> Tuple[str, torch.Tensor, Dict]:
        """
        Generate SVG code and compute per-token log probabilities.

        Returns:
            svg_text: Generated SVG code (string).
            logprob_mean: Mean log probability of generated tokens (tensor scalar).
            debug: Additional debug metadata for logging.
        """
        print(f"[QwenSvgPolicy] generate_svg called", flush=True)
        messages = self.build_messages(observation)
        print(f"[QwenSvgPolicy] Built messages: {len(messages)} messages", flush=True)
        
        # Extract images from the messages for the processor
        images = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        img = item.get("image")
                        if img is not None:
                            images.append(img)
        print(f"[QwenSvgPolicy] Extracted {len(images)} images from messages", flush=True)
        
        # Apply chat template to get the text prompt
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(f"[QwenSvgPolicy] Applied chat template, prompt length: {len(prompt_text)}", flush=True)
        
        # Process text and images together
        model_inputs = self.processor(
            text=[prompt_text],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        )
        print(f"[QwenSvgPolicy] Processed inputs, keys: {list(model_inputs.keys())}", flush=True)
        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}
        print(f"[QwenSvgPolicy] Moved inputs to device", flush=True)

        generate_kwargs = dict(model_inputs)
        generate_kwargs["generation_config"] = self.generation_config

        print(f"[QwenSvgPolicy] Starting model.generate()...", flush=True)
        generated_ids = self.model.generate(**generate_kwargs)
        print(f"[QwenSvgPolicy] Generation complete, output shape: {generated_ids.shape}", flush=True)
        
        prompt_len = model_inputs["input_ids"].shape[1]
        new_token_ids = generated_ids[:, prompt_len:]

        svg_text = self.processor.decode(
            new_token_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        print(f"[QwenSvgPolicy] Decoded SVG text, length: {len(svg_text)}", flush=True)

        print(f"[QwenSvgPolicy] Computing log probabilities...", flush=True)
        logprob_mean = self._logprob_of_generation(
            prompt_input_ids=model_inputs["input_ids"],
            generated_ids=new_token_ids,
            pixel_values=model_inputs.get("pixel_values"),
            image_grid_thw=model_inputs.get("image_grid_thw"),
        )
        print(f"[QwenSvgPolicy] Log prob computed: {logprob_mean.item():.4f}", flush=True)

        debug = {"prompt": prompt_text, "prompt_length": prompt_len}
        return svg_text, logprob_mean, debug

    # Internal helpers -------------------------------------------------------

    def _logprob_of_generation(
        self,
        prompt_input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the mean log probability of the generated tokens.

        Uses teacher-forcing over the concatenated prompt + generated tokens to
        produce a differentiable log probability for REINFORCE updates.
        """
        full_ids = torch.cat([prompt_input_ids, generated_ids], dim=1)
        attention = torch.ones_like(full_ids, device=full_ids.device)

        forward_kwargs = {
            "input_ids": full_ids[:, :-1],
            "attention_mask": attention[:, :-1],
        }
        # Pass processed image tensors if available
        if pixel_values is not None:
            forward_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            forward_kwargs["image_grid_thw"] = image_grid_thw

        outputs = self.model(**forward_kwargs)
        logits = outputs.logits

        gen_logits = logits[:, prompt_input_ids.shape[1] - 1 : -1, :]
        gen_targets = full_ids[:, prompt_input_ids.shape[1] :]

        log_probs = torch.log_softmax(gen_logits, dim=-1)
        token_log_probs = log_probs.gather(2, gen_targets.unsqueeze(-1)).squeeze(-1)

        return token_log_probs.mean()


# ----------------------
# RL training utilities
# ----------------------


@dataclass
class TrainerConfig:
    """Hyperparameters for the simple policy gradient trainer."""

    episodes: int = 5
    max_env_steps: int = 4
    learning_rate: float = 2e-6
    baseline_momentum: float = 0.9
    grad_clip: float = 1.0
    checkpoint_dir: str = "qwen_svg_rl_checkpoints"
    save_every: int = 1


class SvgRlQwenTrainer:
    """
    Minimal REINFORCE trainer that updates Qwen2.5-VL with rendering rewards.

    Can work with both direct environment (SvgRlEnvironment) and HTTP client (SvgRlEnv).
    """

    def __init__(
        self,
        policy: QwenSvgPolicy,
        env_client=None,  # HTTP client or direct environment
        env: Optional[SvgRlEnvironment] = None,  # For backwards compatibility
        learning_rate: float = 2e-6,
        episodes: int = 5,
        max_env_steps: int = 4,
        baseline_momentum: float = 0.9,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "qwen_svg_rl_checkpoints",
        save_every: int = 1,
        config: Optional[TrainerConfig] = None,
    ):
        # Support both env_client and env parameters
        self.env_client = env_client if env_client is not None else env
        if self.env_client is None:
            raise ValueError("Either env_client or env must be provided")

        self.policy = policy

        # Use config if provided, otherwise use individual parameters
        if config is not None:
            self.config = config
            self.learning_rate = config.learning_rate
            self.episodes = config.episodes
            self.max_env_steps = config.max_env_steps
            self.baseline_momentum = config.baseline_momentum
            self.grad_clip = config.grad_clip
            self.checkpoint_dir = config.checkpoint_dir
            self.save_every = config.save_every
        else:
            self.learning_rate = learning_rate
            self.episodes = episodes
            self.max_env_steps = max_env_steps
            self.baseline_momentum = baseline_momentum
            self.grad_clip = grad_clip
            self.checkpoint_dir = checkpoint_dir
            self.save_every = save_every
            self.config = TrainerConfig(
                episodes=episodes,
                max_env_steps=max_env_steps,
                learning_rate=learning_rate,
                baseline_momentum=baseline_momentum,
                grad_clip=grad_clip,
                checkpoint_dir=checkpoint_dir,
                save_every=save_every,
            )

        self.optimizer = torch.optim.AdamW(
            self.policy.model.parameters(),
            lr=self.learning_rate,
        )
        self.baseline = 0.0

    def train(self):
        """
        Run REINFORCE over multiple episodes.

        Each episode renders the target, lets Qwen propose SVG, and performs a
        policy gradient update based on the rendering-aware reward.
        """
        best_reward = float("-inf")
        ckpt_path = Path(self.config.checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)

        for episode_idx in range(1, self.config.episodes + 1):
            obs = self.env.reset()
            done = False
            total_reward = 0.0
            step = 0

            while not done and step < self.config.max_env_steps:
                svg_code, logprob_mean, debug = self.policy.generate_svg(obs)
                action = SvgRlAction(svg_code=svg_code, is_complete=True)
                obs = self.env.step(action)

                reward = obs.reward or 0.0
                total_reward += reward
                advantage = reward - self.baseline
                loss = -advantage * logprob_mean

                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.model.parameters(), self.config.grad_clip
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.baseline = (
                    self.config.baseline_momentum * self.baseline
                    + (1 - self.config.baseline_momentum) * reward
                )

                logger.info(
                    "Episode %d Step %d | Reward %.4f | Advantage %.4f | SVG length %d",
                    episode_idx,
                    step,
                    reward,
                    advantage,
                    len(svg_code),
                )

                done = obs.done or action.is_complete
                step += 1

            if total_reward > best_reward:
                best_reward = total_reward
                self._save_checkpoint(ckpt_path / "best")

            if episode_idx % self.config.save_every == 0:
                self._save_checkpoint(ckpt_path / f"episode_{episode_idx}")

            logger.info(
                "Episode %d finished | Total reward: %.4f | Steps: %d",
                episode_idx,
                total_reward,
                step,
            )

    # Internal helpers -------------------------------------------------------

    def save_checkpoint(self, name: str):
        """
        Persist model and tokenizer weights for later inference.

        Args:
            name: Checkpoint name (e.g., "best", "episode_3")
        """
        ckpt_path = Path(self.checkpoint_dir)
        path = ckpt_path / name
        path.mkdir(parents=True, exist_ok=True)
        self.policy.model.save_pretrained(path)
        self.policy.tokenizer.save_pretrained(path)
        logger.info("Saved checkpoint to %s", path)

    def _save_checkpoint(self, path: Path):
        """Legacy method for backwards compatibility."""
        path.mkdir(parents=True, exist_ok=True)
        self.policy.model.save_pretrained(path)
        self.policy.tokenizer.save_pretrained(path)
        logger.info("Saved checkpoint to %s", path)


# ----------------------
# Inference convenience
# ----------------------


def run_inference(
    model_path: Optional[str] = None,
    target_image_path: Optional[str] = None,
) -> SvgRlObservation:
    """
    Run a single inference pass: load (or initialize) the policy, reset the
    environment, generate SVG, and return the final observation.
    """
    policy_config = QwenPolicyConfig(
        model_name=model_path or QwenPolicyConfig.model_name,
    )
    policy = QwenSvgPolicy(policy_config)

    env = SvgRlEnvironment(target_image_path=target_image_path)
    obs = env.reset()

    svg_code, _, _ = policy.generate_svg(obs)
    result = env.step(SvgRlAction(svg_code=svg_code, is_complete=True))

    logger.info(
        "Inference reward %.4f | SSIM %.4f | Pixel similarity %.4f",
        result.reward,
        result.structural_similarity,
        result.pixel_similarity,
    )
    return result


# -------------
# CLI Entrypoint
# -------------


def _parse_args():  # pragma: no cover - lightweight CLI helper
    import argparse

    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL RL trainer for SVG-RL env."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run RL training loop.")
    train_parser.add_argument("--episodes", type=int, default=5)
    train_parser.add_argument("--max-env-steps", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=2e-6)
    train_parser.add_argument(
        "--checkpoint-dir", type=str, default="qwen_svg_rl_checkpoints"
    )
    train_parser.add_argument(
        "--model-name", type=str, default=QwenPolicyConfig.model_name
    )
    train_parser.add_argument(
        "--target-image", type=str, help="Optional target image path."
    )

    infer_parser = subparsers.add_parser("infer", help="Run a single inference pass.")
    infer_parser.add_argument(
        "--checkpoint", type=str, help="Checkpoint directory to load."
    )
    infer_parser.add_argument(
        "--target-image", type=str, help="Optional target image path."
    )

    return parser.parse_args()


def main():  # pragma: no cover - CLI execution path
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    if args.command == "train":
        policy_cfg = QwenPolicyConfig(model_name=args.model_name)
        policy = QwenSvgPolicy(policy_cfg)
        env = SvgRlEnvironment(target_image_path=args.target_image)
        trainer_cfg = TrainerConfig(
            episodes=args.episodes,
            max_env_steps=args.max_env_steps,
            learning_rate=args.lr,
            checkpoint_dir=args.checkpoint_dir,
        )
        trainer = SvgRlQwenTrainer(env, policy, trainer_cfg)
        trainer.train()
    elif args.command == "infer":
        run_inference(model_path=args.checkpoint, target_image_path=args.target_image)


if __name__ == "__main__":  # pragma: no cover
    main()
