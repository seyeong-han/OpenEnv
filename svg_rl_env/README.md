# SVG-RL Environment

**Rendering-Aware Reinforcement Learning for Vector Graphics Generation**

Based on the paper: ["Rendering-Aware Reinforcement Learning for Vector Graphics Generation" (arxiv 2505.20793)](https://arxiv.org/pdf/2505.20793)

## Overview

The SVG-RL environment enables agents to learn vector graphics generation through visual feedback. The agent generates SVG code, which is rendered and compared against a target image using multiple visual metrics (SSIM, MSE, perceptual distance, edge similarity).

This implementation follows the **RLRF (Rendering Feedback for Reinforcement Learning)** approach where:
1. Agent generates SVG code
2. SVG is rendered to a raster image
3. Multiple visual metrics compare rendered output with target
4. Reward is computed from weighted combination of metrics
5. Agent learns to improve SVG generation based on feedback

## Features

- **Multi-metric Visual Feedback**: SSIM, pixel similarity, perceptual distance, edge similarity, color histogram matching
- **Flexible SVG Generation**: Support for both complete and incremental SVG generation
- **Configurable Targets**: Use custom images or auto-generate simple geometric shapes
- **Detailed Statistics**: Comprehensive metrics for SVG complexity and rendering quality
- **Docker Support**: Easy deployment with pre-configured environment

## Quick Start

### Using Docker (Recommended)

```python
from svg_rl_env import SvgRlAction, SvgRlEnv

# Start environment from Docker image
client = SvgRlEnv.from_docker_image("svg-rl-env:latest")

# Reset to get target image
result = client.reset()
print(f"Target set, SSIM: {result.observation.structural_similarity}")

# Generate SVG
svg_code = '<circle cx="128" cy="128" r="50" fill="red"/>'
result = client.step(SvgRlAction(svg_code=svg_code, is_complete=True))

print(f"Reward: {result.reward}")
print(f"SSIM: {result.observation.structural_similarity:.4f}")
print(f"Pixel Similarity: {result.observation.pixel_similarity:.4f}")

# Cleanup
client.close()
```

### Building Docker Image

```bash
# Build from the svg_rl_env directory
cd svg_rl_env
docker build -t svg-rl-env:latest -f server/Dockerfile .

# Or build from OpenEnv repo root
docker build -t svg-rl-env:latest -f src/envs/svg_rl_env/server/Dockerfile .
```

### Local Development (Requires Cairo)

**Mac:**
```bash
# Install Cairo via Homebrew
brew install cairo pkg-config

# If using Conda: Link to Homebrew's Cairo library
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"

# To make this permanent, add to ~/.zshrc or ~/.bash_profile:
# echo 'export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"' >> ~/.zshrc
```

> **Note for Conda users**: Conda environments are isolated from system libraries. The `DYLD_LIBRARY_PATH` export allows Python to find the Homebrew-installed Cairo library. Alternatively, install Cairo directly in your conda environment with `conda install -c conda-forge cairo`.

**Ubuntu/Debian:**
```bash
sudo apt-get install libcairo2-dev pkg-config python3-dev
```

**Then install the environment:**
```bash
cd svg_rl_env
pip install -e .

# Run server
uvicorn svg_rl_env.server.app:app --host 0.0.0.0 --port 8000
```

## Environment Details

### Action Space

```python
@dataclass
class SvgRlAction(Action):
    svg_code: str          # SVG code to render
    is_complete: bool      # Whether this completes the SVG
```

### Observation Space

```python
@dataclass
class SvgRlObservation(Observation):
    # Visual similarity metrics
    pixel_similarity: float              # MSE-based similarity (0-1)
    structural_similarity: float         # SSIM score (0-1)
    perceptual_distance: float           # Perceptual distance metric (0-1)
    edge_similarity: float               # Edge detection similarity (0-1)
    color_histogram_distance: float      # Color distribution difference (0-1)
    
    # SVG statistics
    svg_complexity: int                  # Number of SVG elements
    svg_valid: bool                      # Whether SVG is valid
    
    # Rendered images (base64 encoded PNG)
    rendered_image: Optional[str]        # Current rendered image
    target_image: Optional[str]          # Target image
    
    # Episode info
    step_number: int
    done: bool
    reward: float
    metadata: Dict
```

### Reward Function

The reward is a weighted combination of multiple visual metrics:

```python
reward = (
    0.3 * pixel_similarity +
    0.4 * structural_similarity +
    0.2 * (1 - perceptual_distance) +
    0.1 * edge_similarity
)

# Bonus for near-perfect match (SSIM > 0.95)
if structural_similarity > 0.95:
    reward += 1.0
```

## Visual Metrics Explained

1. **SSIM (Structural Similarity Index)**: Measures perceived quality difference between images. Considers luminance, contrast, and structure. Range: [0, 1], higher is better.

2. **Pixel Similarity**: Direct pixel-level MSE comparison. Sensitive to exact pixel values. Range: [0, 1], higher is better.

3. **Perceptual Distance**: Measures color difference in LAB color space, which approximates human visual perception. Range: [0, 1], lower is better.

4. **Edge Similarity**: Compares edge maps using Canny edge detection. Focuses on shape matching. Range: [0, 1], higher is better.

5. **Color Histogram Distance**: Compares color distributions using chi-square distance. Range: [0, 1], lower is better.

## Example Usage

See `svg_rl_env_example.py` in the repository root for a complete example.

```python
from svg_rl_env.server.svg_rl_env_environment import SvgRlEnvironment

# Create environment
env = SvgRlEnvironment(
    image_size=(256, 256),
    max_steps=50,
    reward_weights={
        "pixel_similarity": 0.3,
        "structural_similarity": 0.4,
        "perceptual_distance": 0.2,
        "edge_similarity": 0.1,
    }
)

# Reset and get target
obs = env.reset()

# Try different SVG shapes
svg_circle = '<circle cx="128" cy="128" r="60" fill="red"/>'
obs = env.step(SvgRlAction(svg_code=svg_circle, is_complete=True))
print(f"Circle - Reward: {obs.reward:.4f}, SSIM: {obs.structural_similarity:.4f}")

svg_rect = '<rect x="68" y="68" width="120" height="120" fill="blue"/>'
obs = env.step(SvgRlAction(svg_code=svg_rect, is_complete=True))
print(f"Rectangle - Reward: {obs.reward:.4f}, SSIM: {obs.structural_similarity:.4f}")
```

## Integration with RL Algorithms

This environment is compatible with standard RL algorithms. Example with your favorite RL library:

```python
# Pseudo-code for RL training
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    
    while not done:
        # Agent generates SVG code (policy network output)
        svg_code = agent.generate_svg(obs)
        
        # Execute action
        action = SvgRlAction(svg_code=svg_code, is_complete=True)
        obs = env.step(action)
        
        # Store experience for training
        buffer.add(obs.reward, obs.structural_similarity, ...)
        
        # Update policy based on visual feedback
        agent.update()
```

## Qwen2.5-VL RL loop (paper-aligned)

Qwen2.5-VL can be used as the policy to optimize SVG code with rendering-aware rewards (arxiv 2505.20793).

### Option 1: Training via HTTP API (Recommended for Testing)

Start the server and trigger training via REST API:

```bash
# Terminal 1: Start the server
uvicorn svg_rl_env.server.app:app --reload

# Terminal 2: Start training
curl -X POST http://localhost:8000/api/training/start \
     -H "Content-Type: application/json" \
     -d '{"episodes": 1, "max_env_steps": 2}'
```

Monitor training status:
```bash
curl http://localhost:8000/api/training/status
```

Control training:
```bash
# Pause training
curl -X POST http://localhost:8000/api/training/pause

# Resume training
curl -X POST http://localhost:8000/api/training/resume

# Stop training
curl -X POST http://localhost:8000/api/training/stop
```

### Option 2: Command Line Training

1. Install optional dependencies:  
   ```bash
   pip install -e .[qwen]
   ```
2. Train with REINFORCE + EMA baseline (checkpoints under `qwen_svg_rl_checkpoints/`):  
   ```bash
   python -m svg_rl_env.qwen_svg_rl train --episodes 3 --max-env-steps 4
   ```
   - Add `--target-image path/to/target.png` to lock the goal image.
3. Run inference (uses base Qwen or a saved checkpoint):  
   ```bash
   python -m svg_rl_env.qwen_svg_rl infer --checkpoint qwen_svg_rl_checkpoints/best
   ```

### How It Works

The trainer queries Qwen2.5-VL with the target image + current metrics, renders via the environment, and applies a policy-gradient update to maximize the visual reward. See `svg_rl_env/qwen_svg_rl.py` for details.

## Comparison with Paper

This implementation follows the key ideas from arxiv 2505.20793:

| Paper Concept | Implementation |
|--------------|----------------|
| Visual-Language Model generates SVG | Agent generates `svg_code` via `SvgRlAction` |
| SVG rendering using cairo | `cairosvg` library renders to PNG |
| Image comparison metrics | SSIM, MSE, perceptual distance, edge similarity |
| Reward from visual feedback | Weighted combination of metrics |
| Iterative refinement | Incremental SVG generation with `is_complete=False` |

## Environment Configuration

```python
env = SvgRlEnvironment(
    image_size=(256, 256),              # Output image size
    target_image_path="path/to/image",  # Custom target (or None for random)
    max_steps=50,                       # Max steps per episode
    reward_weights={                    # Customize reward weights
        "pixel_similarity": 0.3,
        "structural_similarity": 0.4,
        "perceptual_distance": 0.2,
        "edge_similarity": 0.1,
    }
)
```

## Technical Details

- **SVG Rendering**: Uses `cairosvg` to convert SVG to PNG
- **Image Processing**: OpenCV and scikit-image for metric computation
- **Metrics**: SSIM (structural similarity), MSE (pixel-level), LAB color space (perceptual), Canny edges (shape)
- **Base64 Images**: Rendered and target images are returned as base64-encoded PNGs for visualization

## Citation

If you use this environment in your research, please cite both OpenEnv and the original paper:

```bibtex
@article{openenv2024,
  title={OpenEnv: Agentic Execution Environments},
  author={OpenEnv Team},
  year={2024}
}

@article{rlrf2025,
  title={Rendering-Aware Reinforcement Learning for Vector Graphics Generation},
  journal={arXiv preprint arXiv:2505.20793},
  year={2025}
}
```

## License

BSD 3-Clause License (see LICENSE file in repository root)

## Contributing

Contributions are welcome! Please see the main OpenEnv repository for contribution guidelines.

## Troubleshooting

**Cairo library not found**: Install cairo system library (see Local Development section above)

**Image rendering fails**: Ensure SVG code is valid XML with proper namespace declarations

**Low similarity scores**: Target generation is random - try multiple resets to find learnable targets

## Future Enhancements

- [ ] Support for more complex target images (photos, artwork)
- [ ] Path-based SVG generation (Bezier curves, paths)
- [ ] Multi-step incremental generation
- [ ] Curriculum learning with increasing difficulty
- [ ] Integration with VLM for natural language descriptions
