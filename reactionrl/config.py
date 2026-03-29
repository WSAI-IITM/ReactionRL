"""
Central configuration for the reactionrl package.

Directory paths are auto-detected from the package location, or can be
overridden via the MAIN_DIR environment variable. TrainingConfig holds
all hyperparameters needed for offline RL training.
"""
import os
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path

# Directory paths - auto-detect from package root, or use MAIN_DIR env var
_PACKAGE_DIR = Path(__file__).resolve().parent
MAIN_DIR = Path(os.getenv("MAIN_DIR", str(_PACKAGE_DIR.parent)))
DATASETS_DIR = MAIN_DIR / "datasets"
PRETRAINED_MODELS_DIR = MAIN_DIR / "pretrained_models"
OUTPUT_DIR = MAIN_DIR / "output"


@dataclass
class TrainingConfig:
    """Configuration for offline RL training.

    Controls hyperparameters, model architecture, and training behavior.
    All fields have sensible defaults matching the original training setup.

    Attributes:
        actor_lr: Learning rate for actor optimizer.
        critic_lr: Learning rate for critic optimizer.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        topk: Number of top-k negative samples for PG loss.
        steps: Trajectory length (corresponds to dataset filename).
        model_type: One of "actor", "critic", or "actor-critic".
        actor_loss: Loss type for actor - "mse" or "PG" (policy gradient).
        negative_selection: Strategy for negative sampling.
        seed: Random seed for reproducibility.
        device: Torch device string (e.g. "cpu", "cuda:0").
        num_workers: Number of multiprocessing workers for data preparation.
        train_frac: Fraction of data used for training (rest is validation).
        gin_model_path: Path to pretrained GIN model. Defaults to
            pretrained_models/zinc2m_gin.pth.
        hidden_size: Hidden layer size for actor/critic dense networks.
        actor_num_hidden: Number of hidden layers in actor network.
        critic_num_hidden: Number of hidden layers in critic network.
    """
    # Optimization
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 128
    topk: int = 10

    # Data
    steps: int = 5
    train_frac: float = 0.8
    num_workers: int = min(cpu_count(), 35)

    # Model
    model_type: str = "actor-critic"
    actor_loss: str = "PG"
    negative_selection: str = "combined"
    gin_model_path: str = ""  # empty string means use default
    hidden_size: int = 256
    actor_num_hidden: int = 3
    critic_num_hidden: int = 2

    # Runtime
    seed: int = 42
    device: str = "cpu"

    def get_gin_model_path(self) -> str:
        """Returns the GIN model path, falling back to the default location."""
        if self.gin_model_path:
            return self.gin_model_path
        return str(PRETRAINED_MODELS_DIR / "zinc2m_gin.pth")
