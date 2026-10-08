"""
utils.py — device selection, seeding, and checkpoint save/load for the trained encoder.

A checkpoint (.pt, git-ignored) holds everything needed to restore the model:

    {
      "model_state_dict":     encoder weights (the classification head is NOT needed later,
                                               but is saved too, for resuming training),
      "optimizer_state_dict": optimizer state (resume training),
      "epoch":                which epoch produced it,
      "metric":               {"val_loss": ..., "val_accuracy": ...},
      "config":               the TrainingConfig used,
      "speakers":             {speaker name: label},
      "embedding_dim", "n_mels": architecture sizes
    }
"""

import random
from pathlib import Path

import numpy as np
import torch

from src.config.settings import BEST_CHECKPOINT

from .model import SpeakerClassifier, TrainableSpeakerEncoder


class CheckpointNotFoundError(FileNotFoundError):
    """A trained checkpoint was expected but does not exist."""


def get_device() -> torch.device:
    """CUDA GPU when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> str:
    text = f"Device: {device}\nCUDA available: {torch.cuda.is_available()}"
    if device.type == "cuda":
        text += f"\nGPU name: {torch.cuda.get_device_name(device)}"
    return text


def set_seed(seed: int) -> None:
    """Make weight initialization, shuffling and random crops repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(path: str | Path, classifier: SpeakerClassifier, optimizer: torch.optim.Optimizer | None,
                    epoch: int, metric: dict, config: dict, speakers: dict[str, int]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": classifier.encoder.state_dict(),
        "head_state_dict": classifier.head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metric": metric,
        "config": config,
        "speakers": speakers,
        "embedding_dim": classifier.encoder.embedding_dim,
        "n_mels": classifier.encoder.n_mels,
    }, path)
    return path


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict:
    """Read a checkpoint dict. Clear error if the file is missing — never a silent fallback."""
    path = Path(path)
    if not path.is_file():
        raise CheckpointNotFoundError(
            f"Trained speaker-encoder checkpoint not found: {path}. "
            "Train one first with: python scripts/train_speaker_encoder.py"
        )
    # weights_only=True: only tensors and plain Python data are loaded, no arbitrary code
    return torch.load(path, map_location=device, weights_only=True)


def load_speaker_encoder(path: str | Path = BEST_CHECKPOINT,
                         device: torch.device | None = None) -> TrainableSpeakerEncoder:
    """
    1. read the checkpoint   2. build the architecture with the saved sizes
    3. load the weights      4. model.eval()   5. move to device   6. return the encoder
    """
    device = device or get_device()
    checkpoint = load_checkpoint(path, device)
    encoder = TrainableSpeakerEncoder(embedding_dim=checkpoint["embedding_dim"], n_mels=checkpoint["n_mels"])
    encoder.load_state_dict(checkpoint["model_state_dict"])
    return encoder.eval().to(device)
