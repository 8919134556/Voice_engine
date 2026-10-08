"""
train.py — train the speaker encoder with a speaker-classification objective.

    for each epoch:
        train:      random 2-second crops ─► encoder ─► head ─► cross-entropy ─► Adam step
        validate:   middle 2-second crops ─► loss + accuracy (no weight updates)
        checkpoint: last_model.pt every epoch, best_model.pt when validation loss improves
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import settings
from src.utils.logging import get_logger

from .dataset import Recording, SpeakerDataset
from .losses import speaker_classification_loss
from .model import SpeakerClassifier, TrainableSpeakerEncoder
from .utils import save_checkpoint, set_seed

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """All knobs in one place. Defaults come from src/config/settings.py."""
    epochs: int = settings.EPOCHS
    batch_size: int = settings.BATCH_SIZE
    learning_rate: float = settings.LEARNING_RATE
    weight_decay: float = settings.WEIGHT_DECAY
    segment_frames: int = settings.SEGMENT_FRAMES
    embedding_dim: int = settings.EMBEDDING_DIMENSION
    n_mels: int = settings.N_MELS
    sample_rate: int = settings.SAMPLE_RATE
    seed: int = settings.TRAINING_SEED
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "segment_frames", "embedding_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"TrainingConfig.{name} must be positive.")
        if not self.learning_rate > 0:
            raise ValueError("TrainingConfig.learning_rate must be positive.")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainingResult:
    history: dict                 # per-epoch lists: train_loss, train_accuracy, val_loss, val_accuracy
    best_epoch: int
    best_checkpoint: Path
    last_checkpoint: Path
    classifier: SpeakerClassifier  # the trained model (encoder + head), best weights loaded


def run_epoch(classifier: SpeakerClassifier, loader: DataLoader, loss_fn, device: torch.device,
              optimizer: torch.optim.Optimizer | None = None) -> tuple[float, float]:
    """One pass over `loader`. With an optimizer: training; without: evaluation. Returns (loss, accuracy)."""
    training = optimizer is not None
    classifier.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = classifier(features)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += len(labels)
    return total_loss / max(count, 1), correct / max(count, 1)


def train_speaker_encoder(splits: dict[str, list[Recording]], speakers: dict[str, int],
                          config: TrainingConfig, device: torch.device,
                          checkpoint_dir: str | Path = settings.CHECKPOINT_DIR,
                          verbose: bool = True) -> TrainingResult:
    """Train, keep the best model by validation loss (training loss if there is no val split)."""
    if len(speakers) < settings.MIN_SPEAKERS_FOR_TRAINING:
        raise ValueError(f"Need at least {settings.MIN_SPEAKERS_FOR_TRAINING} speakers to train "
                         f"a speaker classifier; found {len(speakers)}.")
    set_seed(config.seed)
    checkpoint_dir = Path(checkpoint_dir)

    train_set = SpeakerDataset(splits["train"], config.segment_frames, train=True)
    val_set = SpeakerDataset(splits["val"], config.segment_frames, train=False) if splits["val"] else None
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=config.batch_size) if val_set else None

    encoder = TrainableSpeakerEncoder(config.embedding_dim, config.n_mels)
    classifier = SpeakerClassifier(encoder, n_speakers=len(speakers)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    loss_fn = speaker_classification_loss(config.label_smoothing)

    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    best_score, best_epoch = float("inf"), 0
    best_path, last_path = checkpoint_dir / "best_model.pt", checkpoint_dir / "last_model.pt"

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = run_epoch(classifier, train_loader, loss_fn, device, optimizer)
        if val_loader:
            val_loss, val_acc = run_epoch(classifier, val_loader, loss_fn, device)
        else:
            val_loss, val_acc = float("nan"), float("nan")
        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)

        if verbose:
            print(f"Epoch {epoch}/{config.epochs}  Train Loss: {train_loss:.4f}  Train Accuracy: {train_acc:.3f}"
                  f"  Val Loss: {val_loss:.4f}  Val Accuracy: {val_acc:.3f}")

        metric = {"val_loss": val_loss, "val_accuracy": val_acc, "train_loss": train_loss}
        save_checkpoint(last_path, classifier, optimizer, epoch, metric, config.to_dict(), speakers)
        score = val_loss if val_loader else train_loss
        if score < best_score:
            best_score, best_epoch = score, epoch
            save_checkpoint(best_path, classifier, optimizer, epoch, metric, config.to_dict(), speakers)
            logger.info("New best model at epoch %d (score %.4f)", epoch, score)

    best = torch.load(best_path, map_location=device, weights_only=True)
    classifier.encoder.load_state_dict(best["model_state_dict"])
    classifier.head.load_state_dict(best["head_state_dict"])
    classifier.eval()
    return TrainingResult(history, best_epoch, best_path, last_path, classifier)


def save_history(history: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return path


def save_splits(splits: dict[str, list[Recording]], speakers: dict[str, int], path: str | Path) -> Path:
    """Record exactly which files went where (reproducible evaluation; stays in git-ignored outputs/)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"speakers": speakers,
            **{name: [{"path": str(r.path), "speaker": r.speaker, "label": r.label} for r in recs]
               for name, recs in splits.items()}}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_splits(path: str | Path) -> tuple[dict[str, list[Recording]], dict[str, int]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = {name: [Recording(Path(r["path"]), r["speaker"], r["label"]) for r in data[name]]
              for name in ("train", "val", "test")}
    return splits, data["speakers"]
