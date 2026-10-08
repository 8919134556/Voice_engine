"""
Speaker-encoder training (Phase 9): dataset discovery and splitting, a trainable
encoder with a training-only classification head, training loop with checkpoints,
and evaluation (classification accuracy + genuine/impostor verification metrics).
"""

from .dataset import Recording, SpeakerDataset, discover_recordings, discover_speakers, split_recordings
from .model import SpeakerClassifier, TrainableSpeakerEncoder, count_parameters
from .train import TrainingConfig, TrainingResult, train_speaker_encoder
from .utils import (
    CheckpointNotFoundError,
    describe_device,
    get_device,
    load_checkpoint,
    load_speaker_encoder,
    save_checkpoint,
)
