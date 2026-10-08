"""
dataset.py — discover speakers, split recordings, and serve (features, label) pairs.

    dataset/
    ├── speaker_001/ audio_001.wav, audio_002.wav, ...   -> label 0
    ├── speaker_002/ ...                                  -> label 1
    └── ...                                               -> label 2, 3, ...

Folder name = speaker label. Folders are sorted alphabetically and numbered from 0,
so the mapping is the same on every machine. The number of speakers is never
hard-coded, and speakers may have different numbers of recordings.
"""

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.audio.loader import list_audio_files
from src.config.settings import (
    SEGMENT_FRAMES,
    SPLIT_SEED,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION,
)

from .preprocessing import crop_or_pad, load_features


@dataclass(frozen=True)
class Recording:
    path: Path
    speaker: str     # folder name, e.g. "speaker_001"
    label: int       # integer class used by the classifier


def discover_speakers(dataset_dir: str | Path) -> dict[str, int]:
    """
    {speaker folder name: integer label}, for every sub-folder that contains WAV files.
    Raises FileNotFoundError if the dataset folder does not exist.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    speakers = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir() and list_audio_files(p))
    return {name: label for label, name in enumerate(speakers)}


def discover_recordings(dataset_dir: str | Path) -> tuple[list[Recording], dict[str, int]]:
    """All recordings with their labels, plus the speaker -> label mapping."""
    dataset_dir = Path(dataset_dir)
    speaker_to_label = discover_speakers(dataset_dir)
    recordings = [
        Recording(path, speaker, label)
        for speaker, label in speaker_to_label.items()
        for path in list_audio_files(dataset_dir / speaker)
    ]
    return recordings, speaker_to_label


def split_recordings(recordings: list[Recording], train: float = TRAIN_FRACTION,
                     val: float = VAL_FRACTION, test: float = TEST_FRACTION,
                     seed: int = SPLIT_SEED) -> tuple[dict[str, list[Recording]], list[str]]:
    """
    Split BY RECORDING, separately for each speaker:

      - every speaker appears in train (and in val/test when it has enough files),
        which a speaker *classifier* needs: it can only predict speakers it has seen;
      - a WAV file is in exactly ONE split, so no audio leaks from training into evaluation.

    Per speaker with n files: val = round(n*val), test = round(n*test), at least 1 each
    when n >= 3; the rest is train. Speakers with fewer than 3 files go entirely to
    train and are reported in the returned warnings.

    Limitation: val/test contain NEW RECORDINGS of KNOWN speakers. Generalizing to
    completely unseen speakers would need held-out speakers (future work).
    """
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("train + val + test fractions must add up to 1.0")

    by_speaker: dict[str, list[Recording]] = {}
    for rec in recordings:
        by_speaker.setdefault(rec.speaker, []).append(rec)

    rng = random.Random(seed)
    splits: dict[str, list[Recording]] = {"train": [], "val": [], "test": []}
    warnings: list[str] = []
    for speaker in sorted(by_speaker):
        files = sorted(by_speaker[speaker], key=lambda r: r.path.name)
        rng.shuffle(files)
        n = len(files)
        if n < 3:
            splits["train"].extend(files)
            warnings.append(f"{speaker}: only {n} recording(s) -> all used for training, none for val/test.")
            continue
        n_val = max(1, round(n * val))
        n_test = max(1, round(n * test))
        if n - n_val - n_test < 1:  # always keep at least one training file
            n_val, n_test = 1, 1
        splits["val"].extend(files[:n_val])
        splits["test"].extend(files[n_val:n_val + n_test])
        splits["train"].extend(files[n_val + n_test:])
    return splits, warnings


class SpeakerDataset(Dataset):
    """
    features, label = dataset[i]

    features: tensor [1, 80, segment_frames]  (channel, Mel bands, time) — fixed size for batching
    label:    int speaker class

    Features of every recording are computed ONCE (load -> mono -> 16 kHz -> log-Mel ->
    normalize) and kept in memory; each __getitem__ then only cuts a window.
    train=True  -> random window (augmentation);  train=False -> middle window (repeatable).
    """

    def __init__(self, recordings: list[Recording], segment_frames: int = SEGMENT_FRAMES,
                 train: bool = False):
        self.recordings = list(recordings)
        self.segment_frames = segment_frames
        self.train = train
        self._features = [load_features(rec.path) for rec in self.recordings]

    def __len__(self) -> int:
        return len(self.recordings)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        features = crop_or_pad(self._features[index], self.segment_frames, random_crop=self.train)
        return features.unsqueeze(0), self.recordings[index].label

    def full_features(self, index: int) -> torch.Tensor:
        """The whole recording's features [80, time] (used for embedding extraction)."""
        return self._features[index]
