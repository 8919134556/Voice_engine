"""
loader.py — find, load and (optionally) preprocess WAV files.

Data flow:
    WAV file on disk  --load_audio()-->  torch.Tensor [channels, samples] + sample_rate
                      --preprocess_audio()-->  mono, 16 kHz tensor [1, samples]

Convention used everywhere in this project:
    waveform.shape == (channels, samples)
    e.g. 8 seconds of mono 16 kHz audio -> torch.Size([1, 128000])
"""

import os
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from src.config.settings import DATASET_DIR, PROJECT_ROOT, SAMPLE_RATE

# 16 kHz mono is the standard input format for speech models
# (speaker embeddings, ASR, many TTS systems). Value lives in src/config/settings.py.
TARGET_SAMPLE_RATE = SAMPLE_RATE

AUDIO_EXTENSIONS = {".wav"}


def find_dataset_dir(dataset_dir=None):
    """
    Decide where the dataset lives, in this order of priority:
      1. the `dataset_dir` argument (e.g. from --dataset on the command line)
      2. the VOICE_ENGINE_DATASET environment variable
      3. <project root>/dataset
    Raises FileNotFoundError if the folder does not exist.
    """
    if dataset_dir is None:
        dataset_dir = os.environ.get("VOICE_ENGINE_DATASET", DATASET_DIR)

    path = Path(dataset_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {path}")
    return path


def list_speakers(dataset_dir):
    """
    Return every sub-folder of the dataset (one folder = one speaker), sorted by name.
    Hidden folders such as ".ipynb_checkpoints" (created by Jupyter/Colab) are skipped.
    """
    return sorted(p for p in Path(dataset_dir).iterdir() if p.is_dir() and not p.name.startswith("."))


def list_audio_files(speaker_dir):
    """Return all WAV files inside one speaker folder, sorted by name."""
    return sorted(
        p for p in Path(speaker_dir).iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def load_audio(path):
    """
    Load a WAV file into a PyTorch tensor.

    Returns:
        waveform:    torch.float32 tensor, shape (channels, samples), values in [-1.0, 1.0]
        sample_rate: int, samples per second (e.g. 16000)

    Raises an exception if the file is missing or not a valid audio file.
    """
    # always_2d=True gives shape (samples, channels) even for mono files,
    # so mono and stereo are handled by the same code.
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)

    # soundfile uses (samples, channels); PyTorch audio convention is (channels, samples).
    waveform = torch.from_numpy(data.T.copy())
    return waveform, sample_rate


def to_mono(waveform):
    """Average all channels into one: (channels, samples) -> (1, samples)."""
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def resample(waveform, orig_sr, target_sr=TARGET_SAMPLE_RATE):
    """
    Change the sample rate, e.g. 44100 Hz -> 16000 Hz.

    resample_poly applies an anti-aliasing filter, so high frequencies that
    can't be represented at the new rate are removed cleanly instead of
    turning into noise.
    """
    if orig_sr == target_sr:
        return waveform

    # 44100 -> 16000 is the ratio 160/441; dividing by the GCD gives the smallest integers.
    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g

    resampled = resample_poly(waveform.numpy(), up, down, axis=1)
    return torch.from_numpy(resampled.astype(np.float32))


def preprocess_audio(waveform, sample_rate, target_sr=TARGET_SAMPLE_RATE):
    """
    Convert any loaded audio into the project's standard format: mono, 16 kHz.
    Returns (waveform, target_sr).
    """
    waveform = to_mono(waveform)
    waveform = resample(waveform, sample_rate, target_sr)
    return waveform, target_sr


def save_audio(path, waveform, sample_rate):
    """Save a (channels, samples) tensor as a 16-bit PCM WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.numpy().T, sample_rate, subtype="PCM_16")


def load_mono_16k(path):
    """Phase 2 convenience: load_audio() + preprocess_audio() in one call."""
    waveform, sample_rate = load_audio(path)
    return preprocess_audio(waveform, sample_rate)
