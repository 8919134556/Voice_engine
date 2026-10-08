"""
synthetic.py — fake, speech-LIKE test audio and embeddings.

Used by tests, demos and the notebook so the project can run WITHOUT anyone's
personal recordings. None of this is a real voice:
  - "voiced" sounds = a buzzing tone (pitch f0 + harmonics), like a vowel
  - "hiss" = filtered noise, like an "s"
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from src.config.settings import EMBEDDING_DIMENSION, SAMPLE_RATE


def synthetic_speech(seconds: float = 3.0, f0: float = 120.0, seed: int = 0,
                     sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    A mono float32 signal: silence, a vowel-like buzz at pitch `f0`, a short hiss,
    another buzz, silence. Different f0 values give different "fake speakers".
    """
    rng = np.random.default_rng(seed)

    def silence(dur):
        return 0.002 * rng.normal(size=int(sample_rate * dur))

    def voiced(dur, pitch, amp):
        t = np.arange(int(sample_rate * dur)) / sample_rate
        phase = 2 * np.pi * np.cumsum(pitch * (1 + 0.08 * np.sin(2 * np.pi * 3 * t))) / sample_rate
        x = sum(np.sin(k * phase) / k for k in range(1, 20))
        return amp * x / np.abs(x).max() * np.sin(np.pi * t / dur) ** 2

    def hiss(dur, amp):
        n = np.diff(rng.normal(size=int(sample_rate * dur)), prepend=0)
        return amp * n / np.abs(n).max() * np.hanning(len(n))

    part = seconds / 6
    signal = np.concatenate([
        silence(part), voiced(1.5 * part, f0, 0.5), silence(0.5 * part),
        hiss(0.5 * part, 0.15), voiced(1.5 * part, f0 * 1.15, 0.6), silence(part),
    ])
    # Rounding each part can lose a sample or two -> pad/trim to the exact requested length.
    target = int(round(seconds * sample_rate))
    signal = np.pad(signal, (0, max(0, target - len(signal))))[:target]
    return signal.astype(np.float32)


def write_synthetic_dataset(dataset_dir: str | Path, speakers: dict[str, float] | None = None,
                            files_per_speaker: int = 3) -> Path:
    """
    Create   dataset_dir/<speaker>/audio_001.wav ...   with synthetic speech.
    speakers: {folder name: base pitch in Hz}; default is one fake speaker.
    """
    dataset_dir = Path(dataset_dir)
    speakers = speakers or {"speaker_01": 120.0}
    for s_index, (speaker, f0) in enumerate(speakers.items()):
        folder = dataset_dir / speaker
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(1, files_per_speaker + 1):
            audio = synthetic_speech(seconds=2.5 + 0.5 * i, f0=f0 * (1 + 0.03 * i), seed=100 * s_index + i)
            sf.write(str(folder / f"audio_{i:03d}.wav"), audio, SAMPLE_RATE, subtype="PCM_16")
    return dataset_dir


def random_embedding(seed: int = 0, dim: int = EMBEDDING_DIMENSION, scale: float = 1.0) -> np.ndarray:
    """A random float32 vector standing in for a speaker embedding (NOT a real voice)."""
    return (scale * np.random.default_rng(seed).normal(size=dim)).astype(np.float32)
