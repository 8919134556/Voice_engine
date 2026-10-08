"""
preprocessing.py — ONE definition of the model input, shared by training,
validation, testing and embedding generation.

    WAV ─► load_mono_16k()            (Phase 1: mono, 16 kHz)
        ─► Mel spectrogram, log (dB)  (Phase 2)
        ─► per-band normalization     (Phase 3: removes loudness / mic-gain differences)
        ─► [80, time] features

It simply reuses `waveform_to_encoder_input` from Phase 3, so the trained model
always sees exactly the same kind of input at training time and at inference time.

Audio normalization note: the per-band normalization of the log-Mel features
already cancels overall volume (a louder recording adds a constant in dB, which
the mean subtraction removes), so no separate waveform peak-normalization is needed.
"""

import torch

from src.audio.loader import load_mono_16k
from src.config.settings import SEGMENT_FRAMES
from src.embeddings.generate_embeddings import waveform_to_encoder_input


def waveform_to_features(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Mono waveform [1, samples] -> normalized log-Mel features [80, time]."""
    return waveform_to_encoder_input(waveform, sample_rate)[0, 0]


def load_features(path) -> torch.Tensor:
    """WAV file -> normalized log-Mel features [80, time] (any input rate / channel count)."""
    waveform, sample_rate = load_mono_16k(path)
    return waveform_to_features(waveform, sample_rate)


def crop_or_pad(features: torch.Tensor, n_frames: int = SEGMENT_FRAMES,
                random_crop: bool = False, generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Make every example exactly `n_frames` long so examples can be stacked into a batch.

    longer  -> training: a RANDOM window (a little data augmentation: a different
               2-second slice each epoch); evaluation: the MIDDLE window (repeatable)
    shorter -> repeat the recording until it is long enough (better than padding with
               zeros, which would teach the model that silence is part of a voice)
    """
    total = features.shape[1]
    if total < n_frames:
        repeats = -(-n_frames // total)              # ceiling division
        return features.repeat(1, repeats)[:, :n_frames]
    if total == n_frames:
        return features
    if random_crop:
        start = int(torch.randint(0, total - n_frames + 1, (1,), generator=generator))
    else:
        start = (total - n_frames) // 2
    return features[:, start:start + n_frames]
