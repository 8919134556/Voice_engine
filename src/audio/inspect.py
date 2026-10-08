"""
inspect.py — measure a recording and check it for problems.

Data flow:
    WAV path --inspect_audio()--> AudioInfo (numbers + list of issues)
             --format_report()--> human-readable text
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch

from .loader import TARGET_SAMPLE_RATE, load_audio

# Thresholds for the quality checks. Tweak these if they don't suit your recordings.
MIN_DURATION_SEC = 1.0    # shorter than this is "unusually short" for a speech clip
SILENCE_PEAK = 1e-3       # loudest sample below this -> the file is basically silent
CLIPPING_PEAK = 0.999     # loudest sample at/above this -> the recording probably clipped


@dataclass
class AudioInfo:
    """Everything we learned about one audio file."""
    path: Path
    sample_rate: int = 0
    channels: int = 0
    samples: int = 0
    duration_sec: float = 0.0
    shape: tuple = ()
    peak: float = 0.0     # loudest absolute sample value (0.0 .. 1.0)
    rms: float = 0.0      # average loudness (root mean square)
    errors: list = field(default_factory=list)    # file is unusable
    warnings: list = field(default_factory=list)  # file is usable but needs attention

    @property
    def filename(self):
        return self.path.name

    @property
    def is_mono(self):
        return self.channels == 1

    @property
    def ok(self):
        return not self.errors


def inspect_audio(path):
    """Load one file, measure it, and run all the checks. Never raises — problems go into .errors."""
    info = AudioInfo(path=Path(path))

    # 1. Load. A corrupted or non-audio file fails here.
    try:
        waveform, sample_rate = load_audio(path)
    except Exception as exc:
        info.errors.append(f"Could not load file (corrupted or not a WAV?): {exc}")
        return info

    # 2. Basic facts. waveform.shape is (channels, samples).
    info.sample_rate = sample_rate
    info.channels, info.samples = waveform.shape
    info.shape = waveform.shape
    info.duration_sec = info.samples / sample_rate if sample_rate else 0.0

    # 3. Empty file -> nothing else to check.
    if info.samples == 0:
        info.errors.append("File contains no audio samples (empty).")
        return info

    # 4. Loudness statistics.
    if not torch.isfinite(waveform).all():
        info.errors.append("Audio contains NaN/Inf values (corrupted data).")
        return info
    info.peak = waveform.abs().max().item()
    info.rms = waveform.pow(2).mean().sqrt().item()

    # 5. Quality checks.
    if info.peak < SILENCE_PEAK:
        info.errors.append("Audio is silent (check your microphone).")
    if info.duration_sec < MIN_DURATION_SEC:
        info.warnings.append(f"Unusually short: {info.duration_sec:.2f}s (< {MIN_DURATION_SEC}s).")
    if not info.is_mono:
        info.warnings.append(f"Not mono ({info.channels} channels) -> preprocessing will mix to mono.")
    if sample_rate != TARGET_SAMPLE_RATE:
        info.warnings.append(f"Sample rate is {sample_rate} Hz, not {TARGET_SAMPLE_RATE} Hz -> preprocessing will resample.")
    if info.peak >= CLIPPING_PEAK:
        info.warnings.append("Possible clipping (peak at maximum). Record a little quieter.")

    return info


def format_report(info):
    """Turn an AudioInfo into the text block printed by scripts/inspect_dataset.py."""
    lines = [info.filename]

    if info.samples or info.sample_rate:
        channel_label = "mono" if info.is_mono else "stereo" if info.channels == 2 else "multi-channel"
        lines += [
            f"  Sample rate: {info.sample_rate}",
            f"  Channels:    {info.channels} ({channel_label})",
            f"  Samples:     {info.samples}",
            f"  Duration:    {info.duration_sec:.2f} seconds",
            f"  Shape:       {info.shape}",
            f"  Peak / RMS:  {info.peak:.3f} / {info.rms:.4f}",
        ]

    for msg in info.errors:
        lines.append(f"  [ERROR] {msg}")
    for msg in info.warnings:
        lines.append(f"  [WARN]  {msg}")
    if info.ok and not info.warnings:
        lines.append("  [OK]    Mono, 16 kHz, looks good.")

    return "\n".join(lines)
