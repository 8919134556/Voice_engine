"""
audio.py — load, check and clean voice recordings (Step 1).

    original WAV ─► load (mono) ─► resample to 22050 Hz ─► trim silence at start/end
                ─► same loudness for every clip ─► cleaned WAV

Why clean? A voice-cloning model copies everything in the reference audio:
long silences, very quiet or very loud clips, and background noise all make the
cloned voice worse. Clean, consistent references give the best result.
"""

from dataclasses import dataclass, field
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import config


@dataclass
class ClipReport:
    """What we found out about one recording."""
    name: str
    sample_rate: int = 0
    channels: int = 0
    duration_sec: float = 0.0          # original length
    speech_sec: float = 0.0            # length after trimming silence
    peak: float = 0.0
    loudness_db: float = 0.0
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def load_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV file -> (1-D float32 samples in -1..1, sample rate). Stereo is averaged to mono."""
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), sample_rate


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Change the sample rate (e.g. 48000 -> 22050 Hz) with an anti-aliasing filter."""
    if orig_sr == target_sr:
        return audio
    g = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


def loudness_db(audio: np.ndarray) -> float:
    """Average loudness in dBFS (0 dB = maximum possible; speech is usually -30 to -15)."""
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    return 20 * np.log10(rms) if rms > 0 else float("-inf")


def trim_silence(audio: np.ndarray, sample_rate: int,
                 threshold_db: float = config.SILENCE_THRESHOLD_DB,
                 padding_sec: float = config.SILENCE_PADDING_SEC) -> np.ndarray:
    """
    Remove silence at the START and END only (pauses between words are kept).
    Works on 20 ms frames: a frame is "speech" if it is within `threshold_db` of the loudest frame.
    """
    frame = max(1, int(0.02 * sample_rate))
    n_frames = len(audio) // frame
    if n_frames == 0:
        return audio
    frames = audio[: n_frames * frame].reshape(n_frames, frame)
    frame_db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-10)
    speech = np.where(frame_db > frame_db.max() + threshold_db)[0]
    if speech.size == 0:
        return audio
    pad = int(padding_sec * sample_rate)
    start = max(0, speech[0] * frame - pad)
    end = min(len(audio), (speech[-1] + 1) * frame + pad)
    return audio[start:end]


def normalize_loudness(audio: np.ndarray, target_db: float = config.TARGET_LOUDNESS_DB,
                       peak_limit: float = config.PEAK_LIMIT) -> np.ndarray:
    """Scale so the average loudness is `target_db`, but never let the peak exceed `peak_limit`."""
    current = loudness_db(audio)
    if not np.isfinite(current):
        return audio
    gain = 10 ** ((target_db - current) / 20)
    peak = float(np.max(np.abs(audio)))
    if peak * gain > peak_limit:
        gain = peak_limit / peak
    return (audio * gain).astype(np.float32)


def check_clip(path: str | Path) -> tuple[ClipReport, np.ndarray | None]:
    """Load + analyse one file. Returns the report and the cleaned audio (None if unusable)."""
    path = Path(path)
    report = ClipReport(name=path.name)
    try:
        info = sf.info(str(path))
        report.sample_rate, report.channels = info.samplerate, info.channels
        audio, sr = load_mono(path)
    except Exception as exc:
        report.problems.append(f"cannot read file ({exc})")
        return report, None

    report.duration_sec = len(audio) / sr if sr else 0.0
    report.peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    report.loudness_db = loudness_db(audio)
    if audio.size == 0 or report.peak < 1e-3:
        report.problems.append("silent or empty")
        return report, None
    if report.peak >= 0.999:
        report.problems.append("clipped (too loud while recording)")

    cleaned = resample(audio, sr, config.CLEAN_SAMPLE_RATE)
    cleaned = trim_silence(cleaned, config.CLEAN_SAMPLE_RATE)
    cleaned = normalize_loudness(cleaned)
    report.speech_sec = len(cleaned) / config.CLEAN_SAMPLE_RATE
    if report.speech_sec < config.MIN_CLIP_SEC:
        report.problems.append(f"too short ({report.speech_sec:.1f}s of speech)")
    if report.speech_sec > config.MAX_CLIP_SEC:
        report.problems.append(f"too long ({report.speech_sec:.1f}s) - split it into shorter clips")
    return report, cleaned


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int = config.CLEAN_SAMPLE_RATE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")
    return path
