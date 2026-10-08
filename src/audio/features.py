"""
features.py — turn a waveform into spectrograms and simple numeric features.

Data flow:
    waveform (1, samples)
       --stft_power()------------> spectrogram      (freq_bins, frames)
       --mel_filterbank() @ ...--> mel spectrogram  (n_mels, frames)
       --extract_features()------> {duration, rms, zcr, spectral_centroid}

Everything here is plain PyTorch (torch.stft), so it runs the same on CPU, GPU,
Windows and Colab.
"""

import torch

# Standard settings for 16 kHz speech (values live in src/config/settings.py):
#   N_FFT = 400 samples = 25 ms window, HOP_LENGTH = 160 samples = 10 ms step, N_MELS = 80 bands
from src.config.settings import HOP_LENGTH, N_FFT, N_MELS


# ---------------------------------------------------------------------------
# Spectrogram (STFT)
# ---------------------------------------------------------------------------

def stft_magnitude(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Short-Time Fourier Transform:
      1. cut the waveform into short overlapping windows (25 ms each, every 10 ms)
      2. for each window, measure how strong each frequency is

    Input:  waveform (1, samples)
    Output: magnitude (n_fft // 2 + 1, frames) -> (201, frames) with the defaults
            row 0 = 0 Hz, last row = sample_rate / 2 (8000 Hz at 16 kHz)
    """
    window = torch.hann_window(n_fft)  # smooth fade-in/out so window edges don't add fake frequencies
    stft = torch.stft(
        waveform[0],
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    return stft.abs()  # complex numbers -> strength (we ignore phase)


def stft_power(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Power spectrogram = magnitude squared (proportional to energy)."""
    return stft_magnitude(waveform, n_fft, hop_length) ** 2


def power_to_db(spec, top_db=80.0):
    """
    Convert power to decibels (log scale), like our ears perceive loudness.
    Values more than `top_db` below the loudest point are clipped so the
    plot isn't dominated by near-silent noise.
    """
    db = 10.0 * torch.log10(spec.clamp(min=1e-10))
    return db.clamp(min=db.max() - top_db)


# ---------------------------------------------------------------------------
# Mel scale
# ---------------------------------------------------------------------------

def hz_to_mel(hz):
    """HTK Mel formula: equal steps in Mel ~ equal steps in perceived pitch."""
    return 2595.0 * torch.log10(1.0 + torch.as_tensor(hz, dtype=torch.float32) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (torch.as_tensor(mel, dtype=torch.float32) / 2595.0) - 1.0)


def mel_filterbank(sample_rate, n_fft=N_FFT, n_mels=N_MELS, f_min=0.0, f_max=None):
    """
    Build `n_mels` triangular filters spaced evenly on the Mel scale.

    Each filter is a triangle over the linear frequency axis. Low-frequency
    triangles are narrow (fine detail where our ears are sensitive),
    high-frequency triangles are wide (coarse detail where we hear less difference).

    Returns: (n_fft // 2 + 1, n_mels) matrix -> (201, 80) with the defaults.
    """
    f_max = f_max if f_max is not None else sample_rate / 2
    fft_freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)   # Hz of each STFT row

    # n_mels + 2 points evenly spaced in Mel -> converted back to Hz.
    # Filter i rises from point i to point i+1, then falls to point i+2.
    mel_points = torch.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fbank = torch.zeros(len(fft_freqs), n_mels)
    for i in range(n_mels):
        left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        rising = (fft_freqs - left) / (center - left)
        falling = (right - fft_freqs) / (right - center)
        fbank[:, i] = torch.minimum(rising, falling).clamp(min=0)
    return fbank


def mel_center_frequencies(sample_rate, n_mels=N_MELS, f_min=0.0, f_max=None):
    """The Hz value at the peak of each Mel filter (useful for plot labels)."""
    f_max = f_max if f_max is not None else sample_rate / 2
    mel_points = torch.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    return mel_to_hz(mel_points)[1:-1]


def mel_spectrogram(waveform, sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    Spectrogram -> Mel spectrogram: each Mel band = weighted sum of nearby STFT rows.

    (n_mels, freq_bins) @ (freq_bins, frames) -> (n_mels, frames)
    """
    power = stft_power(waveform, n_fft, hop_length)
    fbank = mel_filterbank(sample_rate, n_fft, n_mels)
    return fbank.T @ power


# ---------------------------------------------------------------------------
# Basic features (one number per recording)
# ---------------------------------------------------------------------------

def rms_energy(waveform):
    """Root Mean Square: square every sample, average, square-root. ~ overall loudness."""
    return waveform.pow(2).mean().sqrt().item()


def zero_crossing_rate(waveform):
    """
    Fraction of neighbouring sample pairs where the signal changes sign.
    Noisy sounds ("s", "f", "sh") cross zero often; voiced sounds ("a", "o", "m") less.
    """
    signs = torch.sign(waveform[0])
    crossings = (signs[1:] * signs[:-1] < 0).sum().item()
    return crossings / max(len(signs) - 1, 1)


def spectral_centroid(waveform, sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    "Centre of mass" of the spectrum in Hz: where, on average, the energy sits.
    Computed on the average spectrum of the whole file, so loud (speech) frames
    count more than quiet (silence) frames. Higher = brighter / hissier sound.
    """
    magnitude = stft_magnitude(waveform, n_fft, hop_length)
    avg_spectrum = magnitude.mean(dim=1)                       # (freq_bins,)
    freqs = torch.linspace(0, sample_rate / 2, len(avg_spectrum))
    total = avg_spectrum.sum()
    if total == 0:
        return 0.0
    return ((freqs * avg_spectrum).sum() / total).item()


def extract_features(waveform, sample_rate):
    """All Phase 2 features for one mono waveform, as a dict (one CSV row)."""
    return {
        "duration": waveform.shape[1] / sample_rate,
        "rms": rms_energy(waveform),
        "zcr": zero_crossing_rate(waveform),
        "spectral_centroid": spectral_centroid(waveform, sample_rate),
    }
