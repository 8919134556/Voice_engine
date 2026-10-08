"""Audio utilities for the Voice Engine (Phase 1: loading + inspection, Phase 2: features)."""

from .loader import (
    TARGET_SAMPLE_RATE,
    find_dataset_dir,
    list_speakers,
    list_audio_files,
    load_audio,
    to_mono,
    resample,
    preprocess_audio,
    save_audio,
    load_mono_16k,
)
from .inspect import AudioInfo, inspect_audio, format_report
from .features import stft_power, power_to_db, mel_filterbank, mel_spectrogram, extract_features
