"""
settings.py — the project's important numbers and paths, in ONE place.

Every module imports from here instead of hardcoding values like 128 or 16000,
so changing a setting (e.g. a future 256-D encoder) means editing one line.

This file must not import anything from `src` (everything else imports it).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — always relative to the project root, never absolute (Windows + Colab)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]      # src/config/settings.py -> project root
DATASET_DIR = PROJECT_ROOT / "dataset"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REGISTRY_FILE = PROJECT_ROOT / "voice_profiles.json"

# ---------------------------------------------------------------------------
# Audio (Phases 1-2)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000          # Hz; every recording is converted to 16 kHz mono
MIN_DURATION_SEC = 1.0       # shorter recordings get an "unusually short" warning
N_FFT = 400                  # STFT window: 400 samples = 25 ms at 16 kHz
HOP_LENGTH = 160             # STFT step:   160 samples = 10 ms -> 100 frames/second
N_MELS = 80                  # Mel bands

# ---------------------------------------------------------------------------
# Speaker embeddings (Phases 3-7)
# ---------------------------------------------------------------------------
EMBEDDING_DIMENSION = 128    # size of every speaker embedding / conditioning vector
ENCODER_SEED = 0             # seed for the (untrained) SpeakerEncoder's random weights

# ---------------------------------------------------------------------------
# Verification (Phase 4) — EDUCATIONAL example, not a calibrated threshold
# ---------------------------------------------------------------------------
VERIFICATION_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# Voices (Phases 5-7)
# ---------------------------------------------------------------------------
DEFAULT_LANGUAGE = "en"

# ---------------------------------------------------------------------------
# Logging (Phase 8)
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(levelname)s - %(message)s"


def validate_settings() -> None:
    """Fail fast with a clear message if a setting is nonsense."""
    positive_ints = {
        "SAMPLE_RATE": SAMPLE_RATE, "N_FFT": N_FFT, "HOP_LENGTH": HOP_LENGTH,
        "N_MELS": N_MELS, "EMBEDDING_DIMENSION": EMBEDDING_DIMENSION,
    }
    for name, value in positive_ints.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Setting {name} must be a positive integer, got {value!r}.")
    if HOP_LENGTH > N_FFT:
        raise ValueError("HOP_LENGTH must not be larger than N_FFT.")
    if not -1.0 <= VERIFICATION_THRESHOLD <= 1.0:
        raise ValueError("VERIFICATION_THRESHOLD must be between -1 and 1 (cosine similarity).")
    if not DEFAULT_LANGUAGE or DEFAULT_LANGUAGE != DEFAULT_LANGUAGE.strip().lower():
        raise ValueError("DEFAULT_LANGUAGE must be a lowercase language code such as 'en'.")
