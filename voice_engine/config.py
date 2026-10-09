"""
config.py — all settings in one place. Paths are relative to the project folder,
so the same code works on Windows and in Google Colab.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Your original recordings (PRIVATE, never pushed to GitHub): dataset/<voice name>/*.wav
DATASET_DIR = PROJECT_ROOT / "dataset"

# Cleaned copies made by Step 1 (also private)
CLEAN_DIR = PROJECT_ROOT / "dataset_clean"

# Generated speech and reports (private)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# --- Step 1: audio preparation -------------------------------------------
CLEAN_SAMPLE_RATE = 22050     # Hz, mono. Voice-cloning models read their reference audio at this rate.
TARGET_LOUDNESS_DB = -20.0    # every clip is brought to the same average loudness (dBFS)
PEAK_LIMIT = 0.98             # never louder than this (avoids clipping)
SILENCE_THRESHOLD_DB = -40.0  # quieter than this (relative to the loudest part) counts as silence
SILENCE_PADDING_SEC = 0.15    # keep a little silence before/after speech so words aren't cut
MIN_CLIP_SEC = 2.0            # shorter clips are flagged (too little voice information)
MAX_CLIP_SEC = 30.0           # longer clips are flagged (cloning models prefer shorter references)

# --- Step 2: text -> speech in your voice (XTTS-v2, pretrained voice cloning) ----
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_SAMPLE_RATE = 24000      # Hz, the model's output rate
LANGUAGES = {"en": "English", "hi": "Hindi"}   # XTTS-v2 knows more; this project uses these two
SPEECH_DIR = OUTPUTS_DIR / "speech"
# How much of your reference audio the model listens to (seconds). Longer = closer to your voice.
GPT_COND_LEN = 30
GPT_COND_CHUNK_LEN = 4
MAX_REF_LENGTH = 60
# Randomness of the delivery: lower = steadier/flatter, higher = more expressive but less stable.
TEMPERATURE = 0.75
