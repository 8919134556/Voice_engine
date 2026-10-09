# Voice Engine

Goal: **type text → hear it spoken in my own voice** (English + Hindi), built step by step.

```
your recordings ─► Step 1: clean ─► Step 2: voice cloning TTS ─► Step 3: engine.speak(text) ─► WAV in your voice
```

| Step | What | Status |
|---|---|---|
| 1 | Check and clean the voice recordings | ✅ done |
| 2 | Text → speech in your voice (pretrained voice-cloning model XTTS-v2, English + Hindi, Colab GPU) | ✅ current |
| 3 | Simple engine: `engine.speak("text", language="hi")` | planned |
| Later | More voices (`speaker_02`, …), better quality (more recordings, fine-tuning) | planned |

**Approach:** with ~1–2 minutes of recordings, training a speech model from scratch is not possible
(that needs many hours). Instead Step 2 uses a **pretrained voice-cloning model** that copies a voice from
short reference recordings. The earlier learning phases (1–9: audio analysis, speaker embeddings,
verification, registry, engine architecture) are archived in the branch
[`phases-1-9`](https://github.com/8919134556/Voice_engine/tree/phases-1-9).

## Project layout

```
Voice_engine/
├── dataset/speaker_01/*.wav        # YOUR recordings (private, git-ignored)
├── dataset_clean/                  # cleaned copies from Step 1 (private, git-ignored)
├── voice_engine/                   # the code: config.py, audio.py (Step 1), tts.py (Step 2)
├── scripts/step1_prepare_dataset.py, step2_speak.py
├── outputs/speech/                 # generated speech (private, git-ignored)
├── notebooks/voice_engine.ipynb    # the one notebook, for Google Colab
├── tests/
├── requirements.txt               # Step 1
└── requirements-tts.txt           # Step 2 (coqui-tts)
```

## Step 1 — prepare your recordings

Checks every WAV in `dataset/speaker_01/` (readable, not silent, not clipped, long enough) and saves a cleaned
copy to `dataset_clean/speaker_01/`: mono, 22050 Hz, silence trimmed at the start/end, equal loudness.
Originals are never changed.

**Google Colab (recommended):** open `notebooks/voice_engine.ipynb` from GitHub
(*File → Open notebook → GitHub → `8919134556/Voice_engine`*) and run the cells top to bottom.
Bring in your recordings from Google Drive or by uploading (Step 1a).

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/step1_prepare_dataset.py
python -m unittest discover -s tests -v
```

## Step 2 — text → speech in your voice

Uses **XTTS-v2**, a pretrained multilingual voice-cloning model (via the `coqui-tts` package). It already knows how
to speak English and Hindi; it imitates your voice from the cleaned Step 1 recordings (no training needed).

```
dataset_clean/speaker_01/*.wav ─► voice profile (computed once) ─┐
text + language ("en" / "hi") ───────────────────────────────────┴─► outputs/speech/<voice>_<lang>_<time>.wav (24 kHz)
```

**Google Colab (recommended, T4 GPU):** run the notebook's Step 2 cells — install, accept the license, load the
model + your voice, then type your text (English, or Hindi in Devanagari script).

```python
from voice_engine.tts import VoiceCloner
cloner = VoiceCloner()                 # downloads XTTS-v2 once (~1.8 GB)
cloner.load_voice("speaker_01")
cloner.speak("Hello! This is my voice.", language="en")
cloner.speak("नमस्ते! यह मेरी आवाज़ है।", language="hi")
```

Command line (after Step 1): `COQUI_TOS_AGREED=1 python scripts/step2_speak.py --text "Hello" --language en`

**License:** the XTTS-v2 model weights are under the **Coqui Public Model License (CPML) — non-commercial use only**
(https://coqui.ai/cpml). You accept it yourself (`I_ACCEPT_COQUI_CPML = True` in the notebook, which sets
`COQUI_TOS_AGREED=1`); the code refuses to load the model otherwise.

**Expectations:** with ~1.3 minutes of clean speech the result sounds *like* you, not identical; each run varies a
little. More clean recordings improve it. Your audio is read with `soundfile` (avoids PyTorch/torchcodec version
problems in Colab).

## Privacy

Your recordings are biometric data. `.gitignore` keeps `dataset/` (except its README), `dataset_clean/`,
`outputs/`, all audio files and model/voice files out of GitHub. Never run `git push` from Colab with your
audio in the project, and only add other people's voices with their consent.
