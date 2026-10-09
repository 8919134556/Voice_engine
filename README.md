# Voice Engine

Goal: **type text → hear it spoken in my own voice** (English + Hindi), built step by step.

```
your recordings ─► Step 1: clean ─► Step 2: voice cloning TTS ─► Step 3: engine.speak(text) ─► WAV in your voice
```

| Step | What | Status |
|---|---|---|
| 1 | Check and clean the voice recordings | ✅ current |
| 2 | Text → speech in your voice (pretrained voice-cloning model, English + Hindi, Colab GPU) | next |
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
├── voice_engine/                   # the code: config.py, audio.py
├── scripts/step1_prepare_dataset.py
├── notebooks/voice_engine.ipynb    # the one notebook, for Google Colab
├── tests/
└── requirements.txt
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

## Privacy

Your recordings are biometric data. `.gitignore` keeps `dataset/` (except its README), `dataset_clean/`,
`outputs/`, all audio files and model/voice files out of GitHub. Never run `git push` from Colab with your
audio in the project, and only add other people's voices with their consent.
