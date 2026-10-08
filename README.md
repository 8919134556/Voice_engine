# Voice Engine

An educational, step-by-step **multi-voice engine** built with Python, NumPy and PyTorch.

> **This project currently does NOT synthesize speech.** It manages voice identities and prepares a
> speaker representation (*conditioning*) that a future voice model could use. The speaker encoder is
> **untrained**, so its embeddings demonstrate the architecture — they do not recognize real speakers.

## Purpose

Learn, phase by phase, how a multi-voice system is organized:

- how audio becomes data (waveform → spectrogram → Mel spectrogram),
- how a neural network turns speech into a **speaker embedding**,
- how embeddings are compared (**verification**),
- how many voices are managed by **one engine** through a **registry** keyed by `voice_id`,
- how the selected voice is prepared (**conditioning**) for a future speech model.

## Architecture

```
Audio
  ↓
Speaker Encoder          src/models/speaker_encoder.py      (untrained, 128-D output)
  ↓
Speaker Embedding        src/embeddings/                    (.npy files, private)
  ↓
Voice Registry           src/voice_registry/                (voice_profiles.json + .npy paths)
  ↓
Voice Engine             src/engine/                        (one engine, many voices)
  ↓
Voice Conditioning       src/conditioning/                  (validated, L2-normalized copy)
  ↓
Future Voice Model       (not implemented)
```

How the components work together at runtime:

```
                    Application
                         │
                         ▼
               VoiceEngine  (facade: list / select / resolve / condition / health_check / info)
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
   VoiceProvider (interface)   VoiceConditioner
     = VoiceRegistry today           │
              │                      ▼
              ▼               Speaker Embedding (lazy-loaded, cached in memory)
        VoiceProfile                 │
              └──────────┬───────────┘
                         ▼
                (Future Voice Model) ─► (Future Audio)
```

| Component | Responsibility | Location |
|---|---|---|
| `VoiceRegistry` | storage + metadata management | `src/voice_registry/` |
| `VoiceEngine` | orchestration: lookup, selection, cache, requests | `src/engine/` |
| `VoiceConditioner` | prepare the speaker representation | `src/conditioning/` |
| `SpeakerEncoder` | generate embeddings from audio | `src/models/`, `src/embeddings/` |
| Verification | compare speakers | `src/verification/`, `src/similarity/` |
| Settings | all important constants (sample rate, 128-D, paths, …) | `src/config/settings.py` |
| Logging | standard `logging`; logs voice ids, never audio or raw embeddings | `src/utils/logging.py` |

**One engine, many voices.** There is no `MaleVoiceEngine` or `HindiVoiceEngine`. One `VoiceEngine` manages
`voice_001 … voice_005 …`; gender and languages are only metadata on each `VoiceProfile`. The identity is the `voice_id`.

**Engine lifecycle:** create engine → load registry → validate configuration → select default voice →
load an embedding only when first needed (cache: first request disk → memory, later requests memory) →
condition the selected voice → return the representation.

**`VoiceProvider`** is a small Protocol (`list_voices`, `exists`, `get`, `load_embedding`, `embedding_dim`).
The JSON registry satisfies it today; a database- or API-backed store could replace it later without changing the engine.

```python
from src.engine import VoiceEngine, VoiceRequest

engine = VoiceEngine.from_registry_file("voice_profiles.json", default_voice_id="voice_001")
engine.list_voices()
engine.select_voice("voice_002")
conditioning = engine.condition(VoiceRequest(voice_id="voice_001", language="hi"))
engine.health_check()   # {'status': 'ok', 'voices': 2, 'valid_embeddings': 2, 'problems': {}}
engine.info()           # {'engine': 'VoiceEngine', 'embedding_dimension': 128, 'registered_voices': 2, ...}
```

## Phases completed

| Phase | Topic | Status |
|------:|-------|--------|
| 1 | Audio dataset + inspection | ✅ |
| 2 | Waveform, spectrogram, Mel spectrogram, audio features | ✅ |
| 3 | Speaker embeddings (untrained encoder) | ✅ demo |
| 4 | Speaker verification | ✅ demo |
| 5 | Voice Registry | ✅ |
| 6 | Multi-voice engine core | ✅ |
| 7 | Voice conditioning | ✅ |
| 8 | Production architecture (config, logging, interfaces, health, single notebook) | ✅ |

## Project structure

```
Voice_engine/
├── dataset/README.md            # your recordings go in dataset/<speaker>/ (PRIVATE, git-ignored)
├── src/
│   ├── config/settings.py       # SAMPLE_RATE, EMBEDDING_DIMENSION, DEFAULT_LANGUAGE, paths, ...
│   ├── utils/                   # logging.py, synthetic.py (fake audio/embeddings for tests & demos)
│   ├── audio/                   # loader.py, inspect.py, features.py
│   ├── visualization/plots.py   # waveform / spectrogram / Mel / PCA / pair-score plots
│   ├── models/speaker_encoder.py
│   ├── embeddings/generate_embeddings.py
│   ├── similarity/cosine.py
│   ├── verification/verifier.py
│   ├── voice_registry/          # profile.py (VoiceProfile), registry.py (VoiceRegistry)
│   ├── engine/                  # voice_engine.py, request.py, interfaces.py, exceptions.py
│   └── conditioning/            # conditioner.py, representation.py
├── scripts/                     # command-line tools and demos (see below)
├── tests/                       # unit tests — synthetic data only
├── notebooks/voice_engine.ipynb # the ONE educational notebook (all phases)
├── voice_profiles.example.json  # example registry with fake data
├── requirements.txt             # torch, numpy, soundfile, scipy, matplotlib
└── .gitignore
```

## Installation

**Windows (CPU)** — Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Google Colab:** open the notebook (below) — its first cell clones the repository and installs the requirements.
No GPU is required anywhere; PyTorch uses CUDA automatically when available, otherwise the CPU.

## How to run the tests

```powershell
python -m unittest discover -s tests -v
```

Covers audio utilities, embeddings/verification, the registry, the engine, conditioning and the Phase 8
architecture (settings, logging, provider interface, health check). Tests use temporary folders and synthetic
data — no personal recordings needed.

## How to run the demos

```powershell
python scripts/demo_architecture.py        # Phase 8: full pipeline on synthetic audio, with logging
python scripts/demo_voice_registry.py      # Phase 5
python scripts/demo_voice_engine.py        # Phase 6
python scripts/demo_conditioning.py        # Phase 7
```

Working with your own recordings (`dataset/speaker_01/*.wav`):

```powershell
python scripts/inspect_dataset.py                          # Phase 1: check the WAV files
python scripts/analyze_audio.py                            # Phase 2: plots + features CSV -> outputs/
python scripts/generate_speaker_embeddings.py              # Phase 3: -> embeddings/speaker_01/*.npy
python scripts/verify_speaker.py dataset/speaker_01/audio_001.wav dataset/speaker_01/audio_002.wav
python scripts/evaluate_pairs.py                           # Phase 4: all pairs, FAR/FRR
python scripts/register_voice.py --voice-id voice_001 --name "My Voice" --languages en hi `
    --embedding embeddings/speaker_01/audio_001.npy embeddings/speaker_01/audio_002.npy embeddings/speaker_01/audio_003.npy
python scripts/list_voices.py
python scripts/get_voice.py --voice-id voice_001
python scripts/update_voice.py --voice-id voice_001 --name "New Name"
python scripts/remove_voice.py --voice-id voice_001
python scripts/demo_voice_engine.py --registry voice_profiles.json
```

## How to open the notebook

- **Locally:** `pip install jupyter`, then `jupyter notebook notebooks/voice_engine.ipynb` (or open it in VS Code).
- **Colab:** File → Open notebook → GitHub → `8919134556/Voice_engine` → `notebooks/voice_engine.ipynb`.

Sections: Project Overview · Audio Inspection · Waveform and Spectrogram · Audio Features · Speaker Embedding ·
Speaker Verification · Voice Registry · Multi-Voice Engine · Voice Conditioning · Production Architecture ·
End-to-End Demo. If no recordings are found, the notebook generates **synthetic** audio, so every section runs.

## Privacy rules

Voice recordings are personal data, and speaker embeddings are **derived biometric data**.

- Never committed (see `.gitignore`): `*.wav`, `*.mp3`, `*.flac`, `*.m4a`, `*.ogg`, `*.npy`, `*.npz`,
  `embeddings/`, the contents of `dataset/`, `voice_profiles.json`, model weights (`*.pt`, `*.pth`), `outputs/`.
- Keep recordings locally or in private storage (e.g. private Google Drive). See `dataset/README.md`.
- Tests, demos and the notebook use synthetic data and temporary folders.
- Logs contain voice ids and shapes only — never audio samples or embedding values.
- Only register voices of people who have consented.

## Production Readiness

This is a learning project with a clean architecture — **it is not production-ready.**

| Area | Status |
|---|---|
| Architecture separation | ✅ |
| Voice registry | ✅ |
| Voice selection | ✅ |
| Embedding management | ✅ |
| Conditioning interface | ✅ |
| Testing | ✅ |
| Production speaker model (trained encoder) | ❌ |
| Large multi-speaker training | ❌ |
| Real-time inference | ❌ |
| TTS | ❌ |
| Voice cloning | ❌ |
| Authentication security | ❌ |
| Anti-spoofing | ❌ |
| Distributed deployment | ❌ |
