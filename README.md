# Voice Engine

An educational, step-by-step **multi-voice engine** built with Python, NumPy and PyTorch.

> **This project currently does NOT synthesize speech.** It manages voice identities and prepares a
> speaker representation (*conditioning*) that a future voice model could use. Phase 9 adds a **trainable**
> speaker encoder; it is **experimental** and **not a production authentication system**.

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
Speaker Encoder          src/training/model.py              (trainable, Phase 9; Phase 3's untrained one remains)
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
| `SpeakerEncoder` | generate embeddings from audio (untrained Phase 3 / trained Phase 9) | `src/models/`, `src/training/`, `src/embeddings/` |
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
| 9 | Real (trainable) speaker embedding model | ✅ pipeline (needs a multi-speaker dataset) |

## Phase 9 — Real Speaker Embedding Model

### Why the Phase 3 encoder was not enough
The Phase 3 `SpeakerEncoder` had **random weights**. It turned any sound into 128 numbers, but those numbers did
not describe the speaker — white noise scored ~0.97 cosine similarity against a voice, and every pair "matched".
A speaker embedding only becomes meaningful after **training**.

### Why multiple speakers are required
The model learns *what makes voices different* by being asked to tell speakers apart. With one speaker there is
nothing to contrast (a 1-class classifier is always right), so training needs **at least 2 speakers** — in practice
10+ speakers with 20+ recordings each for a useful experiment, and thousands for a strong model.
The training script refuses to run with fewer than 2 speakers.

### How the model learns (speaker classification)
```
                                            ┌──► classification head ─► speaker id ─► CrossEntropy  (training only)
Mel ─► Conv-BN-ReLU ─► Conv-BN-ReLU ─► Conv-ReLU ─► AdaptivePool ─► Linear ─► 128-D ─┤
                                            └──► L2-normalize ─► speaker embedding        (kept after training)
```
To classify the training speakers correctly, the encoder must put speaker-specific information (timbre, pitch
range, vocal-tract characteristics) into its 128 numbers. After training the head is discarded; the 128-D,
L2-normalized embedding is what verification, the registry and conditioning use. ~160k parameters (Colab-friendly).
Defaults: Adam, learning rate 1e-3, weight decay 1e-4, batch 32, 25 epochs, 2-second training crops — all in
`src/config/settings.py` and overridable on the command line. They are starting points, not tuned values.

### Dataset structure
```
dataset/                     # PRIVATE, git-ignored (or any folder: --dataset <path>)
├── speaker_001/  audio_001.wav, audio_002.wav, ...   -> class 0
├── speaker_002/  ...                                  -> class 1
└── ...                                                -> class 2, 3, ...
```
Folder name = speaker label (sorted alphabetically). Any number of speakers and recordings per speaker; any
sample rate / channel count (converted to 16 kHz mono).

**Split:** per speaker, *by recording*, ~70/15/15 train/val/test. Every speaker appears in training (a classifier
can only learn speakers it sees) and no WAV file is in two splits. Speakers with fewer than 3 files go entirely to
training (reported as a warning). Val/test therefore measure *new recordings of known speakers*; evaluating on
completely unseen speakers is future work.

**Preprocessing** is one shared code path for training, evaluation and embedding generation:
16 kHz mono → Mel spectrogram (n_fft 400, hop 160, 80 Mels) → log (dB) → per-band normalization.
Training uses random 2-second crops; evaluation and embedding generation use the whole recording.

### Inspect → train → evaluate → generate embeddings
```powershell
python scripts/inspect_speaker_dataset.py                    # speakers, files, rates, durations, problems, readiness
python scripts/train_speaker_encoder.py                      # train + checkpoints + plots + test evaluation
python scripts/train_speaker_encoder.py --epochs 30 --batch-size 64 --learning-rate 5e-4 --dataset <path> --output <path>
python scripts/train_speaker_encoder.py --synthetic          # pipeline test on FAKE speakers (no real data needed)
python scripts/evaluate_speaker_encoder.py                   # re-evaluate best_model.pt on the saved test split
python scripts/generate_speaker_embeddings.py --speaker speaker_001 --checkpoint outputs/checkpoints/best_model.pt
python scripts/verify_speaker.py A.wav B.wav --checkpoint outputs/checkpoints/best_model.pt --threshold <from evaluation>
python scripts/evaluate_pairs.py --checkpoint outputs/checkpoints/best_model.pt
```
Outputs (git-ignored): `outputs/checkpoints/best_model.pt` (lowest validation loss) and `last_model.pt`, each with
weights, optimizer state, epoch, metrics, config and speaker labels; and in `outputs/training/`: `history.json`,
`splits.json`, `evaluation.json`, `training_curves.png`, `similarity_distributions.png`, `embeddings_pca.png`.
`load_speaker_encoder()` restores a checkpoint and raises a clear error if it is missing — it never silently falls
back to random weights.

**Evaluation** reports test classification accuracy and, from the 128-D embeddings only, **genuine pairs**
(same speaker) vs **impostor pairs** (different speakers): counts, mean/min/max cosine similarity, FAR / FRR /
accuracy for thresholds 0.30–0.90, and the **EER**. No threshold is assumed — choose one from your own
evaluation (e.g. near the EER threshold). These numbers depend strongly on dataset size and test protocol.

### How the trained encoder connects to the rest
```
Audio ─► Mel Spectrogram ─► Speaker Encoder (trained) ─► 128-D Speaker Embedding (.npy)
      ─► Voice Registry (register_voice.py) ─► Voice Engine ─► Voice Conditioning
```
Trained embeddings have the same format as before (`(128,)`, unit length), so `VoiceProfile`, `VoiceRegistry`,
`VoiceEngine` and `VoiceConditioner` work unchanged. Phase 4 verification uses the trained encoder via
`--checkpoint` (or `create_verifier(checkpoint=...)`); without it, the untrained Phase 3 encoder is used as before.

### Training on Google Colab (recommended — GPU)
1. Open Colab → *Runtime → Change runtime type → GPU*.
2. Open `notebooks/voice_engine.ipynb` from GitHub (its first cell clones the repo and installs requirements), or run
   `!git clone https://github.com/8919134556/Voice_engine.git`, `%cd Voice_engine`, `!pip install -r requirements.txt`.
3. Put your dataset in **private** Google Drive, mount it (`from google.colab import drive; drive.mount(...)`) and
   pass its folder with `--dataset` — no path is hard-coded.
4. `!python scripts/inspect_speaker_dataset.py --dataset <drive-folder>`
5. `!python scripts/train_speaker_encoder.py --dataset <drive-folder> --output <drive-output-folder>`
6. `!python scripts/evaluate_speaker_encoder.py --checkpoint <drive-output-folder>/checkpoints/best_model.pt --splits <drive-output-folder>/training/splits.json`
7. `!python scripts/generate_speaker_embeddings.py --dataset <drive-folder> --speaker <name> --checkpoint <drive-output-folder>/checkpoints/best_model.pt`

Never upload personal audio, checkpoints or embeddings to GitHub.

### Limitations and future work
- **This is an experimental speaker embedding model and not a production-grade speaker recognition system.**
  128 dimensions do not by themselves mean good recognition; the quantity and diversity of data matter most.
- Small datasets give noisy, optimistic or overlapping results; few test pairs make FAR/FRR/EER unreliable.
- Better objectives (not implemented): triplet loss, contrastive loss, AAM-Softmax / ArcFace (angular margin).
- Production systems use stronger encoders such as **ECAPA-TDNN**, **x-vector** systems, **ResNet**-based speaker
  encoders, or modern self-supervised speech representations, trained on thousands of speakers. We deliberately
  build our own small model first to understand the pipeline.

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
│   ├── conditioning/            # conditioner.py, representation.py
│   └── training/                # Phase 9: dataset, preprocessing, model, losses, train, evaluate, utils
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

Covers audio utilities, embeddings/verification, the registry, the engine, conditioning, the Phase 8
architecture (settings, logging, provider interface, health check) and Phase 9 training (dataset discovery,
split, preprocessing, model, checkpoints, evaluation metrics, and a tiny end-to-end training run). Tests use temporary folders and synthetic
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
End-to-End Demo · Real Speaker Embedding Model (Phase 9 training). If no recordings are found, the notebook generates **synthetic** audio, so every section runs.

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
| Trainable speaker encoder + training/evaluation pipeline | ✅ (experimental) |
| Production speaker model (trained on many real speakers) | ❌ |
| Large multi-speaker training | ❌ |
| Real-time inference | ❌ |
| TTS | ❌ |
| Voice cloning | ❌ |
| Authentication security | ❌ |
| Anti-spoofing | ❌ |
| Distributed deployment | ❌ |
