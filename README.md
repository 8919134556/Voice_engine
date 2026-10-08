# Voice Engine

A multi-speaker voice engine, built step by step with Python and PyTorch — as a learning project.

**Current phase:** Phase 5 — Voice Registry
**Current objective:** Manage multiple voice identities: each voice has a unique `voice_id`, metadata (name, optional gender, languages, description) and a reference to its embedding file, saved in `voice_profiles.json`.

## Roadmap

| Phase | Topic | Status |
|------:|-------|--------|
| 1 | Audio dataset + audio inspection | ✅ done |
| 2 | Waveform, spectrogram, audio features | ✅ done |
| 3 | Speaker embeddings | ✅ done (untrained demo) |
| 4 | Speaker similarity / verification | ✅ done (untrained demo) |
| 5 | Multiple voice profiles / Voice Registry | ✅ current |
| 6 | Multi-speaker voice engine | upcoming |
| 7 | Voice selection + speaker conditioning | upcoming |
| 8 | Production-ready architecture | upcoming |

## Phase 1 architecture — loading & inspection

```
WAV file
   ↓
Audio Loader        src/audio/loader.py    (soundfile → torch.Tensor [channels, samples])
   ↓
PyTorch Tensor
   ↓
Audio Inspection    src/audio/inspect.py   (sample rate, channels, duration, quality checks)
   ↓
Information about the recording            scripts/inspect_dataset.py prints the report
```

## Phase 2 architecture — audio representations

```
WAV file
   ↓  load_mono_16k()            src/audio/loader.py     (Phase 1 code: mono, 16 kHz)
Waveform        (1, samples)                              → outputs/phase2/<name>_waveform.png
   ↓  stft_power()               src/audio/features.py   (25 ms windows every 10 ms)
Spectrogram     (201, frames)                             → <name>_spectrogram.png
   ↓  mel_filterbank() @ ...     (80 triangular filters)
Mel spectrogram (80, frames)                              → <name>_mel.png
   ↓  extract_features()
duration, RMS, zero-crossing rate, spectral centroid      → phase2_audio_features.csv
```

## Phase 3 architecture — speaker embeddings

```
WAV
 ↓  load_mono_16k()                    Phase 1 code
16 kHz mono
 ↓  mel_spectrogram() + log + normalize Phase 2 code + src/embeddings/generate_embeddings.py
Mel spectrogram [1, 1, 80, time]
 ↓  SpeakerEncoder                      src/models/speaker_encoder.py (Conv2D-ReLU-Conv2D-ReLU-Pool-Linear)
128-D vector → L2-normalize            → embeddings/<speaker>/<file>.npy  (private, git-ignored)
 ↓  cosine_similarity()                 src/similarity/cosine.py
similarity between recordings
```

> The encoder is **untrained** (random weights). Its vectors are an architectural demonstration,
> **not** speaker recognition. Training on many speakers comes in a later phase.

## Phase 4 architecture — speaker verification

```
AUDIO A -> SpeakerEncoder -> Embedding A ─┐
                                          ├─> cosine similarity -> >= threshold ? -> MATCH / NOT MATCH
AUDIO B -> SpeakerEncoder -> Embedding B ─┘
            (Phase 3, same weights)            src/verification/verifier.py
```

> **DEMONSTRATION ONLY.** The encoder is untrained and the default threshold (0.70) is an educational
> example. With one speaker there are no impostor pairs, so false acceptance cannot be measured.

## Phase 5 architecture — voice registry

```
                 VOICE REGISTRY  (voice_profiles.json — metadata only)
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    voice_001      voice_002      voice_003        <- voice_id = the identity
        │              │              │
    metadata       metadata       metadata         <- name, gender (optional), languages, description
        │              │              │
  embeddings/     embeddings/    embeddings/       <- 128-D vectors in separate .npy files
  voice_001.npy   voice_002.npy  voice_003.npy
```

`src/voice_registry/profile.py` (VoiceProfile) · `src/voice_registry/registry.py` (VoiceRegistry:
register / get / list_voices / update / remove / exists / find_by_language / find_by_gender / save / load).
See `voice_profiles.example.json` for the file format.

## Project structure

```
Voice_engine/
├── dataset/                     # PRIVATE recordings (git-ignored) — see dataset/README.md
├── src/audio/
│   ├── loader.py                # find + load + preprocess (mono, 16 kHz) WAV files
│   ├── inspect.py               # measure a recording and check for problems
│   └── features.py              # STFT, Mel filter bank, RMS, ZCR, spectral centroid
├── src/models/speaker_encoder.py          # small PyTorch SpeakerEncoder (Phase 3)
├── src/embeddings/generate_embeddings.py  # WAV -> Mel -> encoder -> L2-normalized .npy
├── src/similarity/cosine.py               # cosine similarity
├── src/verification/verifier.py          # SpeakerVerifier: embed, compare, threshold (Phase 4)
├── src/voice_registry/                   # VoiceProfile + VoiceRegistry (Phase 5)
├── src/visualization/plots.py   # waveform / spectrogram / Mel / PCA plots (matplotlib)
├── scripts/inspect_dataset.py   # Phase 1 entry point
├── scripts/analyze_audio.py     # Phase 2 entry point
├── scripts/generate_speaker_embeddings.py   # Phase 3 entry point
├── scripts/verify_speaker.py    # Phase 4: compare two WAV files
├── scripts/evaluate_pairs.py    # Phase 4: score all genuine / impostor pairs
├── scripts/register_voice.py, list_voices.py, get_voice.py,
│   update_voice.py, remove_voice.py, demo_voice_registry.py   # Phase 5
├── tests/test_voice_registry.py # Phase 5 unit tests
├── voice_profiles.example.json  # example registry (fake data, committed)
├── voice_profiles.json          # your real registry (PRIVATE, git-ignored)
├── notebooks/01_audio_inspection.ipynb   # Colab: Phase 1
├── notebooks/02_audio_analysis.ipynb     # Colab: Phase 2
├── notebooks/03_speaker_embeddings.ipynb # Colab: Phase 3
├── notebooks/04_speaker_verification.ipynb # Colab: Phase 4
├── embeddings/                  # generated .npy embeddings (PRIVATE, git-ignored)
├── outputs/                     # generated plots + CSV (git-ignored)
├── requirements.txt
└── .gitignore
```

## Run locally (Windows, PowerShell)

```powershell
python --version                      # 3.10 or newer
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # cmd.exe: .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Put your recordings in `dataset/speaker_01/` (`audio_001.wav`, `audio_002.wav`, `audio_003.wav`), then:

```powershell
python scripts/inspect_dataset.py
python scripts/inspect_dataset.py --convert     # optional: write mono 16 kHz copies to dataset_16k/

# Phase 2
python scripts/analyze_audio.py                 # all files in dataset/speaker_01 -> outputs/phase2/
python scripts/analyze_audio.py --file dataset/speaker_01/audio_002.wav
python scripts/analyze_audio.py --show          # also open the plots in windows

# Phase 3
python scripts/generate_speaker_embeddings.py   # -> embeddings/speaker_01/*.npy, outputs/phase3/embeddings_pca.png
python scripts/generate_speaker_embeddings.py --seed 1 --frames 300

# Phase 4
python scripts/verify_speaker.py dataset/speaker_01/audio_001.wav dataset/speaker_01/audio_002.wav
python scripts/verify_speaker.py dataset/speaker_01/audio_002.wav dataset/speaker_01/audio_003.wav --threshold 0.9
python scripts/evaluate_pairs.py                # table + outputs/phase4/pair_scores.png

# Phase 5
python scripts/demo_voice_registry.py           # full demo with fake data in a temp folder
python scripts/register_voice.py --voice-id voice_001 --name "Arjun" --gender male --languages en hi --embedding embeddings/speaker_01/audio_001.npy
python scripts/list_voices.py                   # or: --language hi / --gender female
python scripts/get_voice.py --voice-id voice_001
python scripts/update_voice.py --voice-id voice_001 --name "Arjun Kumar"
python scripts/remove_voice.py --voice-id voice_001
python -m unittest discover -s tests -v         # run the tests
```

## Run on Google Colab

Open `notebooks/01_audio_inspection.ipynb` (Phase 1) , `02_audio_analysis.ipynb` (Phase 2), `03_speaker_embeddings.ipynb` (Phase 3) or `04_speaker_verification.ipynb` (Phase 4) in Colab
(File → Open notebook → GitHub → `8919134556/Voice_engine`) and run the cells top to bottom. No GPU is needed for Phases 1–5; Phases 3–4 use CUDA automatically if one is available. Phase 5 (registry) is pure Python.
Your recordings are read from private Google Drive (`MyDrive/voice_engine/dataset/`) or uploaded for the session.

## Data privacy

Voice recordings are personal data, and speaker embeddings are **derived biometric data**.
`*.wav`, `dataset/` contents, `embeddings/`, `*.npy`, `voice_profiles.json` and model weights (`*.pt`) are git-ignored,
so only code is pushed to GitHub. See [dataset/README.md](dataset/README.md).
