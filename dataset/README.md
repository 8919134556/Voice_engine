# dataset/ — private voice recordings

> **The WAV recordings in this folder contain personal voice data.**
> **Do NOT commit them to a public GitHub repository.**

A voice recording is biometric data: with enough of it, someone could clone your voice.
This repository holds the **code**; the **recordings** stay private.

## Expected layout

```
dataset/
├── README.md          <- this file (the ONLY thing in dataset/ that is committed)
├── speaker_01/
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── audio_003.wav
├── speaker_02/        <- added in later phases
└── ...
```

- One folder per speaker (`speaker_01`, `speaker_02`, ...).
- Ideal format: **WAV, mono, 16 kHz**, 3–15 seconds of clear speech per file.
  Other formats (stereo, 44.1/48 kHz) are fine — `scripts/inspect_dataset.py --convert`
  writes mono 16 kHz copies to `dataset_16k/`.

## Where to keep the recordings

- **Locally** in this folder (they are git-ignored automatically)
- **Private Google Drive** — e.g. `MyDrive/voice_engine/dataset/` (used by the Colab notebook)
- A **private** GitHub repository
- Any other private storage

If the dataset is somewhere else, point the script at it:

```
python scripts/inspect_dataset.py --dataset "D:\my_private_audio\dataset"
```

## How the protection works

`.gitignore` contains:

```
*.wav            # never commit audio files anywhere in the repo
dataset/*        # ignore everything in dataset/ ...
!dataset/README.md   # ... except this README
```

Check before every commit — no `.wav` file should appear here:

```
git status
```

## Intentionally committing a test file (later)

If you ever want to commit a small, **non-personal** test file (e.g. a synthetic beep for automated tests):

- One-off: force-add it — `git add -f tests/fixtures/beep.wav`
- Permanently: add an exception line to `.gitignore`, e.g. `!tests/fixtures/*.wav`

Never do this for real voice recordings in a public repository.
