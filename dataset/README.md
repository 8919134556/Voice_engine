# dataset/ — your private voice recordings

> **These WAV files are personal voice data. They are never committed to GitHub** (see `.gitignore`).
> With enough of someone's audio their voice can be cloned, so keep this folder private.

## Layout

```
dataset/
├── README.md          <- the only file here that goes to GitHub
└── speaker_01/        <- one folder per voice (your voice)
    ├── audio_001.wav
    ├── audio_002.wav
    └── ...
```

Later voices go in `speaker_02/`, `speaker_03/`, … (only with the person's consent).

## Good recordings for voice cloning

- Quiet room, same microphone, normal speaking voice.
- 3–15 seconds per clip; more clips = better (aim for 5–10+ minutes in total over time).
- Any sample rate / mono or stereo is fine — Step 1 converts everything.

## Where to keep them

- On your computer (this folder) — git-ignored.
- In your private Google Drive, for Colab (copy them in with the notebook's Step 1a).
