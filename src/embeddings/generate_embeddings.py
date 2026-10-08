"""
generate_embeddings.py — WAV -> Mel spectrogram -> SpeakerEncoder -> 128-D embedding.

    WAV
     ↓  load_mono_16k()           Phase 1: mono, 16 kHz
     ↓  mel_spectrogram()         Phase 2: [80, time] power
     ↓  log + normalize           this file: make the numbers well-behaved for a network
     ↓  [1, 1, 80, time] tensor   add channel + batch dimensions
     ↓  SpeakerEncoder            src/models/speaker_encoder.py
     ↓  L2-normalize              length 1, so only the direction matters
    numpy array (128,)  ->  embeddings/<speaker>/<file>.npy
"""

from pathlib import Path

import numpy as np
import torch

from src.audio.features import mel_spectrogram, power_to_db
from src.audio.loader import list_audio_files, load_mono_16k


def normalize_mel(mel_db):
    """
    Per-band normalization: for each of the 80 Mel bands, subtract its average
    over time and divide by its spread.

    - Neural networks train best on inputs around 0 with spread ~1.
    - In the log (dB) domain, making the recording louder ADDS a constant to
      every value; subtracting the mean removes it. So the encoder sees the
      *shape* of your voice's spectrum, not how close you were to the mic.
    """
    mean = mel_db.mean(dim=1, keepdim=True)
    std = mel_db.std(dim=1, keepdim=True)
    return (mel_db - mean) / (std + 1e-5)


def pad_or_crop(mel, n_frames):
    """
    Force a Mel spectrogram [n_mels, time] to exactly `n_frames` frames:
    longer -> keep the middle part, shorter -> pad with zeros (= the average after normalization).

    Not needed for our encoder (adaptive pooling accepts any length), but needed
    later when many recordings are stacked into one training batch.
    """
    t = mel.shape[1]
    if t > n_frames:
        start = (t - n_frames) // 2
        return mel[:, start:start + n_frames]
    if t < n_frames:
        return torch.nn.functional.pad(mel, (0, n_frames - t))
    return mel


def waveform_to_encoder_input(waveform, sample_rate, n_frames=None):
    """Mono 16 kHz waveform [1, samples] -> normalized log-Mel tensor [1, 1, 80, time]."""
    mel_db = power_to_db(mel_spectrogram(waveform, sample_rate))   # [80, time]
    mel = normalize_mel(mel_db)
    if n_frames:
        mel = pad_or_crop(mel, n_frames)
    return mel.unsqueeze(0).unsqueeze(0)    # add batch + channel dims: [1, 1, 80, time]


def wav_to_encoder_input(path, n_frames=None):
    """WAV file -> normalized log-Mel tensor shaped [1, 1, 80, time] (batch, channel, mels, time)."""
    waveform, sr = load_mono_16k(path)
    return waveform_to_encoder_input(waveform, sr, n_frames)


def l2_normalize(vector):
    """Scale a vector to length 1: v / ||v||."""
    return vector / (vector.norm(dim=-1, keepdim=True) + 1e-12)


@torch.no_grad()  # inference only: no gradients needed -> faster, less memory
def embed_mel(model, mel, device):
    """[1, 1, 80, time] tensor -> L2-normalized numpy embedding of shape (128,)."""
    embedding = model(mel.to(device))          # [1, 128]
    embedding = l2_normalize(embedding)[0]     # (128,)
    return embedding.cpu().numpy()


def embed_file(model, path, device, n_frames=None):
    """One WAV file -> one (128,) embedding."""
    return embed_mel(model, wav_to_encoder_input(path, n_frames), device)


def generate_for_speaker(model, speaker_dir, out_dir, device, n_frames=None, verbose=True):
    """
    Embed every WAV in `speaker_dir` and save each as `out_dir/<speaker>/<file>.npy`.
    Returns a dict {filename: embedding}.
    """
    speaker_dir = Path(speaker_dir)
    save_dir = Path(out_dir) / speaker_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()  # evaluation mode (matters for layers like dropout/batchnorm in later models)
    embeddings = {}
    for wav_path in list_audio_files(speaker_dir):
        mel = wav_to_encoder_input(wav_path, n_frames)
        embedding = embed_mel(model, mel, device)
        save_path = save_dir / f"{wav_path.stem}.npy"
        np.save(save_path, embedding)
        embeddings[wav_path.name] = embedding

        if verbose:
            first = ", ".join(f"{v:.3f}" for v in embedding[:5])
            print(wav_path.name)
            print(f"  Mel input shape: {list(mel.shape)}  (batch, channel, mel bands, time frames)")
            print(f"  Embedding shape: {embedding.shape}")
            print(f"  First values:    [{first}, ...]")
            print(f"  L2 norm:         {np.linalg.norm(embedding):.3f}")
            print(f"  Saved:           {save_path}\n")
    return embeddings
