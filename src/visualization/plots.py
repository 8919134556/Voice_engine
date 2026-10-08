"""
plots.py — matplotlib figures for waveform, spectrogram and Mel spectrogram.

Every function returns the matplotlib Figure and, if `save_path` is given,
also saves it as a PNG. In a notebook the figure is shown automatically.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

from src.audio.features import HOP_LENGTH, mel_center_frequencies


def _finish(fig, save_path):
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_waveform(waveform, sample_rate, title="Waveform", save_path=None):
    """X = time (s), Y = amplitude (-1 .. 1)."""
    samples = waveform[0]
    time = torch.arange(len(samples)) / sample_rate

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(time.numpy(), samples.numpy(), linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, time[-1].item() if len(time) else 1)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="gray", linewidth=0.5)
    return _finish(fig, save_path)


def plot_spectrogram(spec_db, sample_rate, hop_length=HOP_LENGTH, title="Spectrogram", save_path=None):
    """X = time (s), Y = frequency (Hz, linear), colour = energy (dB)."""
    duration = spec_db.shape[1] * hop_length / sample_rate

    fig, ax = plt.subplots(figsize=(12, 4))
    img = ax.imshow(
        spec_db.numpy(), origin="lower", aspect="auto", cmap="magma",
        extent=[0, duration, 0, sample_rate / 2],
    )
    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=ax, label="Energy (dB)")
    return _finish(fig, save_path)


def plot_mel_spectrogram(mel_db, sample_rate, hop_length=HOP_LENGTH, title="Mel spectrogram", save_path=None):
    """
    X = time (s), Y = Mel band, colour = energy (dB).
    The Y tick labels show the Hz value of each band, so you can see that
    the bands are close together at low Hz and far apart at high Hz.
    """
    n_mels, frames = mel_db.shape
    duration = frames * hop_length / sample_rate

    fig, ax = plt.subplots(figsize=(12, 4))
    img = ax.imshow(
        mel_db.numpy(), origin="lower", aspect="auto", cmap="magma",
        extent=[0, duration, 0, n_mels],
    )
    centers = mel_center_frequencies(sample_rate, n_mels)
    ticks = list(range(0, n_mels, 10))
    ax.set_yticks([t + 0.5 for t in ticks])
    ax.set_yticklabels([f"{centers[t]:.0f}" for t in ticks])
    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mel band centre (Hz)")
    fig.colorbar(img, ax=ax, label="Energy (dB)")
    return _finish(fig, save_path)


def plot_mel_filterbank(fbank, sample_rate, title="Mel filter bank", save_path=None):
    """Draw every triangular Mel filter over the linear frequency axis."""
    freqs = torch.linspace(0, sample_rate / 2, fbank.shape[0])

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(freqs.numpy(), fbank.numpy(), linewidth=0.8)
    ax.set_title(f"{title} ({fbank.shape[1]} filters)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Weight")
    ax.set_xlim(0, sample_rate / 2)
    return _finish(fig, save_path)


def pca_2d(vectors):
    """
    Reduce N vectors of 128 numbers to N points of 2 numbers, keeping as much
    of their spread as possible (Principal Component Analysis via SVD).
    For VISUALIZATION ONLY — PCA is not part of the speaker encoder.
    """
    X = np.asarray(vectors, dtype=np.float64)
    X = X - X.mean(axis=0)                       # centre the cloud of points
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    points = X @ vt[:2].T                        # project onto the 2 main directions
    if points.shape[1] < 2:                      # only 1 recording -> pad to 2 columns
        points = np.pad(points, ((0, 0), (0, 2 - points.shape[1])))
    return points


def plot_embeddings_pca(embeddings, title="Embeddings (PCA to 2-D)", save_path=None):
    """embeddings: dict {name: (128,) vector}. One dot per recording."""
    names = list(embeddings)
    points = pca_2d([embeddings[n] for n in names])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(points[:, 0], points[:, 1], s=80)
    for name, (x, y) in zip(names, points):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6))
    ax.set_title(title)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.margins(0.3)
    return _finish(fig, save_path)


def plot_pair_scores(rows, threshold, title="Pair similarity scores", save_path=None):
    """
    rows: list of (name_a, name_b, "genuine" | "impostor", similarity)
    One horizontal bar per pair, coloured by pair type, with the threshold as a dashed line.
    """
    colors = {"genuine": "tab:blue", "impostor": "tab:orange"}
    labels = [f"{a} vs {b}" for a, b, _, _ in rows]
    scores = [s for _, _, _, s in rows]

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 2.2))
    ax.barh(labels, scores, color=[colors[k] for _, _, k, _ in rows])
    threshold_line = ax.axvline(threshold, color="red", linestyle="--", label=f"threshold = {threshold:g}")
    handles = [Patch(color=c, label=f"{kind} pair") for kind, c in colors.items()
               if any(k == kind for _, _, k, _ in rows)]
    ax.set_xlim(min(0.0, min(scores) - 0.05), 1.0)
    ax.set_xlabel("Cosine similarity")
    ax.invert_yaxis()
    ax.legend(handles=handles + [threshold_line], loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    ax.set_title(title)
    return _finish(fig, save_path)


# ---------------------------------------------------------------------------
# Phase 9: training + evaluation plots
# ---------------------------------------------------------------------------

def plot_training_history(history, save_path=None):
    """Two panels: train/val loss and train/val accuracy per epoch."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_loss"], marker="o", label="training loss")
    ax1.plot(epochs, history["val_loss"], marker="o", label="validation loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("cross-entropy loss"); ax1.set_title("Loss"); ax1.legend()
    ax2.plot(epochs, history["train_accuracy"], marker="o", label="training accuracy")
    ax2.plot(epochs, history["val_accuracy"], marker="o", label="validation accuracy")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.set_ylim(-0.02, 1.02)
    ax2.set_title("Speaker classification accuracy"); ax2.legend()
    return _finish(fig, save_path)


def plot_similarity_distributions(genuine, impostor, threshold_rows=None, eer_threshold=None,
                                  title="Genuine vs impostor cosine similarity", save_path=None):
    """Left: histograms of genuine/impostor scores. Right (optional): FAR and FRR vs threshold."""
    panels = 2 if threshold_rows else 1
    fig, axes = plt.subplots(1, panels, figsize=(6 * panels, 4))
    ax = axes[0] if panels == 2 else axes
    bins = np.linspace(-1, 1, 41)
    ax.hist(genuine, bins=bins, alpha=0.6, label=f"genuine pairs ({len(genuine)})")
    ax.hist(impostor, bins=bins, alpha=0.6, label=f"impostor pairs ({len(impostor)})")
    if eer_threshold is not None:
        ax.axvline(eer_threshold, color="red", linestyle="--", label=f"EER threshold {eer_threshold:.2f}")
    ax.set_xlabel("cosine similarity"); ax.set_ylabel("pairs"); ax.set_title(title); ax.legend()
    if threshold_rows:
        t = [r["threshold"] for r in threshold_rows]
        axes[1].plot(t, [r["far"] for r in threshold_rows], marker="o", label="FAR (false accepts)")
        axes[1].plot(t, [r["frr"] for r in threshold_rows], marker="o", label="FRR (false rejects)")
        axes[1].set_xlabel("threshold"); axes[1].set_ylabel("rate"); axes[1].set_ylim(-0.02, 1.02)
        axes[1].set_title("Error rates vs threshold"); axes[1].legend()
    return _finish(fig, save_path)


def plot_speaker_embeddings_pca(embeddings, labels, label_names=None,
                                title="Speaker embeddings (PCA to 2-D)", save_path=None):
    """One point per recording, coloured by speaker. Qualitative only."""
    points = pca_2d(embeddings)
    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        name = label_names[label] if label_names else f"speaker {label}"
        ax.scatter(points[mask, 0], points[mask, 1], s=50, label=name)
    ax.set_xlabel("PCA component 1"); ax.set_ylabel("PCA component 2"); ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    return _finish(fig, save_path)
