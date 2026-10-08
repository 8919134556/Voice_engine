"""
model.py — the TRAINABLE speaker encoder (Phase 9) and its training-only classifier.

    Mel features [B, 1, 80, T]
        ↓ Conv2D(1→32)   + BatchNorm + ReLU     local time-frequency patterns
        ↓ Conv2D(32→64)  + BatchNorm + ReLU     stride 2: wider patterns, half the size
        ↓ Conv2D(64→128) + ReLU                 stride 2: higher-level patterns
        ↓ AdaptiveAvgPool → (4 frequency bands × 1)   averages over TIME -> any length works
        ↓ Flatten → Linear(512 → 128)
    128-D embedding  ──► encode(): L2-normalized (length 1)

Training only:

                    ┌──► classification head (Linear 128 → n_speakers) ─► speaker id ─► CrossEntropy
    audio ─► encoder┤
                    └──► 128-D embedding   (what we keep and use after training)

The classifier forces the encoder to put speaker-distinguishing information into
the 128 numbers. After training the head is thrown away; only the embedding is used,
so the encoder also works for speakers that were never in the training set.

BatchNorm (new compared with the Phase 3 encoder) keeps the activations in a stable
range during training, which makes training faster and less sensitive to the learning rate.
"""

import torch
from torch import nn

from src.config.settings import EMBEDDING_DIMENSION, N_MELS


class TrainableSpeakerEncoder(nn.Module):

    def __init__(self, embedding_dim: int = EMBEDDING_DIMENSION, n_mels: int = N_MELS):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_mels = n_mels
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 1))        # 4 coarse frequency bands, time averaged
        self.fc = nn.Linear(128 * 4, embedding_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """[B, 1, n_mels, T] -> raw (un-normalized) embedding [B, embedding_dim]."""
        x = self.features(mel)
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """[B, 1, n_mels, T] (or [1, n_mels, T] / [n_mels, T]) -> L2-normalized [B, embedding_dim]."""
        while mel.dim() < 4:
            mel = mel.unsqueeze(0)
        return nn.functional.normalize(self.forward(mel), dim=1)


class SpeakerClassifier(nn.Module):
    """Encoder + classification head. Used ONLY for training."""

    def __init__(self, encoder: TrainableSpeakerEncoder, n_speakers: int):
        super().__init__()
        if n_speakers < 2:
            raise ValueError("Speaker classification needs at least 2 speakers.")
        self.encoder = encoder
        self.head = nn.Linear(encoder.embedding_dim, n_speakers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """[B, 1, n_mels, T] -> logits [B, n_speakers]."""
        return self.head(self.encoder(mel))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
