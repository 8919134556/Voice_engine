"""
speaker_encoder.py — a small, educational speaker encoder (Phase 3).

    Mel spectrogram  [batch, 1, 80, time]
          ↓  Conv2D + ReLU          learn small time-frequency patterns
          ↓  Conv2D + ReLU          combine them into bigger patterns
          ↓  AdaptiveAvgPool2d      average over time -> fixed size, any length works
          ↓  Flatten
          ↓  Linear                 mix everything into 128 numbers
    embedding        [batch, 128]

IMPORTANT: a freshly created SpeakerEncoder has RANDOM weights. It always
produces a 128-number vector, but that vector does not describe *who* is
speaking until the network has been trained on many speakers (a later phase).
"""

import torch
from torch import nn

from src.config.settings import EMBEDDING_DIMENSION, N_MELS


class SpeakerEncoder(nn.Module):

    def __init__(self, embedding_dim=EMBEDDING_DIMENSION, n_mels=N_MELS):
        super().__init__()

        # Conv2D #1: slides 16 small 3x3 filters over the Mel "image".
        # Each filter can learn a local pattern (e.g. a rising pitch, an energy edge).
        # [B, 1, 80, T] -> [B, 16, 80, T]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)

        # Conv2D #2: 32 filters that look at the 16 patterns from conv1.
        # stride=2 halves frequency and time -> combines patterns over a wider area.
        # [B, 16, 80, T] -> [B, 32, 40, T/2]
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)

        # ReLU: keep positive values, set negatives to 0. Without a non-linearity,
        # stacked layers would collapse into one simple linear operation.
        self.relu = nn.ReLU()

        # Adaptive average pooling to (4, 1):
        #   frequency 40 -> 4 coarse bands (low ... high voice regions)
        #   time      T  -> 1  (average over the whole recording)
        # This is what lets a 2-second and an 8-second recording both give
        # the same output size. Speaker identity is a property of the whole
        # recording, not of one moment, so averaging over time makes sense.
        # [B, 32, 40, T/2] -> [B, 32, 4, 1]
        self.pool = nn.AdaptiveAvgPool2d((4, 1))

        # Flatten: [B, 32, 4, 1] -> [B, 128]
        self.flatten = nn.Flatten()

        # Linear: every output number is a weighted mix of all 128 pooled values.
        # [B, 32*4] -> [B, embedding_dim]
        self.fc = nn.Linear(32 * 4, embedding_dim)

    def forward(self, mel):
        """
        mel: [batch, 1, n_mels, time]  (time can be any length)
        returns: [batch, embedding_dim]  (NOT yet L2-normalized)
        """
        x = self.relu(self.conv1(mel))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)


def get_device():
    """Use the GPU (CUDA) when available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    """Number of trainable weights — useful to get a feel for model size."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
