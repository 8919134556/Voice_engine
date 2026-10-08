"""
verifier.py — speaker verification: "are these two recordings from the same speaker?"

             AUDIO A                         AUDIO B
                │                               │
     preprocess + Mel + SpeakerEncoder (Phase 3, same weights for both)
                │                               │
          Embedding A (128,)              Embedding B (128,)
                └──────────► cosine ◄───────────┘
                           similarity
                               │
                  similarity >= threshold ?
                     │                 │
                   MATCH           NOT MATCH

Everything heavy is reused from Phase 3; this file only adds the decision step.
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from src.config.settings import ENCODER_SEED, VERIFICATION_THRESHOLD
from src.embeddings.generate_embeddings import embed_file
from src.models.speaker_encoder import SpeakerEncoder, get_device
from src.similarity.cosine import cosine_similarity

# EDUCATIONAL EXAMPLE ONLY. A real threshold must be chosen from validation data
# (genuine + impostor pairs) for a specific trained encoder.
DEFAULT_THRESHOLD = VERIFICATION_THRESHOLD


@dataclass
class VerificationResult:
    similarity: float
    threshold: float

    @property
    def is_match(self):
        return self.similarity >= self.threshold

    @property
    def decision(self):
        return "MATCH" if self.is_match else "NOT MATCH"


class SpeakerVerifier:

    def __init__(self, encoder, device, trained=False, n_frames=None):
        """
        encoder:  a SpeakerEncoder (Phase 3)
        device:   torch.device("cpu") or torch.device("cuda")
        trained:  True only if real trained weights were loaded — used for warnings
        n_frames: optional crop/pad length for the Mel spectrogram (see Phase 3)
        """
        self.encoder = encoder.to(device).eval()
        self.device = device
        self.trained = trained
        self.n_frames = n_frames

    def generate_embedding(self, audio_path):
        """WAV -> 16 kHz mono -> Mel -> encoder -> L2-normalized (128,) numpy vector."""
        return embed_file(self.encoder, audio_path, self.device, self.n_frames)

    def compare(self, embedding_a, embedding_b):
        """Cosine similarity between two embeddings (-1 .. 1)."""
        return cosine_similarity(embedding_a, embedding_b)

    def verify(self, embedding_a, embedding_b, threshold=DEFAULT_THRESHOLD):
        """Compare, then apply the threshold -> VerificationResult (MATCH / NOT MATCH)."""
        return VerificationResult(self.compare(embedding_a, embedding_b), threshold)

    def verify_files(self, path_a, path_b, threshold=DEFAULT_THRESHOLD):
        """Convenience: two WAV paths -> VerificationResult."""
        return self.verify(self.generate_embedding(path_a), self.generate_embedding(path_b), threshold)


def create_verifier(checkpoint=None, seed=ENCODER_SEED, device=None, n_frames=None):
    """
    Build a SpeakerVerifier around the Phase 3 SpeakerEncoder.

    checkpoint=None -> UNTRAINED encoder with random weights. The seed makes the
                       weights identical to Phase 3's (seed 0), so results are repeatable.
    checkpoint=path -> load trained weights (.pt file) from a future training phase.
    """
    device = device or get_device()
    torch.manual_seed(seed)
    encoder = SpeakerEncoder()

    trained = False
    if checkpoint:
        state = torch.load(Path(checkpoint), map_location=device)
        encoder.load_state_dict(state)
        trained = True
    return SpeakerVerifier(encoder, device, trained=trained, n_frames=n_frames)


UNTRAINED_WARNING = "DEMONSTRATION ONLY - encoder is not trained."


def format_threshold(threshold):
    """0.7 -> '0.70', 0.9998 -> '0.9998' (never round a threshold away)."""
    return f"{threshold:.2f}" if threshold == round(threshold, 2) else f"{threshold:g}"
