"""
representation.py — VoiceConditioning: the prepared speaker representation that a
FUTURE neural voice model would receive. It is data only; it produces no audio.

    VoiceProfile       = identity + metadata   (who: voice_id, name, languages, ...)
    Speaker embedding  = numbers on disk        (what the voice "sounds like", 128-D)
    VoiceConditioning  = validated, L2-normalized, in-memory copy, ready for a model
"""

from dataclasses import dataclass

import numpy as np

EXPECTED_DIM = 128


@dataclass(frozen=True)
class VoiceConditioning:
    voice_id: str
    embedding: np.ndarray        # shape (dimension,), float32, read-only
    dimension: int
    normalized: bool             # True when the L2 norm is ~1.0
    language: str | None = None  # carried over from a VoiceRequest, if one was used

    def __post_init__(self) -> None:
        if self.embedding.shape != (self.dimension,):
            raise ValueError(
                f"embedding shape {self.embedding.shape} does not match dimension {self.dimension}."
            )

    @property
    def l2_norm(self) -> float:
        return float(np.linalg.norm(self.embedding))
