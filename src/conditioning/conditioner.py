"""
conditioner.py — turn a selected voice into a VoiceConditioning.

    VoiceConditioner
          │  asks for the embedding
          ▼
    VoiceEngine  ──►  VoiceRegistry  ──►  VoiceProfile  ──►  .npy on disk
          │
          ▼  (read-only cached array)
    validate  ──►  COPY + L2-normalize  ──►  projector (identity for now)  ──►  VoiceConditioning

The conditioner only talks to the VoiceEngine; it does not know how or where
the registry stores anything. It never writes to disk and never changes the
engine's cached array — it always works on a copy.
"""

import numpy as np

from src.engine import SelectedVoice, VoiceEngine, VoiceEngineError, VoiceRequest

from .representation import EXPECTED_DIM, VoiceConditioning

NORM_TOLERANCE = 1e-5


class ConditioningError(VoiceEngineError, ValueError):
    """The embedding handed to the conditioner is unusable (wrong type/shape, NaN, ...)."""


# ---------------------------------------------------------------------------
# Small pure functions — easy to test on their own
# ---------------------------------------------------------------------------

def validate_embedding_array(embedding: object, expected_dim: int = EXPECTED_DIM) -> np.ndarray:
    """
    Check an in-memory embedding. Raises ConditioningError; never "repairs" anything.
    Returns the same array (typed as np.ndarray) when it is valid.
    """
    if not isinstance(embedding, np.ndarray):
        raise ConditioningError(f"Embedding must be a numpy array, got {type(embedding).__name__}.")
    if embedding.size == 0:
        raise ConditioningError("Embedding is empty.")
    if not np.issubdtype(embedding.dtype, np.number):
        raise ConditioningError(f"Embedding must be numeric, got dtype {embedding.dtype}.")
    if embedding.shape != (expected_dim,):
        raise ConditioningError(f"Embedding has shape {embedding.shape}; expected ({expected_dim},).")
    if not np.all(np.isfinite(embedding)):
        raise ConditioningError("Embedding contains NaN or infinite values.")
    if not np.any(embedding):
        raise ConditioningError("Embedding is all zeros; it cannot be normalized.")
    return embedding


def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    """
    Return a NEW array: embedding / ||embedding||  (length 1, same direction).
    The input array is never modified.
    """
    vector = embedding.astype(np.float32, copy=True)   # copy=True: always a fresh array
    return vector / np.linalg.norm(vector)


# ---------------------------------------------------------------------------
# Placeholder for a future learned projection (NO weights, NO training)
# ---------------------------------------------------------------------------

class EmbeddingProjector:
    """
    Interface for a FUTURE layer that maps the 128-D speaker embedding into the
    size/space a specific voice model expects (e.g. 128 -> 256). A real one would
    be learned together with that model. This base version is the identity: it
    returns an unchanged copy, so Phase 7 adds no new neural network.
    """

    def project(self, embedding: np.ndarray) -> np.ndarray:
        return embedding.copy()


# ---------------------------------------------------------------------------
# The conditioner
# ---------------------------------------------------------------------------

class VoiceConditioner:

    def __init__(self, engine: VoiceEngine, expected_dim: int = EXPECTED_DIM,
                 projector: EmbeddingProjector | None = None):
        self.engine = engine
        self.expected_dim = expected_dim
        self.projector = projector or EmbeddingProjector()

    def condition(self, voice_id: str) -> VoiceConditioning:
        """
        Conditioning for any registered voice. Does NOT change the engine's selected voice.
        Unknown id -> VoiceNotFoundError (from the engine).
        """
        embedding = self.engine.get_embedding(voice_id)
        return self._prepare(voice_id, embedding)

    def condition_current(self) -> VoiceConditioning:
        """Conditioning for the engine's currently selected voice."""
        current = self.engine.get_current_voice()
        if current is None:
            raise ConditioningError("No voice is selected. Call engine.select_voice(...) first.")
        return self.condition(current.voice_id)

    def condition_selected(self, selected: SelectedVoice) -> VoiceConditioning:
        """Conditioning for a SelectedVoice that the engine already resolved."""
        return self._prepare(selected.voice_id, selected.embedding, selected.language)

    def condition_request(self, request: VoiceRequest) -> VoiceConditioning:
        """
        Full path for an application request:
        engine.resolve() (exists? language ok? embedding ok? -> selects the voice)
        then condition the result.
        """
        return self.condition_selected(self.engine.resolve(request))

    def _prepare(self, voice_id: str, embedding: np.ndarray,
                 language: str | None = None) -> VoiceConditioning:
        """validate -> copy + L2-normalize -> project -> freeze -> VoiceConditioning"""
        validate_embedding_array(embedding, self.expected_dim)
        vector = self.projector.project(l2_normalize(embedding))
        vector.setflags(write=False)   # the conditioning result is read-only too
        norm = float(np.linalg.norm(vector))
        return VoiceConditioning(
            voice_id=voice_id,
            embedding=vector,
            dimension=int(vector.shape[0]),
            normalized=abs(norm - 1.0) < NORM_TOLERANCE,
            language=language,
        )
