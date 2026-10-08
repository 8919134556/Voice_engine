"""Voice conditioning (Phase 7): prepare a selected voice's embedding for a future voice model."""

from .conditioner import (
    ConditioningError,
    EmbeddingProjector,
    VoiceConditioner,
    l2_normalize,
    validate_embedding_array,
)
from .representation import EXPECTED_DIM, VoiceConditioning
