"""Voice Registry (Phase 5): manage multiple voice identities by voice_id."""

from .profile import VoiceProfile
from .registry import (
    DEFAULT_REGISTRY_FILE,
    EMBEDDING_DIM,
    DuplicateVoiceError,
    InvalidEmbeddingError,
    VoiceNotFoundError,
    VoiceRegistry,
    VoiceRegistryError,
    average_embeddings,
    validate_embedding,
)
