"""Multi-Voice Engine Core (Phase 6): one engine, many voices, selected by voice_id."""

from src.voice_registry import InvalidEmbeddingError

from .exceptions import (
    ConfigurationError,
    VoiceEngineError,
    VoiceLanguageNotSupportedError,
    VoiceNotFoundError,
)
from .interfaces import VoiceProvider
from .request import SelectedVoice, VoiceRequest
from .voice_engine import VoiceEngine
