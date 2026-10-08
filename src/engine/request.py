"""
request.py — what goes INTO the engine (VoiceRequest) and what comes OUT (SelectedVoice).

    VoiceRequest(voice_id="voice_001", language="hi")
            │
            ▼   engine.resolve(request)
    SelectedVoice(voice_id="voice_001", profile=<VoiceProfile>, embedding=<(128,) array>, language="hi")
"""

from dataclasses import dataclass

import numpy as np

from src.voice_registry import VoiceProfile


@dataclass(frozen=True)
class VoiceRequest:
    """
    An application's request: "I want this voice (optionally: for this language)".
    frozen=True -> a request can't be changed after it is created.
    """

    voice_id: str
    language: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.voice_id, str) or not self.voice_id.strip():
            raise ValueError("VoiceRequest.voice_id must be a non-empty string.")
        # frozen dataclass -> use object.__setattr__ to store the cleaned-up values once
        object.__setattr__(self, "voice_id", self.voice_id.strip())
        language = self.language.strip().lower() if isinstance(self.language, str) else None
        object.__setattr__(self, "language", language or None)  # "" -> None (no language check)


@dataclass(frozen=True)
class SelectedVoice:
    """
    Everything a future speech layer would need about the chosen voice.
    The embedding array is read-only (it is shared with the engine's cache).
    """

    voice_id: str
    profile: VoiceProfile
    embedding: np.ndarray
    language: str | None = None
