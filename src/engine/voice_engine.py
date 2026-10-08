"""
voice_engine.py — ONE engine that manages MANY voices.

    Application
         │  VoiceRequest(voice_id="voice_002", language="hi")
         ▼
    VoiceEngine ──────► VoiceRegistry (Phase 5: the only place voices are stored)
         │                    │
         │               VoiceProfile ──► embedding_path ──► .npy file
         │                                                       │
         │◄──────────── embedding cache (in memory) ◄────────────┘
         ▼
    SelectedVoice (voice_id, profile, embedding)

The engine does NOT generate audio. It only decides WHICH voice is active and
hands out that voice's profile + speaker embedding.
"""

import numpy as np

from src.voice_registry import VoiceProfile, VoiceRegistry

from .exceptions import VoiceLanguageNotSupportedError, VoiceNotFoundError
from .request import SelectedVoice, VoiceRequest


class VoiceEngine:

    def __init__(self, registry: VoiceRegistry, default_voice_id: str | None = None):
        """
        registry:         the Phase 5 VoiceRegistry. The engine keeps NO copy of the voices.
        default_voice_id: optional; must exist; becomes the initially selected voice.
        """
        self.registry = registry
        self._current_voice_id: str | None = None
        # voice_id -> (embedding_path it was loaded from, read-only embedding array)
        self._embedding_cache: dict[str, tuple[str, np.ndarray]] = {}

        if default_voice_id is not None:
            self.select_voice(default_voice_id)

    # ------------------------------------------------------------------
    # Looking up voices (delegated to the registry)
    # ------------------------------------------------------------------

    def list_voices(self) -> list[VoiceProfile]:
        """All registered voice profiles, sorted by voice_id."""
        return self.registry.list_voices()

    def get_voice(self, voice_id: str) -> VoiceProfile:
        """The VoiceProfile for `voice_id`; raises VoiceNotFoundError if it isn't registered."""
        if not self.registry.exists(voice_id):
            raise VoiceNotFoundError(voice_id, [p.voice_id for p in self.list_voices()])
        return self.registry.get(voice_id)

    # ------------------------------------------------------------------
    # Selecting / switching
    # ------------------------------------------------------------------

    def select_voice(self, voice_id: str) -> VoiceProfile:
        """
        Make `voice_id` the current voice. Switching = calling this again with
        another id: the engine stays the same object, only this one field changes.
        """
        profile = self.get_voice(voice_id)  # raises if unknown -> selection unchanged
        self._current_voice_id = voice_id
        return profile

    def get_current_voice(self) -> VoiceProfile | None:
        """
        The currently selected VoiceProfile, or None if no voice has been selected yet.
        (Returns the registry's latest version, so metadata updates are visible.)
        """
        if self._current_voice_id is None:
            return None
        return self.get_voice(self._current_voice_id)

    @property
    def current_voice_id(self) -> str | None:
        return self._current_voice_id

    # ------------------------------------------------------------------
    # Language check (metadata only — no detection, no translation)
    # ------------------------------------------------------------------

    def check_language(self, voice_id: str, language: str) -> VoiceProfile:
        """Raise VoiceLanguageNotSupportedError unless `language` is in the voice's languages."""
        profile = self.get_voice(voice_id)
        if not profile.supports_language(language):
            raise VoiceLanguageNotSupportedError(voice_id, language.strip().lower(), profile.languages)
        return profile

    # ------------------------------------------------------------------
    # Embeddings + cache
    # ------------------------------------------------------------------

    def get_embedding(self, voice_id: str) -> np.ndarray:
        """
        The voice's 128-D speaker embedding (made in earlier phases — never generated here).

        First call:  read the .npy file from disk, validate it, store it in the cache.
        Later calls: return the cached array (no disk access).
        If the voice's embedding_path was changed with registry.update(), the cache
        entry no longer matches and the new file is loaded.

        Validation (exists, loads, numeric, not empty, finite, shape (128,)) is done by
        the Phase 5 registry -> raises InvalidEmbeddingError on a bad file.
        """
        profile = self.get_voice(voice_id)

        cached = self._embedding_cache.get(voice_id)
        if cached is not None and cached[0] == profile.embedding_path:
            return cached[1]

        embedding = self.registry.load_embedding(voice_id)  # validated (128,) float32 array
        embedding.setflags(write=False)  # shared from the cache -> protect it from accidental edits
        self._embedding_cache[voice_id] = (profile.embedding_path, embedding)
        return embedding

    def is_cached(self, voice_id: str) -> bool:
        return voice_id in self._embedding_cache

    def clear_cache(self, voice_id: str | None = None) -> None:
        """Forget one cached embedding, or all of them (voice_id=None)."""
        if voice_id is None:
            self._embedding_cache.clear()
        else:
            self._embedding_cache.pop(voice_id, None)

    # ------------------------------------------------------------------
    # The main entry point for applications
    # ------------------------------------------------------------------

    def resolve(self, request: VoiceRequest) -> SelectedVoice:
        """
        Turn a VoiceRequest into a SelectedVoice:
          1. voice exists?           -> else VoiceNotFoundError
          2. get its VoiceProfile
          3. language supported?     -> else VoiceLanguageNotSupportedError (only if a language was given)
          4. load / reuse embedding  -> else InvalidEmbeddingError
          5. select the voice        (only after every check passed — a failed request
                                      leaves the current selection untouched)
          6. return SelectedVoice
        """
        profile = self.get_voice(request.voice_id)                       # 1 + 2
        if request.language is not None:
            self.check_language(request.voice_id, request.language)      # 3
        embedding = self.get_embedding(request.voice_id)                 # 4
        self.select_voice(request.voice_id)                              # 5
        return SelectedVoice(request.voice_id, profile, embedding, request.language)  # 6

    def __repr__(self) -> str:
        return (f"VoiceEngine(voices={len(self.registry)}, current={self._current_voice_id!r}, "
                f"cached={sorted(self._embedding_cache)})")
