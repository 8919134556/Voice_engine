"""
voice_engine.py — ONE engine that manages MANY voices. The facade applications use.

    Application
         │  VoiceRequest(voice_id="voice_002", language="hi")
         ▼
    VoiceEngine ──────► VoiceProvider (today: the Phase 5 VoiceRegistry, JSON + .npy)
         │                    │
         │               VoiceProfile ──► embedding_path ──► .npy file
         │                                                       │
         │◄──────────── embedding cache (in memory) ◄────────────┘
         ▼
    SelectedVoice (voice_id, profile, embedding)  ──►  engine.condition(...)  ──►  VoiceConditioning

Lifecycle (Phase 8):
    create engine -> (registry already loaded) -> validate configuration -> select default voice
    -> load an embedding only when first needed (lazy) -> condition on request -> return representation

The engine does NOT generate audio, does NOT store voices itself, does NOT
compute embeddings and does NOT verify speakers. Those are separate components.
"""

from pathlib import Path

import numpy as np

from src.config.settings import EMBEDDING_DIMENSION, REGISTRY_FILE, validate_settings
from src.utils.logging import get_logger
from src.voice_registry import VoiceProfile, VoiceRegistry

from .exceptions import ConfigurationError, VoiceLanguageNotSupportedError, VoiceNotFoundError
from .interfaces import VoiceProvider
from .request import SelectedVoice, VoiceRequest

logger = get_logger(__name__)


class VoiceEngine:

    def __init__(self, registry: VoiceProvider, default_voice_id: str | None = None,
                 embedding_dim: int = EMBEDDING_DIMENSION):
        """
        registry:         where voices come from (any VoiceProvider; today a VoiceRegistry).
                          The engine keeps NO copy of the voices.
        default_voice_id: optional; must exist; becomes the initially selected voice.
        embedding_dim:    expected embedding size (from src/config/settings.py).

        Initialization is lightweight: no embedding is loaded here.
        """
        self.registry = registry
        self.embedding_dim = embedding_dim
        self._current_voice_id: str | None = None
        # voice_id -> (embedding_path it was loaded from, read-only embedding array)
        self._embedding_cache: dict[str, tuple[str, np.ndarray]] = {}
        self._conditioner = None  # created on first use (see condition())

        self._validate_configuration()
        logger.info("VoiceEngine created: %d voice(s) available", len(self.registry.list_voices()))

        if default_voice_id is not None:
            self.select_voice(default_voice_id)

    @classmethod
    def from_registry_file(cls, path: str | Path = REGISTRY_FILE,
                           default_voice_id: str | None = None) -> "VoiceEngine":
        """Lifecycle shortcut: load the JSON registry and build the engine in one call."""
        path = Path(path)
        if not path.is_file():
            raise ConfigurationError(f"Registry file not found: {path}. Register a voice first.")
        logger.info("Loading voice registry: %s", path.name)
        return cls(VoiceRegistry(path), default_voice_id=default_voice_id)

    def _validate_configuration(self) -> None:
        try:
            validate_settings()
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        if not isinstance(self.registry, VoiceProvider):
            raise ConfigurationError(
                f"{type(self.registry).__name__} is not a VoiceProvider "
                "(needs list_voices, exists, get, load_embedding, embedding_dim)."
            )
        provider_dim = self.registry.embedding_dim
        if provider_dim != self.embedding_dim:
            raise ConfigurationError(
                f"Embedding size mismatch: provider stores {provider_dim}-D embeddings, "
                f"engine expects {self.embedding_dim}-D (EMBEDDING_DIMENSION)."
            )

    # ------------------------------------------------------------------
    # Looking up voices (delegated to the provider)
    # ------------------------------------------------------------------

    def list_voices(self) -> list[VoiceProfile]:
        """All registered voice profiles, sorted by voice_id."""
        return self.registry.list_voices()

    def get_voice(self, voice_id: str) -> VoiceProfile:
        """The VoiceProfile for `voice_id`; raises VoiceNotFoundError if it isn't registered."""
        logger.debug("Voice lookup: %s", voice_id)
        if not self.registry.exists(voice_id):
            available = [p.voice_id for p in self.list_voices()]
            logger.warning("Voice not found: %s", voice_id)
            raise VoiceNotFoundError(voice_id, available)
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
        logger.info("Selected voice: %s", voice_id)
        return profile

    def get_current_voice(self) -> VoiceProfile | None:
        """
        The currently selected VoiceProfile, or None if no voice has been selected yet.
        (Returns the provider's latest version, so metadata updates are visible.)
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
            code = language.strip().lower()
            logger.warning("Language '%s' not supported by %s", code, voice_id)
            raise VoiceLanguageNotSupportedError(voice_id, code, profile.languages)
        return profile

    # ------------------------------------------------------------------
    # Embeddings + cache
    # ------------------------------------------------------------------

    def get_embedding(self, voice_id: str) -> np.ndarray:
        """
        The voice's speaker embedding (made in earlier phases — never generated here).

        First request:  voice_id -> disk (.npy) -> validate -> memory cache   ("cache miss")
        Later requests: voice_id -> memory cache                             ("cache hit")
        If the voice's embedding_path changed in the registry, the cache entry no
        longer matches and the new file is loaded.

        Validation (exists, loads, numeric, not empty, finite, right size) is done by
        the provider -> raises InvalidEmbeddingError on a bad file.
        """
        profile = self.get_voice(voice_id)

        cached = self._embedding_cache.get(voice_id)
        if cached is not None and cached[0] == profile.embedding_path:
            logger.info("Embedding cache hit: %s", voice_id)
            return cached[1]

        logger.info("Embedding cache miss: %s", voice_id)
        logger.info("Loading embedding: %s", voice_id)          # the id only, never the numbers
        embedding = self.registry.load_embedding(voice_id)
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
    # The main entry points for applications
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

    def condition(self, target: str | VoiceRequest | None = None):
        """
        Facade for Phase 7: get a VoiceConditioning without touching conditioning internals.
          condition()                  -> the currently selected voice
          condition("voice_001")       -> that voice (selection unchanged)
          condition(VoiceRequest(...)) -> resolve (checks + select), then condition
        The work is done by a VoiceConditioner (separate responsibility), created on first use.
        """
        if self._conditioner is None:
            # Imported here because src.conditioning itself imports src.engine.
            from src.conditioning import VoiceConditioner
            self._conditioner = VoiceConditioner(self, expected_dim=self.embedding_dim)
        if target is None:
            return self._conditioner.condition_current()
        if isinstance(target, VoiceRequest):
            return self._conditioner.condition_request(target)
        return self._conditioner.condition(target)

    # ------------------------------------------------------------------
    # Health + info (plain Python dicts; no web endpoint)
    # ------------------------------------------------------------------

    def health_check(self, check_embeddings: bool = True) -> dict:
        """
        Check that the engine can actually serve its voices:
          - the provider is reachable (list_voices works)
          - every profile is a VoiceProfile with a non-empty embedding_path
          - (optional) every embedding file loads, is finite and has the right size

        status: "ok"       everything usable
                "degraded" some voices have problems (listed in "problems")
                "error"    the provider itself failed
        The cache is not touched, so a health check never changes engine state.
        """
        try:
            profiles = self.registry.list_voices()
        except Exception as exc:  # report it instead of crashing: that's the point of a health check
            logger.error("Health check: provider not accessible: %s", exc)
            return {"status": "error", "voices": 0, "valid_embeddings": 0,
                    "problems": {"provider": f"{type(exc).__name__}: {exc}"}}

        problems: dict[str, str] = {}
        valid = 0
        for profile in profiles:
            if not isinstance(profile, VoiceProfile) or not profile.embedding_path:
                problems[getattr(profile, "voice_id", "?")] = "invalid profile structure"
                continue
            if not check_embeddings:
                continue
            try:
                embedding = self.registry.load_embedding(profile.voice_id)
                if embedding.shape != (self.embedding_dim,):
                    raise ValueError(f"shape {embedding.shape}, expected ({self.embedding_dim},)")
                valid += 1
            except Exception as exc:
                problems[profile.voice_id] = f"{type(exc).__name__}: {exc}"

        status = "ok" if not problems else "degraded"
        result = {"status": status, "voices": len(profiles),
                  "valid_embeddings": valid if check_embeddings else None, "problems": problems}
        log = logger.info if status == "ok" else logger.warning
        log("Health check: %s (%d voices, %d problem(s))", status, len(profiles), len(problems))
        return result

    def info(self) -> dict:
        """A small summary of the engine's state."""
        return {
            "engine": type(self).__name__,
            "provider": type(self.registry).__name__,
            "embedding_dimension": self.embedding_dim,
            "registered_voices": len(self.registry.list_voices()),
            "current_voice": self._current_voice_id,
            "cached_embeddings": sorted(self._embedding_cache),
        }

    def __repr__(self) -> str:
        return (f"VoiceEngine(voices={len(self.registry.list_voices())}, current={self._current_voice_id!r}, "
                f"cached={sorted(self._embedding_cache)})")
