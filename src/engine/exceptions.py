"""
exceptions.py — the errors the VoiceEngine can raise.

All of them inherit from VoiceEngineError, so an application can catch
"anything the engine complained about" with a single `except VoiceEngineError`.
"""

from src.voice_registry import VoiceNotFoundError as RegistryVoiceNotFoundError


class VoiceEngineError(Exception):
    """Base class for every VoiceEngine error."""


class VoiceNotFoundError(VoiceEngineError, RegistryVoiceNotFoundError):
    """
    The application asked for a voice_id that is not registered.

    It also inherits from the Phase 5 registry's VoiceNotFoundError, so code that
    already catches the registry error keeps working.
    """

    def __init__(self, voice_id: str, available: list[str] | None = None):
        self.voice_id = voice_id
        message = f"Voice '{voice_id}' is not registered."
        if available is not None:
            message += f" Available voices: {', '.join(available) or '(none)'}."
        super().__init__(message)


class ConfigurationError(VoiceEngineError):
    """The engine was set up inconsistently, e.g. a provider with 256-D embeddings
    while the project is configured for 128-D."""


class VoiceLanguageNotSupportedError(VoiceEngineError):
    """The requested language is not in the voice's `languages` metadata."""

    def __init__(self, voice_id: str, language: str, supported: list[str]):
        self.voice_id = voice_id
        self.language = language
        self.supported = supported
        super().__init__(
            f"Voice '{voice_id}' does not support language '{language}'. "
            f"Supported: {', '.join(supported) or '(none listed)'}."
        )
