"""
interfaces.py — what the VoiceEngine NEEDS from a voice store, written down as a Protocol.

    VoiceEngine ──needs──► VoiceProvider (this interface)
                                ▲
                                │ satisfied by
                ┌───────────────┼──────────────────────────┐
         VoiceRegistry     (future) DatabaseVoiceProvider   (future) ApiVoiceProvider
         JSON + .npy        SQL table + blob storage        remote service

A Protocol is "duck typing with a name": any class that has these methods works,
no inheritance needed. Today the JSON VoiceRegistry (Phase 5) already fits, so
nothing had to change. Tomorrow a database-backed store only needs these five
methods and the VoiceEngine stays exactly the same.
"""

from typing import Protocol, runtime_checkable

import numpy as np

from src.voice_registry import VoiceProfile


@runtime_checkable
class VoiceProvider(Protocol):

    embedding_dim: int

    def list_voices(self) -> list[VoiceProfile]:
        """All profiles, sorted by voice_id."""
        ...

    def exists(self, voice_id: str) -> bool:
        ...

    def get(self, voice_id: str) -> VoiceProfile:
        """The profile; raise an error if the id is unknown."""
        ...

    def load_embedding(self, voice_id: str) -> np.ndarray:
        """The validated (embedding_dim,) vector; raise an error if it is missing or invalid."""
        ...
