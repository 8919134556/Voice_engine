"""
registry.py — VoiceRegistry: a collection of VoiceProfiles, saved as JSON.

                 VOICE REGISTRY  (voice_profiles.json)
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    voice_001      voice_002      voice_003        <- keys: voice_id
        │              │              │
   metadata        metadata       metadata         <- name, gender, languages, ...
        │              │              │
   embedding_path  embedding_path embedding_path   <- reference only
        ▼              ▼              ▼
  voice_001.npy   voice_002.npy  voice_003.npy     <- the 128-D vectors, stored separately

The JSON file holds small, human-readable metadata. The vectors live in .npy
files and are only *referenced* by path.
"""

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config.settings import EMBEDDING_DIMENSION, REGISTRY_FILE

from .profile import VoiceProfile, now_iso

EMBEDDING_DIM = EMBEDDING_DIMENSION
DEFAULT_REGISTRY_FILE = REGISTRY_FILE.name   # "voice_profiles.json"

# Readable field order inside voice_profiles.json.
JSON_FIELD_ORDER = ["name", "gender", "languages", "description", "embedding_path",
                    "created_at", "updated_at", "extra"]

# Fields update() is allowed to change. voice_id and created_at are permanent.
UPDATABLE_FIELDS = {"name", "gender", "languages", "description", "embedding_path", "extra"}


# ---------------------------------------------------------------------------
# Errors — one clear type per problem, all catchable as VoiceRegistryError
# ---------------------------------------------------------------------------

class VoiceRegistryError(Exception):
    """Base class for every registry error."""


class DuplicateVoiceError(VoiceRegistryError):
    pass


class VoiceNotFoundError(VoiceRegistryError):
    pass


class InvalidEmbeddingError(VoiceRegistryError):
    pass


# ---------------------------------------------------------------------------
# Embedding validation
# ---------------------------------------------------------------------------

def validate_embedding(path, expected_dim=EMBEDDING_DIM):
    """
    Check an embedding file and return it as a 1-D float array.
    Raises InvalidEmbeddingError with a clear message if anything is wrong.
    """
    path = Path(path)
    if not path.is_file():
        raise InvalidEmbeddingError(f"Embedding file not found: {path}")
    if path.suffix.lower() != ".npy":
        raise InvalidEmbeddingError(
            f"{path.name} is not a .npy embedding file. Create embeddings from WAV files with "
            f"scripts/generate_speaker_embeddings.py first."
        )
    try:
        # allow_pickle=False: never execute code hidden inside a .npy file
        array = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise InvalidEmbeddingError(f"Could not load {path} as a NumPy array: {exc}") from exc

    if not isinstance(array, np.ndarray):
        raise InvalidEmbeddingError(f"{path} does not contain a single NumPy array.")
    if array.size == 0:
        raise InvalidEmbeddingError(f"{path} is empty.")
    if not np.issubdtype(array.dtype, np.number):
        raise InvalidEmbeddingError(f"{path} is not numeric (dtype {array.dtype}).")

    array = array.squeeze()  # accept (1, 128) as well as (128,)
    if array.shape != (expected_dim,):
        raise InvalidEmbeddingError(
            f"{path} has shape {array.shape}; expected ({expected_dim},)."
        )
    if not np.all(np.isfinite(array)):
        raise InvalidEmbeddingError(f"{path} contains NaN or infinite values.")
    if not np.any(array):
        raise InvalidEmbeddingError(f"{path} is all zeros (not a usable embedding).")
    return array.astype(np.float32)


def average_embeddings(paths, out_path, expected_dim=EMBEDDING_DIM):
    """
    Phase 3 "speaker profile" idea: several recording embeddings -> one
    representative vector (mean, then L2-normalized). Saved to `out_path`.
    Refuses to overwrite an existing file.
    """
    out_path = Path(out_path)
    if out_path.exists():
        raise InvalidEmbeddingError(f"{out_path} already exists; refusing to overwrite it.")
    vectors = [validate_embedding(p, expected_dim) for p in paths]
    mean = np.mean(vectors, axis=0)
    mean = mean / np.linalg.norm(mean)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, mean.astype(np.float32))
    return out_path


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

class VoiceRegistry:

    def __init__(self, path=DEFAULT_REGISTRY_FILE, embedding_dim=EMBEDDING_DIM, autoload=True):
        """
        path:          the JSON file. Relative embedding paths are resolved from its folder.
        embedding_dim: expected embedding size (128 for our SpeakerEncoder).
        autoload:      read the file now if it exists.
        """
        self.path = Path(path)
        self.base_dir = self.path.resolve().parent
        self.embedding_dim = embedding_dim
        self._voices = {}  # voice_id -> VoiceProfile
        if autoload and self.path.exists():
            self.load()

    # -- paths ---------------------------------------------------------------

    def resolve_embedding_path(self, embedding_path):
        """Stored path -> real file path (relative paths are relative to the registry file)."""
        p = Path(embedding_path)
        return p if p.is_absolute() else self.base_dir / p

    def to_stored_path(self, path):
        """Real file path -> what we write in JSON: relative + forward slashes when possible."""
        p = Path(path).resolve()
        try:
            return p.relative_to(self.base_dir).as_posix()
        except ValueError:
            return str(p)  # outside the project: keep the absolute path

    # -- CRUD ----------------------------------------------------------------

    def register(self, profile):
        """
        Add a NEW voice. Steps:
          1. reject a duplicate voice_id (never overwrite silently -> use update())
          2. validate the embedding file
          3. store it in memory (call save() to write to disk)
        """
        if not isinstance(profile, VoiceProfile):
            raise TypeError("register() expects a VoiceProfile.")
        if profile.voice_id in self._voices:
            raise DuplicateVoiceError(
                f"Voice '{profile.voice_id}' is already registered. Use update() to change it."
            )
        validate_embedding(self.resolve_embedding_path(profile.embedding_path), self.embedding_dim)
        self._voices[profile.voice_id] = profile
        return profile

    def get(self, voice_id):
        """Look up one voice by its id (a dictionary lookup — instant even for 10,000 voices)."""
        try:
            return self._voices[voice_id]
        except KeyError:
            raise VoiceNotFoundError(f"Voice '{voice_id}' is not registered.") from None

    def exists(self, voice_id):
        return voice_id in self._voices

    def list_voices(self):
        """All profiles, sorted by voice_id."""
        return [self._voices[v] for v in sorted(self._voices)]

    def update(self, voice_id, /, **changes):
        """
        Change only the fields you pass, e.g. update("voice_001", name="Arjun Kumar").
        (The "/" makes voice_id positional-only, so update("voice_001", voice_id="x")
        reaches our check below and gets a clear error instead of a Python TypeError.)
        Everything else is kept. The new version is fully validated BEFORE it
        replaces the old one, so a failed update changes nothing.
        """
        current = self.get(voice_id)
        unknown = set(changes) - UPDATABLE_FIELDS
        if unknown:
            raise VoiceRegistryError(
                f"Cannot update {sorted(unknown)}. Allowed fields: {sorted(UPDATABLE_FIELDS)}."
            )
        if not changes:
            raise VoiceRegistryError("Nothing to update: no fields given.")

        updated = replace(current, **changes, updated_at=now_iso())  # replace() re-runs validation
        if "embedding_path" in changes:
            validate_embedding(self.resolve_embedding_path(updated.embedding_path), self.embedding_dim)
        self._voices[voice_id] = updated
        return updated

    def remove(self, voice_id):
        """
        Remove a voice from the registry and return its profile.
        The embedding FILE is NOT deleted — it may be shared, backed up or
        needed again. Delete it yourself if you really want it gone.
        """
        profile = self.get(voice_id)
        del self._voices[voice_id]
        return profile

    # -- search --------------------------------------------------------------

    def find_by_language(self, language):
        """Every voice whose languages list contains `language` (case-insensitive)."""
        return [p for p in self.list_voices() if p.supports_language(language)]

    def find_by_gender(self, gender):
        """Filter by the optional gender METADATA. Never use this as an identity check."""
        g = gender.strip().lower()
        return [p for p in self.list_voices() if p.gender == g]

    # -- embeddings ----------------------------------------------------------

    def load_embedding(self, voice_id):
        """Load (and validate) the 128-D vector for a voice — what a future engine will need."""
        return validate_embedding(
            self.resolve_embedding_path(self.get(voice_id).embedding_path), self.embedding_dim
        )

    # -- persistence ---------------------------------------------------------

    def save(self):
        """
        Write all profiles to JSON:  { "voice_001": {name, gender, ...}, ... }
        Written to a temporary file first, then swapped in, so a crash
        mid-write can never leave a half-written (corrupted) registry.
        """
        data = {}
        for profile in self.list_voices():
            entry = profile.to_dict()
            del entry["voice_id"]  # it is already the key
            data[profile.voice_id] = {key: entry[key] for key in JSON_FIELD_ORDER}

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, self.path)
        return self.path

    def load(self):
        """
        Replace the in-memory registry with the contents of the JSON file.
        Embedding files are NOT checked here (they may live on another machine);
        load_embedding() checks them when they are actually used.
        """
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise VoiceRegistryError(f"{self.path} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise VoiceRegistryError(f"{self.path} must contain a JSON object {{voice_id: profile}}.")

        voices = {}
        for voice_id, entry in data.items():
            try:
                voices[voice_id] = VoiceProfile.from_dict(entry, voice_id=voice_id)
            except (TypeError, ValueError) as exc:
                raise VoiceRegistryError(f"Bad entry '{voice_id}' in {self.path}: {exc}") from exc
        self._voices = voices
        return self

    def __len__(self):
        return len(self._voices)

    def __contains__(self, voice_id):
        return voice_id in self._voices
