"""
profile.py — VoiceProfile: everything we know about ONE voice.

    voice_id        "voice_001"          <- the identity (unique, never changes)
    name            "Arjun"              <- human-friendly label (can change)
    gender          "male" / None        <- optional METADATA only, never used for identity
    languages       ["en", "hi"]
    description     free text
    embedding_path  "embeddings/voice_001.npy"   <- reference to the 128-D vector file
    created_at      "2026-10-08T10:15:00+00:00"
    updated_at      None until the first update
    extra           {} free-form dict for future fields (accent, age range, style, ...)
"""

import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone

# Lowercase letters, digits, "_" and "-" only: safe in file names, URLs and JSON keys.
VOICE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def now_iso():
    """Current time as an ISO-8601 string in UTC, e.g. 2026-10-08T10:15:00+00:00."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class VoiceProfile:
    voice_id: str
    name: str
    embedding_path: str
    gender: str | None = None
    languages: list[str] = field(default_factory=list)
    description: str = ""
    created_at: str = field(default_factory=now_iso)
    updated_at: str | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate + normalize as soon as a profile is created, so a bad profile can never exist."""
        if not isinstance(self.voice_id, str) or not VOICE_ID_PATTERN.match(self.voice_id):
            raise ValueError(
                f"Invalid voice_id {self.voice_id!r}: use lowercase letters, digits, '_' or '-' "
                f"(e.g. 'voice_001')."
            )
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string.")
        self.name = self.name.strip()

        if not self.embedding_path or not str(self.embedding_path).strip():
            raise ValueError("embedding_path must not be empty.")
        self.embedding_path = str(self.embedding_path)

        # Gender is optional, free-form metadata. Empty or missing -> None.
        if isinstance(self.gender, str) and self.gender.strip():
            self.gender = self.gender.strip().lower()
        else:
            self.gender = None

        # Languages: lowercase codes, no duplicates, original order kept. "EN" -> "en".
        if isinstance(self.languages, str):
            self.languages = [self.languages]
        cleaned = []
        for lang in self.languages:
            code = str(lang).strip().lower()
            if code and code not in cleaned:
                cleaned.append(code)
        self.languages = cleaned

        self.description = (self.description or "").strip()
        self.extra = dict(self.extra or {})

    def supports_language(self, language):
        return language.strip().lower() in self.languages

    def to_dict(self):
        """Profile -> plain dict (for JSON). voice_id is included."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data, voice_id=None):
        """
        Plain dict (from JSON) -> VoiceProfile.
        Unknown keys are kept in `extra` instead of being lost, so a newer
        file can be read by older code without dropping information.
        """
        data = dict(data)
        if voice_id is not None:
            data["voice_id"] = voice_id
        known = {f.name for f in fields(cls)}
        extra = dict(data.pop("extra", {}) or {})
        for key in list(data):
            if key not in known:
                extra[key] = data.pop(key)
        return cls(**data, extra=extra)
