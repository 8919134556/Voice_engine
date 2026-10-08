"""
Phase 5 — register a new voice in the Voice Registry.

Usage (run from the project root):
    python scripts/register_voice.py --voice-id voice_001 --name "Arjun" --gender male \
        --languages en hi --description "Indian English/Hindi male voice" \
        --embedding embeddings/speaker_01/audio_001.npy

    # Several recording embeddings -> averaged into one profile vector embeddings/voice_001.npy
    python scripts/register_voice.py --voice-id voice_001 --name "Arjun" --languages en hi \
        --embedding embeddings/speaker_01/audio_001.npy embeddings/speaker_01/audio_002.npy embeddings/speaker_01/audio_003.npy
"""

import argparse
import sys
from pathlib import Path

from _registry_cli import PROJECT_ROOT, add_registry_arg, print_profile

from src.voice_registry import (
    EMBEDDING_DIM,
    VoiceProfile,
    VoiceRegistry,
    VoiceRegistryError,
    average_embeddings,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Register a new voice.")
    parser.add_argument("--voice-id", required=True, help="Unique id, e.g. voice_001")
    parser.add_argument("--name", required=True, help="Display name")
    parser.add_argument("--gender", help="Optional metadata (not used for identity)")
    parser.add_argument("--languages", nargs="+", default=[], help="Language codes, e.g. en hi ta")
    parser.add_argument("--description", default="", help="Free-text description")
    parser.add_argument("--embedding", nargs="+", required=True,
                        help="One .npy file, or several to average into embeddings/<voice-id>.npy")
    parser.add_argument("--dim", type=int, default=EMBEDDING_DIM, help="Expected embedding size")
    add_registry_arg(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        registry = VoiceRegistry(args.registry, embedding_dim=args.dim)

        # Check the id first, so we don't create an averaged file for a voice we will reject.
        if registry.exists(args.voice_id):
            print(f"[ERROR] Voice '{args.voice_id}' already exists. "
                  f"Use scripts/update_voice.py to change it.")
            return 1

        averaging = len(args.embedding) > 1
        if averaging:
            embedding_file = PROJECT_ROOT / "embeddings" / f"{args.voice_id}.npy"
        else:
            embedding_file = Path(args.embedding[0])

        profile = VoiceProfile(                                 # validates id, name, languages
            voice_id=args.voice_id,
            name=args.name,
            gender=args.gender,
            languages=args.languages,
            description=args.description,
            embedding_path=registry.to_stored_path(embedding_file),
        )

        if averaging:  # only after the profile is known to be valid
            average_embeddings(args.embedding, embedding_file, args.dim)
            print(f"Averaged {len(args.embedding)} embeddings -> {embedding_file}")

        registry.register(profile)                              # duplicate + embedding checks
        registry.save()
    except (VoiceRegistryError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        print("The voice was NOT registered.")
        return 1

    print(f"Registered voice '{profile.voice_id}' in {registry.path}\n")
    print_profile(profile, registry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
