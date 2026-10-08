"""
Phase 5 — change some fields of an existing voice. Fields you don't pass are kept.

Usage:
    python scripts/update_voice.py --voice-id voice_001 --name "Arjun Kumar"
    python scripts/update_voice.py --voice-id voice_001 --languages en hi ta
    python scripts/update_voice.py --voice-id voice_001 --gender ""      # clear the gender metadata
    python scripts/update_voice.py --voice-id voice_001 --embedding embeddings/voice_001_v2.npy
"""

import argparse
import sys

from _registry_cli import add_registry_arg, print_profile

from src.voice_registry import VoiceRegistry, VoiceRegistryError


def main():
    parser = argparse.ArgumentParser(description="Update fields of a registered voice.")
    parser.add_argument("--voice-id", required=True)
    parser.add_argument("--name")
    parser.add_argument("--gender", help='Use --gender "" to clear it')
    parser.add_argument("--languages", nargs="+", help="REPLACES the whole language list")
    parser.add_argument("--description")
    parser.add_argument("--embedding", help="New embedding .npy file (validated)")
    add_registry_arg(parser)
    args = parser.parse_args()

    try:
        registry = VoiceRegistry(args.registry)

        # Only the options that were actually given (None = not supplied).
        changes = {
            "name": args.name,
            "gender": args.gender,
            "languages": args.languages,
            "description": args.description,
        }
        changes = {k: v for k, v in changes.items() if v is not None}
        if args.embedding is not None:
            changes["embedding_path"] = registry.to_stored_path(args.embedding)

        if not changes:
            print("[ERROR] Nothing to update. Pass at least one of "
                  "--name, --gender, --languages, --description, --embedding.")
            return 1

        profile = registry.update(args.voice_id, **changes)
        registry.save()
    except (VoiceRegistryError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        print("Nothing was changed.")
        return 1

    print(f"Updated {sorted(changes)} for '{profile.voice_id}'.\n")
    print_profile(profile, registry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
