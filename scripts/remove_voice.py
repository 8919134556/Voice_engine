"""
Phase 5 — remove a voice from the registry.

The embedding .npy file is NOT deleted: removing a registry entry should never
silently destroy data. Delete the file yourself if you want it gone.

Usage:
    python scripts/remove_voice.py --voice-id voice_001          # asks for confirmation
    python scripts/remove_voice.py --voice-id voice_001 --yes    # no question (for scripts)
"""

import argparse
import sys

from _registry_cli import add_registry_arg, print_profile

from src.voice_registry import VoiceRegistry, VoiceRegistryError


def confirm(question):
    try:
        return input(f"{question} [y/N]: ").strip().lower() in {"y", "yes"}
    except EOFError:  # no keyboard available (e.g. piped input) -> treat as "no"
        return False


def main():
    parser = argparse.ArgumentParser(description="Remove a voice from the registry.")
    parser.add_argument("--voice-id", required=True)
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation question")
    add_registry_arg(parser)
    args = parser.parse_args()

    try:
        registry = VoiceRegistry(args.registry)
        profile = registry.get(args.voice_id)
    except VoiceRegistryError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print_profile(profile, registry)
    if not args.yes and not confirm(f"\nRemove '{args.voice_id}' from the registry?"):
        print("Cancelled. Nothing was removed.")
        return 0

    registry.remove(args.voice_id)
    registry.save()
    print(f"\nRemoved '{args.voice_id}' from {registry.path}.")
    print(f"The embedding file was kept: {registry.resolve_embedding_path(profile.embedding_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
