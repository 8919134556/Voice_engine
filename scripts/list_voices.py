"""
Phase 5 — list registered voices.

Usage:
    python scripts/list_voices.py
    python scripts/list_voices.py --language hi
    python scripts/list_voices.py --gender female
"""

import argparse
import sys

from _registry_cli import add_registry_arg, print_profile

from src.voice_registry import VoiceRegistry, VoiceRegistryError


def main():
    parser = argparse.ArgumentParser(description="List registered voices.")
    parser.add_argument("--language", help="Only voices that support this language code")
    parser.add_argument("--gender", help="Only voices with this gender metadata")
    add_registry_arg(parser)
    args = parser.parse_args()

    try:
        registry = VoiceRegistry(args.registry)
    except VoiceRegistryError as exc:
        print(f"[ERROR] {exc}")
        return 1

    voices = registry.find_by_language(args.language) if args.language else registry.list_voices()
    if args.gender:
        wanted = {p.voice_id for p in registry.find_by_gender(args.gender)}
        voices = [p for p in voices if p.voice_id in wanted]

    print("Available Voices")
    print("----------------")
    if not voices:
        print("(none)" if len(registry) == 0 else "(no voices match the filter)")
        if not registry.path.exists():
            print(f"No registry file yet at {registry.path}. Register a voice first.")
        return 0
    for profile in voices:
        print()
        print_profile(profile, full=False)
    print(f"\n{len(voices)} of {len(registry)} voice(s) shown.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
