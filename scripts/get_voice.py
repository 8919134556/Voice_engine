"""
Phase 5 — show one voice.

Usage:
    python scripts/get_voice.py --voice-id voice_001
"""

import argparse
import sys

from _registry_cli import add_registry_arg, print_profile

from src.voice_registry import VoiceRegistry, VoiceRegistryError


def main():
    parser = argparse.ArgumentParser(description="Show one registered voice.")
    parser.add_argument("--voice-id", required=True)
    add_registry_arg(parser)
    args = parser.parse_args()

    try:
        registry = VoiceRegistry(args.registry)
        profile = registry.get(args.voice_id)
    except VoiceRegistryError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print_profile(profile, registry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
