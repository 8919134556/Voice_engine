"""
Shared helpers for the Phase 5 registry scripts (register / list / get / update / remove).
The leading underscore marks it as a helper, not a script you run directly.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.voice_registry import DEFAULT_REGISTRY_FILE  # noqa: E402

DEFAULT_REGISTRY_PATH = PROJECT_ROOT / DEFAULT_REGISTRY_FILE


def add_registry_arg(parser):
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH),
                        help=f"Registry JSON file (default: {DEFAULT_REGISTRY_FILE} in the project root)")


def print_profile(profile, registry=None, full=True):
    """Print one profile in a readable block."""
    print(profile.voice_id)
    print(f"  Name:           {profile.name}")
    print(f"  Gender:         {profile.gender or '-'}")
    print(f"  Languages:      {', '.join(profile.languages) or '-'}")
    if full:
        print(f"  Description:    {profile.description or '-'}")
        print(f"  Embedding path: {profile.embedding_path}")
        if registry is not None:
            resolved = registry.resolve_embedding_path(profile.embedding_path)
            print(f"  Embedding file: {'found' if resolved.is_file() else 'MISSING'} ({resolved})")
        print(f"  Created at:     {profile.created_at}")
        print(f"  Updated at:     {profile.updated_at or '-'}")
        if profile.extra:
            print(f"  Extra:          {profile.extra}")
