"""
Phase 6 — Multi-Voice Engine Core demo. No audio is generated; no GPU needed.

By default everything runs in a temporary folder with FAKE random embeddings,
so no personal recordings are needed and nothing in your project is changed.

Usage (run from the project root):
    python scripts/demo_voice_engine.py
    python scripts/demo_voice_engine.py --registry voice_profiles.json   # use your real registry (read-only)
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import (  # noqa: E402
    VoiceEngine,
    VoiceLanguageNotSupportedError,
    VoiceNotFoundError,
    VoiceRequest,
)
from src.voice_registry import VoiceProfile, VoiceRegistry  # noqa: E402

# Made-up example voices. Embeddings are random test vectors, NOT real voices.
EXAMPLE_VOICES = [
    dict(voice_id="voice_001", name="Arjun", gender="male", languages=["en", "hi"],
         description="Indian English/Hindi male voice"),
    dict(voice_id="voice_002", name="Priya", gender="female", languages=["en"],
         description="English female voice"),
    dict(voice_id="voice_003", name="Kavya", gender="female", languages=["en", "hi", "ta"],
         description="English/Hindi/Tamil female voice"),
]


def build_demo_registry(folder: Path) -> VoiceRegistry:
    """Create a registry with fake L2-normalized embeddings inside `folder`."""
    (folder / "embeddings").mkdir()
    registry = VoiceRegistry(folder / "voice_profiles.json")
    for seed, info in enumerate(EXAMPLE_VOICES):
        vector = np.random.default_rng(seed).normal(size=128).astype(np.float32)
        np.save(folder / "embeddings" / f"{info['voice_id']}.npy", vector / np.linalg.norm(vector))
        registry.register(VoiceProfile(**info, embedding_path=f"embeddings/{info['voice_id']}.npy"))
    registry.save()
    return registry


def step(n: int, title: str) -> None:
    print(f"\n{'=' * 60}\n{n}. {title}\n{'=' * 60}")


def run_demo(registry: VoiceRegistry) -> int:
    step(1, "Voice Registry loaded")
    print(f"Registry file: {registry.path}")
    print(f"Voices registered: {len(registry)}")
    if len(registry) == 0:
        print("[ERROR] The registry is empty. Register voices first (scripts/register_voice.py).")
        return 1

    step(2, "Create ONE VoiceEngine")
    engine = VoiceEngine(registry)
    engine_id = id(engine)
    print(engine)
    print(f"Current voice: {engine.get_current_voice()}  (nothing selected yet)")

    step(3, "Available voices")
    for p in engine.list_voices():
        print(f"- {p.voice_id}  {p.name:<8} languages={', '.join(p.languages) or '-':<10} gender={p.gender or '-'}")

    voice_ids = [p.voice_id for p in engine.list_voices()]
    first = voice_ids[0]

    step(4, f"Select {first}")
    engine.select_voice(first)
    print(f"Selected: {engine.current_voice_id}")

    step(5, "Current voice")
    current = engine.get_current_voice()
    print(f"{current.voice_id}: {current.name} ({current.description or 'no description'})")

    step(6, "Load its embedding (disk -> cache)")
    print(f"cached before: {engine.is_cached(first)}")
    embedding = engine.get_embedding(first)
    print(f"Embedding shape: {embedding.shape}, first values {np.round(embedding[:4], 3)}")
    print(f"cached after:  {engine.is_cached(first)}")
    again = engine.get_embedding(first)
    print(f"Second call returns the SAME cached array (no disk read): {again is embedding}")

    if len(voice_ids) > 1:
        second = voice_ids[1]
        step(7, f"Switch to {second}")
        print("Switching voice...")
        engine.select_voice(second)
        print(f"Selected: {engine.current_voice_id} ({engine.get_current_voice().name})")

        step(8, "Same engine?")
        print(f"Engine object before: {engine_id}   after: {id(engine)}   same: {id(engine) == engine_id}")
        print("Voice Engine remains the same. Only the selected voice changed.")
    else:
        print("\n(Only one voice registered - skipping the switch demo.)")

    step(9, "Resolve a VoiceRequest")
    # Ask for a Hindi-capable voice if there is one; otherwise just the first voice, no language.
    hindi_voices = engine.registry.find_by_language("hi")
    if hindi_voices:
        request = VoiceRequest(voice_id=hindi_voices[0].voice_id, language="hi")
    else:
        request = VoiceRequest(voice_id=first)
    selected = engine.resolve(request)
    print(f"Request:  {request}")
    print(f"Result:   SelectedVoice(voice_id={selected.voice_id!r}, name={selected.profile.name!r}, "
          f"embedding shape={selected.embedding.shape}, language={selected.language!r})")
    print(f"Current voice is now: {engine.current_voice_id}")

    step(10, "Language validation and unknown voices")
    english_only = next((p.voice_id for p in engine.list_voices() if not p.supports_language("hi")), None)
    before = engine.current_voice_id
    if english_only:
        try:
            engine.resolve(VoiceRequest(voice_id=english_only, language="hi"))
        except VoiceLanguageNotSupportedError as exc:
            print(f"REJECTED (as expected): {exc}")
    try:
        engine.resolve(VoiceRequest(voice_id="voice_999"))
    except VoiceNotFoundError as exc:
        print(f"REJECTED (as expected): {exc}")
    print(f"Failed requests did not change the selection: still {engine.current_voice_id} (was {before})")

    print(f"\n{engine}")
    print("\nDemo finished. No audio was generated - Phase 6 only manages voice identity and selection.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="VoiceEngine demo.")
    parser.add_argument("--registry", help="Use an existing registry JSON (read-only) instead of fake voices")
    args = parser.parse_args()

    if args.registry:
        if not Path(args.registry).is_file():
            print(f"[ERROR] Registry file not found: {args.registry}")
            return 1
        return run_demo(VoiceRegistry(args.registry))

    work_dir = Path(tempfile.mkdtemp(prefix="voice_engine_demo_"))
    try:
        print(f"Using a temporary demo registry with FAKE embeddings: {work_dir}")
        return run_demo(build_demo_registry(work_dir))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
