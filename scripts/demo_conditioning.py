"""
Phase 7 — Voice Conditioning demo. Produces NO audio; no GPU needed.

By default it uses a temporary registry with FAKE random embeddings, so no
personal recordings are needed and nothing in your project is changed.

Usage (run from the project root):
    python scripts/demo_conditioning.py
    python scripts/demo_conditioning.py --registry voice_profiles.json   # your real registry (read-only)
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.conditioning import VoiceConditioner, VoiceConditioning  # noqa: E402
from src.engine import VoiceEngine, VoiceRequest  # noqa: E402
from src.voice_registry import VoiceProfile, VoiceRegistry  # noqa: E402

EXAMPLE_VOICES = [
    dict(voice_id="voice_001", name="Arjun", gender="male", languages=["en", "hi"]),
    dict(voice_id="voice_002", name="Priya", gender="female", languages=["en", "hi", "ta"]),
]


def build_demo_registry(folder: Path) -> VoiceRegistry:
    """Fake embeddings that are deliberately NOT normalized, so the effect is visible."""
    (folder / "embeddings").mkdir()
    registry = VoiceRegistry(folder / "voice_profiles.json")
    for seed, info in enumerate(EXAMPLE_VOICES):
        vector = (3.0 * np.random.default_rng(seed).normal(size=128)).astype(np.float32)
        np.save(folder / "embeddings" / f"{info['voice_id']}.npy", vector)
        registry.register(VoiceProfile(**info, embedding_path=f"embeddings/{info['voice_id']}.npy"))
    registry.save()
    return registry


def show(c: VoiceConditioning, stored: np.ndarray) -> None:
    print(f"Voice ID:            {c.voice_id}")
    print(f"Embedding dimension: {c.dimension}")
    print(f"Normalized:          {c.normalized}")
    print(f"L2 norm:             {c.l2_norm:.4f}   (stored embedding: {np.linalg.norm(stored):.4f})")
    print(f"Language:            {c.language or '-'}")
    print(f"First values:        {np.round(c.embedding[:4], 4)}")


def run_demo(registry: VoiceRegistry) -> int:
    print("1. VoiceRegistry loaded:", registry.path, f"({len(registry)} voices)")
    if len(registry) == 0:
        print("[ERROR] The registry is empty. Register voices first (scripts/register_voice.py).")
        return 1

    engine = VoiceEngine(registry)
    print("2. VoiceEngine created: ", engine)
    conditioner = VoiceConditioner(engine)
    print("3. VoiceConditioner created (uses the engine; never touches the registry directly)")

    print("\n4. Available voices:")
    voice_ids = [p.voice_id for p in engine.list_voices()]
    for p in engine.list_voices():
        print(f"   - {p.voice_id}  {p.name}  ({', '.join(p.languages) or '-'})")

    first = voice_ids[0]
    print(f"\n5. Select {first}")
    engine.select_voice(first)

    print("\n6-7. Conditioning for the selected voice")
    print("-" * 50)
    conditioning = conditioner.condition_current()
    show(conditioning, engine.get_embedding(first))
    print("The stored embedding was NOT changed; conditioning used a normalized copy.")

    if len(voice_ids) > 1:
        second = voice_ids[1]
        print(f"\nAnother voice via a VoiceRequest ({second}, language 'en')")
        print("-" * 50)
        other = conditioner.condition_request(VoiceRequest(voice_id=second, language="en"))
        show(other, engine.get_embedding(second))
        cosine = float(conditioning.embedding @ other.embedding)  # both length 1 -> dot = cosine
        print(f"\nCosine similarity {first} vs {second}: {cosine:.4f}  (independent representations)")
    else:
        print("\n(Only one voice registered - skipping the second voice.)")

    print("\nDone. VoiceConditioning is only a prepared representation - no speech was generated.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Voice conditioning demo.")
    parser.add_argument("--registry", help="Use an existing registry JSON (read-only) instead of fake voices")
    args = parser.parse_args()

    if args.registry:
        if not Path(args.registry).is_file():
            print(f"[ERROR] Registry file not found: {args.registry}")
            return 1
        return run_demo(VoiceRegistry(args.registry))

    work_dir = Path(tempfile.mkdtemp(prefix="voice_conditioning_demo_"))
    try:
        print(f"Using a temporary demo registry with FAKE embeddings: {work_dir}\n")
        return run_demo(build_demo_registry(work_dir))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
