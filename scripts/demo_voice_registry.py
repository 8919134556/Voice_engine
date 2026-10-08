"""
Phase 5 — Voice Registry demo. Needs NO personal recordings and NO GPU.

Everything happens in a temporary folder with FAKE random embeddings, so your
real voice_profiles.json and embeddings/ are never touched.

Usage:
    python scripts/demo_voice_registry.py
    python scripts/demo_voice_registry.py --keep     # keep the temp folder to look at the JSON
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.voice_registry import (  # noqa: E402
    DuplicateVoiceError,
    InvalidEmbeddingError,
    VoiceProfile,
    VoiceRegistry,
)

# Example voices. Names are made up; embeddings are random test vectors, NOT real voices.
EXAMPLE_VOICES = [
    dict(voice_id="voice_001", name="Arjun", gender="male", languages=["en", "hi"],
         description="Indian English/Hindi male voice"),
    dict(voice_id="voice_002", name="Priya", gender="female", languages=["en", "hi", "ta"],
         description="English/Hindi/Tamil female voice"),
    dict(voice_id="voice_003", name="Narrator", gender=None, languages=["en"],
         description="Neutral English narration voice (no gender metadata)"),
]


def make_fake_embedding(path, seed, dim=128):
    """A random L2-normalized vector, standing in for a real speaker embedding."""
    vector = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    np.save(path, vector / np.linalg.norm(vector))


def step(n, title):
    print(f"\n{'=' * 60}\nStep {n}: {title}\n{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Voice Registry demo (fake data, temp folder).")
    parser.add_argument("--keep", action="store_true", help="Keep the temporary folder afterwards")
    args = parser.parse_args()

    work_dir = Path(tempfile.mkdtemp(prefix="voice_registry_demo_"))
    try:
        emb_dir = work_dir / "embeddings"
        emb_dir.mkdir()
        registry_file = work_dir / "voice_profiles.json"
        print(f"Demo folder (temporary): {work_dir}")

        step(1, "Create an empty registry")
        registry = VoiceRegistry(registry_file)
        print(f"Voices in registry: {len(registry)}")

        step(2, "Register example voices (fake embeddings)")
        for seed, info in enumerate(EXAMPLE_VOICES):
            make_fake_embedding(emb_dir / f"{info['voice_id']}.npy", seed)
            profile = VoiceProfile(**info, embedding_path=f"embeddings/{info['voice_id']}.npy")
            registry.register(profile)
            print(f"  registered {profile.voice_id} ({profile.name})")

        print("\n  Trying to register voice_001 again ...")
        try:
            registry.register(VoiceProfile(voice_id="voice_001", name="Someone else",
                                           embedding_path="embeddings/voice_001.npy"))
        except DuplicateVoiceError as exc:
            print(f"  REJECTED (as expected): {exc}")

        print("\n  Trying to register a voice with a broken embedding (64 numbers instead of 128) ...")
        np.save(emb_dir / "bad.npy", np.ones(64, dtype=np.float32))
        try:
            registry.register(VoiceProfile(voice_id="voice_bad", name="Bad",
                                           embedding_path="embeddings/bad.npy"))
        except InvalidEmbeddingError as exc:
            print(f"  REJECTED (as expected): {exc}")

        step(3, "List voices")
        for p in registry.list_voices():
            print(f"  {p.voice_id}  {p.name:<9} gender={p.gender or '-':<7} languages={', '.join(p.languages)}")

        step(4, "Retrieve one voice by voice_id")
        p = registry.get("voice_002")
        print(f"  {p.voice_id}: {p.name}, {p.languages}, embedding -> {p.embedding_path}")
        print(f"  embedding loaded: shape {registry.load_embedding('voice_002').shape}")

        step(5, "Update a voice (only the name changes)")
        before = registry.get("voice_001")
        after = registry.update("voice_001", name="Arjun Kumar")
        print(f"  name:      {before.name!r} -> {after.name!r}")
        print(f"  languages: {after.languages} (unchanged)")
        print(f"  updated_at: {after.updated_at}")

        step(6, "Search by language")
        for lang in ["hi", "ta", "en", "fr"]:
            ids = [p.voice_id for p in registry.find_by_language(lang)] or ["(none)"]
            print(f"  {lang}: {', '.join(ids)}")
        print(f"  gender metadata 'female': {[p.voice_id for p in registry.find_by_gender('female')]}")

        step(7, "Save registry to JSON")
        registry.save()
        print(f"  saved {len(registry)} voices to {registry_file.name}:\n")
        print("  " + registry_file.read_text(encoding="utf-8").rstrip().replace("\n", "\n  "))
        print("\n  Note: the JSON holds PATHS to the embeddings, not the 128 numbers themselves.")

        step(8, "Load the registry again (a brand-new object, as if the program restarted)")
        reloaded = VoiceRegistry(registry_file)
        print(f"  voices loaded: {len(reloaded)}")

        step(9, "Verify the voices still exist")
        all_ok = True
        for p in registry.list_voices():
            same = reloaded.exists(p.voice_id) and reloaded.get(p.voice_id) == p
            all_ok &= same
            print(f"  {p.voice_id}: {'OK - identical after reload' if same else 'MISMATCH'}")
        print(f"  voice_001 name after reload: {reloaded.get('voice_001').name!r}")
        assert json.loads(registry_file.read_text(encoding="utf-8")).keys() == {"voice_001", "voice_002", "voice_003"}
        print("\nDemo finished successfully." if all_ok else "\nDemo found a problem!")
        return 0 if all_ok else 1
    finally:
        if args.keep:
            print(f"\nKept demo folder: {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
