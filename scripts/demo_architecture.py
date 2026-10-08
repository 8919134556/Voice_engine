"""
Phase 8 — end-to-end architecture demo, with logging. No audio is generated; no GPU needed.

    synthetic audio ─► SpeakerEncoder ─► embeddings ─► VoiceProfile ─► VoiceRegistry
        ─► VoiceEngine (health, info, select, switch, cache) ─► VoiceConditioning

Everything happens in a temporary folder with SYNTHETIC audio (fake buzzing
"voices"), so no personal recordings are needed and your project is untouched.

Usage (run from the project root):
    python scripts/demo_architecture.py
    python scripts/demo_architecture.py --quiet     # hide the INFO log lines
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.embeddings.generate_embeddings import generate_for_speaker  # noqa: E402
from src.engine import VoiceEngine, VoiceLanguageNotSupportedError, VoiceRequest  # noqa: E402
from src.models.speaker_encoder import SpeakerEncoder, get_device  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.synthetic import write_synthetic_dataset  # noqa: E402
from src.voice_registry import VoiceProfile, VoiceRegistry, average_embeddings  # noqa: E402

# Two FAKE speakers (base pitch in Hz) and the voice metadata we give them.
SPEAKERS = {"speaker_01": 120.0, "speaker_02": 210.0}
VOICES = {
    "speaker_01": dict(voice_id="voice_001", name="Demo Voice A", gender="male", languages=["en", "hi"]),
    "speaker_02": dict(voice_id="voice_002", name="Demo Voice B", gender="female", languages=["en"]),
}


def banner(text: str) -> None:
    print(f"\n{'=' * 64}\n{text}\n{'=' * 64}")


def build_registry(work: Path) -> Path:
    """Audio -> SpeakerEncoder -> per-recording embeddings -> averaged voice embedding -> registry."""
    banner("1. Audio  (synthetic speakers, written as WAV files)")
    dataset = write_synthetic_dataset(work / "dataset", SPEAKERS, files_per_speaker=3)
    for speaker in SPEAKERS:
        print(f"  {speaker}: {len(list((dataset / speaker).glob('*.wav')))} WAV files")

    banner("2. SpeakerEncoder -> embeddings  (UNTRAINED, seed from settings)")
    torch.manual_seed(settings.ENCODER_SEED)
    device = get_device()
    encoder = SpeakerEncoder().to(device)
    print(f"  device: {device}, embedding dimension: {settings.EMBEDDING_DIMENSION}")

    banner("3. VoiceProfile + VoiceRegistry")
    registry_file = work / "voice_profiles.json"
    registry = VoiceRegistry(registry_file)
    for speaker, meta in VOICES.items():
        per_file = generate_for_speaker(encoder, dataset / speaker, work / "embeddings", device, verbose=False)
        voice_npy = work / "embeddings" / f"{meta['voice_id']}.npy"
        average_embeddings(sorted((work / "embeddings" / speaker).glob("*.npy")), voice_npy)
        registry.register(VoiceProfile(**meta, embedding_path=registry.to_stored_path(voice_npy)))
        print(f"  {meta['voice_id']} <- {speaker} ({len(per_file)} recordings averaged)")
    registry.save()
    print(f"  saved {len(registry)} voices to {registry_file.name}")
    return registry_file


def use_engine(registry_file: Path) -> None:
    banner("4. VoiceEngine lifecycle: create -> validate -> default voice (lazy loading)")
    engine = VoiceEngine.from_registry_file(registry_file, default_voice_id="voice_001")
    print(f"  info():         {engine.info()}")
    print(f"  health_check(): {engine.health_check()}")

    banner("5. Voice selection + embedding cache")
    engine.get_embedding("voice_001")          # miss -> disk
    engine.get_embedding("voice_001")          # hit  -> memory
    print(f"  current voice: {engine.current_voice_id}, cached: {engine.info()['cached_embeddings']}")

    banner("6. VoiceRequest -> resolve -> VoiceConditioning (via the engine facade)")
    conditioning = engine.condition(VoiceRequest(voice_id="voice_001", language="hi"))
    print(f"  Voice ID: {conditioning.voice_id}   dimension: {conditioning.dimension}   "
          f"normalized: {conditioning.normalized}   L2 norm: {conditioning.l2_norm:.4f}")

    banner("7. Switch voice (same engine object) + language check")
    engine_id = id(engine)
    other = engine.condition(VoiceRequest(voice_id="voice_002", language="en"))
    print(f"  now: {engine.current_voice_id}, conditioning dimension {other.dimension}, "
          f"same engine: {id(engine) == engine_id}")
    try:
        engine.condition(VoiceRequest(voice_id="voice_002", language="hi"))
    except VoiceLanguageNotSupportedError as exc:
        print(f"  REJECTED (as expected): {exc}")
    print(f"\n  final info(): {engine.info()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end architecture demo (synthetic data).")
    parser.add_argument("--quiet", action="store_true", help="Do not show INFO log lines")
    args = parser.parse_args()
    setup_logging("WARNING" if args.quiet else "INFO")

    work = Path(tempfile.mkdtemp(prefix="voice_engine_architecture_"))
    try:
        use_engine(build_registry(work))
        print("\nDone. The pipeline ends at VoiceConditioning - this project does NOT synthesize speech.")
        print("Note: the encoder is untrained, so the embeddings are architectural placeholders.")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
