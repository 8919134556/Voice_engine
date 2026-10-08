"""
Phase 4 — compare two WAV files: same speaker or not?

Usage (run from the project root):
    python scripts/verify_speaker.py dataset/speaker_01/audio_001.wav dataset/speaker_01/audio_002.wav
    python scripts/verify_speaker.py A.wav B.wav --threshold 0.9
    python scripts/verify_speaker.py A.wav B.wav --checkpoint models/encoder.pt   # future: trained weights
"""

import argparse
import sys
from pathlib import Path

# Make `import src...` work when this file is run directly as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.verification.verifier import (  # noqa: E402
    DEFAULT_THRESHOLD,
    UNTRAINED_WARNING,
    create_verifier,
    format_threshold,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4: speaker verification between two WAV files.")
    parser.add_argument("audio_a", help="First WAV file")
    parser.add_argument("audio_b", help="Second WAV file")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Decision threshold (default {DEFAULT_THRESHOLD}, an educational example)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the untrained weights (same as Phase 3)")
    parser.add_argument("--checkpoint", help="Path to trained encoder weights (.pt) — none exist yet")
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.audio_a, args.audio_b):
        if not Path(path).is_file():
            print(f"[ERROR] File not found: {path}")
            return 1

    verifier = create_verifier(checkpoint=args.checkpoint, seed=args.seed)

    try:
        emb_a = verifier.generate_embedding(args.audio_a)
        emb_b = verifier.generate_embedding(args.audio_b)
    except Exception as exc:
        print(f"[ERROR] Could not process audio: {exc}")
        return 1
    result = verifier.verify(emb_a, emb_b, args.threshold)

    print("Speaker Verification")
    print("--------------------")
    if not verifier.trained:
        print(f"*** {UNTRAINED_WARNING} ***\n")
    print(f"Device:              {verifier.device}")
    print(f"Audio A:             {Path(args.audio_a).name}")
    print(f"Audio B:             {Path(args.audio_b).name}")
    print(f"Embedding dimension: {emb_a.shape[0]}")
    print(f"Cosine similarity:   {result.similarity:.4f}")
    print(f"Threshold:           {format_threshold(result.threshold)}")
    print(f"Decision:            {result.decision}")
    if not verifier.trained:
        print(f"\n*** {UNTRAINED_WARNING} ***")
        print("The decision comes from random weights; it says nothing about who is speaking.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
