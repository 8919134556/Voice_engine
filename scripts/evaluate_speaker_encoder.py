"""
Phase 9 — evaluate a trained checkpoint again on the SAME test split used during training.

Usage (run from the project root):
    python scripts/evaluate_speaker_encoder.py
    python scripts/evaluate_speaker_encoder.py --checkpoint outputs/checkpoints/best_model.pt \
        --splits outputs/training/splits.json
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.training import (  # noqa: E402
    CheckpointNotFoundError,
    SpeakerClassifier,
    describe_device,
    get_device,
    load_checkpoint,
    load_speaker_encoder,
)
from src.training.evaluate import evaluate_encoder  # noqa: E402
from src.training.train import load_splits  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_speaker_encoder import LIMITATION, print_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained speaker encoder.")
    parser.add_argument("--checkpoint", default=str(settings.BEST_CHECKPOINT))
    parser.add_argument("--splits", default=str(settings.TRAINING_OUTPUT_DIR / "splits.json"))
    args = parser.parse_args()

    device = get_device()
    print(describe_device(device) + "\n")
    try:
        checkpoint = load_checkpoint(args.checkpoint, device)
        encoder = load_speaker_encoder(args.checkpoint, device)
    except CheckpointNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1
    if not Path(args.splits).is_file():
        print(f"[ERROR] Split file not found: {args.splits} (it is written by train_speaker_encoder.py)")
        return 1
    splits, speakers = load_splits(args.splits)
    missing = [str(r.path) for r in splits["test"] if not r.path.is_file()]
    if missing:
        print(f"[ERROR] {len(missing)} test file(s) are missing, e.g. {missing[0]}")
        return 1

    classifier = SpeakerClassifier(encoder, n_speakers=len(speakers)).to(device)
    classifier.head.load_state_dict(checkpoint["head_state_dict"])
    classifier.eval()
    print(f"Checkpoint: epoch {checkpoint['epoch']}, metric {checkpoint['metric']}\n")
    report = evaluate_encoder(classifier, splits["test"], checkpoint["config"]["segment_frames"], device)["report"]
    print_report(report)
    print(f"\n{LIMITATION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
