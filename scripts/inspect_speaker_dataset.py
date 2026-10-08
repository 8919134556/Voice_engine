"""
Phase 9 — check a multi-speaker dataset BEFORE training.

Usage (run from the project root):
    python scripts/inspect_speaker_dataset.py
    python scripts/inspect_speaker_dataset.py --dataset <path-to-your-private-dataset>

Reports speakers, files per speaker, sample rates, channels, durations,
corrupt/silent files and too-short files, and whether training is possible.
"""

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.inspect import inspect_audio  # noqa: E402
from src.config import settings  # noqa: E402
from src.training.dataset import discover_recordings, split_recordings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a speaker dataset before training.")
    parser.add_argument("--dataset", default=str(settings.DATASET_DIR), help="Dataset folder (one sub-folder per speaker)")
    args = parser.parse_args()

    try:
        recordings, speakers = discover_recordings(args.dataset)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"Dataset: {Path(args.dataset).resolve()}\n")
    print(f"Speakers:    {len(speakers)}")
    print(f"Audio files: {len(recordings)}\n")
    if not recordings:
        print("[ERROR] No WAV files found. Expected dataset/<speaker>/<file>.wav")
        return 1

    per_speaker = Counter(r.speaker for r in recordings)
    for speaker, label in speakers.items():
        print(f"  {speaker:<20} label {label:<3} {per_speaker[speaker]:>4} files")

    rates, channels, durations = Counter(), Counter(), []
    broken, too_short, other_warnings = [], [], 0
    for rec in recordings:
        info = inspect_audio(rec.path)          # Phase 1 checks: corrupt, empty, silent, short, ...
        if not info.ok:
            broken.append(f"{rec.speaker}/{rec.path.name}: {'; '.join(info.errors)}")
            continue
        rates[info.sample_rate] += 1
        channels[info.channels] += 1
        durations.append(info.duration_sec)
        if info.duration_sec < settings.MIN_DURATION_SEC:
            too_short.append(f"{rec.speaker}/{rec.path.name} ({info.duration_sec:.2f}s)")

    print("\nSample rates: " + ", ".join(f"{sr} Hz ({n} files)" for sr, n in sorted(rates.items())))
    if any(sr != settings.SAMPLE_RATE for sr in rates):
        print(f"  -> will be resampled to {settings.SAMPLE_RATE} Hz automatically")
    print("Channels:     " + ", ".join(f"{c} ({n} files)" for c, n in sorted(channels.items())))
    if durations:
        print(f"Duration:     min {min(durations):.2f}s   mean {statistics.mean(durations):.2f}s   "
              f"max {max(durations):.2f}s   total {sum(durations) / 60:.1f} min")
    print(f"Corrupt/unusable files: {len(broken)}")
    for line in broken:
        print(f"  [ERROR] {line}")
    print(f"Too-short files (< {settings.MIN_DURATION_SEC}s): {len(too_short)}")
    for line in too_short:
        print(f"  [WARN]  {line}")

    splits, warnings = split_recordings(recordings)
    print(f"\nPlanned split (per speaker, by recording): train {len(splits['train'])}, "
          f"val {len(splits['val'])}, test {len(splits['test'])}")
    for w in warnings:
        print(f"  [WARN]  {w}")

    print("\nReadiness:")
    if len(speakers) < settings.MIN_SPEAKERS_FOR_TRAINING:
        print(f"  [NOT READY] {len(speakers)} speaker(s). Speaker classification needs at least "
              f"{settings.MIN_SPEAKERS_FOR_TRAINING}. Add more speaker folders (5-10+ recommended).")
        return 2
    if min(per_speaker.values()) < 10:
        print("  [WARN] Some speakers have fewer than 10 recordings; results will be very noisy.")
    if len(speakers) < 10:
        print(f"  [WARN] Only {len(speakers)} speakers: fine for learning the pipeline, far too few "
              "for a reliable speaker model.")
    print("  [OK] Training is possible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
