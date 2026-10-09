"""
STEP 1 — check and clean your voice recordings.

    dataset/speaker_01/*.wav  (your originals, never changed)
            │  check: readable? silent? clipped? long enough?
            │  clean: mono, 22050 Hz, trim start/end silence, equal loudness
            ▼
    dataset_clean/speaker_01/*.wav  (+ a report)

Usage (from the project folder):
    python scripts/step1_prepare_dataset.py
    python scripts/step1_prepare_dataset.py --voice speaker_01
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_engine import config  # noqa: E402
from voice_engine.audio import check_clip, save_wav  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 1: check and clean voice recordings.")
    parser.add_argument("--voice", default="speaker_01", help="Folder name inside dataset/ (default: speaker_01)")
    args = parser.parse_args()

    source = config.DATASET_DIR / args.voice
    target = config.CLEAN_DIR / args.voice
    files = sorted(source.glob("*.wav"))
    if not files:
        print(f"[ERROR] No WAV files found in {source}")
        print("Put your recordings in dataset/<voice name>/ (in Colab: upload them first).")
        return 1

    print(f"STEP 1 - checking {len(files)} recordings in {source}\n")
    print(f"{'file':<16}{'rate':>7}{'ch':>4}{'length':>9}{'speech':>9}{'peak':>7}{'loudness':>10}  status")
    print("-" * 78)
    good, total_speech = 0, 0.0
    for path in files:
        report, cleaned = check_clip(path)
        status = "OK" if report.ok else "; ".join(report.problems)
        print(f"{report.name:<16}{report.sample_rate:>7}{report.channels:>4}{report.duration_sec:>8.2f}s"
              f"{report.speech_sec:>8.2f}s{report.peak:>7.2f}{report.loudness_db:>8.1f}dB  {status}")
        if cleaned is not None and report.ok:
            save_wav(target / path.name, cleaned)
            good += 1
            total_speech += report.speech_sec

    print("-" * 78)
    print(f"Usable recordings: {good} of {len(files)}")
    print(f"Clean speech:      {total_speech:.1f} seconds ({total_speech / 60:.1f} minutes)")
    print(f"Cleaned copies:    {target}  (mono, {config.CLEAN_SAMPLE_RATE} Hz, equal loudness)")
    print("\nWhat this means for voice cloning:")
    if total_speech >= 30:
        print("  [OK] Enough clean speech to clone your voice (Step 2).")
    else:
        print("  [WARN] Less than 30 s of clean speech - cloning may sound less like you. Record a few more clips.")
    print("  More clean recordings (5-10+ minutes) will improve quality later.")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
