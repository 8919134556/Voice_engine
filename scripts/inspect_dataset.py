"""
Phase 1 — inspect every WAV recording in the dataset.

Usage (run from the project root):
    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --dataset <path-to-your-private-dataset>
    python scripts/inspect_dataset.py --convert      # also write mono 16 kHz copies to dataset_16k/

Exit code is 0 when everything loaded fine, 1 otherwise.
"""

import argparse
import sys
from pathlib import Path

# Make `import src...` work when this file is run directly as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.loader import (  # noqa: E402
    find_dataset_dir,
    list_audio_files,
    list_speakers,
    load_audio,
    preprocess_audio,
    save_audio,
)
from src.audio.inspect import format_report, inspect_audio  # noqa: E402

# The three recordings Phase 1 expects for the first speaker.
EXPECTED_SPEAKER = "speaker_01"
EXPECTED_FILES = ["audio_001.wav", "audio_002.wav", "audio_003.wav"]


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect WAV recordings in the Voice Engine dataset.")
    parser.add_argument("--dataset", help="Path to the dataset folder (default: ./dataset)")
    parser.add_argument("--convert", action="store_true",
                        help="Also save mono 16 kHz copies into dataset_16k/ next to the dataset")
    return parser.parse_args()


def convert_file(wav_path, dataset_dir, output_root):
    """Load -> mono + 16 kHz -> save, keeping the same speaker/file layout."""
    waveform, sr = load_audio(wav_path)
    waveform, sr = preprocess_audio(waveform, sr)
    out_path = output_root / wav_path.relative_to(dataset_dir)
    save_audio(out_path, waveform, sr)
    return out_path


def main():
    args = parse_args()

    # 1. Find the dataset folder.
    try:
        dataset_dir = find_dataset_dir(args.dataset)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        print("Create it and put your recordings in dataset/speaker_01/ (see dataset/README.md).")
        return 1
    print(f"Dataset: {dataset_dir}\n")

    # 2. Find the speaker folders.
    speakers = list_speakers(dataset_dir)
    if not speakers:
        print("[ERROR] No speaker folders found. Expected e.g. dataset/speaker_01/")
        return 1

    problems = 0
    total_files = 0
    total_seconds = 0.0
    output_root = dataset_dir.parent / "dataset_16k"

    for speaker_dir in speakers:
        wav_files = list_audio_files(speaker_dir)
        print("=" * 60)
        print(f"{speaker_dir.name}  ({len(wav_files)} WAV files)")
        print("=" * 60)

        # 3. Phase 1 check: are the three expected files present for speaker_01?
        if speaker_dir.name == EXPECTED_SPEAKER:
            found = {p.name for p in wav_files}
            for name in EXPECTED_FILES:
                if name not in found:
                    print(f"[ERROR] Missing expected file: {name}")
                    problems += 1

        # 4 + 5 + 6. Load each file, print its info, and check it.
        for wav_path in wav_files:
            info = inspect_audio(wav_path)
            print(format_report(info))
            print()

            total_files += 1
            total_seconds += info.duration_sec
            if not info.ok:
                problems += 1
                continue

            # 7. Optional preprocessing to mono 16 kHz.
            if args.convert:
                out_path = convert_file(wav_path, dataset_dir, output_root)
                print(f"  -> converted copy saved: {out_path}\n")

    if EXPECTED_SPEAKER not in {s.name for s in speakers}:
        print(f"[ERROR] Expected speaker folder not found: {EXPECTED_SPEAKER}")
        problems += 1

    # Summary
    print("=" * 60)
    print(f"Speakers: {len(speakers)}   Files: {total_files}   "
          f"Total audio: {total_seconds:.1f} s   Problems: {problems}")
    if problems == 0:
        print("All recordings loaded successfully.")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
