"""
Phase 2 — waveform, spectrogram, Mel spectrogram and basic audio features.

Usage (run from the project root):
    python scripts/analyze_audio.py                                   # all files in dataset/speaker_01
    python scripts/analyze_audio.py --file dataset/speaker_01/audio_002.wav
    python scripts/analyze_audio.py --speaker speaker_02
    python scripts/analyze_audio.py --dataset /content/drive/MyDrive/voice_engine/dataset
    python scripts/analyze_audio.py --show                            # also open plot windows

Output (default folder: outputs/phase2/):
    <name>_waveform.png, <name>_spectrogram.png, <name>_mel.png   for every analyzed file
    phase2_audio_features.csv                                      the comparison table
"""

import argparse
import csv
import sys
from pathlib import Path

# Make `import src...` work when this file is run directly as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

# Choose the plotting backend BEFORE pyplot is imported:
# "Agg" draws straight to PNG files (no windows), unless --show was requested.
if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from src.audio.loader import find_dataset_dir, list_audio_files, load_mono_16k  # noqa: E402
from src.audio.features import extract_features, mel_spectrogram, power_to_db, stft_power  # noqa: E402
from src.visualization.plots import plot_mel_spectrogram, plot_spectrogram, plot_waveform  # noqa: E402

FEATURE_COLUMNS = ["filename", "duration", "rms", "zcr", "spectral_centroid"]


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: visualize and measure WAV recordings.")
    parser.add_argument("--file", help="Analyze one WAV file instead of a whole speaker folder")
    parser.add_argument("--dataset", help="Path to the dataset folder (default: ./dataset)")
    parser.add_argument("--speaker", default="speaker_01", help="Speaker folder to analyze (default: speaker_01)")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "phase2"), help="Output folder")
    parser.add_argument("--show", action="store_true", help="Open the plots in windows as well as saving them")
    return parser.parse_args()


def collect_files(args):
    """Return the list of WAV paths to analyze."""
    if args.file:
        path = Path(args.file)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return [path]

    speaker_dir = find_dataset_dir(args.dataset) / args.speaker
    if not speaker_dir.is_dir():
        raise FileNotFoundError(f"Speaker folder not found: {speaker_dir}")
    files = list_audio_files(speaker_dir)
    if not files:
        raise FileNotFoundError(f"No WAV files in {speaker_dir}")
    return files


def analyze_file(path, out_dir):
    """WAV -> waveform -> spectrogram -> Mel spectrogram -> features. Returns the feature row."""
    # Part 1: load (Phase 1 code) + mono + 16 kHz
    waveform, sr = load_mono_16k(path)
    print(path.name)
    print(f"  Sample rate: {sr}")
    print(f"  Samples:     {waveform.shape[1]}")
    print(f"  Duration:    {waveform.shape[1] / sr:.2f} seconds")
    print(f"  Shape:       {waveform.shape}")

    # Part 3: STFT -> spectrogram (freq_bins, frames)
    spec_db = power_to_db(stft_power(waveform))
    print(f"  Spectrogram: {tuple(spec_db.shape)}  (frequency bins x time frames)")

    # Part 4: Mel filter bank -> Mel spectrogram (n_mels, frames)
    mel_db = power_to_db(mel_spectrogram(waveform, sr))
    print(f"  Mel spec:    {tuple(mel_db.shape)}  (Mel bands x time frames)")

    # Part 2-4: plots
    stem = path.stem
    plot_waveform(waveform, sr, f"Waveform — {path.name}", out_dir / f"{stem}_waveform.png")
    plot_spectrogram(spec_db, sr, title=f"Spectrogram — {path.name}", save_path=out_dir / f"{stem}_spectrogram.png")
    plot_mel_spectrogram(mel_db, sr, title=f"Mel spectrogram — {path.name}", save_path=out_dir / f"{stem}_mel.png")
    print(f"  Plots saved: {out_dir / stem}_*.png\n")

    # Part 5: basic features
    return {"filename": path.name, **extract_features(waveform, sr)}


def print_table(rows):
    """Part 6: the comparison table."""
    header = f"{'filename':<18}{'duration (s)':>13}{'RMS':>10}{'ZCR':>10}{'centroid (Hz)':>15}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['filename']:<18}{r['duration']:>13.2f}{r['rms']:>10.4f}"
              f"{r['zcr']:>10.4f}{r['spectral_centroid']:>15.1f}")


def save_csv(rows, csv_path):
    """Part 7: write the feature table to CSV."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "filename": r["filename"],
                "duration": f"{r['duration']:.3f}",
                "rms": f"{r['rms']:.6f}",
                "zcr": f"{r['zcr']:.6f}",
                "spectral_centroid": f"{r['spectral_centroid']:.2f}",
            })


def main():
    args = parse_args()

    try:
        files = collect_files(args)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in files:
        try:
            rows.append(analyze_file(path, out_dir))
        except Exception as exc:  # one bad file shouldn't stop the others
            print(f"[ERROR] {path.name}: {exc}\n")
        if not args.show:
            plt.close("all")  # free memory; figures are already saved

    if not rows:
        print("[ERROR] No files could be analyzed.")
        return 1

    print_table(rows)
    csv_path = out_dir / "phase2_audio_features.csv"
    save_csv(rows, csv_path)
    print(f"\nFeatures saved: {csv_path}")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
