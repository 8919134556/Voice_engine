"""
Phase 3 / 9 — generate speaker embeddings for one speaker folder.
Without --checkpoint: the UNTRAINED Phase 3 encoder. With --checkpoint: the TRAINED Phase 9 encoder.

Usage (run from the project root):
    python scripts/generate_speaker_embeddings.py
    python scripts/generate_speaker_embeddings.py --speaker speaker_02
    python scripts/generate_speaker_embeddings.py --dataset <path-to-your-private-dataset>
    python scripts/generate_speaker_embeddings.py --seed 1        # different random network
    python scripts/generate_speaker_embeddings.py --frames 300    # crop/pad every input to 3 s
    python scripts/generate_speaker_embeddings.py --checkpoint outputs/checkpoints/best_model.pt   # TRAINED

Output:
    embeddings/<speaker>/<file>.npy        one (128,) vector per recording  (git-ignored, private)
    outputs/phase3/embeddings_pca.png      2-D picture of the embeddings     (git-ignored)
"""

import argparse
import sys
from pathlib import Path

# Make `import src...` work when this file is run directly as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # save plots to files, no windows

import torch  # noqa: E402

from src.audio.loader import TARGET_SAMPLE_RATE, find_dataset_dir  # noqa: E402
from src.config.settings import ENCODER_SEED  # noqa: E402
from src.embeddings.generate_embeddings import (  # noqa: E402
    embed_mel,
    generate_for_speaker,
    waveform_to_encoder_input,
)
from src.models.speaker_encoder import SpeakerEncoder, count_parameters, get_device  # noqa: E402
from src.similarity.cosine import cosine_similarity, pairwise_similarities  # noqa: E402
from src.training.utils import CheckpointNotFoundError, load_speaker_encoder  # noqa: E402
from src.visualization.plots import plot_embeddings_pca  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Generate speaker embeddings (untrained or trained encoder).")
    parser.add_argument("--dataset", help="Path to the dataset folder (default: ./dataset)")
    parser.add_argument("--speaker", default="speaker_01", help="Speaker folder (default: speaker_01)")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "embeddings"), help="Where to save .npy files")
    parser.add_argument("--seed", type=int, default=ENCODER_SEED, help="Random seed for the untrained weights")
    parser.add_argument("--frames", type=int, help="Crop/pad every Mel spectrogram to this many frames")
    parser.add_argument("--checkpoint", help="Trained encoder checkpoint (Phase 9), e.g. outputs/checkpoints/best_model.pt")
    return parser.parse_args()


def control_sounds(seconds=3.0, sr=TARGET_SAMPLE_RATE):
    """Two sounds that are definitely NOT your voice, to test the untrained encoder."""
    torch.manual_seed(123)
    t = torch.arange(int(seconds * sr)) / sr
    return {
        "white noise": 0.1 * torch.randn(1, len(t)),
        "440 Hz beep": 0.3 * torch.sin(2 * torch.pi * 440 * t).unsqueeze(0),
    }


def main():
    args = parse_args()

    try:
        speaker_dir = find_dataset_dir(args.dataset) / args.speaker
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1
    if not speaker_dir.is_dir() or not any(speaker_dir.glob("*.wav")):
        print(f"[ERROR] No WAV files found in {speaker_dir}")
        return 1

    # 1. Build the encoder: TRAINED from a checkpoint, or UNTRAINED with repeatable random weights.
    device = get_device()
    trained = bool(args.checkpoint)
    if trained:
        try:
            model = load_speaker_encoder(args.checkpoint, device)
        except CheckpointNotFoundError as exc:
            print(f"[ERROR] {exc}")
            return 1
    else:
        torch.manual_seed(args.seed)
        model = SpeakerEncoder().to(device)   # size from src/config/settings.py

    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"Model:  {type(model).__name__}, {count_parameters(model):,} parameters")
    if trained:
        print(f"        TRAINED - {args.checkpoint}\n")
    else:
        print(f"        UNTRAINED - random weights (seed {args.seed})\n")

    # 2-8. WAV -> mono 16 kHz -> Mel -> encoder -> 128-D -> L2-normalize -> .npy
    embeddings = generate_for_speaker(model, speaker_dir, args.out, device, args.frames)

    # Cosine similarity between every pair of recordings
    print("Cosine similarity between recordings (same speaker):")
    for a, b, sim in pairwise_similarities(embeddings):
        print(f"  {Path(a).stem} vs {Path(b).stem}:   {sim:.4f}")

    # Control experiment: compare your voice against sounds that are clearly not a voice.
    first_name = sorted(embeddings)[0]
    print(f"\nControl experiment - {Path(first_name).stem} vs non-voice sounds:")
    for name, waveform in control_sounds().items():
        control = embed_mel(model, waveform_to_encoder_input(waveform, TARGET_SAMPLE_RATE, args.frames), device)
        print(f"  {Path(first_name).stem} vs {name:<12} {cosine_similarity(embeddings[first_name], control):.4f}")

    if not trained:
        print(
            "\n[NOTE] This encoder is UNTRAINED. These similarity values are NOT speaker\n"
            "       verification results: the network has never learned what a voice is.\n"
            "       If noise or a beep also scores high, that is exactly why training is needed."
        )

    # Visualization: 128-D -> PCA -> 2-D (for looking only)
    if len(embeddings) >= 2:
        plot_path = PROJECT_ROOT / "outputs" / "phase3" / "embeddings_pca.png"
        plot_embeddings_pca(
            {Path(n).stem: e for n, e in embeddings.items()},
            title=f"{args.speaker} embeddings ({'trained' if trained else 'untrained'}, PCA to 2-D)",
            save_path=plot_path,
        )
        print(f"\nPCA plot saved: {plot_path}")
    print(f"Embeddings saved in: {Path(args.out) / args.speaker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
