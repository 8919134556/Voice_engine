"""
Phase 4 — score every pair of recordings in the dataset.

    genuine pair  = two recordings from the SAME speaker folder
    impostor pair = recordings from DIFFERENT speaker folders
                    (only appear once you add speaker_02, speaker_03, ... — never invented)

Usage (run from the project root):
    python scripts/evaluate_pairs.py
    python scripts/evaluate_pairs.py --threshold 0.9
    python scripts/evaluate_pairs.py --dataset <path-to-your-private-dataset>

Output: table in the terminal + outputs/phase4/pair_scores.png (git-ignored)
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

# Make `import src...` work when this file is run directly as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # save plots to files, no windows

from src.audio.loader import find_dataset_dir, list_audio_files, list_speakers  # noqa: E402
from src.config.settings import ENCODER_SEED  # noqa: E402
from src.verification.verifier import (  # noqa: E402
    DEFAULT_THRESHOLD,
    UNTRAINED_WARNING,
    create_verifier,
    format_threshold,
)
from src.visualization.plots import plot_pair_scores  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4: score genuine (and impostor) pairs.")
    parser.add_argument("--dataset", help="Path to the dataset folder (default: ./dataset)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Decision threshold (default {DEFAULT_THRESHOLD}, an educational example)")
    parser.add_argument("--seed", type=int, default=ENCODER_SEED, help="Seed for the untrained weights (same as Phase 3)")
    parser.add_argument("--checkpoint", help="Trained encoder checkpoint, e.g. outputs/checkpoints/best_model.pt (Phase 9)")
    return parser.parse_args()


def build_pairs(recordings):
    """
    recordings: list of (speaker, path)
    returns: list of (rec_a, rec_b, "genuine" | "impostor")
    """
    return [
        (a, b, "genuine" if a[0] == b[0] else "impostor")
        for a, b in combinations(recordings, 2)
    ]


def error_rates(rows, threshold):
    """
    FRR = genuine pairs wrongly rejected / all genuine pairs   (false reject: real speaker turned away)
    FAR = impostor pairs wrongly accepted / all impostor pairs (false accept: wrong person let in)
    Returns None for a rate when there are no pairs of that type.
    """
    genuine = [s for _, _, kind, s in rows if kind == "genuine"]
    impostor = [s for _, _, kind, s in rows if kind == "impostor"]
    frr = sum(s < threshold for s in genuine) / len(genuine) if genuine else None
    far = sum(s >= threshold for s in impostor) / len(impostor) if impostor else None
    return frr, far, len(genuine), len(impostor)


def label(rec):
    speaker, path = rec
    return f"{speaker}/{path.stem}"


def main():
    args = parse_args()
    try:
        dataset_dir = find_dataset_dir(args.dataset)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    recordings = [(spk.name, p) for spk in list_speakers(dataset_dir) for p in list_audio_files(spk)]
    if len(recordings) < 2:
        print(f"[ERROR] Need at least 2 WAV files under {dataset_dir} to form a pair.")
        return 1

    try:
        verifier = create_verifier(checkpoint=args.checkpoint, seed=args.seed)
    except FileNotFoundError as exc:  # includes CheckpointNotFoundError
        print(f"[ERROR] {exc}")
        return 1
    if not verifier.trained:
        print(f"*** {UNTRAINED_WARNING} ***\n")

    # Embed every recording ONCE, then reuse it for all its pairs.
    embeddings = {rec: verifier.generate_embedding(rec[1]) for rec in recordings}

    rows = []
    for a, b, kind in build_pairs(recordings):
        sim = verifier.compare(embeddings[a], embeddings[b])
        rows.append((label(a), label(b), kind, sim))

    # Table
    width = max(len(f"{a} vs {b}") for a, b, _, _ in rows)
    print(f"{'Pair':<{width}}  {'Type':<9} {'Similarity':>10}  Decision (threshold {format_threshold(args.threshold)})")
    print("-" * (width + 46))
    for a, b, kind, sim in rows:
        decision = "MATCH" if sim >= args.threshold else "NOT MATCH"
        print(f"{a + ' vs ' + b:<{width}}  {kind:<9} {sim:>10.4f}  {decision}")

    # Error rates at this threshold
    frr, far, n_gen, n_imp = error_rates(rows, args.threshold)
    print(f"\nGenuine pairs:  {n_gen}    Impostor pairs: {n_imp}")
    print(f"False Rejection Rate (FRR): {frr:.0%}" if frr is not None else "FRR: n/a (no genuine pairs)")
    if far is not None:
        print(f"False Acceptance Rate (FAR): {far:.0%}")
    else:
        print("False Acceptance Rate (FAR): CANNOT BE MEASURED - no impostor pairs.")
        print("  Add a second speaker (dataset/speaker_02/...) to create impostor pairs.")

    plot_path = PROJECT_ROOT / "outputs" / "phase4" / "pair_scores.png"
    plot_pair_scores(rows, args.threshold, save_path=plot_path,
                     title="Pair similarity scores" + ("" if verifier.trained else " (UNTRAINED - demo only)"))
    print(f"\nPlot saved: {plot_path}")

    if not verifier.trained:
        print(f"\n*** {UNTRAINED_WARNING} ***")
        print("These scores come from random weights. They are not speaker verification results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
