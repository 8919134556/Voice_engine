"""
Phase 9 — train the speaker encoder (speaker classification), then evaluate it.

Usage (run from the project root; GPU recommended, e.g. Google Colab):
    python scripts/train_speaker_encoder.py
    python scripts/train_speaker_encoder.py --dataset <path-to-dataset> --epochs 25 --batch-size 32
    python scripts/train_speaker_encoder.py --synthetic        # pipeline test on FAKE speakers (outputs/synthetic_dataset)

Outputs (all git-ignored, under --output, default outputs/):
    checkpoints/best_model.pt, checkpoints/last_model.pt
    training/history.json, training/splits.json, training/evaluation.json
    training/training_curves.png, training/similarity_distributions.png, training/embeddings_pca.png
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from src.config import settings  # noqa: E402
from src.training import (  # noqa: E402
    TrainingConfig,
    TrainableSpeakerEncoder,
    count_parameters,
    describe_device,
    discover_recordings,
    get_device,
    split_recordings,
    train_speaker_encoder,
)
from src.training.evaluate import evaluate_encoder  # noqa: E402
from src.training.train import save_history, save_splits  # noqa: E402
from src.utils.synthetic import write_synthetic_speaker_dataset  # noqa: E402
from src.visualization.plots import (  # noqa: E402
    plot_similarity_distributions,
    plot_speaker_embeddings_pca,
    plot_training_history,
)

LIMITATION = ("This is an experimental speaker embedding model and not a production-grade "
              "speaker recognition system.")


def parse_args():
    p = argparse.ArgumentParser(description="Train the Phase 9 speaker encoder.")
    p.add_argument("--dataset", default=str(settings.DATASET_DIR), help="Dataset folder (one sub-folder per speaker)")
    p.add_argument("--output", default=str(settings.OUTPUTS_DIR), help="Output folder (checkpoints, plots)")
    p.add_argument("--epochs", type=int, default=settings.EPOCHS)
    p.add_argument("--batch-size", type=int, default=settings.BATCH_SIZE)
    p.add_argument("--learning-rate", type=float, default=settings.LEARNING_RATE)
    p.add_argument("--synthetic", action="store_true",
                   help="Ignore --dataset and train on a generated FAKE multi-speaker dataset (pipeline test)")
    return p.parse_args()


def print_report(report: dict) -> None:
    g, i = report["genuine"], report["impostor"]
    print(f"Test recordings:            {report['test_recordings']}")
    print(f"Test classification accuracy: {report['test_accuracy']:.3f}  (loss {report['test_loss']:.4f})")
    print(f"Genuine pairs:  {g['count']:>5}   mean similarity "
          + (f"{g['mean']:.3f}  (min {g['min']:.3f}, max {g['max']:.3f})" if g["count"] else "n/a"))
    print(f"Impostor pairs: {i['count']:>5}   mean similarity "
          + (f"{i['mean']:.3f}  (min {i['min']:.3f}, max {i['max']:.3f})" if i["count"] else "n/a"))
    print("\nThreshold   FAR     FRR     Accuracy")
    for row in report["thresholds"]:
        print(f"  {row['threshold']:.2f}     {row['far']:.3f}   {row['frr']:.3f}   {row['accuracy']:.3f}")
    if report["eer"] is not None:
        print(f"\nEER: {report['eer']:.3f} at threshold {report['eer_threshold']:.3f}")
    print("These numbers depend on the dataset size and test protocol (test = new recordings of "
          "TRAINING speakers). They are not a security guarantee.")


def main() -> int:
    args = parse_args()
    out = Path(args.output)
    dataset = Path(args.dataset)
    if args.synthetic:
        # Kept inside the (git-ignored) output folder so evaluate_speaker_encoder.py can re-use it.
        dataset = out / "synthetic_dataset"
        shutil.rmtree(dataset, ignore_errors=True)
        write_synthetic_speaker_dataset(dataset, n_speakers=6, files_per_speaker=12)
        print("*** SYNTHETIC DATASET: fake speakers generated for a pipeline test. "
              "Results say nothing about real voices. ***\n")
    return run(args, dataset, out)


def run(args, dataset: Path, out: Path) -> int:
    # 1-4. dataset, speakers, recordings, split
    try:
        recordings, speakers = discover_recordings(dataset)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1
    print(f"Dataset:    {dataset.resolve()}")
    print(f"Speakers:   {len(speakers)}")
    print(f"Recordings: {len(recordings)}")
    if len(speakers) < settings.MIN_SPEAKERS_FOR_TRAINING:
        print(f"\n[ERROR] Speaker classification needs at least {settings.MIN_SPEAKERS_FOR_TRAINING} speakers "
              f"(found {len(speakers)}). Add more speaker folders, e.g. dataset/speaker_002/. "
              "Try the pipeline with --synthetic.")
        return 2
    splits, warnings = split_recordings(recordings)
    print(f"Split:      train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])}")
    for w in warnings:
        print(f"  [WARN] {w}")

    # 5-6. device + model
    device = get_device()
    print("\n" + describe_device(device))
    config = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)
    print(f"\nModel: TrainableSpeakerEncoder, {count_parameters(TrainableSpeakerEncoder()):,} parameters, "
          f"{config.embedding_dim}-D embedding (+ training-only classification head over {len(speakers)} speakers)")
    print(f"Config: {config.to_dict()}\n")

    # 7-9. train + checkpoints + history
    training_dir = out / "training"
    save_splits(splits, speakers, training_dir / "splits.json")
    result = train_speaker_encoder(splits, speakers, config, device, checkpoint_dir=out / "checkpoints")
    save_history(result.history, training_dir / "history.json")
    plot_training_history(result.history, training_dir / "training_curves.png")
    h = result.history
    print(f"\nBest epoch: {result.best_epoch}   best checkpoint: {result.best_checkpoint}")
    print(f"Final epoch: train acc {h['train_accuracy'][-1]:.3f}, val acc {h['val_accuracy'][-1]:.3f}")

    # 10-11. evaluation on the test split
    if not splits["test"]:
        print("\n[WARN] No test recordings (too few files per speaker) -> evaluation not run.")
        return 0
    print("\n=== Evaluation on the TEST split (best checkpoint) ===")
    evaluation = evaluate_encoder(result.classifier, splits["test"], config.segment_frames, device)
    report, arrays = evaluation["report"], evaluation["arrays"]
    report["best_epoch"] = result.best_epoch
    report["final_train_accuracy"] = h["train_accuracy"][-1]
    report["final_val_accuracy"] = h["val_accuracy"][-1]
    (training_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(report)

    names = {label: name for name, label in speakers.items()}
    plot_similarity_distributions(arrays["genuine"], arrays["impostor"], report["thresholds"],
                                  report["eer_threshold"], save_path=training_dir / "similarity_distributions.png")
    plot_speaker_embeddings_pca(arrays["embeddings"], arrays["labels"], names,
                                title="Test embeddings by speaker (PCA, qualitative only)",
                                save_path=training_dir / "embeddings_pca.png")
    print(f"\nPlots and reports saved in: {training_dir}")
    print(f"\n{LIMITATION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
