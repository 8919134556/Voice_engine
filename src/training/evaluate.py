"""
evaluate.py — how good is the trained encoder?

1. Classification accuracy on the test recordings (can it name the training speakers?).
2. Speaker verification on the test recordings, using ONLY the 128-D embeddings:
       genuine pairs  = two recordings of the SAME speaker
       impostor pairs = recordings of DIFFERENT speakers
   cosine similarity for every pair -> distributions -> FAR / FRR / accuracy per
   threshold -> EER (where FAR = FRR).

These numbers depend heavily on the dataset size and the test protocol. With few
speakers/recordings they are rough indications, not security guarantees.
"""

from itertools import combinations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.embeddings.generate_embeddings import embed_mel

from .dataset import Recording, SpeakerDataset
from .losses import speaker_classification_loss
from .model import SpeakerClassifier, TrainableSpeakerEncoder
from .preprocessing import load_features
from .train import run_epoch

DEFAULT_THRESHOLDS = [round(t, 2) for t in np.arange(0.30, 0.9001, 0.05)]


def classification_accuracy(classifier: SpeakerClassifier, recordings: list[Recording],
                            segment_frames: int, device: torch.device) -> tuple[float, float]:
    """(loss, accuracy) on middle crops of `recordings`."""
    loader = DataLoader(SpeakerDataset(recordings, segment_frames, train=False), batch_size=32)
    return run_epoch(classifier, loader, speaker_classification_loss(), device)


def extract_embeddings(encoder: TrainableSpeakerEncoder, recordings: list[Recording],
                       device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Embed each WHOLE recording (same path as embedding generation: features -> encoder -> L2).
    Returns (embeddings [N, 128], labels [N]).
    """
    encoder.eval()
    vectors = [embed_mel(encoder, load_features(r.path)[None, None], device) for r in recordings]
    return np.stack(vectors), np.array([r.label for r in recordings])


def pair_scores(embeddings: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity of every pair, split into (genuine, impostor). Embeddings are unit length."""
    genuine, impostor = [], []
    for i, j in combinations(range(len(labels)), 2):
        score = float(embeddings[i] @ embeddings[j])
        (genuine if labels[i] == labels[j] else impostor).append(score)
    return np.array(genuine), np.array(impostor)


def summarize(scores: np.ndarray) -> dict:
    if len(scores) == 0:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {"count": int(len(scores)), "mean": float(scores.mean()),
            "min": float(scores.min()), "max": float(scores.max())}


def error_rates(genuine: np.ndarray, impostor: np.ndarray, threshold: float) -> dict:
    """
    At one threshold (score >= threshold -> MATCH):
      FAR = impostor pairs accepted / impostor pairs     (wrong person let in)
      FRR = genuine pairs rejected  / genuine pairs      (right person turned away)
      accuracy = correct decisions / all pairs (dominated by impostors when they outnumber genuine pairs)
    """
    far = float(np.mean(impostor >= threshold)) if len(impostor) else float("nan")
    frr = float(np.mean(genuine < threshold)) if len(genuine) else float("nan")
    correct = np.sum(genuine >= threshold) + np.sum(impostor < threshold)
    total = len(genuine) + len(impostor)
    return {"threshold": threshold, "far": far, "frr": frr,
            "accuracy": float(correct / total) if total else float("nan")}


def threshold_table(genuine: np.ndarray, impostor: np.ndarray,
                    thresholds: list[float] = DEFAULT_THRESHOLDS) -> list[dict]:
    return [error_rates(genuine, impostor, t) for t in thresholds]


def equal_error_rate(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """
    (EER, threshold): try every observed score as a threshold and pick the one where
    FAR and FRR are closest; EER = their average there. Needs both pair types.
    """
    if len(genuine) == 0 or len(impostor) == 0:
        raise ValueError("EER needs both genuine and impostor pairs.")
    candidates = np.unique(np.concatenate([genuine, impostor]))
    best = min(candidates, key=lambda t: abs(np.mean(impostor >= t) - np.mean(genuine < t)))
    far, frr = np.mean(impostor >= best), np.mean(genuine < best)
    return float((far + frr) / 2), float(best)


def evaluate_encoder(classifier: SpeakerClassifier, test_recordings: list[Recording],
                     segment_frames: int, device: torch.device) -> dict:
    """Full test report (plain dict) + the raw arrays needed for plots."""
    test_loss, test_acc = classification_accuracy(classifier, test_recordings, segment_frames, device)
    embeddings, labels = extract_embeddings(classifier.encoder, test_recordings, device)
    genuine, impostor = pair_scores(embeddings, labels)
    report = {
        "test_recordings": len(test_recordings),
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "genuine": summarize(genuine),
        "impostor": summarize(impostor),
        "thresholds": threshold_table(genuine, impostor),
        "eer": None,
        "eer_threshold": None,
    }
    if len(genuine) and len(impostor):
        report["eer"], report["eer_threshold"] = equal_error_rate(genuine, impostor)
    arrays = {"embeddings": embeddings, "labels": labels, "genuine": genuine, "impostor": impostor}
    return {"report": report, "arrays": arrays}
