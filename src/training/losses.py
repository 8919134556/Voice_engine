"""
losses.py — the training objective.

Phase 9 uses plain SPEAKER CLASSIFICATION with cross-entropy:

    logits = classifier(mel)                 # one score per training speaker
    loss   = CrossEntropy(logits, label)     # low when the correct speaker gets the top score

To classify well, the encoder must put speaker-specific information into its
128-D embedding — that is the representation we keep.

Future improvements (NOT implemented here):
  - Triplet loss:     distance(anchor, same speaker) + margin < distance(anchor, other speaker)
  - Contrastive loss: pull same-speaker pairs together, push different pairs apart
  - AAM-Softmax / ArcFace: classification with an angular margin, which directly makes
                      cosine similarity a better verification score (used by ECAPA-TDNN)
"""

import torch
from torch import nn


def speaker_classification_loss(label_smoothing: float = 0.0) -> nn.Module:
    """Cross-entropy over speaker classes. A little label smoothing can reduce overconfidence."""
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of examples whose highest-scoring class is the correct speaker."""
    return (logits.argmax(dim=1) == labels).float().mean().item()
