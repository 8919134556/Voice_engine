"""
Tests for speaker embeddings + verification (Phases 3-4).
They test the PIPELINE (shapes, normalization, threshold logic), not recognition
quality — the encoder is untrained. Synthetic audio only.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import EMBEDDING_DIMENSION, N_MELS, VERIFICATION_THRESHOLD  # noqa: E402
from src.embeddings.generate_embeddings import generate_for_speaker, pad_or_crop  # noqa: E402
from src.models.speaker_encoder import SpeakerEncoder, get_device  # noqa: E402
from src.similarity.cosine import cosine_similarity, pairwise_similarities  # noqa: E402
from src.utils.synthetic import write_synthetic_dataset  # noqa: E402
from src.verification.verifier import VerificationResult, create_verifier  # noqa: E402


class TestCosine(unittest.TestCase):

    def test_known_values(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0)
        self.assertAlmostEqual(cosine_similarity([3, 4], [6, 8]), 1.0)     # length doesn't matter
        self.assertEqual(cosine_similarity([0, 0], [1, 0]), 0.0)

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1, 2, 3], [1, 2])

    def test_pairs(self):
        pairs = pairwise_similarities({"a": [1, 0], "b": [0, 1], "c": [1, 1]})
        self.assertEqual([(a, b) for a, b, _ in pairs], [("a", "b"), ("a", "c"), ("b", "c")])


class TestEncoder(unittest.TestCase):

    def test_any_length_gives_fixed_size(self):
        model = SpeakerEncoder().eval()
        for frames in (50, 300):
            with torch.no_grad():
                out = model(torch.randn(2, 1, N_MELS, frames))
            self.assertEqual(tuple(out.shape), (2, EMBEDDING_DIMENSION))

    def test_pad_or_crop(self):
        mel = torch.randn(N_MELS, 120)
        self.assertEqual(pad_or_crop(mel, 100).shape[1], 100)
        self.assertEqual(pad_or_crop(mel, 150).shape[1], 150)

    def test_device_is_cpu_or_cuda(self):
        self.assertIn(get_device().type, {"cpu", "cuda"})


class TestVerificationPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        write_synthetic_dataset(cls.root / "ds", {"speaker_01": 120.0}, files_per_speaker=2)
        cls.files = sorted((cls.root / "ds" / "speaker_01").glob("*.wav"))
        cls.verifier = create_verifier(device=torch.device("cpu"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_embedding_shape_and_norm(self):
        emb = self.verifier.generate_embedding(self.files[0])
        self.assertEqual(emb.shape, (EMBEDDING_DIMENSION,))
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=5)
        self.assertFalse(self.verifier.trained)

    def test_same_seed_is_repeatable(self):
        again = create_verifier(device=torch.device("cpu"))
        np.testing.assert_allclose(self.verifier.generate_embedding(self.files[0]),
                                   again.generate_embedding(self.files[0]), rtol=1e-6)

    def test_threshold_decision(self):
        self.assertEqual(VerificationResult(0.80, 0.70).decision, "MATCH")
        self.assertEqual(VerificationResult(0.70, 0.70).decision, "MATCH")       # >= is a match
        self.assertEqual(VerificationResult(0.69, 0.70).decision, "NOT MATCH")
        result = self.verifier.verify_files(self.files[0], self.files[1], VERIFICATION_THRESHOLD)
        self.assertEqual(result.threshold, VERIFICATION_THRESHOLD)
        self.assertTrue(-1.0 <= result.similarity <= 1.0)

    def test_generate_for_speaker_saves_npy(self):
        out = self.root / "emb"
        embeddings = generate_for_speaker(SpeakerEncoder(), self.root / "ds" / "speaker_01", out,
                                          torch.device("cpu"), verbose=False)
        self.assertEqual(len(embeddings), 2)
        saved = sorted(p.name for p in (out / "speaker_01").glob("*.npy"))
        self.assertEqual(saved, ["audio_001.npy", "audio_002.npy"])


if __name__ == "__main__":
    unittest.main()
