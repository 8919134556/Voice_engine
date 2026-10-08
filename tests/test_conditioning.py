"""
Phase 7 tests for VoiceConditioner / VoiceConditioning.

Uses only temporary folders and synthetic random embeddings — never personal recordings.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conditioning import (  # noqa: E402
    ConditioningError,
    VoiceConditioner,
    VoiceConditioning,
    l2_normalize,
    validate_embedding_array,
)
from src.engine import (  # noqa: E402
    InvalidEmbeddingError,
    SelectedVoice,
    VoiceEngine,
    VoiceLanguageNotSupportedError,
    VoiceNotFoundError,
    VoiceRequest,
)
from src.voice_registry import VoiceProfile, VoiceRegistry  # noqa: E402


def random_embedding(seed: int, scale: float = 5.0) -> np.ndarray:
    """Synthetic, deliberately NOT normalized (norm far from 1) so normalization is visible."""
    return (scale * np.random.default_rng(seed).normal(size=128)).astype(np.float32)


class ConditioningTestCase(unittest.TestCase):
    """voice_001 (en, hi) and voice_002 (en), with un-normalized random embeddings."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "embeddings").mkdir()
        self.registry = VoiceRegistry(self.root / "voice_profiles.json")
        self.add_voice("voice_001", ["en", "hi"], random_embedding(1))
        self.add_voice("voice_002", ["en"], random_embedding(2))
        self.engine = VoiceEngine(self.registry)
        self.conditioner = VoiceConditioner(self.engine)

    def tearDown(self):
        self._tmp.cleanup()

    def npy_path(self, voice_id: str) -> Path:
        return self.root / "embeddings" / f"{voice_id}.npy"

    def add_voice(self, voice_id: str, languages: list[str], array: np.ndarray) -> None:
        np.save(self.npy_path(voice_id), array)
        self.registry.register(VoiceProfile(voice_id=voice_id, name=voice_id.title(),
                                            languages=languages,
                                            embedding_path=f"embeddings/{voice_id}.npy"))

    def selected_with(self, embedding: object) -> SelectedVoice:
        """A hand-made SelectedVoice, to feed bad arrays straight into the conditioner."""
        return SelectedVoice("voice_001", self.registry.get("voice_001"), embedding)


class TestVoiceConditioner(ConditioningTestCase):

    # 1
    def test_valid_embedding_produces_conditioning(self):
        result = self.conditioner.condition("voice_001")
        self.assertIsInstance(result, VoiceConditioning)
        self.assertEqual(result.voice_id, "voice_001")

    # 2
    def test_dimension_is_128(self):
        result = self.conditioner.condition("voice_001")
        self.assertEqual(result.dimension, 128)
        self.assertEqual(result.embedding.shape, (128,))

    # 3 + 4
    def test_output_is_normalized(self):
        result = self.conditioner.condition("voice_001")
        self.assertTrue(result.normalized)
        self.assertAlmostEqual(result.l2_norm, 1.0, places=5)
        # same direction as the original: cosine similarity ~ 1
        original = np.load(self.npy_path("voice_001"))
        cosine = float(original @ result.embedding / np.linalg.norm(original))
        self.assertAlmostEqual(cosine, 1.0, places=5)

    # 5
    def test_original_embedding_unchanged(self):
        file_hash_before = hashlib.sha256(self.npy_path("voice_001").read_bytes()).hexdigest()
        cached = self.engine.get_embedding("voice_001")
        cached_before = cached.copy()

        result = self.conditioner.condition("voice_001")

        self.assertEqual(hashlib.sha256(self.npy_path("voice_001").read_bytes()).hexdigest(),
                         file_hash_before)                                   # file on disk
        np.testing.assert_array_equal(self.engine.get_embedding("voice_001"), cached_before)  # cache
        self.assertGreater(np.linalg.norm(cached_before), 2.0)              # still NOT normalized
        self.assertFalse(np.shares_memory(result.embedding, cached))        # result is a copy
        with self.assertRaises(ValueError):
            result.embedding[0] = 1.0                                        # result is read-only

    def test_direct_array_input_is_not_modified(self):
        array = random_embedding(7)
        before = array.copy()
        self.conditioner.condition_selected(self.selected_with(array))
        np.testing.assert_array_equal(array, before)

    # 6
    def test_empty_embedding_rejected(self):
        with self.assertRaises(ConditioningError):
            self.conditioner.condition_selected(self.selected_with(np.array([], dtype=np.float32)))

    # 7
    def test_wrong_dimension_rejected(self):
        for bad in [np.ones(64, np.float32), np.ones(256, np.float32), np.ones((1, 128), np.float32)]:
            with self.assertRaises(ConditioningError, msg=str(bad.shape)):
                self.conditioner.condition_selected(self.selected_with(bad))

    def test_wrong_dimension_on_disk_rejected_by_engine(self):
        np.save(self.npy_path("voice_002"), np.ones(64, np.float32))   # corrupt after registering
        with self.assertRaises(InvalidEmbeddingError):
            self.conditioner.condition("voice_002")

    # 8
    def test_nan_rejected(self):
        bad = random_embedding(3)
        bad[10] = np.nan
        with self.assertRaises(ConditioningError):
            self.conditioner.condition_selected(self.selected_with(bad))

    # 9
    def test_infinite_rejected(self):
        bad = random_embedding(3)
        bad[0] = -np.inf
        with self.assertRaises(ConditioningError):
            self.conditioner.condition_selected(self.selected_with(bad))

    def test_non_array_non_numeric_and_zero_rejected(self):
        for bad in [[0.1] * 128, np.array(["x"] * 128), np.zeros(128, np.float32)]:
            with self.assertRaises(ConditioningError):
                validate_embedding_array(bad)

    # 10
    def test_unknown_voice_rejected(self):
        with self.assertRaises(VoiceNotFoundError):
            self.conditioner.condition("voice_999")
        with self.assertRaises(VoiceNotFoundError):
            self.conditioner.condition_request(VoiceRequest("voice_999"))

    # 11
    def test_request_conditioning(self):
        result = self.conditioner.condition_request(VoiceRequest(voice_id="voice_001", language="hi"))
        self.assertEqual(result.voice_id, "voice_001")
        self.assertEqual(result.language, "hi")
        self.assertTrue(result.normalized)
        self.assertEqual(self.engine.current_voice_id, "voice_001")   # resolved through the engine

    def test_request_with_unsupported_language_rejected(self):
        with self.assertRaises(VoiceLanguageNotSupportedError):
            self.conditioner.condition_request(VoiceRequest(voice_id="voice_002", language="hi"))

    # 12
    def test_multiple_voices_independent(self):
        a = self.conditioner.condition("voice_001")
        b = self.conditioner.condition("voice_002")
        self.assertEqual((a.voice_id, b.voice_id), ("voice_001", "voice_002"))
        self.assertFalse(np.allclose(a.embedding, b.embedding))
        self.assertFalse(np.shares_memory(a.embedding, b.embedding))
        a_again = self.conditioner.condition("voice_001")              # unaffected by voice_002
        np.testing.assert_array_equal(a.embedding, a_again.embedding)

    def test_condition_does_not_change_selection(self):
        self.engine.select_voice("voice_002")
        self.conditioner.condition("voice_001")
        self.assertEqual(self.engine.current_voice_id, "voice_002")

    def test_condition_current(self):
        with self.assertRaises(ConditioningError):                     # nothing selected yet
            self.conditioner.condition_current()
        self.engine.select_voice("voice_002")
        self.assertEqual(self.conditioner.condition_current().voice_id, "voice_002")


class TestHelpers(unittest.TestCase):

    def test_l2_normalize_returns_new_unit_vector(self):
        x = np.array([3.0, 4.0] + [0.0] * 126, dtype=np.float32)
        y = l2_normalize(x)
        self.assertAlmostEqual(float(np.linalg.norm(y)), 1.0, places=6)
        np.testing.assert_allclose(y[:2], [0.6, 0.8], rtol=1e-6)        # 3-4-5 triangle
        self.assertEqual(float(x[0]), 3.0)                               # input untouched

    def test_conditioning_dataclass_checks_shape(self):
        with self.assertRaises(ValueError):
            VoiceConditioning("voice_001", np.ones(64, np.float32), dimension=128, normalized=False)


if __name__ == "__main__":
    unittest.main()
