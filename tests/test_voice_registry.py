"""
Phase 5 tests for VoiceProfile / VoiceRegistry.

Uses only temporary folders and random test vectors — never personal recordings.

Run from the project root:
    python -m unittest discover -s tests -v
(or `pytest tests` if you have pytest installed)
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.voice_registry import (  # noqa: E402
    DuplicateVoiceError,
    InvalidEmbeddingError,
    VoiceNotFoundError,
    VoiceProfile,
    VoiceRegistry,
    VoiceRegistryError,
    average_embeddings,
)


class RegistryTestCase(unittest.TestCase):
    """Each test gets a fresh temp folder with an embeddings/ subfolder and an empty registry."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "embeddings").mkdir()
        self.registry_file = self.root / "voice_profiles.json"
        self.registry = VoiceRegistry(self.registry_file)

    def tearDown(self):
        self._tmp.cleanup()

    def make_embedding(self, name, array=None, seed=0):
        """Write a .npy test file and return its path relative to the registry folder."""
        if array is None:
            array = np.random.default_rng(seed).normal(size=128).astype(np.float32)
        np.save(self.root / "embeddings" / name, array)
        return f"embeddings/{name}"

    def make_profile(self, voice_id="voice_001", languages=("en", "hi"), gender="male", seed=0):
        path = self.make_embedding(f"{voice_id}.npy", seed=seed)
        return VoiceProfile(voice_id=voice_id, name=f"Name {voice_id}", gender=gender,
                            languages=list(languages), description="test voice", embedding_path=path)


class TestVoiceProfile(RegistryTestCase):

    def test_normalizes_fields(self):
        p = VoiceProfile(voice_id="voice_001", name="  Arjun ", gender=" Male ",
                         languages=["EN", "hi", "en"], embedding_path="embeddings/x.npy")
        self.assertEqual(p.name, "Arjun")
        self.assertEqual(p.gender, "male")
        self.assertEqual(p.languages, ["en", "hi"])   # lowercased, duplicates removed
        self.assertTrue(p.created_at)
        self.assertIsNone(p.updated_at)

    def test_gender_is_optional(self):
        p = VoiceProfile(voice_id="voice_001", name="A", embedding_path="e.npy")
        self.assertIsNone(p.gender)

    def test_rejects_bad_voice_id_and_name(self):
        for bad_id in ["", "Voice 1", "voice/001", "VOICE_001"]:
            with self.assertRaises(ValueError, msg=bad_id):
                VoiceProfile(voice_id=bad_id, name="A", embedding_path="e.npy")
        with self.assertRaises(ValueError):
            VoiceProfile(voice_id="voice_001", name="   ", embedding_path="e.npy")

    def test_dict_round_trip_keeps_unknown_fields(self):
        p = self.make_profile()
        data = p.to_dict()
        data["accent"] = "indian"                   # a field from a hypothetical newer version
        restored = VoiceProfile.from_dict(data)
        self.assertEqual(restored.extra, {"accent": "indian"})
        self.assertEqual(restored.voice_id, p.voice_id)


class TestRegistry(RegistryTestCase):

    # 1. register
    def test_register_voice(self):
        self.registry.register(self.make_profile())
        self.assertTrue(self.registry.exists("voice_001"))
        self.assertIn("voice_001", self.registry)
        self.assertEqual(len(self.registry), 1)

    # 2. get
    def test_get_voice(self):
        profile = self.make_profile()
        self.registry.register(profile)
        self.assertEqual(self.registry.get("voice_001"), profile)

    def test_get_missing_voice_raises(self):
        with self.assertRaises(VoiceNotFoundError):
            self.registry.get("voice_999")

    # 3. list
    def test_list_voices_sorted(self):
        for vid in ["voice_003", "voice_001", "voice_002"]:
            self.registry.register(self.make_profile(vid))
        self.assertEqual([p.voice_id for p in self.registry.list_voices()],
                         ["voice_001", "voice_002", "voice_003"])

    # 4. duplicates
    def test_duplicate_voice_id_rejected(self):
        self.registry.register(self.make_profile())
        original = self.registry.get("voice_001")
        duplicate = VoiceProfile(voice_id="voice_001", name="Impostor",
                                 embedding_path=self.make_embedding("other.npy", seed=5))
        with self.assertRaises(DuplicateVoiceError):
            self.registry.register(duplicate)
        self.assertEqual(self.registry.get("voice_001"), original)   # NOT overwritten

    # 5. update
    def test_update_changes_only_given_fields(self):
        self.registry.register(self.make_profile())
        updated = self.registry.update("voice_001", name="Arjun Kumar")
        self.assertEqual(updated.name, "Arjun Kumar")
        self.assertEqual(updated.languages, ["en", "hi"])          # kept
        self.assertEqual(updated.gender, "male")                   # kept
        self.assertEqual(updated.description, "test voice")        # kept
        self.assertIsNotNone(updated.updated_at)

    def test_update_rejects_identity_fields_and_bad_values(self):
        self.registry.register(self.make_profile())
        before = self.registry.get("voice_001")
        with self.assertRaises(VoiceRegistryError):
            self.registry.update("voice_001", voice_id="voice_002")
        with self.assertRaises(ValueError):
            self.registry.update("voice_001", name="")
        with self.assertRaises(InvalidEmbeddingError):
            self.registry.update("voice_001", embedding_path="embeddings/missing.npy")
        self.assertEqual(self.registry.get("voice_001"), before)   # failed updates change nothing

    def test_update_missing_voice_raises(self):
        with self.assertRaises(VoiceNotFoundError):
            self.registry.update("voice_404", name="x")

    # 6. remove
    def test_remove_voice_keeps_embedding_file(self):
        self.registry.register(self.make_profile())
        removed = self.registry.remove("voice_001")
        self.assertFalse(self.registry.exists("voice_001"))
        self.assertTrue((self.root / removed.embedding_path).is_file())
        with self.assertRaises(VoiceNotFoundError):
            self.registry.remove("voice_001")

    # 7. save / load
    def test_save_and_load(self):
        for i, vid in enumerate(["voice_001", "voice_002"]):
            self.registry.register(self.make_profile(vid, seed=i))
        self.registry.update("voice_002", description="changed")
        self.registry.save()

        data = json.loads(self.registry_file.read_text(encoding="utf-8"))
        self.assertEqual(set(data), {"voice_001", "voice_002"})
        self.assertEqual(data["voice_001"]["embedding_path"], "embeddings/voice_001.npy")
        self.assertNotIn("voice_id", data["voice_001"])            # the key IS the id

        reloaded = VoiceRegistry(self.registry_file)                # brand-new object
        self.assertEqual(reloaded.list_voices(), self.registry.list_voices())
        self.assertEqual(reloaded.get("voice_002").description, "changed")
        np.testing.assert_allclose(reloaded.load_embedding("voice_001"),
                                   np.load(self.root / "embeddings" / "voice_001.npy"))

    def test_load_rejects_corrupt_json(self):
        self.registry_file.write_text("{ not json", encoding="utf-8")
        with self.assertRaises(VoiceRegistryError):
            VoiceRegistry(self.registry_file)

    # 8. language filtering
    def test_find_by_language(self):
        self.registry.register(self.make_profile("voice_001", ["en", "hi"], seed=1))
        self.registry.register(self.make_profile("voice_002", ["en", "hi", "ta"], gender="female", seed=2))
        self.registry.register(self.make_profile("voice_003", ["en"], gender=None, seed=3))

        ids = lambda profiles: [p.voice_id for p in profiles]   # noqa: E731
        self.assertEqual(ids(self.registry.find_by_language("hi")), ["voice_001", "voice_002"])
        self.assertEqual(ids(self.registry.find_by_language("TA")), ["voice_002"])
        self.assertEqual(ids(self.registry.find_by_language("en")), ["voice_001", "voice_002", "voice_003"])
        self.assertEqual(self.registry.find_by_language("fr"), [])
        self.assertEqual(ids(self.registry.find_by_gender("female")), ["voice_002"])

    # 9. invalid embeddings
    def test_invalid_embeddings_rejected(self):
        bad_files = {
            "missing.npy": None,                                           # file does not exist
            "wrong_dim.npy": np.ones(64, dtype=np.float32),
            "empty.npy": np.array([], dtype=np.float32),
            "nan.npy": np.full(128, np.nan, dtype=np.float32),
            "inf.npy": np.r_[np.ones(127), np.inf].astype(np.float32),
            "zeros.npy": np.zeros(128, dtype=np.float32),
            "matrix.npy": np.ones((2, 128), dtype=np.float32),
            "text.npy": np.array(["a"] * 128),
        }
        for name, array in bad_files.items():
            if array is not None:
                self.make_embedding(name, array)
        (self.root / "embeddings" / "garbage.npy").write_bytes(b"this is not numpy")
        bad_files["garbage.npy"] = None

        for name in bad_files:
            profile = VoiceProfile(voice_id="voice_001", name="A", embedding_path=f"embeddings/{name}")
            with self.assertRaises(InvalidEmbeddingError, msg=name):
                self.registry.register(profile)
            self.assertFalse(self.registry.exists("voice_001"), msg=name)

    def test_accepts_row_vector_shape(self):
        path = self.make_embedding("row.npy", np.ones((1, 128), dtype=np.float32))
        self.registry.register(VoiceProfile(voice_id="voice_001", name="A", embedding_path=path))
        self.assertEqual(self.registry.load_embedding("voice_001").shape, (128,))

    def test_average_embeddings(self):
        paths = [self.root / self.make_embedding(f"r{i}.npy", seed=i) for i in range(3)]
        out = average_embeddings(paths, self.root / "embeddings" / "voice_001.npy")
        vector = np.load(out)
        self.assertEqual(vector.shape, (128,))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)
        with self.assertRaises(InvalidEmbeddingError):               # never overwrites
            average_embeddings(paths, out)


if __name__ == "__main__":
    unittest.main()
