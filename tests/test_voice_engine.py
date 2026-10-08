"""
Phase 6 tests for the VoiceEngine.

Uses only temporary folders and random test vectors — never personal recordings.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import (  # noqa: E402
    InvalidEmbeddingError,
    SelectedVoice,
    VoiceEngine,
    VoiceEngineError,
    VoiceLanguageNotSupportedError,
    VoiceNotFoundError,
    VoiceRequest,
)
from src.voice_registry import VoiceProfile, VoiceRegistry  # noqa: E402
from src.voice_registry import VoiceNotFoundError as RegistryVoiceNotFoundError  # noqa: E402


class EngineTestCase(unittest.TestCase):
    """
    Fresh temp registry for each test:
        voice_001  languages en, hi
        voice_002  languages en
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "embeddings").mkdir()
        self.registry = VoiceRegistry(self.root / "voice_profiles.json")
        self.add_voice("voice_001", ["en", "hi"], seed=1)
        self.add_voice("voice_002", ["en"], seed=2)
        self.engine = VoiceEngine(self.registry)

    def tearDown(self):
        self._tmp.cleanup()

    def write_npy(self, name, array):
        np.save(self.root / "embeddings" / name, array)
        return f"embeddings/{name}"

    def add_voice(self, voice_id, languages, seed=0, array=None):
        if array is None:
            array = np.random.default_rng(seed).normal(size=128).astype(np.float32)
        path = self.write_npy(f"{voice_id}.npy", array)
        self.registry.register(VoiceProfile(voice_id=voice_id, name=voice_id.title(),
                                            languages=languages, embedding_path=path))


class TestVoiceEngine(EngineTestCase):

    # 1
    def test_list_voices(self):
        self.assertEqual([p.voice_id for p in self.engine.list_voices()], ["voice_001", "voice_002"])

    # 2
    def test_get_voice(self):
        profile = self.engine.get_voice("voice_001")
        self.assertEqual(profile.voice_id, "voice_001")
        self.assertIs(profile, self.registry.get("voice_001"))   # same object: no second database

    # 3
    def test_unknown_voice_raises(self):
        with self.assertRaises(VoiceNotFoundError) as ctx:
            self.engine.get_voice("voice_999")
        self.assertIn("voice_001", str(ctx.exception))            # helpful: lists available voices
        self.assertIsInstance(ctx.exception, VoiceEngineError)
        self.assertIsInstance(ctx.exception, RegistryVoiceNotFoundError)
        for call in (self.engine.select_voice, self.engine.get_embedding):
            with self.assertRaises(VoiceNotFoundError):
                call("voice_999")

    # 4 + 5
    def test_select_and_current_voice(self):
        self.assertIsNone(self.engine.get_current_voice())         # documented: None before selecting
        self.engine.select_voice("voice_001")
        self.assertEqual(self.engine.current_voice_id, "voice_001")
        self.assertEqual(self.engine.get_current_voice().voice_id, "voice_001")

    def test_select_unknown_keeps_previous_selection(self):
        self.engine.select_voice("voice_001")
        with self.assertRaises(VoiceNotFoundError):
            self.engine.select_voice("voice_999")
        self.assertEqual(self.engine.current_voice_id, "voice_001")

    # 6
    def test_switch_voices_same_engine(self):
        engine_before = self.engine
        self.engine.select_voice("voice_001")
        self.engine.select_voice("voice_002")
        self.assertIs(self.engine, engine_before)
        self.assertEqual(self.engine.get_current_voice().voice_id, "voice_002")

    def test_current_voice_reflects_registry_updates(self):
        self.engine.select_voice("voice_001")
        self.registry.update("voice_001", name="Renamed")
        self.assertEqual(self.engine.get_current_voice().name, "Renamed")

    # 7
    def test_embedding_loads_correctly(self):
        embedding = self.engine.get_embedding("voice_001")
        expected = np.load(self.root / "embeddings" / "voice_001.npy")
        self.assertEqual(embedding.shape, (128,))
        np.testing.assert_allclose(embedding, expected)

    # 8
    def test_embedding_cache(self):
        self.assertFalse(self.engine.is_cached("voice_001"))
        first = self.engine.get_embedding("voice_001")
        self.assertTrue(self.engine.is_cached("voice_001"))

        # Delete the file: a second call can only succeed if it uses the cache.
        (self.root / "embeddings" / "voice_001.npy").unlink()
        second = self.engine.get_embedding("voice_001")
        self.assertIs(second, first)

        self.engine.clear_cache("voice_001")
        with self.assertRaises(InvalidEmbeddingError):             # now it must read the (missing) file
            self.engine.get_embedding("voice_001")

    def test_cached_embedding_is_read_only(self):
        embedding = self.engine.get_embedding("voice_001")
        with self.assertRaises(ValueError):
            embedding[0] = 99.0

    def test_cache_reloads_when_embedding_path_changes(self):
        old = self.engine.get_embedding("voice_001")
        new_path = self.write_npy("voice_001_v2.npy", np.ones(128, dtype=np.float32))
        self.registry.update("voice_001", embedding_path=new_path)
        new = self.engine.get_embedding("voice_001")
        self.assertFalse(np.allclose(old, new))
        np.testing.assert_allclose(new, np.ones(128))

    # 9
    def test_invalid_embedding_dimension_rejected(self):
        self.add_voice("voice_003", ["en"])
        self.write_npy("voice_003.npy", np.ones(64, dtype=np.float32))   # corrupt it after registering
        with self.assertRaises(InvalidEmbeddingError):
            self.engine.get_embedding("voice_003")
        self.assertFalse(self.engine.is_cached("voice_003"))

    # 10
    def test_non_finite_embedding_rejected(self):
        self.add_voice("voice_003", ["en"])
        bad = np.ones(128, dtype=np.float32)
        bad[5] = np.nan
        self.write_npy("voice_003.npy", bad)
        with self.assertRaises(InvalidEmbeddingError):
            self.engine.get_embedding("voice_003")

    # 11
    def test_language_supported(self):
        selected = self.engine.resolve(VoiceRequest(voice_id="voice_001", language="hi"))
        self.assertEqual(selected.language, "hi")
        self.engine.check_language("voice_001", "EN")              # case-insensitive, no error

    # 12
    def test_unsupported_language_raises(self):
        self.engine.select_voice("voice_001")
        with self.assertRaises(VoiceLanguageNotSupportedError) as ctx:
            self.engine.resolve(VoiceRequest(voice_id="voice_002", language="hi"))
        self.assertEqual(ctx.exception.supported, ["en"])
        self.assertEqual(self.engine.current_voice_id, "voice_001")  # failed request changes nothing

    # 13
    def test_resolve_request(self):
        selected = self.engine.resolve(VoiceRequest(voice_id="voice_002"))
        self.assertIsInstance(selected, SelectedVoice)
        self.assertEqual(selected.voice_id, "voice_002")
        self.assertEqual(selected.profile.voice_id, "voice_002")
        self.assertIsNone(selected.language)                       # no language -> no language check
        np.testing.assert_allclose(selected.embedding, np.load(self.root / "embeddings" / "voice_002.npy"))
        self.assertEqual(self.engine.current_voice_id, "voice_002")
        self.assertTrue(self.engine.is_cached("voice_002"))

    def test_resolve_with_bad_embedding_keeps_selection(self):
        self.engine.select_voice("voice_001")
        self.add_voice("voice_003", ["en"])
        self.write_npy("voice_003.npy", np.zeros(10, dtype=np.float32))
        with self.assertRaises(InvalidEmbeddingError):
            self.engine.resolve(VoiceRequest(voice_id="voice_003"))
        self.assertEqual(self.engine.current_voice_id, "voice_001")

    # 14
    def test_default_voice(self):
        engine = VoiceEngine(self.registry, default_voice_id="voice_002")
        self.assertEqual(engine.get_current_voice().voice_id, "voice_002")
        with self.assertRaises(VoiceNotFoundError):
            VoiceEngine(self.registry, default_voice_id="voice_999")


class TestVoiceRequest(unittest.TestCase):

    def test_normalizes_language(self):
        self.assertEqual(VoiceRequest("voice_001", " HI ").language, "hi")
        self.assertIsNone(VoiceRequest("voice_001", "").language)
        self.assertIsNone(VoiceRequest("voice_001").language)

    def test_rejects_empty_voice_id(self):
        for bad in ["", "   ", None]:
            with self.assertRaises(ValueError):
                VoiceRequest(bad)

    def test_is_immutable(self):
        request = VoiceRequest("voice_001", "hi")
        with self.assertRaises(Exception):
            request.voice_id = "voice_002"


if __name__ == "__main__":
    unittest.main()
