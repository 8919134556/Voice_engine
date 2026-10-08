"""
Phase 8 tests: configuration, logging, the VoiceProvider interface, engine
lifecycle (lazy loading, facade), health_check() and info().
Synthetic embeddings in temporary folders only.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings, validate_settings  # noqa: E402
from src.conditioning import VoiceConditioning  # noqa: E402
from src.engine import (  # noqa: E402
    ConfigurationError,
    VoiceEngine,
    VoiceNotFoundError,
    VoiceProvider,
    VoiceRequest,
)
from src.utils.logging import get_logger, setup_logging  # noqa: E402
from src.utils.synthetic import random_embedding  # noqa: E402
from src.voice_registry import VoiceProfile, VoiceRegistry  # noqa: E402


class TestSettings(unittest.TestCase):

    def test_defaults(self):
        self.assertEqual(settings.SAMPLE_RATE, 16000)
        self.assertEqual(settings.EMBEDDING_DIMENSION, 128)
        self.assertEqual(settings.DEFAULT_LANGUAGE, "en")
        validate_settings()

    def test_paths_are_inside_project(self):
        for path in (settings.DATASET_DIR, settings.EMBEDDINGS_DIR, settings.REGISTRY_FILE):
            self.assertEqual(path.resolve().parent if path.suffix else path.resolve().parent,
                             settings.PROJECT_ROOT)

    def test_modules_use_settings(self):
        from src.audio.loader import TARGET_SAMPLE_RATE
        from src.conditioning.representation import EXPECTED_DIM
        from src.voice_registry.registry import EMBEDDING_DIM
        self.assertEqual({TARGET_SAMPLE_RATE}, {settings.SAMPLE_RATE})
        self.assertEqual({EXPECTED_DIM, EMBEDDING_DIM}, {settings.EMBEDDING_DIMENSION})


class EngineFixture(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "embeddings").mkdir()
        self.registry_file = self.root / "voice_profiles.json"
        self.registry = VoiceRegistry(self.registry_file)
        for i, (vid, langs) in enumerate([("voice_001", ["en", "hi"]), ("voice_002", ["en"])]):
            np.save(self.root / "embeddings" / f"{vid}.npy", random_embedding(i, scale=4.0))
            self.registry.register(VoiceProfile(voice_id=vid, name=vid, languages=langs,
                                                embedding_path=f"embeddings/{vid}.npy"))
        self.registry.save()

    def tearDown(self):
        self._tmp.cleanup()


class TestProviderInterface(EngineFixture):

    def test_registry_satisfies_protocol(self):
        self.assertIsInstance(self.registry, VoiceProvider)

    def test_engine_accepts_any_provider(self):
        """A completely different store (here: a dict) works without changing the engine."""
        registry = self.registry

        class InMemoryProvider:
            embedding_dim = 128

            def list_voices(self):
                return registry.list_voices()

            def exists(self, voice_id):
                return registry.exists(voice_id)

            def get(self, voice_id):
                return registry.get(voice_id)

            def load_embedding(self, voice_id):
                return registry.load_embedding(voice_id)

        engine = VoiceEngine(InMemoryProvider(), default_voice_id="voice_002")
        self.assertEqual(engine.get_current_voice().voice_id, "voice_002")

    def test_non_provider_rejected(self):
        with self.assertRaises(ConfigurationError):
            VoiceEngine(object())

    def test_dimension_mismatch_rejected(self):
        with self.assertRaises(ConfigurationError):
            VoiceEngine(VoiceRegistry(self.registry_file, embedding_dim=256))


class TestLifecycle(EngineFixture):

    def test_from_registry_file(self):
        engine = VoiceEngine.from_registry_file(self.registry_file, default_voice_id="voice_001")
        self.assertEqual(engine.current_voice_id, "voice_001")
        with self.assertRaises(ConfigurationError):
            VoiceEngine.from_registry_file(self.root / "missing.json")

    def test_embeddings_are_loaded_lazily(self):
        engine = VoiceEngine(self.registry, default_voice_id="voice_001")
        self.assertEqual(engine.info()["cached_embeddings"], [])          # nothing loaded at startup
        engine.get_embedding("voice_001")
        self.assertEqual(engine.info()["cached_embeddings"], ["voice_001"])

    def test_condition_facade(self):
        engine = VoiceEngine(self.registry, default_voice_id="voice_001")
        current = engine.condition()
        by_id = engine.condition("voice_002")
        by_request = engine.condition(VoiceRequest("voice_002", "en"))
        for c in (current, by_id, by_request):
            self.assertIsInstance(c, VoiceConditioning)
            self.assertTrue(c.normalized)
        self.assertEqual(current.voice_id, "voice_001")
        self.assertEqual(engine.current_voice_id, "voice_002")             # the request selected it
        with self.assertRaises(VoiceNotFoundError):
            engine.condition("voice_999")


class TestHealthAndInfo(EngineFixture):

    def test_healthy(self):
        engine = VoiceEngine(self.registry)
        health = engine.health_check()
        self.assertEqual(health, {"status": "ok", "voices": 2, "valid_embeddings": 2, "problems": {}})
        self.assertFalse(engine.is_cached("voice_001"))                     # health check is side-effect free

    def test_degraded_when_embedding_missing_or_wrong_size(self):
        np.save(self.root / "embeddings" / "voice_002.npy", np.ones(64, np.float32))
        (self.root / "embeddings" / "voice_001.npy").unlink()
        health = VoiceEngine(self.registry).health_check()
        self.assertEqual(health["status"], "degraded")
        self.assertEqual(health["valid_embeddings"], 0)
        self.assertEqual(set(health["problems"]), {"voice_001", "voice_002"})

    def test_error_when_provider_fails(self):
        engine = VoiceEngine(self.registry)
        engine.registry = None                                              # simulate a broken store
        self.assertEqual(engine.health_check()["status"], "error")

    def test_info(self):
        engine = VoiceEngine(self.registry, default_voice_id="voice_002")
        info = engine.info()
        self.assertEqual(info["engine"], "VoiceEngine")
        self.assertEqual(info["embedding_dimension"], 128)
        self.assertEqual(info["registered_voices"], 2)
        self.assertEqual(info["current_voice"], "voice_002")
        json.dumps(info)                                                    # plain, serializable data


class TestLogging(EngineFixture):

    def test_logger_names(self):
        self.assertEqual(get_logger("src.engine.voice_engine").name, "voice_engine.engine.voice_engine")
        setup_logging("WARNING")
        setup_logging("WARNING")                                             # no duplicate handlers
        root = get_logger("x").parent
        self.assertEqual(sum(getattr(h, "_voice_engine", False) for h in root.handlers), 1)

    def test_engine_logs_events_but_not_embeddings(self):
        engine = VoiceEngine(self.registry)
        with self.assertLogs("voice_engine", level="INFO") as logs:
            engine.select_voice("voice_001")
            engine.get_embedding("voice_001")
            engine.get_embedding("voice_001")
            engine.condition("voice_001")
        text = "\n".join(logs.output)
        for expected in ["Selected voice: voice_001", "Embedding cache miss: voice_001",
                         "Loading embedding: voice_001", "Embedding cache hit: voice_001",
                         "Conditioned voice: voice_001"]:
            self.assertIn(expected, text)
        first_value = f"{engine.get_embedding('voice_001')[0]:.4f}"
        self.assertNotIn(first_value, text)                                 # no raw numbers in logs
        self.assertNotIn("[", text.replace("['", ""))

    def test_errors_are_logged(self):
        engine = VoiceEngine(self.registry)
        with self.assertLogs("voice_engine", level="WARNING") as logs:
            with self.assertRaises(VoiceNotFoundError):
                engine.select_voice("voice_999")
        self.assertIn("Voice not found: voice_999", logs.output[0])


if __name__ == "__main__":
    unittest.main()
