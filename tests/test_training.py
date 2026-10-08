"""
Phase 9 tests: dataset discovery/labels/split, preprocessing, the trainable encoder,
checkpoints, training config, evaluation utilities, and a tiny end-to-end training
run whose embeddings flow into verification and the Voice Registry/Engine.

Synthetic audio and tensors only — never personal recordings.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings  # noqa: E402
from src.embeddings.generate_embeddings import embed_file, waveform_to_encoder_input  # noqa: E402
from src.engine import VoiceEngine, VoiceRequest  # noqa: E402
from src.training import (  # noqa: E402
    CheckpointNotFoundError,
    SpeakerClassifier,
    SpeakerDataset,
    TrainableSpeakerEncoder,
    TrainingConfig,
    discover_recordings,
    discover_speakers,
    load_checkpoint,
    load_speaker_encoder,
    save_checkpoint,
    split_recordings,
    train_speaker_encoder,
)
from src.training.evaluate import (  # noqa: E402
    equal_error_rate,
    error_rates,
    evaluate_encoder,
    pair_scores,
    threshold_table,
)
from src.training.preprocessing import crop_or_pad, load_features, waveform_to_features  # noqa: E402
from src.utils.synthetic import synthetic_speech, write_synthetic_speaker_dataset  # noqa: E402
from src.verification.verifier import create_verifier  # noqa: E402
from src.voice_registry import VoiceProfile, VoiceRegistry  # noqa: E402

CPU = torch.device("cpu")


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class TestDiscoveryAndSplit(TempDirCase):

    def setUp(self):
        super().setUp()
        self.ds = write_synthetic_speaker_dataset(self.root / "ds", n_speakers=3, files_per_speaker=7)
        # uneven speaker + a folder without audio (must be ignored)
        write_synthetic_speaker_dataset(self.root / "extra", n_speakers=1, files_per_speaker=2)
        (self.root / "extra" / "speaker_001").rename(self.ds / "speaker_zzz")
        (self.ds / "notes").mkdir()

    def test_speakers_and_labels(self):
        speakers = discover_speakers(self.ds)
        self.assertEqual(speakers, {"speaker_001": 0, "speaker_002": 1, "speaker_003": 2, "speaker_zzz": 3})

    def test_recordings_unequal_counts(self):
        recordings, speakers = discover_recordings(self.ds)
        self.assertEqual(len(recordings), 3 * 7 + 2)
        for rec in recordings:
            self.assertEqual(rec.label, speakers[rec.speaker])

    def test_missing_dataset(self):
        with self.assertRaises(FileNotFoundError):
            discover_speakers(self.root / "nope")

    def test_split_no_leakage_and_coverage(self):
        recordings, _ = discover_recordings(self.ds)
        splits, warnings = split_recordings(recordings)
        paths = [r.path for split in splits.values() for r in split]
        self.assertEqual(len(paths), len(set(paths)))                       # no file in two splits
        self.assertEqual(len(paths), len(recordings))                       # nothing lost
        for speaker in ["speaker_001", "speaker_002", "speaker_003"]:        # 7 files -> 5 / 1 / 1
            counts = [sum(r.speaker == speaker for r in splits[s]) for s in ("train", "val", "test")]
            self.assertEqual(counts, [5, 1, 1])
        self.assertEqual(sum(r.speaker == "speaker_zzz" for r in splits["train"]), 2)
        self.assertTrue(any("speaker_zzz" in w for w in warnings))          # too small -> reported

    def test_split_is_reproducible(self):
        recordings, _ = discover_recordings(self.ds)
        a, _ = split_recordings(recordings, seed=1)
        b, _ = split_recordings(recordings, seed=1)
        self.assertEqual([r.path for r in a["test"]], [r.path for r in b["test"]])

    def test_dataset_item(self):
        recordings, _ = discover_recordings(self.ds)
        dataset = SpeakerDataset(recordings[:4], segment_frames=120, train=True)
        features, label = dataset[0]
        self.assertEqual(tuple(features.shape), (1, settings.N_MELS, 120))
        self.assertEqual(label, recordings[0].label)
        self.assertEqual(len(dataset), 4)


class TestPreprocessing(TempDirCase):

    def test_same_features_as_phase3_embedding_path(self):
        waveform = torch.from_numpy(synthetic_speech(seconds=2.0)).unsqueeze(0)
        ours = waveform_to_features(waveform, settings.SAMPLE_RATE)
        phase3 = waveform_to_encoder_input(waveform, settings.SAMPLE_RATE)[0, 0]
        self.assertTrue(torch.equal(ours, phase3))
        self.assertEqual(ours.shape[0], settings.N_MELS)

    def test_load_features_resamples(self):
        import soundfile as sf
        path = self.root / "a.wav"
        sf.write(str(path), synthetic_speech(seconds=1.0, sample_rate=48000), 48000)
        features = load_features(path)
        self.assertEqual(features.shape[0], settings.N_MELS)
        self.assertAlmostEqual(features.shape[1], 1.0 * settings.SAMPLE_RATE / settings.HOP_LENGTH, delta=2)

    def test_crop_or_pad(self):
        x = torch.arange(300.0).repeat(80, 1)
        self.assertTrue(torch.equal(crop_or_pad(x, 100), x[:, 100:200]))   # middle window
        short = crop_or_pad(x[:, :40], 100)                                 # repeated, not zero-padded
        self.assertEqual(short.shape[1], 100)
        self.assertTrue(torch.equal(short[:, 40:80], x[:, :40]))
        g = torch.Generator().manual_seed(0)
        self.assertEqual(crop_or_pad(x, 100, random_crop=True, generator=g).shape, (80, 100))


class TestModel(unittest.TestCase):

    def test_output_shapes(self):
        encoder = TrainableSpeakerEncoder().eval()
        for frames in (50, 200, 431):
            with torch.no_grad():
                raw = encoder(torch.randn(3, 1, settings.N_MELS, frames))
            self.assertEqual(tuple(raw.shape), (3, settings.EMBEDDING_DIMENSION))

    def test_encode_is_l2_normalized_128(self):
        encoder = TrainableSpeakerEncoder().eval()
        with torch.no_grad():
            emb = encoder.encode(torch.randn(settings.N_MELS, 150))           # [80, T] accepted
        self.assertEqual(tuple(emb.shape), (1, 128))
        self.assertAlmostEqual(float(emb.norm()), 1.0, places=5)

    def test_classifier(self):
        classifier = SpeakerClassifier(TrainableSpeakerEncoder(), n_speakers=5)
        self.assertEqual(tuple(classifier(torch.randn(2, 1, 80, 100)).shape), (2, 5))
        with self.assertRaises(ValueError):
            SpeakerClassifier(TrainableSpeakerEncoder(), n_speakers=1)


class TestCheckpoints(TempDirCase):

    def test_save_load_roundtrip(self):
        torch.manual_seed(0)
        classifier = SpeakerClassifier(TrainableSpeakerEncoder(), n_speakers=3)
        optimizer = torch.optim.Adam(classifier.parameters())
        path = save_checkpoint(self.root / "ck" / "best_model.pt", classifier, optimizer, epoch=4,
                               metric={"val_loss": 0.5}, config=TrainingConfig().to_dict(),
                               speakers={"a": 0, "b": 1, "c": 2})
        data = load_checkpoint(path)
        for key in ("model_state_dict", "optimizer_state_dict", "epoch", "metric", "config", "speakers"):
            self.assertIn(key, data)
        self.assertEqual(data["epoch"], 4)

        encoder = load_speaker_encoder(path, CPU)
        self.assertFalse(encoder.training)                                   # eval mode
        x = torch.randn(1, 1, 80, 120)
        classifier.encoder.eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(encoder(x), classifier.encoder(x)))

    def test_missing_checkpoint_is_an_error(self):
        with self.assertRaises(CheckpointNotFoundError):
            load_speaker_encoder(self.root / "missing.pt", CPU)
        with self.assertRaises(FileNotFoundError):
            create_verifier(checkpoint=self.root / "missing.pt", device=CPU)


class TestConfig(unittest.TestCase):

    def test_defaults_from_settings(self):
        config = TrainingConfig()
        self.assertEqual(config.embedding_dim, 128)
        self.assertEqual(config.batch_size, settings.BATCH_SIZE)
        self.assertEqual(config.learning_rate, settings.LEARNING_RATE)
        self.assertEqual(config.sample_rate, 16000)
        settings.validate_settings()

    def test_invalid_values(self):
        for bad in ({"epochs": 0}, {"batch_size": -1}, {"learning_rate": 0}):
            with self.assertRaises(ValueError):
                TrainingConfig(**bad)


class TestEvaluationUtils(unittest.TestCase):

    def test_pair_scores_counts(self):
        emb = np.eye(4, 128, dtype=np.float32)
        emb[1] = emb[0]
        genuine, impostor = pair_scores(emb, np.array([0, 0, 1, 1]))
        self.assertEqual((len(genuine), len(impostor)), (2, 4))
        self.assertAlmostEqual(float(genuine.max()), 1.0)

    def test_error_rates(self):
        genuine, impostor = np.array([0.9, 0.8, 0.4]), np.array([0.1, 0.5, 0.95])
        row = error_rates(genuine, impostor, 0.6)
        self.assertAlmostEqual(row["far"], 1 / 3)
        self.assertAlmostEqual(row["frr"], 1 / 3)
        self.assertAlmostEqual(row["accuracy"], 4 / 6)
        self.assertEqual(len(threshold_table(genuine, impostor)), 13)        # 0.30 ... 0.90

    def test_eer(self):
        eer, threshold = equal_error_rate(np.array([0.9, 0.85, 0.8]), np.array([0.1, 0.2, 0.3]))
        self.assertEqual(eer, 0.0)
        self.assertTrue(0.3 < threshold <= 0.8)
        eer, _ = equal_error_rate(np.array([0.9, 0.4]), np.array([0.5, 0.1]))
        self.assertAlmostEqual(eer, 0.5)
        with self.assertRaises(ValueError):
            equal_error_rate(np.array([0.9]), np.array([]))


class TestTinyTrainingRun(TempDirCase):
    """A real (tiny) training run: proves the pipeline end to end, not model quality."""

    def test_train_evaluate_and_use_embeddings(self):
        ds = write_synthetic_speaker_dataset(self.root / "ds", n_speakers=3, files_per_speaker=6)
        recordings, speakers = discover_recordings(ds)
        splits, _ = split_recordings(recordings)
        config = TrainingConfig(epochs=3, batch_size=8, segment_frames=100)
        result = train_speaker_encoder(splits, speakers, config, CPU,
                                       checkpoint_dir=self.root / "ck", verbose=False)

        self.assertTrue(result.best_checkpoint.is_file() and result.last_checkpoint.is_file())
        self.assertEqual(len(result.history["train_loss"]), 3)
        self.assertTrue(all(np.isfinite(result.history["train_loss"])))

        report = evaluate_encoder(result.classifier, splits["test"], 100, CPU)["report"]
        # 6 files per speaker -> 1 test file each -> no same-speaker pair, 3 cross-speaker pairs
        self.assertEqual(report["genuine"]["count"], 0)
        self.assertEqual(report["impostor"]["count"], 3)
        self.assertIsNone(report["eer"])                                     # EER needs both pair types

        # trained encoder -> Phase 4 verifier -> (128,) unit embeddings
        verifier = create_verifier(checkpoint=result.best_checkpoint, device=CPU)
        self.assertTrue(verifier.trained)
        emb = verifier.generate_embedding(recordings[0].path)
        self.assertEqual(emb.shape, (128,))
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=5)

        # trained embedding -> Phase 5 registry -> Phase 6 engine -> Phase 7 conditioning
        (self.root / "embeddings").mkdir()
        encoder = load_speaker_encoder(result.best_checkpoint, CPU)
        np.save(self.root / "embeddings" / "voice_001.npy", embed_file(encoder, recordings[0].path, CPU))
        registry = VoiceRegistry(self.root / "voice_profiles.json")
        registry.register(VoiceProfile(voice_id="voice_001", name="Trained", languages=["en"],
                                       embedding_path="embeddings/voice_001.npy"))
        engine = VoiceEngine(registry)
        conditioning = engine.condition(VoiceRequest("voice_001", "en"))
        self.assertEqual(conditioning.dimension, 128)
        self.assertTrue(conditioning.normalized)
        self.assertEqual(engine.health_check()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
