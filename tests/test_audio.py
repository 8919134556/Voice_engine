"""
Tests for the audio utilities (Phases 1-2): loading, preprocessing, inspection, features.
Synthetic audio in temporary folders only — never personal recordings.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.features import (  # noqa: E402
    extract_features,
    mel_filterbank,
    mel_spectrogram,
    power_to_db,
    stft_power,
    zero_crossing_rate,
)
from src.audio.inspect import inspect_audio  # noqa: E402
from src.audio.loader import list_audio_files, load_audio, load_mono_16k, preprocess_audio  # noqa: E402
from src.config.settings import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE  # noqa: E402
from src.utils.synthetic import synthetic_speech, write_synthetic_dataset  # noqa: E402


class AudioTestCase(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, data, sr=SAMPLE_RATE):
        path = self.root / name
        sf.write(str(path), data, sr)
        return path


class TestLoader(AudioTestCase):

    def test_load_mono_shape(self):
        path = self.write("a.wav", synthetic_speech(seconds=2.0))
        waveform, sr = load_audio(path)
        self.assertEqual(sr, SAMPLE_RATE)
        self.assertEqual(waveform.shape, (1, 2 * SAMPLE_RATE))
        self.assertEqual(waveform.dtype, torch.float32)

    def test_preprocess_stereo_44k_to_mono_16k(self):
        audio = synthetic_speech(seconds=1.0, sample_rate=44100)
        path = self.write("stereo.wav", np.stack([audio, audio], axis=1), sr=44100)
        waveform, sr = load_mono_16k(path)
        self.assertEqual(sr, SAMPLE_RATE)
        self.assertEqual(waveform.shape[0], 1)
        self.assertAlmostEqual(waveform.shape[1] / sr, len(audio) / 44100, places=2)

    def test_preprocess_is_noop_for_mono_16k(self):
        w = torch.randn(1, 1600)
        out, sr = preprocess_audio(w, SAMPLE_RATE)
        self.assertTrue(torch.equal(out, w))

    def test_list_audio_files_sorted_wav_only(self):
        write_synthetic_dataset(self.root / "ds", files_per_speaker=3)
        (self.root / "ds" / "speaker_01" / "notes.txt").write_text("x")
        names = [p.name for p in list_audio_files(self.root / "ds" / "speaker_01")]
        self.assertEqual(names, ["audio_001.wav", "audio_002.wav", "audio_003.wav"])


class TestInspect(AudioTestCase):

    def test_good_file(self):
        info = inspect_audio(self.write("ok.wav", synthetic_speech(seconds=2.0)))
        self.assertTrue(info.ok)
        self.assertTrue(info.is_mono)
        self.assertEqual(info.warnings, [])

    def test_problems_detected(self):
        self.assertFalse(inspect_audio(self.write("silent.wav", np.zeros(SAMPLE_RATE))).ok)
        short = inspect_audio(self.write("short.wav", synthetic_speech(seconds=0.5)))
        self.assertTrue(any("short" in w.lower() for w in short.warnings))
        broken = self.root / "broken.wav"
        broken.write_bytes(b"not audio")
        self.assertFalse(inspect_audio(broken).ok)


class TestFeatures(unittest.TestCase):

    def setUp(self):
        self.waveform = torch.from_numpy(synthetic_speech(seconds=2.0)).unsqueeze(0)

    def test_shapes(self):
        frames = self.waveform.shape[1] // HOP_LENGTH + 1
        self.assertEqual(tuple(stft_power(self.waveform).shape), (N_FFT // 2 + 1, frames))
        self.assertEqual(tuple(mel_spectrogram(self.waveform, SAMPLE_RATE).shape), (N_MELS, frames))
        self.assertEqual(tuple(mel_filterbank(SAMPLE_RATE).shape), (N_FFT // 2 + 1, N_MELS))

    def test_power_to_db_is_finite_and_capped(self):
        db = power_to_db(stft_power(self.waveform))
        self.assertTrue(torch.isfinite(db).all())
        self.assertLessEqual(float(db.max() - db.min()), 80.0 + 1e-4)

    def test_features_react_as_expected(self):
        base = extract_features(self.waveform, SAMPLE_RATE)
        quiet = extract_features(self.waveform * 0.3, SAMPLE_RATE)
        self.assertAlmostEqual(base["duration"], 2.0, places=2)
        self.assertAlmostEqual(quiet["rms"], base["rms"] * 0.3, places=4)     # volume -> RMS only
        self.assertAlmostEqual(quiet["zcr"], base["zcr"], places=6)
        self.assertAlmostEqual(quiet["spectral_centroid"], base["spectral_centroid"], places=1)

    def test_zero_crossing_rate_of_alternating_signal(self):
        alternating = torch.tensor([[1.0, -1.0] * 50])
        self.assertAlmostEqual(zero_crossing_rate(alternating), 1.0)


if __name__ == "__main__":
    unittest.main()
