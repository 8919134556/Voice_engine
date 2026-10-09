"""
Step 1 tests — synthetic audio only (never personal recordings).
Run from the project folder:  python -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_engine import config  # noqa: E402
from voice_engine.audio import check_clip, loudness_db, normalize_loudness, resample, trim_silence  # noqa: E402

SR = 48000


def tone(seconds: float, amp: float = 0.3, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


class TestAudio(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, audio, sr=SR):
        path = self.root / name
        sf.write(str(path), audio, sr)
        return path

    def test_resample_length(self):
        out = resample(tone(1.0), SR, config.CLEAN_SAMPLE_RATE)
        self.assertEqual(len(out), config.CLEAN_SAMPLE_RATE)

    def test_trim_keeps_speech_removes_edges(self):
        audio = np.concatenate([silence(1.0), tone(2.0), silence(1.5)])
        trimmed = trim_silence(audio, SR)
        expected = 2.0 + 2 * config.SILENCE_PADDING_SEC
        self.assertAlmostEqual(len(trimmed) / SR, expected, delta=0.05)

    def test_normalize_loudness_and_peak_limit(self):
        quiet = normalize_loudness(tone(1.0, amp=0.01))
        self.assertAlmostEqual(loudness_db(quiet), config.TARGET_LOUDNESS_DB, delta=0.1)
        spiky = np.zeros(SR, dtype=np.float32)
        spiky[100] = 0.5                                   # one loud click, very low average
        self.assertLessEqual(np.abs(normalize_loudness(spiky)).max(), config.PEAK_LIMIT + 1e-6)

    def test_check_good_stereo_clip(self):
        mono = np.concatenate([silence(0.5), tone(3.0), silence(0.5)])
        report, cleaned = check_clip(self.write("ok.wav", np.stack([mono, mono], axis=1)))
        self.assertTrue(report.ok, report.problems)
        self.assertEqual((report.sample_rate, report.channels), (SR, 2))
        self.assertAlmostEqual(report.speech_sec, 3.0 + 2 * config.SILENCE_PADDING_SEC, delta=0.05)
        self.assertEqual(cleaned.ndim, 1)

    def test_problems_detected(self):
        cases = {
            "silent.wav": silence(3.0),
            "short.wav": np.concatenate([silence(1.0), tone(0.5), silence(1.0)]),
            "clipped.wav": np.clip(tone(3.0, amp=2.0), -1, 1),
        }
        for name, audio in cases.items():
            report, _ = check_clip(self.write(name, audio))
            self.assertFalse(report.ok, name)
        broken = self.root / "broken.wav"
        broken.write_bytes(b"not audio")
        report, cleaned = check_clip(broken)
        self.assertFalse(report.ok)
        self.assertIsNone(cleaned)


if __name__ == "__main__":
    unittest.main()
