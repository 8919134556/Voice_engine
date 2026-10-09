"""
Step 2 tests. The real XTTS-v2 model (1.8 GB, GPU) is NOT used here: a small stand-in
model checks that our code passes the right things in and saves the right file out.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_engine import config  # noqa: E402
from voice_engine.tts import (  # noqa: E402
    LicenseNotAcceptedError,
    VoiceCloner,
    _soundfile_load_audio,
    find_reference_clips,
)


class FakeXTTS:
    """Records calls like XTTS-v2 would receive them; returns 1 s of quiet audio."""

    def __init__(self):
        self.latent_calls, self.inference_calls = [], []

    def get_conditioning_latents(self, audio_path, **kwargs):
        self.latent_calls.append((audio_path, kwargs))
        return "gpt_latent", "speaker_embedding"

    def inference(self, text, language, gpt_cond_latent, speaker_embedding, **kwargs):
        self.inference_calls.append((text, language, gpt_cond_latent, speaker_embedding, kwargs))
        return {"wav": 0.1 * np.sin(np.linspace(0, 100, config.XTTS_SAMPLE_RATE))}


class TestVoiceCloner(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.clips = []
        for i in range(3):
            path = self.root / "clean" / "speaker_01" / f"audio_{i:03d}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(path), 0.2 * np.sin(np.linspace(0, 300, 22050)), 22050)
            self.clips.append(path)
        self.model = FakeXTTS()
        self.cloner = VoiceCloner(model=self.model)

    def tearDown(self):
        self._tmp.cleanup()

    def test_reference_clips_found_and_missing(self):
        self.assertEqual(find_reference_clips("speaker_01", self.root / "clean"), self.clips)
        with self.assertRaises(FileNotFoundError):
            find_reference_clips("nobody", self.root / "clean")

    def test_voice_profile_built_once(self):
        self.assertEqual(self.cloner.load_voice("speaker_01", self.clips), 3)
        paths, kwargs = self.model.latent_calls[0]
        self.assertEqual(paths, [str(p) for p in self.clips])
        self.assertEqual(kwargs["gpt_cond_len"], config.GPT_COND_LEN)
        for lang in ("en", "hi"):
            self.cloner.speak("text", lang, "speaker_01", self.root / f"{lang}.wav")
        self.assertEqual(len(self.model.latent_calls), 1)            # profile reused, not rebuilt
        self.assertEqual(self.cloner.loaded_voices, ["speaker_01"])

    def test_speak_writes_24khz_wav(self):
        self.cloner.load_voice("speaker_01", self.clips)
        path = self.cloner.speak("नमस्ते, यह मेरी आवाज़ है।", "HI", "speaker_01", self.root / "out" / "hi.wav")
        audio, sr = sf.read(str(path))
        self.assertEqual(sr, config.XTTS_SAMPLE_RATE)
        self.assertEqual(len(audio), config.XTTS_SAMPLE_RATE)
        text, language, latent, embedding, kwargs = self.model.inference_calls[0]
        self.assertEqual((language, latent, embedding), ("hi", "gpt_latent", "speaker_embedding"))
        self.assertTrue(kwargs["enable_text_splitting"])

    def test_bad_input_rejected(self):
        with self.assertRaises(ValueError):
            self.cloner.speak("   ", "en")
        with self.assertRaises(ValueError):
            self.cloner.speak("Bonjour", "fr")

    def test_license_must_be_accepted(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COQUI_TOS_AGREED", None)
            with self.assertRaises(LicenseNotAcceptedError):
                VoiceCloner()

    def test_soundfile_loader_matches_xtts_contract(self):
        audio = _soundfile_load_audio(str(self.clips[0]), 24000)      # 22050 Hz file -> 24000 Hz
        self.assertEqual(tuple(audio.shape), (1, 24000))
        self.assertLessEqual(float(audio.abs().max()), 1.0)


if __name__ == "__main__":
    unittest.main()
