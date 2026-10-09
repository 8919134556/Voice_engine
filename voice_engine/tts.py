"""
tts.py — Step 2: speak any text in your voice with XTTS-v2 (a pretrained voice-cloning model).

    your cleaned recordings ──► model.get_conditioning_latents()  ──► "voice profile"
                                (done ONCE per voice, kept in memory)        │
    text + language ─────────────────────────────────────────────────────────┤
                                                                             ▼
                                model.inference() ──► 24 kHz audio ──► outputs/speech/*.wav

XTTS-v2 was trained by Coqui on thousands of hours of speech, so it already knows how
to talk. It does NOT learn your voice permanently; it imitates it from the reference
clips every time you load the voice (zero-shot cloning).

License: the XTTS-v2 model weights are under the Coqui Public Model License (CPML),
non-commercial use only — https://coqui.ai/cpml. You must accept it yourself
(environment variable COQUI_TOS_AGREED=1, set by the notebook only after you agree).
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config
from .audio import load_mono, resample


class LicenseNotAcceptedError(RuntimeError):
    """The XTTS-v2 model license (CPML) has not been accepted."""


def find_reference_clips(voice: str, clean_dir: Path = config.CLEAN_DIR) -> list[Path]:
    """The Step 1 cleaned recordings of `voice` — these are what the model imitates."""
    clips = sorted((Path(clean_dir) / voice).glob("*.wav"))
    if not clips:
        raise FileNotFoundError(
            f"No cleaned recordings in {Path(clean_dir) / voice}. Run Step 1 first: "
            f"python scripts/step1_prepare_dataset.py --voice {voice}"
        )
    return clips


def _soundfile_load_audio(audiopath, sampling_rate):
    """
    Replacement for XTTS's own load_audio(), which uses torchaudio.load. With new PyTorch
    versions torchaudio.load needs an extra package (torchcodec) that must exactly match the
    installed PyTorch — a frequent install problem on Colab. Reading with soundfile avoids it.
    Same result: a [1, samples] float tensor at `sampling_rate`, values in -1..1.
    """
    import torch
    audio, sr = load_mono(audiopath)
    audio = np.clip(resample(audio, sr, sampling_rate), -1.0, 1.0)
    return torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)


class VoiceCloner:
    """
    cloner = VoiceCloner()                      # downloads XTTS-v2 once (~1.8 GB), uses the GPU if there is one
    cloner.load_voice("speaker_01")             # listen to your recordings once
    cloner.speak("Hello!", language="en")       # -> outputs/speech/speaker_01_en_<time>.wav
    """

    def __init__(self, device: str | None = None, model=None):
        """`model` is only for tests (a stand-in object); normally XTTS-v2 is loaded here."""
        self._voices: dict[str, tuple] = {}
        if model is not None:
            self.model, self.device = model, device or "cpu"
            return

        if os.environ.get("COQUI_TOS_AGREED") != "1":
            raise LicenseNotAcceptedError(
                "XTTS-v2 is licensed under the non-commercial Coqui Public Model License "
                "(https://coqui.ai/cpml). Read it, and if you agree set COQUI_TOS_AGREED=1 "
                "(in the notebook: I_ACCEPT_COQUI_CPML = True)."
            )
        import torch
        import TTS.tts.models.xtts as xtts_module
        from TTS.api import TTS

        xtts_module.load_audio = _soundfile_load_audio      # see _soundfile_load_audio()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TTS(config.XTTS_MODEL, progress_bar=False).to(self.device).synthesizer.tts_model

    @property
    def loaded_voices(self) -> list[str]:
        return sorted(self._voices)

    def load_voice(self, voice: str = "speaker_01", reference_clips: list[Path] | None = None) -> int:
        """Build the voice profile from the reference clips. Returns how many clips were used."""
        clips = reference_clips or find_reference_clips(voice)
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[str(p) for p in clips],
            gpt_cond_len=config.GPT_COND_LEN,
            gpt_cond_chunk_len=config.GPT_COND_CHUNK_LEN,
            max_ref_length=config.MAX_REF_LENGTH,
        )
        self._voices[voice] = (gpt_cond_latent, speaker_embedding)
        return len(clips)

    def speak(self, text: str, language: str = "en", voice: str = "speaker_01",
              out_path: str | Path | None = None) -> Path:
        """Turn `text` into speech in `voice` and save it as a WAV file. Returns the file path."""
        text = (text or "").strip()
        if not text:
            raise ValueError("Text is empty.")
        language = language.strip().lower()
        if language not in config.LANGUAGES:
            raise ValueError(f"Language '{language}' not supported here. Use one of: {', '.join(config.LANGUAGES)}.")
        if voice not in self._voices:
            self.load_voice(voice)

        gpt_cond_latent, speaker_embedding = self._voices[voice]
        result = self.model.inference(
            text, language, gpt_cond_latent, speaker_embedding,
            temperature=config.TEMPERATURE,
            enable_text_splitting=True,          # long text is spoken sentence by sentence
        )
        audio = np.asarray(result["wav"], dtype=np.float32)

        if out_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = config.SPEECH_DIR / f"{voice}_{language}_{stamp}.wav"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio, config.XTTS_SAMPLE_RATE)
        return out_path
