"""
STEP 2 — type text, get a WAV file in your voice (XTTS-v2, GPU recommended: Google Colab).

Usage (from the project folder, after Step 1 and after accepting the model license):
    COQUI_TOS_AGREED=1 python scripts/step2_speak.py --text "Hello, this is my voice." --language en
    COQUI_TOS_AGREED=1 python scripts/step2_speak.py --text "नमस्ते, यह मेरी आवाज़ है।" --language hi

The license (non-commercial CPML, https://coqui.ai/cpml) must be accepted by you:
COQUI_TOS_AGREED=1 means "I have read and agree to it".
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_engine import config  # noqa: E402
from voice_engine.tts import LicenseNotAcceptedError, VoiceCloner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 2: speak text in your voice.")
    parser.add_argument("--text", required=True, help="What to say")
    parser.add_argument("--language", default="en", choices=sorted(config.LANGUAGES), help="en or hi")
    parser.add_argument("--voice", default="speaker_01", help="Voice folder in dataset_clean/ (default: speaker_01)")
    parser.add_argument("--out", help="Output WAV path (default: outputs/speech/<voice>_<lang>_<time>.wav)")
    args = parser.parse_args()

    try:
        start = time.time()
        cloner = VoiceCloner()
        print(f"Model loaded on {cloner.device} in {time.time() - start:.0f}s")
        if cloner.device == "cpu":
            print("[NOTE] No GPU: this works, but is slow. Use Colab with a T4 GPU for faster results.")
        start = time.time()
        n = cloner.load_voice(args.voice)
        print(f"Voice '{args.voice}' loaded from {n} recordings in {time.time() - start:.1f}s")
        start = time.time()
        path = cloner.speak(args.text, args.language, args.voice, args.out)
    except (LicenseNotAcceptedError, FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1
    print(f"Speech ({config.LANGUAGES[args.language]}) generated in {time.time() - start:.1f}s")
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
