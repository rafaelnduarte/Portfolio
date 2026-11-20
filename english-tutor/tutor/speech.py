"""Open-source speech components (all run locally, no paid APIs).

- PhonemeRecognizer: wav2vec2 model that outputs the SOUNDS spoken (IPA),
  not the words. This is what makes pronunciation feedback possible.
- Transcriber: Whisper (faster-whisper) for WHAT was said.
- Voice: Kokoro text-to-speech for the teacher's voice.

Models download on first use (Hugging Face for the first two, GitHub for Kokoro).
"""
from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
MODELS_DIR = Path(os.environ.get("TUTOR_MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))

PHONEME_MODEL = os.environ.get("TUTOR_PHONEME_MODEL", "facebook/wav2vec2-lv-60-espeak-cv-ft")
WHISPER_MODEL = os.environ.get("TUTOR_WHISPER_MODEL", "small.en")
KOKORO_VOICE = os.environ.get("TUTOR_VOICE", "af_heart")
KOKORO_URLS = {
    "kokoro-v1.0.int8.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}


def load_audio(path: str | Path) -> np.ndarray:
    """Any audio file -> mono float32 at 16 kHz (uses ffmpeg, handles webm/ogg from browsers)."""
    import subprocess

    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


class PhonemeRecognizer:
    @cached_property
    def _model(self):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        torch.set_num_threads(max(1, os.cpu_count() or 1))
        proc = AutoProcessor.from_pretrained(PHONEME_MODEL)
        model = Wav2Vec2ForCTC.from_pretrained(PHONEME_MODEL).eval()
        return proc, model

    def recognize(self, audio: np.ndarray) -> list[str]:
        import torch

        proc, model = self._model
        inputs = proc(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            logits = model(inputs.input_values).logits
        ids = torch.argmax(logits, dim=-1)
        text = proc.batch_decode(ids)[0]
        return [p for p in text.split(" ") if p]


class Transcriber:
    @cached_property
    def _model(self):
        from faster_whisper import WhisperModel

        return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray) -> str:
        # A literal prompt nudges Whisper to keep disfluencies instead of "cleaning up".
        segments, _ = self._model.transcribe(
            audio, language="en", beam_size=1, condition_on_previous_text=False,
            initial_prompt="Umm, I, uh, I goed to the... the store yesterday.",
        )
        return " ".join(s.text.strip() for s in segments).strip()


class Voice:
    def __init__(self, voice: str = KOKORO_VOICE, speed: float = 0.95):
        self.voice, self.speed = voice, speed

    @staticmethod
    def _ensure_files() -> tuple[Path, Path]:
        import urllib.request

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for name, url in KOKORO_URLS.items():
            dest = MODELS_DIR / name
            if not dest.exists():
                urllib.request.urlretrieve(url, dest)
        return MODELS_DIR / "kokoro-v1.0.int8.onnx", MODELS_DIR / "voices-v1.0.bin"

    @cached_property
    def _tts(self):
        from kokoro_onnx import Kokoro

        return Kokoro(*map(str, self._ensure_files()))

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        samples, sr = self._tts.create(text, voice=self.voice, speed=self.speed, lang="en-us")
        return samples, sr

    def to_wav(self, text: str, path: str | Path) -> Path:
        import soundfile as sf

        samples, sr = self.synthesize(text)
        sf.write(str(path), samples, sr)
        return Path(path)
