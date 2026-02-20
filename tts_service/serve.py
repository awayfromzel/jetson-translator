"""
FastAPI-based Piper TTS microservice.

Loads all configured voices at startup and exposes:
- GET /health
- POST /synthesize

Synthesizes speech in-memory and returns raw WAV audio.
Designed to run in a separate Python environment to
avoid dependency conflicts with the main application.
"""

import os
import io
import wave
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from piper.voice import PiperVoice


# ---- env/config ----
VOICE_DIR = Path(os.environ.get("PIPER_VOICE_DIR", "")).expanduser().resolve()
USE_CUDA = os.environ.get("PIPER_USE_CUDA", "0").strip().lower() in ("1", "true", "yes", "y")
# NOTE: PIPER_SPEAKER_RATE can be added if needed, but Piper will embed the correct rate in the WAV.

VOICE_MAP = {
    "eng_Latn": "en_GB-cori-high",
    "ita_Latn": "it_IT-paola-medium",
    "spa_Latn": "es_MX-claude-high",
    "por_Latn": "pt_BR-faber-medium",
    "fra_Latn": "fr_FR-siwis-medium",
}


def _voice_paths(lang_code: str) -> tuple[Path, Path]:
    if lang_code not in VOICE_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown lang_code: {lang_code}")

    name = VOICE_MAP[lang_code]
    onnx = VOICE_DIR / f"{name}.onnx"
    cfg  = VOICE_DIR / f"{name}.onnx.json"

    if not onnx.is_file() or not cfg.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Voice files missing for {lang_code}: {onnx} / {cfg}",
        )
    return onnx, cfg


def _load_voice(lang_code: str) -> PiperVoice:
    onnx, cfg = _voice_paths(lang_code)

    # PiperVoice.load signature varies slightly across versions; this matches piper-tts 1.3.0
    try:
        voice = PiperVoice.load(str(onnx), str(cfg), use_cuda=USE_CUDA)
    except TypeError:
        # fallback if build doesn't accept use_cuda kwarg
        voice = PiperVoice.load(str(onnx), str(cfg))

    print(f"[TTS] loaded {lang_code} -> {onnx.name} (cuda={USE_CUDA})")
    return voice


class SynthesizeIn(BaseModel):
    text: str
    lang_code: str


app = FastAPI()

VOICES: dict[str, PiperVoice] = {}
LOCKS: dict[str, threading.Lock] = {}


@app.on_event("startup")
def startup():
    if not VOICE_DIR.is_dir():
        raise RuntimeError(f"PIPER_VOICE_DIR is not a directory: {VOICE_DIR}")

    for lang in VOICE_MAP.keys():
        VOICES[lang] = _load_voice(lang)
        LOCKS[lang] = threading.Lock()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/synthesize")
def synthesize(payload: SynthesizeIn):
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")

    if payload.lang_code not in VOICES:
        raise HTTPException(status_code=400, detail=f"Unknown lang_code: {payload.lang_code}")

    voice = VOICES[payload.lang_code]
    lock = LOCKS[payload.lang_code]

    # IMPORTANT: synthesize_wav expects a wave.Wave_write object, not a filepath.
    buf = io.BytesIO()

    try:
        with lock:
            with wave.open(buf, "wb") as wf:
                voice.synthesize_wav(text, wf)

        data = buf.getvalue()

        # Sanity check: WAV header is 44 bytes; if that's all we have, no audio samples were written.
        if len(data) <= 44:
            raise HTTPException(status_code=500, detail="Generated WAV is empty (0 frames)")

        return Response(content=data, media_type="audio/wav")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS synth failed: {e}")