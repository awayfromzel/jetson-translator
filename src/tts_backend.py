# tts_backend.py
from abc import ABC, abstractmethod
import os, datetime, subprocess, shutil

class BaseTTS(ABC):
    @abstractmethod
    def synthesize(self, text: str, lang_code: str) -> str | None:
        """Return path to wav, or None if cannot synthesize."""


class PiperSubprocessTTS(BaseTTS):
    def __init__(self, piper_bin: str, voice_dir: str, out_dir: str, voice_map: dict, spk_device: str):
        self.piper_bin = piper_bin
        self.voice_dir = voice_dir
        self.out_dir = out_dir
        self.voice_map = voice_map
        self.spk_device = spk_device

    def _voice_paths(self, lang_code):
        name = self.voice_map.get(lang_code)
        if not name:
            return None
        onnx = os.path.join(self.voice_dir, f"{name}.onnx")
        cfg  = os.path.join(self.voice_dir, f"{name}.onnx.json")
        if not (os.path.isfile(onnx) and os.path.isfile(cfg)):
            return None
        return onnx, cfg

    def synthesize(self, text: str, lang_code: str) -> str | None:
        paths = self._voice_paths(lang_code)
        if not paths:
            return None
        onnx, cfg = paths

        os.makedirs(self.out_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = os.path.join(self.out_dir, f"translation_{lang_code}_{stamp}.wav")

        proc = subprocess.Popen(
            [self.piper_bin, "-m", onnx, "-c", cfg, "--output-file", wav_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        _, stderr = proc.communicate(text, timeout=60)
        if proc.returncode != 0:
            print(f"[TTS] Piper error: {stderr.strip()}")
            return None

        return wav_path