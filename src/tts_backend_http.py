import uuid
from pathlib import Path

import requests


class PiperHTTPServiceTTS:
    def __init__(self, base_url: str, out_dir: str):
        self.base_url = base_url.rstrip("/")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(self, text: str, lang_code: str) -> str | None:
        payload = {"text": text, "lang_code": lang_code}

        try:
            r = requests.post(
                f"{self.base_url}/synthesize",
                json=payload,
                timeout=180,
            )
        except Exception as e:
            print(f"[TTS HTTP] request failed: {e}")
            return None

        if r.status_code != 200:
            print(f"[TTS HTTP] error {r.status_code}: {r.text}")
            return None

        wav_path = self.out_dir / f"tts_{lang_code}_{uuid.uuid4().hex}.wav"
        wav_path.write_bytes(r.content)
        return str(wav_path)
