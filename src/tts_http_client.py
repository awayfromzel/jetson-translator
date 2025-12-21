import os
import requests
import datetime

class PiperHttpTTS:
    def __init__(self, base_url: str, out_dir: str, timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.out_dir = out_dir
        self.timeout_s = timeout_s
        os.makedirs(self.out_dir, exist_ok=True)

    def synthesize(self, text: str, lang_code: str, out_basename_prefix: str = "translation") -> str | None:
        url = f"{self.base_url}/synthesize"
        payload = {"text": text, "lang_code": lang_code}

        r = requests.post(url, json=payload, timeout=self.timeout_s)
        if r.status_code != 200:
            print(f"[TTS-HTTP] error {r.status_code}: {r.text}")
            return None

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = os.path.join(self.out_dir, f"{out_basename_prefix}_{lang_code}_{stamp}.wav")
        with open(wav_path, "wb") as f:
            f.write(r.content)
        return wav_path
